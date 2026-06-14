"""Shared library for the H1/H2 hypothesis experiment pipeline.

Pipeline stages (see ``run_h1.py`` / ``run_h2.py``, ``analyze_results.py``,
``plot_results.py``):

1. **runs**       -- play many games per configuration, with >=5 base seeds
                     each, and dump raw per-game results to JSON.
2. **statistics** -- pool results per configuration, compute win rates and
                     Wilson 95% confidence intervals.
3. **hypothesis / significance tests** -- exact two-sided binomial test
                     against p0=0.5, plus a cross-seed consistency check.
4. **plots**      -- bar charts (win rate + CI per config) and, where a
                     configuration is swept over MCTS iteration budgets, a
                     win-rate-vs-iterations curve with a CI band.

GPU note
--------
This MCTS implementation is a branch-heavy, object-oriented Python game-tree
search (board cloning, recursive capture-DFS, random rollouts). None of that
vectorizes onto a GPU without rewriting the game engine in array/tensor form,
so there is no CUDA path here. Parallelism is CPU-only, via root
parallelization (independent trees per worker process); use ``--workers`` to
match the machine's core count.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

from experiments import run_match
from game import CheckersGame
from mcts import mcts_search_parallel

# Root-parallel MCTS splits ``iterations`` across ``num_workers`` processes
# and assigns each worker a seed derived from the search seed and its worker
# index (see ``mcts.mcts_search_parallel``), so the *number* of workers is
# part of what makes a search reproducible -- not just the seed. To make the
# no-flags default invocation of run_h1..run_h4 reproducible across machines
# with different core counts, the CLI default is a fixed constant rather than
# ``os.cpu_count()``. Pass ``--workers`` explicitly for faster (but then
# machine-specific) runs.
DEFAULT_WORKERS = 4


# ──────────────────────────────────────── seeds

def derive_seeds(base_seed: int, n: int) -> list[int]:
    """Deterministically derive ``n`` base seeds from one base seed.

    Each derived seed is itself used by ``run_match`` to derive per-game
    seeds, so a large odd stride keeps the resulting seed sequences from
    overlapping.
    """
    return [base_seed + i * 10007 for i in range(n)]


# ──────────────────────────────────────── statistics

def wilson_ci(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95% CI)."""
    if n == 0:
        return (0.0, 1.0)
    phat = wins / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return max(0.0, lo), min(1.0, hi)


def binomial_two_sided_pvalue(wins: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial test p-value for H0: P(success) = p0.

    Sums the probabilities of all outcomes at least as extreme as the
    observed one (standard "small p-value" definition), avoiding a
    dependency on scipy.
    """
    if n == 0:
        return 1.0
    obs = math.comb(n, wins) * p0 ** wins * (1 - p0) ** (n - wins)
    total = 0.0
    for k in range(n + 1):
        pk = math.comb(n, k) * p0 ** k * (1 - p0) ** (n - k)
        if pk <= obs * (1 + 1e-9):
            total += pk
    return min(1.0, total)


# ──────────────────────────────────────── experiment configuration

@dataclass
class ExperimentConfig:
    """One (p1 vs p2, iteration budget) configuration to repeat across seeds.

    ``mcts_search_units`` is the total number of MCTS-search "units" of work
    performed per ply pair (one ply by each side), used only for time
    estimation: it should equal ``iterations`` if only one side runs MCTS
    (H1: UCT vs heuristic), or ``2 * iterations`` if both sides run MCTS at
    the same budget (H2: RAVE vs UCT).
    """
    name: str
    p1: str
    p2: str
    iterations: int
    p1_heuristic: str = "base"
    p2_heuristic: str = "base"
    mcts_search_units: int = field(default=0)

    def __post_init__(self) -> None:
        if self.mcts_search_units == 0:
            self.mcts_search_units = self.iterations


# ──────────────────────────────────────── calibration & time budgeting

def calibrate_seconds_per_iteration(iterations: int = 200, workers: int = 4,
                                    seed: int = 0) -> float:
    """Time a single root-parallel UCT search from the initial position.

    Returns wall-clock seconds per MCTS iteration (averaged across
    ``workers`` worker processes), used to estimate game durations.
    """
    game = CheckersGame()
    t0 = time.time()
    mcts_search_parallel(game, iterations, variant="uct", seed=seed,
                         num_workers=workers)
    elapsed = time.time() - t0
    return elapsed / iterations


def estimate_game_seconds(cfg: ExperimentConfig, sec_per_iter: float,
                          plies_per_side: int = 30) -> float:
    """Rough wall-clock estimate for one game under ``cfg``.

    Assumes ``plies_per_side`` MCTS searches per side (a typical game is
    ~60-150 plies total, i.e. ~30-75 per side).
    """
    return cfg.mcts_search_units * sec_per_iter * plies_per_side


def allocate_games_per_seed(configs: list[ExperimentConfig], sec_per_iter: float,
                            time_budget_s: float, seeds: int,
                            min_games: int = 1, max_games: int = 10,
                            plies_per_side: int = 30
                            ) -> tuple[dict[str, int], float]:
    """Distribute a time budget across configs as extra games-per-seed.

    Every config starts at ``min_games`` games per seed (so with ``seeds``
    base seeds, every config gets at least ``seeds * min_games`` games --
    satisfying the ">=5 base seeds per configuration" requirement when
    ``seeds >= 5``). Remaining time is greedily spent one game-per-seed at a
    time on whichever config is currently cheapest, up to ``max_games``.

    Returns ``(games_per_seed_by_config_name, estimated_total_seconds)``.
    """
    cost_per_game_per_seed = {
        c.name: estimate_game_seconds(c, sec_per_iter, plies_per_side)
        for c in configs
    }
    games = {c.name: min_games for c in configs}
    total = sum(cost_per_game_per_seed[c.name] * games[c.name] * seeds for c in configs)

    while True:
        candidates = [c for c in configs if games[c.name] < max_games]
        if not candidates:
            break
        cheapest = min(candidates, key=lambda c: cost_per_game_per_seed[c.name])
        step = cost_per_game_per_seed[cheapest.name] * seeds
        if total + step > time_budget_s:
            break
        games[cheapest.name] += 1
        total += step

    return games, total


# ──────────────────────────────────────── running configs

def run_config(cfg: ExperimentConfig, seeds: list[int], games_per_seed: int,
               max_moves: int, num_workers: int | None,
               log_dir: str | None, run_tag: str) -> dict:
    """Run ``games_per_seed`` games for each base seed under ``cfg``.

    Returns a dict with the config and one ``run_match`` summary per seed
    (each summary already contains per-game results, win/draw counts, and
    average node counts -- see ``experiments.run_match``).
    """
    per_seed = []
    for seed in seeds:
        summary = run_match(
            cfg.p1, cfg.p2, games_per_seed, cfg.iterations, seed,
            max_moves, verbose=False, log_dir=log_dir,
            run_tag=f"{run_tag}{cfg.name}_seed{seed}_",
            p1_heuristic=cfg.p1_heuristic, p2_heuristic=cfg.p2_heuristic,
            num_workers=num_workers,
        )
        per_seed.append({"seed": seed, "summary": summary})
    return {"config": asdict(cfg), "per_seed": per_seed}


# ──────────────────────────────────────── CLI / stage-1 driver

def build_arg_parser(description: str, default_iterations: list[int],
                     default_output_dir: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help="CPU worker processes for root-parallel MCTS "
                        f"(default: {DEFAULT_WORKERS}, fixed for "
                        "reproducibility -- changing this changes results)")
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of base seeds (>=5 recommended)")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--iterations", type=int, nargs="+",
                   default=default_iterations,
                   help="MCTS iteration budgets to sweep ('depths')")
    p.add_argument("--time-budget-min", type=float, default=110,
                   help="Soft wall-clock budget in minutes for this run "
                        "(default 110, i.e. just under 2h)")
    p.add_argument("--min-games-per-seed", type=int, default=1,
                   help="Minimum games per seed per config "
                        "(guarantees >= seeds * this many games per config)")
    p.add_argument("--max-games-per-seed", type=int, default=10,
                   help="Cap on games per seed per config when filling the "
                        "time budget")
    p.add_argument("--max-moves", type=int, default=200,
                   help="Half-move limit per game (draw if exceeded)")
    p.add_argument("--plies-per-side", type=int, default=30,
                   help="Assumed plies per side per game, used only for "
                        "time-budget estimation")
    p.add_argument("--calibration-iterations", type=int, default=200,
                   help="Iterations used to measure ms/iteration before "
                        "allocating the time budget")
    p.add_argument("--output-dir", type=str, default=default_output_dir)
    return p


def run_stage1(args: argparse.Namespace, configs: list[ExperimentConfig]) -> str:
    """Stage 1 ("uruchomienia"): run all configs, write raw_results.json.

    Returns the path to the written ``raw_results.json``.
    """
    if args.seeds < 5:
        print(f"WARNING: --seeds={args.seeds} < 5; the konspekt requires "
              f">=5 base seeds per configuration.")

    output_dir = args.output_dir
    games_dir = os.path.join(output_dir, "games")
    os.makedirs(games_dir, exist_ok=True)

    workers = args.workers
    print(f"Calibrating: {args.calibration_iterations} UCT iterations on "
          f"{workers} worker(s)...")
    sec_per_iter = calibrate_seconds_per_iteration(args.calibration_iterations, workers)
    print(f"  -> {sec_per_iter * 1000:.3f} ms/iteration (aggregate across workers)")

    seeds = derive_seeds(args.base_seed, args.seeds)
    time_budget_s = args.time_budget_min * 60
    games_per_seed, est_total_s = allocate_games_per_seed(
        configs, sec_per_iter, time_budget_s, len(seeds),
        args.min_games_per_seed, args.max_games_per_seed, args.plies_per_side,
    )

    print("\nPlanned run:")
    for c in configs:
        g = games_per_seed[c.name]
        est_s = estimate_game_seconds(c, sec_per_iter, args.plies_per_side) * g * len(seeds)
        print(f"  {c.name:>14s}: {c.p1} vs {c.p2}, iterations={c.iterations}, "
              f"{g} games/seed x {len(seeds)} seeds = {g * len(seeds)} games "
              f"(~{est_s / 60:.1f} min)")
    print(f"  Estimated total: ~{est_total_s / 60:.1f} min "
          f"(budget: {args.time_budget_min:.0f} min)\n")

    run_tag = time.strftime("%Y%m%d_%H%M%S_")
    raw_path = os.path.join(output_dir, "raw_results.json")
    all_results: list[dict] = []
    start = time.time()
    for c in configs:
        g = games_per_seed[c.name]
        print(f"=== {c.name}: {c.p1} vs {c.p2} (iterations={c.iterations}, "
              f"{g} games/seed x {len(seeds)} seeds) ===")
        result = run_config(c, seeds, g, args.max_moves, args.workers, games_dir, run_tag)
        all_results.append(result)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"  done ({(time.time() - start) / 60:.1f} min elapsed total)\n")

    print(f"Raw results written to {raw_path}")
    print(f"Total elapsed: {(time.time() - start) / 60:.1f} min")
    return raw_path


def run_full_pipeline(raw_path: str, hypothesis: str, output_dir: str) -> None:
    """Stages 2-4 ("statystyki" -> "testy hipotez i istotności" ->
    "generowanie wykresów"), run immediately after :func:`run_stage1` so a
    single no-flags command (``python run_h1.py`` etc.) executes the whole
    runs -> statistics -> hypothesis tests -> plots pipeline.

    Writes ``<output_dir>/stats.json`` and ``stats.csv`` (see
    ``analyze_results.py``) and PNG plots under ``<output_dir>/plots/`` (see
    ``plot_results.py``).
    """
    # Imported lazily: analyze_results imports from this module at module
    # load time, so importing it back at module level here would be circular.
    from analyze_results import compute_stats, print_summary, write_stats
    from plot_results import generate_plots

    with open(raw_path, encoding="utf-8") as f:
        raw_results = json.load(f)

    stats = compute_stats(raw_results, hypothesis)
    stats_path, csv_path = write_stats(stats, output_dir)
    print_summary(stats)
    print(f"\nWritten to {stats_path} and {csv_path}")

    plots_dir = os.path.join(output_dir, "plots")
    for path in generate_plots(stats, plots_dir):
        print(f"Wrote {path}")
