"""H1 -- stage 1 ("uruchomienia"): pure UCT vs. the heuristic player.

Konspekt H1: "Czysty UCT z wystarczającym limitem iteracji (>=10000 na ruch)
gra istotnie skuteczniej od gracza czysto heurystycznego, osiągając >60%
wygranych w meczu z naprzemienną zmianą kolorów."

This sweeps several UCT iteration budgets ("depths") against the heuristic
player, each repeated across >=5 base seeds (with colors alternating within
each seed's games, via ``experiments.run_match``). Only UCT runs MCTS, so the
per-config cost scales with ``iterations`` alone.

This is a complete, self-contained pipeline run: stage 1 ("uruchomienia")
writes ``<output-dir>/raw_results.json`` (one entry per iteration budget,
each containing one ``run_match`` summary per base seed) plus replayable
per-game logs under ``<output-dir>/games/``; stages 2-4 (statistics,
significance tests, plots -- see ``analyze_results.py``/``plot_results.py``)
then run automatically, writing ``stats.json``/``stats.csv`` and PNGs under
``<output-dir>/plots/``.

Reproducibility
---------------
With **no flags**, this script uses fixed defaults (base seed 0, 5 derived
base seeds, fixed worker count -- see ``hypothesis_lib.DEFAULT_WORKERS``) and
is fully deterministic: the same machine-independent sequence of games is
played every time. Pass ``--workers`` to use more cores for a faster but
machine-specific (non-reproducible) "rich" run.

Time budget: default ``--time-budget-min 80`` keeps the whole run under ~80 min.

Usage
-----
    python run_h1.py

    # faster, richer, but no longer strictly reproducible across machines:
    python run_h1.py --workers 16 --iterations 500 1000 2000 5000 10000
"""

from __future__ import annotations

from hypothesis_lib import ExperimentConfig, build_arg_parser, run_full_pipeline, run_stage1

DEFAULT_ITERATIONS = [500, 1000, 2000, 3000, 5000]


def build_configs(iterations_list: list[int]) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            name=f"uct{it}",
            p1="uct", p2="heuristic",
            iterations=it,
            mcts_search_units=it,  # only p1 (UCT) runs MCTS
        )
        for it in iterations_list
    ]


def main() -> None:
    parser = build_arg_parser(
        description="H1: UCT vs heuristic, swept over iteration budgets "
                     "(stage 1: runs)",
        default_iterations=DEFAULT_ITERATIONS,
        default_output_dir="output/h1",
    )
    args = parser.parse_args()

    if 10000 not in args.iterations:
        print("NOTE: H1 specifically claims a result at >=10000 iterations; "
              "consider including 10000 in --iterations.")

    configs = build_configs(args.iterations)
    raw_path = run_stage1(args, configs)
    run_full_pipeline(raw_path, "h1", args.output_dir)


if __name__ == "__main__":
    main()
