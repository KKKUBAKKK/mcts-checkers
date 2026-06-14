"""H3 -- stage 1 ("uruchomienia"): UCT + progressive bias vs. plain UCT,
same iteration budget.

Konspekt H3: "UCT z heurystycznym progressive bias wygrywa >60% partii z
czystym UCT, a jego przewaga jest największa przy małym limicie iteracji
(<=2000/ruch) ... Wraz ze wzrostem liczby iteracji oczekujemy zaniku
przewagi."

This sweeps several shared iteration budgets ("depths"), each repeated
across >=5 base seeds (colors alternate within each seed's games, via
``experiments.run_match``). Both sides run MCTS at the same budget, so the
per-config cost is roughly ``2 * iterations``. The default sweep spans both
sides of the <=2000 threshold (500..10000) so ``plot_results.py``'s
win-rate-vs-iterations curve can show the expected fading advantage.

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

Time budget: default ``--time-budget-min 110`` keeps the whole run under 2h.

Usage
-----
    python run_h3.py

    # faster, richer, but no longer strictly reproducible across machines:
    python run_h3.py --workers 16 --iterations 500 1000 2000 5000 10000
"""

from __future__ import annotations

from hypothesis_lib import ExperimentConfig, build_arg_parser, run_full_pipeline, run_stage1

# Both sides run MCTS, so each config costs ~2x an H1 config at the same
# iteration count -- mirror H1's sweep so the <=2000 vs >2000 trend is
# visible while keeping the same overall shape as H1/H2.
DEFAULT_ITERATIONS = [500, 1000, 2000, 5000, 10000]


def build_configs(iterations_list: list[int]) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            name=f"progressive{it}",
            p1="progressive", p2="uct",
            iterations=it,
            mcts_search_units=2 * it,  # both sides run MCTS
        )
        for it in iterations_list
    ]


def main() -> None:
    parser = build_arg_parser(
        description="H3: UCT+progressive bias vs UCT at equal iteration "
                     "budgets (stage 1: runs)",
        default_iterations=DEFAULT_ITERATIONS,
        default_output_dir="output/h3",
    )
    args = parser.parse_args()

    if not any(it <= 2000 for it in args.iterations):
        print("NOTE: H3 specifically claims the largest advantage at "
              "<=2000 iterations; consider including a budget <=2000 in "
              "--iterations.")

    configs = build_configs(args.iterations)
    raw_path = run_stage1(args, configs)
    run_full_pipeline(raw_path, "h3", args.output_dir)


if __name__ == "__main__":
    main()
