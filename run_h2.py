"""H2 -- stage 1 ("uruchomienia"): UCT+RAVE vs. plain UCT, same iteration budget.

Konspekt H2: "UCT + RAVE przy tym samym budżecie iteracji gra istotnie
skuteczniej od czystego UCT, osiągając >60% wygranych."

This sweeps several shared iteration budgets ("depths"), each repeated
across >=5 base seeds (colors alternate within each seed's games, via
``experiments.run_match``). Both sides run MCTS at the same budget, so the
per-config cost is roughly ``2 * iterations``.

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
    python run_h2.py

    # faster, richer, but no longer strictly reproducible across machines:
    python run_h2.py --workers 16 --iterations 500 1000 2000 5000
"""

from __future__ import annotations

from hypothesis_lib import ExperimentConfig, build_arg_parser, run_full_pipeline, run_stage1

# RAVE vs UCT both run MCTS per ply, so each config costs ~2x an H1 config at
# the same iteration count.
DEFAULT_ITERATIONS = [500, 1000, 2000, 5000]


def build_configs(iterations_list: list[int]) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            name=f"rave{it}",
            p1="rave", p2="uct",
            iterations=it,
            mcts_search_units=2 * it,  # both sides run MCTS
        )
        for it in iterations_list
    ]


def main() -> None:
    parser = build_arg_parser(
        description="H2: UCT+RAVE vs UCT at equal iteration budgets "
                     "(stage 1: runs)",
        default_iterations=DEFAULT_ITERATIONS,
        default_output_dir="output/h2",
    )
    args = parser.parse_args()

    configs = build_configs(args.iterations)
    raw_path = run_stage1(args, configs)
    run_full_pipeline(raw_path, "h2", args.output_dir)


if __name__ == "__main__":
    main()
