"""H2 -- stage 1 ("uruchomienia"): UCT+RAVE vs. plain UCT, same iteration budget.

Konspekt H2: "UCT + RAVE przy tym samym budżecie iteracji gra istotnie
skuteczniej od czystego UCT, osiągając >60% wygranych."

This sweeps several shared iteration budgets ("depths"), each repeated
across >=5 base seeds (colors alternate within each seed's games, via
``experiments.run_match``). Both sides run MCTS at the same budget, so the
per-config cost is roughly ``2 * iterations``.

Output: ``<output-dir>/raw_results.json`` (one entry per iteration budget,
each containing one ``run_match`` summary per base seed) plus replayable
per-game logs under ``<output-dir>/games/``.

Next pipeline stages: ``analyze_results.py`` (statistics + significance),
then ``plot_results.py``.

Usage
-----
    python run_h2.py --workers 16 --seeds 5 --iterations 500 1000 2000 5000

This script only plans and (when run) executes games -- it does not analyze
or plot results.
"""

from __future__ import annotations

from hypothesis_lib import ExperimentConfig, build_arg_parser, run_stage1

# RAVE vs UCT both run MCTS per ply, so each config costs ~2x an H1 config at
# the same iteration count -- use a smaller default sweep to stay in budget.
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
    run_stage1(args, configs)


if __name__ == "__main__":
    main()
