"""H1 -- stage 1 ("uruchomienia"): pure UCT vs. the heuristic player.

Konspekt H1: "Czysty UCT z wystarczającym limitem iteracji (>=10000 na ruch)
gra istotnie skuteczniej od gracza czysto heurystycznego, osiągając >60%
wygranych w meczu z naprzemienną zmianą kolorów."

This sweeps several UCT iteration budgets ("depths") against the heuristic
player, each repeated across >=5 base seeds (with colors alternating within
each seed's games, via ``experiments.run_match``). Only UCT runs MCTS, so the
per-config cost scales with ``iterations`` alone.

Output: ``<output-dir>/raw_results.json`` (one entry per iteration budget,
each containing one ``run_match`` summary per base seed) plus replayable
per-game logs under ``<output-dir>/games/``.

Next pipeline stages: ``analyze_results.py`` (statistics + significance),
then ``plot_results.py``.

Usage
-----
    python run_h1.py --workers 16 --seeds 5 --iterations 500 1000 2000 5000 10000

This script only plans and (when run) executes games -- it does not analyze
or plot results.
"""

from __future__ import annotations

from hypothesis_lib import ExperimentConfig, build_arg_parser, run_stage1

DEFAULT_ITERATIONS = [500, 1000, 2000, 5000, 10000]


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
    run_stage1(args, configs)


if __name__ == "__main__":
    main()
