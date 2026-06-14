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

Output: ``<output-dir>/raw_results.json`` (one entry per iteration budget,
each containing one ``run_match`` summary per base seed) plus replayable
per-game logs under ``<output-dir>/games/``.

Next pipeline stages: ``analyze_results.py`` (statistics + significance),
then ``plot_results.py``.

Usage
-----
    python run_h3.py --workers 16 --seeds 5 --iterations 500 1000 2000 5000 10000

This script only plans and (when run) executes games -- it does not analyze
or plot results.
"""

from __future__ import annotations

from hypothesis_lib import ExperimentConfig, build_arg_parser, run_stage1

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
    run_stage1(args, configs)


if __name__ == "__main__":
    main()
