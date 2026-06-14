"""H4 -- stage 1 ("uruchomienia"): edge-favouring vs. center-favouring
positional heuristic ("h").

Konspekt H4: "heurystyki premiujące figury przy krawędzi planszy są
skuteczniejsze niż heurystyki premiujące figury w centrum. Hipotezę
zweryfikujemy ... wstawione do obu graczy wykorzystujących heurystykę
(gracza czysto heurystycznego oraz UCT + progressive bias). Mecze
rozegramy w dwóch konfiguracjach ... heurystyka-krawędzie vs
heurystyka-centrum. Hipotezę uznamy za potwierdzoną, jeśli w obu
konfiguracjach wariant 'krawędzie' osiąga >50% wygranych."

Two configurations, each repeated across >=5 base seeds (colors alternate
within each seed's games, via ``experiments.run_match``):

1. ``heuristic_edge_vs_center`` -- ``HeuristicPlayer(heuristic="edge")`` vs.
   ``HeuristicPlayer(heuristic="center")``. Neither side runs MCTS, so this
   configuration is essentially free and gets as many games as
   ``--max-games-per-seed`` allows.
2. ``progressive_edge_vs_center_<it>`` (one per entry in ``--iterations``)
   -- ``progressive`` (UCT + progressive bias) with the bias term ``h``
   computed under the "edge" heuristic vs. the same under "center", at a
   shared iteration budget. Both sides run MCTS, so cost scales with
   ``2 * iterations``.

The default iteration budget (2000) sits at the H3 "small budget" end where
the progressive-bias term has the most influence.

Output: ``<output-dir>/raw_results.json`` plus replayable per-game logs
under ``<output-dir>/games/``.

Next pipeline stages: ``analyze_results.py`` (statistics + significance),
then ``plot_results.py``.

Usage
-----
    python run_h4.py --workers 16 --seeds 5 --iterations 2000

This script only plans and (when run) executes games -- it does not analyze
or plot results.
"""

from __future__ import annotations

from hypothesis_lib import ExperimentConfig, build_arg_parser, run_stage1

DEFAULT_ITERATIONS = [2000]


def build_configs(iterations_list: list[int]) -> list[ExperimentConfig]:
    configs = [
        ExperimentConfig(
            name="heuristic_edge_vs_center",
            p1="heuristic", p2="heuristic",
            iterations=0,
            p1_heuristic="edge", p2_heuristic="center",
            mcts_search_units=0,  # neither side runs MCTS
        )
    ]
    for it in iterations_list:
        configs.append(ExperimentConfig(
            name=f"progressive_edge_vs_center_{it}",
            p1="progressive", p2="progressive",
            iterations=it,
            p1_heuristic="edge", p2_heuristic="center",
            mcts_search_units=2 * it,  # both sides run MCTS
        ))
    return configs


def main() -> None:
    parser = build_arg_parser(
        description="H4: edge-favouring vs center-favouring heuristic, "
                     "for the heuristic player and progressive-bias UCT "
                     "(stage 1: runs)",
        default_iterations=DEFAULT_ITERATIONS,
        default_output_dir="output/h4",
    )
    args = parser.parse_args()

    configs = build_configs(args.iterations)
    run_stage1(args, configs)


if __name__ == "__main__":
    main()
