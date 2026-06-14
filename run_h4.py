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

This is a complete, self-contained pipeline run: stage 1 ("uruchomienia")
writes ``<output-dir>/raw_results.json`` plus replayable per-game logs under
``<output-dir>/games/``; stages 2-4 (statistics, significance tests, plots --
see ``analyze_results.py``/``plot_results.py``) then run automatically,
writing ``stats.json``/``stats.csv`` (including the H4-specific
``both_configs_hold`` verdict) and PNGs under ``<output-dir>/plots/``.

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
    python run_h4.py

    # faster, richer, but no longer strictly reproducible across machines:
    python run_h4.py --workers 16 --iterations 2000
"""

from __future__ import annotations

from hypothesis_lib import ExperimentConfig, build_arg_parser, run_full_pipeline, run_stage1

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
    raw_path = run_stage1(args, configs)
    run_full_pipeline(raw_path, "h4", args.output_dir)


if __name__ == "__main__":
    main()
