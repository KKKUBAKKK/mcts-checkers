"""Unit tests for the pure helper functions in ``hypothesis_lib``.

These do not run any MCTS games (no calls to ``run_match`` / ``run_config`` /
``calibrate_seconds_per_iteration``), so they are safe to run even though the
H1/H2 experiment pipelines themselves should not be executed yet.
"""

from __future__ import annotations

import math

from hypothesis_lib import (
    ExperimentConfig,
    allocate_games_per_seed,
    binomial_two_sided_pvalue,
    derive_seeds,
    estimate_game_seconds,
    wilson_ci,
)
from run_h1 import build_configs as build_h1_configs
from run_h2 import build_configs as build_h2_configs
from run_h3 import build_configs as build_h3_configs
from run_h4 import build_configs as build_h4_configs
from analyze_results import analyze_config


def test_derive_seeds_deterministic_and_distinct():
    seeds_a = derive_seeds(0, 5)
    seeds_b = derive_seeds(0, 5)
    assert seeds_a == seeds_b
    assert len(set(seeds_a)) == 5
    assert seeds_a[0] == 0


def test_derive_seeds_different_base_seeds_differ():
    assert derive_seeds(0, 5) != derive_seeds(1, 5)


def test_wilson_ci_bounds_and_symmetry():
    lo, hi = wilson_ci(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0

    # zero games -> maximally uninformative interval
    assert wilson_ci(0, 0) == (0.0, 1.0)

    # a clean sweep gives a CI anchored near the top
    lo_all, hi_all = wilson_ci(10, 10)
    assert math.isclose(hi_all, 1.0)
    assert 0.0 < lo_all < 1.0


def test_wilson_ci_narrows_with_more_games():
    lo_small, hi_small = wilson_ci(5, 10)
    lo_large, hi_large = wilson_ci(50, 100)
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_binomial_two_sided_pvalue_extremes():
    # exactly 50/50 with an even count -> p-value of 1 (most likely outcome)
    assert math.isclose(binomial_two_sided_pvalue(5, 10, p0=0.5), 1.0)

    # all wins out of many games is extremely unlikely under p0=0.5
    assert binomial_two_sided_pvalue(10, 10, p0=0.5) < 0.01

    # zero games -> no evidence, p-value of 1
    assert binomial_two_sided_pvalue(0, 0) == 1.0


def test_binomial_two_sided_pvalue_monotonic_in_extremity():
    p_close = binomial_two_sided_pvalue(6, 10, p0=0.5)
    p_far = binomial_two_sided_pvalue(9, 10, p0=0.5)
    assert p_far < p_close


def test_experiment_config_defaults_search_units_to_iterations():
    cfg = ExperimentConfig(name="x", p1="uct", p2="heuristic", iterations=1000)
    assert cfg.mcts_search_units == 1000


def test_experiment_config_explicit_search_units_preserved():
    cfg = ExperimentConfig(name="x", p1="rave", p2="uct", iterations=1000,
                            mcts_search_units=2000)
    assert cfg.mcts_search_units == 2000


def test_estimate_game_seconds_scales_with_units_and_plies():
    cfg1 = ExperimentConfig(name="x", p1="uct", p2="heuristic", iterations=1000)
    cfg2 = ExperimentConfig(name="y", p1="rave", p2="uct", iterations=1000,
                             mcts_search_units=2000)
    sec_per_iter = 0.001
    est1 = estimate_game_seconds(cfg1, sec_per_iter, plies_per_side=30)
    est2 = estimate_game_seconds(cfg2, sec_per_iter, plies_per_side=30)
    assert math.isclose(est2, 2 * est1)


def test_allocate_games_per_seed_respects_minimum_and_budget():
    configs = [
        ExperimentConfig(name="cheap", p1="uct", p2="heuristic", iterations=100),
        ExperimentConfig(name="expensive", p1="uct", p2="heuristic", iterations=100000),
    ]
    sec_per_iter = 0.001
    seeds = 5

    # generous budget: both configs reach at least min_games
    games, total = allocate_games_per_seed(
        configs, sec_per_iter, time_budget_s=1e9, seeds=seeds,
        min_games=2, max_games=10, plies_per_side=30,
    )
    assert games["cheap"] >= 2
    assert games["expensive"] >= 2
    assert total > 0

    # tiny budget: still guarantees the minimum for every config
    games_tiny, _ = allocate_games_per_seed(
        configs, sec_per_iter, time_budget_s=0, seeds=seeds,
        min_games=1, max_games=10, plies_per_side=30,
    )
    assert games_tiny["cheap"] == 1
    assert games_tiny["expensive"] == 1


def test_allocate_games_per_seed_caps_at_max_games():
    configs = [ExperimentConfig(name="cheap", p1="uct", p2="heuristic", iterations=1)]
    games, _ = allocate_games_per_seed(
        configs, sec_per_iter=1e-9, time_budget_s=1e9, seeds=5,
        min_games=1, max_games=3, plies_per_side=1,
    )
    assert games["cheap"] == 3


def test_run_h1_build_configs():
    configs = build_h1_configs([500, 10000])
    assert [c.iterations for c in configs] == [500, 10000]
    for c in configs:
        assert c.p1 == "uct"
        assert c.p2 == "heuristic"
        assert c.mcts_search_units == c.iterations


def test_run_h2_build_configs():
    configs = build_h2_configs([500, 1000, 2000])
    assert [c.iterations for c in configs] == [500, 1000, 2000]
    for c in configs:
        assert c.p1 == "rave"
        assert c.p2 == "uct"
        assert c.mcts_search_units == 2 * c.iterations


def test_run_h3_build_configs():
    configs = build_h3_configs([500, 2000, 10000])
    assert [c.iterations for c in configs] == [500, 2000, 10000]
    for c in configs:
        assert c.p1 == "progressive"
        assert c.p2 == "uct"
        assert c.mcts_search_units == 2 * c.iterations


def test_run_h4_build_configs():
    configs = build_h4_configs([2000])
    assert len(configs) == 2

    heur = configs[0]
    assert heur.name == "heuristic_edge_vs_center"
    assert heur.p1 == "heuristic" and heur.p2 == "heuristic"
    assert heur.p1_heuristic == "edge" and heur.p2_heuristic == "center"
    assert heur.mcts_search_units == 0

    prog = configs[1]
    assert prog.name == "progressive_edge_vs_center_2000"
    assert prog.p1 == "progressive" and prog.p2 == "progressive"
    assert prog.p1_heuristic == "edge" and prog.p2_heuristic == "center"
    assert prog.iterations == 2000
    assert prog.mcts_search_units == 4000


def test_run_h4_build_configs_multiple_iteration_budgets():
    configs = build_h4_configs([1000, 2000])
    names = [c.name for c in configs]
    assert names == [
        "heuristic_edge_vs_center",
        "progressive_edge_vs_center_1000",
        "progressive_edge_vs_center_2000",
    ]


def _fake_summary(games, p1_wins, p2_wins, draws):
    return {"games": games, "p1_wins": p1_wins, "p2_wins": p2_wins, "draws": draws}


def test_analyze_config_strong_consistent_result_holds():
    entry = {
        "config": {"name": "uct10000", "p1": "uct", "p2": "heuristic",
                    "iterations": 10000},
        "per_seed": [
            {"seed": 0, "summary": _fake_summary(10, 8, 2, 0)},
            {"seed": 1, "summary": _fake_summary(10, 9, 1, 0)},
            {"seed": 2, "summary": _fake_summary(10, 7, 3, 0)},
            {"seed": 3, "summary": _fake_summary(10, 8, 2, 0)},
            {"seed": 4, "summary": _fake_summary(10, 9, 1, 0)},
        ],
    }
    result = analyze_config(entry)

    assert result["games"] == 50
    assert result["p1_wins"] == 41
    assert result["p1_win_rate"] > 0.6
    assert result["pvalue_binomial"] < 0.05
    assert result["seeds_p1_ahead"] == 5
    assert result["consistent_across_seeds"] is True
    assert result["hypothesis_holds"] is True


def test_analyze_config_roughly_even_result_does_not_hold():
    entry = {
        "config": {"name": "uct500", "p1": "uct", "p2": "heuristic",
                    "iterations": 500},
        "per_seed": [
            {"seed": 0, "summary": _fake_summary(10, 5, 5, 0)},
            {"seed": 1, "summary": _fake_summary(10, 4, 6, 0)},
            {"seed": 2, "summary": _fake_summary(10, 6, 4, 0)},
            {"seed": 3, "summary": _fake_summary(10, 5, 5, 0)},
            {"seed": 4, "summary": _fake_summary(10, 4, 6, 0)},
        ],
    }
    result = analyze_config(entry)

    assert result["p1_win_rate"] == 0.48
    assert result["hypothesis_holds"] is False
    assert result["win_rate_above_threshold"] is False


def test_analyze_config_h4_threshold_is_50pct():
    entry = {
        "config": {"name": "heuristic_edge_vs_center", "p1": "heuristic",
                    "p2": "heuristic", "iterations": 0,
                    "p1_heuristic": "edge", "p2_heuristic": "center"},
        "per_seed": [
            {"seed": 0, "summary": _fake_summary(10, 6, 4, 0)},
            {"seed": 1, "summary": _fake_summary(10, 6, 4, 0)},
            {"seed": 2, "summary": _fake_summary(10, 6, 4, 0)},
            {"seed": 3, "summary": _fake_summary(10, 6, 4, 0)},
            {"seed": 4, "summary": _fake_summary(10, 6, 4, 0)},
        ],
    }
    # 0.6 win rate fails the H1-H3 threshold of 0.6 (not strictly greater)
    # but passes the H4 threshold of 0.5.
    result_h1 = analyze_config(entry, win_rate_threshold=0.6)
    result_h4 = analyze_config(entry, win_rate_threshold=0.5)

    assert result_h1["win_rate_above_threshold"] is False
    assert result_h4["win_rate_above_threshold"] is True
