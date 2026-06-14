"""Tests for players, MCTS node counting and game-log replay."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game import (  # noqa: E402
    CheckersGame, WHITE, BLACK, EMPTY, BOARD_SIZE,
    WHITE_PIECE, BLACK_PIECE, WHITE_KING,
)
from mcts import mcts_search, evaluate, HEURISTICS  # noqa: E402
from players import HeuristicPlayer, MCTSPlayer, RandomPlayer  # noqa: E402
from experiments import play_game, write_game_log, replay_game_log, make_player  # noqa: E402


def empty_game(current_player=WHITE):
    g = CheckersGame()
    g.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    g.current_player = current_player
    return g


# ──────────────────────────────────────── heuristic

def test_heuristic_uses_shared_evaluator_signature():
    # HeuristicPlayer must rely on mcts.evaluate (no private _evaluate left)
    assert not hasattr(HeuristicPlayer, "_evaluate")


def test_heuristic_takes_free_capture():
    g = empty_game(WHITE)
    g.board[5][4] = WHITE_PIECE
    g.board[4][3] = BLACK_PIECE   # free capture -> wins all material
    player = HeuristicPlayer(seed=0)
    move = player.choose_move(g)
    assert move == [(5, 4), (3, 2)]


def test_evaluate_symmetry():
    g = CheckersGame()
    # symmetric start position evaluates to ~0.5 from white's perspective,
    # for every heuristic variant (the positional bonus is symmetric too)
    for h in HEURISTICS:
        assert abs(evaluate(g, h) - 0.5) < 1e-9


# ──────────────────────────────────────── H4 heuristic variants (edge/center)

def test_edge_heuristic_favours_side_columns():
    # a single white king on a side column vs. the same king centrally placed
    g_edge = empty_game(WHITE)
    g_edge.board[5][0] = WHITE_KING
    g_edge.board[0][9] = BLACK_PIECE

    g_center = empty_game(WHITE)
    g_center.board[5][4] = WHITE_KING
    g_center.board[0][9] = BLACK_PIECE

    assert evaluate(g_edge, "edge") > evaluate(g_center, "edge")


def test_center_heuristic_favours_central_columns():
    g_edge = empty_game(WHITE)
    g_edge.board[5][0] = WHITE_KING
    g_edge.board[0][9] = BLACK_PIECE

    g_center = empty_game(WHITE)
    g_center.board[5][4] = WHITE_KING
    g_center.board[0][9] = BLACK_PIECE

    assert evaluate(g_center, "center") > evaluate(g_edge, "center")


def test_base_heuristic_ignores_column():
    g_edge = empty_game(WHITE)
    g_edge.board[5][0] = WHITE_KING
    g_edge.board[0][9] = BLACK_PIECE

    g_center = empty_game(WHITE)
    g_center.board[5][4] = WHITE_KING
    g_center.board[0][9] = BLACK_PIECE

    assert evaluate(g_edge, "base") == evaluate(g_center, "base")


def test_heuristic_player_injectable_heuristic():
    # one white piece with two equal-material moves: to column 3 (further
    # from center) or column 5 (closer to center); a fixed black piece
    # elsewhere keeps evaluate() non-degenerate.
    g = empty_game(WHITE)
    g.board[5][4] = WHITE_PIECE
    g.board[0][9] = BLACK_PIECE

    edge_player = HeuristicPlayer(seed=0, heuristic="edge")
    center_player = HeuristicPlayer(seed=0, heuristic="center")

    assert edge_player.choose_move(g) == [(5, 4), (4, 3)]
    assert center_player.choose_move(g) == [(5, 4), (4, 5)]


def test_make_player_passes_heuristic_to_progressive():
    g = CheckersGame()
    player = make_player("progressive", iterations=20, seed=3, heuristic="edge")
    assert isinstance(player, MCTSPlayer)
    assert player.heuristic == "edge"
    move, nodes = mcts_search(g, iterations=20, variant="progressive",
                              seed=3, heuristic="edge")
    assert move in g.get_legal_moves()
    assert nodes > 1


# ──────────────────────────────────────── MCTS node counting

def test_mcts_search_returns_move_and_node_count():
    g = CheckersGame()
    move, nodes = mcts_search(g, iterations=50, variant="uct", seed=1)
    assert move in g.get_legal_moves()
    assert isinstance(nodes, int)
    assert nodes > 1  # at least the root plus some expansions


def test_mcts_player_records_node_count():
    g = CheckersGame()
    player = MCTSPlayer(variant="uct", iterations=40, parallel=False, seed=7)
    move = player.choose_move(g)
    assert move in g.get_legal_moves()
    assert player.last_node_count > 1


def test_non_mcts_players_have_zero_node_count():
    assert RandomPlayer().last_node_count == 0
    assert HeuristicPlayer().last_node_count == 0


# ──────────────────────────────────────── game log replay

def test_game_log_roundtrip(tmp_path):
    p1 = RandomPlayer(seed=1)
    p2 = RandomPlayer(seed=2)
    result = play_game(p1, p2, max_moves=80)

    path = tmp_path / "game.txt"
    write_game_log(str(path), {"max_moves": 80, "winner": "x"},
                   result["move_sequence"])

    replayed = replay_game_log(str(path))
    # replaying the recorded moves reproduces the exact final outcome
    assert replayed.move_count == result["moves"]
    assert replayed.get_winner() == result["winner"]
