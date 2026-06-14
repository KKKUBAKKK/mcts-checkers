"""Tests for players, MCTS node counting and game-log replay."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game import CheckersGame, WHITE, EMPTY, BOARD_SIZE, WHITE_PIECE, BLACK_PIECE  # noqa: E402
from mcts import mcts_search, evaluate  # noqa: E402
from players import HeuristicPlayer, MCTSPlayer, RandomPlayer  # noqa: E402
from experiments import play_game, write_game_log, replay_game_log  # noqa: E402


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
    # symmetric start position evaluates to ~0.5 from white's perspective
    assert abs(evaluate(g) - 0.5) < 1e-9


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
