"""Unit tests for the 10x10 checkers move generator and rules.

Covers the custom rules of this project:
- no mandatory capture (captures are optional)
- flying kings (move/capture along the whole diagonal)
- chain captures, promotion and terminal detection.

Run with:  python -m pytest tests/ -q
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game import (  # noqa: E402
    CheckersGame, BOARD_SIZE,
    WHITE, BLACK, EMPTY,
    WHITE_PIECE, WHITE_KING, BLACK_PIECE, BLACK_KING,
)


def empty_game(current_player=WHITE, max_moves=200):
    """A game with a cleared board, for constructing test positions."""
    g = CheckersGame(max_moves=max_moves)
    g.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    g.current_player = current_player
    return g


def is_capture(move):
    """A move is a capture iff any step jumps more than one square."""
    (r0, c0), (r1, c1) = move[0], move[1]
    return abs(r1 - r0) > 1 or len(move) > 2 or abs(c1 - c0) > 1


# ──────────────────────────────────────── initial position

def test_initial_board_counts():
    g = CheckersGame()
    whites = sum(1 for row in g.board for p in row if p == WHITE_PIECE)
    blacks = sum(1 for row in g.board for p in row if p == BLACK_PIECE)
    assert whites == 20
    assert blacks == 20
    assert g.current_player == WHITE


def test_initial_moves_are_simple_forward_only():
    g = CheckersGame()
    moves = g.get_legal_moves()
    # opening position has no captures available
    assert all(not is_capture(m) for m in moves)
    # white front rank (row 6) has pieces on odd columns: 1,3,5,7,9
    # giving 2+2+2+2+1 = 9 forward moves
    assert len(moves) == 9


# ──────────────────────────────────────── regular piece captures

def test_single_piece_capture_available_and_applied():
    g = empty_game(WHITE)
    g.board[5][4] = WHITE_PIECE
    g.board[4][3] = BLACK_PIECE  # landing (3,2) is empty
    moves = g.get_legal_moves()
    cap = [(5, 4), (3, 2)]
    assert cap in moves
    g.make_move(cap)
    assert g.board[3][2] == WHITE_PIECE
    assert g.board[4][3] == EMPTY  # captured
    assert g.board[5][4] == EMPTY


def test_capture_is_optional():
    """No mandatory capture: simple moves coexist with available captures."""
    g = empty_game(WHITE)
    g.board[5][4] = WHITE_PIECE
    g.board[4][3] = BLACK_PIECE
    moves = g.get_legal_moves()
    assert any(is_capture(m) for m in moves)       # capture exists
    assert any(not is_capture(m) for m in moves)   # but is not forced


def test_chain_capture():
    g = empty_game(WHITE)
    g.board[5][4] = WHITE_PIECE
    g.board[4][3] = BLACK_PIECE   # capture -> land (3,2)
    g.board[2][3] = BLACK_PIECE   # then capture -> land (1,4)
    moves = g.get_legal_moves()
    chain = [(5, 4), (3, 2), (1, 4)]
    assert chain in moves
    g.make_move(chain)
    assert g.board[1][4] == WHITE_PIECE
    assert g.board[4][3] == EMPTY
    assert g.board[2][3] == EMPTY


def test_regular_piece_captures_backwards():
    """Regular pieces may capture in all 4 directions (here backwards)."""
    g = empty_game(WHITE)
    g.board[5][4] = WHITE_PIECE
    g.board[6][5] = BLACK_PIECE   # behind the white piece; land (7,6) empty
    moves = g.get_legal_moves()
    assert [(5, 4), (7, 6)] in moves


# ──────────────────────────────────────── flying king

def test_flying_king_moves_along_whole_diagonal():
    g = empty_game(WHITE)
    g.board[5][5] = WHITE_KING
    moves = g.get_legal_moves()
    # king can slide all the way to a corner
    assert [(5, 5), (0, 0)] in moves
    assert [(5, 5), (9, 9)] in moves


def test_flying_king_capture_multiple_landing_squares():
    g = empty_game(WHITE)
    g.board[5][5] = WHITE_KING
    g.board[2][2] = BLACK_PIECE   # empty squares before and after
    moves = g.get_legal_moves()
    # king may land on any empty square beyond the captured piece
    assert [(5, 5), (1, 1)] in moves
    assert [(5, 5), (0, 0)] in moves
    g.make_move([(5, 5), (0, 0)])
    assert g.board[0][0] == WHITE_KING
    assert g.board[2][2] == EMPTY


# ──────────────────────────────────────── promotion

def test_promotion_to_king():
    g = empty_game(WHITE)
    g.board[1][2] = WHITE_PIECE
    g.board[9][9] = BLACK_PIECE   # keep both colours on the board
    g.make_move([(1, 2), (0, 1)])
    assert g.board[0][1] == WHITE_KING


def test_black_promotion_to_king():
    g = empty_game(BLACK)
    g.board[8][3] = BLACK_PIECE
    g.board[0][0] = WHITE_PIECE
    g.make_move([(8, 3), (9, 4)])
    assert g.board[9][4] == BLACK_KING


# ──────────────────────────────────────── terminal detection

def test_winner_when_opponent_has_no_pieces():
    g = empty_game(WHITE)
    g.board[5][5] = BLACK_PIECE   # only black remains
    assert g.get_winner() == BLACK
    assert g.is_terminal()


def test_winner_when_player_has_no_moves():
    g = empty_game(WHITE)
    g.board[0][0] = WHITE_PIECE   # on top rank, cannot move forward/capture
    g.board[5][5] = BLACK_PIECE
    assert g.get_legal_moves() == []
    assert g.get_winner() == BLACK  # white to move but stuck -> black wins


def test_draw_on_max_moves():
    g = CheckersGame(max_moves=10)
    g.move_count = 10
    assert g.get_winner() == 0
    assert g.is_terminal()


# ──────────────────────────────────────── clone independence

def test_clone_is_independent():
    g = CheckersGame()
    h = g.clone()
    h.make_move(h.get_legal_moves()[0])
    assert g.move_count == 0
    assert h.move_count == 1
