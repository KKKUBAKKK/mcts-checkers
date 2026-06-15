"""10x10 Checkers game engine.

Rules:
- 10x10 board, pieces on dark squares
- No mandatory capture (captures are optional)
- Kings (damki) are flying kings: move/capture along entire diagonal
- Regular pieces move forward only, capture in all 4 directions
- Promotion when ending a turn on the last row
- Win by eliminating all opponent pieces or leaving them with no moves
- Draw after max_moves total half-moves
"""

from __future__ import annotations

EMPTY = 0
WHITE_PIECE = 1
WHITE_KING = 2
BLACK_PIECE = -1
BLACK_KING = -2

WHITE = 1
BLACK = -1

BOARD_SIZE = 10

DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


class CheckersGame:
    __slots__ = ("board", "current_player", "move_count", "max_moves",
                 "captured_white", "captured_black", "captured")

    def __init__(self, max_moves: int = 200):
        self.board: list[list[int]] = self._init_board()
        self.current_player: int = WHITE
        self.move_count: int = 0
        self.max_moves: int = max_moves
        self.captured_white: list[int] = []
        self.captured_black: list[int] = []
        self.captured: list[int] = []

    @staticmethod
    def _init_board() -> list[list[int]]:
        board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for r in range(4):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1:
                    board[r][c] = BLACK_PIECE
        for r in range(6, BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1:
                    board[r][c] = WHITE_PIECE
        return board

    def clone(self) -> CheckersGame:
        g = CheckersGame.__new__(CheckersGame)
        g.board = [row[:] for row in self.board]
        g.current_player = self.current_player
        g.move_count = self.move_count
        g.max_moves = self.max_moves
        g.captured_white = self.captured_white[:]
        g.captured_black = self.captured_black[:]
        g.captured = self.captured[:]
        return g

    @staticmethod
    def _in_bounds(r: int, c: int) -> bool:
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def _is_own(self, r: int, c: int, player: int) -> bool:
        p = self.board[r][c]
        return (p > 0) if player == WHITE else (p < 0)

    def _is_opponent(self, r: int, c: int, player: int) -> bool:
        p = self.board[r][c]
        return (p < 0) if player == WHITE else (p > 0)

    def get_legal_moves(self) -> list[list[tuple[int, int]]]:
        """Return all legal moves for the current player.

        Each move is a list of board positions: [(r0,c0), (r1,c1), ...]
        - Length 2 for simple moves or single captures
        - Length >2 for chain captures
        """
        moves: list[list[tuple[int, int]]] = []
        player = self.current_player
        board = self.board
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if not self._is_own(r, c, player):
                    continue
                piece = board[r][c]
                if piece in (WHITE_KING, BLACK_KING):
                    self._king_simple_moves(r, c, moves)
                    self._king_captures(r, c, moves)
                else:
                    self._piece_simple_moves(r, c, player, moves)
                    self._piece_captures(r, c, moves)
        return moves

    def _piece_simple_moves(self, r: int, c: int, player: int,
                            out: list[list[tuple[int, int]]]) -> None:
        fwd = -1 if player == WHITE else 1
        for dc in (-1, 1):
            nr, nc = r + fwd, c + dc
            if self._in_bounds(nr, nc) and self.board[nr][nc] == EMPTY:
                out.append([(r, c), (nr, nc)])

    def _piece_captures(self, r: int, c: int,
                        out: list[list[tuple[int, int]]]) -> None:
        self._piece_cap_dfs(r, c, [(r, c)], set(), out)

    def _piece_cap_dfs(self, r: int, c: int,
                       path: list[tuple[int, int]],
                       captured: set[tuple[int, int]],
                       out: list[list[tuple[int, int]]]) -> None:
        found = False
        board = self.board
        player = self.current_player
        for dr, dc in DIRECTIONS:
            mr, mc = r + dr, c + dc
            lr, lc = r + 2 * dr, c + 2 * dc
            if (not self._in_bounds(lr, lc)
                    or (mr, mc) in captured
                    or not self._is_opponent(mr, mc, player)
                    or board[lr][lc] != EMPTY):
                continue
            found = True
            old_from = board[r][c]
            old_mid = board[mr][mc]
            board[r][c] = EMPTY
            board[mr][mc] = EMPTY
            board[lr][lc] = old_from
            self._piece_cap_dfs(lr, lc, path + [(lr, lc)],
                                captured | {(mr, mc)}, out)
            board[r][c] = old_from
            board[mr][mc] = old_mid
            board[lr][lc] = EMPTY
        if not found and len(path) > 1:
            out.append(list(path))

    def _king_simple_moves(self, r: int, c: int,
                           out: list[list[tuple[int, int]]]) -> None:
        board = self.board
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            while self._in_bounds(nr, nc) and board[nr][nc] == EMPTY:
                out.append([(r, c), (nr, nc)])
                nr += dr
                nc += dc

    def _king_captures(self, r: int, c: int,
                       out: list[list[tuple[int, int]]]) -> None:
        self._king_cap_dfs(r, c, [(r, c)], set(), out)

    def _king_cap_dfs(self, r: int, c: int,
                      path: list[tuple[int, int]],
                      captured: set[tuple[int, int]],
                      out: list[list[tuple[int, int]]]) -> None:
        found = False
        board = self.board
        player = self.current_player
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            while self._in_bounds(nr, nc) and board[nr][nc] == EMPTY:
                nr += dr
                nc += dc
            if (not self._in_bounds(nr, nc)
                    or not self._is_opponent(nr, nc, player)
                    or (nr, nc) in captured):
                continue
            mr, mc = nr, nc
            lr, lc = mr + dr, mc + dc
            while self._in_bounds(lr, lc) and board[lr][lc] == EMPTY:
                found = True
                old_king = board[r][c]
                old_mid = board[mr][mc]
                board[r][c] = EMPTY
                board[mr][mc] = EMPTY
                board[lr][lc] = old_king
                self._king_cap_dfs(lr, lc, path + [(lr, lc)],
                                   captured | {(mr, mc)}, out)
                board[r][c] = old_king
                board[mr][mc] = old_mid
                board[lr][lc] = EMPTY
                lr += dr
                lc += dc
        if not found and len(path) > 1:
            out.append(list(path))

    def make_move(self, move: list[tuple[int, int]]) -> None:
        board = self.board
        sr, sc = move[0]
        er, ec = move[-1]
        piece = board[sr][sc]

        for i in range(len(move) - 1):
            r1, c1 = move[i]
            r2, c2 = move[i + 1]
            dr = 1 if r2 > r1 else -1
            dc_dir = 1 if c2 > c1 else -1
            rr, cc = r1 + dr, c1 + dc_dir
            while (rr, cc) != (r2, c2):
                if board[rr][cc] != EMPTY:
                    captured_piece = board[rr][cc]
                    if captured_piece > 0:
                        self.captured_white.append(captured_piece)
                    else:
                        self.captured_black.append(captured_piece)
                    self.captured.append(captured_piece)
                    board[rr][cc] = EMPTY
                    break
                rr += dr
                cc += dc_dir

        board[sr][sc] = EMPTY

        if piece == WHITE_PIECE and er == 0:
            piece = WHITE_KING
        elif piece == BLACK_PIECE and er == BOARD_SIZE - 1:
            piece = BLACK_KING

        board[er][ec] = piece
        self.current_player = -self.current_player
        self.move_count += 1

    def get_winner(self) -> int | None:
        """Return WHITE, BLACK, 0 (draw), or None (ongoing)."""
        if self.move_count >= self.max_moves:
            return 0
        has_white = has_black = False
        for row in self.board:
            for p in row:
                if p > 0:
                    has_white = True
                elif p < 0:
                    has_black = True
                if has_white and has_black:
                    break
            if has_white and has_black:
                break
        if not has_white:
            return BLACK
        if not has_black:
            return WHITE
        if not self.get_legal_moves():
            return -self.current_player
        return None

    def is_terminal(self) -> bool:
        return self.get_winner() is not None

    def to_dict(self) -> dict:
        return {
            "board": self.board,
            "current_player": self.current_player,
            "move_count": self.move_count,
            "legal_moves": [list(map(list, m)) for m in self.get_legal_moves()],
            "winner": self.get_winner(),
            "captured_white": self.captured_white,
            "captured_black": self.captured_black,
            "captured": self.captured,
        }

