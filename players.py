"""Player abstractions for checkers.

Players
-------
- RandomPlayer       : picks a random legal move
- HeuristicPlayer    : greedy one-ply lookahead using a static evaluator
- MCTSPlayer         : configurable MCTS (variant, iterations, concurrency)
"""

from __future__ import annotations

import os
import random
from typing import Optional

from game import (
    CheckersGame, WHITE, BLACK, BOARD_SIZE,
    WHITE_PIECE, WHITE_KING, BLACK_PIECE, BLACK_KING,
)
from mcts import mcts_search, mcts_search_parallel, evaluate


class RandomPlayer:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose_move(self, game: CheckersGame) -> list[tuple[int, int]]:
        moves = game.get_legal_moves()
        return self.rng.choice(moves)


class HeuristicPlayer:
    """One-ply greedy player using piece-value + positional evaluation.

    Evaluation favours:
    - Material (kings worth 3x a piece)
    - Advancement (pieces closer to promotion)
    - Center control
    - King mobility
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose_move(self, game: CheckersGame) -> list[tuple[int, int]]:
        moves = game.get_legal_moves()
        player = game.current_player
        best_score = -float("inf")
        best_moves: list[list[tuple[int, int]]] = []

        for move in moves:
            g = game.clone()
            g.make_move(move)
            score = self._evaluate(g, player)
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return self.rng.choice(best_moves)

    @staticmethod
    def _evaluate(game: CheckersGame, player: int) -> float:
        score = 0.0
        center = BOARD_SIZE / 2 - 0.5
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = game.board[r][c]
                if p == 0:
                    continue
                piece_player = WHITE if p > 0 else BLACK
                sign = 1.0 if piece_player == player else -1.0
                # material
                if abs(p) == 1:
                    val = 1.0
                    # advancement bonus
                    if piece_player == WHITE:
                        val += 0.1 * (BOARD_SIZE - 1 - r)
                    else:
                        val += 0.1 * r
                else:
                    val = 3.0
                # center control
                dist = abs(c - center) + abs(r - center)
                val += 0.05 * (BOARD_SIZE - dist)
                score += sign * val
        return score


class MCTSPlayer:
    """Configurable MCTS player.

    Parameters
    ----------
    variant : str
        ``"uct"``, ``"rave"``, or ``"progressive"``
    iterations : int
        Number of MCTS iterations per move
    parallel : bool
        Use root-parallel search across multiple processes
    num_workers : int
        Number of parallel workers (defaults to CPU count)
    c : float
        Exploration constant
    seed : int | None
        Base random seed for reproducibility
    """

    def __init__(self, variant: str = "uct", iterations: int = 2000,
                 parallel: bool = True, num_workers: int | None = None,
                 c: float = 1.414, seed: int | None = None):
        self.variant = variant
        self.iterations = iterations
        self.parallel = parallel
        self.num_workers = num_workers or max(1, os.cpu_count() or 1)
        self.c = c
        self.seed = seed
        self._call_count = 0

    def choose_move(self, game: CheckersGame) -> list[tuple[int, int]]:
        move_seed = None
        if self.seed is not None:
            move_seed = self.seed + self._call_count * 1000
        self._call_count += 1

        if self.parallel and self.num_workers > 1:
            return mcts_search_parallel(
                game, self.iterations, self.variant, self.c,
                seed=move_seed, num_workers=self.num_workers,
            )
        return mcts_search(
            game, self.iterations, self.variant, self.c, seed=move_seed,
        )
