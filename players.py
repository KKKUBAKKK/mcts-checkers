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

from game import CheckersGame, WHITE, BLACK
from mcts import mcts_search, mcts_search_parallel, evaluate


class RandomPlayer:
    last_node_count: int = 0

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose_move(self, game: CheckersGame) -> list[tuple[int, int]]:
        moves = game.get_legal_moves()
        return self.rng.choice(moves)


class HeuristicPlayer:
    """One-ply greedy player using the shared static evaluator.

    Uses exactly the same evaluation function ``mcts.evaluate`` that the
    progressive-bias MCTS variant uses as its heuristic term ``h(state)``,
    so that comparisons between the heuristic player and progressive bias are
    consistent (per the project konspekt). The evaluation is a linear
    combination of material (piece = 1.0, king = 3.0) and a small advancement
    bonus, returned normalised as ``white / (white + black)`` in ``[0, 1]``.
    """

    last_node_count: int = 0

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
            # evaluate() is from WHITE's perspective; flip for BLACK so that a
            # higher score always means "better for the player to move".
            score = evaluate(g)
            if player == BLACK:
                score = 1.0 - score
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return self.rng.choice(best_moves)


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
        # number of tree nodes visited during the most recent choose_move call
        self.last_node_count: int = 0

    def choose_move(self, game: CheckersGame) -> list[tuple[int, int]]:
        move_seed = None
        if self.seed is not None:
            move_seed = self.seed + self._call_count * 1000
        self._call_count += 1

        if self.parallel and self.num_workers > 1:
            move, nodes = mcts_search_parallel(
                game, self.iterations, self.variant, self.c,
                seed=move_seed, num_workers=self.num_workers,
            )
        else:
            move, nodes = mcts_search(
                game, self.iterations, self.variant, self.c, seed=move_seed,
            )
        self.last_node_count = nodes
        return move
