"""MCTS engine with UCT variants and root-parallel concurrency.

Variants
--------
- ``uct``      : vanilla UCB1 selection
- ``rave``     : UCT + RAVE (Rapid Action Value Estimation / AMAF)
- ``progressive`` : UCT + progressive bias from a heuristic evaluator
"""

from __future__ import annotations

import math
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from game import CheckersGame, WHITE, BLACK, BOARD_SIZE
from game import WHITE_PIECE, WHITE_KING, BLACK_PIECE, BLACK_KING, EMPTY


# Heuristic variants for `evaluate()`, injectable into HeuristicPlayer and the
# progressive-bias MCTS variant (see H4: edge-favouring vs. center-favouring
# positional evaluation).
HEURISTICS = ("base", "edge", "center")

_BOARD_CENTER = (BOARD_SIZE - 1) / 2.0


def _positional_bonus(c: int, heuristic: str) -> float:
    """Column-based positional bonus added to a piece's material value."""
    dist_from_center = abs(c - _BOARD_CENTER)
    if heuristic == "edge":
        return 0.05 * dist_from_center
    if heuristic == "center":
        return 0.05 * (_BOARD_CENTER - dist_from_center)
    return 0.0


def evaluate(game: CheckersGame, heuristic: str = "base") -> float:
    """Static evaluation from WHITE's perspective in [0, 1].

    ``heuristic`` selects a positional bonus (see :data:`HEURISTICS`):
    - ``"base"``   : material + advancement only (no positional term).
    - ``"edge"``   : additional bonus for pieces near the side columns.
    - ``"center"`` : additional bonus for pieces near the central columns.
    """
    white_score = black_score = 0.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = game.board[r][c]
            if p == EMPTY:
                continue
            bonus = _positional_bonus(c, heuristic)
            if p == WHITE_PIECE:
                white_score += 1.0 + 0.05 * (BOARD_SIZE - 1 - r) + bonus
            elif p == WHITE_KING:
                white_score += 3.0 + bonus
            elif p == BLACK_PIECE:
                black_score += 1.0 + 0.05 * r + bonus
            elif p == BLACK_KING:
                black_score += 3.0 + bonus
    total = white_score + black_score
    if total == 0:
        return 0.5
    return white_score / total


class MCTSNode:
    __slots__ = (
        "game", "parent", "move", "children",
        "visits", "value",
        "rave_visits", "rave_value",
        "untried_moves",
    )

    def __init__(self, game: CheckersGame, parent: Optional[MCTSNode] = None,
                 move: Optional[list[tuple[int, int]]] = None):
        self.game = game
        self.parent = parent
        self.move = move
        self.children: list[MCTSNode] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.rave_visits: int = 0
        self.rave_value: float = 0.0
        self.untried_moves: list[list[tuple[int, int]]] | None = None

    def _ensure_untried(self) -> None:
        if self.untried_moves is None:
            self.untried_moves = self.game.get_legal_moves()

    def is_fully_expanded(self) -> bool:
        self._ensure_untried()
        return len(self.untried_moves) == 0  # type: ignore[arg-type]

    def is_terminal(self) -> bool:
        return self.game.is_terminal()


def ucb1(node: MCTSNode, child: MCTSNode, c: float) -> float:
    exploit = child.value / child.visits
    explore = c * math.sqrt(math.log(node.visits) / child.visits)
    return exploit + explore


def ucb1_rave(node: MCTSNode, child: MCTSNode, c: float,
              b_param: float = 0.001) -> float:
    beta = child.rave_visits / (child.visits + child.rave_visits
                                + 4 * b_param * b_param * child.visits * child.rave_visits + 1e-9)
    q_mcts = child.value / max(child.visits, 1)
    q_rave = child.rave_value / max(child.rave_visits, 1)
    q = (1 - beta) * q_mcts + beta * q_rave
    explore = c * math.sqrt(math.log(node.visits) / child.visits)
    return q + explore


def ucb1_progressive(node: MCTSNode, child: MCTSNode, c: float,
                     w: float = 0.5, heuristic: str = "base") -> float:
    exploit = child.value / child.visits
    explore = c * math.sqrt(math.log(node.visits) / child.visits)
    # evaluate() is from WHITE's perspective; exploit/explore are from the
    # perspective of the player choosing among `node`'s children, so flip
    # the heuristic term to match when that player is BLACK.
    h = evaluate(child.game, heuristic)
    if node.game.current_player == BLACK:
        h = 1.0 - h
    bias = w * h / (child.visits + 1)
    return exploit + explore + bias


def _select(node: MCTSNode, variant: str, c: float,
            heuristic: str = "base") -> MCTSNode:
    while not node.is_terminal() and node.is_fully_expanded():
        if not node.children:
            break
        if variant == "rave":
            node = max(node.children, key=lambda ch: ucb1_rave(node, ch, c))
        elif variant == "progressive":
            node = max(node.children, key=lambda ch: ucb1_progressive(node, ch, c, heuristic=heuristic))
        else:
            node = max(node.children, key=lambda ch: ucb1(node, ch, c))
    return node


def _expand(node: MCTSNode) -> MCTSNode:
    node._ensure_untried()
    if not node.untried_moves:
        return node
    move = node.untried_moves.pop()  # type: ignore[union-attr]
    child_game = node.game.clone()
    child_game.make_move(move)
    child = MCTSNode(child_game, parent=node, move=move)
    node.children.append(child)
    return child


def _simulate(game: CheckersGame, rng: random.Random) -> float:
    """Random rollout. Returns result from WHITE's perspective: 1/0/0.5."""
    g = game.clone()
    depth = 0
    max_depth = 150
    while not g.is_terminal() and depth < max_depth:
        moves = g.get_legal_moves()
        if not moves:
            break
        g.make_move(rng.choice(moves))
        depth += 1
    winner = g.get_winner()
    if winner == WHITE:
        return 1.0
    elif winner == BLACK:
        return 0.0
    return 0.5


def _move_key(move: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(move)


def _count_nodes(root: MCTSNode) -> int:
    """Total number of nodes in the search tree (i.e. nodes visited/expanded)."""
    count = 0
    stack = [root]
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(node.children)
    return count


def _backpropagate(node: MCTSNode, result: float,
                   variant: str,
                   sim_moves: list[tuple[tuple[int, int], ...]] | None = None) -> None:
    """Backpropagate result. For RAVE, also update AMAF values."""
    cur: MCTSNode | None = node
    while cur is not None:
        cur.visits += 1
        # value is from the perspective of the player who MADE the move leading here
        # If the parent's player == WHITE, then a WHITE win (1.0) is good for this node
        if cur.parent is not None:
            parent_player = cur.parent.game.current_player
            cur.value += result if parent_player == WHITE else (1.0 - result)
        else:
            cur.value += result if cur.game.current_player == BLACK else (1.0 - result)

        if variant == "rave" and sim_moves is not None and cur.parent is not None:
            sim_set = set(sim_moves)
            for ch in cur.parent.children:
                if ch.move is not None and _move_key(ch.move) in sim_set:
                    ch.rave_visits += 1
                    parent_player = cur.parent.game.current_player
                    ch.rave_value += result if parent_player == WHITE else (1.0 - result)

        cur = cur.parent


def mcts_search(game: CheckersGame, iterations: int, variant: str = "uct",
                c: float = 1.414, seed: int | None = None,
                heuristic: str = "base"
                ) -> tuple[list[tuple[int, int]], int]:
    """Run MCTS and return ``(best_move, num_nodes)``.

    ``num_nodes`` is the number of tree nodes visited (expanded) during the
    search — used as a search-effort metric in experiments. ``heuristic``
    selects the positional evaluation variant (see :data:`HEURISTICS`) used
    by the ``"progressive"`` variant's bias term; ignored otherwise.
    """
    rng = random.Random(seed)
    root = MCTSNode(game.clone())

    for _ in range(iterations):
        node = _select(root, variant, c, heuristic)
        if not node.is_terminal():
            node = _expand(node)
        result = _simulate(node.game, rng)
        sim_moves = None
        if variant == "rave":
            sim_moves = []
            g2 = node.game.clone()
            depth = 0
            while not g2.is_terminal() and depth < 150:
                moves = g2.get_legal_moves()
                if not moves:
                    break
                m = rng.choice(moves)
                sim_moves.append(_move_key(m))
                g2.make_move(m)
                depth += 1
        _backpropagate(node, result, variant, sim_moves)

    if not root.children:
        moves = game.get_legal_moves()
        return (moves[0] if moves else []), _count_nodes(root)

    best = max(root.children, key=lambda ch: ch.visits)
    return best.move, _count_nodes(root)


def _worker(args: tuple) -> dict:
    """Run MCTS in a subprocess.

    Returns ``{"counts": {move_str: visits}, "nodes": int}`` — per-move visit
    counts (aggregated across workers for move selection) and the number of
    tree nodes this worker visited.
    """
    game_board, current_player, move_count, max_moves, iterations, variant, c, seed, heuristic = args
    game = CheckersGame.__new__(CheckersGame)
    game.board = game_board
    game.current_player = current_player
    game.move_count = move_count
    game.max_moves = max_moves
    game.captured_white = []
    game.captured_black = []
    game.captured = []

    rng = random.Random(seed)
    root = MCTSNode(game.clone())

    for _ in range(iterations):
        node = _select(root, variant, c, heuristic)
        if not node.is_terminal():
            node = _expand(node)
        result = _simulate(node.game, rng)
        sim_moves = None
        if variant == "rave":
            sim_moves = []
            g2 = node.game.clone()
            depth = 0
            while not g2.is_terminal() and depth < 150:
                moves = g2.get_legal_moves()
                if not moves:
                    break
                m = rng.choice(moves)
                sim_moves.append(_move_key(m))
                g2.make_move(m)
                depth += 1
        _backpropagate(node, result, variant, sim_moves)

    counts: dict[str, int] = {}
    for ch in root.children:
        key = str(ch.move)
        counts[key] = ch.visits
    return {"counts": counts, "nodes": _count_nodes(root)}


def mcts_search_parallel(game: CheckersGame, iterations: int,
                         variant: str = "uct", c: float = 1.414,
                         seed: int | None = None,
                         num_workers: int = 4,
                         heuristic: str = "base",
                         executor: Optional[ProcessPoolExecutor] = None,
                         ) -> tuple[list[tuple[int, int]], int]:
    """Root-parallel MCTS: split iterations across workers, aggregate visits.

    Returns ``(best_move, num_nodes)`` where ``num_nodes`` is the total number
    of tree nodes visited summed across all worker trees. ``heuristic``
    selects the positional evaluation variant (see :data:`HEURISTICS`) used
    by the ``"progressive"`` variant's bias term; ignored otherwise.

    ``executor``, if given, is a pre-created ``ProcessPoolExecutor`` (with at
    least ``num_workers`` workers) that is reused instead of spawning a new
    pool for this call. Creating a process pool is expensive (especially on
    Windows, which uses ``spawn``), so callers making many searches (e.g. one
    per move of a game) should pass a long-lived executor.
    """
    base_seed = seed if seed is not None else random.randint(0, 2**31)
    per_worker = iterations // num_workers
    remainder = iterations % num_workers

    args_list = []
    for i in range(num_workers):
        n = per_worker + (1 if i < remainder else 0)
        if n == 0:
            continue
        args_list.append((
            [row[:] for row in game.board],
            game.current_player,
            game.move_count,
            game.max_moves,
            n, variant, c, base_seed + i, heuristic,
        ))

    aggregated: dict[str, int] = {}
    total_nodes = 0
    if executor is not None:
        results = executor.map(_worker, args_list)
    else:
        with ProcessPoolExecutor(max_workers=len(args_list)) as local_executor:
            results = list(local_executor.map(_worker, args_list))
    for result in results:
        total_nodes += result["nodes"]
        for move_str, count in result["counts"].items():
            aggregated[move_str] = aggregated.get(move_str, 0) + count

    if not aggregated:
        moves = game.get_legal_moves()
        return (moves[0] if moves else []), total_nodes

    best_move_str = max(aggregated, key=lambda k: aggregated[k])

    legal = game.get_legal_moves()
    for m in legal:
        if str(m) == best_move_str:
            return m, total_nodes

    return (legal[0] if legal else []), total_nodes
