"""Experiment runner: automated games between AI players with seed control.

Usage examples
--------------
    # Quick test: 10 games, UCT vs heuristic
    python experiments.py --p1 uct --p2 heuristic -n 10

    # Full experiment: UCT vs RAVE, 50 games, seed 42
    python experiments.py --p1 uct --p2 rave -n 50 --seed 42 --iterations 3000

    # All pairs round-robin
    python experiments.py --round-robin -n 20 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

from game import CheckersGame, WHITE, BLACK
from players import MCTSPlayer, HeuristicPlayer, RandomPlayer


PLAYER_TYPES = ["uct", "rave", "progressive", "heuristic", "random"]

_WINNER_NAME = {WHITE: "white", BLACK: "black", 0: "draw", None: "draw"}


# ──────────────────────────────────────── game logging / replay

def _serialize_move(move: list[tuple[int, int]]) -> str:
    """Serialize a move (path of board squares) as space-separated 'r,c' tokens."""
    return " ".join(f"{r},{c}" for r, c in move)


def _parse_move(line: str) -> list[tuple[int, int]]:
    out = []
    for token in line.split():
        r, c = token.split(",")
        out.append((int(r), int(c)))
    return out


def write_game_log(path: str, meta: dict,
                   move_sequence: list[list[tuple[int, int]]]) -> None:
    """Write a single game to a human-readable, replayable text file.

    Header lines are ``key: value`` pairs (metadata + seeds for
    reproducibility); after a ``moves:`` marker each line is one move encoded
    as space-separated ``row,col`` board squares. See :func:`replay_game_log`.
    """
    lines = ["# Checkers 10x10 game record",
             "# rules: no_mandatory_capture, flying_kings"]
    for key, value in meta.items():
        lines.append(f"{key}: {value}")
    lines.append("moves:")
    lines.extend(_serialize_move(m) for m in move_sequence)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def replay_game_log(path: str) -> CheckersGame:
    """Re-apply a recorded game's moves and return the final game state.

    Used to verify that a logged game reproduces deterministically.
    """
    with open(path, encoding="utf-8") as f:
        raw = [ln.rstrip("\n") for ln in f]

    meta: dict[str, str] = {}
    move_lines: list[str] = []
    in_moves = False
    for ln in raw:
        if not in_moves:
            if ln.startswith("#") or not ln.strip():
                continue
            if ln.strip() == "moves:":
                in_moves = True
                continue
            key, _, value = ln.partition(":")
            meta[key.strip()] = value.strip()
        elif ln.strip():
            move_lines.append(ln)

    max_moves = int(meta.get("max_moves", 200))
    game = CheckersGame(max_moves=max_moves)
    for ml in move_lines:
        game.make_move(_parse_move(ml))
    return game


def make_player(name: str, iterations: int, seed: int | None):
    if name == "heuristic":
        return HeuristicPlayer(seed=seed)
    if name == "random":
        return RandomPlayer(seed=seed)
    return MCTSPlayer(variant=name, iterations=iterations, parallel=True, seed=seed)


def play_game(p1, p2, max_moves: int = 200, verbose: bool = False) -> dict:
    """Play a single game. p1 is WHITE, p2 is BLACK. Returns result dict.

    The result captures the full move sequence (for replay) plus the average
    number of tree nodes each player visited per move (search-effort metric;
    0 for non-MCTS players).
    """
    game = CheckersGame(max_moves=max_moves)
    t0 = time.time()

    move_sequence: list[list[tuple[int, int]]] = []
    white_nodes: list[int] = []
    black_nodes: list[int] = []

    while not game.is_terminal():
        if game.current_player == WHITE:
            move = p1.choose_move(game)
            white_nodes.append(getattr(p1, "last_node_count", 0))
        else:
            move = p2.choose_move(game)
            black_nodes.append(getattr(p2, "last_node_count", 0))
        move_sequence.append([tuple(p) for p in move])
        game.make_move(move)
        if verbose:
            print(f"  Move {game.move_count}: {'W' if game.current_player == BLACK else 'B'} -> {move}")

    elapsed = time.time() - t0
    winner = game.get_winner()

    def _avg(xs: list[int]) -> float:
        return round(sum(xs) / len(xs), 1) if xs else 0.0

    return {
        "winner": winner,
        "moves": game.move_count,
        "time": round(elapsed, 2),
        "move_sequence": move_sequence,
        "white_avg_nodes": _avg(white_nodes),
        "black_avg_nodes": _avg(black_nodes),
    }


def run_match(p1_name: str, p2_name: str, num_games: int,
              iterations: int, base_seed: int | None,
              max_moves: int, verbose: bool,
              log_dir: str | None = None, run_tag: str = "") -> dict:
    """Run a match of num_games between two player types.

    Each player plays both colors. For each pair of games (i*2, i*2+1),
    colors are swapped. When ``log_dir`` is set, every game is written to a
    replayable text log under that directory.
    """
    p1_wins = p2_wins = draws = 0
    total_moves = 0
    total_time = 0.0
    p1_nodes_total = 0.0
    p2_nodes_total = 0.0
    results = []

    for i in range(num_games):
        seed_i = (base_seed + i * 100) if base_seed is not None else None
        black_seed = (seed_i + 1) if seed_i is not None else None
        # alternate colors
        if i % 2 == 0:
            white_name, black_name = p1_name, p2_name
        else:
            white_name, black_name = p2_name, p1_name
        white = make_player(white_name, iterations, seed_i)
        black = make_player(black_name, iterations, black_seed)

        result = play_game(white, black, max_moves, verbose)
        w = result["winner"]

        # map color-based stats back to the logical p1/p2 players
        if i % 2 == 0:
            p1_nodes_total += result["white_avg_nodes"]
            p2_nodes_total += result["black_avg_nodes"]
            if w == WHITE:
                p1_wins += 1
            elif w == BLACK:
                p2_wins += 1
            else:
                draws += 1
        else:
            p1_nodes_total += result["black_avg_nodes"]
            p2_nodes_total += result["white_avg_nodes"]
            if w == WHITE:
                p2_wins += 1
            elif w == BLACK:
                p1_wins += 1
            else:
                draws += 1

        total_moves += result["moves"]
        total_time += result["time"]
        results.append({
            "game": i + 1,
            "white": white_name,
            "black": black_name,
            "winner_color": _WINNER_NAME.get(w, "draw"),
            "moves": result["moves"],
            "time_s": result["time"],
            "white_avg_nodes": result["white_avg_nodes"],
            "black_avg_nodes": result["black_avg_nodes"],
        })

        if log_dir is not None:
            meta = {
                "white_player": white_name,
                "white_seed": seed_i,
                "black_player": black_name,
                "black_seed": black_seed,
                "iterations": iterations,
                "max_moves": max_moves,
                "winner": _WINNER_NAME.get(w, "draw"),
                "total_moves": result["moves"],
                "white_avg_nodes": result["white_avg_nodes"],
                "black_avg_nodes": result["black_avg_nodes"],
            }
            fname = f"{run_tag}{p1_name}_vs_{p2_name}_g{i+1:03d}.txt"
            write_game_log(os.path.join(log_dir, fname), meta,
                           result["move_sequence"])

        tag = "W" if (w == WHITE) else ("B" if w == BLACK else "D")
        print(f"  Game {i+1}/{num_games}: {white_name}(W) vs {black_name}(B) "
              f"-> {tag} in {result['moves']} moves ({result['time']:.1f}s)")

    summary = {
        "p1": p1_name,
        "p2": p2_name,
        "games": num_games,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "p1_win_rate": round(p1_wins / num_games, 3),
        "avg_moves": round(total_moves / num_games, 1),
        "avg_time_s": round(total_time / num_games, 1),
        "p1_avg_nodes": round(p1_nodes_total / num_games, 1),
        "p2_avg_nodes": round(p2_nodes_total / num_games, 1),
        "game_results": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Checkers MCTS experiments")
    parser.add_argument("--p1", type=str, default="uct", choices=PLAYER_TYPES)
    parser.add_argument("--p2", type=str, default="heuristic", choices=PLAYER_TYPES)
    parser.add_argument("-n", "--num-games", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=2000,
                        help="MCTS iterations per move (for MCTS players)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--round-robin", action="store_true",
                        help="Run all pairs of player types")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save per-game results to CSV file")
    parser.add_argument("--log-dir", type=str, default="output/games",
                        help="Directory for replayable per-game logs "
                             "(set to '' or 'none' to disable)")
    args = parser.parse_args()

    log_dir: str | None = args.log_dir
    if log_dir in ("", "none", "None"):
        log_dir = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
    run_tag = time.strftime("%Y%m%d_%H%M%S_")

    all_summaries = []

    if args.round_robin:
        for i, p1 in enumerate(PLAYER_TYPES):
            for p2 in PLAYER_TYPES[i + 1:]:
                print(f"\n{'='*60}")
                print(f"Match: {p1} vs {p2}")
                print(f"{'='*60}")
                summary = run_match(p1, p2, args.num_games, args.iterations,
                                    args.seed, args.max_moves, args.verbose,
                                    log_dir, run_tag)
                all_summaries.append(summary)
                print(f"\n  Result: {p1} {summary['p1_wins']}W / "
                      f"{p2} {summary['p2_wins']}W / {summary['draws']}D")
    else:
        print(f"\nMatch: {args.p1} vs {args.p2} ({args.num_games} games)")
        print(f"{'='*60}")
        summary = run_match(args.p1, args.p2, args.num_games, args.iterations,
                            args.seed, args.max_moves, args.verbose,
                            log_dir, run_tag)
        all_summaries.append(summary)
        print(f"\n  Result: {args.p1} {summary['p1_wins']}W / "
              f"{args.p2} {summary['p2_wins']}W / {summary['draws']}D")
        print(f"  {args.p1} win rate: {summary['p1_win_rate']:.1%}")
        print(f"  Avg moves: {summary['avg_moves']}, Avg time: {summary['avg_time_s']}s")
        print(f"  Avg nodes/move: {args.p1}={summary['p1_avg_nodes']}, "
              f"{args.p2}={summary['p2_avg_nodes']}")
    if log_dir is not None:
        print(f"\nPer-game logs written to {log_dir}/")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nResults saved to {args.output}")

    if args.csv:
        rows = []
        for s in all_summaries:
            for gr in s["game_results"]:
                rows.append(gr)
        if rows:
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"Per-game CSV saved to {args.csv}")


if __name__ == "__main__":
    main()
