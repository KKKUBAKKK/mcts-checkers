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
import sys
import time

from game import CheckersGame, WHITE, BLACK
from players import MCTSPlayer, HeuristicPlayer, RandomPlayer


PLAYER_TYPES = ["uct", "rave", "progressive", "heuristic", "random"]


def make_player(name: str, iterations: int, seed: int | None):
    if name == "heuristic":
        return HeuristicPlayer(seed=seed)
    if name == "random":
        return RandomPlayer(seed=seed)
    return MCTSPlayer(variant=name, iterations=iterations, parallel=True, seed=seed)


def play_game(p1, p2, max_moves: int = 200, verbose: bool = False) -> dict:
    """Play a single game. p1 is WHITE, p2 is BLACK. Returns result dict."""
    game = CheckersGame(max_moves=max_moves)
    t0 = time.time()

    while not game.is_terminal():
        if game.current_player == WHITE:
            move = p1.choose_move(game)
        else:
            move = p2.choose_move(game)
        game.make_move(move)
        if verbose:
            print(f"  Move {game.move_count}: {'W' if game.current_player == BLACK else 'B'} -> {move}")

    elapsed = time.time() - t0
    winner = game.get_winner()
    return {
        "winner": winner,
        "moves": game.move_count,
        "time": round(elapsed, 2),
    }


def run_match(p1_name: str, p2_name: str, num_games: int,
              iterations: int, base_seed: int | None,
              max_moves: int, verbose: bool) -> dict:
    """Run a match of num_games between two player types.

    Each player plays both colors. For each pair of games (i*2, i*2+1),
    colors are swapped.
    """
    p1_wins = p2_wins = draws = 0
    total_moves = 0
    total_time = 0.0
    results = []

    for i in range(num_games):
        seed_i = (base_seed + i * 100) if base_seed is not None else None
        # alternate colors
        if i % 2 == 0:
            white_name, black_name = p1_name, p2_name
            white = make_player(p1_name, iterations, seed_i)
            black = make_player(p2_name, iterations, seed_i + 1 if seed_i is not None else None)
        else:
            white_name, black_name = p2_name, p1_name
            white = make_player(p2_name, iterations, seed_i)
            black = make_player(p1_name, iterations, seed_i + 1 if seed_i is not None else None)

        result = play_game(white, black, max_moves, verbose)
        w = result["winner"]

        if i % 2 == 0:
            if w == WHITE:
                p1_wins += 1
            elif w == BLACK:
                p2_wins += 1
            else:
                draws += 1
        else:
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
            "winner_color": {1: "white", -1: "black", 0: "draw", None: "draw"}.get(w, "draw"),
            "moves": result["moves"],
            "time_s": result["time"],
        })

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
    args = parser.parse_args()

    all_summaries = []

    if args.round_robin:
        for i, p1 in enumerate(PLAYER_TYPES):
            for p2 in PLAYER_TYPES[i + 1:]:
                print(f"\n{'='*60}")
                print(f"Match: {p1} vs {p2}")
                print(f"{'='*60}")
                summary = run_match(p1, p2, args.num_games, args.iterations,
                                    args.seed, args.max_moves, args.verbose)
                all_summaries.append(summary)
                print(f"\n  Result: {p1} {summary['p1_wins']}W / "
                      f"{p2} {summary['p2_wins']}W / {summary['draws']}D")
    else:
        print(f"\nMatch: {args.p1} vs {args.p2} ({args.num_games} games)")
        print(f"{'='*60}")
        summary = run_match(args.p1, args.p2, args.num_games, args.iterations,
                            args.seed, args.max_moves, args.verbose)
        all_summaries.append(summary)
        print(f"\n  Result: {args.p1} {summary['p1_wins']}W / "
              f"{args.p2} {summary['p2_wins']}W / {summary['draws']}D")
        print(f"  {args.p1} win rate: {summary['p1_win_rate']:.1%}")
        print(f"  Avg moves: {summary['avg_moves']}, Avg time: {summary['avg_time_s']}s")

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
