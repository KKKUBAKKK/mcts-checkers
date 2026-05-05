"""Flask web application for human-vs-AI checkers."""

from __future__ import annotations

import os
from flask import Flask, jsonify, render_template, request

from game import CheckersGame
from players import MCTSPlayer, HeuristicPlayer, RandomPlayer

app = Flask(__name__)

# ── global game state (single-user server) ──────────────────────────────────
game: CheckersGame = CheckersGame()
ai_player: MCTSPlayer | HeuristicPlayer | RandomPlayer = MCTSPlayer(
    variant="uct", iterations=2000, parallel=True,
)
human_color: int = 1  # WHITE by default


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def state():
    return jsonify(game.to_dict() | {"human_color": human_color})


@app.route("/api/new_game", methods=["POST"])
def new_game():
    global game, ai_player, human_color
    data = request.get_json(force=True, silent=True) or {}

    human_color = data.get("human_color", 1)
    variant = data.get("variant", "uct")
    iterations = int(data.get("iterations", 2000))
    max_moves = int(data.get("max_moves", 200))

    game = CheckersGame(max_moves=max_moves)

    if variant == "heuristic":
        ai_player = HeuristicPlayer()
    elif variant == "random":
        ai_player = RandomPlayer()
    else:
        ai_player = MCTSPlayer(
            variant=variant, iterations=iterations, parallel=True,
        )

    resp = game.to_dict() | {"human_color": human_color}

    # If AI goes first, make its move immediately
    if game.current_player != human_color and not game.is_terminal():
        move = ai_player.choose_move(game)
        game.make_move(move)
        resp = game.to_dict() | {"human_color": human_color, "ai_move": [list(p) for p in move]}

    return jsonify(resp)


@app.route("/api/move", methods=["POST"])
def human_move():
    global game
    data = request.get_json(force=True)
    move_raw = data.get("move")
    if move_raw is None:
        return jsonify({"error": "no move provided"}), 400

    move = [tuple(p) for p in move_raw]

    # validate
    legal = game.get_legal_moves()
    if move not in legal:
        return jsonify({"error": "illegal move", "legal": [list(map(list, m)) for m in legal]}), 400

    game.make_move(move)
    resp = game.to_dict() | {"human_color": human_color}
    return jsonify(resp)


@app.route("/api/ai_move", methods=["POST"])
def ai_move():
    global game
    if game.is_terminal():
        return jsonify(game.to_dict() | {"human_color": human_color})

    move = ai_player.choose_move(game)
    game.make_move(move)
    resp = game.to_dict() | {"human_color": human_color, "ai_move": [list(p) for p in move]}
    return jsonify(resp)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
