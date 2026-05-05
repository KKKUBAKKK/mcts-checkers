# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University research project implementing Monte Carlo Tree Search (MCTS) / Upper Confidence Bound applied to Trees (UCT) as AI for a two-player perfect-information game: **10x10 checkers** with custom rules:
- 10x10 board
- **No mandatory capture** (capturing is optional)
- Kings (damki) capture along the **entire diagonal** (flying kings)

This is a research project — the code is a means to run experiments, not the deliverable itself. The deliverables are a research outline (konspekt), final report, and presentation.

## Setup and Commands

```bash
# Create venv and install dependencies
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# Run web UI (human vs AI) — open http://localhost:8765 in browser
PORT=8765 ./venv/bin/python app.py

# Run experiments (automated AI vs AI games)
./venv/bin/python experiments.py --p1 uct --p2 heuristic -n 10 --seed 42
./venv/bin/python experiments.py --round-robin -n 20 --seed 0 --output results.json --csv results.csv
```

## Architecture

- **game.py** — Game engine: board state, move generation (regular pieces + flying kings), chain captures, promotion, terminal detection. All state in `CheckersGame` class.
- **mcts.py** — MCTS with three UCT variants (`uct`, `rave`, `progressive`) and root-parallel concurrency via `multiprocessing.ProcessPoolExecutor`. Contains shared heuristic `evaluate()`.
- **players.py** — Player abstractions: `RandomPlayer`, `HeuristicPlayer` (greedy one-ply), `MCTSPlayer` (configurable variant/iterations/parallelism/seed).
- **app.py** — Flask web server with REST API (`/api/state`, `/api/new_game`, `/api/move`, `/api/ai_move`).
- **experiments.py** — CLI experiment runner with color-swapping, seed control, JSON/CSV output.
- **templates/index.html + static/** — Browser-based UI with click-to-move interaction.

## Research Requirements

The project compares four player strategies:
1. Baseline UCT (no modifications)
2. UCT + RAVE (Rapid Action Value Estimation) — from literature
3. UCT + Progressive Bias — from literature
4. Heuristic evaluator (greedy, material + positional)

Additionally: one custom hypothesis related to checkers rules, and human-vs-AI testing with at least 5 people.

## Key Design Constraints

- **Reproducibility**: All players accept a `seed` parameter. Experiments alternate colors and use deterministic seeds derived from a base seed.
- **Concurrency**: MCTS uses root parallelization — independent trees across worker processes, aggregated visit counts.
- **Cross-platform**: Python + Flask + browser UI. Runs on macOS, Windows, Linux.
