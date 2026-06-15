"""Cross-hypothesis summary plots for the presentation.

Reads the per-hypothesis ``output/h{1,2,3}/stats.json`` (win-rate sweeps over
MCTS iteration budgets) and ``output/h4/stats.json`` and produces two
overview figures under ``output/summary/``:

1. ``summary_winrate_vs_iterations.png`` -- overlays the H1/H2/H3 win-rate
   curves (each "modification vs. its baseline") on one log-x axis, with
   per-point 95% Wilson CI error bars and the chance/threshold reference
   lines. One slide that contrasts how the three approaches scale with the
   iteration budget.
2. ``summary_headline_bar.png`` -- one bar per hypothesis at a representative
   budget (the modification's win rate vs. its baseline) with 95% CI error
   bars; the H4 custom hypothesis contributes its two configurations. A
   single "scoreboard" slide.

Reference lines: chance = 0.5, threshold = 0.6 (H1-H3) / 0.5 (H4).

Usage
-----
    ./venv/bin/python plot_summary.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "output/summary"

# One colour per hypothesis, reused across both figures.
COLORS = {"h1": "#4C72B0", "h2": "#C44E52", "h3": "#55A868", "h4": "#8172B3"}

# Human-readable "modification vs baseline" labels.
PAIR_LABEL = {
    "h1": "H1: UCT vs heuristic",
    "h2": "H2: RAVE vs UCT",
    "h3": "H3: progressive bias vs UCT",
}


def load_stats(hypothesis: str) -> dict:
    with open(f"output/{hypothesis}/stats.json", encoding="utf-8") as f:
        return json.load(f)


def plot_combined_curve(output_dir: str) -> str:
    """Overlay the H1/H2/H3 win-rate-vs-iterations sweeps on one axis."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for h in ("h1", "h2", "h3"):
        configs = sorted(load_stats(h)["configs"], key=lambda c: c["iterations"])
        xs = [c["iterations"] for c in configs]
        ys = [c["p1_win_rate"] for c in configs]
        err_lo = [c["p1_win_rate"] - c["ci95_lo"] for c in configs]
        err_hi = [c["ci95_hi"] - c["p1_win_rate"] for c in configs]
        ax.errorbar(xs, ys, yerr=[err_lo, err_hi], marker="o", capsize=3,
                    color=COLORS[h], linewidth=2, elinewidth=1, alpha=0.9,
                    label=PAIR_LABEL[h])

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.axhline(0.6, color="red", linestyle="--", linewidth=1, label="threshold (60%)")

    ax.set_xscale("log")
    ax.set_xlabel("MCTS iterations per move")
    ax.set_ylabel("modification win rate vs. baseline")
    ax.set_ylim(0, 1)
    ax.set_title("MCTS modifications: win rate vs. iteration budget\n"
                 "(N=5 games/point -- wide 95% CI, preliminary)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    path = os.path.join(output_dir, "summary_winrate_vs_iterations.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_headline_bar(output_dir: str) -> str:
    """One representative bar per hypothesis (+ both H4 configs)."""
    # (label, win_rate, ci_lo, ci_hi, hypothesis-key-for-colour, threshold)
    bars: list[tuple[str, float, float, float, str, float]] = []

    # H1/H2/H3: pick the representative budget for the slide.
    rep_iters = {"h1": 5000, "h2": 1000, "h3": 2000}
    rep_label = {
        "h1": "H1: UCT vs heur.\n(5000 it.)",
        "h2": "H2: RAVE vs UCT\n(1000 it., best)",
        "h3": "H3: progr. vs UCT\n(2000 it.)",
    }
    for h in ("h1", "h2", "h3"):
        c = next(c for c in load_stats(h)["configs"]
                 if c["iterations"] == rep_iters[h])
        bars.append((rep_label[h], c["p1_win_rate"], c["ci95_lo"],
                     c["ci95_hi"], h, 0.6))

    # H4: both configs (edge as p1).
    h4 = load_stats("h4")["configs"]
    h4_label = {
        "heuristic_edge_vs_center": "H4: edge vs center\n(heuristic)",
        "progressive_edge_vs_center_2000": "H4: edge vs center\n(progressive)",
    }
    for c in h4:
        bars.append((h4_label.get(c["name"], c["name"]), c["p1_win_rate"],
                     c["ci95_lo"], c["ci95_hi"], "h4", 0.5))

    labels = [b[0] for b in bars]
    wr = [b[1] for b in bars]
    err_lo = [b[1] - b[2] for b in bars]
    err_hi = [b[3] - b[1] for b in bars]
    colors = [COLORS[b[4]] for b in bars]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(bars))
    ax.bar(x, wr, yerr=[err_lo, err_hi], capsize=4, color=colors, alpha=0.85)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.axhline(0.6, color="red", linestyle="--", linewidth=1, label="threshold H1-H3 (60%)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("win rate (first player) with 95% CI")
    ax.set_title("Headline results across hypotheses\n"
                 "(N=5 games/config, N=10 for H4 -- preliminary, no result reaches p<0.05)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    path = os.path.join(output_dir, "summary_headline_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for path in (plot_combined_curve(OUTPUT_DIR), plot_headline_bar(OUTPUT_DIR)):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
