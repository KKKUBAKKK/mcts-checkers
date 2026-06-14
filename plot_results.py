"""Stage 4 ("generowanie wykresów") of the H1-H4 hypothesis pipeline.

Reads ``stats.json`` (written by ``analyze_results.py``) and produces:

1. A bar chart of ``p1``'s win rate with 95% Wilson CI error bars per
   configuration, with reference lines at 0.5 (chance) and the hypothesis's
   win-rate threshold (0.6 for H1-H3, 0.5 for H4).
2. If the configurations sweep MCTS iteration budgets for a single (p1, p2)
   pair (more than one distinct ``iterations`` value), a line plot of win
   rate vs. iterations with a shaded 95% CI band.

PNGs are written to ``<output-dir>/``.

Usage
-----
    python plot_results.py --input output/h1/stats.json --output-dir output/h1/plots
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_bar_chart(configs: list[dict], hypothesis: str, threshold: float,
                    output_dir: str) -> str:
    names = [c["name"] for c in configs]
    win_rates = [c["p1_win_rate"] for c in configs]
    err_lo = [wr - c["ci95_lo"] for wr, c in zip(win_rates, configs)]
    err_hi = [c["ci95_hi"] - wr for wr, c in zip(win_rates, configs)]

    p1_name = configs[0]["p1"] if configs else "p1"
    p2_name = configs[0]["p2"] if configs else "p2"

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bars = ax.bar(names, win_rates, yerr=[err_lo, err_hi], capsize=4,
                   color="#4C72B0", alpha=0.85)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1,
               label=f"threshold ({threshold:.0%})")

    for bar, c in zip(bars, configs):
        if c["hypothesis_holds"]:
            bar.set_color("#55A868")

    ax.set_ylim(0, 1)
    ax.set_ylabel(f"{p1_name} win rate")
    ax.set_title(f"{hypothesis.upper()}: {p1_name} vs {p2_name} win rate "
                  f"(95% CI)")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"{hypothesis}_winrate_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_iterations_curve(configs: list[dict], hypothesis: str, threshold: float,
                            output_dir: str) -> str | None:
    # Only meaningful for a single, homogeneous (p1, p2) sweep over
    # iteration budgets (H1/H2/H3); H4 mixes heuristic and progressive
    # configs at different (or zero) iteration counts.
    pairs = {(c["p1"], c["p2"]) for c in configs}
    iterations = sorted({c["iterations"] for c in configs})
    if len(iterations) <= 1 or len(pairs) != 1:
        return None

    by_iter = sorted(configs, key=lambda c: c["iterations"])
    xs = [c["iterations"] for c in by_iter]
    ys = [c["p1_win_rate"] for c in by_iter]
    lo = [c["ci95_lo"] for c in by_iter]
    hi = [c["ci95_hi"] for c in by_iter]

    p1_name = configs[0]["p1"] if configs else "p1"
    p2_name = configs[0]["p2"] if configs else "p2"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, marker="o", color="#4C72B0", label=f"{p1_name} win rate")
    ax.fill_between(xs, lo, hi, color="#4C72B0", alpha=0.2, label="95% CI")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1,
               label=f"threshold ({threshold:.0%})")

    ax.set_xscale("log")
    ax.set_xlabel("MCTS iterations per move")
    ax.set_ylabel(f"{p1_name} win rate")
    ax.set_ylim(0, 1)
    ax.set_title(f"{hypothesis.upper()}: {p1_name} vs {p2_name} win rate "
                  f"vs. iteration budget")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, f"{hypothesis}_winrate_vs_iterations.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: plot generation for H1/H2")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to stats.json")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        stats = json.load(f)

    hypothesis = stats["hypothesis"]
    configs = stats["configs"]
    threshold = stats.get("win_rate_threshold", 0.6)

    os.makedirs(args.output_dir, exist_ok=True)

    if not configs:
        print("No configurations found in stats.json; nothing to plot.")
        return

    bar_path = plot_bar_chart(configs, hypothesis, threshold, args.output_dir)
    print(f"Wrote {bar_path}")

    curve_path = plot_iterations_curve(configs, hypothesis, threshold, args.output_dir)
    if curve_path:
        print(f"Wrote {curve_path}")
    else:
        print("Single iteration budget (or mixed config types); skipping "
              "win-rate-vs-iterations curve.")


if __name__ == "__main__":
    main()
