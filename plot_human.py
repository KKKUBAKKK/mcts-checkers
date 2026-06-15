"""Pie charts of human-vs-MCTS results (5 testers, one game each per variant).

Human wins: progressive bias 1/5, the other two variants 2/5. Transparent
background so the charts sit on either a light or dark slide.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCENT = "#b56a38"   # human (deck accent)
AI = "#2a2620"       # MCTS (deck dark piece)
N = 5

variants = [
    ("best",   "Najlepszy MCTS", 1),
    ("medium", "Średni MCTS",    2),
    ("worst",  "Najgorszy MCTS", 2),
]

OUT = "output/human"
os.makedirs(OUT, exist_ok=True)

for key, _title, human_wins in variants:
    ai_wins = N - human_wins
    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, _ = ax.pie(
        [human_wins, ai_wins],
        colors=[ACCENT, AI],
        startangle=90, counterclock=False,
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
    )
    # count labels centered in each slice, white for contrast
    for w, val in zip(wedges, [human_wins, ai_wins]):
        ang = (w.theta2 + w.theta1) / 2
        import math
        x = 0.6 * math.cos(math.radians(ang))
        y = 0.6 * math.sin(math.radians(ang))
        ax.text(x, y, str(val), ha="center", va="center",
                color="white", fontsize=34, fontweight="bold")
    ax.set(aspect="equal")
    fig.patch.set_alpha(0)
    fig.tight_layout(pad=0)
    path = os.path.join(OUT, f"human_{key}.png")
    fig.savefig(path, dpi=150, transparent=True)
    plt.close(fig)
    print(f"Wrote {path}  (człowiek {human_wins} / MCTS {ai_wins})")
