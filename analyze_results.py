"""Stage 2+3 ("statystyki" + "testy hipotez i istotności") of the H1-H4
hypothesis pipeline.

Reads ``raw_results.json`` (written by ``run_h1.py``/``run_h2.py``/
``run_h3.py``/``run_h4.py``), pools results per configuration across base
seeds, and computes:

- pooled win rate for ``p1`` (out of all games, draws counted as losses
  against the win-rate threshold) and a Wilson 95% confidence interval,
- an exact two-sided binomial test against ``p0=0.5`` on decisive games
  (draws excluded) -- the significance test,
- per-seed win rates, for a cross-seed consistency check,
- a verdict: the hypothesis ("p1 beats p2 by a wide enough, significant, and
  seed-consistent margin") holds iff win_rate > threshold AND p-value < 0.05
  AND p1 has the higher win rate in a strict majority of seeds.

The win-rate threshold is 0.6 for H1/H2/H3 (konspekt: ">60% wygranych") and
0.5 for H4 (konspekt: "wariant krawędzie osiąga >50% wygranych" in *both*
configurations). For H4, an extra top-level ``both_configs_hold`` field
records whether the edge-favouring side wins both configurations.

Output: ``<output-dir>/stats.json`` and ``<output-dir>/stats.csv``.

Next pipeline stage: ``plot_results.py``.

Usage
-----
    python analyze_results.py --input output/h1/raw_results.json \\
        --hypothesis h1 --output-dir output/h1
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from hypothesis_lib import binomial_two_sided_pvalue, wilson_ci

WIN_RATE_THRESHOLDS = {"h1": 0.6, "h2": 0.6, "h3": 0.6, "h4": 0.5}
ALPHA = 0.05


def analyze_config(entry: dict, win_rate_threshold: float = 0.6) -> dict:
    """Pool per-seed summaries for one configuration and run the hypothesis
    significance test against ``win_rate_threshold``."""
    config = entry["config"]
    per_seed = entry["per_seed"]

    games = p1_wins = p2_wins = draws = 0
    seed_rows = []
    seeds_p1_ahead = 0

    for ps in per_seed:
        s = ps["summary"]
        games += s["games"]
        p1_wins += s["p1_wins"]
        p2_wins += s["p2_wins"]
        draws += s["draws"]
        seed_win_rate = s["p1_wins"] / s["games"] if s["games"] else 0.0
        if seed_win_rate > 0.5:
            seeds_p1_ahead += 1
        seed_rows.append({
            "seed": ps["seed"],
            "games": s["games"],
            "p1_wins": s["p1_wins"],
            "p2_wins": s["p2_wins"],
            "draws": s["draws"],
            "p1_win_rate": round(seed_win_rate, 3),
        })

    decisive = p1_wins + p2_wins
    win_rate = p1_wins / games if games else 0.0
    ci_lo, ci_hi = wilson_ci(p1_wins, games)
    pvalue = binomial_two_sided_pvalue(p1_wins, decisive, p0=0.5)

    n_seeds = len(per_seed)
    consistent = seeds_p1_ahead > n_seeds / 2
    significant = pvalue < ALPHA
    high_win_rate = win_rate > win_rate_threshold
    verdict = high_win_rate and significant and consistent

    return {
        "name": config["name"],
        "p1": config["p1"],
        "p2": config["p2"],
        "iterations": config["iterations"],
        "p1_heuristic": config.get("p1_heuristic", "base"),
        "p2_heuristic": config.get("p2_heuristic", "base"),
        "games": games,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "p1_win_rate": round(win_rate, 3),
        "ci95_lo": round(ci_lo, 3),
        "ci95_hi": round(ci_hi, 3),
        "pvalue_binomial": round(pvalue, 5),
        "n_seeds": n_seeds,
        "seeds_p1_ahead": seeds_p1_ahead,
        "consistent_across_seeds": consistent,
        "win_rate_threshold": win_rate_threshold,
        "win_rate_above_threshold": high_win_rate,
        "pvalue_below_0.05": significant,
        "hypothesis_holds": verdict,
        "per_seed": seed_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2+3: statistics + significance tests for H1-H4")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to raw_results.json")
    parser.add_argument("--hypothesis", type=str, choices=["h1", "h2", "h3", "h4"],
                        required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        raw_results = json.load(f)

    threshold = WIN_RATE_THRESHOLDS[args.hypothesis]
    configs = [analyze_config(entry, threshold) for entry in raw_results]

    out = {"hypothesis": args.hypothesis, "win_rate_threshold": threshold,
           "configs": configs}
    if args.hypothesis == "h4":
        out["both_configs_hold"] = bool(configs) and all(
            c["win_rate_above_threshold"] for c in configs)

    os.makedirs(args.output_dir, exist_ok=True)
    stats_path = os.path.join(args.output_dir, "stats.json")
    csv_path = os.path.join(args.output_dir, "stats.csv")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    csv_fields = [
        "name", "p1", "p2", "iterations", "p1_heuristic", "p2_heuristic",
        "games", "p1_wins", "p2_wins", "draws", "p1_win_rate",
        "ci95_lo", "ci95_hi", "pvalue_binomial", "n_seeds", "seeds_p1_ahead",
        "consistent_across_seeds", "win_rate_threshold",
        "win_rate_above_threshold", "pvalue_below_0.05", "hypothesis_holds",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for c in configs:
            writer.writerow({k: c[k] for k in csv_fields})

    print(f"{args.hypothesis.upper()} statistics "
          f"(win-rate threshold > {threshold:.0%}):\n")
    for c in configs:
        verdict = "HOLDS" if c["hypothesis_holds"] else "not supported"
        print(f"  {c['name']:>28s}: {c['p1']} win rate = {c['p1_win_rate']:.1%} "
              f"(95% CI [{c['ci95_lo']:.1%}, {c['ci95_hi']:.1%}]), "
              f"p={c['pvalue_binomial']:.4f}, "
              f"{c['seeds_p1_ahead']}/{c['n_seeds']} seeds favor {c['p1']} "
              f"-> {verdict}")

    if args.hypothesis == "h4":
        verdict = "HOLDS" if out["both_configs_hold"] else "not supported"
        print(f"\n  H4 overall (edge wins >50% in both configurations) "
              f"-> {verdict}")

    print(f"\nWritten to {stats_path} and {csv_path}")


if __name__ == "__main__":
    main()
