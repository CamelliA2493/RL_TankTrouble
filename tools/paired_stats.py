"""配对统计：胜率 McNemar 精确检验、J/U 的按局配对 bootstrap 区间。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def load(directory):
    data = json.loads((Path(directory) / "metrics.json").read_text())
    rows = {row["seed"]: row for row in data["episodes_detail"]}
    return data, rows


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n) * 2
    return min(1.0, p)


def pooled_j(rows):
    frames = sum(r["j"]["frames"] for r in rows)
    covered = sum(r["j"]["covered"] for r in rows)
    return 1000.0 * covered / max(frames, 1)


def bootstrap(rows_a, rows_b, metric, iterations=4000, seed=7):
    rng = np.random.default_rng(seed)
    seeds = list(rows_a)
    size = len(seeds)
    values = []
    for _ in range(iterations):
        sample = rng.choice(seeds, size=size, replace=True)
        a = metric([rows_a[s] for s in sample])
        b = metric([rows_b[s] for s in sample])
        values.append(b - a)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)), float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("cand")
    args = parser.parse_args()
    data_a, rows_a = load(args.base)
    data_b, rows_b = load(args.cand)
    seeds = sorted(set(rows_a) & set(rows_b))
    b = sum(1 for s in seeds if rows_a[s]["outcome"] == "win" and rows_b[s]["outcome"] != "win")
    c = sum(1 for s in seeds if rows_a[s]["outcome"] != "win" and rows_b[s]["outcome"] == "win")
    wins_a = sum(rows_a[s]["outcome"] == "win" for s in seeds)
    wins_b = sum(rows_b[s]["outcome"] == "win" for s in seeds)
    report = dict(
        paired=len(seeds),
        baseline_wins=wins_a, candidate_wins=wins_b,
        baseline_only=b, candidate_only=c,
        mcnemar_exact_p=mcnemar(b, c),
        win_rate_diff=(wins_b - wins_a) / len(seeds),
        win_diff_ci=bootstrap(rows_a, rows_b, lambda rows: sum(r["outcome"] == "win" for r in rows) / len(rows)),
        jitter=dict(
            baseline=pooled_j([rows_a[s] for s in seeds]),
            candidate=pooled_j([rows_b[s] for s in seeds]),
        ),
    )
    report["jitter"]["relative"] = (report["jitter"]["candidate"] - report["jitter"]["baseline"]) / max(report["jitter"]["baseline"], 1e-9)
    lo, hi, mean = bootstrap(rows_a, rows_b, pooled_j)
    report["jitter"]["diff_ci"] = [lo, hi]
    report["jitter"]["relative_ci"] = [lo / max(report["jitter"]["baseline"], 1e-9),
                                       hi / max(report["jitter"]["baseline"], 1e-9)]
    for label in ("opportunities", "effective"):
        total_a = sum(rows_a[s]["u"][label] for s in seeds)
        total_b = sum(rows_b[s]["u"][label] for s in seeds)
        report[label] = dict(baseline=total_a, candidate=total_b)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
