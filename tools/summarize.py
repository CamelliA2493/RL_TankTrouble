"""把多个 runs 目录的指标汇总为 Markdown 表。"""
import argparse
import json
from pathlib import Path


def load(directory):
    return json.loads((Path(directory) / "metrics.json").read_text())


def row(directory, label=None):
    data = load(directory)
    j, u = data["jitter"], data["items"]
    o = data["outcomes"]
    return (f"| {label or Path(directory).name} | {data['episodes']} | "
            f"{o.get('win',0)}/{o.get('loss',0)}/{o.get('draw',0)}/{o.get('timeout',0)} | "
            f"{100*data['win_rate']:.1f}% | {j['j']:.1f} | {j['j_dodge']:.1f} | "
            f"{100*j['stationary']:.1f}% | {u['opportunities']} | {u['picked']} | {u['used']} | "
            f"{u['effective']} | {100*u['effective_rate']:.1f}% | {u['effective_per_100_games']:.2f} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--labels", nargs="*")
    args = parser.parse_args()
    print("| 运行 | 局数 | 胜/负/平/超时 | 胜率 | J | J(避弹) | 静止 | 机会 | 拾取 | 使用 | 有效 | U | 每百局有效 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for index, directory in enumerate(args.dirs):
        label = args.labels[index] if args.labels and index < len(args.labels) else None
        print(row(directory, label))


if __name__ == "__main__":
    main()
