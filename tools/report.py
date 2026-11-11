"""汇总多个 runs 目录的 metrics.json，输出对比表（配对差值）。"""
import argparse
import json
from pathlib import Path


def row(directory):
    data = json.loads((Path(directory) / "metrics.json").read_text())
    j, u = data["jitter"], data["items"]
    return dict(name=Path(directory).name, episodes=data["episodes"],
                win=data["outcomes"].get("win", 0), loss=data["outcomes"].get("loss", 0),
                draw=data["outcomes"].get("draw", 0), timeout=data["outcomes"].get("timeout", 0),
                win_rate=data["win_rate"], j=j["j"], j_dodge=j["j_dodge"],
                stationary=j["stationary"], flips=j["flips"],
                opportunities=u["opportunities"], picked=u["picked"], used=u["used"],
                effective=u["effective"], u=u["effective_rate"],
                per100=u["effective_per_100_games"],
                approach_rate=u["approach_rate"], pickup_rate=u["pickup_rate"],
                use_rate=u["use_rate"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--pair", nargs=2, help="两个目录做配对差值")
    args = parser.parse_args()
    rows = [row(d) for d in args.dirs]
    header = ("name", "ep", "W", "L", "D", "T", "win%", "J", "Jdodge", "still", "flips", "opp", "pick", "use", "eff", "U%", "per100")
    print("  ".join(f"{h:>10}" for h in header))
    for r in rows:
        values = [r["name"], r["episodes"], r["win"], r["loss"], r["draw"], r["timeout"],
                  round(100 * r["win_rate"], 1), round(r["j"], 1), round(r["j_dodge"], 1),
                  round(100 * r["stationary"], 1), r["flips"], r["opportunities"], r["picked"],
                  r["used"], r["effective"], round(100 * r["u"], 1), round(r["per100"], 2)]
        print("  ".join(f"{str(v):>10}" for v in values))
    if args.pair and len(rows) == 2:
        a, b = rows
        print(f"pair {a['name']} -> {b['name']}: "
              f"dJ={100*(b['j']-a['j'])/max(a['j'],1e-9):+.1f}% "
              f"dU={100*(b['u']-a['u'])/max(a['u'],1e-9):+.1f}% "
              f"dU_pp={100*(b['u']-a['u']):+.2f} dWin_pp={100*(b['win_rate']-a['win_rate']):+.2f} "
              f"dStill_pp={100*(b['stationary']-a['stationary']):+.2f}")


if __name__ == "__main__":
    main()
