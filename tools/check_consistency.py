"""比较 evaluate 与 deploy 的逐帧动作序列，验证评估/部署一致性。"""
import argparse
import json
from pathlib import Path


def load_evaluate(directory, seed):
    episode = json.loads((Path(directory) / f"{seed}.json").read_text())
    return [row["action"] for row in episode["trace"]], episode["info"]["outcome"], episode["info"]["frames"]


def load_deploy(directory):
    actions = []
    for line in (Path(directory) / "frames.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "action" in row:
            actions.append(row["action"])
    summary = json.loads((Path(directory) / "summary.json").read_text())
    return actions, summary["info"]["outcome"], summary["info"]["frames"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--deploy", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    expected, outcome_a, frames_a = load_evaluate(args.evaluation, args.seed)
    actual, outcome_b, frames_b = load_deploy(args.deploy)
    length = min(len(expected), len(actual))
    mismatches = [(i, expected[i], actual[i]) for i in range(length) if expected[i] != actual[i]]
    report = dict(seed=args.seed, evaluation_actions=len(expected), deploy_actions=len(actual),
                  outcome_evaluation=outcome_a, outcome_deploy=outcome_b,
                  frames_evaluation=frames_a, frames_deploy=frames_b,
                  mismatches=len(mismatches), first_mismatches=mismatches[:5],
                  match=not mismatches and outcome_a == outcome_b and frames_a == frames_b)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["match"] else 1)


if __name__ == "__main__":
    main()
