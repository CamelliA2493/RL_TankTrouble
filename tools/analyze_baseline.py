"""基线/候选逐局归因：失败类型、抖动来源、道具机会细节。只读 evaluate --trace 产物。"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load(directory):
    return [json.loads(path.read_text()) for path in sorted(Path(directory).glob("[0-9]*.json"))]


def turn_sign(action):
    return int(action) // 2 % 3 - 1


def flip_causes(episode):
    """逐翻转归因：模板切换、紧急避弹、瞄准误差过零、航点切换、导航误差。"""
    trace = episode["trace"]
    counts = Counter()
    examples = defaultdict(list)
    for frame in range(1, len(trace)):
        if turn_sign(trace[frame - 1]["action"]) == 0 or turn_sign(trace[frame]["action"]) == 0:
            continue
        if turn_sign(trace[frame - 1]["action"]) == turn_sign(trace[frame]["action"]):
            continue
        tags = []
        if trace[frame].get("tm") != trace[frame - 1].get("tm"):
            tags.append("template_switch")
        if trace[frame].get("dg") or trace[frame - 1].get("dg"):
            tags.append("dodge")
        detail = trace[frame]["detail"]
        ordinary = detail.get("ordinary") or detail
        reason = detail["reason"]
        if reason in ("align", "alignment_quantized") or ordinary.get("reason") in ("align", "alignment_quantized"):
            error = ordinary.get("heading_error")
            if error is not None and abs(error) < 0.15:
                tags.append("aim_cross_zero")
            else:
                tags.append("aim_large_error")
        if reason == "navigate" or ordinary.get("reason") == "navigate":
            tags.append("navigate_error")
        if reason == "template_navigate":
            previous = trace[frame - 1]["detail"].get("ordinary") or trace[frame - 1]["detail"]
            if previous.get("waypoint") != ordinary.get("waypoint"):
                tags.append("waypoint_switch")
            else:
                tags.append("template_navigate_same_waypoint")
        if reason == "template_retreat":
            tags.append("retreat")
        if not tags:
            tags.append("other")
        for tag in set(tags):
            counts[tag] += 1
            if len(examples[tag]) < 3:
                examples[tag].append(dict(seed=episode["seed"], frame=trace[frame]["frame"],
                                          reason=reason, detail=detail))
    return counts, examples


def episode_failure(episode, usernames):
    trace = episode["trace"]
    info = episode["info"]
    own = usernames[0] if usernames else "Player 1"
    enemy = usernames[1] if len(usernames) > 1 else "Laika"
    hits = episode.get("hits", [])
    death_frame = info.get("player_death_frame")
    killers = [h for h in hits if h["victim"] == own]
    kills = [h for h in hits if h["victim"] == enemy]
    death = None
    if death_frame is not None:
        context = [r for r in trace if abs(r["frame"] - death_frame) <= 2]
        if context:
            record = min(context, key=lambda r: abs(r["frame"] - death_frame))
            s = record["s"]
            death = dict(frame=death_frame, template=record.get("tm"), reason=record["detail"].get("reason"),
                         own_weapon=s[4], enemy_weapon=s[15],
                         distance=round(math.hypot(s[0] - s[11], s[1] - s[12]), 3),
                         last_hit_by=[h["owner"] for h in killers if abs(h["frame"] - death_frame) <= 3])
    return dict(seed=episode["seed"], outcome=info["outcome"], frames=info["frames"],
                own_hits=len(kills), enemy_hits=len(killers),
                suicide=sum(1 for h in killers if h["owner"] == own),
                death=death,
                templates=dict(Counter(r.get("tm") for r in trace)),
                weapons=dict(Counter(r["s"][4] for r in trace)),
                last_reason=trace[-1]["detail"].get("reason") if trace else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    episodes = load(args.dir)
    total = Counter()
    cause_total = Counter()
    examples = {}
    failures = []
    for episode in episodes:
        total[episode["info"]["outcome"]] += 1
        causes, sample = flip_causes(episode)
        cause_total.update(causes)
        usernames = episode.get("usernames") or []
        if episode["info"]["outcome"] != "win":
            failures.append(episode_failure(episode, usernames))
        for tag, rows in sample.items():
            examples.setdefault(tag, []).extend(rows[:1])
    # 模板分布与静止帧
    template_frames = Counter()
    template_reasons = defaultdict(Counter)
    for episode in episodes:
        for record in episode["trace"]:
            template_frames[str(record.get("tm"))] += 1
            template_reasons[str(record.get("tm"))][record["detail"]["reason"]] += 1
    # 抖动覆盖帧按模板
    jitter_by_template = Counter()
    for episode in episodes:
        metrics_path = Path(args.dir) / "metrics.json"
        if not metrics_path.exists():
            break
        detail = {row["seed"]: row for row in json.loads(metrics_path.read_text())["episodes_detail"]}
        row = detail.get(episode["seed"])
        if not row:
            continue
        trace = episode["trace"]
        n = len(trace)
        positions = np.array([[r["s"][0], r["s"][1]] for r in trace], dtype=float)
        moves = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        signs = np.array([turn_sign(r["action"]) for r in trace])
        for start in range(0, n - 8):
            movement = float(moves[start:start + 8 - 1].sum())
            nonzero = signs[start:start + 8]
            nonzero = nonzero[nonzero != 0]
            changes = int((np.diff(nonzero) != 0).sum()) if len(nonzero) > 1 else 0
            if movement < 0.25 and changes >= 2:
                for frame in range(start, start + 8):
                    jitter_by_template[str(trace[frame].get("tm"))] += 1
    report = dict(outcomes=dict(total), flips=dict(cause_total),
                  jitter_covered_by_template=dict(jitter_by_template),
                  template_frames=dict(template_frames),
                  template_reasons={k: dict(v) for k, v in template_reasons.items()},
                  failures=failures, examples=examples,
                  items=json.loads((Path(args.dir) / "metrics.json").read_text())["items"] if (Path(args.dir) / "metrics.json").exists() else {})
    (Path(args.dir) / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    slim = {k: report[k] for k in ("outcomes", "flips", "jitter_covered_by_template", "template_frames", "items")}
    slim["failure_count"] = len(failures)
    slim["failures"] = failures
    print(json.dumps(slim, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
