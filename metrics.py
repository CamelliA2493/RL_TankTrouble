"""从 evaluate.py --trace 的逐局记录计算抖动 J、道具 U 与配对统计。

指标定义在 runs/phase0/PROTOCOL.md 中冻结；本文件只做确定性计算，不接触游戏。
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from flash_navigation import maze_distances

JITTER_WINDOW = 8
JITTER_MOVE = 0.25
JITTER_FLIPS = 2
DETOUR_BUDGET = 6          # 仅报告：相对敌方的绕行边数
MAX_APPROACH_EDGES = 8     # 机会要求己方到箱子的图距离上限（v1.1）
PICKUP_RADIUS = 0.8        # 格
WEAPON_WINDOW = 250        # 帧，拾取后武器有效期上限
OPPORTUNITY_MIN_FRAMES = 1


def wilson(success, total, z=1.96):
    if total <= 0:
        return [0.0, 0.0]
    p = success / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def build_walls(episode):
    maze = np.asarray(episode["maze"], dtype=np.float32)
    geometry = episode["geometry"]
    width, height = int(geometry["width"]), int(geometry["height"])
    grid = np.zeros((3, height + 1, width + 1), dtype=np.float32)
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            cell = maze[x, y]
            grid[0, y, x] = cell[0]
            grid[1, y + 1, x] = cell[1]
            grid[2, y, x] = cell[2]
    grid[1, 0, :width] = grid[1, height, :width] = 1
    grid[2, :height, 0] = grid[2, :height, width] = 1
    walls = np.ascontiguousarray(grid[1:, :height + 1, :width + 1] > .5, dtype=np.uint8)
    return width, height, walls, maze_distances(width, height, walls.tobytes())


def turn_sign(action):
    return int(action) // 2 % 3 - 1


def jitter(episode):
    trace = episode["trace"]
    n = len(trace)
    if n < JITTER_WINDOW + 1:
        return dict(frames=n, covered=0, j=0.0, dodge_frames=0, dodge_covered=0, j_dodge=0.0,
                    stationary=0.0, flips=0, flip_reasons={})
    positions = np.array([[r["s"][0], r["s"][1]] for r in trace], dtype=float)
    moves = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    signs = np.array([turn_sign(r["action"]) for r in trace])
    dodge = np.array([int(r.get("dg", 0)) for r in trace])
    covered = np.zeros(n, dtype=bool)
    dodge_covered = np.zeros(n, dtype=bool)
    flips_at = Counter()
    reasons = Counter()
    flips = 0
    for start in range(0, n - JITTER_WINDOW):
        movement = float(moves[start:start + JITTER_WINDOW - 1].sum())
        window_signs = signs[start:start + JITTER_WINDOW]
        nonzero = window_signs[window_signs != 0]
        changes = int((np.diff(nonzero) != 0).sum()) if len(nonzero) > 1 else 0
        if movement < JITTER_MOVE and changes >= JITTER_FLIPS:
            covered[start:start + JITTER_WINDOW] = True
            if dodge[start:start + JITTER_WINDOW].any():
                dodge_covered[start:start + JITTER_WINDOW] = True
    for frame in range(n - 1):
        if signs[frame] == 0 or signs[frame + 1] == 0:
            continue
        if signs[frame] != signs[frame + 1]:
            flips += 1
            reasons[trace[frame + 1]["detail"]["reason"]] += 1
    dodge_mask = dodge.astype(bool)
    return dict(
        frames=n,
        covered=int(covered.sum()),
        j=1000.0 * covered.sum() / n,
        dodge_frames=int(dodge_mask.sum()),
        dodge_covered=int(dodge_covered.sum()),
        j_dodge=1000.0 * dodge_covered.sum() / max(int(dodge_mask.sum()), 1),
        stationary=float((moves < 1e-3).mean()),
        flips=flips,
        flip_reasons=dict(reasons),
        coverage_by_frame=int(covered.sum()),
    )


def _tile(position, width, height):
    x, y = position
    return (int(np.clip(math.floor(x), 0, width - 1)), int(np.clip(math.floor(y), 0, height - 1)))


def items(episode, width, height, distances, usernames, hits, fires):
    """机会：己方存活、箱子已落地、迷宫可达、绕行预算内；同局同箱只记一次。"""
    trace = episode["trace"]
    n = len(trace)
    if n == 0:
        return dict(opportunities=0, opportunities_by_weapon={}, picked=0, picked_by_enemy=0,
                    approached=0, used=0, effective=0, defensive_unattributed=0,
                    per_opportunity=[])
    own_name = usernames[0] if usernames else "Player 1"
    enemy_name = usernames[1] if len(usernames) > 1 else "Laika"
    present = {}
    opportunities = {}
    for frame in range(n):
        record = trace[frame]
        s = record["s"]
        if not s[3]:
            continue
        own = _tile((s[0], s[1]), width, height)
        enemy_alive = bool(s[14])
        enemy = _tile((s[11], s[12]), width, height) if enemy_alive else None
        for crate in record.get("cr", []):
            name, cx, cy, landed = crate
            if not landed:
                continue
            present.setdefault(name, []).append(frame)
            if name in opportunities:
                continue
            tile = _tile((cx, cy), width, height)
            own_id = own[1] * width + own[0]
            crate_id = tile[1] * width + tile[0]
            distance = distances[own_id, crate_id]
            if not np.isfinite(distance):
                continue
            detour = 0.0
            if enemy_alive:
                enemy_id = enemy[1] * width + enemy[0]
                to_enemy = distances[crate_id, enemy_id]
                between = distances[own_id, enemy_id]
                if np.isfinite(to_enemy) and np.isfinite(between):
                    detour = float(distance) + float(to_enemy) - float(between)
            # v1.1：机会只取决于己方能否在固定预算内到达箱子；敌方位置另报，不改变分母。
            feasible = distance <= MAX_APPROACH_EDGES
            if feasible:
                opportunities[name] = dict(name=name, frame=frame, detour=detour, distance=float(distance))
    rows = []
    for name, info in opportunities.items():
        frames = present.get(name, [])
        later = [f for f in frames if f >= info["frame"]]
        approached = False
        for f in later:
            crate = record_crate(trace[f], name)
            s = trace[f]["s"]
            if crate is not None and np.linalg.norm(crate - np.array([s[0], s[1]])) <= PICKUP_RADIUS:
                approached = True
        picked_by_us = False
        picked_by_enemy = False
        pickup_frame = None
        for f in later:
            if f + 1 >= n:
                break
            if any(c[0] == name for c in trace[f + 1].get("cr", [])):
                continue
            crate = record_crate(trace[f], name)
            if crate is None:
                break
            s = trace[f]["s"]
            own_distance = float(np.linalg.norm(crate - np.array([s[0], s[1]])))
            enemy_alive = bool(s[14])
            enemy_distance = float(np.linalg.norm(crate - np.array([s[11], s[12]]))) if enemy_alive else math.inf
            if own_distance <= PICKUP_RADIUS and own_distance < enemy_distance:
                picked_by_us = True
                pickup_frame = f + 1
            elif enemy_alive and enemy_distance <= PICKUP_RADIUS:
                picked_by_enemy = True
            break
        weapon = None
        equipment = None
        if picked_by_us and pickup_frame is not None:
            before = trace[pickup_frame - 1]["s"]
            after = trace[pickup_frame]["s"]
            if after[4] != before[4]:
                weapon = after[4]
            if after[5] != before[5]:
                equipment = after[5]
        used = False
        effective = False
        unattributed = False
        if picked_by_us and pickup_frame is not None:
            window_end = min(n, pickup_frame + WEAPON_WINDOW)
            if weapon is not None:
                for f in range(pickup_frame, window_end):
                    if trace[f]["s"][4] != weapon:
                        window_end = f
                        break
                used = any(e["owner"] == own_name and e.get("weapon") == weapon
                           and pickup_frame <= e["frame"] < window_end for e in fires)
                for hit in hits:
                    frame = hit["frame"]
                    if (hit["owner"] == own_name and hit["victim"] == enemy_name
                            and pickup_frame <= frame < window_end and frame < n
                            and trace[frame]["s"][4] == weapon):
                        effective = True
            elif equipment is not None:
                used = True  # 装备拾取后立即生效；效果无法从原版事件归因，单列。
                unattributed = True
        rows.append(dict(name=name, opportunity_frame=info["frame"], detour=info["detour"],
                         distance=info["distance"], approached=approached, picked_by_us=picked_by_us,
                         picked_by_enemy=picked_by_enemy, weapon=weapon, equipment=equipment,
                         used=used, effective=effective, unattributed=unattributed))
    opportunities_count = len(rows)
    picked = sum(r["picked_by_us"] for r in rows)
    used = sum(r["used"] for r in rows)
    effective = sum(r["effective"] for r in rows)
    by_weapon = Counter(r["weapon"] for r in rows if r["picked_by_us"])
    return dict(opportunities=opportunities_count,
                opportunities_by_weapon=dict(by_weapon),
                approached=sum(r["approached"] for r in rows),
                picked=picked,
                picked_by_enemy=sum(r["picked_by_enemy"] for r in rows),
                used=used,
                effective=effective,
                defensive_unattributed=sum(r["unattributed"] for r in rows),
                per_opportunity=rows)


def record_crate(record, name):
    for crate in record.get("cr", []):
        if crate[0] == name:
            return np.array([crate[1], crate[2]])
    return None


def analyze(directory):
    directory = Path(directory)
    episodes = []
    for path in sorted(directory.glob("[0-9]*.json")):
        episodes.append(json.loads(path.read_text()))
    if not episodes:
        raise SystemExit("没有逐局记录")
    results = []
    for episode in episodes:
        width, height, walls, distances = build_walls(episode)
        j = jitter(episode)
        u = items(episode, width, height, distances, episode.get("usernames", []),
                  episode.get("hits", []), episode.get("fires", []))
        outcome = episode["info"]["outcome"]
        results.append(dict(seed=episode["seed"], outcome=outcome, frames=episode["info"]["frames"],
                            j=j, u=u, suicide=sum(1 for h in episode.get("hits", [])
                                                  if h["owner"] == (episode.get("usernames") or ["Player 1"])[0]
                                                  and h["victim"] == (episode.get("usernames") or ["Player 1"])[0])))
    outcomes = Counter(r["outcome"] for r in results)
    total = len(results)
    wins = outcomes.get("win", 0)
    controllable = sum(r["j"]["frames"] for r in results)
    covered = sum(r["j"]["covered"] for r in results)
    dodge_frames = sum(r["j"]["dodge_frames"] for r in results)
    dodge_covered = sum(r["j"]["dodge_covered"] for r in results)
    flip_reasons = Counter()
    for r in results:
        flip_reasons.update(r["j"]["flip_reasons"])
    opportunities = sum(r["u"]["opportunities"] for r in results)
    approached = sum(r["u"]["approached"] for r in results)
    picked = sum(r["u"]["picked"] for r in results)
    used = sum(r["u"]["used"] for r in results)
    effective = sum(r["u"]["effective"] for r in results)
    unattributed = sum(r["u"]["defensive_unattributed"] for r in results)
    summary = dict(
        episodes=total, outcomes=dict(outcomes), win_rate=wins / total if total else 0.0,
        win_wilson=wilson(wins, total),
        timeout=outcomes.get("timeout", 0), draws=outcomes.get("draw", 0), losses=outcomes.get("loss", 0),
        frames=dict(total=sum(r["frames"] for r in results),
                    mean=float(np.mean([r["frames"] for r in results]))),
        jitter=dict(controllable_frames=controllable, covered_frames=covered,
                    j=1000.0 * covered / max(controllable, 1),
                    dodge_frames=dodge_frames, dodge_covered=dodge_covered,
                    j_dodge=1000.0 * dodge_covered / max(dodge_frames, 1),
                    stationary=float(np.mean([r["j"]["stationary"] for r in results])),
                    flips=sum(r["j"]["flips"] for r in results),
                    flip_reasons=dict(flip_reasons)),
        items=dict(opportunities=opportunities, approached=approached, picked=picked, used=used,
                   effective=effective, defensive_unattributed=unattributed,
                   approach_rate=approached / opportunities if opportunities else 0.0,
                   pickup_rate=picked / opportunities if opportunities else 0.0,
                   use_rate=used / picked if picked else 0.0,
                   effective_rate=effective / opportunities if opportunities else 0.0,
                   effective_per_100_games=100.0 * effective / total if total else 0.0,
                   opportunity_per_100_games=100.0 * opportunities / total if total else 0.0),
        suicide=sum(r["suicide"] for r in results),
        episodes_detail=results,
    )
    (directory / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.dir)
    slim = {k: v for k, v in summary.items() if k != "episodes_detail"}
    print(json.dumps(slim, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
