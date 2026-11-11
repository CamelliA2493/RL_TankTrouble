import json, hashlib, time
from pathlib import Path
root = Path(r"E:\tank")
files = ["weights/pretrained.pt","runtime/TankTrouble_v4.0.swf",
         "bin/tank-flash-runner.exe","bin/tank-flash-window.exe","bin/ruffle.exe",
         "tactics.py","policy.py","trainer.py","evaluate.py","flash_env.py",
         "flash_config.py","flash_navigation.py","deploy.py","metrics.py"]
hashes = {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}
seed_plan = {
  "train": {"start": 1000000000, "end": 1199999999, "rule": "按 run 台账顺序分配，每局一个，不重复"},
  "diagnostic": {"start": 1200000000, "end": 1249999999, "rule": "诊断/归因专用，可挑典型场景但不得进入选模或正式集"},
  "supplement": {"start": 1250000000, "end": 1250000199, "rule": "仅当 U 机会<100 时启用的预登记补充诊断集，单列成绩"},
  "selection": {"start": 1300000000, "end": 1300000199, "rule": "200 局选模，按序全用，不挑地图"},
  "formal": {"start": 1400000000, "end": 1400000999, "rule": "1000 局正式盲测，冻结候选后只开一次"},
  "excluded": "所有 <1e9 的既有种子区间（含旧正式 930000000–930000199）一律排除",
}
protocol = {
  "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "objective": "原版 Flash v4.0 对官方 Laika：从 weights/pretrained.pt 执行阶段 0–5，99% 胜率后训练",
  "weights_sha256": hashes["weights/pretrained.pt"],
  "swf_sha256": hashes["runtime/TankTrouble_v4.0.swf"],
  "ruffle_commit": "89a7049965f3090e706265670df309a5128233ac",
  "source_hashes": {k: v for k, v in hashes.items() if k.endswith(".py")},
  "executor_hashes": {k: v for k, v in hashes.items() if k.startswith("bin/")},
  "environment": {
    "server": {"host":"connect.bjb1.seetacloud.com:47741","gpu":"NVIDIA GeForce RTX 4090 24564MiB",
               "cpu_quota":"16 cores (cgroup 1600000/100000)","ram_gb":1007,"disk":"50G, 35G free",
               "python":"3.10.22","torch":"2.5.1+cu118","numpy":"1.26.4","gymnasium":"1.0.0"},
    "local": {"os":"Windows","python":"3.10.11","torch":"2.5.1+cpu"}
  },
  "sampling": {"decoder":"sample-policy（deterministic=False）","policy_seed_offset":100000,
               "physics_seed":"reset(seed=map_seed)","decision_frames":13,
               "controller":"tactics.control 逐帧，紧急避弹可即时打断"},
  "seed_plan": seed_plan,
  "metrics": {
    "win_rate": "胜/全部开局；负、平、1500帧超时均计入分母；Wilson 95%",
    "jitter_J": {
      "controllable_frame": "env.step 前 player0 存活的一帧；auto_settle 帧不计",
      "window": "8 个连续可控帧",
      "movement": "窗口内 7 次逐帧位移的格数之和 < 0.25",
      "turn_reversal": "转向指令 action//2%3-1 忽略 0 后相邻符号变化 >= 2 次",
      "report": "合并重叠窗口，覆盖帧数/可控帧数*1000",
      "extra": "紧急避弹子集、静止占比（逐帧位移<1e-3）、自杀、超时"
    },
    "items_U": {
      "opportunity": "己方存活、箱子已落地、迷宫图可达、绕行预算内；同局同箱只记首次",
      "detour_budget_edges": 6,
      "max_approach_edges": 12,
      "pickup_radius_tiles": 0.8,
      "weapon_window_frames": 250,
      "effective": "武器有效期内 hits 事件 owner=己方 victim=Laika 且该帧当前武器为拾取武器",
      "defensive": "护盾/ElToro 效果无法用原版事件归因，单列不计入有效",
      "report": "U=有效/机会；接近率、拾取率、拾取后使用率、有效使用率、每百局有效次数"
    }
  },
  "gates": {
    "phase2": {"j_relative_drop": 0.30, "selection_win_drop_max_pp": 1.0, "timeout_increase_max_pp": 1.0},
    "phase3": {"opportunities_min": 100, "u_relative_gain": 0.20, "u_absolute_gain_pp": 2.0,
               "effective_per_100_games_increase": True, "j_hold": True, "win_guardrail": "同上"},
    "phase4": {"selection_wins_min": 198, "episodes": 200, "j_gate": True, "u_gate": True},
    "phase5": {"formal_wins_min": 990, "episodes": 1000, "j_relative_drop": 0.30,
               "u_relative_gain": 0.20, "u_absolute_gain_pp": 2.0}
  },
  "baseline_history": {"episodes": 200, "outcomes": {"win": 176, "loss": 16, "draw": 8, "timeout": 0},
                       "note": "旧正式集 930000000–930000199，仅作历史对照，不进入新协议"}
}
(root/"experiments").mkdir(exist_ok=True)
(root/"experiments/protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
for name, info in [("selection", seed_plan["selection"]), ("formal", seed_plan["formal"]),
                   ("supplement", seed_plan["supplement"]), ("diagnostic_sample", {"start":1200000000,"end":1200000099})]:
    seeds = list(range(info["start"], info["end"]+1))
    (root/f"experiments/seeds_{name}.json").write_text(json.dumps(seeds))
print(json.dumps({k: protocol[k] for k in ("frozen_at","weights_sha256","swf_sha256")}, indent=2))
print("seed counts", {n: len(json.loads((root/f'experiments/seeds_{n}.json').read_text())) for n in ('selection','formal','supplement','diagnostic_sample')})
