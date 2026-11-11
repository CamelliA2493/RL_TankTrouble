"""20 局原生窗口与评估的逐帧动作一致性验证（Windows，本地运行）。"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / "python" / "python.exe"


def window_run(seed, frames=240):
    output = ROOT / "runs" / "window_consistency" / str(seed)
    output.mkdir(parents=True, exist_ok=True)
    record = output / "record.jsonl"
    if record.exists():
        record.unlink()
    command = [str(PYTHON), str(ROOT / "windowed_flash_play.py"), "--mode", "laika",
               "--seed", str(seed), "--frames", str(frames), "--record", str(record), "--chase-crates"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
    rows = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()] if record.exists() else []
    first_round = [row for row in rows if row.get("round") == 1]
    return first_round, result.returncode


def evaluation_run(seed):
    output = ROOT / "runs" / "window_consistency" / f"eval_{seed}"
    if output.exists():
        import shutil
        shutil.rmtree(output)
    command = [str(PYTHON), str(ROOT / "evaluate.py"), "--model", str(ROOT / "weights/pretrained.pt"),
               "--sample-policy", "--episodes", "1", "--workers", "1", "--seed", str(seed),
               "--trace", "--chase-crates", "--output", str(output)]
    subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
    episode = json.loads((output / f"{seed}.json").read_text())
    return [row["action"] for row in episode["trace"]], episode["info"]["outcome"]


def main():
    seeds = [710010000 + i for i in range(20)]
    rows = []
    for seed in seeds:
        window_actions, code = window_run(seed)
        eval_actions, outcome = evaluation_run(seed)
        window = [row["action"] for row in window_actions]
        length = min(len(window), len(eval_actions))
        mismatches = [(i, window[i], eval_actions[i]) for i in range(length) if window[i] != eval_actions[i]]
        key_mismatches = [row["frame"] for row in window_actions
                          if row.get("model_keys") is not None and
                          sorted(row["model_keys"]) != sorted(row.get("applied_keys") or [])]
        rows.append(dict(seed=seed, window_frames=len(window), evaluation_frames=len(eval_actions),
                         compared=length, action_mismatches=len(mismatches), key_mismatches=len(key_mismatches),
                         outcome=outcome, first_mismatch=mismatches[:3]))
        print(json.dumps(rows[-1]), flush=True)
    summary = dict(seeds=len(rows), total_compared=sum(r["compared"] for r in rows),
                   total_action_mismatches=sum(r["action_mismatches"] for r in rows),
                   total_key_mismatches=sum(r["key_mismatches"] for r in rows),
                   games=rows)
    (ROOT / "runs" / "window_consistency" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("seeds", "total_compared", "total_action_mismatches",
                                               "total_key_mismatches")}))


if __name__ == "__main__":
    sys.exit(main())
