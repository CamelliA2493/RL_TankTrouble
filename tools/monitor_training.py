"""服务器端训练监控：等待 run 完成，按累计帧间隔评估检查点并汇总。

在服务器后台运行，避免训练期间反复远程推理。只读 run 目录并写日志。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/tank-flash")
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/p4_entropy0_500k"
SEEDS = 1200000100
EPISODES = 100
TARGETS = [100000, 200000, 300000, 400000, 500000]
LOG = ROOT / "runs" / (RUN.name + "_eval.log")


def log(message):
    with LOG.open("a") as stream:
        stream.write(message + "\n")


def wait_done(timeout=5 * 3600):
    started = time.time()
    while time.time() - started < timeout:
        record = RUN / "run.json"
        if record.exists():
            try:
                if json.loads(record.read_text()).get("status") == "complete":
                    return True
            except json.JSONDecodeError:
                pass
        if not (RUN / "progress.jsonl").exists() and time.time() - started > 900:
            return False
        time.sleep(60)
    return False


def main():
    log(f"monitor start {time.strftime('%H:%M:%S')} run={RUN.name}")
    if not wait_done():
        log("training did not complete; evaluate whatever exists")
    rows = []
    if (RUN / "progress.jsonl").exists():
        rows = [json.loads(line) for line in (RUN / "progress.jsonl").read_text().splitlines()]
    if not rows:
        log("no progress rows")
        return
    chosen = []
    for target in TARGETS:
        best = min(rows, key=lambda row: abs(row["frames"] - target))
        if best["iteration"] not in [c["iteration"] for c in chosen]:
            chosen.append(best)
    log("chosen " + json.dumps([dict(iteration=c["iteration"], frames=c["frames"]) for c in chosen]))
    for row in chosen:
        checkpoint = RUN / f"checkpoint_{row['iteration']:03d}.pt"
        if not checkpoint.exists():
            log(f"missing {checkpoint.name}")
            continue
        output = ROOT / "runs" / f"{RUN.name}_eval_{row['frames']//1000:03d}k"
        subprocess.run(["rm", "-rf", str(output)], check=True)
        command = ["python", "evaluate.py", "--model", str(checkpoint),
                   "--tactics", str(RUN / "source_tactics.py"), "--sample-policy",
                   "--seed", str(SEEDS), "--episodes", str(EPISODES), "--workers", "16",
                   "--trace", "--chase-crates", "--output", str(output)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"eval failed {row['frames']}: {result.stderr[-800:]}")
            continue
        subprocess.run(["python", "metrics.py", "--dir", str(output)], cwd=ROOT,
                       capture_output=True, text=True)
        metrics = json.loads((output / "metrics.json").read_text())
        log(json.dumps(dict(frames=row["frames"], iteration=row["iteration"],
                            outcomes=metrics["outcomes"], win_rate=metrics["win_rate"],
                            jitter=round(metrics["jitter"]["j"], 2),
                            opportunities=metrics["items"]["opportunities"],
                            effective=metrics["items"]["effective"])))
    log(f"monitor done {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
