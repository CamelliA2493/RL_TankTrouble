"""冻结模型控制原版SWF；原生窗口负责渲染和真人键盘输入。"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT/"weights/pretrained.pt"
DEFAULT_RUNNER = ROOT/"bin/tank-flash-window.exe"


def load_runtime(model):
    if model == DEFAULT_MODEL:
        metadata = json.loads((ROOT/"weights/pretrained.json").read_text())
        if hashlib.sha256(model.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("预训练权重哈希不一致")
    sys.path.insert(0, str(ROOT))
    return {name:importlib.import_module(name) for name in ("deploy","flash_env","flash_config")}


def decide(runtime, agent, state, config):
    tanks = state.get("tanks", [])
    # 原版清场和换图期间可以没有坦克，等待原版自行建立下一回合。
    if state.get("timeline") != 53 or len(tanks) != 2 or not all(tanks) or not tanks[0].get("alive"):
        return 8, [], {"reason":"awaiting_round_or_dead"}
    observed = dict(state, frame=state["round_frame"])
    obs = runtime["flash_env"].make_observation(observed, config)
    action, detail = agent.act(obs, state["round_frame"])
    return action, runtime["flash_env"].action_keys(action,tanks[0]), detail


def play(args, runtime, bridge, agent, config):
    state = bridge.request(op="reset",seed=args.seed,mode="duel" if args.mode=="human" else "laika",observe=False)
    current_round = None
    settled = set()
    stream = args.record.open("w",encoding="utf-8") if args.record else None
    with stream if stream else nullcontext():
        while True:
            started = time.monotonic()
            if state["round"] != current_round:
                current_round = state["round"]
                agent.reset(args.seed+current_round-1)
                print(f"回合 {current_round} 开始",flush=True)
            tanks = state.get("tanks",[])
            if state["root"].get("frozen") and len(tanks)==2 and all(tanks) and current_round not in settled:
                settled.add(current_round)
                alive=[bool(t.get("alive")) for t in tanks]
                result="模型获胜" if alive==[True,False] else "模型落败" if alive==[False,True] else "平局"
                print(result,flush=True)
            action, keys, detail = decide(runtime,agent,state,config)
            try:
                following = bridge.request(op="step",keys=keys,frames=1)
            except (RuntimeError, BrokenPipeError):
                try:
                    exit_code=bridge.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    raise
                if exit_code==0:
                    return
                raise
            if stream:
                stream.write(json.dumps(dict(frame=state["frame"],round=current_round,action=action,
                    model_keys=keys,applied_keys=following["applied_keys"],detail=detail))+"\n")
                stream.flush()
            state = following
            if args.frames and state["frame"]>=args.frames:
                return
            time.sleep(max(0,1/args.fps-(time.monotonic()-started)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["human","laika"],default="human")
    parser.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    parser.add_argument("--runner",type=Path,default=DEFAULT_RUNNER)
    parser.add_argument("--seed",type=int,default=710004000)
    parser.add_argument("--fps",type=int,default=25)
    parser.add_argument("--record",type=Path)
    parser.add_argument("--frames",type=int,default=0)
    parser.add_argument("--chase-crates",action="store_true")
    args=parser.parse_args()
    args.model=args.model.resolve();args.runner=args.runner.resolve()
    if not 1<=args.fps<=120 or not 0<=args.seed<2**31:
        parser.error("fps须为1至120，seed须为非负32位整数")
    if args.record:
        args.record.parent.mkdir(parents=True,exist_ok=True)
    if sys.stdout is None:
        log=Path(os.environ.get("LOCALAPPDATA",tempfile.gettempdir()))/"TankTrouble/model_client.log"
        log.parent.mkdir(parents=True,exist_ok=True)
        sys.stdout=log.open("a",encoding="utf-8",buffering=1);sys.stderr=sys.stdout
    runtime=load_runtime(args.model)
    config=runtime["flash_config"].FlashConfig(runner=str(args.runner),movie=str(ROOT/"runtime/TankTrouble_v4.0.swf"))
    agent=runtime["deploy"].HybridController(args.model,sample_policy=True,chase_crates=args.chase_crates)
    bridge=runtime["flash_env"].FlashBridge(config)
    print("原版SWF窗口：模型控制红色1P；真人控制绿色2P（方向键、M）。关闭窗口退出。",flush=True)
    try:
        play(args,runtime,bridge,agent,config)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()


if __name__=="__main__":
    main()
