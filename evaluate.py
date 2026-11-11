"""原版官方Laika整局评估，保存逐帧控制与真实命中事件。"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np

from flash_config import FlashConfig,GAME_SHA256,RUFFLE_COMMIT
from flash_env import OriginalFlashEnv


def play(task):
    seed,output,enable_dodge,options=task
    spec=importlib.util.spec_from_file_location("frozen_tactics",Path(output)/"source_tactics.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    network=None
    if options["model"]:
        import torch
        torch.set_num_threads(1)
        policy_spec=importlib.util.spec_from_file_location("frozen_policy",Path(output)/"source_policy.py")
        policy_module=importlib.util.module_from_spec(policy_spec)
        policy_spec.loader.exec_module(policy_module)
        network=policy_module.HybridPolicy().eval()
        network.load_state_dict(torch.load(Path(output)/"model.pt",map_location="cpu",weights_only=False)["policy"])
        # 采样解码也按地图固定随机流，便于对照和逐帧复现。
        torch.manual_seed(seed+options["policy_seed_offset"])
    env=OriginalFlashEnv(FlashConfig(mode="laika",max_episode_frames=1500,
        auto_settle_dead=True,observe_events=True))
    trace=[];memory={};weapons=Counter();reasons=Counter();started=time.monotonic()
    rng=np.random.default_rng(seed+1234567)
    selected=options["template"]
    hidden=None;aggression=.5;next_decision=0;last_action=8;last_decision=0
    try:
        obs,_=env.reset(seed=seed)
        while True:
            weapon=env.state["tanks"][0]["currentWeapon"];weapons[weapon]+=1
            previous_template=selected
            if options["random_templates"] and env.state["frame"]%13==0:
                selected=int(rng.integers(7))
            if network is not None and env.state["frame"]>=next_decision:
                context=[0 if selected is None else selected/6,aggression,last_action//6-1,
                         last_action//2%3-1,last_action%2,(env.state["frame"]-last_decision)/25]
                encoded=policy_module.encode_observation(obs,context)
                with torch.no_grad():
                    logits,hidden=network(torch.from_numpy(encoded).reshape(1,1,403),hidden)
                    choice,_,_,_=network.actions(logits,deterministic=not options["sample_policy"])
                new_template=int(choice[0,0,0])
                selected=new_template
                aggression=float(choice[0,0,2])
                last_decision=env.state["frame"];next_decision=last_decision+13
            if selected!=previous_template:
                memory.clear()
            if selected is None:
                action,detail=module.control(obs,memory,enable_dodge=enable_dodge)
            else:
                action,detail=module.control(obs,memory,enable_dodge=enable_dodge,template=selected,aggression=aggression)
            if selected is not None:
                detail["selected_template"]=selected
                detail["aggression"]=aggression
            last_action=action
            trace.append(dict(frame=env.state["frame"],action=action,detail=detail))
            reasons[detail["reason"]]+=1
            obs,_,ended,limited,info=env.step(action)
            if ended or limited:
                break
        result=dict(seed=seed,info=info,weapons=dict(weapons),reasons=dict(reasons),
                    hits=env.state["hits"],trace=trace,seconds=time.monotonic()-started)
        (Path(output)/f"{seed}.json").write_text(json.dumps(result))
        return dict(seed=seed,outcome=info["outcome"],frames=info["frames"],reasons=dict(reasons),
                    weapons=dict(weapons),seconds=result["seconds"])
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=710000500)
    parser.add_argument("--episodes",type=int,default=8)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--without-dodge",action="store_true")
    parser.add_argument("--tactics",type=Path,default=Path("tactics.py"))
    parser.add_argument("--template",type=int,choices=range(7))
    parser.add_argument("--random-templates",action="store_true")
    parser.add_argument("--model",type=Path)
    parser.add_argument("--sample-policy",action="store_true")
    parser.add_argument("--policy-seed-offset",type=int,default=100000)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    assert not (args.random_templates and args.template is not None)
    assert not (args.model and (args.random_templates or args.template is not None))
    assert not args.sample_policy or args.model
    assert args.episodes>0 and args.workers>0 and 0<=args.seed and args.seed+args.episodes<=2**31
    args.output.mkdir(parents=True,exist_ok=False)
    sources={}
    for name in ["tactics.py","evaluate.py","flash_env.py","flash_config.py"]:
        blob=(args.tactics if name=="tactics.py" else Path(name)).read_bytes()
        (args.output/("source_"+name)).write_bytes(blob)
        sources[name]=hashlib.sha256(blob).hexdigest()
    if args.model:
        blob=args.model.read_bytes();(args.output/"model.pt").write_bytes(blob)
        sources["model.pt"]=hashlib.sha256(blob).hexdigest()
        blob=Path("policy.py").read_bytes();(args.output/"source_policy.py").write_bytes(blob)
        sources["policy.py"]=hashlib.sha256(blob).hexdigest()
    manifest=dict(kind="original_laika_tactical_integration_pilot",arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_hashes=sources,swf_sha256=GAME_SHA256,ruffle_commit=RUFFLE_COMMIT,status="running",goal_passed=False)
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    rows=[];started=time.monotonic()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        options=dict(template=args.template,random_templates=args.random_templates,model=bool(args.model),
                     sample_policy=args.sample_policy,policy_seed_offset=args.policy_seed_offset)
        tasks=[(seed,str(args.output),not args.without_dodge,options) for seed in range(args.seed,args.seed+args.episodes)]
        for row in pool.imap_unordered(play,tasks):
            rows.append(row);print(json.dumps(row),flush=True)
    report=dict(manifest,status="complete",episodes=len(rows),outcomes=dict(Counter(r["outcome"] for r in rows)),
                rows=sorted(rows,key=lambda r:r["seed"]),seconds=time.monotonic()-started)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ["status","episodes","outcomes","seconds"]}),flush=True)


if __name__=="__main__":
    main()
