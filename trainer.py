"""完整原版回合采样与循环PPO，检查点保存在完整rollout更新边界。"""
import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from flash_config import FlashConfig
from flash_env import OriginalFlashEnv
from policy import HybridPolicy,encode_observation
from tactics import control


PPO_CONFIG=dict(lr=3e-4,clip=.2,entropy=.01,value_coef=.5,epochs=4,
                minibatch_episodes=4,gamma=1.,gae_lambda=1.,max_grad_norm=.5,decision_frames=13)


def collect_episode(task):
    weights,seed,rng_seed,deterministic,options=task
    torch.set_num_threads(1)
    model=HybridPolicy().eval()
    model.load_state_dict(torch.load(weights,map_location="cpu",weights_only=False)["policy"])
    torch.manual_seed(rng_seed)
    env=OriginalFlashEnv(FlashConfig(mode="laika",max_episode_frames=1500,auto_settle_dead=True,
                                     reward_mode=options["reward_mode"]))
    observations=[];choices=[];old_probs=[];values=[];rewards=[];durations=[];primitive=[]
    hidden=None;memory={};selected=None;aggression=.5;last_action=8;last_decision=0
    try:
        obs,_=env.reset(seed=seed)
        done=False
        while not done:
            start=env.state["frame"]
            context=[0 if selected is None else selected/6,aggression,last_action//6-1,
                     last_action//2%3-1,last_action%2,(start-last_decision)/25]
            encoded=encode_observation(obs,context)
            with torch.no_grad():
                logits,hidden=model(torch.from_numpy(encoded).reshape(1,1,403),hidden)
                choice,prob,_,value=model.actions(logits,deterministic=deterministic)
            new_template=int(choice[0,0,0])
            if new_template!=selected:
                memory.clear()
            selected=new_template;aggression=float(choice[0,0,2]);last_decision=start
            observations.append(encoded);choices.append(choice[0,0].numpy())
            old_probs.append(float(prob));values.append(float(value))
            total=0.
            while env.state["frame"]<start+PPO_CONFIG["decision_frames"]:
                action,_=control(obs,memory,template=selected,aggression=aggression,
                                 chase_crates=options["chase_crates"])
                primitive.append(action);last_action=action
                obs,reward,ended,limited,info=env.step(action)
                elapsed=env.state["frame"]-start
                total+=PPO_CONFIG["gamma"]**(elapsed-1)*reward
                done=ended or limited
                if done:
                    break
            rewards.append(total);durations.append(env.state["frame"]-start)
        return dict(seed=seed,obs=np.asarray(observations),actions=np.asarray(choices),
            old_probs=np.asarray(old_probs,dtype=np.float32),values=np.asarray(values,dtype=np.float32),
            rewards=np.asarray(rewards,dtype=np.float32),durations=np.asarray(durations),info=info,
            primitive_actions=np.asarray(primitive,dtype=np.uint8),
            actions_sha256=hashlib.sha256(bytes(primitive)).hexdigest())
    finally:
        env.close()


def advantages(episode):
    values=episode["values"];result=np.empty_like(values);following=0.
    for step in reversed(range(len(values))):
        discount=PPO_CONFIG["gamma"]**int(episode["durations"][step])
        decay=PPO_CONFIG["gae_lambda"]**int(episode["durations"][step])
        next_value=values[step+1] if step+1<len(values) else 0.
        delta=episode["rewards"][step]+discount*next_value-values[step]
        following=delta+discount*decay*following;result[step]=following
    return result,result+values


def update(model,optimizer,episodes,rng,device):
    maximum_error=0.
    with torch.no_grad():
        for row in episodes:
            x=torch.as_tensor(row["obs"],device=device).unsqueeze(0)
            logits,_=model(x)
            p,_=model.evaluate_actions(logits,torch.as_tensor(row["actions"],device=device).unsqueeze(0))
            maximum_error=max(maximum_error,float((p[0]-torch.as_tensor(row["old_probs"],device=device)).abs().max()))
    assert maximum_error<2e-4,("在线采样概率重算不一致",maximum_error)
    adv,targets=zip(*(advantages(row) for row in episodes))
    all_adv=np.concatenate(adv);mean=float(all_adv.mean());std=max(float(all_adv.std()),1e-6)
    metrics=[]
    model.train()
    for _ in range(PPO_CONFIG["epochs"]):
        order=rng.permutation(len(episodes))
        for begin in range(0,len(order),PPO_CONFIG["minibatch_episodes"]):
            probs=[];entropy=[];predictions=[];old=[];normalized=[];returns=[]
            for index in order[begin:begin+PPO_CONFIG["minibatch_episodes"]]:
                row=episodes[index]
                logits,_=model(torch.as_tensor(row["obs"],device=device).unsqueeze(0))
                p,e=model.evaluate_actions(logits,torch.as_tensor(row["actions"],device=device).unsqueeze(0))
                probs.append(p[0]);entropy.append(e[0]);predictions.append(logits[0,:,16])
                old.append(torch.as_tensor(row["old_probs"],device=device))
                normalized.append(torch.as_tensor((adv[index]-mean)/std,device=device))
                returns.append(torch.as_tensor(targets[index],device=device))
            p,e,v,old_p,a,ret=[torch.cat(items) for items in [probs,entropy,predictions,old,normalized,returns]]
            ratio=(p-old_p).exp()
            policy_loss=-torch.minimum(ratio*a,ratio.clamp(1-PPO_CONFIG["clip"],1+PPO_CONFIG["clip"])*a).mean()
            value_loss=(v-ret).square().mean()
            loss=policy_loss+PPO_CONFIG["value_coef"]*value_loss-PPO_CONFIG["entropy"]*e.mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),PPO_CONFIG["max_grad_norm"])
            assert torch.isfinite(norm)
            optimizer.step()
            metrics.append(dict(policy_loss=float(policy_loss),value_loss=float(value_loss),entropy=float(e.mean()),
                approx_kl=float(((ratio-1)-(p-old_p)).mean()),clip_fraction=float(((ratio-1).abs()>PPO_CONFIG["clip"]).float().mean())))
    model.eval()
    return dict(probability_max_error=maximum_error,**{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--initial",type=Path,default=Path("weights/pretrained.pt"))
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--frames",type=int,default=20000)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--episodes-per-rollout",type=int,default=16)
    parser.add_argument("--seed-start",type=int,default=710030000)
    parser.add_argument("--seed",type=int,default=456)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--from-scratch",action="store_true")
    parser.add_argument("--reward-mode",choices=["standard","win_events"],default="standard")
    parser.add_argument("--chase-crates",action="store_true")
    parser.add_argument("--optimizer",choices=["adam","adamw"],default="adam")
    parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--weight-decay",type=float,default=0.)
    parser.add_argument("--entropy",type=float,default=.01)
    args=parser.parse_args()
    PPO_CONFIG["entropy"]=args.entropy
    assert not args.output.exists() and args.frames>0 and args.workers>0 and args.episodes_per_rollout>0
    assert 0<=args.seed_start<2**31 and not (args.from_scratch and args.resume)
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(args.seed)
    model=HybridPolicy().to(args.device)
    decay=[p for n,p in model.named_parameters() if p.ndim>1 and not n.endswith("bias")]
    no_decay=[p for n,p in model.named_parameters() if p.ndim<=1 or n.endswith("bias")]
    groups=[dict(params=decay,weight_decay=args.weight_decay),dict(params=no_decay,weight_decay=0.)]
    if args.optimizer=="adamw":
        optimizer=torch.optim.AdamW(groups,lr=args.lr)
    else:
        optimizer=torch.optim.Adam(groups,lr=args.lr)
    train_config=dict(reward_mode=args.reward_mode,chase_crates=args.chase_crates,
                      optimizer=args.optimizer,lr=args.lr,weight_decay=args.weight_decay,
                      entropy=args.entropy)
    source_names=["policy.py","tactics.py","trainer.py","flash_config.py","flash_env.py"]
    blobs={n:Path(n).read_bytes() for n in source_names}
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in blobs.items()}
    state=None if args.from_scratch else torch.load(args.resume or args.initial,map_location=args.device,weights_only=False)
    if state is not None:model.load_state_dict(state["policy"])
    frames=games=iteration=0
    if args.resume:
        assert state["source_hashes"]==hashes and state["ppo_config"]==PPO_CONFIG
        assert state.get("train_config")==train_config,("训练配置与断点不一致",state.get("train_config"),train_config)
        assert state["seed_start"]==args.seed_start and state["training_seed"]==args.seed
        assert state["episodes_per_rollout"]==args.episodes_per_rollout
        optimizer.load_state_dict(state["optimizer"])
        frames,games,iteration=state["frames"],state["games"],state["iteration"]
        rng.bit_generator.state=state["numpy_rng"];torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    args.output.mkdir(parents=True)
    for name,blob in blobs.items():(args.output/("source_"+name)).write_bytes(blob)
    record=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_hashes=hashes,ppo_config=PPO_CONFIG,train_config=train_config,status="running",goal_passed=False)
    (args.output/"run.json").write_text(json.dumps(record,indent=2))
    started=time.monotonic()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        while frames<args.frames:
            assert args.seed_start+games+args.episodes_per_rollout<2**31
            weights=args.output/"sampling.pt"
            torch.save(dict(policy={k:v.detach().cpu() for k,v in model.state_dict().items()}),weights)
            options=dict(reward_mode=args.reward_mode,chase_crates=args.chase_crates)
            tasks=[(str(weights),args.seed_start+games+i,args.seed+games+i+100000,False,options)
                   for i in range(args.episodes_per_rollout)]
            episodes=pool.map(collect_episode,tasks)
            torch.save(episodes,args.output/f"rollout_{iteration+1:03d}.pt")
            metrics=update(model,optimizer,episodes,rng,args.device)
            frames+=sum(e["info"]["frames"] for e in episodes);games+=len(episodes);iteration+=1
            checkpoint=dict(policy=model.state_dict(),optimizer=optimizer.state_dict(),frames=frames,games=games,
                iteration=iteration,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                source_hashes=hashes,ppo_config=PPO_CONFIG,train_config=train_config,
                seed_start=args.seed_start,training_seed=args.seed,
                episodes_per_rollout=args.episodes_per_rollout)
            path=args.output/f"checkpoint_{iteration:03d}.pt"
            torch.save(checkpoint,path.with_suffix(".pending"));path.with_suffix(".pending").replace(path)
            result=dict(iteration=iteration,frames=frames,games=games,metrics=metrics,
                outcomes=dict(Counter(e["info"]["outcome"] for e in episodes)),seconds=time.monotonic()-started)
            with (args.output/"progress.jsonl").open("a") as stream:stream.write(json.dumps(result)+"\n")
            print(json.dumps(result),flush=True)
    record["status"]="complete";(args.output/"run.json").write_text(json.dumps(record,indent=2))


if __name__=="__main__":
    main()
