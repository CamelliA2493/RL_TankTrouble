"""便携混合策略入口；物理、对手和胜负仍由原版 SWF 执行。"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from flash_config import FlashConfig
from flash_env import OriginalFlashEnv
from policy import HybridPolicy, encode_observation
from tactics import control


DECISION_FRAMES = 13


class HybridController:
    def __init__(self, weights, sample_policy=False, policy_seed_offset=100000):
        torch.set_num_threads(1)
        self.model = HybridPolicy().eval()
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False)["policy"])
        self.sample_policy = sample_policy
        self.policy_seed_offset = policy_seed_offset
        self.reset()

    def reset(self, seed=0):
        self.hidden = None
        self.memory = {}
        self.selected = None
        self.aggression = .5
        self.last_action = 8
        self.last_decision = 0
        self.next_decision = 0
        # 独立随机流避免可视化、其他对局或第三方代码扰动策略采样。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + self.policy_seed_offset)
            self.rng_state = torch.get_rng_state()

    @torch.inference_mode()
    def act(self, obs, frame):
        if frame >= self.next_decision:
            context = [0 if self.selected is None else self.selected / 6, self.aggression,
                       self.last_action // 6 - 1, self.last_action // 2 % 3 - 1,
                       self.last_action % 2, (frame - self.last_decision) / 25]
            encoded = encode_observation(obs, context)
            outputs, self.hidden = self.model(torch.from_numpy(encoded).reshape(1, 1, 403), self.hidden)
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self.rng_state)
                choice, _, _, _ = self.model.actions(outputs, deterministic=not self.sample_policy)
                self.rng_state = torch.get_rng_state()
            selected = int(choice[0, 0, 0])
            if selected != self.selected:
                self.memory.clear()
            self.selected = selected
            self.aggression = float(choice[0, 0, 2])
            self.last_decision = frame
            self.next_decision = frame + DECISION_FRAMES
        action, detail = control(obs, self.memory, template=self.selected, aggression=self.aggression)
        self.last_action = action
        detail.update(selected_template=self.selected, aggression=self.aggression)
        return action, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent/"weights/pretrained.pt")
    parser.add_argument("--seed", type=int, default=710004000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-policy", action="store_true")
    parser.add_argument("--policy-seed-offset", type=int, default=100000)
    args = parser.parse_args()
    # 正式保留集只能通过冻结评估入口使用。
    assert not 930000000 <= args.seed <= 930000199
    args.output.mkdir(parents=True, exist_ok=False)
    agent = HybridController(args.model, args.sample_policy, args.policy_seed_offset)
    env = OriginalFlashEnv(FlashConfig(mode="laika", max_episode_frames=1500,
                                      auto_settle_dead=True, observe_events=True))
    try:
        obs, _ = env.reset(seed=args.seed)
        agent.reset(args.seed)
        with (args.output / "frames.jsonl").open("w") as stream:
            while True:
                frame = env.state["frame"]
                action, detail = agent.act(obs, frame)
                stream.write(json.dumps(dict(frame=frame, action=action, detail=detail, state=env.state)) + "\n")
                obs, _, ended, limited, info = env.step(action)
                if ended or limited:
                    stream.write(json.dumps(dict(frame=env.state["frame"], state=env.state, info=info)) + "\n")
                    break
        result = dict(seed=args.seed, info=info, model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                      renderer="original_state_replay", sample_policy=args.sample_policy,
                      policy_seed_offset=args.policy_seed_offset, goal_passed=False)
        (args.output / "summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
