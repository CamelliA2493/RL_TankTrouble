# Laika99 后训练交付

## 结论（如实声明）

**未达到 99% 胜率目标，也未通过阶段 5 的全部门槛。** 本次在预算内完成了阶段 0–5 的工程与
测量闭环，交付实际结果、最佳候选和未通过项，不宣称达标。

| 指标 | 基线（原 88% 系统） | 候选（阶段 2/3 战术） | 门槛 | 判定 |
|---|---|---|---|---|
| 正式 1000 局胜率 | 902/1000 = **90.2%** [88.2, 91.9] | 873/1000 = **87.3%** [85.1, 89.2] | ≥990/1000 | 未通过 |
| 抖动 J（每千可控帧） | 522.4 | **350.0** | 相对 −30% | 通过（−33.0%，CI [−35.3, −30.8]） |
| 道具 U | 0/72 = 0% | 0/49 = 0%（7 拾取/6 使用） | +20% 且 +2pp | 未通过（合计机会 85/65 <100） |
| 每百局有效使用 | 0.0 | 0.0 | 增加 | 未通过 |
| 部署一致性（20 局、3594 帧） | — | 逐帧动作/按键 0 差异 | 一致 | 通过 |

配对检验：正式集胜率差 **−2.9pp**（95% CI [−5.2, −0.6]，McNemar p=0.019）；
补充诊断集 +5.0pp（CI [−0.5, +10.5]，p=0.11）——两套种子方向相反，存在地图集异质性。
J 差 −172.4/千帧（CI [−184.4, −160.7]）。

## 交付物

| 文件 | 说明 |
|---|---|
| `candidate_model.pt` | 最佳候选权重，SHA256 `113b6694…b74287`（与 `weights/pretrained.pt` 相同；阶段 4 训练未超过预训练，按计划保留预训练权重） |
| `candidate_tactics.py` | 通过阶段 2 J 门槛与阶段 3 道具规则的战术源码，SHA256 `2014031e…37283` |
| `candidate_policy.py` | 403 维观测/GRU 策略源码，SHA256 `71adb77f…f6bd` |
| `candidate_config.json` | 采样解码、策略种子偏移、决策周期、追箱开关与哈希 |
| `baseline_tactics.py` | 基线冻结战术，SHA256 `112bb685…2eff`（正式对照与 88% 历史成绩使用） |
| `EVALUATION.md` | 阶段 0–5 结果、指标定义与未通过项 |
| `highlights.mp4` | 4 局模型 vs 官方 Laika 精彩对局（带标题卡，1000×750，25fps） |

原始 `weights/pretrained.pt` 未改动（哈希在阶段 0 与交付时一致）。所有实验在服务器
`runs/<name>/` 独立目录中，含冻结源码、逐局记录与指标。

## 启动与复现

本地双击 `启动游戏.cmd`（选项 4 观看模型对 Laika）。候选模型的窗口/回放：

```bash
# 原生窗口：候选战术 + 候选权重（Windows）
python windowed_flash_play.py --mode laika --seed 710004000 \
  --model delivery/candidate_model.pt --chase-crates --record runs/window.jsonl

# 无窗口逐帧回放（部署路径，输出每帧动作与状态）
python deploy.py --model delivery/candidate_model.pt --seed 710004000 \
  --chase-crates --output runs/deploy_check

# 评估（采样解码，与训练/部署一致）
python evaluate.py --model delivery/candidate_model.pt \
  --tactics delivery/candidate_tactics.py --sample-policy --chase-crates \
  --seed 1400000000 --episodes 1000 --workers 16 --trace --output runs/recheck

# 基线（原 88% 系统）
python evaluate.py --model weights/pretrained.pt \
  --tactics delivery/baseline_tactics.py --sample-policy \
  --seed 1400000000 --episodes 1000 --workers 16 --trace --output runs/recheck_base
```

运行候选战术需要同时提供 `candidate_tactics.py`、`candidate_policy.py` 与项目其余环境文件
（`flash_env.py`、`flash_config.py`、`flash_navigation.py`、`bin/`、`runtime/`）。服务器上的
复现命令见 `EVALUATION.md`。

## 训练恢复

阶段 4 的训练入口已支持奖励模式、优化器、学习率与追箱开关；检查点含源码哈希、PPO 配置、
训练配置和 RNG 状态：

```bash
python trainer.py --resume runs/p4_main500k/checkpoint_029.pt \
  --output runs/resume_check --frames 200000 \
  --lr 3e-4 --optimizer adam --reward-mode standard --chase-crates \
  --seed-start 1000000000
```

`--resume` 会校验源码、PPO 配置、训练配置、随机种子与采样批次；不匹配会拒绝恢复。重新从
预训练权重开始新实验：

```bash
python trainer.py --initial weights/pretrained.pt --output runs/new \
  --frames 500000 --workers 16 --episodes-per-rollout 16 --lr 3e-4 --chase-crates
```

## 未通过项与原因

1. **99% 胜率**：正式 873/1000（候选）/ 902/1000（基线），差距 117/88 胜。低层战术的 J 修复
   让坦克在对准阶段继续前进，静止占比从 73% 降到 56%，但换来更多近距离交火与自杀/同时死亡
   （候选 68 负 59 平 vs 基线 54 负 44 平）。
2. **胜率护栏**：阶段 2 的 J 修复在选模集上 −2.5pp（McNemar p=0.42），正式集上 −2.9pp
   （p=0.019）。根因是反弹弹自击：自击裕度把自杀从 12 次降到 8 次，剩余 3 次为击杀后旧弹返回。
3. **道具 U**：追箱与开火修复把拾取从 1 次提升到 7 次、使用 6 次，但 6 次激光开火均未命中，
   有效使用仍为 0；正式集机会 49 < 100。根因是特殊武器沿直线/不同弹道开火，而开火判定沿用
   反弹弹几何（对激光应要求直射）。
4. **后训练**：Adam 1e-4 与 AdamW 1e-4 各 5 万帧在 100 局诊断集上同为 91% 且动作一致；
   Adam 3e-4 续训 10 万帧降到 86%（预训练 89%），熵从 1.34 升到 1.49。判定 PPO 在当前
   稀疏奖励、16 局/rollout 配置下未超过预训练，保留预训练权重。
