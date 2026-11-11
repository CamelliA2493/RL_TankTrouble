# 坦克动荡 ：混合策略强化学习

双击 **启动游戏.cmd**，选择玩法：

| 选项 | 玩法 |
|---|---|
| 1 | 本地双人或三人，共用键盘，按原版说明加入 |
| 2 | 用户对战官方Laika，方向键移动/转向，M加入及开火 |
| 3 | 用户对战预训练模型：你控制绿色2P（方向键、M），模型控制红色1P |
| 4 | 观看模型控制红色坦克对战Laika |

全部使用原版SWF画面与游戏逻辑。经典模式使用完整Ruffle播放器；模型模式使用逐帧同步的原生GPU窗口，关闭窗口即可退出。模型模式目前未接入音效与鼠标菜单。默认目标25FPS，计算较慢时放慢墙钟速度，不跳过物理帧。

本地完整运行包已内置Windows独立CPU运行时，不需要安装Python、conda、CUDA或WSL。已有conda环境未改动。Git源码仓库不包含该运行时和执行器，直接克隆后尚不能双击启动。

## Git仓库与运行包

`.gitignore`排除了`python/`、`bin/`、训练输出、缓存和本机配置，但保留`weights/pretrained.pt`及其评估元数据。完整运行包与原生执行器适合单独通过Release分发；仅推送源码不会上传本地依赖。

其他机器需要另行安装`requirements.txt`中的依赖，并提供`bin/`下对应平台的执行器。`activate.sh`默认指向当前服务器的环境路径，在其他机器上需要调整；原生Windows桥接的构建入口为`native/build_windows.sh`，依赖外部固定版本Ruffle源码和Rust交叉编译环境。

原版SWF的来源和哈希记录在`runtime/sources.json`；该来源记录不代表已确认再分发许可。公开发布原版游戏资源前需确认其许可。`native/Ruffle_LICENSE.md`仅说明Ruffle的许可；本项目自己的代码尚未指定开源许可证。

## 保留内容

- `weights/pretrained.pt`：唯一保留的模型检查点，神经参数未改动。
- `weights/pretrained.json`：模型结构、哈希及原版Laika评估记录。
- `trainer.py`、`policy.py`、`tactics.py`、`evaluate.py`、`deploy.py`：训练、战术、评估与推理代码。
- `flash_env.py`、`flash_config.py`、`flash_navigation.py`：原版环境与桥接。
- `runtime/`、`bin/`、`python/`：必要游戏资源、执行器及CPU运行时。
- `native/`：原生桥接与启动器源码、构建依据和许可证。

预训练网络为403维观察、hidden128、GRU32的混合策略。算法负责战术，网络选择高层模板及攻击性。冻结完整系统在原版Flash v4.0对官方Laika的200局测试中为176胜、16负、8平，胜率88%。

## 以后继续训练

Linux服务器复用原有CUDA环境，在项目目录执行：

```bash
source activate.sh
chmod +x bin/tank-flash-runner
python trainer.py --output runs/finetune --frames 200000
```

默认从`weights/pretrained.pt`加载策略参数并建立新的优化器，属于预训练后的微调。若要随机初始化，加`--from-scratch`。参数集中在`trainer.py`的`PPO_CONFIG`。

新训练的断点恢复：

```bash
python trainer.py --resume runs/finetune/checkpoint_005.pt --output runs/resumed --frames 400000
```

`--resume`会检查源码、PPO配置、随机种子和采样批次；使用的是累计帧预算。保留下来的原始预训练检查点适合默认`--initial`微调，整理前的旧源码哈希与当前目录布局不同，不用它直接执行`--resume`。

Windows可用内置CPU运行时检查训练流程：

```bat
python\python.exe trainer.py --device cpu --workers 1 --episodes-per-rollout 2 --frames 1000 --output runs/debug
```

评估微调后的模型时保持采样解码：

```bash
python evaluate.py --model weights/pretrained.pt --sample-policy --episodes 100 --seed 920010000 --output runs/evaluation
```

原版电影与资源字节未修改，来源及哈希见`runtime/sources.json`。游戏首次启动可能需要数秒。
