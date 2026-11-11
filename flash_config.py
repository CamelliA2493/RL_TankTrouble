from dataclasses import dataclass
from pathlib import Path
import os


PROJECT = Path(__file__).resolve().parent
RUFFLE_COMMIT = "89a7049965f3090e706265670df309a5128233ac"
GAME_SHA256 = "188062aff7f7d969abb7f453f661d68f77d6ad9faf95baeaf8029a882c840a96"
OBSERVATION_SCHEMA = "flash-structured-v1"
CONTROL_OBSERVATION_SCHEMA = "flash-structured-controls-v2"
SHOT_PATH_OBSERVATION_SCHEMA = "flash-structured-shot-paths-v3"
NAVIGATION_OBSERVATION_SCHEMA = "flash-structured-navigation-v4"
WEAPONS = ("bullet", "laser", "frag", "gatling", "deathRay", "homing", "mine", "remote", "electric", "unknown")
ENTITY_TYPES = ("electricbullet", "deathray", "bullet", "laser", "frag", "gatling", "homing", "mine", "remote", "crate", "aimer", "shield", "eltoro")


@dataclass
class FlashConfig:
    mode: str = "laika"  # 原版单人 Laika；duel 为原版双人，第二辆可由外部策略控制
    max_episode_frames: int = 1500
    action_repeat: int = 1
    auto_settle_dead: bool = False  # 旧实验默认逐帧；新训练在己方死亡后自动执行原版结算
    include_controls: bool = False  # 显式版本开关，旧模型的观测不变
    include_shot_paths: bool = False  # 原版 AI 的静态目标弹道查询，不提供专家按键
    include_navigation: bool = False  # 只读公开墙体的图距离，不输出方向或按键
    observe_events: bool = False
    seed_pool: tuple = ()
    timeout_penalty: float = 0.0
    reward_mode: str = "standard"  # win_events 提前给出击毁信号并在被追平时扣回，训练需 gamma=1
    max_entities: int = 256
    max_width: int = 12
    max_height: int = 10
    request_timeout: float = 90.0
    runner: str = str(PROJECT / ("bin/tank-flash-runner.exe" if os.name=="nt" else "bin/tank-flash-runner"))
    movie: str = str(PROJECT / "runtime/TankTrouble_v4.0.swf")
    navigation_reward_scale: float = 0.25


def observation_schema(config):
    if config.include_navigation:
        return NAVIGATION_OBSERVATION_SCHEMA
    if config.include_shot_paths:
        return SHOT_PATH_OBSERVATION_SCHEMA
    return CONTROL_OBSERVATION_SCHEMA if config.include_controls else OBSERVATION_SCHEMA
