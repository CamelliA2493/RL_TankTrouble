"""统一启动入口；经典模式使用原版播放器，模型模式使用逐帧控制桥接。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
MODES = ("multiplayer", "user-laika", "user-model", "model-laika")
sys.dont_write_bytecode = True
for setting in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(setting,"1")


def command_for(mode, players=2):
    if mode in ("multiplayer", "user-laika"):
        label = f"controls{players}Player" if mode == "multiplayer" else "controls1Player"
        return [str(ROOT/"bin/ruffle.exe"),"--width","1000","--height","750","--storage","memory",
                "--tcp-connections","deny","--open-url-mode","deny","-PgameMode="+label,
                str(ROOT/"runtime/bootstrap.swf")]
    return [sys.executable,"-B","-X","utf8",str(ROOT/"windowed_flash_play.py"),
            "--mode","human" if mode == "user-model" else "laika"]


def start(mode, players=2):
    command = command_for(mode, players)
    print("正在启动原版游戏…", flush=True)
    if mode == "user-model":
        print("你：绿色2P，方向键移动/转向，M开火。模型：红色1P。", flush=True)
    elif mode in ("multiplayer", "user-laika"):
        print("按原版控制说明画面提示加入并开始。", flush=True)
    options = dict(stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) if mode in ("multiplayer","user-laika") else {}
    code = subprocess.run(command,cwd=ROOT/"runtime",**options).returncode
    if code:
        print(f"游戏退出代码：{code}。请检查bin和runtime是否完整。")
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--players", type=int, choices=[2,3], default=2)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    if args.print_command:
        import json
        print(json.dumps(command_for(args.mode or "multiplayer", args.players), ensure_ascii=False))
        return
    if args.mode:
        raise SystemExit(start(args.mode, args.players))
    while True:
        print("\n坦克动荡 v4.0\n1. 多人游玩（本地2人/3人）\n2. 用户对战Laika\n3. 用户对战预训练模型\n4. 观看模型对战Laika\n0. 退出")
        try:
            choice = input("选择玩法 [0-4]：").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if choice == "0":
            return
        if choice not in {"1","2","3","4"}:
            continue
        players = 2
        if choice == "1":
            answer = input("人数 [2/3，默认2]：").strip()
            if answer not in {"", "2", "3"}:
                print("原版本地多人支持2人或3人。")
                continue
            players = int(answer or "2")
        start(MODES[int(choice)-1], players)


if __name__ == "__main__":
    main()
