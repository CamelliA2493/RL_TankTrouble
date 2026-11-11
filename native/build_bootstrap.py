"""使用JPEXS编译独立启动器，不改写原版电影。传入ffdec-cli.jar路径。"""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile
import zlib


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ffdec", type=Path)
    args = parser.parse_args()
    raw = (ROOT/"runtime/TankTrouble_v4.0.swf").read_bytes()
    body = zlib.decompress(raw[8:]) if raw[:3] == b"CWS" else raw[8:]
    rect_size = (5 + 4*(body[0] >> 3) + 7)//8
    header = body[:rect_size+2] + struct.pack("<H", 1)
    tags = struct.pack("<HI", (69 << 6) | 4, 0)
    tags += struct.pack("<H", (12 << 6) | 1) + b"\0" + struct.pack("<HH", 1 << 6, 0)
    with tempfile.TemporaryDirectory(prefix="tank-bootstrap-") as directory:
        blank = Path(directory)/"blank.swf"
        blank.write_bytes(b"FWS"+raw[3:4]+struct.pack("<I",len(header+tags)+8)+header+tags)
        scripts = subprocess.check_output(["java","-jar",str(args.ffdec.resolve()),"-dumpAS2","-exportNames",str(blank)],text=True)
        script = next(line.strip() for line in scripts.splitlines() if "DoAction" in line)
        subprocess.run(["java","-jar",str(args.ffdec.resolve()),"-replace",str(blank),
                        str(ROOT/"runtime/bootstrap.swf"),script,str(ROOT/"native/bootstrap.as")],check=True)


if __name__ == "__main__":
    main()
