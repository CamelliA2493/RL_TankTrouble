#!/bin/bash
# 重算若干 runs 目录的 metrics 并汇总。
set -e
cd /root/autodl-tmp/tank-flash
source activate.sh
for d in "$@"; do
  python metrics.py --dir "$d" > /dev/null
done
python tools/report.py "$@"
