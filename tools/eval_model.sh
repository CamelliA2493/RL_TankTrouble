#!/bin/bash
# 评估单个检查点：tools/eval_model.sh <model.pt> <output_dir> <seed> <episodes> [额外 evaluate 参数]
set -e
cd /root/autodl-tmp/tank-flash
source activate.sh
model="$1"; output="$2"; seed="$3"; episodes="$4"; shift 4
rm -rf "$output"
python evaluate.py --model "$model" --sample-policy --seed "$seed" --episodes "$episodes" \
  --workers 16 --trace --output "$output" "$@"
python metrics.py --dir "$output" > /dev/null
python tools/report.py "$output"
