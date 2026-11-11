#!/bin/bash
# 正式盲测：基线与候选在 1000 个新正式开局上配对评估。候选冻结后只运行一次。
# 用法: tools/formal_eval.sh <candidate.pt> [额外候选 evaluate 参数]
set -e
cd /root/autodl-tmp/tank-flash
source activate.sh
model="$1"; shift || true
rm -rf runs/p5_formal_base runs/p5_formal_cand
python evaluate.py --model weights/pretrained.pt --tactics runs/p1_baseline200/source_tactics.py \
  --sample-policy --seed 1400000000 --episodes 1000 --workers 16 --trace \
  --output runs/p5_formal_base
python evaluate.py --model "$model" --sample-policy --seed 1400000000 --episodes 1000 \
  --workers 16 --trace --output runs/p5_formal_cand "$@"
python metrics.py --dir runs/p5_formal_base
python metrics.py --dir runs/p5_formal_cand
python tools/paired_stats.py runs/p5_formal_base runs/p5_formal_cand
