# 用 source 加载，不要直接 bash 执行。
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/tank-runtime/conda/flash-rl
export CARGO_HOME=/root/autodl-tmp/tank-runtime/cargo
export RUSTUP_HOME=/root/autodl-tmp/tank-runtime/rustup
export CARGO_TARGET_DIR=/root/autodl-tmp/tank-runtime/cargo-target
export PATH="$CARGO_HOME/bin:$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
