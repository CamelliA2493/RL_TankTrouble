#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT/activate.sh"
python "$PROJECT/native/prepare_windowed_host.py"
export CARGO_BUILD_JOBS=4
export CARGO_PROFILE_RELEASE_OPT_LEVEL=2
export CARGO_PROFILE_RELEASE_LTO=false
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=64
export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc
cd "${TANK_RUNTIME:-/root/autodl-tmp/tank-runtime}/visual-sources/ruffle"
cargo build --locked --release --target x86_64-pc-windows-gnu -p ruffle_core --example tank_window --example tank_rl
cp "$CARGO_TARGET_DIR/x86_64-pc-windows-gnu/release/examples/tank_window.exe" "$PROJECT/bin/tank-flash-window.exe"
cp "$CARGO_TARGET_DIR/x86_64-pc-windows-gnu/release/examples/tank_rl.exe" "$PROJECT/bin/tank-flash-runner.exe"
