#!/usr/bin/env bash
set -eo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$OPS_DIR/.." && pwd)"
cd "$ROOT"
export PATH="/data/conda/envs/isaaclab_zym/bin:$PATH"
export PYTHONPATH="/home/zym/roboparty_train/rsl_rl:/home/zym/roboparty_train/robo_newton${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export HEADLESS=1
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROBO_TRAIN_LOG_ROOT=/data/zym/roboparty_train/logs/rsl_rl
export XDG_CACHE_HOME=/data/zym/roboparty_train/cache
export TORCHINDUCTOR_CACHE_DIR=/data/zym/roboparty_train/cache/torchinductor
export TRITON_CACHE_DIR=/data/zym/roboparty_train/cache/triton
export WARP_CACHE_PATH=/data/zym/roboparty_train/cache/warp
export CUDA_CACHE_PATH=/data/zym/roboparty_train/cache/cuda
export WANDB_DIR=/data/zym/roboparty_train/wandb
export WANDB_CACHE_DIR=/data/zym/roboparty_train/cache/wandb
export TMPDIR=/data/zym/roboparty_train/tmp
mkdir -p "$ROBO_TRAIN_LOG_ROOT" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
    "$TRITON_CACHE_DIR" "$WARP_CACHE_PATH" "$CUDA_CACHE_PATH" \
    "$WANDB_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

exec python -m torch.distributed.run --standalone --nproc_per_node=8 \
    robo_newton/scripts/rsl_rl/train.py --task RP1-PUFO --num_envs 1024 \
    --device cuda --distributed --visualizer none --logger wandb \
    --run_name pufo_v5_local8gpu1024_20261006 "$@"
