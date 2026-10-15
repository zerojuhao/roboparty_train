#!/usr/bin/env bash
# Source this file on H200-2; it does not start training.

if [ ! -f /data/zym/conda/envs/isaaclab_zym/bin/activate ]; then
    echo "Missing H200-2 environment: /data/zym/conda/envs/isaaclab_zym" >&2
    return 1 2>/dev/null || exit 1
fi

source /data/zym/conda/envs/isaaclab_zym/bin/activate
export PYTHONPATH="/home/zym/roboparty_train/rsl_rl:/home/zym/roboparty_train/robo_newton${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# User accepted the NVIDIA Omniverse EULA on 2026-10-03.
export OMNI_KIT_ACCEPT_EULA=YES
export HEADLESS=1
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
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
