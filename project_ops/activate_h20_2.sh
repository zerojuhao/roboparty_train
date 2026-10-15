#!/usr/bin/env bash
# Source on H20-2 (SSH port 51821). This does not start training.

if [ ! -f /data1/zym/conda/envs/isaaclab_zym/bin/activate ]; then
    echo "Missing H20-2 environment: /data1/zym/conda/envs/isaaclab_zym" >&2
    return 1 2>/dev/null || exit 1
fi

source /data1/zym/conda/envs/isaaclab_zym/bin/activate
export PYTHONPATH="/home/zym/roboparty_train/rsl_rl:/home/zym/roboparty_train/robo_newton${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Preserve the user's existing NVIDIA EULA acceptance.
export OMNI_KIT_ACCEPT_EULA=YES
export HEADLESS=1
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export ROBO_TRAIN_LOG_ROOT=/data1/zym/roboparty_train/logs/rsl_rl
export XDG_CACHE_HOME=/data1/zym/roboparty_train/cache
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export WANDB_DIR=/data1/zym/roboparty_train/wandb
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export TMPDIR=/data1/zym/roboparty_train/tmp
mkdir -p "$ROBO_TRAIN_LOG_ROOT" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
    "$TRITON_CACHE_DIR" "$WARP_CACHE_PATH" "$CUDA_CACHE_PATH" \
    "$WANDB_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"
