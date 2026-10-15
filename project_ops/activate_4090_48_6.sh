#!/usr/bin/env bash
# Source on 4090-48-6; this does not start training.

ENV_PREFIX=/home/zym/conda/envs/isaaclab_zym
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    echo "Missing 4090-48-6 environment: $ENV_PREFIX" >&2
    return 1 2>/dev/null || exit 1
fi

if [ -f /data/conda/bin/activate ]; then
    source /data/conda/bin/activate "$ENV_PREFIX"
else
    export PATH="$ENV_PREFIX/bin:$PATH"
    export CONDA_PREFIX="$ENV_PREFIX"
fi
export PYTHONPATH="/home/zym/roboparty_train/rsl_rl:/home/zym/roboparty_train/robo_newton${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export HEADLESS=1
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export ROBO_TRAIN_LOG_ROOT=/home/zym/roboparty_train_data/logs/rsl_rl
export XDG_CACHE_HOME=/home/zym/roboparty_train_data/cache
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export WANDB_DIR=/home/zym/roboparty_train_data/wandb
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export TMPDIR=/home/zym/roboparty_train_data/tmp
mkdir -p "$ROBO_TRAIN_LOG_ROOT" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
    "$TRITON_CACHE_DIR" "$WARP_CACHE_PATH" "$CUDA_CACHE_PATH" \
    "$WANDB_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"
