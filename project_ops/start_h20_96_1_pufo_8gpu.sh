#!/usr/bin/env bash
set -eo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$OPS_DIR/.." && pwd)"
cd "$ROOT"
source "$OPS_DIR/activate_h20_96_1.sh"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

exec python -m torch.distributed.run --standalone --nproc_per_node=8 \
    robo_newton/scripts/rsl_rl/train.py --task RP1-PUFO --num_envs 1024 \
    --device cuda --distributed --visualizer none --logger tensorboard \
    --run_name pufo_v5_h20_96_1_8gpu1024 "$@"
