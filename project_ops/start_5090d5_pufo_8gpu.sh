#!/usr/bin/env bash
set -eo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$OPS_DIR/.." && pwd)"
cd "$ROOT"
source "$OPS_DIR/activate_5090d5.sh"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

train_entrypoint=(robo_newton/scripts/rsl_rl/train.py)
case "${PUFO_BIND_GPU_LOCAL_CPUS:-0}" in
    0) ;;
    1) train_entrypoint=(robo_newton/scripts/bind_gpu_local_rank.py "${train_entrypoint[@]}") ;;
    *) echo "PUFO_BIND_GPU_LOCAL_CPUS must be 0 or 1" >&2; exit 2 ;;
esac

exec python -m torch.distributed.run --standalone --nproc_per_node=8 \
    "${train_entrypoint[@]}" --task RP1-PUFO --num_envs 1024 \
    --device cuda --distributed --visualizer none --logger wandb \
    --run_name pufo_contactguard_5090d5_8gpu1024 "$@"
