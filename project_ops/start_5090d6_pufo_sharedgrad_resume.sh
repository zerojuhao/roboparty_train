#!/usr/bin/env bash
set -eo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$OPS_DIR/.."
source "$OPS_DIR/activate_5090d6.sh"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PUFO_RESUME_RUN=2026-10-13_velocity_ab_initial
PUFO_RESUME_MODEL=model_124805120.pt
test -f "$ROBO_TRAIN_LOG_ROOT/rp1_pufo_slim512/$PUFO_RESUME_RUN/$PUFO_RESUME_MODEL"

exec python -m torch.distributed.run --standalone --nproc_per_node=8 \
    robo_newton/scripts/rsl_rl/train.py --task RP1-PUFO-SLIM512 --num_envs 1024 \
    --device cuda --distributed --visualizer none --logger wandb \
    --resume --load_run "^${PUFO_RESUME_RUN}$" --checkpoint '^model_124805120[.]pt$' \
    --run_name pufo_slim512x6_sharedgrad_resume124805120_5090d6 \
    agent.velocity_estimator_updates_encoder=true \
    agent.num_env_steps=1000000000 \
    env.scene.camera.ray_alignment=base
