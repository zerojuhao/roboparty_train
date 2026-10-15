#!/bin/bash
# usage: launch_cycle.sh RUN_NAME LOAD_RUN CKPT MAX_ITERS [extra hydra overrides...]
set -eo pipefail
RUN_NAME=$1; LOAD_RUN=$2; CKPT=$3; MAXIT=$4; shift 4
cd /home/zym/roboparty_train
source /data/zym/conda/envs/isaaclab_zym/bin/activate
export PYTHONPATH="/home/zym/roboparty_train/rsl_rl:/home/zym/roboparty_train/robo_newton${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES HEADLESS=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROBO_TRAIN_LOG_ROOT=/data/zym/roboparty_train/logs/rsl_rl
export XDG_CACHE_HOME=/data/zym/roboparty_train/cache
export TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/torchinductor TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export WARP_CACHE_PATH=$XDG_CACHE_HOME/warp CUDA_CACHE_PATH=$XDG_CACHE_HOME/cuda
export WANDB_DIR=/data/zym/roboparty_train/wandb WANDB_CACHE_DIR=$XDG_CACHE_HOME/wandb TMPDIR=/data/zym/roboparty_train/tmp
exec python -m torch.distributed.run --standalone --nproc_per_node=8 \
  robo_newton/scripts/rsl_rl/train.py --task=RP1-AMP-Path --visualizer none --num_envs=1024 \
  --device cuda --distributed --max_iterations=$MAXIT --resume --load_run="$LOAD_RUN" \
  --checkpoint="$CKPT" --run_name="$RUN_NAME" --logger=tensorboard \
  agent.algorithm.learning_rate=2e-5 agent.algorithm.schedule=fixed agent.save_interval=500 \
  env.commands.single_axis_velocity.ranges.lin_vel_x=[-0.5,2.0] \
  env.curriculum.velocity_tracking.params.min_velocity=1.5 \
  env.curriculum.velocity_tracking.params.max_velocity=2.5 \
  env.curriculum.velocity_tracking.params.up_threshold=0.85 \
  env.curriculum.velocity_tracking.params.required_windows=5 \
  env.curriculum.velocity_tracking.params.step_size=0.3 \
  env.rewards.is_alive.weight=2.5 env.rewards.undesired_contacts.weight=-2.0 \
  env.rewards.flat_orientation_l2.weight=0.0 \
  env.rewards.lean_orientation_l2.weight=-6.0 \
  env.rewards.lean_orientation_l2.params.forward_tolerance=0.2617993877991494 \
  env.rewards.lean_orientation_l2.params.backward_scale=16.0 \
  env.rewards.track_lin_vel_xy_exp.weight=0.8 \
  env.rewards.lateral_command_offset.weight=-0.3 \
  env.rewards.lateral_command_offset.params.speed_gain=0.5 \
  env.rewards.turn_command_offset.weight=-0.3 \
  env.rewards.feet_air_time.params.turn_reward_scale=0.5 \
  "$@"