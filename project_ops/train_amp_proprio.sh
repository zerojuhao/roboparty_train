#!/bin/bash
set -eo pipefail
cd /home/zym/roboparty_train
source /data/zym/conda/envs/isaaclab_zym/bin/activate
set -u
export PYTHONPATH="/home/zym/roboparty_train/rsl_rl:/home/zym/roboparty_train/robo_newton${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES HEADLESS=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export CUDA_VISIBLE_DEVICES="${AMP_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ROBO_TRAIN_LOG_ROOT=/data/zym/roboparty_train/logs/rsl_rl
export XDG_CACHE_HOME=/data/zym/roboparty_train/cache
export TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/torchinductor TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export WARP_CACHE_PATH=$XDG_CACHE_HOME/warp CUDA_CACHE_PATH=$XDG_CACHE_HOME/cuda
export WANDB_DIR=/data/zym/roboparty_train/wandb WANDB_CACHE_DIR=$XDG_CACHE_HOME/wandb TMPDIR=/data/zym/roboparty_train/tmp
exec python -m torch.distributed.run --standalone --nproc_per_node="${AMP_NPROC_PER_NODE:-8}" \
  robo_newton/scripts/rsl_rl/train.py --task RP1-AMP-Path --visualizer none \
  --num_envs 1024 --device cuda --distributed --logger tensorboard \
  --max_iterations 5000 --run_name c18_rewards_no_actor_path_fresh_20261012 \
  env.commands.single_axis_velocity.ranges.lin_vel_x=[-0.5,1.2] \
  env.curriculum.velocity_tracking.params.min_velocity=1.2 \
  env.curriculum.velocity_tracking.params.max_velocity=4.0 \
  env.curriculum.velocity_tracking.params.step_size=0.1 \
  env.curriculum.velocity_tracking.params.up_threshold=0.8 \
  env.curriculum.velocity_tracking.params.up_required_windows=5 \
  env.curriculum.velocity_tracking.params.down_required_windows=2 \
  env.rewards.is_alive.weight=2.5 env.rewards.undesired_contacts.weight=-2.0 \
  env.rewards.track_lin_vel_xy_exp.weight=0.8 \
  env.rewards.lateral_command_offset.weight=-0.1 \
  env.rewards.lateral_command_offset.params.std=0.5 \
  env.rewards.lean_orientation_l2.params.forward_tolerance=0.17453292519943295 \
  env.rewards.high_speed_double_flight.params.max_flight_time=0.22 "$@"
