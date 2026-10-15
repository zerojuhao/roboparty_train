#!/bin/bash
set -euo pipefail
cd /home/zym/roboparty_train

source_run=2026-10-15_02-31-41_amp_clean_v29_physics_peak_braking_power_from47500_5000_resume_20261015
checkpoint=/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/$source_run/model_48000.pt
if [[ ! -f "$checkpoint" ]]; then
    echo "Missing source checkpoint: $checkpoint" >&2
    exit 1
fi

exec bash project_ops/train_amp_proprio.sh \
    --resume --load_run "$source_run" --checkpoint model_48000.pt \
    --run_name amp_clean_v31_braking_power01_from48000_5000_resume_20261015 \
    --max_iterations 5000 --resume_learning_rate 1e-5 \
    agent.algorithm.learning_rate=1e-5 agent.algorithm.schedule=fixed \
    agent.algorithm.entropy_coef=0.002 \
    env.rewards.track_lin_vel_xy_exp.weight=1.2 \
    env.rewards.track_lin_vel_xy_exp.params.std=0.5 \
    env.rewards.track_lin_vel_xy_exp.params.use_braking_reference=false \
    env.rewards.track_lin_vel_xy_exp.params.braking_reward_scale=1.0 \
    env.rewards.track_lin_vel_xy_exp.params.braking_min_speed=0.2 \
    env.rewards.track_lin_vel_xy_exp.params.braking_max_command=0.1 \
    env.rewards.joint_torques_l2.weight=-1e-5 \
    env.rewards.joint_torque_saturation_l2.weight=-0.1 \
    env.rewards.joint_torque_saturation_l2.params.limit_ratio=0.95 \
    env.rewards.joint_torque_saturation_l2.params.use_substep_peak=true \
    env.rewards.braking_power_peak_l2.weight=-0.1 \
    env.rewards.braking_power_peak_l2.params.power_threshold=2500.0 \
    env.rewards.braking_torque_step_l2.weight=-0.01 \
    env.commands.single_axis_velocity.braking_duration_range=null \
    env.commands.single_axis_velocity.acceleration_duration_range=null \
    env.commands.single_axis_velocity.stop_start_time_range=[0.5,1.5] \
    env.commands.single_axis_velocity.stop_peak_speed_probability=0.5 \
    env.commands.single_axis_velocity.stop_sequence_probability=0.3 \
    env.commands.single_axis_velocity.stop_hold_time_range=[5.0,8.0] \
    env.rewards.lateral_command_offset.weight=-0.5 \
    env.rewards.lateral_command_offset.params.std=0.5 \
    env.rewards.feet_slide.weight=-0.5 \
    env.rewards.feet_close_y_gauss.params.threshold=0.18 \
    env.rewards.feet_close_y_gauss.weight=-0.5 \
    env.rewards.knee_close_y_gauss.params.threshold=0.18 \
    env.rewards.knee_close_y_gauss.weight=-0.5 \
    env.rewards.sound_suppression.weight=-5e-4 \
    env.rewards.stand_still_joint_deviation.weight=-0.1 \
    env.rewards.hip_yaw_deviation_l1.weight=-0.1 \
    env.rewards.elbow_position_range_l2.weight=-0.05 \
    env.rewards.elbow_position_range_l2.params.angle_scale=0.08726646259971647 \
    env.rewards.elbow_position_range_l2.params.min_linear_command=1.5 "$@"
