# RP1 AMP Raw-Step Braking

Prepared on 2026-10-14 for 5090D-6. Syncing these files does not start training.

## Training Change

- Actor and critic velocity-command observations remain raw, including an
  immediate 4 -> 0 m/s stop. Observation sizes and ONNX deployment are unchanged.
- Only `track_lin_vel_xy_exp` uses the command term's `tracking_command`.
  Linear command reductions/reversals get a cubic smoothstep braking reference.
  Acceleration and yaw tracking remain unchanged.
- A 4 m/s reduction takes a uniformly sampled 2-3 seconds in the reward
  reference; smaller drops scale proportionally, with a 0.1-second minimum.
  This is a training target, not a guaranteed stopping time or safety limiter.
- The reference finishes at the raw target. Continued motion after a stop
  then loses velocity-tracking reward. Resets discard old references; an
  interrupted braking segment restarts from its current reference.
- Restore standing-pose weight to -0.1, the v21 baseline. The v22 -0.5
  experiment learned to stay above the measured-speed gate.
- No PD, action filter, torque limit, actor observations or electrical model
  changes. Existing action-rate penalty stays -0.01 for this first experiment.

## Resume Later

Source checkpoint and its saved configuration must be present at:

```text
/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/
2026-10-13_16-54-50_amp_clean_v21_elbow05_entropy002_from30500_8gpu_5000_resume_20261014/
model_35000.pt
```

Check GPU availability before starting. The launcher uses eight GPUs by default;
`AMP_CUDA_VISIBLE_DEVICES` and `AMP_NPROC_PER_NODE` can select only free GPUs.

```bash
bash project_ops/train_amp_braking_resume.sh
```

Keep learning rate 5e-5, fixed schedule, entropy coefficient 0.002, optimizer
state and checkpoint curriculum (max velocity 4 m/s). Add 5000 iterations,
save every 500, ending at model_39999.pt.

Run the fixed evaluator in a separate tmux window when starting training:

```bash
source_run=2026-10-13_16-54-50_amp_clean_v21_elbow05_entropy002_from30500_8gpu_5000_resume_20261014
run_name=amp_clean_v23_rawstep_smooth_braking1to2s_from35000_5000_resume_20261014
run_root=/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MUJOCO_GL=egl \
  /data/zym/conda/envs/isaaclab_zym/bin/python \
  project_ops/scripts/watch_amp_checkpoint_evaluations.py \
  --run-root "$run_root" --run-name "$run_name" \
  --baseline "$run_root/$source_run/model_35000.pt" \
  --final-iteration 39999 --timeout-hours 6 --turn-tests \
  --xml /data/zym/roboparty_train/robo_newton_data/robots/roboparty/rp1.3/mjcf/rp1_flat.xml \
  --output-dir "/data/zym/roboparty_train/sim2sim_checks/$run_name"
```

## Evaluation

Keep the existing 31 abrupt-stop timings with raw 4 -> 0 commands: do not
filter the inference command to hide abrupt-stop failures. In addition to
speed, lateral drift, stop duration/distance, pose and slip, reports now
include the first three seconds' target-position increments, peak joint
torque, summed mechanical braking power and braking energy. Power/torque
peaks include all four 200-Hz physics substeps per 50-Hz policy step.

These are simulation proxies, not measured bus voltage or electrical
regeneration power. Hardware validation still needs bus voltage/current,
drive fault codes, a protected test setup and adequate stopping space.
