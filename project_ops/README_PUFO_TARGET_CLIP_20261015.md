# PUFO absolute position target clipping

Date: 2026-10-15 (Asia/Shanghai). Evaluation host: H20-96-4.

## Change

The final absolute PD target is clipped to each joint's physical position
range, after action rescaling and default-position offset. The raw action
history is unchanged. Physical positions/velocities are never projected,
and collision shapes, filtering, solver settings and reward weights are unchanged.
The 24 physical joint ranges in the RP1 URDF and MJCF match exactly.

- `robo_newton/scripts/mujoco/sim2sim_rp1_pufo.py`: clipping on by default;
  `prepare_action_target` is shared with the measured simulation.
- `project_ops/scripts/evaluate_pufo_push_recovery.py`: `--target-clip-ab`
  requires the same checkpoint and a fixed push pulse. Left is the old
  unclipped controller; right is target clipping. Both raw and applied
  targets and physical joint ranges are saved in the trace archives.
- `robo_newton/robolab/tasks/manager_based/ufo/mdp/rewards.py`:
  `JointLimitedBodyImpactPositionAction` clips processed absolute targets.
  The old `BodyImpactJointPositionAction` remains unchanged for saved configs.
- PUFO's new default action class uses clipping. Its training contract is
  `pufo_h200_alignment_v7_target_clip`. The existing loader intentionally
  rejects a direct v6 resume, preventing reuse of incompatible replay.
  A future v6-to-v7 training continuation needs an explicit weights/optimizer
  migration with fresh replay; this task did not perform that migration.

Files were synchronized between local 5090D-5 and H20-96-4's canonical
`/home/zym/roboparty_train/robo_newton` package. H20 also has an older
`/home/zym/roboparty_train/robolab` directory, which can shadow the canonical
package when running `python -m unittest` from the repository root. Tests
were rerun from `robo_newton` to resolve the intended package. The unrelated
older root-level copy was not overwritten or removed.

The existing `rp1-pufo-jvel10x` training was not stopped or restarted and
continues with its already-loaded, unclipped action implementation. No new
model was trained here. The separate real-robot deployment package was not
modified; its PD target path still needs the same rule before deployment.

## Controlled Evaluation

Both panels use:

`/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo/2026-10-15_03-41-14_pufo_1024x6_depthaug50_jvel10x_resume_h20_96_4_8gpu1024/model_336003072.pt`

This deliberately uses the previously tested 336M model, not a later checkpoint.

- Flat-ground, 30-second rollouts, forward and backward separately.
- Hold deployment context row 0, source dance1_subject2 frame 50; future
  window 5, gamma 0.8; heading and velocity inputs masked to zero.
- Waist COM force: world +/-X, 300 N, starts at 3 s with a 0.1 s ramp,
  requested cutoff 3.75 s, actual control-tick cutoff 3.76 s.
- Identical force arrays and impulse (210 Ns) for both panels. No automatic
  pose resets, lifting, state clipping or changes to self-collision.
- Recovery confirmation requires 50 consecutive 50 Hz samples of upright,
  low-speed standing with both feet loaded and no significant body support.
  Times below are from force removal to confirmation, including the stable
  standing verification interval.

| Direction | Unclipped | Target clipped |
| --- | --- | --- |
| Forward | No recovery by 30 s | Recovery confirmed after 3.16 s; stable through end |
| Backward | Recovery confirmed after 2.50 s | Recovery confirmed after 2.44 s; stable through end |

The clipped forward rollout no longer reaches the stuck left-knee pose:
minimum left-knee angle is -1.720 rad, versus the previous persistent
approximately -2.907 rad state. This is a single deterministic test per
direction, not a general recovery-success-rate estimate.

## Limits and Remaining Risks

- Applied target violations: exactly zero in both clipped trials.
- Actual joint angles can still overshoot. At 50 Hz, maximum clipped-trial
  overshoot was 0.04871 rad at the right ankle pitch (forward, 3.62 s), and
  0.01195 rad at the left ankle roll (backward, 4.62 s). These are sampled
  maxima, not guaranteed physics-substep maxima.
- Backward recovery-window whole-body speed RMS: 2.946 -> 2.538 rad/s;
  however sampled foot net vertical contact force peak: 783.7 -> 1844.2 N.
  The interval includes the fall impact, not only the subsequent get-up.
- Clipped forward trial's left wrist reaches approximately 50.77 rad/s
  during impact at 4.185 s. Target clipping is not a velocity limiter.
- Do not claim general gentler contact or hardware safety from successful
  recovery. This is CPU Torch/MuJoCo evaluation, not the complete ONNX
  deployment stack with hardware filters, delays and motor/sensor dynamics.

## Artifacts and Checks

Local artifacts:

`/home/zym/roboparty_train/videos/pufo_target_clip_336m_20261015/`

Each `forward` / `backward` folder contains `comparison.mp4`,
`before_traces.npz`, `after_traces.npz`, and `report.json`. Local
`review_frames.jpg` files show 4 s, 7 s and 29 s for visual inspection.
Top-level `setup.json` and `report.json` retain exact settings and results.

Remote artifacts:

`/data/zym/roboparty_train/evaluations/pufo_target_clip_336m_20261015/`

Console log:

`/data/zym/roboparty_train/evaluations/pufo_target_clip_336m_20261015.log`

Verification:

- Unclipped root position, joint position, PD samples, force, context and
  targets exactly match the previous matched-pulse 336M evaluation.
- Force and context arrays exactly match between panels.
- Both videos fully decoded: 751 frames, 25 fps, 30.04 s, 1280x740.
  Frames at 4 s, 7 s and 29 s were visually inspected.
- Local tests: 14 contact/action tests, 3 target-clip tests, 4 recovery
  helpers, 9 stair/comparison helpers passed.
- H20 tests: 14 contact/action tests (one P3-reference test skipped because
  its reference archive is absent), 3 target-clip tests, 4 recovery helpers,
  9 stair/comparison helpers and one real-model action-history test passed.

Reproduce on H20 (output directory must not exist):

```bash
cd /home/zym/roboparty_train
clip_checkpoint=/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo/2026-10-15_03-41-14_pufo_1024x6_depthaug50_jvel10x_resume_h20_96_4_8gpu1024/model_336003072.pt
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=2 \
  /data/zym/conda/envs/isaaclab_zym/bin/python -u \
  project_ops/scripts/evaluate_pufo_push_recovery.py \
  --before "$clip_checkpoint" --after "$clip_checkpoint" \
  --output /data/zym/roboparty_train/evaluations/pufo_target_clip_336m_repeat \
  --target-clip-ab --fixed-pulse --force 300 --push-end 3.75 --duration 30
```
