# PUFO depth and standing stability: 2026-10-13

## Fixed checkpoint and scope

The latest published checkpoint at the start of inspection on 5090D-6 was:

```text
/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo_slim512/
2026-10-12_03-13-19_pufo_slim512x6_5090d6_8gpu1024/model_115204096.pt
SHA256: 11feac87dd90bb5214065764fc84342ffd8dfd044e2fb1db0dea9653de197e62
```

Remote and local checkpoint hashes match. All experiments below use this fixed
115,204,096-step snapshot, not subsequent checkpoints from the live job.
The saved actor has six 512-wide layers. Depth input is 8 x 18 x 36 with
37-sample history, skip 5, integer delay 0/1, and crop (18, 0, 14, 14).
The runtime camera follows full base roll, pitch, and yaw even though this run's
serialized configuration still says `yaw`. No camera mode was changed for testing.

The subsequent shared-gradient continuation published
`model_134406144.pt` at 134,406,144 global steps. Its remote and local SHA256 is
`fb1daebe4e3a98f193d556ebea2180fb9cd0201a07c72b66afea7f67b0cd171d`. The
latest standing checks below use that fixed snapshot while the live continuation
continues independently.

Artifacts are under:

```text
/data/zym/roboparty_train/migration/pufo_depth_diagnosis_20261013/
```

Tests use CPU deterministic actor inference and the existing MuJoCo sim2sim
tool, with 50 Hz control, 200 Hz PD, and 400 Hz physics. The source AMP job and
the 5090D-6 PUFO training job were not stopped, restarted, or reconfigured.
No model weights or production training/runtime configuration were changed.

## Baseline videos

### Latest checkpoint standing check

For `model_134406144.pt`, default joints with zero velocity and frame-50 context
were held for 20 s on flat ground. The robot remained upright and did not show
sustained backward drift: forward displacement was +0.00489 m and the maximum backward excursion
was only 0.00383 m. Maximum tilt was 1.38 degrees and late tilt RMS was 0.606
degrees. The dataset frame-50 pose held for 15 s had +0.0250 m forward displacement,
0.000435 m maximum backward excursion, 0.859 degrees maximum tilt, and 0.475
degrees late tilt RMS. Neither run reached the 0.35 m root-height threshold.

The same default-start metric at the earlier 115,204,096-step snapshot was
0.008656 rad/step target-increment RMS and 2.440 degrees late tilt RMS. The
latest checkpoint is therefore substantially calmer; this result is consistent
with continued training and does not isolate a gradient-path effect.

| Test | Result | Video |
| --- | --- | --- |
| Hold dance1_subject2 frame 50, dataset pose, 15 s | Flat stays upright; forward displacement +0.083 m, no backward excursion | `hold50_15s.mp4` |
| Follow dance1_subject2 from frame 45 (0.9 s), 20 s | Flat stays upright; mean joint L2 error 0.947 rad; tracking is not exact | `dance45_20s.mp4` |
| Hold frame 50 from nominal default joints, yaw-only initial orientation, zero velocity, 20 s | Upright but persistent lower-body oscillation | `same_state_egl/default50_baseline_vs_true_velocity.mp4` |

The default start uses the reference root height without extra height correction.
It is a specified counterfactual start, not every possible real deployment start.
The comparison video renders trajectories from `default50/rollouts.npz`:
left is unchanged baseline, right replaces only the velocity estimate with
the receiver's own simulated true yaw-frame velocity. Live depth remains enabled.
All three videos passed complete FFmpeg decoding; 12 s frames were inspected.

The old checkpoint's flat hold test drifted backward 0.855 m in 10 s.
This new snapshot did not reproduce that failure from the dataset pose.
Continuous reference motion also translates: its root XY displacement is
[-0.380, -0.287] m over this interval, versus [-0.870, -1.015] m for the flat
policy. Movement during the dance is not all unintended standing drift.

## Matched closed-loop interventions

All weights, target context, and physics remain fixed. Interventions are
diagnostic counterfactuals, not proposed deployment settings. Stair depth donors
evolve on their own trajectories; those swaps are not same-state comparisons.

Late-half target-increment RMS is the RMS change in the 24 effective joint
targets between adjacent 50 Hz control steps, in rad/step, not rad/s.

| Default-start frame 50, 20 s (`default50_controls/report.json`) | Late target-increment RMS, rad/step | Late tilt RMS, degrees |
| --- | ---: | ---: |
| Baseline | 0.008656 | 2.440 |
| Depth delay zero | 0.007673 | 2.305 |
| Prefill proprioceptive history with initial state | 0.009116 | 2.656 |
| Freeze initial depth image history | 0.003977 | 1.617 |
| Swap direct actor depth latent only | 0.000585 | 1.035 |
| Swap estimator depth latent only | 0.001011 | 1.151 |
| True simulated velocity in place of estimate | 0.000636 | 0.867 |
| Zero velocity estimate | 0.000785 | 1.080 |
| Freeze initial velocity estimate | 0.003785 | 2.641 |
| Zero velocity command | 0.000562 | 0.874 |
| Zero depth pixels | 0.000552 | 3.594 |

The earlier ten-mode default test (`default50`) gives baseline 0.008361 and
true-velocity 0.000650 rad/step, a 92.2% reduction. The expanded thirteen-mode
repeat gives a 92.6% reduction. Batch-size floating-point differences lead to
some trajectory differences, so individual numbers should not be treated as
bitwise reproducible; the large intervention effect reproduces.

The dataset-pose hold has baseline RMS 0.000765 rad/step (`ablation50`, 10 s),
roughly an order of magnitude smaller than default-start baseline. Default
start can enter an oscillatory regime even with the same target and weights.
Filling initial proprioception history and removing depth delay do not resolve it.

In continuous dance (`follow45`, 20 s), zero-depth late joint error RMS increases
from 0.2646 to 0.3045 rad, about 15%. Zero depth also increases the backward
excursion and maximum tilt. Static stability with zero depth is not evidence
that removing depth is a valid general fix. Zero normalized pixels are an
out-of-distribution image, not equivalent to removing the sensor branch.

## Why depth changes control

The network has two active routes from the depth encoder: direct policy input
and velocity-estimator input. The estimate is fed back into the actor. Its
`detach()` blocks training gradients along that connection, not its influence
on actions during physical rollout.

With a base-following camera, robot roll/pitch and robot occlusion change the
image even on flat ground. The relevant physical loop is:

```text
robot motion -> depth -> direct actor / estimated velocity -> targets -> robot motion
```

Replacing only estimated velocity by true velocity leaves the direct visual
route live and reduces default-start oscillation by over 92%. Zeroing the
estimate gives a similar reduction. This identifies the estimated-velocity
feedback as an important mediator, not just a correlation between image and
jitter. It does not prove the estimator is the sole cause: changing the direct
visual route also stabilizes this example, and target commands interact with it.
Freezing an inaccurate initial estimate only partially helps and produces
0.234 m forward displacement; it is not a fix either.

The baseline default-start estimate has a persistent lateral bias of about
0.035 m/s relative to the simulated velocity. Left ankle pitch is the largest
target-increment contributor; its oscillation is approximately 1.8 Hz.
Holding a motion frame does not make all command/context velocities zero.
Zero-command testing leaves the motion context unchanged and therefore is not
a completely consistent stationary reference task.

The same-state probe (`same_state_egl/report.json`) keeps proprioception and
motion context fixed, changing only depth. At frame 50, flat versus stair depth
has pixel MAE 1.562 uint8 levels, but the direct route changes joint targets by
0.01763 rad RMS. The estimator-only route changes them by 0.00483 rad RMS and
changes estimated XY velocity by [-0.0479, +0.0404] m/s. Both paths therefore
have measurable causal sensitivity. Images here repeat a static pose across
depth history; this probe does not model moving temporal history.

Camera offset pitch perturbations of +/-2 degrees also change actions, but
are sensitivity tests, not evidence that the current camera extrinsics are wrong.
Full roll/pitch/yaw following remains the intended physical camera behavior.

## Training evidence and remaining uncertainty

`training_audit.json` summarizes the last 30 scalar samples no later than this
checkpoint. Local depth-gradient audit median estimator/actor norm ratio is
0.001470 (0.147%). This does not support the hypothesis that estimator auxiliary
gradients dominate the shared encoder in this run. These are rank-local raw
gradient diagnostics, not a reconstruction of Adam updates or a proof that
multi-task gradient conflict is absent.

Training velocity MAE medians are approximately [0.117, 0.110, 0.086] m/s.
`false_motion_when_stopped` is a fraction: median 9.36% of samples with true XY
speed below 0.1 m/s predict speed above 0.2 m/s. It is not an error in m/s.
Aggregate motion evaluation still reports joint L2 1.203 rad, velocity L2
9.707 rad/s, and acceleration L2 281.7 rad/s^2. These aggregate metrics do not
isolate flat-ground startup behavior.

This checkpoint's replay is version 2 without terminal observation snapshots,
and the live job began before the reset-boundary collector fix. See
`PUFO_RESET_BOUNDARY_FIX_20261012.md`. This is a separate known training-data
risk, not a demonstrated explanation of this particular depth oscillation.
The live process was not restarted; disk fixes do not update already loaded code.

The closed-loop tests here run MuJoCo, not the training Newton physics engine.
The stair model is the existing static H200 5 m-grid export, not this run's
10 m-grid training terrain. Global root-height thresholds are inappropriate on
negative-height stairs and must not be used to claim a stair fall. Old camera
pixel-parity checks do not substitute for a new saved-config Newton parity test.

## Next experiments

1. Reproduce the default-start and dataset-start hold in the exact saved Newton
   environment, retaining live depth and comparing estimated versus true velocity.
   Repeat several held frames and nominal starts, logging per-axis bias, image
   change, target change, tilt, and foot contacts.
2. Evaluate a fully stationary reference with consistent zero velocity command
   and context, distinct from holding a moving dance frame. Test low-speed
   estimator calibration rather than blindly disabling vision.
3. Before a controlled continuation-training comparison, start a new process
   with the reset-boundary fix and compatible replay handling. Restarting the
   current job requires a separate user decision.
4. Use the resulting evidence to compare low-speed estimator supervision or
   separating estimator visual features from direct control features. Retain
   dance and stair tracking as acceptance tests; static jitter alone is insufficient.

## Reproduction

Seven metric regression tests pass, covering heading-relative drift, yaw-invariant
tilt, late-window reference alignment, target-increment units, and flat-height
threshold timing. At frame 50 the split diagnostic inference equals native
`Policy.act` exactly; the 13-row batch differs by at most 3.28e-7 actor units.
CPU-only runs print a harmless Warp no-CUDA warning because CUDA is intentionally
hidden; all reported runs and tests completed successfully.

Use an unused output directory for every invocation; scripts reject existing
directories instead of overwriting results.

```bash
source project_ops/activate_5090d6.sh
PUFO_DIAG=/data/zym/roboparty_train/migration/pufo_depth_diagnosis_20261013
PUFO_CP="$PUFO_DIAG/slim512x6_115204096/model_115204096.pt"
CUDA_VISIBLE_DEVICES='' python project_ops/scripts/diagnose_pufo_depth_stability.py \
  --checkpoint "$PUFO_CP" --frame 50 --duration 20 --start-default \
  --output "$PUFO_DIAG/default50_new_repeat"
CUDA_VISIBLE_DEVICES=7 MUJOCO_GL=egl python project_ops/scripts/probe_pufo_depth_inputs.py \
  --checkpoint "$PUFO_CP" --output "$PUFO_DIAG/same_state_new_repeat" \
  --rollouts "$PUFO_DIAG/default50_new_repeat/rollouts.npz"
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s project_ops/scripts \
  -p test_diagnose_pufo_depth_stability.py -v
```
