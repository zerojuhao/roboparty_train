# 5090D-6 shared-gradient PUFO continuation

## Decision

On 2026-10-13 the user chose to continue the original training design instead
of running the stop-gradient A/B experiment. The velocity estimator still reads
the shared depth latent and its supervised loss still updates the depth encoder.
The original SLIM512x6 networks, optimizer settings, depth processing, rewards,
and one-billion global environment-step budget are preserved.

The same 20-second default-start, frame-50 test improved between the original
115,204,096 and 124,805,120-step snapshots without stopping encoder gradients:

| Metric | 115,204,096 steps | 124,805,120 steps |
| --- | ---: | ---: |
| Late joint-target increment RMS, rad/control step | 0.008656 | 0.000617 |
| Late body tilt RMS, degrees | 2.440 | 1.676 |

Target-increment RMS decreased approximately 92.9%. This is evidence against
assuming shared backpropagation is inherently unstable, not proof that every
motion or startup state has converged. The old and new tests both use thirteen
matched counterfactual rollouts, 50 Hz control, and the same default-start setup.

## Active Training

Host: `zym@10.0.10.132:22` (5090D-6).

```text
tmux: rp1-pufo-sharedgrad-resume-5090d6
torchrun PID at startup: 2113179
task: RP1-PUFO-SLIM512
layout: 8 GPUs, 1024 environments per GPU
run directory:
/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo_slim512/
2026-10-13_03-28-57_pufo_slim512x6_sharedgrad_resume124805120_5090d6
console log:
/data/zym/roboparty_train/migration/pufo_sharedgrad_resume_20261013/train.log
W&B: https://wandb.ai/dtu-yiming/rp1_pufo/runs/cez5d06s
```

Launch script: `project_ops/start_5090d6_pufo_sharedgrad_resume.sh`.
It resumes the exact frozen `model_124805120.pt`, not a moving latest-checkpoint
selection. Do not run the script again while the current tmux task is active.

Frozen source:

```text
/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo_slim512/
2026-10-13_velocity_ab_initial/model_124805120.pt
SHA256: e735eb40e707dd93461e603bd4e4bbf3d6c25459c930a3d82234dd405989b8fa
```

The directory name predates the decision to cancel A/B; the checkpoint is an
unchanged copy of the original run's snapshot. Model weights, observation
normalizer, optimizer states, and training counters were restored. The frozen
copy has no replay shards, so both the resumed policy and fresh simulator
collect a new replay window before optimizer updates. Warmup completed at
global step 124,895,232; normal update phases subsequently advanced beyond it.

The stop-gradient attempt failed during configuration parsing, before loading
training weights or doing any optimizer updates. Its logs and `experiment.json`
remain under `migration/pufo_velocity_stopgrad_20261013`. There is no active A/B
process or queued A group. The unused A/B launcher's camera path was corrected
from `scene.depth_camera` to `scene.camera`; it was not restarted.

## Verified Configuration

Comparing saved original configuration against the resumed run showed only:

- Agent resume metadata and run name, plus the newly explicit
  `velocity_estimator_updates_encoder: true` (the old implicit behavior).
- `compute_final_obs: true` and the extra `transition_kinematics` observation
  used by the reset-boundary replay fix, not by the actor's observation input.
- Camera alignment label `yaw` -> `base`; the original implementation already
  followed full base roll/pitch/yaw, so this makes the label reflect its behavior.
- The output log directory.

Actor/critic/encoder shapes, learning rates, losses, camera processing and noise,
physics parameters, commands, actions, events and reward settings are unchanged.
The remote checkpoint contract remains `pufo_h200_alignment_v5`. The local
workspace's separate v6 configuration was not copied over the remote v5 config.

The new process runs the previously deployed reset-boundary collector fix.
Legacy terminal states are not recoverable, and the old replay was not reused.
Consequently later improvements cannot be attributed solely to extra training
steps; corrected replay collection is also a change relative to the old process.

## Velocity Accuracy Baseline

The resumed checkpoint completed evaluation of all 902 motions before updating:

| Axis | MAE, m/s | RMSE, m/s | Signed bias, m/s |
| --- | ---: | ---: | ---: |
| X | 0.097986 | 0.149801 | 0.004843 |
| Y | 0.090320 | 0.128675 | 0.024454 |
| Z | 0.053148 | 0.076083 | 0.006101 |

These are equal-weight averages of per-motion metrics, not a pooled-frame
statistic. They compare the estimate with actual robot yaw-frame velocity at
the same pre-action state, not with the requested reference velocity.
Evaluation joint L2 was 1.187928 rad. This initial evaluation is a baseline,
not a result of the continuation's updates.

TensorBoard/W&B records:

- Training: `train/velocity_estimator/mae_{x,y,z}`,
  `rmse_{x,y,z}`, `bias_{x,y,z}`, `stopped_mae_{x,y,z}`,
  `stopped_fraction`, and `false_motion_when_stopped`.
- Tracking evaluation: `eval/velocity_estimator_mae_{x,y,z}_m_s`,
  `velocity_estimator_rmse_{x,y,z}_m_s`, and
  `velocity_estimator_bias_{x,y,z}_m_s`.
- Shared encoder supervision: `train/depth_gradient_local/estimator_actor_ratio`.

Training low-speed metrics use true horizontal speed below 0.1 m/s; the false
motion threshold is predicted horizontal speed above 0.2 m/s. Interpret the
stopped MAE alongside stopped fraction: training batches without stopped samples
have a masked value of zero, not evidence of perfect stationary accuracy.

## Fixed-Input Comparison

Two frozen MuJoCo observation corpora, 100 samples each at 0.2-second intervals,
are stored on 5090D-6:

```text
/data/zym/roboparty_train/migration/pufo_velocity_stopgrad_20261013/corpora/default50.npz
/data/zym/roboparty_train/migration/pufo_velocity_stopgrad_20261013/corpora/follow45.npz
```

The former is nominal default startup holding frame 50; the latter is a 20-second
dance from frame 45. Holding a frame retains its motion command/context and is
not a completely stationary reference. These are two diagnostic conditions,
not coverage of the entire motion dataset or Newton training distribution.

Run `project_ops/scripts/compare_pufo_velocity.py` on the frozen initial model
and a later checkpoint, with both corpora, to compare identical raw inputs and
true velocities. Each checkpoint uses its own saved observation normalizer.
This controls for the fact that a changed policy visits different states during
closed-loop evaluation. Use a new output JSON path; the tool refuses overwrite.
Then rerun standing and dancing rollouts to evaluate physical stability as well.

The initial model's combined corpus MAE is [0.02522, 0.01855, 0.02574] m/s;
RMSE is [0.03257, 0.03050, 0.04013] m/s. Its lower error than the full-motion
evaluation reflects the much narrower, different state distribution.

## Validation

Local regression run: 71 tests passed, 22 subtests passed, one opt-in CUDA test
skipped. Remote gradient regressions: 10 tests passed using the existing isolated
test dependencies, without installing packages into the training environment.
The active run passed initial 902-motion evaluation, replay warmup, and real
optimizer updates with finite reported metrics. The saved configuration and
logged encoder-gradient ratio were independently checked after startup.

All original model snapshots and logs were retained. Original remote source
files before the diagnostic edits are archived at:

```text
/data/zym/roboparty_train/migration/pufo_velocity_stopgrad_20261013/source_before_stopgrad.tar
```
