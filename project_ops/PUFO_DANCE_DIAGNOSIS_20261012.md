# PUFO dance1_subject2 drift and jitter diagnosis

Checkpoint evaluated before its external deletion:
`model_393601024.pt`, run
`2026-10-10_11-36-11_pufo_mass0812_pd0713_angvel025_notrim_resume_from336_20261010`.

## What the interventions establish

The dominant causal input path for the observed sim2sim flat-ground drift is
depth -> direct actor fusion -> joint actions. It is not primarily the
depth-assisted velocity estimator. This identifies the behavioral mechanism,
not which training objective or dataset imbalance originally produced it.

Identical checkpoint, held frame 50, 10-second flat-ground rollouts:

| Intervention | Forward displacement | Target-increment RMS at 5-10 s |
| --- | ---: | ---: |
| Original depth | -0.855 m | 0.01734 rad |
| Stair depth in direct actor branch only | +0.002 m | 0.00076 rad |
| Stair depth in velocity-estimator branch only | -0.954 m | 0.01689 rad |
| Stair depth in both branches | +0.017 m | 0.00076 rad |

These swaps are diagnostic counterfactuals, not deployable fixes. The donor
stair camera and the receiver flat robot can follow different trajectories.
Stair-depth replacement does not establish correct dance tracking.

Additional 15-second controls did not eliminate flat-ground drift:
zero velocity command (-1.252 m), fully zeroed frozen-reference velocities
(-1.351 m), prefilled proprioceptive history (-1.269 m), true instead of
estimated velocity (-1.151 m), and action smoothing alpha=0.33 (-0.982 m).
An actual flat tile on the exported terrain map behaved like the plane, and
changing ground friction from 0.9 to 1.0 did not resolve the 10-second drift.
These controls exclude those changes as sufficient remedies, not all possible
physical-model differences.

## Actual training camera versus sim2sim camera

One Newton/Warp evaluation environment used the robot USD cached at this run's
startup, with camera mounting randomization and observation noise disabled.
Both sensors were queried at the same pose. All robot-body position differences
were below 2.8e-7 m. Training cropping, clipping, blur, normalization and image
quantization were compared, including dynamic robot mesh occlusion.

| Dataset frame | Depth MAE (uint8) | Maximum pixel error | Action-target change RMS |
| --- | ---: | ---: | ---: |
| 45 | 0.00309 | 1 | 0.000200 rad |
| 50 | 0.00463 | 1 | 0.000087 rad |
| 100 | 0.00617 | 1 | 0.000139 rad |
| 200 | 0.00309 | 1 | 0.000147 rad |

This strongly excludes a starting-pose camera projection/preprocessing/mesh
bug as the source of the observed large drift. It does not cover every pose
later visited by an unstable policy. The camera's serialized `yaw` alignment
is misleading: camera initialization uses full `base` rotation, consistent
with sim2sim. The exporter uses fixed one-frame depth delay, whereas this run
trained with integer delays 0 or 1; a dedicated delay-zero rollout remains
untested. Static repeated-image parity does not test temporal sampling.

Raw depth, processed images, ray geometry and same-state actions are retained
in `/data/zym/roboparty_train/migration/pufo_dance_diagnosis_20261012/`.
Reproduction tool: `project_ops/scripts/diagnose_pufo_depth_parity.py`.
It needs the checkpoint and adjacent saved `params/env.yaml`, `params/agent.yaml`.
The current test environment uses the repository's evaluation config; it is
not a complete reconstruction of every saved training physics parameter.

## Why the first moments look unsettled

At frame 50 there are initially no MuJoCo contacts. Left-foot collision geometry
is 2.6-4.5 mm above the plane; right-foot geometry is 7.5-8.8 mm above it.
The initial actor target differs from the dataset joints by L2=0.902 rad.
Largest target offsets are right shoulder yaw (-0.423 rad), right wrist
(+0.384 rad), right shoulder roll (+0.379 rad), left shoulder roll (-0.340 rad)
and left wrist (-0.320 rad). Thus a nearly nominal waist does not imply a
supported or policy-equilibrium whole-body pose. Startup landing and abrupt
upper-body target changes plausibly contribute to the early disturbance;
their individual contributions have not been isolated by a closed-loop control.

The initial reference also has nonzero root velocity, but zero-command and
zero-reference-velocity controls show that this is not sufficient to explain
sustained drift. A root staying above 0.35 m establishes only no low-root fall,
not stability, low jitter or correct tracking.

## Remaining limits

The user confirmed the original PUFO logs/checkpoints were deleted during this
investigation. Camera parity completed because that process had already loaded
the weights. No model weights, training configurations or running training jobs
were modified by this diagnosis. A same-checkpoint training-physics closed-loop
rollout and training gradient/reward balance audit can no longer be completed
without another copy of that checkpoint. The surviving deployment ONNX export
belongs to an older model (297607168) and must not be substituted as the latest.

Reward imbalance, poor flat-ground generalization, shared-encoder gradient
conflict and physics transfer mismatch remain candidate upstream causes, not
proven conclusions. The architecture has a direct visual fusion branch;
`VisualResidualActor` is not a separately gated additive visual correction to a
stable proprioceptive controller. Simply deleting this branch from an already
trained network is not a validated fix.
