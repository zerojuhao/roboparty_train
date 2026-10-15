# PUFO Reset-Boundary Replay Fix

The collector now separates the pre-reset action outcome from the reset
observation used by the next policy decision. Isaac Lab snapshots `final_obs`
before reset and exposes `_final_obs` to identify its valid environments. The
snapshot is cloned so reset cannot mutate camera/action buffers underneath it.
PUFO also exposes the actual world heading and root velocity for computing
action-outcome metadata at the boundary.

Replay keeps ordinary next observations in the following record and stores
full terminal observations only for reset environments. Terminal histories are
built before the collector clears proprioceptive history. Sampling uses
action-aligned outcome flags and includes the final valid start of a window.
Episodes of one action, asynchronous resets, terminated and truncated outcomes,
ring overwrite and sequence sampling are covered by regression tests.

Replay snapshots use version 3 and include terminal observations. Legacy
snapshots cannot recover their lost terminal states; resuming their models
retains weights, optimizer and counters but collects fresh replay during warmup.
The network architecture and `pufo_h200_alignment_v5` model contract did not change.

## Validation

An isolated local Newton/Warp PUFO run used 8 environments, 0.06-second episodes,
production networks, BF16, CUDA Graph physics, batch size 8 and one minibatch per
update. It completed 640 environment steps and 77 optimizer updates without
changing any existing training job. Compile was disabled and tracking evaluation
was disabled to keep the test bounded. This is a pipeline test, not a model-quality
or same-configuration convergence comparison.

Artifacts:
`/data/zym/roboparty_train/migration/reset_boundary_fix_20261012/rp1_pufo/2026-10-12_08-44-58_reset_boundary_smoke/`

The `model_512.pt` snapshot and referenced rank-local replay passed an independent
audit: 170 terminal outcomes survived serialization; 1,361 of 4,096 sampled
transitions were terminal outcomes and matched every saved terminal observation
field. Terminal last actions equaled the executed actions, and proprioceptive
history retained the preceding observation's action. The next reset observations
had cleared last actions and different robot states. Model and optimizer tensors
were finite.

The last-pair regression was also run against the original sampler and failed
because it omitted the final valid pair; it passes after the fix.

## 5090D-6 Status

The public endpoint `zym@111.0.22.33:51132` timed out, but SSH to
`zym@10.0.10.132:22` succeeded using the same key. Its host key matched the
existing known-hosts entry. The scoped RoboLab/RSL-RL changes, regression tests,
documentation and Isaac Lab environment fix were synced and checksum-verified.
The remote camera alignment (`yaw`) was preserved, as were the custom SLIM512
configuration and task registrations.

Original target files were backed up before syncing to:
`/data/zym/roboparty_train/migration/reset_boundary_fix_20261012_0915/source_before_sync.tar`

Remote CPU regressions passed: 72 tests and 17 subtests, with one opt-in eight-GPU
NCCL test skipped. Test dependencies were installed in an isolated directory,
not into the training environment. Existing training GPU processes were untouched.

The running SLIM512 job's `model_28803072.pt` checkpoint was also tested read-only
on CPU with the patched agent. It loaded the actor, backward map and observation
normalizer without architecture changes, and produced finite 4x24 actions on
synthetic observations. All 610 model tensors and 1,050 optimizer tensors were
finite. This checks checkpoint/inference compatibility, not rollout quality or
convergence. Its rank-local replay metadata was version 2, with no saved terminal
outcomes, confirming that the running job still uses the legacy collector.

The existing eight-GPU training launcher (PID 3232030) was not restarted. An
already running Python process continues using its loaded code. Resume from its
model checkpoint with the saved training configuration and the existing local
RSL-RL `PYTHONPATH`; the patched collector will discard legacy replay and warm up
with the loaded policy while retaining weights, optimizer and step counters.
