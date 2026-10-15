# PUFO Ground-Ray Failure Investigation

## 2026-10-07 Follow-up: Previous Guard Missed a Third Captured Sliver

The formal eight-GPU run stopped after its last 18,481,152-step log with
`PUFO root-ground ray missed terrain`. Its failing physical state was not
saved, so that exact historical event cannot be replayed. The other seven
ranks then timed out in the next-observation finite-check collective.

The follow-up reproduced a remaining upstream NaN with the OLD project
collision guard enabled: seed 45, 1024 worlds, 0.6-second episodes, random
actions in [-0.5, 0.5], CUDA Graph enabled, nconmax=512, njmax=4096. At step
2388, world 673 (terrain level 0, type 3, episode age 17) first failed at
`cholesky:search`. Inputs were finite and overflow flags were zero. The
stage probe stopped before the downstream ray query.

Exact-input replay with that old guard reproduced NaN. CPU MuJoCo remained
finite without warnings. GPU contact pair 2424/751 had a position error of
826.93 m relative to CPU. Replacing ONLY this contact position with the CPU
position made GPU integration finite (max absolute acceleration 790.21,
versus CPU 790.35). This is an offline causal control, not a training fallback.

The missed triangle had normalized squared area just above the old 1e-12
threshold. Stock affine weights were approximately
(-5.04841e6, 5.05613e6, -7724.53), summing to 0.466797 instead of one.
The old guard checked near-collinearity and non-finiteness, but not this
finite catastrophic cancellation.

The first strengthened affine guard used a float32-relative 64-ULP edge
tolerance (squared for the normalized area comparison), and rejected affine
results whose partition-of-unity error exceeded 1e-3. It made the exact-input
replay finite (max absolute acceleration 779.27), and passed four 10,000-step
profiles. However, closest-edge recovery alone still displaced the seed-45
contact by 0.263 m and produced a substantially different normal. That was
not accepted as the final repair.

The final `mjwarp_contact_guard.py` also intercepts EPA face construction.
Ill-conditioned faces are marked INVALID, not DELETED: their original plane
is retained for visibility/horizon construction, but EPA selects another
valid face for the witness. A failed/zero-distance original attach returns
without accessing unwritten or out-of-capacity face storage. The bounded
affine guard remains a last resort. Ordinary faces and signed outside-triangle
coordinates retain the upstream calculation.

All three captured bad contact pairs remain present. The final face-selection
repair gives CPU/GPU position differences of 0.0068 mm (seed 43),
0.0153 mm (seed 44), and 0.0251 mm (seed 45); the normals are also close.
Maximum absolute GPU acceleration is 476.79 / 655.50 / 790.18 respectively,
compared to CPU 477.55 / 652.84 / 790.35. This does not replace observations,
substitute CPU results during training, discard these contacts, change solver
type, disable CUDA Graph, or edit installed libraries.

The runner now propagates all ordinary environment-step exceptions through
the same collective as successful ranks, and saves available ground-ray
failure details with the before/after snapshot. The training entrypoint also
preserves a nonzero exit code when Kit closes after an exception.

Diagnostics gained `--profile_policy_checkpoint` to exercise trained actors
and expert contexts without optimizer updates, and replay gained
`--require-finite` for machine-checkable regressions. High-terrain tests must
set `env.curriculum.terrain_levels.params.initial_max_level=9` as well as
`env.scene.terrain.max_init_terrain_level=9`: the curriculum otherwise
resamples levels 0-2 at reset. Each profile prints the actual level range.

Evidence under `/data/zym/roboparty_train/migration/`:

- `ground_recheck_seed45.pt`, `ground_recheck_seed45.log`: new captured input.
- `ground_recheck_seed45_replay.log`: old guard still fails.
- `ground_recheck_seed45_trace.log`: failing affine coordinates.
- `ground_recheck_seed45_cpu_position.log`: position-only causal control.
- `ground_recheck_seed45_fixed.log`: interim affine-only guard passes finiteness,
  but its remaining geometric error is why face selection was also repaired.
- `ground_fix_replay_seed43.log`, `ground_fix_replay_seed44.log`: interim replays.
- `ground_fix_faces43.log`, `ground_fix_faces44.log`, `ground_fix_faces45.log`:
  face-selection experiment and matching `*_matrices.pt` geometric evidence.
- `ground_fix_eight_gpu_tests.log`: real eight-GPU NCCL exception propagation.
- `ground_final_replay_seed43.log`, `ground_final_replay_seed44.log`,
  `ground_final_replay_seed45.log`: final project guard, identical captured
  inputs, `--require-finite`, all exit zero.
- `ground_final_seed45_matrices.pt`: final contact-position comparison.
- `ground_final_unit_tests.log`: 106 environment/physics tests pass. Six
  contact-guard tests also pass on CUDA. Runner tests pass all 29 cases,
  including the opt-in real eight-GPU NCCL failure-propagation test.
- `ground_final_8gpu_training_smoke.log`: final eight-GPU, 1024 environments
  per GPU, 32 rollout iterations, 262,144 total environment transitions;
  exit zero. After warmup, 21 update phases (16 minibatches each) and three
  evaluations at steps 0 / 131,072 / 262,144 complete. All 75 TensorBoard
  scalar tags contain finite values. This short randomly initialized run
  validates execution, not tracking quality or convergence.

The eight-GPU smoke used eight evaluation motions and eight evaluation
environments to bound runtime; it does not establish full-dataset evaluation
performance. It used TensorBoard, not the formal WandB run. Formal training
was not restarted and the old checkpoint/replay were not removed.

### Final-Code Stress Results

All four profiles below completed 10,000 steps with 1024 environments each
(40,960,000 environment transitions in total), printed `[CAPACITY] PASS`,
and exited zero. All retained Newton, CUDA Graph, nconmax=512 and njmax=4096.
The probe verified 39 active stage fields inside the physics graph, checked
capacity on every physics tick, and checked observations/rewards for finite
values. There were no NaNs, root-ground misses or capacity overflows. Normal
startup/importer warnings are not counted as these runtime failures.

| Log basename | Actions | Episode seconds | Actual initial levels | Peak contacts | Peak constraints/world | Peak solver iterations |
| --- | --- | --- | --- | --- | --- | --- |
| `ground_final_seed45.log` | Uniform [-0.5, 0.5] | 0.6 | 0-2 | 16211 | 244 | 10 |
| `ground_final_hard48.log` | Uniform [-0.5, 0.5] | 0.6 | 0-9 | 15074 | 212 | 11 |
| `ground_final_hard49.log` | Uniform [-0.5, 0.5] | 10 | 0-9 | 14903 | 204 | 10 |
| `ground_final_policy50.log` | Stochastic checkpoint actor + expert contexts | 10 | 0-9 | 36711 | 291 | 11 |

The policy profile loaded the preserved `model_9601024.pt` without optimizer
or replay, and did not update its parameters. The eight-GPU smoke above
separately exercised replay, parameter updates and sharded evaluation.
All bounded training/profile processes exited. These results validate the
captured failure cases and this stress scope, not indefinite training or the
exact historical formal-run state that was never saved.

The sections below describe earlier investigations chronologically. Bounded
stress-test success is not proof that every possible numerical failure is
eliminated.

## Current Conclusion: Captured Contact-Generation Failure

The follow-up investigation captured TWO full real failure snapshots and
reproduced both outside IsaacLab, without a learner, policy, replay buffer,
reset manager or CUDA Graph. The failure is now localized to erroneous GPU
convex-contact witnesses, followed by a float32 Cholesky failure. Earlier
unresolved statements below describe the chronological investigation, not
the current conclusion.

The observed chain is:

```text
near-degenerate GJK/EPA face and unstable affine witness construction
  -> finite but physically impossible cylinder/box contact position
  -> excessively large contact lever arm and constraint Jacobian
  -> ill-conditioned contact Hessian
  -> non-finite Cholesky search / acceleration
  -> non-finite integrated root state
  -> invalid ray origin and missing ground hit
```

Both captures used 1024 environments, terrain levels 0-9, finite random
actions in [-0.5, 0.5], 0.6-second episodes, CUDA Graph enabled, nconmax=512,
njmax=4096 and EPA horizon=64. Neither had a detected capacity overflow.
Installed versions: MuJoCo 3.12.0, MuJoCo Warp 3.12.0, Warp 1.17.0,
Newton 1.6.0. No installed library was edited.

| Capture | Seed 43 | Seed 44 |
| --- | --- | --- |
| Environment step | 4006 | 2358 |
| World | 490 | 578 |
| Terrain level/type index | 2 / 2 | 2 / 3 |
| Episode age | 15 steps (0.30 s) | 17 steps (0.34 s) |
| First instrumented invalid field | solve:qacc | cholesky:search |
| Bad contact geom pair | 2462 / 264 | 2418 / 872 |
| Collision shapes | right foot cylinder / terrain box | left foot cylinder / terrain box |
| Contact position error vs CPU | 466.5233 m | 505.1137 m |
| Reported penetration depth | 0.01264 mm | 0.03376 mm |
| CPU max absolute qacc, no warnings | 477.5455 | 652.8423 |

The first capture predates the internal Cholesky probe. Its standalone replay
identifies that failure inside Cholesky as well. The second capture freezes
the first failed Cholesky context inside the physics graph, before subsequent
iterations reuse the scratch arrays. Its saved Hessian is padded to 32x32;
the active system is 30x30.

For seed 43, the GPU witness was (-482.85095, -98.61082, 18.03124), whereas
CPU MuJoCo returned (-25.09038, -10.07246, 1.90059). For seed 44, the GPU
witness was (-528.62750, 3.47057, -34.04543), versus the CPU witness
(-24.56349, -0.14313, -1.69893). These are finite values, so a finite-only
contact probe does not identify them as the first invalid stage.

### Causal Controls on the Exact Saved Inputs

| Replay change | Seed 43 | Seed 44 |
| --- | --- | --- |
| Stock GPU physics | NaN | NaN |
| CPU MuJoCo | finite, zero warnings | finite, zero warnings |
| Replace only the one bad contact position with CPU position | finite, max qacc 502.4457 | finite, max qacc 658.3912 |
| Replace that position and contact frame with CPU values | finite, max qacc 477.4639 | finite, max qacc 652.8406 |
| Clear warm-start | still NaN | not tested |
| Restore EPA horizon from 64 to 512 | still NaN | not tested |
| Use CG instead of Newton | finite, but wrong witness remains | not tested |
| Compute affine coordinates in float64, return float32 weights | still NaN, larger Hessian | not tested |

The position-only intervention preserves the GPU contact frame, depth,
friction, input state and original Newton solver. Only one contact position
is replaced and constraints are rebuilt. This isolates the bad position as
a causal trigger; the closer CPU agreement when also correcting the frame
shows that the normal/frame may be inaccurate too. CPU witness substitution
is an offline diagnostic, NOT a production fallback or proposed training path.

With stock seed-43 contacts, the Hessian maximum eigenvalue reached about
3.79 million; a later stored Hessian had a minimum eigenvalue of -0.01731.
Correcting the bad witness and frame reduced the maximum to about 180-279
and preserved a positive minimum near 0.01026. Seed 44 also failed float32
Cholesky with a very ill-conditioned matrix; a float64 eigenvalue calculation
on a stored matrix can remain positive despite the float32 factorization
failure. Negative eigenvalues are not a necessary condition for this failure.

### Localized Collision Arithmetic

In the installed `mujoco_warp/_src/collision_gjk.py`, `_attach_face` projects
the origin onto a face plane. `_epa_witness` calls `_tri_affine_coord` on the
selected face and uses those weights directly on world-space support points.
Tracing the unmodified calculation in the seed-43 replay gives:

```text
v1 = ( 0.209621429, -0.372461319, -1.26361847e-05)
v2 = ( 0.209623337, -0.372461319, -1.26361847e-05)
v3 = (-0.390407562, -0.372463226, -1.26361847e-05)
projected point = (0, 0, -1.26361847e-05)
weights = (1.3805e+08, -1.38049e+08, -440.816)
float32 weight sum = 7.18427
```

The triangle is nearly collinear and the projection is outside it. Enormous
cancelling weights make witness reconstruction unstable. `_epa_witness`
returns penetration depth from the face-plane distance independently of
these witness coordinates, explaining why tiny reported depth coexists with
a contact hundreds of meters away. Merely making the affine division double
precision did not repair the invalid face/witness geometry in the experiment.

The repair target is robust contact generation: handle degenerate EPA faces
and invalid witnesses, and validate a geometrically correct alternative.
Do not treat CG, NaN replacement, increased scanner height, or silently
discarding such contacts as a validated fix.

The repository now contains this repair in
`robo_newton/robolab/utils/mjwarp_contact_guard.py`. The PUFO-specific
`NewtonPUFOMJWarpManager` installs it before MuJoCo Warp kernels are built and
restores the upstream function when the solver is cleared. The guard changes
only near-collinear triangle handling; normal EPA triangles retain the stock
affine calculation. It is process-local and does not modify site-packages.
The exact saved seed-43 and seed-44 replays pass with this project guard.
Long multi-seed rollout validation is still required before claiming that
all possible contact failures are eliminated.

Validation after the repository change:

- Project-guard replay of `nan_hunt_seed43.pt`: finite solve and integration,
  `qacc` max-abs 480.63.
- Project-guard replay of `nan_hunt_seed44_v2.pt`: finite solve and integration,
  `qacc` max-abs 711.78.
- Real `RP1-PUFO` smoke rollout: 16 environments, 120 random-action steps,
  all rewards and observations finite.
- Guarded Newton stress rollout: 1024 environments, 4500 random-action steps
  at the original `nconmax=512`/`njmax=4096` capacities; this passed the
  previous seed-43 failure location around step 4006 and finished with all
  instrumented stages finite.
- RSL-RL runner startup: 16 environments, two rollout iterations, no NaN or
  exception. The smoke command used `--skip_tracking_eval`; this flag only
  disables the expensive initial evaluation and does not change its default.
- UFO test suite: 98 tests passed.

The saved cases show that neither deep root burial, learning instability,
replay storage, reset execution at the failing instant, CUDA Graph execution,
nor reduced EPA horizon is required to reproduce these failures. This does
not prove that all historical NaNs have the same cause, or that other reset
or capacity problems can never exist. The earlier world-632 failure has no
complete solver snapshot and cannot be retrospectively proven identical.

### Reproduction and Evidence

All evidence files are under `/data/zym/roboparty_train/migration/`:

- `nan_hunt_seed43.log`, `nan_hunt_seed43.pt`
- `nan_hunt_seed44_v2.log`, `nan_hunt_seed44_v2.pt`
- `nan_hunt_seed43_matrices.pt`, `nan_hunt_seed44_matrices.pt`
- `nan_hunt_seed43_replay_affine_trace.log`
- `nan_hunt_seed43_replay_affine64.log`
- `nan_hunt_seed43_replay_cpu_position.log`, `nan_hunt_seed44_replay_cpu_position.log`
- `nan_hunt_seed43_replay_cpu_witness.log`, `nan_hunt_seed44_replay_cpu_witness.log`

For example, run the stock replay, then add `--cpu-witness --cpu-position-only`
to isolate the effect of correcting the erroneous contact position:

```bash
CUDA_VISIBLE_DEVICES=5 /data/conda/envs/isaaclab_zym/bin/python \
  /home/zym/roboparty_train/project_ops/scripts/diagnose_pufo_nan_snapshot.py \
  /data/zym/roboparty_train/migration/nan_hunt_seed43.pt
```

Only load trusted snapshots: they include serialized MuJoCo model objects.
The replay prints results and can exit successfully even when it reports
NaN; inspect `SOLVE` and `INTEGRATION`, not just the shell exit status.

The follow-up diagnostic probe additionally checks Newton/MuJoCo conversion,
reset/interval event boundaries and first-failed Cholesky context. Stateful
event wrappers preserve the original `.reset()` interface. No default physics,
training, reset, CUDA Graph, or observation-fallback behavior was changed.

Verification: all 97 UFO environment unit tests passed, including the
first-failure retention, active-world filtering, frozen Cholesky context,
graph recapture and stateful-event reset tests. The four diagnostic modules
passed Python compilation and `git diff --check` passed. All bounded
diagnostic processes completed; no persistent training was started.
Test output: `nan_hunt_final_tests.log` in the evidence directory.

An unfiltered 16-environment, 60-step Compute Sanitizer initcheck using LBVH
for diagnostic geometry construction passed with zero reported errors
(`nan_lbvh_initcheck.log`). This is a limited control, not a proof of global
memory safety. A separate run excluding cuBQL kernels generated downstream
uninitialized-read reports; excluding producers makes these reports
inconclusive and they are NOT the established cause of the captured NaNs.

## Captured Failure (2026-10-06)

The missing-root-ground exception was reproduced locally with seed 43,
1024 environments, terrain levels 0-9, random actions in [-0.5, 0.5],
0.6-second episodes, CUDA Graph enabled and the current solver capacities.
No learner or learned policy was involved.

At environment step 1996, world 632 failed on terrain type 3
(`low_stairs_down`), level 0. Its episode age was 16 steps (0.32 seconds),
and its reset flag was false. Root position, linear/angular velocity and
sensor position were already NaN. The downward ray direction was finite,
but its origin was NaN, so re-casting and refreshing the sensor could not
produce a valid ground hit. There was no detected solver-capacity overflow.

The preceding eight saved root states were finite. The final finite position
was (-45.3161354, -0.1986861, -0.8116484), with linear velocity approximately
(-0.5922, -0.9064, -0.1230) m/s. Reconstructing the staircase and re-casting at
that XY yielded ground z=-1.496383: the base was 0.684735 m above the surface.
The previous positions crossed a step-height transition from -1.603267 to
-1.496383. This is evidence against an already deeply buried base being the
direct reason for this captured miss. It does NOT rule out foot/body contact
penetration or contact-solver numerical failure as the upstream trigger.

The immediate failure chain is invalid simulation state -> invalid sensor
origin -> missing ray hit -> generic ground-observation RuntimeError.
The exception occurs before the runner's next-observation NaN check. The
current distributed exception adapter only special-cases capacity overflow,
so this RuntimeError also bypasses that coordinated failure path.

The precise operation that first produced NaN has not been captured. This
diagnostic records whole environment-step root states, not every solver
substep's joint positions, velocities, controls, contacts and accelerations.
Do not label the solver, reset, push event or capacity optimization as the
proven upstream cause without those additional measurements. The earlier
seed-42 failure had no equivalent snapshot, so it cannot be conclusively
assigned the same cause based only on the matching exception text.

## Checks

- Seed 42: 3000 frequent-reset steps passed; peak 14,760 pooled contacts,
  212 constraints per world and 11 solver iterations.
- Seed 43: failure at step 1996 as above; peak 14,919 pooled contacts,
  220 constraints per world and 10 solver iterations.
- Seed 42 with the normal 10-second episode duration: 1000 steps passed;
  peak 14,229 pooled contacts, 208 constraints per world and nine iterations.
  All three bounded simulation processes exited; no persistent training was started.
- Static scans rebuilt all 70 terrain tiles with generator seed 0 and global
  NumPy seeds 42/43. Each scan sampled 459,270 positions within +/-1 m of tile
  centers at 0.025 m spacing. CPU high-origin scans had no misses. GPU scans
  from a high origin and 2.05/2.3/3.0 m above sampled ground also had no misses.
  These are geometry checks on reconstructed meshes, not a simulation-state
  replay or a proof that every possible GPU ray is valid.
- Four capacity/diagnostic unit tests and five ground-observation tests passed.

## Diagnostic Changes Only

`robo_newton/robolab/utils/physics_memory.py` retains eight recent root states
in the opt-in physics profile. On failure it prints the root-reference misses,
actual ray origins/directions, root state, reset flags, terrain identity,
reset lift, independent re-casts, high-origin probes with 1 mm XY offsets,
and explicitly refreshed scans. Refreshing uses a per-world mask because
`force_recompute=True` alone does not mark all worlds outdated in this backend.
These probes execute only after failure and do not continue the rollout.

No reset, physics, observation fallback or normal training behavior was changed.
Increasing scanner height or replacing missing ground with zero would mask
the symptom without repairing NaN state.

Local logs:

- `/data/zym/roboparty_train/migration/ground_miss_stress_3000.log`
- `/data/zym/roboparty_train/migration/ground_miss_stress_seed43.log`
- `/data/zym/roboparty_train/migration/ground_miss_full_episode.log`

The `[GROUND_DIAGNOSTIC]` record in the seed-43 log contains the captured data.
The logs use Python JSON's NaN/Infinity representations for invalid tensors.

## Follow-up: First-Invalid-Stage Probe

An opt-in `--profile_physics_nan /absolute/path/snapshot.pt` switch now
instruments the MuJoCo Warp input, collision, constraint construction,
position/velocity dynamics, actuation, acceleration solve, constraint solve,
and implicit integration. The checks run inside the physics CUDA Graph.
Only active contacts/constraints are checked, not unused capacity slots.
The first invalid field is retained per world; pre-substep inputs are frozen
for that world and saved with randomized model fields and solver options.
The profile stops instead of masking invalid state or continuing training.

Important diagnostic correction: the first three follow-up runs installed
the Python wrappers AFTER the environment had already captured its graph.
Those wrappers did not instrument the existing graph. Their successful
4000-step seed-42, 4000-step seed-43 and 6000-step normal-episode seed-43
runs are final-observation/reward/root-state checks ONLY, not evidence that
every internal solver field stayed finite. The profile now invalidates and
re-captures the graph after installing the probe, verifies that integration
checks were registered, and prints `[PHYSICS_NAN_PROBE]` with active fields.
The graph is also invalidated when removing the diagnostic wrappers.

The normal-episode run peaked at 14,518 pooled contacts, 208 constraints per
world and 10 iterations. Neither it nor the two short-episode runs reproduced
the original failure. A fixed seed alone has not reproduced the same rollout.

`project_ops/scripts/diagnose_pufo_nan_snapshot.py` supports a trusted local
snapshot replay in CPU MuJoCo and staged GPU MuJoCo Warp, with optional
zero-warmstart/CG comparisons and Hessian eigenvalue diagnostics. Saving and
replaying a synthetic free-body model passed on both CPU and GPU; this is NOT
a replay of the original RP1 failure, for which no full solver snapshot exists.
Three probe kernel tests and all 94 UFO environment tests passed, including
the added graph-recapture/opt-in regression test. A synthetic
CUDA Graph replay also detected an injected non-finite value.

After correcting graph capture, two bounded runs verified all 26 active
stage-field checks and completed without NaN:

| Seed | Environments | Steps | Episode seconds | Peak contacts | Peak constraints/world | Peak iterations |
| --- | --- | --- | --- | --- | --- | --- |
| 43 | 1024 | 3000 | 0.6 | 14929 | 220 | 12 |
| 42 | 1024 | 3000 | 0.6 | 14754 | 212 | 11 |

Both used levels 0-9, actions in [-0.5, 0.5], the current solver capacities
and CUDA Graph enabled. Logs are `nan_graph_probe_seed43.log` and
`nan_graph_probe_seed42.log` under the migration directory. No actual failure
snapshot was generated in these runs. Their success does NOT establish that
the intermittent NaN is fixed. All bounded diagnostics exited; no persistent
training process was started. The exact first-invalid operation remains
unresolved. Contact/solver/integration numerics remain hypotheses, not a
confirmed root cause; Newton state conversion and event paths are not fully
instrumented by these MuJoCo-stage checks.

## Compute Sanitizer Lead

A 16-environment initcheck run reached its 360-second timeout during model
initialization, so it did not validate the rollout. It reported uninitialized
reads in cuBQL mesh BVH construction. A standalone 100-triangle mesh reproduced
303 reports with `bvh_constructor="cubql"`; the same input using Warp's default
GPU constructor reported zero. Both built the mesh successfully.

The full small-mesh report consists of 100 `initPrims +0x90` reports, 200
`writeNodes +0xb0` reports, one host-copy report, and two node-copy reports.
Inspection of the installed sm_89 binary shows the first two locations are
read-modify-write operations while initializing packed bitfields. Both paths
subsequently overwrite the entire packed value. The two node-copy reports
refer to node 1, the unused sibling of the root in the cuBQL layout. These
reports are not proof of corrupted active collision geometry or the cause
of RP1 NaN. In particular, this PUFO configuration uses MuJoCo contacts, not
Newton's separate cuBQL collision pipeline. No BVH backend was changed.

Related upstream source (used to interpret the local binary, not to establish
that a particular upstream revision is installed):

- https://github.com/NVIDIA/cuBQL/blob/main/cuBQL/builder/cuda/sm_builder.h
- https://github.com/NVIDIA/cuBQL/blob/main/cuBQL/builder/cuda/sah_builder.h

Additional logs under `/data/zym/roboparty_train/migration/`:

- `nan_stage_seed42.log`, `nan_stage_seed43.log`, `nan_stage_long_episode.log`
- `nan_initcheck.log`, `nan_initcheck_mesh.log`, `nan_initcheck_cubql_full.log`
- `nan_cubql_initPrims_sass.txt`
- `nan_probe_synthetic.log`, `nan_probe_synthetic_replay.log`
- `nan_probe_unit_tests.log`
- `nan_probe_final_tests.log`, `nan_probe_capacity_tests.log`

No default reset, training loss, CUDA Graph, or invalid-observation fallback
was changed. The PUFO collision guard is an intentional production physics
change, scoped to the PUFO Newton manager only.
