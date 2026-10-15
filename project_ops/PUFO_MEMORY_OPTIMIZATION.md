# PUFO Memory Optimization

## Scope

The optimized PUFO profile keeps CUDA Graphs, 1024 training environments per
rank, the 256 evaluation-environment ceiling, persistent eight-shard evaluation,
network sizes, batch size, depth resolution/history, replay capacity, motion
prioritization, rewards, reset behavior, and solver accuracy settings unchanged.
Only workspace capacities and the evaluator's unused training modules change;
overflow detection now fails explicitly instead of silently continuing.

## Reproduced Cause

On a 32GB 5090 D, a single 1024-environment physics-only process (no learner,
replay or evaluation process) exhausted device memory during first CUDA Graph
instantiation with nconmax=2048 and njmax=8192. After reset it used 16.29 GiB;
at failure it used 31.35 GiB, with only 0.005 GiB free. This rules out replay or
evaluation as necessary causes of this failure.

The installed MuJoCo Warp collision_convex.py allocates EPA scratch arrays per
naccdmax, which defaults to nconmax times environment count. At 35 CCD iterations,
the six main arrays alone require about 9.76 GiB per collision invocation for
2048 x 1024 slots. Captured substeps have additional allocations; this estimate
is not a complete graph-memory measurement.

The older tracked configuration used nconmax=512, njmax=4096 and one substep.
The failing configuration used 2048, 8192 and two substeps. Older production run
configs have been removed, so the tracked version is evidence of code changes,
not a recovered configuration of the last successful run. The retained v5 smoke
test used only 16 environments and did not establish 1024-environment capacity.
The latest local eight-GPU launch also failed in wp_cuda_graph_create_exec:
/data/zym/roboparty_train/logs/launch/pufo_v5_local8gpu1024_20261006_025353.log.
Those 48GB cards were shared with other users, so nominal VRAM was not the
available budget. No other users' jobs were stopped for this work.

## Changes

- Restore allocation limits to nconmax=512 and njmax=4096. Contact capacity is
  pooled across worlds (524,288 slots at 1024 environments); constraint capacity
  is per world. Keep two substeps, 600 solver iterations, 80 line-search
  iterations, tolerance 0.01, and the same collision geometry/integrator.
- Use NewtonPUFOMJWarpManager to reject any MuJoCo Warp overflow after every
  physics tick, before auto-reset can clear flags. Reset collision correction
  also checks overflow. This includes EPA overflow, not just contact/constraint
  capacity. A failure must be investigated, not suppressed to keep training going.
- Increase EPA horizon workspace from 24 to 64 entries. The first full-scale
  test exposed EPA_HORIZON overflow (flag 256), distinct from contact/constraint
  exhaustion. This changes workspace, not EPA iterations or tolerance. MuJoCo
  Warp 3.12 exposes this through a private module constant; the PUFO backend
  scopes the override to its simulator lifetime and restores it on failure or
  teardown. Dependency upgrades require a compatibility check.
- Propagate a training rank's capacity failure through the existing distributed
  finite-state check, so peer ranks also exit instead of waiting in NCCL.
- Evaluation constructs only actor (including depth encoder and velocity
  estimator), backward map and observation normalizer. It receives only those
  state dictionaries. The production architecture's model payload shrinks from
  0.7291 GiB to 0.0409 GiB per evaluator. Network operations and weights are shared
  with training implementation; full checkpoint loading remains supported.
- Add opt-in physics memory/capacity diagnostics and optional trajectory traces.
  Tests use explicit finite-state and overflow checks, not process exit status
  alone (the simulator wrapper can print a traceback yet exit with status zero).

## Validation (2026-10-06)

The candidate limits passed 200 random-action steps at 1024 environments on
one 5090 D with CUDA Graphs enabled. Simulation memory was 12.85 GiB; observed
peak total contacts were 15,050, per-world constraints 192, solver iterations 8.
No overflow or non-finite state was observed. This short low-level-terrain test
does not establish a bound for all future policies or terrain levels.

CPU regression tests compare the inference-only agent's actions and backward
encodings against a full agent at identical inputs/weights, with zero tolerance.
They also check omitted modules, overflow handling, backend factory resolution,
EPA override restoration and two-process failure propagation. All 130 focused
tests passed locally: 89 environment, 26 evaluation/runner, nine checkpoint,
and six compact-depth replay tests. The 89 environment tests also passed remotely.

Two bounded production-size runs passed on 5090D-6: eight GPUs, 1024 environments
per GPU, production batch/network/replay, CUDA Graph enabled, and persistent
evaluation workers capped at 256 environments (112/113 actual per shard).
Each run completed 131,072 global environment steps, 80 optimizer updates per
rank, and two evaluations covering all 902 motions. Training continued after
the second evaluation. The checkpoint run saved the model and replay for all
eight ranks; all 1,624 saved model/optimizer/replay tensors were finite, and
the replay manifests retained single-frame 18 x 36 uint8 depth storage.
All 385 logged scalar values in the checkpoint run were finite as well.
This verifies a partially populated replay snapshot, not full-buffer disk capacity.

Two-second device-memory samples during the first successful full run peaked
at 26,219 MiB (25.60 GiB), leaving about 5.75 GiB of the observed 32,110 MiB
usable device-memory budget (after driver reservation).
This is a sampled peak, not an instantaneous allocator high-water mark.

A local 1024-environment, 600-step frequent-reset test passed on terrain levels
0-2. A separate test explicitly covering levels 0-9 reached over 500 steps
without solver overflow but failed because a root-ground ray missed terrain.
Its sampled capacity peaks were 14,554 total contacts and 208 constraints per
world. Higher-terrain observation robustness is therefore NOT validated; the
missing-ray failure remains a separate issue and must not be hidden by a
placeholder ground height.
A repeat of the same level-0-9 stress configuration passed all 600 steps
(14,524 contacts, 208 constraints per world, nine solver iterations). This
non-reproduction does not resolve the earlier missing-ray error or establish
its exact cause. The opt-in diagnostic now prints failing-step positions and
terrain levels for any ray-miss worlds; no observation fallback was introduced.
Follow-up testing captured a seed-43 ground-ray failure with NaN root state
and NaN ray origin. See [PUFO_GROUND_RAY_DIAGNOSTIC.md](PUFO_GROUND_RAY_DIAGNOSTIC.md).
The upstream cause of that NaN remains unproven; the memory validation does
not establish long-run numerical stability.

Fixed-action 64-environment, 40-step comparisons were also run against the old
capacities. Mean joint-position differences were 0.000767 rad for old/new versus
0.000813 rad between two old-config repeats. Mean reward differences were
0.002187 versus 0.002537 respectively. This is consistent with the existing
non-deterministic contact solver's run-to-run variation, not proof of bitwise
equivalence or identical long-run learning. These short runs do not validate
policy convergence or tracking quality.

Remote diagnostic logs are under /data/zym/roboparty_train/migration:
capacity_baseline_1024.log, capacity_512_4096_1024.log,
memory_optimized_8gpu1024_smoke_r2.log (EPA overflow before the fix),
memory_optimized_8gpu1024_smoke_r3.log (passed),
memory_optimized_8gpu1024_checkpoint.log (passed), and
memory_optimized_8gpu1024_memory.csv. Local high-terrain diagnostics are in
/data/zym/roboparty_train/migration/capacity_alllevels_reset_stress.log (failed)
and capacity_alllevels_diagnostic.log (repeat passed).

## Diagnostic Usage

After activating the matching host environment, run train.py with
--profile_physics_steps 200 and --diagnostic_action_scale 0.1, retaining the usual
--task RP1-PUFO --num_envs 1024 --device cuda:0 --visualizer none arguments.
This does not construct a training runner or write training checkpoints.
--profile_physics_trace /absolute/path/trace.pt additionally records actions,
root state, joint positions/velocities and rewards for fixed-seed comparisons.
Profile limits may be overridden through env.sim.physics.solver_cfg.nconmax
and env.sim.physics.solver_cfg.njmax. Always require the final [CAPACITY] PASS.
