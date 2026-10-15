# Shared PUFO / Parkour Physics Guard

PUFO and the parkour task family use `NewtonGuardedMJWarpManager` for their
Newton/MuJoCo Warp backend. PUFO's old configuration/import paths remain
available for saved configs. AMP-only tasks and the PhysX backend are unchanged.
The guard is installed before solver construction and CUDA Graph capture.

The contact algorithm is the validated EPA face-selection repair described in
`PUFO_GROUND_RAY_DIAGNOSTIC.md`; this change shares that repair, not a new
collision approximation. Each task keeps its own solver precision, integration,
substeps, contact capacity and constraint capacity. The shared manager reserves
at least 64 EPA horizon entries and rejects overflow after every physics tick.

## Authored Terrain Collision (2026-10-08)

This supersedes the mesh-rasterization approach described in the historical
parkour follow-up below. Newton PUFO and parkour now opt into
`FiledTerrainGeneratorCfg.native_collision`. PhysX and other tasks keep their
existing generation path. The importer retains each authored component before
concatenation, including PUFO's surrounding guard tiles and walls.

`robolab/terrains/native_collision.py` builds the observation mesh and collider
from the same height samples. Samples are carried on USD prim attributes and
imported by the project-owned `NewtonGuardedMJWarpManager` override. No installed
IsaacLab/Newton files or global heightfield registries are modified. The import
transposes X/Y samples into MuJoCo's row-Y/column-X convention, keeps the signed
minimum height in the shape transform with `min_z=0`, and explicitly forwards
dynamic friction and restitution. This matches Newton's USD friction mapping;
it does not introduce independent static/dynamic friction into MuJoCo.

PUFO keeps its native box stairs, terrain proportions, 10 m tiles, flat border,
guard ring and curriculum. Its rough/slope/obstacle height fields no longer
undergo mesh ray-rasterization. Parkour pyramid and trapezoid stairs are divided
into separate solid box treads; Perlin surface variation is retained as native
heightfield caps, with matching perimeter faces in the sensor mesh. Tread
boundaries are halfway between the original samples; cap grids use half the
original spacing so original sample locations and the boundaries both align.
This deliberately replaces the old slope-corrected cliff mesh; it is not
bitwise geometry equivalence with PhysX. Other sampled surfaces use their raw
grid without shifted cliff vertices. Arbitrary overhangs are not heightfields.

Native fields live under a dedicated `native_heightfields` subtree. Newton's
USD import excludes that one subtree after injection, avoiding thousands of
individual regexes and duplicate colliders. An unexpected non-field child is
rejected. Missing source samples or non-identity USD parent transforms also
fail explicitly instead of silently rasterizing or convexifying the surface.
Legacy OBJ caches cannot preserve this metadata and are not used by this path.

Fine parkour tread caps exposed a separate MuJoCo Warp limit: the 50-candidate
per-heightfield/geom collision workspace overflowed (`OverflowType.HFIELD=32`)
even though global contact/constraint buffers were not full. Parkour now sets
`heightfield_contact_capacity=512`; PUFO retains 50. The project manager enlarges
the candidate vector/matrices before kernel construction, with an isolated
kernel-factory cache, and restores the original objects on teardown/failure.
The version/source-hash compatibility gate still runs first. This remains a
process-local private-interface adaptation, not an upstream fix or installed
package edit. Contact reduction still selects at most four contacts per pair;
overflow checks and physics CUDA Graphs remain enabled. The explicit CUDA
regression reproduces the stock overflow and verifies zero overflow and four
finite contacts after expansion.

Regression tests cover signed/flat/asymmetric heights, MuJoCo ray agreement,
material forwarding, rough stair caps, all active parkour terrain families,
PUFO guard placement/origins, and exclusion-container safety. Bounded stress
tests can select maximum difficulty without changing training defaults:

```bash
python robo_newton/scripts/smoke_newton_env.py --task RP1-PUFO --device cuda:0 --num_envs 64 --steps 1000 --random_actions --episode_length_s 2 --terrain_difficulty 1 --seed 44 --check_pufo_contacts --check_pufo_replay --require_kit
python robo_newton/scripts/smoke_newton_env.py --task RP1-Parkour --device cuda:0 --num_envs 32 --steps 1000 --random_actions --episode_length_s 2 --terrain_difficulty 1 --seed 43 --check_parkour --require_kit
ROBO_TEST_HFIELD_CUDA=1 python -m unittest discover -s robo_newton/robolab/tasks/manager_based/parkour/tests -p test_hfield_capacity.py
```

### Native Terrain Validation Results

- 118 UFO/environment/physics tests passed. Parkour discovery passed 23 CPU
  tests and skipped the opt-in CUDA test; that CUDA test then passed separately.
- PUFO: 16 environments/300 steps, then maximum difficulty with 64 environments
  and 1,000 random-action steps for each of seeds 43 and 44. Observations,
  rewards and contact forces stayed finite; no ray failure or capacity overflow
  occurred. Forced boundary reset, raw contact-slot agreement and compact
  current/next depth-history reconstruction passed. The native solver contained
  2,335 boxes and 82 heightfields; stair ray error was about 1.1e-7 m.
- PUFO training smoke: 32 collection iterations, three actual optimizer-update
  phases, all reported losses finite. Only smoke settings reduced replay size,
  batch size, seed steps and update count; model architecture and terrain were
  unchanged. Torch compilation and tracking evaluation were skipped for this
  bounded test. This does not validate tracking evaluation or convergence.
- Parkour: maximum difficulty, seed 43, 32 environments/1,000 random-action
  steps passed. The first step had zero terminations; later random-action falls
  produced normal orientation/root-height terminations, not mass resets every
  step. No non-finite observations/rewards or capacity overflows occurred.
- Parkour PPO/AMP: two iterations passed with symmetry and physics CUDA Graph
  enabled. Iteration times were 7.80/2.63 seconds; each checkpoint had 376 finite
  tensors. Full scene setup took about 211 seconds in this concurrent test run;
  solver construction took 5.94 seconds. There were 1,931 boxes and 1,124 fields.
  More independent rough tread surfaces increase geometry/setup cost; this is
  not a claim of training speed or memory improvement.
- Initial parkour attempts correctly stopped on HFIELD overflow at capacity 50.
  The final `v3` tests use 512 and passed. The isolated CUDA regression's log
  intentionally includes the stock overflow before testing the repair.

Final logs under `/data/zym/roboparty_train/migration/`:
`native_terrain_pufo_stress_v2_20261008.log`,
`native_terrain_pufo_train_20261008.log`,
`native_terrain_parkour_stress_v3_20261008.log`,
`native_terrain_parkour_train_v3_20261008.log`, and
`native_hfield_capacity_cuda_20261008.log`.
All bounded test processes exited. No long-running training or remote deployment
was started; other users' jobs were untouched. These checks do not guarantee
absence of every long-run numerical failure or prove policy convergence.

## Shared Terrain Import and All-Contact Stumble (2026-10-08)

Both tasks now inherit `robolab.terrains.native_importer.NativeTerrainImporter`.
The common path disables collision on the combined observation mesh and imports
authored heightfields, exact boxes (including rotated boxes), and other convex
components separately. Parkour retains its strip-merging generator and virtual
edges; PUFO retains its original unwelded observation mesh and guard generator.
Terrain distributions, materials, reset policies, reward weights, and task-specific
solver capacities are unchanged. PhysX and plane/USD terrain keep their existing
import paths. Compatibility names remain in the task modules.

Both tasks validate collider counts and sample collision heights before rollout
(Parkour startup, PUFO first contact-aware reset). The height check compares
downward rays through the observation mesh against static-only MuJoCo collision
geometry, using a private model copy so robot occlusion cannot affect the check.
It probes every tile, plus representative component edges and rough samples,
and rejects hit/miss disagreement or errors over 1 mm. This is bounded sampling,
not an exhaustive geometric proof or a check of dynamic contact response.
Exact shared box boundaries are avoided by sub-millimetre probe offsets: float32
box centers/extents can leave sub-micron seams at mathematically shared edges.
Both ray implementations receive the same float32 starting coordinates.
PUFO additionally retains its denser, task-specific stair-edge probes.

The new Parkour probes caught an existing rotated-box import error: the old
importer passed WXYZ quaternions to IsaacLab 3's XYZW shape-spawn interface.
Axis-aligned boxes concealed the error, but rotated obstacle collision surfaces
were displaced relative to the observation mesh. The shared importer now uses
XYZW, with a regression that checks actual USD-transformed box corners.

`robolab.sensors.terrain_contact.TerrainContactSensorCfg` selects the shared
Newton sensor. Standard net body forces and optional air/contact timers are
still maintained. One pass over all reported contacts computes the maximum
`max(norm(Fxy) - horizontal_ratio * abs(Fz), 0)` per sensed body, before contacts
are summed. A scalar history retains all four physics ticks of a policy step;
the reward selects foot bodies by name and applies the existing 20 N threshold,
100 N scale, [0,1] clamp, and time/body maximum. Reset only clears the selected
environments. Non-finite forces remain non-finite reward signals rather than
being hidden by the clamp. The ratio in the reward must match the sensor ratio.

This replaces PUFO's top-two-contact slots and Parkour's net-foot-force stumble.
Two large supporting forces can no longer exclude a smaller side impact.
Contacts already discarded by the physics solver remain unavailable. PhysX
retains its body-net-force fallback; all-contact equivalence is Newton-only.
Parkour retains `track_air_time=True` and weight -1; PUFO retains no air-time
tracking and weight -2. Their RL time scaling remains different. Other Parkour
net-force-history consumers now also see four ticks instead of three.

The initial GPU attempt caught a backend configuration conversion error for the
new project-only ratio field. The sensor now extracts that field before passing
a base-only config to the backend. Regression tests cover this conversion,
weak side impacts under multiple supports, opposite contact directions,
multiple shapes per body, body selection, full-step history, masked reset,
invalid forces, geometry offsets/misses, gaps, and exclusion of robot geometry.

### Shared Component Validation Results

- UFO discovery: 120 tests passed. Parkour discovery: 28 passed and the
  pre-existing opt-in heightfield-capacity CUDA test skipped. The nine shared
  contact/alignment tests also passed with the reduction/history kernels on CUDA.
- PUFO: maximum difficulty, 32 environments, 1,000 random-action steps for seed
  45 (`v3`) and seed 46 (`v4`). Both exited zero, including forced boundary
  truncation/reset, all-contact statistics versus raw Newton forces, and exact
  compact depth replay reconstruction. Seed 46 used the final XYZW box importer;
  1,075 probe rays agreed, maximum finite hit-height error 1.55e-6 m.
- Parkour: maximum difficulty, seed 45, 32 environments, 1,000 random-action
  steps with the final XYZW importer (`v3`), exit zero. All 1,460 probe rays
  agreed, maximum finite hit-height error 1.43e-6 m. First-step terminations
  were zero; later random-action falls had ordinary termination reasons.
  All-contact statistics matched raw Newton forces. Rewards, observations and
  contact histories stayed finite with physics CUDA Graphs and guards enabled.
- Probes just outside component/map edges may miss both representations; only
  hit/miss disagreement is an error. These bounded tests do not prove policy
  convergence, exhaustive collision equivalence, or long-run stability.

Logs under `/data/zym/roboparty_train/migration/`:
`shared_terrain_contacts_ufo_tests_v3_20261008.log`,
`shared_terrain_contacts_parkour_tests_v3_20261008.log`,
`shared_terrain_contacts_pufo_v3_20261008.log`,
`shared_terrain_contacts_pufo_v4_20261008.log`, and
`shared_terrain_contacts_parkour_v3_20261008.log`.
Earlier unsuffixed/`v2` smoke logs include the configuration-conversion error
and the deliberately fail-closed terrain mismatch diagnostics, not successful
runs. All test processes exited. No formal training, remote synchronization,
other-user task changes, or installed dependency edits were performed.

## Parkour Stair-Edge Startup Check (2026-10-08)

Parkour additionally checks every generated tile of the active ascending and
inverted pyramid/trapezoid stair families. The generator retains references to
each tile's collision components, with family/row/column labels; their world
transforms remain the same as the native collision components. This adds no
terrain, reset, reward, material, or solver changes.

`parkour/stair_validation.py` derives edges from the actual axis-aligned boxes
and authored heightfield caps, including adjoining walls. Every distinct
rectangular edge receives three longitudinal samples and probes on both sides
at 2 mm and 10 mm offsets (reduced for very narrow components). Identical
box/cap edges are deduplicated; no global sampling quota can exclude an entire
stair level. Exact corners and midpoints are avoided to reduce ambiguous seams.
This remains discrete edge sampling, not a continuous proof or a guarantee of
detecting sub-millimetre gaps between sample locations.

Expected heights come from the full observation mesh, including rough caps,
not just flat box tops. The existing static-only MuJoCo ray comparison rejects
hit/miss disagreement or height errors above 1 mm. Errors include the stair
family, row, column, XY position and both heights. Successful startup logs show
tile/probe counts, per-family coverage, maximum height error and check duration.
An empty stair tile is rejected; a terrain preset containing no stairs skips
this additional check. The check shares the existing one-time startup guard;
the smoke script no longer repeats the full check after reset.

Regression tests cover near/far probes on both sides, ascending/descending rough
stairs, a 3 mm collision-edge shift, omitted rough caps, missing collisions,
filled valleys, world-space tile metadata, deduplication, and once-only startup.

Validation: 36 Parkour tests passed (one existing opt-in CUDA test skipped),
including eight new stair-edge tests; all 120 UFO regression tests passed.
The RP1-Parkour GPU smoke used seed 46, maximum terrain difficulty, 16
environments and 300 random-action steps. It exited zero with finite rewards,
observations and contact histories, zero first-step terminations and no solver
overflow. The startup edge check ran once: 40 tiles, 91,260 probes, all hits,
maximum height error 1.89e-6 m, elapsed 5.448 s. Each family had 10 tiles;
pyramid/inverted-pyramid/trapezoid/inverted-trapezoid counts were respectively
33,756 / 33,792 / 13,968 / 9,744 probes. Physics CUDA Graphs remained enabled.
This is bounded execution validation, not a long-run training guarantee.

Logs under `/data/zym/roboparty_train/migration/`:
`parkour_stair_edges_tests_20261008.log`,
`parkour_stair_edges_ufo_regression_20261008.log`, and
`parkour_stair_edges_smoke_20261008.log`.
All test processes exited. No formal training or remote deployment was started.

## Dependency Compatibility

Use `robo_newton/requirements-physics.txt` in the existing IsaacLab environment.
This file pins the four physics packages, not the entire training environment.
Startup checks their versions, the two collision source files' SHA-256 hashes,
and the patched interfaces. Unknown versions or source changes stop startup
instead of silently omitting protection. Even a harmless source edit requires
reviewing/updating the compatibility record. `[CONTACT_GUARD]` reports the
enabled revision and runtime versions.

No site-packages files are edited. The Python replacement is process-wide
while the simulator is alive; use separate processes for independent physics
backends. Installation failure and simulator teardown restore the original
functions and EPA capacity. This is not a general multi-simulator patch registry.

## Upgrades and Validation

Do not merely update the version/hash allowlist to suppress a startup error.
Review the upstream face flags, affine coordinate semantics and collision
kernel construction first. Run contact-guard tests on CPU and CUDA, shared
backend/configuration tests, the saved failure replays, and per-task stress
and training smoke tests before accepting a new dependency combination.

The full historical failure snapshots remain external diagnostic artifacts in
`/data/zym/roboparty_train/migration/`; they are not bundled with a fresh clone.
The repository includes portable tests for captured degenerate triangles,
normal signed coordinates, face selection and capacity exhaustion.

Example (with the validated environment and project packages installed):

```bash
python -m unittest discover -s robo_newton/robolab/tasks/manager_based/ufo/tests -p 'test_shared_physics_guard.py'
PUFO_GUARD_TEST_DEVICE=cuda:0 python -m unittest discover -s robo_newton/robolab/tasks/manager_based/ufo/tests -p 'test_contact_guard.py'
python robo_newton/scripts/smoke_newton_env.py --task RP1-Parkour --device cuda:0 --num_envs 16 --steps 150 --random_actions --episode_length_s 0.6 --require_kit
```

## Parkour Terrain and Symmetry Follow-Up

The immediate parkour resets were independent of the contact guard. The custom
terrain importer sent the entire map to MuJoCo as one mesh collider, which uses
convex-hull collision. The hull enclosed the terrain's walls and spawn locations.
An instrumented 16-environment run recorded base-contact termination in every
environment on the first step, base forces around 47--101 kN and a contact
penetration of 1.22 m. The same failure occurred with the contact guard disabled.

`parkour/terrain_importer.py` now retains the generator's authored components
before concatenation. The original combined mesh remains the observation and
virtual-edge source but has collision disabled on Newton. Height fields are
imported separately at the configured horizontal resolution; boxes, including
rotated boxes and walls, use equivalent native box colliders. Other convex
solids remain individual meshes. Zero-thickness box floors become height fields.
Adjacent axis-aligned infill strips with identical X/Z bounds are merged along
Y without changing their solid union. Inverted trapezoid stairs otherwise add
about 1,900 boxes per tile; Newton 1.6 materializes quadratic same-static-body
filter pairs for them. Keeping all strips separately caused excessive startup
time and host memory growth, even though the geometry was correct.
This does not change terrain sampling, spawn origins, reset height, rewards or
termination thresholds. PhysX retains triangle-mesh collision.

Height fields are a rasterized collision representation, not an exact triangle
surface at every steep corner. The regression checks preserve the authored
observation mesh and component transforms; they do not prove sub-grid contact
equivalence. Unsupported mesh surfaces are rejected, not silently convexified.
Legacy terrain OBJ caches lose component boundaries, so this importer requires
`use_cache=False` (the current parkour default).

RP1 symmetry now allocates the augmented nested TensorDict with `obs.repeat(2)`
and passes `env` to the critic mirror function. Both errors are fixed without
disabling augmentation or mirror loss.

```bash
python -m unittest discover -s robo_newton/robolab/tasks/manager_based/parkour/tests
python robo_newton/scripts/smoke_newton_env.py --task RP1-Parkour --device cuda:0 --num_envs 16 --steps 150 --random_actions --episode_length_s 0.6 --check_parkour --require_kit
```

The additional smoke flag checks that the solver contains the expected
independent colliders, rejects first-step termination, and prints termination
reasons on subsequent steps. Finite observations/rewards alone are insufficient:
a broken environment can return finite values while resetting on every step.

## Follow-Up Validation (2026-10-08)

- 15 parkour regression tests passed. The 116-test UFO/environment/physics
  suite also passed during this follow-up.
- The working tree disables `stairs_gap_platforms`; that existing change was
  preserved. Command ranges and the nested locomotion orientation reward now
  drop its inactive overrides rather than failing during initialization/step.
- Full RP1-Parkour terrain imported as 1,011 boxes and 144 height fields, with
  exact solver-count verification and no whole-map mesh collider. Scene creation
  took 136.7 seconds; model finalization took 4.66 seconds.
- 16 environments, 150 random-action steps, 0.6-second episode limit: exited
  zero. The first step terminated zero environments. No base-contact, root-height,
  bad-orientation or boundary terminations occurred; the 80 total timeouts were
  the five expected 30-step resets of all 16 environments. Observations/rewards
  stayed finite and no solver-capacity overflow was reported.
- Two PPO/AMP iterations completed with data augmentation, mirror loss and
  physics CUDA Graph enabled. Both checkpoints contained 376 finite tensors;
  actor, critic and discriminator parameters changed between checkpoints.
  Main optimizer step counters advanced from 12 to 24. Iteration times were
  6.43 and 2.91 seconds, excluding initialization.
- This is short-run execution validation, not evidence of convergence or a
  guarantee against every long-run/high-difficulty physics failure.

Final logs: `/data/zym/roboparty_train/migration/parkour_collision_smoke_v7_20261008.log`
and `/data/zym/roboparty_train/migration/parkour_collision_train_v7_20261008.log`.
Checkpoints: `/data/zym/roboparty_train/logs/rsl_rl/rp1_parkour/2026-10-08_12-00-33_parkour_collision_symmetry_smoke_v7_20261008/`.
Both bounded GPU test processes have exited. No formal training was launched.

## Initial Local Validation (2026-10-08, Before Follow-Up)

- 115 environment/physics unit tests passed, including shared RP1/RPO config
  preservation, compatibility rejection and installation rollback.
- Six contact-guard tests passed on CUDA. All three historical full physics
  snapshots (seeds 43/44/45) replayed with `--affine project --require-finite`
  and exited zero.
- RP1-Parkour, 16 environments, 150 random-action steps, 0.6-second episode
  limit: exited zero; observations/rewards remained finite and no capacity
  overflow was reported. All 16 environments returned done on EVERY step.
  This validates repeated reset execution, not sustained locomotion or
  high-difficulty terrain stability. The immediate termination needs a
  separate investigation; its cause was not established here.
- A two-iteration RP1-Parkour training attempt installed the guard and collected
  its first rollout, then failed at the first update in the pre-existing
  `mdp/symmetry/rp1.py:64`: `obs_aug` is used before initialization. Symmetry
  was not disabled or modified to bypass this failure. Full parkour training
  is therefore NOT validated by this change.

Logs are under `/data/zym/roboparty_train/migration/`:
`shared_guard_all_ufo_tests_20261008.log`,
`shared_guard_seed{43,44,45}_replay_20261008.log`,
`parkour_shared_guard_smoke_20261008.log`, and
`parkour_shared_guard_train_20261008.log`.

Only local project code was changed; no remote deployment, existing training
restart, installed-library edit, or formal training launch was performed.
