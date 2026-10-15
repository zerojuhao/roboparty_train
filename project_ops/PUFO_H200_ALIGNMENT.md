# PUFO H200 Alignment

Reference: H200-2 `/home/zym/UFO`, run
`rp1_p3_depth_origpd_clip10_8gpu_fresh_20260925`, inspected 2026-10-05.

New RP1-PUFO runs use `training_contract=pufo_h200_alignment_v6`:

- V6 adds `penalty_body_impact` with weight -1 for `waist_yaw_link` and all
  shoulder/hip links. One reward function uses each body's strongest net total
  force vector over the four physics ticks and its pre-action world velocity.
  Load severity is Smooth L1 (beta .25) of `relu(force / body_weight - .05)`.
  Incoming speed scales it by `clamp(speed / 1 m/s, 0, 2)^2`; the sum is capped
  at 4. This follows P3's impact calculation except that its static baseline
  `1 +` is removed: static pressing, separating motion and velocity orthogonal
  to the total force have zero penalty. Total force includes friction, so this
  does not exclude all sliding contacts. Newton forces point onto the body, opposite to MJLab's
  convention, so their direction is reversed for the incoming-speed projection.
  The action term snapshots velocities before physics; all reward calculations
  remain in `mdp/rewards.py`. Other bodies and termination rules are unchanged.
  V5 and older replay has different auxiliary reward
  targets; use a fresh run or the original saved configuration/code to continue
  those checkpoints. The runner rejects loading them under V6.

- V5 aligns attachment P3 contact rewards and boundary handling: stumble keeps
  the two strongest total-force contacts per foot geometry over all four physics
  ticks; slippage uses total contact force and a 5%-of-current-body-weight gate,
  with weight -4. Global map edges truncate at scanner radius + 0.5 m, including
  the guard/border extent (internal tile crossings do not truncate). Infinite
  evaluation planes have no boundary. `obs_state_*` uses every robot joint.
  Previous replay rewards have different semantics, so V4 and older checkpoints
  cannot silently resume under V5. Plain UFO reward functions are unchanged.

- V4 uses local-ground root height, positive raised-terrain deltas, root-centered
  20 m terrain scans, and compact single-frame depth replay. Older observation
  contracts are not resume-compatible; see `REPLAY_CHECKPOINTS.md`.

- 10 m terrain tiles (user override from the H200 5 m reference). Stairs, platforms and walls retain independent box
  colliders; only height-field components are rasterized. The sensor uses the
  original visual mesh. Startup validates solver collider counts and probes
  both sides of stair edges against the authored box heights.
- Select the same 40 motion identities (902 ten-second clips), without
  modifying/deleting the 55-motion source dataset. Full evaluation enables
  weights `2 ** (2 * clamp(EMD, 0.5, 2))` for expert/reset sampling.
- Ordinary starts use fresh dataset pose/velocity, randomized yaw, motion root
  height + 5 cm, terrain footprint clearance, then actual contact penetration
  correction (one contact pass, lift = 1.25 * penetration + 5 cm only when
  penetrating). No joint position noise or changes
  to default joint zeros.
- 30% training starts use four lying orientations (side, side, supine, prone),
  default joint pose and zero velocities, matching the H200 RP1 adapter. These
  are intentional recovery starts, not noisy dataset poses. Falling alone does
  not terminate an episode. Evaluation uses reference states, not lying starts.
- Depth is 8 x 18 x 36, crop (18, 0, 14, 14), max-fill invalid depth, cap at
  2.5 m before sigma-1 Gaussian blur, normalize [.1, 2.5] m, no image DR.
  A 38-frame history supports inclusive integer delays 0/1/2. Camera mounting
  DR is +/-2 cm and roll/pitch/yaw +/-3/10/3 degrees, sampled once per environment
  at startup. Reset history is refreshed only after the final pose is written.

EMD priorities update both reset motion sampling and expert/context trajectory
sampling. Velocity estimation uses the H200/Parkour MSE objective and reports
per-axis errors, stopped-motion false positives and depth-ablation sensitivity.

PUFO defaults to two shared critic groups: discriminator-reward/auxiliary Q
and heading/velocity Q. Context heads keep independent rewards, valid masks,
bootstrap discounts and actor coefficients. `shared_context_value_net=True`
uses `heading_critic` architecture and `lr_heading_critic` for one joint
optimizer; `velocity_critic` architecture/lr apply only to separate critics.
Disabling either context head uses the remaining independent critic. Models
and optimizers with the previous separate context layout cannot be resumed
into the shared layout; use their saved configuration for legacy evaluation.

PD gains, action scale and network widths are deliberately unchanged. Except
for the V5 slippage alignment and V6 body-impact addition above, reward weights are unchanged.
This is an alignment of the diagnosed contracts,
not a claim that the Newton and MJLab pipelines are identical. Rough terrain
still uses height fields; only the box terrain geometry is exact.

## Checkpoints

The v3 profile restores 36-pixel width while retaining the aligned max-fill
and history semantics. Both v1/v2 18 x 32 checkpoints and older 18 x 36
checkpoints with different preprocessing remain incompatible. The runner
rejects mixing old checkpoints with this new profile, including weight-only
loads. Old runs can still be evaluated using their original saved configuration;
the MuJoCo export script reads crop/history/fill behavior from that configuration.
Use a fresh run for the aligned profile. No production training is started by
these edits, and no prior runs or checkpoints are deleted.

## H200 Initialization And Resume

H200 `humanoidverse/agents/envs/robot_adapter.py` implements lying starts and
reset-only penetration correction. `humanoidverse_mjlab.py:reset_idx` samples
motion/time, rotates pose and velocities together, queries local ground,
corrects penetration, then refreshes depth and clears action/history buffers.
The policy itself learns to get up; there is no separate recovery controller.

For training resume, `humanoidverse/training/workspace.py` checks
`checkpoint/train_status.json`. It constructs a fresh agent when absent;
otherwise it loads agent/optimizer state and local/global/update counters.
Rank-local replay shards are loaded when present, but physical simulation
states are reset, not restored mid-trajectory. Missing replay requires an
explicit `--resume-replay-warmup-steps` for policy-only collection before updates;
that option defaults to zero. This differs from our existing automatic
missing-replay warmup. H200 also rejects incompatible world sizes/depth contracts.

## Validation

The simulation results below describe the original 5 m alignment profile,
not a simulation validation of the subsequent 10 m tile and 36-pixel width overrides.

- 62 UFO/PUFO tests, 30 runner/checkpoint tests and 9 MuJoCo export tests pass.
- Formal `train.py` physics diagnostic: 16 environments, 80 control steps,
  320 physics ticks, four autoreset rounds. All checked states/observations/
  rewards finite; reset joint-position error zero.
- Compiled solver: 744 static boxes and 128 per-component height fields.
  28 staircase probes agree with authored heights to 5.06e-8 m.
- Small single-GPU training: 16 environments, batch 32, shortened warmup,
  two evaluation clips and seven actual optimizer updates pass. This is a
  smoke test, not a convergence result or eight-GPU throughput benchmark.
- Logs: `/tmp/pufo_h200_train_diagnostic.log`,
  `/tmp/pufo_h200_update_smoke.log`; smoke runs under
  `/tmp/pufo_h200_alignment_runs/`. No smoke processes remain running.

## Single-Pass Reset Validation

- 68 UFO/PUFO unit tests pass, including one-pass contact refresh, selective
  lifting, self-contact filtering and configurable clearance.
- With 10 m tiles and 8 x 18 x 36 depth, a 16-environment physics diagnostic
  passed 80 control steps / 320 physics ticks and four autoreset rounds.
  Checked reset poses had zero residual terrain penetration and zero joint
  initialization error; maximum additional contact lift was 0.056974 m.
- Residual contact verification is diagnostic-only, not an extra correction
  pass during normal training. This sample does not prove clearance for every
  possible pose or terrain contact. Log: `/tmp/pufo_single_pass_reset_diagnostic.log`.

## V5 Contact And Boundary Validation

- 84 UFO/PUFO tests, 23 evaluation tests, 9 checkpoint tests and 6 compact-depth
  replay tests pass. Contact regressions cover separate support/side impacts,
  strongest-two selection, all four physics ticks, partial reset, total-force
  slip projection, and randomized-body-weight load thresholds. Observation-state
  metrics include the last joint for both 24- and 29-joint inputs.
- GPU smoke: 16 environments, 80 control steps, 320 physics ticks and repeated
  asynchronous autoresets. Contact slots match raw Newton forces; an injected
  global-edge crossing truncates (does not terminate) and resets. Observations
  and rewards remain finite, and compact replay reconstructs real-camera current
  and next histories exactly. No formal training was started.
- Log: `/tmp/pufo_contact_alignment_smoke2.log`.

## V6 Body Impact Validation

- 87 tests and 30 subtests pass; one opt-in eight-GPU NCCL test is skipped.
  Checks cover the P3 formula without its static baseline, peak force direction,
  pre-action velocity snapshots, body selection, mass scaling and penalty caps.
- GPU smoke: eight environments, 64 control steps and repeated autoresets with
  random actions. Observations and rewards remain finite, and the new impact
  term produces nonzero penalties. This validates the simulation integration,
  not training convergence. No model updates or remote training changes were made.
