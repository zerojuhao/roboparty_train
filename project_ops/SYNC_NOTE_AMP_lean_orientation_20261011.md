# Sync note — RP1 AMP-Path parkour-style lean orientation (2026-10-11)

## Goal
Fight early falls on RP1 AMP-Path with a parkour-like forward-lean deadzone and
steep out-of-band pitch/roll penalty. Soft zero cost inside tolerance; large
backward scale outside. PUFO / parkour untouched.

## Parkour reference (unchanged this change)
- Helper: obo_newton/robolab/tasks/manager_based/parkour/mdp/terrain_family.py
  → symmetric_pitch_roll_penalty
- Reward: parkour/mdp/rewards.py → link_orientation
- Wired as pelvis_orientation_l2 weight -6.0 in parkour_env_cfg.py

## Changed on 4090 host (luobo1 via ZERO)
Absolute paths on this machine:

1. /home/zym/roboparty_train/robo_newton/robolab/tasks/manager_based/amp/mdp/rewards.py
   - Added AMP-local symmetric_pitch_roll_penalty (no parkour import)
   - Added lean_orientation_l2 on root projected_gravity_b
2. /home/zym/roboparty_train/robo_newton/robolab/tasks/manager_based/amp/rp1_amp_env_cfg.py
   - Added RP1PathRewardsCfg:
     - lat_orientation_l2.weight = 0.0 (avoid double-count)
     - lean_orientation_l2 weight -6.0, orward_tolerance=15°, ackward_scale=16
   - RP1AmpPathEnvCfg and RP1AmpPathEnvCfg_PLAY use RP1PathRewardsCfg
   - Base RP1-AMP / RP1RewardsCfg still uses lat_orientation_l2 at -1.0
3. /home/zym/roboparty_train/project_ops/SYNC_NOTE_AMP_lean_orientation_20261011.md (this note)

## Behavior check (unit math)
- Forward lean 10° with 15° tol → penalty 0
- Forward lean 30° → positive excess penalty
- Backward lean 10° → 16× flat g_xy L2 (matches parkour scale)

## Not changed / do not touch
- Parkour / PUFO / UFO trees (not edited for this feature)
- Base RP1-AMP (non-Path) reward defaults
- No training started on 4090

## Sync to 5090D-5 / 5090D-6
Copy at least:
1. obo_newton/robolab/tasks/manager_based/amp/mdp/rewards.py
2. obo_newton/robolab/tasks/manager_based/amp/rp1_amp_env_cfg.py
3. project_ops/SYNC_NOTE_AMP_lean_orientation_20261011.md

## Hydra overrides (test on 5090D-5)
Task id: RP1-AMP-Path (defaults already enable lean)

`	ext
env.rewards.lean_orientation_l2.weight=-6.0
env.rewards.lean_orientation_l2.params.forward_tolerance=0.2617993877991494
env.rewards.lean_orientation_l2.params.backward_scale=16.0
`

Tighten / loosen:
`	ext
env.rewards.lean_orientation_l2.weight=-8.0
env.rewards.lean_orientation_l2.params.forward_tolerance=0.17453292519943295
`

A/B restore symmetric flat:
`	ext
env.rewards.lean_orientation_l2.weight=0.0
env.rewards.flat_orientation_l2.weight=-1.0
`

Launch (from oboparty_train, local RP1 URDF workflow):
`ash
python robo_newton/scripts/rsl_rl/train.py --task RP1-AMP-Path \
  env.rewards.lean_orientation_l2.weight=-6.0 \
  env.rewards.lean_orientation_l2.params.forward_tolerance=0.2618
`
