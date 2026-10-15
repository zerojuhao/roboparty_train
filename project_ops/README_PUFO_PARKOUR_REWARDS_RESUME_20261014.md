# PUFO Parkour Reward Resume

Host: H20-96-4 (`zym@61.49.53.32:51823`). Source root:
`/home/zym/roboparty_train`.

## Changes

PUFO adds the following terms with the existing Parkour weights:

| PUFO term | Parkour configuration term | Weight |
| --- | --- | --- |
| sound_suppression_acc_per_foot | sound_suppression | -5e-3 |
| joint_acc_l2 | dof_acc_l2 | -2.5e-7 |
| joint_vel_l2 | dof_vel_l2 | -1e-4 |

The foot penalty uses world-frame center-of-mass vertical velocity and
acceleration, gated by nonzero absolute vertical contact force, matching Parkour.
Both joint penalties select all joints. Other rewards, network architecture,
observations, action scaling, and the eight-GPU / 1024-environments-per-rank
configuration are unchanged.

Checkpoint saving now records reward term weights. Resume keeps model,
optimizer, normalizer, and training counters, but collects fresh replay if the
saved reward contract differs or is unversioned. This avoids mixing stored old
auxiliary rewards with the new objective. Existing checkpoint/replay files are
not rewritten. The configured warmup is 10,240 transitions per rank.

## Restart

The original run was interrupted with Ctrl-C only after unit tests and simulation
smoke validation passed. The original eight workers exited before restart.
The latest published checkpoint was at 192,004,096 global steps; unsaved progress
after that checkpoint is not restored (last console update: 194,183,168).

```bash
cd /home/zym/roboparty_train
bash project_ops/start_h20_96_4_pufo_8gpu.sh \
  --resume \
  --load_run 2026-10-13_14-25-22_pufo_1024x6_depthaug50_h20_96_4_8gpu1024 \
  --checkpoint model_192004096.pt \
  --run_name pufo_1024x6_depthaug50_parkour_rewards_resume_h20_96_4_8gpu1024
```

- Tmux: `rp1-pufo-h20-96-4`.
- Console: `/data/zym/roboparty_train/logs/launch/pufo_parkour_rewards_resume_20261014.log`.
- New run: `/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo/2026-10-14_10-28-57_pufo_1024x6_depthaug50_parkour_rewards_resume_h20_96_4_8gpu1024`.

## Verification

- PUFO suite: 130 tests, OK, 1 skipped (reference package unavailable).
- Checkpoint/replay suite: 12 tests passed, including eight-process restoration,
  changed/missing reward contract warmup, and optimizer preservation.
- Newton smoke: 16 environments, 32 random-action steps, finite observations
  and rewards, all three new terms evaluated.
- Smoke command requires `--require_kit` for URDF import on this host.
- Unit log: `/data/zym/roboparty_train/logs/launch/pufo_parkour_rewards_unit_20261014.log`.
- Smoke log: `/data/zym/roboparty_train/logs/launch/pufo_parkour_rewards_smoke_kit_20261014.log`.
- Startup tracking evaluation completed for all 902 clips across eight ranks.
- Resume warmup completed and updates advanced from 23,427 to at least 23,435
  (192,151,552 global environment steps), with finite losses and all nine reward
  components logged. The three added weighted penalties at that update were
  approximately -4.0098 (foot), -0.4742 (joint acceleration), and -0.0375 (joint
  velocity). These are startup diagnostics, not evidence of hardware improvement.
- W&B: https://wandb.ai/dtu-yiming/rp1_pufo/runs/9amclfvp
