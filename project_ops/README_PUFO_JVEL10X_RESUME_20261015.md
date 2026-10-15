# PUFO joint velocity 10x resume

Host: H20-96-4 (`zym@61.49.53.32:51823`). Source:
`/home/zym/roboparty_train`. Date: 2026-10-15 (Asia/Shanghai).

## Scope

Only `PUFORewardsCfg.joint_vel_l2.weight` changes, from `-1e-4` to
`-1e-3`, as explicitly requested. The formula and all-joint selection
are unchanged. Landing penalty remains `-5e-3`, joint acceleration
remains `-2.5e-7`, and all other training settings are unchanged.
Parkour's own configuration is not changed. The weight regression test
now checks the intentional 10x velocity multiplier while preserving
the Parkour landing/acceleration weights.

## Resume source

- Run: `2026-10-14_10-28-57_pufo_1024x6_depthaug50_parkour_rewards_resume_h20_96_4_8gpu1024`.
- Checkpoint: `model_326402048.pt`, 326,402,048 global environment steps.
- Saved topology: 8 ranks, 1024 environments per rank, 39,844 environment iterations.
- Last old-run console update: 334,667,776 steps, update 40,832.
- The 8,265,728 steps after the published checkpoint were not restored.
  The existing interrupt handler does not save an additional checkpoint.
- Old checkpoints, replay snapshots, and logs were not modified or deleted.
- The old tmux training was interrupted; all eight workers and their eight
  tracking-evaluation children were confirmed exited before replacement.

The normal resume path preserves model, optimizer, normalizer and training
counters. Because the saved reward weights differ, the runner rejects
the old replay for training and collects fresh transitions before updates.
This is a continuation experiment, not a from-scratch run or an equal-step
causal ablation.

## Launch

```bash
cd /home/zym/roboparty_train
bash project_ops/start_h20_96_4_pufo_8gpu.sh \
  --resume \
  --load_run 2026-10-14_10-28-57_pufo_1024x6_depthaug50_parkour_rewards_resume_h20_96_4_8gpu1024 \
  --checkpoint model_326402048.pt \
  --run_name pufo_1024x6_depthaug50_jvel10x_resume_h20_96_4_8gpu1024
```

- New tmux session: `rp1-pufo-jvel10x`.
- New run: `2026-10-15_03-41-14_pufo_1024x6_depthaug50_jvel10x_resume_h20_96_4_8gpu1024`
  under `/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo` (directory timestamp is UTC).
- W&B: https://wandb.ai/dtu-yiming/rp1_pufo/runs/puy7e9ko
- Console log: `/data/zym/roboparty_train/logs/launch/pufo_jvel10x_resume_20261015.log`.
- Source edits synchronized between local 5090D-5 and H20-96-4 only.

## Verification

- Five reward tests passed locally and on H20-96-4.
- Twelve checkpoint/replay tests passed locally, including changed-reward
  replay rejection and optimizer preservation.
- Checkpoint was opened read-only with Torch; reward weights, model and
  optimizer payloads, topology, and step counters were present.
- Saved env.yaml comparison differs only in joint_vel_l2.weight and log_dir.
  Saved agent.yaml differs only in resume run/checkpoint and new run name.
- Startup tracking evaluation completed across all 902 clips / eight ranks.
- Reward-contract change was detected and old replay was not restored.
  Fresh replay warmup completed at 326,492,160 steps (11 time steps/rank).
- Actual optimizer updates advanced through at least update 39,827 and
  326,516,736 global steps with finite reported losses/rewards.
- First update: joint_vel_l2 raw 506.8719, weighted -0.5069, confirming
  the new -0.001 weight. These startup numbers do not establish better
  movement quality; compare dance, get-up and pushed-stair behavior later.
