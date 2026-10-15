# 5090D-6 Migration

Target: `zym@111.0.22.33`, SSH port `51132`, key `~/.ssh/id_rsa`.

## Layout

- Code and editable IsaacLab/RSL-RL sources: `/home/zym/roboparty_train`
- Python environment: `/data/zym/conda/envs/isaaclab_zym`
- All RoboLab motions and robot assets: `/data/zym/roboparty_train/robo_newton_data`
- Logs, checkpoints, replay, caches and W&B files: `/data/zym/roboparty_train`
- Migration archive and validation logs: `/data/zym/roboparty_train/migration`

Dataset and log paths in the project link to the data disk. The unused original
IsaacLab `.venv` is replaced by a link to the relocated Python environment.
Current local source edits are included. Previous training runs, checkpoints,
replay buffers, compilation caches, credentials and exported videos are not
copied during initial migration. W&B authentication was subsequently copied
at the user's request; the remote credential file is owner-readable only.
The packed environment is prefix-rewritten for the exact path above.

This migration uses `pufo_h200_alignment_v5`, including contact-point foot
impact penalties, body-weight-gated slippage, global terrain boundary protection
and all 24 joints in `obs_state_*` evaluation. It preserves the per-frame CNN
plus GRU, 8 x 18 x 36 depth input, shared critic groups, compact depth replay,
motion prioritization and dataset-based reset joint positions.

## Activate

```bash
cd /home/zym/roboparty_train
source ./activate_5090d6.sh
python --version
```

Use the host-specific scripts above, not the H20/H200/local launchers.

## Start Training Later

Migration does not start training or reserve GPUs. Check `nvidia-smi` first.
For a fresh eight-GPU run with 1024 environments per GPU:

```bash
tmux new -s rp1-pufo-5090d6
cd /home/zym/roboparty_train
bash start_5090d6_pufo_8gpu.sh
```

The launcher now writes directly to W&B (`dtu-yiming/rp1_pufo`), retains local
TensorBoard logs, and saves replay with checkpoints by default. Override with
`--logger tensorboard` for a local-only run. Monitor free data-disk space: multi-rank replay
snapshots can consume hundreds of GiB. v4 checkpoints are incompatible with v5.
At the user's request, the temporary memory workarounds have been reverted:
Newton CUDA Graph execution uses its enabled default, and evaluation uses the
original 256-env per-rank ceiling (112 or 113 environments for each shard of
902 motions across eight ranks). Eight-shard evaluation and persistent worker
reuse remain enabled. The 32GB host previously failed full-scale CUDA Graph
instantiation. Subsequent workspace and evaluator optimizations now pass
bounded eight-GPU / 1024-env training with Graph enabled; see
[PUFO_MEMORY_OPTIMIZATION.md](PUFO_MEMORY_OPTIMIZATION.md) for measurements and
the remaining high-terrain observation issue. No persistent production training
was started as part of these optimization tests.

## Validation Commands

```bash
source ./activate_5090d6.sh
python robo_newton/scripts/validate_migrated_runtime.py
python -m torch.distributed.run --standalone --nproc_per_node=8 \
    robo_newton/scripts/validate_migrated_runtime.py --distributed
python -m unittest discover -s robo_newton/robolab/tasks/manager_based/ufo/tests -q
```

## Verified Results (2026-10-06)

- Code and both datasets matched rsync content checksums.
- The environment archive SHA-256 matched on both hosts:
  `264debd0fb56123701427e143e00cffcbfce5fe761d51b6e13453a00f9036a82`.
- All eight 32GB RTX 5090 D GPUs passed CUDA matrix multiplication and NCCL
  all-reduce with the existing 580.126.09 driver. No driver change was needed.
- Torch 2.11.0+cu128 compilation and a Warp 1.17.0 CUDA kernel passed on sm_120.
- All 55 manifest motion files loaded with finite pose/velocity arrays.
- All 122 regression tests passed: 84 environment, 32 runner/checkpoint,
  and 6 compact-depth replay tests.
- Real `train.py` smoke testing completed 1,280 environment steps and 150
  agent updates, using 16 environments, batch size 32, BF16, torch.compile,
  the production network architecture and v5 reward/observation contracts.
- Tracking evaluation completed at steps 0, 512 and 1024 using the same
  evaluator process, two motions per round. Training continued after evaluation.
- Logged scalar values, saved model parameters, optimizer state and replay
  tensor chunks were finite. The checkpoint and replay were saved and read back.
- All test processes exited; no persistent training or W&B upload was started.

Logs are in `/data/zym/roboparty_train/migration`: `runtime_validation.log`,
`nccl_validation.log`, `ufo_unit_tests.log`, `runner_unit_tests.log`,
`replay_unit_tests.log`, `train_eval_smoke.log`, and `artifact_validation.log`.
The isolated test run is under `migration/smoke_runs`; production launches
write to `logs/rsl_rl` instead. Compilation caches remain on the data disk.

The smoke run uses short warmup and 0.4-second episodes to exercise resets;
its tracking scores are not a convergence result. Those small-scale checks
establish migration functionality, not long-run stability or full-scale
throughput. The subsequent production-size launch is described below.

## Full-Scale Startup Follow-Up

The first eight-GPU / 1024-env launch exposed a constructor-only observation
probe issue: before the first reset, the clone staging grid extends beyond
the finite terrain. Observation shape inference now tolerates missing ground
only before IsaacLab assigns the observation manager. Real reset and rollout
observations still reject missing root-ground rays; terrain/reset semantics
and the v5 checkpoint contract are unchanged. This fix is present locally and
on 5090D-6, and all 85 environment regression tests passed remotely.

The failed attempts produced no training updates and their console logs are
kept. Evaluation workers now see only their assigned physical GPU, addressed
as logical CUDA device 0, and unused PyTorch caches are released across the
evaluation boundary. All 35 runner/checkpoint regression tests passed remotely.
Newton CUDA Graph instantiation still exceeded 32GB at 1024 environments. A
temporary retry disabled that optimization, but that override has since been
reverted at the user's request.

The stopped W&B-enabled retry used tmux session `rp1-pufo-5090d6` and console log
`/data/zym/roboparty_train/logs/launch/pufo_v5_5090d6_8gpu1024_wandb_nograph_20261006.log`.
Its W&B run is https://wandb.ai/dtu-yiming/rp1_pufo/runs/zy86tk3a .
Initial eight-shard evaluation completed for all 902 clips, and rollout passed
the first physics step. No first training-loss report was observed before the
user requested stopping this run. Its training and evaluation processes have
all exited. Logs are retained; these startup checks do not establish long-run
stability.

Production replay remains 5,120,000 transitions per rank, stored on CPU.
Each transition stores one 18 x 36 uint8 depth frame; eight-frame histories
are reconstructed at sampling. Depth storage is about 3.09 GiB per rank
(previously 24.72 GiB for eight stored uint8 frames). All replay fields total
about 33.66 GiB per rank, or 269.32 GiB across eight ranks. The observed host
memory increase on allocation was about 269 GiB. A full replay checkpoint
has a similar tensor payload; allow room for old and new snapshots to coexist
while saving. Model weights, simulator memory and serialization overhead are
additional.
