# 5090D-5 Migration

Target: `zym@111.0.22.33`, SSH port `51131`.

## Layout

- Code, including editable IsaacLab and RSL-RL: `/home/zym/roboparty_train`
- Python environment: `/data/zym/conda/envs/isaaclab_zym`
- All RoboLab datasets/assets: `/data/zym/roboparty_train/robo_newton_data`
- Logs, replay, checkpoints, W&B and caches: `/data/zym/roboparty_train`
- Migration validation logs: `/data/zym/roboparty_train/migration`

The source includes the 2026-10-07 EPA face-selection repair, coordinated
distributed physics failures, compact depth replay, and per-update reward/loss
logging. Newton and physics CUDA Graph remain enabled. The current local
environment is packed with its destination prefix rewritten; no external
library patch is required for the contact fix.

Old training logs/checkpoints/replay and compilation caches are not migrated.
W&B authentication is transferred separately with owner-only permissions;
credentials are not stored in this repository. New training creates a new run.

## Activate

```bash
cd /home/zym/roboparty_train
source ./project_ops/activate_5090d5.sh
```

## Start Later

Migration does not start a persistent training run or reserve GPUs. Check GPU
availability and data-disk capacity before starting:

```bash
tmux new -s rp1-pufo-5090d5
cd /home/zym/roboparty_train
bash project_ops/start_5090d5_pufo_8gpu.sh
```

The launcher uses eight GPUs, 1024 environments per GPU, W&B project `rp1_pufo`,
and replay checkpoints enabled by default. It does not resume an old run.
Replay resides on CPU and can require about 269 GiB across eight ranks; allow
room for both old and new replay snapshots during checkpoint writes.

### Optional Startup CPU Placement

`PUFO_BIND_GPU_LOCAL_CPUS=1` enables an experimental per-rank launcher that binds
CPUs before importing the simulator or allocating replay. It queries each rank's
visible GPU PCI location and intersects its local CPUs with the process's allowed
CPU set. An empty intersection fails startup rather than widening permissions.
The default is `0`, preserving the existing launcher behavior.

```bash
PUFO_BIND_GPU_LOCAL_CPUS=1 bash project_ops/start_5090d5_pufo_8gpu.sh --resume \
  --load_run RUN_DIRECTORY_NAME --checkpoint model_STEP.pt
```

This encourages local first-touch anonymous memory allocation, but does not set
a strict memory policy or migrate existing pages. Verify `/proc/PID/numa_maps`
and steady-state training throughput before retaining it. To revert, unset the
variable (or set it to `0`) on the next launch. Do not restart before a complete
model/replay checkpoint unless a rollback of unsaved progress is acceptable.

## Validation

Verified on 2026-10-07:

- Code and both datasets match rsync content checksums, including symlinks.
- The freshly packed environment archive matches SHA-256 on both hosts:
  `828730d71d8d6d2ccfcbbdcd632442afd8b2dbc5bdc20eb08a8fe1dce27db7fd`.
- All eight GPUs pass CUDA matrix multiplication and NCCL all-reduce.
  Torch compilation and a Warp kernel pass on sm_120; all 55 manifest motion
  files and the robot asset are readable and finite where applicable.
- All 106 environment regression tests pass.
- A bounded single-GPU smoke completes 80 iterations with 16 environments
  (1280 transitions), 19 update phases with two minibatches each, and three
  two-motion evaluations at steps 0 / 512 / 1024. The persistent evaluation
  process is reused. All 75 TensorBoard scalar tags contain finite values.
- W&B authentication resolves to `zym20000911`, default entity `dtu-yiming`.
  The smoke run is finished with 22 synced history rows containing both
  training/reward and evaluation metrics, all finite:
  https://wandb.ai/dtu-yiming/rp1_pufo/runs/229rropa
- All migration test training/evaluation processes exited normally. No
  persistent training was started and no other user's task was stopped.

At the user's request this was only a simple migration smoke, not a full
eight-GPU 1024-environment training test or a long stability test. Smoke-only
overrides were batch size 32, replay capacity 4096, warmup 64 samples, update
interval 64 samples, two minibatches per phase, and 0.4-second episodes.
The normal launcher/configuration was not changed by these overrides. Newton,
CUDA Graph, the real network architecture and depth observations stayed enabled.
No training checkpoint was due during the smoke, so checkpoint/replay saving
was not revalidated on this host. Three captured physics snapshots and an
optional extended validation script were copied but not run, per the request
to keep testing simple.

Logs under `/data/zym/roboparty_train/migration`:

- `runtime_validation_5090d5.log`
- `nccl_validation_5090d5.log`
- `ufo_unit_tests_5090d5.log`
- `train_eval_smoke_5090d5.log`

The smoke artifacts are under
`/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo/2026-10-07_11-37-14_migration_smoke_5090d5_20261007`.
The target system uses UTC for directory timestamps; the date above is also
2026-10-07 in Asia/Shanghai. GPU availability is not reserved by this migration.

## Full-Scale Validation And Training

On 2026-10-07, a subsequent eight-GPU test with 1024 environments per GPU
completed normally (exit 0). This supersedes the small-smoke-only limitations
above, but is not a long-duration stability or convergence test.

- Production network, batch size 1024, replay capacity 5,120,000 per rank,
  CPU replay, Newton and CUDA Graph were retained.
- 20 rollout iterations collected 163,840 transitions and completed nine
  update phases with 16 minibatches each.
- All 902 motions were evaluated across eight persistent workers at steps
  0 and 131,072. Training continued after the second evaluation.
- All 75 TensorBoard scalar tags were finite. The saved checkpoint contained
  1,352 finite model/optimizer tensors; all eight replay shards passed
  tensor shape, dtype, coverage and finiteness checks. Each shard retained
  single-frame 18x36 uint8 depth storage.
- Later update phases took about 4.7-4.9 seconds (roughly 1,680-1,700 FPS).
  Initial evaluation took 52 seconds including setup; the reused full
  evaluation took 25 seconds. First-update compilation took 148 seconds.
- Test log: `/data/zym/roboparty_train/migration/fullscale_smoke_20261007_223001.log`
- Test artifacts: `/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo/2026-10-07_14-33-41_fullscale_smoke_5090d5_20261007_223001`

A fresh production run was launched at 22:44 Asia/Shanghai, without resuming
the test checkpoint. Only its run name was overridden; normal evaluation
(3,200,000 steps) and checkpoint (9,600,000 steps) intervals apply. W&B and
replay checkpoint saving are enabled. Existing artifacts and other users'
jobs were not removed or stopped.

- tmux session: `rp1-pufo-5090d5`
- Run name: `pufo_contactguard_5090d5_8gpu1024_20261007_2243`
- W&B: https://wandb.ai/dtu-yiming/rp1_pufo/runs/mgxv9lcv
- Console log: `/data/zym/roboparty_train/logs/launch/pufo_contactguard_5090d5_8gpu1024_20261007_2243.log`

To attach on 5090D-5, run `tmux attach -t rp1-pufo-5090d5`.
