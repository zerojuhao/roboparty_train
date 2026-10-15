# H20-96-4 Migration

Target: `zym@61.49.53.32:51823`, hostname `bjdb-h20-node-107`,
internal address `172.31.0.107`. Completed on 2026-10-13.

## Layout

- Current source: `/home/zym/roboparty_train`
- Python environment: `/data/zym/conda/envs/isaaclab_zym`
- Assets: `/data/zym/roboparty_train/{robo_newton_data,ufo_data}`
- Runtime logs and caches: `/data/zym/roboparty_train/{logs,cache,tmp}`
- Validation logs: `/data/zym/roboparty_train/migration/h20_96_4_20261013`

The environment and required assets were copied directly from H20-96-1
(`172.31.0.104`) over internal SSH. Its source checkout was used as a base;
checksum-based synchronization applied the current local source changes.
Temporary SSH agent forwarding was used without copying private keys.
Historical checkpoints, training logs and W&B credentials were not copied.
No production training was started during migration. W&B is online in the activation script;
the user's local W&B login was subsequently copied with owner-only access.

## Activate

```bash
cd /home/zym/roboparty_train
source project_ops/activate_h20_96_4.sh
```

Use `--logger wandb` when launching training (the PUFO default). Do not use another
host's launcher, which may configure W&B or different runtime paths.

## Verification

- Environment rsync content checksums matched H20-96-1, excluding generated
  Python bytecode and log files.
- Current depth encoder, estimator and activation script SHA256 matched local.
- Python 3.12.14, Torch 2.11.0+cu128, Warp 1.17.0, Newton 1.6.0,
  MuJoCo and MuJoCo-Warp 3.12.0 imported successfully.
- All eight H20 GPUs passed CUDA matrix multiplication and NCCL all-reduce.
- Torch compilation and Warp CUDA kernel/cache placement passed.
- PUFO configuration, robot assets and 55 manifest motion files passed.
- UFO/PUFO unit tests: 126 run, OK, 1 skipped.

Validation logs: `runtime_validation.log`, `nccl_validation.log`,
`ufo_unit_tests.log`. These checks establish migration functionality, not
training convergence or full-scale training throughput.

## Production Training

Started on 2026-10-13 at 14:24:46 UTC in tmux session
`rp1-pufo-h20-96-4`, using all eight GPUs with 1024 environments per rank
(8192 total). This is a fresh `RP1-PUFO` run, not a checkpoint resume.
The existing 1024-wide, six-block actor configuration and shared encoder
gradient are retained. Depth scale, convolution and pixel-failure augmentations
each use probability 0.5; Gaussian depth noise remains enabled.

```bash
cd /home/zym/roboparty_train
bash project_ops/start_h20_96_4_pufo_8gpu.sh
```

Attach to the running job:

```bash
ssh -p 51823 zym@61.49.53.32
tmux attach -t rp1-pufo-h20-96-4
```

- Console: `/data/zym/roboparty_train/logs/launch/pufo_h20_96_4_8gpu1024_20261013.log`
- Run: `/data/zym/roboparty_train/logs/rsl_rl/rp1_pufo/2026-10-13_14-25-22_pufo_1024x6_depthaug50_h20_96_4_8gpu1024`
- W&B: https://wandb.ai/dtu-yiming/rp1_pufo/runs/4bcre5gd

Startup verification: all 902 initial evaluation clips completed; the first
four update phases reached 122880 global environment steps with finite losses.
