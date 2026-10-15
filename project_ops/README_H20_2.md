# H20-2 Migration

Target: `zym@223.167.85.187`, port `51821`, SSH key `~/.ssh/id_rsa`.
This is H20-2, not H200-2 (port 51823).

## Layout

- Code and editable IsaacLab/RSL-RL sources: `/home/zym/roboparty_train`
- Python environment: `/data1/zym/conda/envs/isaaclab_zym`
- All RoboLab motions and robot assets: `/data1/zym/roboparty_train/robo_newton_data`
- Logs, checkpoints, replay, caches, W&B files: `/data1/zym/roboparty_train`
- Migration archive and verification logs: `/data1/zym/roboparty_train/migration`

The code tree links its dataset/log directories to the data disk. The unused
original IsaacLab `.venv` is replaced by a link to the packed Python environment.
All current local source edits are included. No previous training logs,
checkpoints, replay buffers, compilation caches, or credentials are copied.
Historical source archives and unrelated exported videos are not required and
are not copied. The environment archive is built for the exact destination
above; do not unpack it into another prefix.

The migration preserves the current per-frame CNN + GRU visual encoder,
8 x 18 x 36 depth input, shared critic groups, and EMD-prioritized motion
sampling. It does not change the training algorithm or start a long run.

## Activate

```bash
cd /home/zym/roboparty_train
source ./activate_h20_2.sh
python --version
```

Do not use `activate_h2002.sh` or `start_local_pufo_8gpu.sh` on this machine;
those files refer to other hosts' environment paths.

## Start Training Later

Check GPU availability with `nvidia-smi` first. Migration does not reserve GPUs.
For eight GPUs and 1024 environments per GPU:

```bash
tmux new -s rp1-pufo-h20-2
cd /home/zym/roboparty_train
bash start_h20_2_pufo_8gpu.sh
```

The launcher starts a fresh run, writes TensorBoard logs, and saves replay
buffers with checkpoints by default. W&B credentials are not migrated;
authenticate on H20-2 separately before selecting W&B logging. Eight-rank
replay checkpoints can occupy hundreds of GiB, so monitor free space on
`/data1` during training.

## Verification

```bash
source ./activate_h20_2.sh
python robo_newton/scripts/validate_migrated_runtime.py
python -m torch.distributed.run --standalone --nproc_per_node=8 \
    robo_newton/scripts/validate_migrated_runtime.py --distributed
python -m unittest discover -s robo_newton/robolab/tasks/manager_based/ufo/tests -q
```

## Verified Results (2026-10-05)

- Code and both datasets matched rsync content checksums.
- Environment archive SHA-256 matched on both hosts:
  `c5629e774465c40f03134a80269d177cdd2959af07dfd2fbd5442d4a5dd7f505`.
- All eight H20-3e GPUs passed CUDA matrix multiplication and NCCL all-reduce.
- Torch compilation and a Warp CUDA kernel passed without changing the driver.
- All 55 manifest motion files loaded with finite pose/velocity arrays.
- 104 CPU regression tests passed: 73 UFO, 22 runner, and 9 checkpoint tests.
- A CUDA BF16 network-update/checkpoint test passed all eight architecture
  combinations, including the visual model and both shared critic groups.
- The real PUFO simulation passed 80 control steps / 320 physics substeps with
  16 environments, 0.4-second episodes, and diagnostic action amplitude 0.15.
  Reset poses matched the selected motion/recovery state, and residual reset
  penetration was zero. States, rewards and observations remained finite.

Logs: `runtime_validation.log`, `nccl_validation.log`, `ufo_unit_tests.log`,
`runner_unit_tests.log`, `checkpoint_unit_tests.log`,
`cuda_bf16_update_tests.log`, and `physics_validation.log` in the data-disk
`migration` directory. First-use compilation on H20 took several minutes;
the generated caches remain on `/data1` for later startup.

The tests have exited. No persistent training or W&B upload was started.
These checks establish migration functionality, not long-run convergence or
stability of eight-GPU training at 1024 environments per rank.
