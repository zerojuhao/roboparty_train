# H20-96-1 Migration

Target: `zym@61.49.53.32`, SSH port `51820`, hostname `bjdb-h20-node-104`.
Hardware: eight NVIDIA H20 GPUs, 97871 MiB each; driver 580.126.20.

## Layout

- Source code, including current uncommitted edits: `/home/zym/roboparty_train`
- Python environment: `/data/zym/conda/envs/isaaclab_zym`
- Robot assets and motions: `/data/zym/roboparty_train/{robo_newton_data,ufo_data}`
- Logs and caches: `/data/zym/roboparty_train/{logs,cache,tmp,wandb}`
- Migration archives and validation logs:
  `/data/zym/roboparty_train/migration/h20_96_1_20261012`
- Preserved dance diagnosis poses/report:
  `/data/zym/roboparty_train/migration/pufo_dance_diagnosis_20261012`

Absolute source/environment paths are preserved for editable packages. The
IsaacLab SDK sources are included, but its dangling source-host worktree `.git`
pointer is excluded. The root project's Git metadata is included. Private keys,
credentials, old training logs, replay buffers, historical exports and general
compilation caches are not copied. A small robot USD asset cache needed by the
saved depth diagnostic is included explicitly.

The migration preserves the current PUFO implementation, not a new training
algorithm: 8 x 18 x 36 depth, a 37-frame history, integer delays 0/1, current
preprocessing, shared critic groups and EMD prioritization. The depth camera uses `ray_alignment="base"`
and follows robot roll, pitch and yaw. The separate terrain scanner keeps its
existing yaw alignment. See `PUFO_DANCE_DIAGNOSIS_20261012.md` for the depth
ablation results and remaining uncertainties.

## Activate

```bash
cd /home/zym/roboparty_train
source project_ops/activate_h20_96_1.sh
```

## Start Training Later

Check GPU availability first; migration does not reserve GPUs. Other users
started jobs on all eight GPUs during migration (about 39 GiB per GPU at the
time of verification); do not stop those jobs. Coordinate GPU use before
starting a production run. For eight ranks and 1024 environments per rank:

```bash
nvidia-smi
cd /home/zym/roboparty_train
bash project_ops/start_h20_96_1_pufo_8gpu.sh
```

The launcher starts a fresh run and uses TensorBoard logging. No persistent
training is started during migration. The latest source PUFO training directory
and checkpoint were deleted before migration; the original run cannot be
resumed without a separately supplied compatible checkpoint. Do not substitute
an older ONNX export for a resumable training checkpoint. Replay checkpoints
can occupy hundreds of GiB; monitor `/data` free space during training.

## Verification

```bash
source project_ops/activate_h20_96_1.sh
python project_ops/scripts/validate_migrated_runtime.py
python -m torch.distributed.run --standalone --nproc_per_node=8 \
    project_ops/scripts/validate_migrated_runtime.py --distributed
python -m unittest discover -s robo_newton/robolab/tasks/manager_based/ufo/tests -q
```

The training environment does not include pytest. Verification-only pytest
8.4.2 and its dependencies are installed in
`/data/zym/roboparty_train/migration/h20_96_1_20261012/test_deps`, not in the
training environment. To repeat the RSL checks:

```bash
PYTHONPATH=/data/zym/roboparty_train/migration/h20_96_1_20261012/test_deps:$PYTHONPATH \
    python -m pytest -q rsl_rl/tests/runners/test_fbcpr_evaluation.py \
    rsl_rl/tests/runners/test_replay_checkpoint.py \
    rsl_rl/tests/algorithms/test_pufo_visual_contract.py \
    rsl_rl/tests/algorithms/test_gradient_audit.py \
    rsl_rl/tests/storage/test_compact_depth_replay.py
```

Archive SHA-256 values:

- Environment: `fcc161fc403eeb940bf5a947d40bda8ca17011f6e09f58c098f13dc632dc78a6`
- Assets: `cfd06a51073780aba75a03fc2b610ffa8c9a9b4e91a0dc3f9ea56cfa7aba3bd3`

## Verified Results (2026-10-12)

- Both archive SHA-256 values matched before extraction. Code, assets and
  environment package files passed rsync content checksums; generated Python
  bytecode and Kit logs are excluded from the environment comparison.
- All eight H20 GPUs passed CUDA matrix multiplication and NCCL all-reduce.
- Torch compilation and a Warp CUDA kernel passed; caches use the data disk.
- PUFO configuration, robot assets and all 55 manifest motion files passed.
- 121 UFO/PUFO regression tests passed, including full-rotation camera and
  depth preprocessing/history checks. Two stale tests were adapted to the
  existing five-value reset API; the reset implementation was not changed.
- 56 focused RSL tests passed; one optional eight-GPU evaluation-worker test
  was skipped. The separate eight-rank NCCL validation above passed.
- Real PUFO simulation passed 80 control steps / 320 physics substeps with
  16 environments, 0.4-second episodes and diagnostic action amplitude 0.15.
  Checked states, observations and rewards remained finite. Checked reset
  joint errors and residual terrain penetration were zero. Terrain solver
  validation found 2385 boxes and 82 heightfields; stair-probe error was
  9.11e-8 m. First-use CUDA graph capture took about 214 seconds.

Runtime versions: Python 3.12, Torch 2.11.0+cu128, Warp 1.17.0, Newton 1.6.0,
MuJoCo/MuJoCo-Warp 3.12.0 and installed RSL-RL metadata 5.5.1. The project's
editable RSL sources are selected through the activation script. Existing
physics package version overrides are preserved; no dependency downgrade or
driver change was performed.

Validation logs are in the migration directory: `archive_validation.log`,
`runtime_validation.log`, `nccl_validation.log`, `ufo_unit_tests.log`,
`rsl_unit_tests.log` and `physics_validation.log`. Archives and resumable parts
are retained there. Diagnostic processes have exited; unrelated users' jobs
remain untouched. No persistent training, optimizer training run or W&B upload
was started. These checks establish migration functionality, not long-run
training convergence or resolution of the flat-ground backward-drift problem.
