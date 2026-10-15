# H200-2 Training Environment

Migration target: `zym@223.167.85.187`, SSH port `51823`.

## Current Status: 2026-10-03 Reset and NaN Fixes

The original eight-GPU run stopped after a non-finite physical state; it has
not been restarted. The initial-launch and live-mirror notes below describe
the earlier run, not currently running services. Do not reuse its old PID.

Reset now samples the motion ID and time before writing the root/joint state.
The later command-manager reset consumes that sample without sampling twice.
Joint position noise was subsequently disabled at the user's request: joint
positions and velocities now come directly from the sampled motion frame,
without changing the reference zero. This matches the archive's sampling
order, not every reset detail. The archive supports Gaussian initial noise
but its saved configurations leave the overall noise multiplier at zero;
root-height placement still differs and has not been changed.

Loss scalars now average all updates in each 20-phase reporting interval,
then average across ranks. Evaluation angle/velocity/acceleration metrics use
explicit radian-based units and temporal differences; new CSV output is
`tracking_eval_v2.csv`. Old metric history is not retroactively corrected.
The W&B mirror now follows every event file. The stopped run was backfilled
through global step 106422272.

The first observed NaN was in simulator root/joint state at env iteration
13008, shortly after the synchronized reset at 13000. Dataset arrays and all
1886 floating tensors in the last saved checkpoint were finite. This does
not establish the exact solver failure or exclude later optimizer failures.
Rollout non-finite checks now stop all ranks together and save before/after
physics snapshots instead of leaving peers waiting for gradient collectives.

Validation: 56 UFO/PUFO tests, 11 runner tests (including a two-process
non-finite propagation test), and one TensorBoard mirror test passed locally.
On H200-2, a 16-environment diagnostic passed 1100 control steps / 4400 physics
substeps, with reset-frame checks at initialization and steps 500 and 1000.
A separate 520-step, 16-environment training smoke test completed optimizer
updates, recorded finite losses, crossed a timeout reset and saved
`model_8192.pt`. This is not an eight-GPU long-run stability guarantee.
Diagnostic logs are `migration/reset_contract_diagnostic.log` and
`migration/resetfix_update_smoke.log` under `/data/zym/roboparty_train`.

## Layout

- Code, local edits, IsaacLab SDK and RSL-RL: `/home/zym/roboparty_train`
- Relocated training environment: `/data/zym/conda/envs/isaaclab_zym`
- All RoboLab motions and robot assets: `/data/zym/roboparty_train/robo_newton_data`
- Legacy source archives: `/data/zym/roboparty_train/archives`
- Logs, checkpoints, caches, W&B files and temporary files: `/data/zym/roboparty_train`

The corresponding project directories are symlinks into the data disk.
The unused original IsaacLab `.venv` is not duplicated; its path links to the
relocated training environment. Old training logs, checkpoints and compilation
caches are not migrated. The environment archive has its installation prefix
rewritten for the exact destination above; do not extract it elsewhere.

## Activate

```bash
cd /home/zym/roboparty_train
source ./activate_h2002.sh
python --version
```

The activation script routes generated training files and caches to `/data`.
It does not change shell startup files or start training. Credentials are not
copied from the source machine; configure W&B login on H200-2 separately if needed.

## First-Launch License Confirmation

The installed Isaac Sim / Omniverse package asks for NVIDIA's EULA confirmation
on first import, including when the training launcher uses the Newton backend.
The user accepted the agreement and authorized confirmation on 2026-10-03.
`activate_h2002.sh` now sets `OMNI_KIT_ACCEPT_EULA=YES` for this environment.
To check the package import after activation:

```bash
python -c "import isaacsim"
```

Acceptance is supplied through the environment variable, not a marker file.

## Start a Fresh Run Manually

Check GPU availability first; the migration does not reserve GPUs. In a tmux
session, after activation, run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
  --standalone --nproc_per_node=8 robo_newton/scripts/rsl_rl/train.py \
  --task RP1-PUFO --num_envs 1024 --device cuda --distributed \
  --visualizer none --logger tensorboard --run_name resetfix_h2002_8gpu1024_20261003
```

Do not add `--resume` for a fresh run. Each rank has 1024 environments.
Reset joint positions and velocities come directly from the sampled motion
frame, without added joint noise; the default reference positions remain unchanged.
Other training settings and the synchronized episode timeout are unchanged.

## CPU Regression Checks

```bash
python -m unittest \
  robolab.tasks.manager_based.ufo.tests.test_reset_contract \
  robolab.tasks.manager_based.ufo.tests.test_action_config \
  robolab.tasks.manager_based.ufo.tests.test_motion_indexing -v
```

## Migration Validation

The environment archive SHA-256 matched on both hosts. Code and dataset contents
were compared with rsync checksums. On H200-2, Torch CUDA matrix multiplication,
Torch compilation, a Warp CUDA kernel, asset/configuration imports, and all
13 regression tests above passed.

The initial simulation check stopped at the license prompt. After authorization,
a 16-environment test reached solver initialization, then was interrupted at the
user's request to launch eight-GPU training directly. The short test therefore
did not validate optimizer updates. Validation logs are in
`/data/zym/roboparty_train/migration/`.

The eight-GPU run uses tmux session `rp1-pufo-h2002`, 1024 environments per GPU,
TensorBoard logging, and no resume. Its console log is
`/data/zym/roboparty_train/logs/launch/rp1_pufo_8gpu1024_20261003.log`.
Inspect the live log for current initialization and training status.
The run reached 253,952 environment steps and 20 update phases with finite
reported critic and actor losses. This verifies startup and initial updates,
not long-run convergence or stability.

## Live W&B Mirror

The running trainer still writes TensorBoard locally. A separate CPU-only process
in tmux session `rp1-pufo-wandb-sync` backfills and follows its scalar metrics at
https://wandb.ai/dtu-yiming/rp1_pufo/runs/tb4715dd3b7b634c without restarting training.
The mirror keeps original global environment steps and uploads the parameter
YAML files. It does not upload model checkpoints or evaluation videos.

The latest metric step is held until the next step arrives so partially written
scalar groups are not committed early. On normal training exit it drains the last
step and finishes. Console output is in
`/data/zym/roboparty_train/logs/launch/rp1_pufo_wandb_sync_20261003.log`.
To restart only the mirror, first ensure no other copy is running, activate the
environment, and use the current torchrun PID:

```bash
  --run-dir /data/zym/roboparty_train/logs/rsl_rl/rp1_pufo/2026-10-03_01-03-58_resetfix_h2002_8gpu1024_20261003 \
  --entity dtu-yiming --project rp1_pufo --training-pid 4151599
```

The run ID is derived from the source directory, and a restarted mirror resumes
from the cloud run's step. Credentials remain in the existing H200-2 login store.
