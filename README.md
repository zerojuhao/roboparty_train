# roboparty_train

[English](README.md) | [中文](README_CN.md)

RP1 and RPO policy training with Isaac Lab 3.0, Newton/MJWarp, and a locally modified RSL-RL. Tasks include AMP, Parkour/SSR, UFO/PUFO, and BeyondMimic.

## Repository

```text
robo_newton/robolab/       Robot definitions, tasks, sensors, and utilities
robo_newton/scripts/       Training, evaluation, export, and diagnostic tools
rsl_rl/rsl_rl/             Modified algorithms and runners
rsl_rl/tests/              Training-library tests
robo_newton/robolab/data/  Local robot assets and unified motion data
```

Source directories are ordinary tracked directories, not Git submodules. Isaac Lab is installed separately. `IsaacLab/`, `IsaacLab-3.0/`, `project_ops/`, logs, checkpoints, and the local data bundle are excluded from Git. The robot assets and unified training motions are kept together under `robo_newton/robolab/data/` for each checkout.

## Installation

The current environment uses Linux, NVIDIA GPUs, Python 3.12, and a Newton-capable Isaac Lab 3.0 checkout. Follow the [SDK installation guide](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/setup/installation/index.html). This repository does not pin or distribute the SDK checkout; arbitrary newer upstream revisions are not guaranteed compatible.

[requirements-physics.txt](robo_newton/requirements-physics.txt) pins MuJoCo 3.12.0, mujoco-warp 3.12.0, Warp 1.17.0, and Newton 1.6.0. Do not upgrade without collision regressions. Use this repository's RSL-RL fork.

In the activated Isaac Lab Python environment:

```bash
git clone https://github.com/zerojuhao/roboparty_train.git
cd roboparty_train
python -m pip install -r robo_newton/requirements-physics.txt
python -m pip install -e ./robo_newton -e ./rsl_rl
export PYTHONPATH="$PWD/rsl_rl:$PWD/robo_newton${PYTHONPATH:+:$PYTHONPATH}"
export ROBO_TRAIN_LOG_ROOT="$PWD/logs/rsl_rl"
```

The repository includes the robot assets and unified datasets below. A checkout therefore has the expected runtime paths before importing task configurations or starting training.

## Motion Data

`robo_newton/robolab/data` is the checkout-local data root. The unified NPZ bundle and robot resources are stored there. Data lookup order is `--data_root`, `ROBO_TRAIN_DATA_ROOT`, then this local data root.

```text
robo_newton/robolab/data/
  robots/rp1/  RP1 URDF, MJCF, and meshes
  robots/rpo/    RPO URDF, MJCF, and meshes
  motions/
    rp1/                  RP1 AMP, Parkour, and SSR
    rp1_upper_run_bounded/ RP1 AMP upper-body reference
    rpo/                  RPO AMP, Parkour, and BeyondMimic
    rp1_lafan/            UFO/PUFO manifest.json and npz/
    conversion_report.json
```

All training clips use `roboparty_motion_v1` NPZ with `allow_pickle=False`. Metadata includes `format`, `fps`, `quaternion_order="xyzw"`, `joint_order="urdf"`, and `joint_names`. Identity rotation is `[0, 0, 0, 1]`. Joint columns follow non-fixed URDF declaration order; robot configurations explicitly select this order. Loaders validate names and map columns once at startup.

| Track | Arrays |
| --- | --- |
| Common | `joint_pos` |
| AMP | `root_pos_w`, `root_quat_w`, `key_body_pos_w`, `key_body_names`, `loop_mode` |
| Full-body | `joint_vel`, `body_names`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w` |

Positions are metres, joint angles radians, and world-space velocities retain source semantics. Full-body tracks put `base_link` first. AMP retains its forward-difference velocity calculation and key-body measurements. Conversion does not invent missing body tracks, retarget, resample, or smooth motions. Compression is lossless; the conversion report records source/output hashes.

UFO/PUFO defaults to `motions/rp1_lafan/manifest.json`; override with `UFO_MOTION_MANIFEST`. A `.npz` extension alone does not establish compatible conventions.

Retarget motions with external tools and prepare NPZ files that follow the schema above before training. Legacy retargeting, CSV/PKL preparation, and one-time migration scripts have been removed. Runtime validation and name-based mapping remain in [motion_npz.py](robo_newton/robolab/utils/motion_npz.py); this module does not retarget motions or infer missing tracks or source conventions.

The current slimmed local dataset contains 125 clips, approximately 752 MiB in total:

| Directory | Clips | NPZ Size |
| --- | ---: | ---: |
| `rp1` | 36 | 2.1 MiB |
| `rpo` | 33 | 6.0 MiB |
| `rp1_upper_run_bounded` | 1 | Less than 0.1 MiB |
| `rp1_lafan/npz` | 55 | 743.5 MiB |

These files belong under `robo_newton/robolab/data/motions/`. Original motions, intermediate PKLs, and duplicate backups are not needed. Keep the LAFAN manifest and relevant provenance records alongside the NPZs. Robot assets belong under the sibling `robots/` directory.

Motion data does not inherit the source-code license. [Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) is released under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). Retargeting does not remove upstream restrictions; confirm redistribution rights before publishing modified motion data. This also applies to LAFAN-derived clips stored in other folders.

## Training

Use `robo_newton/scripts/train_newton.py` as the portable entry point from any working directory. It dispatches to the RSL-RL `train.py` backend and sets project/data paths. Common task IDs: `RP1-AMP`, `RP1-AMP-Path`, `RP1-Parkour`, `RP1-SSR`, `RP1-UFO`, `RP1-PUFO`, `RPO-AMP`, `RPO-Parkour`.

```bash
python robo_newton/scripts/train_newton.py --rl_library rsl_rl \
  --task RP1-AMP-Path --num_envs 1024 --device cuda:0 \
  --visualizer none --logger tensorboard
```

Multi-GPU example (`num_envs` per worker):

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  robo_newton/scripts/train_newton.py --rl_library rsl_rl --task RP1-PUFO \
  --num_envs 1024 --device cuda --distributed --visualizer none --logger tensorboard
```

Resume compatible runs with `--resume --load_run RUN --checkpoint model_N.pt`. The URDF migration changes AMP discriminator inputs, including upper-body inputs. Legacy AMP/Parkour/SSR/BeyondMimic checkpoints are rejected until explicitly migrated; equal tensor dimensions do not imply compatible joint semantics. Start a new run for the new convention. Running processes are not migrated automatically.

RP1 UFO and PUFO use the same selected 40-clip LAFAN motion subset. UFO remains a flat, non-visual task, but now uses the shared terrain-contact reward terms, proprioceptive yaw-frame velocity estimator, and heading/velocity context critics; PUFO adds the depth and terrain branches.

## Evaluation and Export

Use absolute checkpoint paths to avoid ambiguity between log roots:

```bash
python robo_newton/scripts/rsl_rl/play_amp.py \
  --task RP1-AMP-Path-Play --num_envs 1 \
  --checkpoint /path/to/run/model_N.pt --visualizer none

python robo_newton/scripts/rsl_rl/play_parkour.py \
  --task RP1-Parkour-Play --num_envs 1 \
  --checkpoint /path/to/run/model_N.pt --exportonnx --visualizer none

python robo_newton/scripts/rsl_rl/export_pufo_deploy.py \
  --load_run /path/to/pufo_run --checkpoint model_N.pt --motion dance1_subject2
```

PUFO export also needs saved run configuration and motion data. MuJoCo checks live under `robo_newton/scripts/mujoco/`; select the entry point and deployment contract matching the policy.

PUFO sim2sim assets live in `robo_newton/robolab/data/robots/rp1/mjcf/`: `rp1_simple_terrain/` contains the terrain, and `robot_24dof_newton_physics.json` contains nominal compiled collision/contact/solver data. The shared `robolab.utils.mujoco_physics` utility applies these assets without loading a checkpoint; motor limits are resolved from the URDF and supplied actuator configuration. Other tasks can use their own assets through the same utility, not inherit PUFO parameters implicitly. `sim2sim_rp1_pufo.py --physics-asset PATH` selects an alternative asset. Regenerate assets with `scripts/tools/export_mujoco_physics.py --model MODEL.mjb --urdf ROBOT.urdf --output ASSET.json --physics-dt SECONDS` from a nonrandomized, single-robot Newton model with a ground plane.

The terrain assets `rp1_simple_terrain/rp1_simple_terrain.xml` and `rp1_simple_terrain/meta.json` are pre-generated; the old machine-specific terrain exporter has been removed. The metadata supplies reset positions and terrain descriptions, while `robot_24dof_newton_physics.json` supplies Newton collision, contact, and solver parameters. Both JSON files are required by PUFO sim2sim and use indented formatting. The physics exporter above regenerates only the physics JSON, not the terrain. Install both terrain files from the robot asset bundle; an external UFO checkout is not needed to run sim2sim.

With Isaac Lab 3, `effort_limit_sim` is a solver-joint limit alias, not the explicit actuator's internal torque limit. Set `actuator_effort_limit` explicitly when that is intended. Sim2sim preserves this distinction for saved configurations; this asset reorganization does not change training motor settings.

All RP1 sim2sim entry points (AMP, Parkour, PUFO) use `robolab.utils.mujoco_actuators.RoboPartyMujocoActuator`, which instantiates the actual training `RoboPartyActuator` for PD, torque-speed rolloff and braking. No motor torque values are embedded in these scripts or physics assets. AMP/PUFO use the run's saved robot configuration; Parkour accepts `--env-config params/env.yaml`, otherwise uses `--checkpoint`'s run configuration or the current `roboparty.py` defaults. Nominal evaluation disables actuator delay and random feed-forward torque.

## Script Guide

Paths below are relative to `robo_newton/scripts/`. Standalone diagnostic tools need not be imported by training to remain useful. This inventory documents their roles, not a claim that every legacy GUI entry point has been validated on the current Newton stack.

| Entry | Purpose and Status |
| --- | --- |
| `rsl_rl/train.py`, `rsl_rl/play*.py`, `rsl_rl/export_pufo_deploy.py` | Main training, task-specific evaluation, and policy export. Keep. |
| `smoke_newton_env.py` | Creates, resets, and steps an environment without PPO; checks finite outputs and optional Parkour colliders, PUFO contacts, and compact depth replay. Keep for regressions. |
| `tools/export_mujoco_physics.py` | Regenerates collision/contact/solver JSON from a nominal compiled model. Not required on each run, but needed after physics-asset changes. Keep. |
| `tools/list_envs.py` | Lists registered task IDs. Useful, though currently starts AppLauncher. |
| `tools/visualize_robot.py`, `tools/visualize_terrain.py`, `tools/visualize_depth_domain_randomization.py` | Inspect robot joint/link order, terrain, and camera noise. Keep as optional diagnostics; these use AppLauncher-based scenes. |
| `bind_gpu_local_rank.py` | Optional torchrun wrapper binding each worker to CPUs local to its GPU. Requires explicit `CUDA_VISIBLE_DEVICES` and Linux NUMA topology. |
| `train_newton.py`, `play_newton.py` | Thin Isaac Lab CLI wrappers, not equivalents of all custom runner features. `smoke_newton_env.py` imports `train_newton.py` for kitless initialization; do not remove it in isolation. |
| `mujoco/sim2sim_*.py` | Deployment checks with task-specific observation/action contracts. Keep the entry matching the policy; older RPO variants need a separate compatibility audit before consolidation. |

CPU regression tests for motion loading and RP1 sim2sim:

```bash
MUJOCO_GL=egl python -m pytest -q robo_newton/robolab/utils/tests
```

The sim2sim integration tests require local robot assets; some motion checks skip if the data bundle is absent. A GPU environment smoke test is separate from these unit tests:

```bash
python robo_newton/scripts/smoke_newton_env.py \
  --task RP1-AMP --num_envs 2 --steps 4 --device cuda:0
```

## References

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Newton](https://github.com/newton-physics/newton)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [MuJoCo](https://github.com/google-deepmind/mujoco)
