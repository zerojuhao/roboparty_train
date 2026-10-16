# roboparty_train

[English](README.md) | [中文](README_CN.md)

面向 RP1、RPO 的策略训练仓库，使用 Isaac Lab 3.0、Newton/MJWarp 和本地修改的 RSL-RL，包含 AMP、Parkour/SSR、UFO/PUFO 和 BeyondMimic。

## 仓库结构

```text
robo_newton/robolab/       机器人配置、任务、传感器和工具函数
robo_newton/scripts/       训练、测试、导出和诊断工具
rsl_rl/rsl_rl/             修改后的算法和 runner
rsl_rl/tests/              训练库测试
robo_newton/data/          本地机器人资产与动作数据，不纳入 Git
```

源码目录由本仓库直接管理，不是 Git submodule。Isaac Lab 单独安装；`IsaacLab/`、`IsaacLab-3.0/`、`project_ops/`、日志、checkpoint 和本地运行数据均不上传。当前仓库尚未包含动作数据的发布副本。

## 环境与安装

当前环境使用 Linux、NVIDIA GPU、Python 3.12 和支持 Newton 的 Isaac Lab 3.0 源码。参见 [SDK 官方安装指南](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/setup/installation/index.html)。本仓库不附带或锁定 SDK 源码版本，不能保证任意更新的上游版本都直接兼容。

[requirements-physics.txt](robo_newton/requirements-physics.txt) 固定 MuJoCo 3.12.0、mujoco-warp 3.12.0、Warp 1.17.0、Newton 1.6.0，升级前需执行碰撞回归测试。RSL-RL 必须使用本仓库的修改版本。

在已激活的 Isaac Lab Python 环境中执行：

```bash
git clone https://github.com/zerojuhao/roboparty_train.git
cd roboparty_train
python -m pip install -r robo_newton/requirements-physics.txt
python -m pip install -e ./robo_newton -e ./rsl_rl
export PYTHONPATH="$PWD/rsl_rl:$PWD/robo_newton${PYTHONPATH:+:$PYTHONPATH}"
export ROBO_TRAIN_LOG_ROOT="$PWD/logs/rsl_rl"
```

私有仓库需要 GitHub 访问权限。导入任务配置或训练前，需要准备下面的机器人资产和动作数据。

## 统一数据集

`robo_newton/data` 可以链接到本地数据盘，默认目录为：

```text
robo_newton/data/
  robots/roboparty/rp1.3/  RP1 URDF、MJCF、网格
  robots/roboparty/rpo/    RPO URDF、MJCF、网格
  motions/newton/
    rp1/                  RP1 AMP、Parkour、SSR
    rp1_upper_run_bounded/ RP1 AMP 上半身参考动作
    rpo/                  RPO AMP、Parkour、BeyondMimic
    rp1_lafan/            UFO/PUFO 的 manifest.json 和 npz/
    conversion_report.json
```

训练数据统一为 `roboparty_motion_v1` NPZ，使用 `allow_pickle=False` 读取。元数据包含 `format`、`fps`、`quaternion_order="xyzw"`、`joint_order="urdf"`、`joint_names`。单位四元数为 `[0, 0, 0, 1]`。

关节按 URDF 中非固定关节的声明顺序排列，不依赖 PhysX 或求解器的隐式顺序。机器人配置明确采用相同顺序，加载器按名字校验并在启动时映射。

| 数据类型 | 数组字段 |
| --- | --- |
| 公共字段 | `joint_pos` |
| AMP | `root_pos_w`、`root_quat_w`、`key_body_pos_w`、`key_body_names`、`loop_mode` |
| 全身跟踪 | `joint_vel`、`body_names`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w` |

位置单位为米，关节角度为弧度，世界坐标系速度保留来源语义。全身数据的第一个身体为 `base_link`。AMP 保留原有前向差分速度算法和关键身体位置，不凭空补全缺失的全身轨迹。转换不重新重定向、不重采样、不平滑动作；NPZ 压缩无损，转换报告记录来源与输出的哈希。

UFO/PUFO 默认读取 `motions/newton/rp1_lafan/manifest.json`，可用 `UFO_MOTION_MANIFEST` 指定其他统一格式数据。后缀同为 `.npz` 不代表四元数和关节排列兼容。

请使用外部工具完成动作重定向，并在训练前准备符合上述规范的 NPZ 数据。历史重定向、CSV/PKL 处理和一次性迁移脚本已移除。训练运行时的数据校验与按名称映射仍由 [motion_npz.py](robo_newton/robolab/utils/motion_npz.py) 提供；该模块不会重新重定向、补全缺失轨迹或猜测来源数据的排列约定。

当前精简后的数据集共 125 段动作，总计约 752 MiB：

| 目录 | 动作数 | NPZ 体积 |
| --- | ---: | ---: |
| `rp1` | 36 | 2.1 MiB |
| `rpo` | 33 | 6.0 MiB |
| `rp1_upper_run_bounded` | 1 | 小于 0.1 MiB |
| `rp1_lafan/npz` | 55 | 743.5 MiB |

这里统计的是本地训练数据包，不代表源码仓库已经附带这些文件。训练不需要原始动作、中间 PKL 或重复备份；NPZ 旁仍需保留 LAFAN 清单及相关来源记录。机器人资产另行准备。

动作数据不自动适用源码许可证。[Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) 使用 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) 许可。重定向不会消除上游许可限制，发布修改后的动作数据前需确认再分发权利；其他目录中的 LAFAN 衍生片段同样需要确认。

## 训练

从仓库根目录运行，主要训练入口是 `robo_newton/scripts/rsl_rl/train.py`。常用任务：`RP1-AMP`、`RP1-AMP-Path`、`RP1-Parkour`、`RP1-SSR`、`RP1-UFO`、`RP1-PUFO`、`RPO-AMP`、`RPO-Parkour`。

```bash
python robo_newton/scripts/rsl_rl/train.py \
  --task RP1-AMP-Path --num_envs 1024 --device cuda:0 \
  --visualizer none --logger tensorboard
```

多 GPU 示例，`num_envs` 是每个进程的环境数：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  robo_newton/scripts/rsl_rl/train.py --task RP1-PUFO \
  --num_envs 1024 --device cuda --distributed --visualizer none --logger tensorboard
```

兼容的训练使用 `--resume --load_run RUN --checkpoint model_N.pt` 续训。URDF 顺序迁移会改变 AMP 判别器输入，包括上半身输入。旧 AMP/Parkour/SSR/BeyondMimic checkpoint 在显式迁移前会被 runner 拒绝加载，不能仅凭张量维度相同就直接续训。新规则应从新训练开始；已运行进程不会自动迁移。

RP1 UFO 与 PUFO 共用筛选后的 40 条 LAFAN 动作。UFO 仍是无视觉的平面任务，但现在使用统一的地形接触奖励、本体偏航坐标系速度估计器以及 heading/velocity context critic；PUFO 在此基础上增加深度图和地形分支。

## 测试与导出

使用绝对 checkpoint 路径，避免日志根目录歧义：

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

PUFO 导出还需要对应训练配置和动作数据。MuJoCo 验证脚本位于 `robo_newton/scripts/mujoco/`，应选择与策略任务及部署接口匹配的入口。

PUFO sim2sim 资源位于 `robo_newton/data/robots/roboparty/rp1.3/mjcf/`：`rp1_simple_terrain/` 保存地形，`robot_24dof_newton_physics.json` 保存非随机化模型编译后的碰撞、接触和求解器参数。公共工具 `robolab.utils.mujoco_physics` 不加载 checkpoint；电机限幅由 URDF 和传入的执行器配置解析。其他任务可使用同一工具加载自己的物理资产，不会自动套用 PUFO 参数。`sim2sim_rp1_pufo.py --physics-asset PATH` 可选择其他资产。可从非随机化、单机器人加地面的 Newton 模型执行 `scripts/tools/export_mujoco_physics.py --model MODEL.mjb --urdf ROBOT.urdf --output ASSET.json --physics-dt SECONDS` 重新生成。这些文件属于本地数据，需随机器人资产分发。

地形资产 `rp1_simple_terrain/rp1_simple_terrain.xml` 与 `rp1_simple_terrain/meta.json` 是预生成的，旧的服务器专用地形导出脚本已移除。地形元数据提供重置位置和地形描述，`robot_24dof_newton_physics.json` 提供 Newton 的碰撞、接触和求解器参数；两个 JSON 都是 PUFO sim2sim 所需文件，使用缩进格式保存。上述物理导出工具只生成物理 JSON，不生成地形。使用时需从机器人资源包安装两个地形文件；运行 sim2sim 不依赖外部 UFO 代码目录。

Isaac Lab 3 中，`effort_limit_sim` 是仿真关节限幅的旧字段，并不直接等同于显式执行器内部的扭矩限幅；需要内部限幅时应明确配置 `actuator_effort_limit`。sim2sim 保留旧配置的实际解析语义，本次资源整理不修改训练电机参数。

RP1 的 AMP、Parkour、PUFO 三个 sim2sim 入口统一使用 `robolab.utils.mujoco_actuators.RoboPartyMujocoActuator`，实际实例化训练端的 `RoboPartyActuator`，复用 PD、扭矩随转速衰减及制动逻辑，脚本和物理资产不再写死电机扭矩。AMP/PUFO 使用训练运行保存的机器人配置；Parkour 可用 `--env-config params/env.yaml`，否则读取 `--checkpoint` 所属运行的配置，均未指定时使用当前 `roboparty.py` 默认值。标称评估关闭执行器随机延迟和随机前馈扭矩。

## 脚本用途

下表路径相对于 `robo_newton/scripts/`。独立诊断工具没有被训练代码导入，不代表没有用途。这里说明用途和维护建议，不代表所有旧 GUI 入口都已在当前 Newton 环境验证通过。

| 入口 | 用途与维护建议 |
| --- | --- |
| `rsl_rl/train.py`、`rsl_rl/play*.py`、`rsl_rl/export_pufo_deploy.py` | 主要训练、各任务评估和策略导出入口，保留。 |
| `smoke_newton_env.py` | 不启动 PPO，直接创建、重置、步进环境，检查观测和奖励是否有限；还可检查 Parkour 碰撞体、PUFO 接触和单帧深度回放重建。建议保留用于回归。 |
| `tools/export_mujoco_physics.py` | 从标称编译模型重新生成碰撞、接触和求解器 JSON。不是每次运行都调用，但更新物理资产时仍需要，建议保留。 |
| `tools/list_envs.py` | 列出注册任务 ID，有用，但当前会启动 AppLauncher。 |
| `tools/visualize_robot.py`、`tools/visualize_terrain.py`、`tools/visualize_depth_domain_randomization.py` | 检查机器人关节/身体顺序、地形和相机噪声，作为可选诊断工具保留；目前使用 AppLauncher 场景。 |
| `bind_gpu_local_rank.py` | 可选 torchrun 包装器，将各进程绑定到对应 GPU 附近的 CPU；依赖显式 `CUDA_VISIBLE_DEVICES` 和 Linux NUMA 拓扑。 |
| `train_newton.py`、`play_newton.py` | Isaac Lab CLI 的轻量包装，不等同于完整的自定义 runner 入口。`smoke_newton_env.py` 会导入 `train_newton.py` 完成无 Kit 初始化，不能单独删除。 |
| `mujoco/sim2sim_*.py` | 各任务部署验证入口，观测和动作接口不同。保留与策略匹配的版本，旧 RPO 变体需单独核对兼容性后再合并。 |

动作加载和 RP1 sim2sim 的 CPU 回归测试：

```bash
MUJOCO_GL=egl python -m pytest -q robo_newton/robolab/utils/tests
```

sim2sim 集成测试需要本地机器人资产，缺少动作数据时部分检查会跳过。GPU 环境冒烟测试与上述单元测试分开运行：

```bash
python robo_newton/scripts/smoke_newton_env.py \
  --task RP1-AMP --num_envs 2 --steps 4 --device cuda:0
```

## 参考项目

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Newton](https://github.com/newton-physics/newton)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [MuJoCo](https://github.com/google-deepmind/mujoco)
