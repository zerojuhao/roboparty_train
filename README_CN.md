# ATOM01-Train

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RL](https://img.shields.io/badge/RSL_RL-3.3.0-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

[English](README.md) | [中文](README_CN.md)

## 概述

本仓库提供了使用 IsaacLab 训练足式机器人的工作流。它提供环境的高透明度和低重构难度，并使用 isaaclab 组件简化流程。代码库基于 IsaacLab 构建，支持 Sim2Sim 传输到 MuJoCo，并具有模块化架构，便于无缝定制和扩展。

**维护者**: 刘志浩
**联系方式**: ZhihaoLiu_hit@163.com

**主要特性:**

- `易于重组` 提供直接工作流程，允许精细定义环境逻辑。
- `隔离性` 在核心 Isaac Lab 仓库之外工作，确保开发工作保持自包含。
- `长期支持` 本仓库将随着 isaac sim 和 isaac lab 的更新而更新，并将长期支持。



## 安装

ATOM01-Train 基于最新版本的 IsaacSim/IsaacLab 构建，建议跟随 ATOM01-Train 的最新更新。

- 按照[安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)安装 Isaac Lab。我们推荐使用 conda 安装，因为它简化了从终端调用 Python 脚本的过程。

- 独立于 Isaac Lab 安装克隆此仓库（即在 `IsaacLab` 目录之外）：

```bash
git clone https://github.com/Roboparty/atom01_train.git
```

- 使用已安装 Isaac Lab 的 python 解释器安装库：

```bash
cd atom01_train
git submodule update --init --recursive
cd robolab
pip install -e .
cd ..
cd rsl_rl
pip install -e .
cd ..
```

- 通过运行以下命令验证扩展是否正确安装，打印可用环境列表：

```bash
python robolab/scripts/tools/list_envs.py
```

## 用法

### 训练
```bash
python robolab/scripts/rsl_rl/train.py --task=<ENV_NAME> --headless --logger=tensorboard --num_envs=8192
```

### 测试
```bash
python robolab/scripts/rsl_rl/play.py --task=<ENV_NAME> --num_envs=1
```
### 测试(AMP)
```bash
python robolab/scripts/rsl_rl/play_amp.py --task=Atom01-AMP-Play --num_envs=1
```
### 测试(Beyondmimic)
```bash
python robolab/scripts/rsl_rl/play_bm.py --task=Atom01-BeyondMimic --num_envs=1
```
### 测试(Parkour)
```bash
python robolab/scripts/rsl_rl/play_parkour.py --task=Atom01-Parkour-Play --num_envs=1
```
导出 onnx 模型时, 请设置 `num_envs=1` 并添加 `--exportonnx`
```bash
python robolab/scripts/rsl_rl/play_parkour.py --task=Atom01-Parkour-Play --num_envs=1 --exportonnx
```

### Sim2Sim
```bash
python robolab/scripts/mujoco/sim2sim_atom01.py --load_model "{exported/policy.pt model full path here}"
```

### 数据集准备
对于 AMP 和 BeyondMimic 获取数据集的工作流程, 请查看 [GMR](https://github.com/Roboparty/GMR).

通过GMR获得的数据集中的关节顺序与机器人的 URDF 和 XML 中的顺序一致，但这与 Isaac Lab 所使用的顺序不同。因此，我们需要准备一个包含关节映射信息的`.yaml`文件（如`scripts/tools/retarget/config/atom01.yaml`所示），并在训练前使用`scripts/tools/retarget/dataset_retarget.py`重新排序关节序列。


## 参考和致谢
本项目仓库建立在巨人的肩膀上。
* [IsaacLab](https://github.com/isaac-sim/IsaacLab)
* [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
* [legged_gym](https://github.com/leggedrobotics/legged_gym)
* [legged_lab](https://github.com/zitongbai/legged_lab)
* [robot_lab](https://github.com/fan-ziqi/robot_lab)
* [InstinctLab](https://github.com/project-instinct/InstinctLab)