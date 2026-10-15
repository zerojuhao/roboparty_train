# PBFM 部署 Context 对齐修改方案

## 目标

让另一套实机部署代码严格匹配当前 `context_1.npz`、P3 tracking 导出逻辑和训练时的 actor 输入契约，避免对已经平滑过的 tracking context 再次做时间平滑。

## 已确认的 Context 契约

当前导出的 context 不是单帧原始 `B(s_t)`，而是 tracking sliding context：

```text
z_raw[t] = B(next_obs[t])
z_context[t] = project_z(mean(z_raw[t : t + 8]))
```

由于 tracking 导出使用 `backward_obs[1:]`，从动作时间索引看，它等价于：

```text
context[t] = project_z(mean(B(frame[t+1]), ..., B(frame[t+8])))
```

最后不足 8 帧时使用剩余帧。每个时间索引仍然有一行 `z`，但每一行已经包含未来 8 帧的信息。

当前 context 文件应满足：

```text
z:                              float32 [T, 256]
heading_relative_xy:            float32 [T, 2]
heading_valid:                  uint8   [T, 1]
reference_velocity_heading0_xy: float32 [T, 2]
velocity_valid:                 uint8   [T, 1]
```

`z` 在 `project_z` 后通常接近范数 `sqrt(256) = 16`。不要因为范数约为 16 就把它当成未平滑的单帧 latent。

## 必须修改的部署逻辑

### 1. 禁止二次 latent 平滑

如果部署端已有如下逻辑：

```cpp
weighted_latent_window(context.z, frame, gamma, window_size, actor_z);
```

则对当前 context 必须使用：

```yaml
window_size: 1
```

或者直接等价为：

```cpp
copy(context.z[frame], actor_z);
```

`gamma` 在 `window_size=1` 时不影响结果，可以保留原值，但不能再使用 `window_size=3/5` 等窗口。当前旧部署的 `gamma=0.8, window_size=3` 会对已经经过 8 帧 tracking 平均的 `z` 再平滑一次。

建议同时把 C++ 参数默认值也改成 1，避免配置缺失时回退到旧值：

```cpp
size_t window_size = 1;
```

配置文件和 C++ 默认值都应保持一致。

### 2. frame 索引必须直接对齐

播放第 `frame` 个动作时使用：

```text
actor_z = context.z[frame]
heading = context.heading_relative_xy[frame]
velocity = context.reference_velocity_heading0_xy[frame]
```

不要在部署端额外使用 `frame + 1`，也不要把 context 当作原始 B 输出后再自行向前偏移。`[1:]` 的未来帧偏移已经在导出阶段完成。

### 3. standing 状态和播放状态分开

未开始播放、`frame == 0` 时继续使用 `standing_z`，不要使用 tracking context：

```text
holding = (!playing && frame == 0)
holding -> actor_z = standing_z
playing -> actor_z = context.z[frame]
```

按下播放/开始 tracking 的瞬间：

1. 从 IMU 当前四元数计算当前机体 heading。
2. 将它保存为 `start_heading`。
3. 从第 0 个 context frame 开始推理。

## Heading 和 Velocity 输入契约

### current heading

`current_heading` 不是导出时固定的值，而是每个控制周期从实机 IMU 四元数计算得到的当前机体朝向。

### target heading

`context.heading_relative_xy[frame]` 是相对于导出时参考 heading 的二维单位向量。播放开始时将它旋转到实机的 `start_heading`：

```text
target_world = rotate_xy(heading_relative_xy[frame], start_heading)
```

然后按 actor 所需的当前机体坐标系计算：

```text
heading_input[0] = 1 - dot(current_heading, target_world)
heading_input[1] = cross2d(current_heading, target_world)
```

因此刚开始播放时，如果 context 的第 0 帧目标方向与 `start_heading` 对齐，送入 actor 的 heading input 应接近 `[0, 0]`。这不表示世界坐标中的 `current_heading` 必须为 0。

### velocity

`reference_velocity_heading0_xy[frame]` 同样先使用 `start_heading` 变换到世界坐标，再使用当前 IMU heading 变换到当前机体坐标：

```text
velocity_world = rotate_xy(reference_velocity_heading0_xy[frame], start_heading)
velocity_input[0] = dot(current_heading, velocity_world)
velocity_input[1] = cross2d(current_heading, velocity_world)
velocity_input[2] = velocity_valid[frame] ? 1 : 0
```

当 `velocity_valid == 0` 时，前两个 velocity 分量必须清零。

## 不要做的修改

- 不要把当前 `z` 再做 3 帧或 5 帧滑动平均。
- 不要把 `heading_relative_xy` 直接当作当前机体坐标输入。
- 不要把 `reference_velocity_heading0_xy` 直接当作当前机体 velocity 输入。
- 不要把 `current_heading` 在每帧重置为 0；只在播放开始时捕获 `start_heading`。
- 不要把 `standing_z` 和 tracking `z` 混用。
- 不要用原始单帧 `B(s_t)` 的导出方案配合当前已经导出的 context；两种方案的部署窗口配置不同。

## 推荐修改顺序

1. 加载 context 时检查五个字段、shape、dtype、帧数一致性和有限值。
2. 将 latent temporal window 设置为 `1`，并将 C++ 默认值同步为 `1`。
3. 确认播放帧使用同一个 `frame` 索引读取 z、heading、velocity。
4. 保留 `standing_z` 的启动保持逻辑。
5. 在播放开始瞬间捕获 IMU `start_heading`，之后保持该 anchor 不变。
6. 按上面的公式转换 heading 和 velocity。
7. 检查 actor 输入顺序仍为：

   ```text
   dof_pos(24), dof_vel(24), gravity_b(3), ang_vel(3),
   last_action(24), perception(256), z(256), heading(2), velocity(3)
   ```

## 验证计划

### 离线验证

- `window_size=1` 时，部署侧 `actor_z` 与 `context.z[frame]` 逐元素一致。
- `window_size=3` 时应明确检测为不符合当前 context 契约，不能默默运行。
- 首帧 context、heading、velocity 的帧数和索引一致。
- 所有 actor 输入均为有限值。
- 最后一帧不会越界。

### 仿真验证

- 使用同一个 `pbfm_actor.onnx` 和 `context_1.npz`。
- sim2sim 直接使用 `context.z[frame]`，不要再平滑。
- 分别测试从 motion pose 启动和从实机默认关节姿态启动；后者更接近实机。
- 记录首帧 actor 输入、首帧 action、重力向量和 IMU heading。

### 实机验证

至少记录以下字段：

```text
frame, playing, current_heading, start_heading,
context_z_norm, actor_z_norm,
heading_input, velocity_input,
standing_z_used, action
```

重点确认：

- 未播放时使用 `standing_z`，而不是 tracking z。
- 开始播放后 `frame=0` 只切换一次到 tracking context。
- `actor_z` 没有二次窗口平滑。
- 首帧 heading input 接近预期，且不会因 anchor 每帧变化而跳变。
- action 不出现非有限值或异常突变。

## 当前已准备的参考补丁

当前仓库已有最小配置补丁：

`project_ops/pbfm_deploy_context_fix.patch`

它将 PBFM 配置中的 `window_size` 从 `3` 改为 `1`。迁移到另一份部署代码时，应按本文件同步检查 C++ 默认值、frame 索引和 heading/velocity 转换，而不是只改 YAML。
