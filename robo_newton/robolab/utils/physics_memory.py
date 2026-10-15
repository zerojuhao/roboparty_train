"""Opt-in Newton capacity and memory diagnostics; no training policy changes."""

import json
from collections import deque

import torch


def memory_snapshot(device, label):
    import warp as wp

    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    result = {
        "stage": label,
        "device_used_gib": (total - free) / 2**30,
        "device_free_gib": free / 2**30,
        "torch_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
        "torch_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
        "warp_pool_used_gib": wp.get_mempool_used_mem_current(str(device)) / 2**30,
        "warp_pool_peak_gib": wp.get_mempool_used_mem_high(str(device)) / 2**30,
    }
    print("[MEMORY] " + json.dumps(result), flush=True)
    return result


def check_solver_overflow(data):
    import warp as wp

    overflow = wp.to_torch(data.overflow)
    if torch.any(overflow != 0):
        ids = (overflow != 0).nonzero().flatten()[:16]
        raise RuntimeError(f"MuJoCo Warp capacity overflow: worlds={ids.tolist()} flags={overflow[ids].tolist()}")


def solver_capacity(solver):
    import warp as wp

    data = solver.mjw_data
    check_solver_overflow(data)
    return {
        "contacts": int(wp.to_torch(data.nacon).max()),
        "constraints_per_world": int(wp.to_torch(data.nefc).max()),
        "solver_iterations": int(wp.to_torch(data.solver_niter).max()),
    }


def diagnose_ground_miss(env, recent_states):
    """Probe a failed scan without stepping physics or substituting observations."""
    import warp as wp
    from isaaclab.utils.warp import raycast_mesh
    from robolab.utils.tensors import as_torch_tensor

    sensor = env.scene.sensors["terrain_priv"]
    local = as_torch_tensor(sensor.ray_starts)
    index = int(local[0, :, :2].square().sum(-1).argmin())
    hits = as_torch_tensor(sensor.data.ray_hits_w).clone()
    ids = (~torch.isfinite(hits[:, index, 2])).nonzero().flatten()
    if not len(ids):
        return {"root_miss_worlds": []}
    starts = wp.to_torch(sensor._ray_starts_w)[ids, index].clone()
    directions = wp.to_torch(sensor._ray_directions_w)[ids, index].clone()
    root = as_torch_tensor(env.scene["robot"].data.root_state_w)[ids].clone()
    mesh = sensor.meshes[(sensor.cfg.mesh_prim_paths[0], sensor._device)]
    top = float(wp.to_torch(mesh.points)[:, 2].max()) + 10.0
    original = raycast_mesh(starts, directions, mesh, max_dist=sensor.cfg.max_distance)[0]
    offsets = starts.new_tensor([[0, 0, 0], [.001, 0, 0], [-.001, 0, 0],
                                [0, .001, 0], [0, -.001, 0]])
    probes = starts[:, None, :] + offsets[None, :, :]
    probes[..., 2] = top
    down = torch.zeros_like(probes)
    down[..., 2] = -1
    high_hits = raycast_mesh(probes, down, mesh, max_dist=1000.)[0]
    root_probe = root[:, :3].clone()
    root_probe[:, 2] = top
    root_ground = raycast_mesh(root_probe, down[:, 0].contiguous(), mesh, max_dist=1000.)[0]
    result = {
        "root_miss_worlds": ids.tolist(), "reference_index": index,
        "root_state": root.tolist(), "ray_start": starts.tolist(),
        "ray_direction": directions.tolist(), "recast_hits": original.tolist(),
        "high_hits_center_plus_1mm": high_hits.tolist(), "root_ground": root_ground.tolist(),
        "sensor_position": as_torch_tensor(sensor.data.pos_w)[ids].tolist(),
        "episode_length": env.episode_length_buf[ids].tolist(),
        "reset_buf": env.reset_buf[ids].tolist(),
        "terrain_levels": env.scene.terrain.terrain_levels[ids].tolist(),
        "terrain_types": env.scene.terrain.terrain_types[ids].tolist(),
        "env_origins": as_torch_tensor(env.scene.env_origins)[ids].tolist(),
        "reset_lift": env._pufo_reset_lift[ids].tolist(),
        "recent": [{"step": step, "root_state": state[ids].tolist(),
                    "episode_length": age[ids].tolist()} for step, state, age in recent_states],
    }
    # force_recompute alone does not set the per-world outdated mask in this backend.
    # Re-run only the failed worlds without resetting drift or changing physics.
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    mask[ids] = True
    sensor._update_buffers_impl(wp.from_torch(mask))
    result["refreshed_hits"] = as_torch_tensor(sensor.data.ray_hits_w)[ids, index].tolist()
    result["refreshed_starts"] = wp.to_torch(sensor._ray_starts_w)[ids, index].tolist()
    return result


def make_profile_policy(env, agent_cfg, checkpoint):
    """Exercise a trained actor and expert contexts without replay or optimization."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners.fbcpr_runner import FbCprRunner

    cfg = agent_cfg.to_dict()
    cfg.update(logger="tensorboard", use_mix_rollout=False, torch_compile_mode=None)
    runner = FbCprRunner(RslRlVecEnvWrapper(env), cfg, log_dir=env.cfg.log_dir,
                        device=env.device, inference_only=True)
    runner.load(checkpoint, load_optimizer=False)
    runner._build_expert_buffer()

    @torch.no_grad()
    def policy(obs):
        runner.history.reset((env.episode_length_buf == 0).nonzero().flatten())
        parts = runner._obs_dict(obs)
        heading, _ = runner._root_heading_velocity()
        runner._context = runner.agent.advance_rollout_context(
            runner._context, env.episode_length_buf, heading, runner.replay)
        parts.update(runner.agent.context_observations(runner._context, heading))
        action = runner._env_action(runner.agent.act(parts, runner._context.z, mean=False))
        runner.history.add(runner._current_obs_values(obs))
        return action

    return policy, runner.writer


def profile_physics(env, steps, action_scale=0.0, trace_path=None, nan_path=None, policy=None):
    from isaaclab_newton.physics.newton_manager import NewtonManager

    solver = NewtonManager._solver
    peaks = {"contacts": 0, "constraints_per_world": 0, "solver_iterations": 0}
    memory_snapshot(env.device, "environment_constructed")
    obs, _ = env.reset()
    memory_snapshot(env.device, "reset")
    levels = getattr(env.scene.terrain, "terrain_levels", None)
    print("[CAPACITY] " + json.dumps({
        "envs": env.num_envs, "njmax": env.cfg.sim.physics.solver_cfg.njmax,
        "nconmax": env.cfg.sim.physics.solver_cfg.nconmax,
        "use_cuda_graph": env.cfg.sim.physics.use_cuda_graph,
        "graph_conditional": bool(solver.mjw_model.opt.graph_conditional),
        "terrain_level_range": None if levels is None else [int(levels.min()), int(levels.max())],
    }), flush=True)
    original_step = env.sim.step
    probe = None
    if nan_path:
        from robolab.utils.physics_nan import PhysicsNaNProbe

        probe = PhysicsNaNProbe(solver, nan_path)
        probe.install()
        probe.install_events(env)
        # The environment may have captured physics before diagnostics were installed.
        # Re-capture on the next step so the checks execute during graph replay.
        NewtonManager._graph = None
        NewtonManager._graph_capture_pending = True

    def measured_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        if probe is not None:
            if "integration:qvel" not in probe.labels:
                raise RuntimeError("Physics NaN probe did not execute inside the solver graph")
            probe.raise_if_failed(env)
        # Check every physics tick, before the environment can reset overflow flags.
        values = solver_capacity(solver)
        for key, value in values.items():
            peaks[key] = max(peaks[key], value)
        return result

    env.sim.step = measured_step
    actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    trace = []
    recent_states = deque(maxlen=8)
    try:
        for step in range(steps):
            from robolab.utils.tensors import as_torch_tensor

            recent_states.append((step, as_torch_tensor(env.scene["robot"].data.root_state_w).clone(),
                                  env.episode_length_buf.clone()))
            if policy is not None:
                actions = policy(obs)
            elif action_scale:
                actions.uniform_(-action_scale, action_scale)
            obs, reward, _, _, _ = env.step(actions)
            if probe is not None:
                probe.raise_if_failed(env)
            if step == 0 and probe is not None:
                print("[PHYSICS_NAN_PROBE] " + json.dumps({"active_fields": list(probe.labels)}), flush=True)
            if not torch.isfinite(reward).all() or any(not torch.isfinite(v).all() for v in obs.values()):
                raise FloatingPointError(f"Non-finite physics profile result at step {step}")
            if trace_path:
                from robolab.utils.tensors import as_torch_tensor

                robot = env.scene["robot"].data
                trace.append({key: as_torch_tensor(value).detach().cpu().clone() for key, value in {
                    "action": actions, "root_state": robot.root_state_w,
                    "joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel,
                    "reward": reward,
                }.items()})
            if step == 0 or (step + 1) % 100 == 0:
                memory_snapshot(env.device, f"step_{step + 1}")
                print("[CAPACITY] " + json.dumps(peaks), flush=True)
    except BaseException:
        memory_snapshot(env.device, "failure")
        from robolab.utils.tensors import as_torch_tensor

        sensor = env.scene.sensors.get("terrain_priv")
        if sensor is not None:
            hits = as_torch_tensor(sensor.data.ray_hits_w)
            bad = (~torch.isfinite(hits[..., 2])).any(dim=-1).nonzero().flatten()[:16]
            root = as_torch_tensor(env.scene["robot"].data.root_pos_w)
            print("[CAPACITY_FAILURE] " + json.dumps({
                "step": step + 1, "peaks": peaks, "ray_miss_worlds": bad.tolist(),
                "root_positions": root[bad].tolist(),
                "terrain_levels": None if levels is None else levels[bad].tolist(),
                "env_origins": as_torch_tensor(env.scene.env_origins)[bad].tolist(),
            }), flush=True)
            try:
                print("[GROUND_DIAGNOSTIC] " + json.dumps(diagnose_ground_miss(env, recent_states)), flush=True)
            except Exception as diagnostic_error:
                print(f"[GROUND_DIAGNOSTIC_ERROR] {diagnostic_error!r}", flush=True)
        if probe is not None and not probe.snapshot_saved:
            probe.raise_if_failed(env)
        raise
    finally:
        env.sim.step = original_step
        if probe is not None:
            probe.close()
            NewtonManager._graph = None
            NewtonManager._graph_capture_pending = True
    if trace_path:
        torch.save({key: torch.stack([frame[key] for frame in trace]) for key in trace[0]}, trace_path)
    print("[CAPACITY] PASS " + json.dumps({"steps": steps, **peaks}), flush=True)
