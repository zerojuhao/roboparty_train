"""Opt-in CUDA-graph-compatible MuJoCo Warp stage diagnostics."""

import functools
import json
from pathlib import Path

import torch
import warp as wp


@wp.kernel
def check_worlds(values: wp.array2d[float], flags: wp.array[int], code: int):
    world, column = wp.tid()
    if not wp.isfinite(values[world, column]):
        wp.atomic_cas(flags, world, 0, code)


@wp.kernel
def check_constraints(values: wp.array2d[float], counts: wp.array[int], flags: wp.array[int], code: int):
    world, column = wp.tid()
    if column < counts[world] and not wp.isfinite(values[world, column]):
        wp.atomic_cas(flags, world, 0, code)


@wp.kernel
def check_contacts(values: wp.array2d[float], worlds: wp.array[int], count: wp.array[int],
                   flags: wp.array[int], code: int):
    contact, column = wp.tid()
    if contact < count[0]:
        world = worlds[contact]
        if world >= 0 and world < flags.shape[0] and not wp.isfinite(values[contact, column]):
            wp.atomic_cas(flags, world, 0, code)


@wp.kernel
def remember_input(values: wp.array2d[float], saved: wp.array2d[float], flags: wp.array[int]):
    world, column = wp.tid()
    if flags[world] == 0:
        saved[world, column] = values[world, column]


@wp.kernel
def check_bodies(values: wp.array2d[float], worlds: wp.array[int], flags: wp.array[int], code: int):
    body, column = wp.tid()
    world = worlds[body]
    if world >= 0 and world < flags.shape[0] and not wp.isfinite(values[body, column]):
        wp.atomic_cas(flags, world, 0, code)


@wp.kernel
def check_active(values: wp.array2d[float], done: wp.array[bool], flags: wp.array[int], code: int):
    world, column = wp.tid()
    if not done[world] and not wp.isfinite(values[world, column]):
        wp.atomic_cas(flags, world, 0, code)


@wp.kernel
def freeze_failed(values: wp.array2d[float], saved: wp.array2d[float],
                  flags: wp.array[int], frozen: wp.array[int]):
    world, column = wp.tid()
    if flags[world] != 0 and frozen[world] == 0:
        saved[world, column] = values[world, column]


@wp.kernel
def mark_frozen(flags: wp.array[int], frozen: wp.array[int]):
    world = wp.tid()
    if flags[world] != 0:
        frozen[world] = 1


class PhysicsNaNProbe:
    """Freeze each failed world's input at its first invalid dynamics stage."""

    INPUTS = ("qpos", "qvel", "qacc_warmstart", "qfrc_applied", "xfrc_applied", "ctrl", "act")

    def __init__(self, solver, path):
        self.solver, self.data = solver, solver.mjw_data
        self.path = Path(path)
        self.snapshot_saved = False
        self.flags = wp.zeros(self.data.nworld, dtype=wp.int32, device=solver.device)
        self.labels = {}
        self.views = {}
        self.originals = []
        self.inputs = {}
        self.newton_views = {}
        self.newton_states = {}
        self.context_views = {}
        self.context_saved = {}
        self.context_frozen = wp.zeros_like(self.flags)
        for name in self.INPUTS:
            view = self.view(name)
            if view.size:
                self.inputs[name] = wp.zeros_like(view)
        wp.load_module(module=__name__, device=solver.device)

    def value(self, name):
        value = self.data
        for part in name.split("."):
            value = getattr(value, part)
        return value

    def view(self, name):
        if name not in self.views:
            tensor = wp.to_torch(self.value(name))
            self.views[name] = wp.from_torch(tensor.reshape(tensor.shape[0], -1))
        return self.views[name]

    def check(self, stage, names):
        for name in names:
            view = self.view(name)
            if not view.size:
                continue
            label = stage + ":" + name
            code = self.labels.setdefault(label, len(self.labels) + 1)
            if name.startswith("contact."):
                kernel = check_contacts
                inputs = [view, self.data.contact.worldid, self.data.nacon, self.flags, code]
            elif name.startswith("efc."):
                kernel = check_constraints
                inputs = [view, self.data.nefc, self.flags, code]
            else:
                kernel, inputs = check_worlds, [view, self.flags, code]
            wp.launch(kernel, dim=view.shape, inputs=inputs, device=self.solver.device)

    def patch(self, owner, name, stage, fields, before=False):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if before:
                for key, saved in self.inputs.items():
                    view = self.view(key)
                    wp.launch(remember_input, dim=view.shape, inputs=[view, saved, self.flags],
                              device=self.solver.device)
                self.check(stage, fields)
            result = original(*args, **kwargs)
            if not before:
                self.check(stage, fields)
            return result

        setattr(owner, name, wrapped)

    def check_newton(self, state, stage):
        model = self.solver.model
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            value = getattr(state, name)
            key = (name, id(value))
            if key not in self.newton_views:
                tensor = wp.to_torch(value)
                rows = tensor.shape[0] if name.startswith("body_") else self.data.nworld
                self.newton_views[key] = wp.from_torch(tensor.reshape(rows, -1))
            view = self.newton_views[key]
            label = stage + ":" + name
            self.newton_states[label] = view
            code = self.labels.setdefault(label, len(self.labels) + 1)
            if name.startswith("body_"):
                kernel, inputs = check_bodies, [view, model.body_world, self.flags, code]
            else:
                kernel, inputs = check_worlds, [view, self.flags, code]
            wp.launch(kernel, dim=view.shape, inputs=inputs, device=self.solver.device)

    def install_newton(self):
        # PUFO has identical articulation coordinate layouts in every world.
        model = self.solver.model
        counts = torch.bincount(wp.to_torch(model.joint_world).long(), minlength=self.data.nworld)
        if len(counts) != self.data.nworld or not bool((counts == counts[0]).all()):
            raise ValueError("Newton NaN diagnostics require homogeneous per-world joint layouts")
        original_input = self.solver._update_mjc_data
        original_output = self.solver._update_newton_state
        self.originals.extend(((self.solver, "_update_mjc_data", original_input),
                               (self.solver, "_update_newton_state", original_output)))

        @functools.wraps(original_input)
        def convert_input(data, model, state=None, *args, **kwargs):
            if state is not None:
                self.check_newton(state, "newton_input")
            result = original_input(data, model, state, *args, **kwargs)
            self.check("newton_to_mujoco", ("qpos", "qvel"))
            return result

        @functools.wraps(original_output)
        def convert_output(model, state, *args, **kwargs):
            result = original_output(model, state, *args, **kwargs)
            self.check_newton(state, "newton_output")
            return result

        self.solver._update_mjc_data = convert_input
        self.solver._update_newton_state = convert_output

    def install_events(self, env):
        from isaaclab_newton.physics.newton_manager import NewtonManager

        def instrument(original, label):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                self.check_newton(NewtonManager.backend.state_0, "before_" + label)
                result = original(*args, **kwargs)
                self.check_newton(NewtonManager.backend.state_0, "after_" + label)
                return result
            if hasattr(original, "reset"):
                wrapped.reset = original.reset
            return wrapped

        manager = env.event_manager
        for mode in ("reset", "interval"):
            for name, config in zip(manager._mode_term_names.get(mode, ()),
                                    manager._mode_term_cfgs.get(mode, ())):
                self.originals.append((config, "func", config.func))
                config.func = instrument(config.func, mode + ":" + name)

    def install_cholesky(self, module):
        original = module._cholesky_factorize_solve
        self.originals.append((module, "_cholesky_factorize_solve", original))

        @functools.wraps(original)
        def wrapped(model, data, context, *args, **kwargs):
            result = original(model, data, context, *args, **kwargs)
            views = {}
            for name in ("h", "hfactor", "grad", "search", "search_dot", "newton_decrement", "Jaref"):
                value = getattr(context, name)
                if not value.size:
                    continue
                key = (name, id(value))
                if key not in self.context_views:
                    tensor = wp.to_torch(value)
                    self.context_views[key] = wp.from_torch(tensor.reshape(self.data.nworld, -1))
                view = views[name] = self.context_views[key]
                if name not in self.context_saved:
                    self.context_saved[name] = wp.zeros_like(view)
                if name in ("search", "search_dot", "newton_decrement"):
                    label = "cholesky:" + name
                    code = self.labels.setdefault(label, len(self.labels) + 1)
                    wp.launch(check_active, dim=view.shape, inputs=[view, context.done, self.flags, code],
                              device=self.solver.device)
            # Preserve the first failed factorization, before later substeps reuse scratch.
            for name, view in views.items():
                wp.launch(freeze_failed, dim=view.shape,
                          inputs=[view, self.context_saved[name], self.flags, self.context_frozen],
                          device=self.solver.device)
            wp.launch(mark_frozen, dim=self.data.nworld, inputs=[self.flags, self.context_frozen],
                      device=self.solver.device)
            return result

        module._cholesky_factorize_solve = wrapped

    def install(self):
        import mujoco_warp
        from mujoco_warp._src import forward, collision_driver, constraint, solver

        self.patch(mujoco_warp, "step", "input", self.INPUTS, before=True)
        self.patch(collision_driver, "collision", "collision", ("contact.dist", "contact.pos", "contact.frame"))
        self.patch(constraint, "make_constraint", "constraint", ("efc.pos", "efc.D", "efc.aref"))
        self.patch(forward, "fwd_position", "position", ("qpos", "M"))
        self.patch(forward, "fwd_velocity", "velocity", ("qvel", "qfrc_bias", "qfrc_passive"))
        self.patch(forward, "fwd_actuation", "actuation", ("qfrc_actuator",))
        self.patch(forward, "fwd_acceleration", "acceleration", ("qfrc_smooth", "qacc_smooth", "qLD", "qLDiagInv"))
        self.patch(solver, "solve", "solve", ("qacc", "qfrc_constraint", "efc.force"))
        self.patch(forward, "implicit", "integration", ("qpos", "qvel"))
        self.install_cholesky(solver)
        if hasattr(self.solver, "_update_newton_state"):
            self.install_newton()

    def raise_if_failed(self, env):
        flags = wp.to_torch(self.flags)
        ids = (flags != 0).nonzero().flatten()
        if not len(ids):
            return
        labels = {code: label for label, code in self.labels.items()}
        stages = [labels[int(code)] for code in flags[ids]]
        snapshot = {"worlds": ids.cpu(), "stages": stages,
                    "timestep": wp.to_torch(self.solver.mjw_model.opt.timestep).cpu(),
                    "episode_length": env.episode_length_buf[ids].cpu(),
                    "terrain_levels": env.scene.terrain.terrain_levels[ids].cpu(),
                    "terrain_types": env.scene.terrain.terrain_types[ids].cpu(),
                    "before": {key: wp.to_torch(value)[ids].cpu() for key, value in self.inputs.items()},
                    "after": {}, "model_arrays": {}, "mujoco_model": self.solver.mj_model}
        for name in set(self.views) | {"efc.J", "efc.type", "efc.id", "nefc", "overflow"}:
            tensor = wp.to_torch(self.value(name))
            if name.startswith("contact."):
                count = int(wp.to_torch(self.data.nacon)[0])
                worlds = wp.to_torch(self.data.contact.worldid)[:count]
                selected = torch.isin(worlds, ids)
                snapshot["after"][name] = tensor[:count][selected].cpu()
                snapshot["after"]["contact.worldid"] = worlds[selected].cpu()
                snapshot["after"]["contact.geom"] = wp.to_torch(self.data.contact.geom)[:count][selected].cpu()
            else:
                snapshot["after"][name] = tensor[ids].cpu()
        # Preserve randomized, world-specific model fields for a standalone replay.
        import dataclasses

        for field in dataclasses.fields(self.solver.mjw_model):
            value = getattr(self.solver.mjw_model, field.name)
            if isinstance(value, wp.array) and value.ndim > 1 and value.shape[0] == self.data.nworld:
                snapshot["model_arrays"][field.name] = wp.to_torch(value)[ids].cpu()
        for section in ("opt", "stat"):
            snapshot[section] = {}
            for field in dataclasses.fields(getattr(self.solver.mjw_model, section)):
                value = getattr(getattr(self.solver.mjw_model, section), field.name)
                if isinstance(value, wp.array):
                    tensor = wp.to_torch(value)
                    snapshot[section][field.name] = (
                        tensor[ids] if tensor.shape[0] == self.data.nworld else tensor
                    ).cpu()
                elif isinstance(value, (int, float, bool, str)):
                    snapshot[section][field.name] = value
        snapshot["block_dim"] = self.solver.mjw_model.block_dim
        snapshot["solver_context"] = {name: wp.to_torch(value)[ids].cpu()
                                      for name, value in self.context_saved.items()}
        snapshot["solver_context_frozen"] = wp.to_torch(self.context_frozen)[ids].cpu()
        snapshot["newton"] = {}
        for name, view in self.newton_states.items():
            tensor = wp.to_torch(view)
            if name.split(":")[-1].startswith("body_"):
                worlds = wp.to_torch(self.solver.model.body_world)
                selected = torch.isin(worlds, ids)
                snapshot["newton"][name] = tensor[selected].cpu()
                snapshot["newton"]["body_world"] = worlds[selected].cpu()
            else:
                snapshot["newton"][name] = tensor[ids].cpu()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(snapshot, self.path)
        self.snapshot_saved = True
        print("[PHYSICS_NAN] " + json.dumps({"worlds": ids.tolist(), "stages": stages,
                                              "snapshot": str(self.path)}), flush=True)
        raise FloatingPointError(f"First invalid MuJoCo Warp stages: {stages}; snapshot={self.path}")

    def close(self):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals.clear()
