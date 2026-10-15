"""Read a trusted local physics snapshot and replay its frozen input, without IsaacLab."""

import argparse
import copy
import json

import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw
from mujoco_warp._src import collision_convex, forward, solver, constraint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--world-index", type=int, default=0)
    parser.add_argument("--zero-warmstart", action="store_true")
    parser.add_argument("--solver", choices=("Newton", "CG"))
    parser.add_argument("--dump", help="Save replay contacts and Hessians for numerical analysis.")
    parser.add_argument("--require-finite", action="store_true", help="Fail the regression check on non-finite dynamics.")
    parser.add_argument("--affine", choices=("stock", "trace", "float64", "clamp", "project", "faces"), default="stock")
    parser.add_argument("--epa-horizon", type=int, default=64)
    parser.add_argument("--cpu-witness", action="store_true",
                        help="Diagnostic only: replace GPU witnesses >10 cm from matching CPU contacts.")
    parser.add_argument("--cpu-position-only", action="store_true",
                        help="With --cpu-witness, retain the GPU contact frame to isolate position errors.")
    args = parser.parse_args()
    if args.cpu_position_only and not args.cpu_witness:
        parser.error("--cpu-position-only requires --cpu-witness")
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    index = args.world_index
    model = copy.copy(snapshot["mujoco_model"])
    model.opt.timestep = float(snapshot["timestep"].flatten()[0])
    if args.solver:
        model.opt.solver = getattr(mujoco.mjtSolver, "mjSOL_" + args.solver.upper())
    for key, value in snapshot["model_arrays"].items():
        target = getattr(model, key, None)
        if isinstance(target, np.ndarray) and target.shape == tuple(value[index].shape):
            target[:] = value[index].numpy()
    cpu = mujoco.MjData(model)
    mujoco.mj_setConst(model, cpu)
    for key, value in snapshot["before"].items():
        getattr(cpu, key)[:] = value[index].numpy().reshape(getattr(cpu, key).shape)
    if args.zero_warmstart:
        cpu.qacc_warmstart[:] = 0
    print("INPUT", json.dumps({key: {"finite": bool(torch.isfinite(value[index]).all()),
                                      "max_abs": float(value[index].abs().max())}
                                for key, value in snapshot["before"].items()}), flush=True)
    cpu_input = copy.copy(cpu)
    mujoco.mj_step(model, cpu)
    print("CPU", json.dumps({"finite": bool(np.isfinite(cpu.qpos).all() and np.isfinite(cpu.qvel).all()),
                             "max_abs_qacc": float(np.abs(cpu.qacc).max()), "contacts": cpu.ncon,
                             # CPU MuJoCo may auto-reset bad state and return finite outputs.
                             "warnings": {mujoco.mjtWarning(i).name: int(warning.number)
                                          for i, warning in enumerate(cpu.warning) if warning.number}}), flush=True)
    details = {"cpu_contacts": [{"geom": [int(c.geom1), int(c.geom2)], "dist": float(c.dist),
                                  "pos": c.pos.copy(), "frame": c.frame.copy()} for c in cpu.contact],
               "cpu_qacc": cpu.qacc.copy(), "hessians": []}
    wp.init()
    collision_convex.MJ_MAX_EPAHORIZON = args.epa_horizon
    if args.affine != "stock":
        if args.affine == "project":
            from robolab.utils import mjwarp_contact_guard

            mjwarp_contact_guard.install()
        else:
            from pufo_collision_diagnostic import install

            install(args.affine)
    with wp.ScopedDevice(args.device):
        gpu_model = mjw.put_model(model)
        gpu_data = mjw.put_data(model, cpu_input, nworld=1, nconmax=512, njmax=4096)
        for key, value in snapshot["model_arrays"].items():
            target = wp.to_torch(getattr(gpu_model, key))
            target.copy_(value[index:index + 1].reshape(target.shape).to(target.device))
        for section in ("opt", "stat"):
            for key, value in snapshot.get(section, {}).items():
                owner = getattr(gpu_model, section)
                target = getattr(owner, key)
                if torch.is_tensor(value):
                    dest = wp.to_torch(target)
                    src = value[index:index + 1] if value.shape[0] > 1 else value
                    dest.copy_(src.reshape(dest.shape).to(dest.device))
                else:
                    setattr(owner, key, value)
        if args.solver:
            gpu_model.opt.solver = getattr(mujoco.mjtSolver, "mjSOL_" + args.solver.upper())
        if "block_dim" in snapshot:
            gpu_model.block_dim = snapshot["block_dim"]
        gpu_model.opt.timestep.fill_(model.opt.timestep)
        for key, value in snapshot["before"].items():
            target = wp.to_torch(getattr(gpu_data, key))
            target.copy_(value[index:index + 1].reshape(target.shape).to(target.device))
        if args.zero_warmstart:
            gpu_data.qacc_warmstart.zero_()

        def report(stage, names):
            result = {}
            for name in names:
                obj = gpu_data
                for part in name.split("."):
                    obj = getattr(obj, part)
                values = wp.to_torch(obj)
                if name.startswith("efc."):
                    values = values[:, :int(wp.to_torch(gpu_data.nefc)[0])]
                result[name] = {"nonfinite": int((~torch.isfinite(values)).sum()),
                                "max_abs": float(values.abs().max()) if values.numel() else 0}
            print(stage, json.dumps(result), flush=True)

        original = solver._cholesky_factorize_solve

        def inspect_hessian(m, d, ctx, *a, **kw):
            h = wp.to_torch(ctx.h)[0, :m.nv, :m.nv].double().cpu()
            h = h.triu() + h.triu(1).T
            finite = bool(torch.isfinite(h).all())
            if finite:
                eigenvalues = torch.linalg.eigvalsh(h)
                print("HESSIAN", json.dumps({"min_eigenvalue": float(eigenvalues.min()),
                                              "max_eigenvalue": float(eigenvalues.max())}), flush=True)
            else:
                print("HESSIAN nonfinite", flush=True)
            if args.dump:
                details["hessians"].append({"h": h.clone(), "grad": wp.to_torch(ctx.grad).cpu().clone(),
                                            "efc_state": wp.to_torch(d.efc.state).cpu().clone()})
            result = original(m, d, ctx, *a, **kw)
            print("CHOLESKY_RESULT", json.dumps({"search_finite": bool(torch.isfinite(wp.to_torch(ctx.search)).all())}),
                  flush=True)
            return result

        solver._cholesky_factorize_solve = inspect_hessian
        try:
            forward.fwd_position(gpu_model, gpu_data, factorize=False)
            if args.cpu_witness:
                count = int(wp.to_torch(gpu_data.nacon)[0])
                pairs = wp.to_torch(gpu_data.contact.geom)
                positions = wp.to_torch(gpu_data.contact.pos)
                frames = wp.to_torch(gpu_data.contact.frame)
                replacements = []
                for i in range(count):
                    pair = pairs[i].cpu().tolist()
                    candidates = [c for c in cpu.contact if [int(c.geom1), int(c.geom2)] == pair]
                    if not candidates:
                        continue
                    old = positions[i].cpu().numpy()
                    closest = min(candidates, key=lambda c: np.linalg.norm(old - c.pos))
                    error = float(np.linalg.norm(old - closest.pos))
                    if error > .1:
                        positions[i].copy_(torch.as_tensor(closest.pos, device=positions.device))
                        if not args.cpu_position_only:
                            frames[i].copy_(torch.as_tensor(closest.frame.reshape(3, 3), device=frames.device))
                        replacements.append({"pair": pair, "position_error_m": error})
                constraint.make_constraint(gpu_model, gpu_data)
                print("CPU_WITNESS_REPLACEMENTS", json.dumps(replacements), flush=True)
            report("POSITION", ("qpos", "M", "efc.D", "efc.aref"))
            if args.dump:
                count = int(wp.to_torch(gpu_data.nacon)[0])
                details["gpu_contacts"] = {name: wp.to_torch(getattr(gpu_data.contact, name))[:count].cpu().clone()
                                           for name in ("geom", "pos", "frame", "dist", "worldid")}
                details["gpu_constraints"] = {name: wp.to_torch(getattr(gpu_data.efc, name)).cpu().clone()
                                              for name in ("J", "D", "aref", "type", "id")}
                details["nefc"] = wp.to_torch(gpu_data.nefc).cpu().clone()
                details["M"] = wp.to_torch(gpu_data.M).cpu().clone()
            forward.fwd_velocity(gpu_model, gpu_data)
            report("VELOCITY", ("qvel", "qfrc_bias", "qfrc_passive"))
            forward.fwd_actuation(gpu_model, gpu_data)
            report("ACTUATION", ("qfrc_actuator",))
            forward.fwd_acceleration(gpu_model, gpu_data, factorize=True)
            report("ACCELERATION", ("qacc_smooth", "qLD", "qLDiagInv"))
            solver.solve(gpu_model, gpu_data)
            report("SOLVE", ("qacc", "qfrc_constraint", "efc.force"))
            forward.implicit(gpu_model, gpu_data)
            report("INTEGRATION", ("qpos", "qvel"))
            if args.require_finite:
                for name in ("qpos", "qvel", "qacc"):
                    if not torch.isfinite(wp.to_torch(getattr(gpu_data, name))).all():
                        raise FloatingPointError(f"Replay produced non-finite {name}")
        finally:
            solver._cholesky_factorize_solve = original
            if args.dump:
                torch.save(details, args.dump)


if __name__ == "__main__":
    main()
