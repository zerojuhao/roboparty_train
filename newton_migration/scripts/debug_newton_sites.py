"""Print Newton clone-plan body labels before launching a training probe."""

import sys

from train_newton import run_train_cli
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.sensors.ray_caster.legacy_ray_caster import _LegacyNewtonRayCasterMixin


original_resolve = _LegacyNewtonRayCasterMixin._resolve_target_owner_exprs


def inspect_target(self, prim_expr):
    if "/Robot/" in prim_expr:
        from isaaclab.sim import SimulationContext

        stage = SimulationContext.instance().stage
        paths = [str(prim.GetPath()) for prim in stage.Traverse() if "/Robot" in str(prim.GetPath())]
        print(f"TARGET {prim_expr}; ROBOT PRIMS {paths[:150]}", flush=True)
    return original_resolve(self, prim_expr)


_LegacyNewtonRayCasterMixin._resolve_target_owner_exprs = inspect_target

original = NewtonManager._cl_inject_sites.__func__


def inspect_sites(cls, builder, source_builders):
    for name, source_builder in source_builders.items():
        print(f"SOURCE {name}: {list(source_builder.body_label)}", flush=True)
    print(f"MAIN: {list(builder.body_label)}", flush=True)
    return original(cls, builder, source_builders)


NewtonManager._cl_inject_sites = classmethod(inspect_sites)

sys.exit(run_train_cli())
