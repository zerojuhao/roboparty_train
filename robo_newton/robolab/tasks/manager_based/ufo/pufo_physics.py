"""Compatibility names for saved PUFO configs; implementation is shared."""

from isaaclab.utils import configclass
from robolab.utils.guarded_mjwarp import GuardedMJWarpSolverCfg, NewtonGuardedMJWarpManager


@configclass
class PUFOMJWarpSolverCfg(GuardedMJWarpSolverCfg):
    class_type: str = "robolab.tasks.manager_based.ufo.pufo_physics:NewtonPUFOMJWarpManager"


# An alias keeps lifecycle state identical across both task entry points.
NewtonPUFOMJWarpManager = NewtonGuardedMJWarpManager
