"""Shared PUFO/parkour collision and capacity guard, including CUDA Graph replay."""

from functools import lru_cache

from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab.utils import configclass

from robolab.utils.physics_memory import check_solver_overflow


@configclass
class GuardedMJWarpSolverCfg(MJWarpSolverCfg):
    class_type: str = "robolab.utils.guarded_mjwarp:NewtonGuardedMJWarpManager"
    epa_horizon_capacity: int = 64
    heightfield_contact_capacity: int = 50
    """Candidate contacts per hfield/geom pair, before reduction to four contacts."""


class NewtonGuardedMJWarpManager(NewtonMJWarpManager):
    _original_epa_horizon = None
    _contact_guard_installed = False
    _original_hfield_settings = None

    @classmethod
    def _configure_heightfield_capacity(cls, capacity):
        import warp as wp
        from mujoco_warp._src import collision_convex

        if type(capacity) is not int or capacity < 50 or capacity > 2048:
            raise ValueError("heightfield_contact_capacity must be an integer in [50, 2048]")
        if cls._original_hfield_settings is not None:
            raise RuntimeError("Heightfield capacity is already configured for a live solver")
        if capacity == collision_convex.MJ_MAXCONPAIR:
            return
        names = ("MJ_MAXCONPAIR", "vec_maxconpair", "mat_maxconpair", "ccd_hfield_kernel_builder")
        cls._original_hfield_settings = {name: getattr(collision_convex, name) for name in names}
        collision_convex.MJ_MAXCONPAIR = capacity
        collision_convex.vec_maxconpair = wp.types.vector(length=capacity, dtype=float)
        collision_convex.mat_maxconpair = wp.types.matrix(shape=(capacity, 3), dtype=float)
        # The upstream cache key omits these compile-time capacities. Give this
        # solver its own factory cache; never reuse a kernel compiled for 50.
        factory = cls._original_hfield_settings["ccd_hfield_kernel_builder"].__wrapped__
        collision_convex.ccd_hfield_kernel_builder = lru_cache(maxsize=None)(factory)
        print(f"[CONTACT_GUARD] heightfield candidate capacity={capacity} (final manifold remains <=4)", flush=True)

    @classmethod
    def _restore_heightfield_capacity(cls):
        if cls._original_hfield_settings is not None:
            from mujoco_warp._src import collision_convex

            for name, value in cls._original_hfield_settings.items():
                setattr(collision_convex, name, value)
            cls._original_hfield_settings = None

    @classmethod
    def _inject_terrain_heightfields(cls, stage, builder, *, root_paths):
        from robolab.terrains.native_collision import inject_heightfields

        authored = inject_heightfields(stage, builder, root_paths)
        return authored + super()._inject_terrain_heightfields(stage, builder, root_paths=root_paths)

    @classmethod
    def _create_solver(cls, model, solver_cfg):
        from mujoco_warp._src import collision_convex
        from robolab.utils import mjwarp_contact_guard

        capacity = solver_cfg.epa_horizon_capacity
        if type(capacity) is not int or capacity < 24:
            raise ValueError("epa_horizon_capacity must be an integer >= 24")
        try:
            # Validate/install before compiling collision kernels or capturing graphs.
            mjwarp_contact_guard.install()
            cls._contact_guard_installed = True
            if cls._original_epa_horizon is None:
                cls._original_epa_horizon = collision_convex.MJ_MAX_EPAHORIZON
            collision_convex.MJ_MAX_EPAHORIZON = max(capacity, cls._original_epa_horizon)
            cls._configure_heightfield_capacity(solver_cfg.heightfield_contact_capacity)
            return super()._create_solver(model, solver_cfg)
        except BaseException:
            cls._restore_heightfield_capacity()
            cls._restore_epa_horizon()
            mjwarp_contact_guard.restore()
            cls._contact_guard_installed = False
            raise

    @classmethod
    def _restore_epa_horizon(cls):
        if cls._original_epa_horizon is not None:
            from mujoco_warp._src import collision_convex

            collision_convex.MJ_MAX_EPAHORIZON = cls._original_epa_horizon
            cls._original_epa_horizon = None

    @classmethod
    def clear(cls):
        try:
            super().clear()
        finally:
            cls._restore_heightfield_capacity()
            cls._restore_epa_horizon()
            from robolab.utils import mjwarp_contact_guard

            mjwarp_contact_guard.restore()
            cls._contact_guard_installed = False

    @classmethod
    def _check_solver_status(cls):
        super()._check_solver_status()
        # Check before auto-reset can clear a world's overflow flags. This is
        # outside graph capture; changing allocation limits must never drop contacts silently.
        if not cls._solver.use_mujoco_cpu:
            check_solver_overflow(cls._solver.mjw_data)
