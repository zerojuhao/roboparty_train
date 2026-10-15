"""Shared backend dispatch, compatibility rejection and lifecycle recovery."""

from dataclasses import fields
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mujoco_warp._src import collision_convex, collision_gjk
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
from robolab.tasks.manager_based.amp.amp_env_cfg import PhysicsPresetCfg
from robolab.tasks.manager_based.parkour.rp1_parkour_env_cfg import RP1ParkourEnvCfg
from robolab.tasks.manager_based.parkour.rpo_parkour_env_cfg import RPOParkourEnvCfg
from robolab.tasks.manager_based.ufo.pufo_physics import NewtonPUFOMJWarpManager
from robolab.utils import mjwarp_contact_guard as guard
from robolab.utils.guarded_mjwarp import GuardedMJWarpSolverCfg, NewtonGuardedMJWarpManager


class TestSharedPhysicsGuard(unittest.TestCase):
    def tearDown(self):
        guard.restore()
        NewtonGuardedMJWarpManager._restore_epa_horizon()
        NewtonGuardedMJWarpManager._restore_heightfield_capacity()
        NewtonGuardedMJWarpManager._contact_guard_installed = False

    def test_parkour_preserves_all_existing_solver_parameters(self):
        baseline = PhysicsPresetCfg()
        for task in (RP1ParkourEnvCfg, RPOParkourEnvCfg):
            cfg = task()
            physics = cfg.sim.physics.default
            self.assertIs(physics, cfg.sim.physics.newton_mjwarp)
            self.assertIsInstance(physics.solver_cfg, GuardedMJWarpSolverCfg)
            self.assertEqual(physics.class_type, GuardedMJWarpSolverCfg().class_type)
            for field in fields(baseline.newton_mjwarp.solver_cfg):
                if field.name != "class_type":
                    self.assertEqual(getattr(physics.solver_cfg, field.name),
                                     getattr(baseline.newton_mjwarp.solver_cfg, field.name), field.name)
            self.assertEqual(physics.num_substeps, baseline.newton_mjwarp.num_substeps)
            self.assertEqual(physics.use_cuda_graph, baseline.newton_mjwarp.use_cuda_graph)
            self.assertEqual(cfg.sim.physics.isaacsim_physx.class_type,
                             baseline.isaacsim_physx.class_type)
        self.assertNotIsInstance(PhysicsPresetCfg().default.solver_cfg, GuardedMJWarpSolverCfg)

    def test_pufo_alias_uses_same_lifecycle_state(self):
        self.assertIs(NewtonPUFOMJWarpManager, NewtonGuardedMJWarpManager)

    def test_current_runtime_passes_compatibility_check(self):
        self.assertEqual(guard.validate_compatibility(), guard.VALIDATED_VERSIONS)

    def test_unknown_version_fails_before_mutating_functions(self):
        original = collision_gjk._tri_affine_coord
        with patch.object(guard, "version", return_value="999.0"):
            with self.assertRaisesRegex(RuntimeError, "unvalidated physics dependencies"):
                guard.install()
        self.assertIs(collision_gjk._tri_affine_coord, original)
        self.assertIsNone(guard._original_affine)

    def test_changed_source_fails_before_installation(self):
        with patch.dict(guard._SOURCE_HASHES, {"collision_gjk": "changed"}):
            with self.assertRaisesRegex(RuntimeError, "unvalidated collision_gjk source"):
                guard.install()
        self.assertIsNone(guard._original_affine)

    def test_changed_interface_fails_before_installation(self):
        with patch.object(collision_gjk, "_attach_face", object()):
            with self.assertRaisesRegex(RuntimeError, "interface _attach_face"):
                guard.install()
        self.assertIsNone(guard._original_affine)

    def test_installation_failure_restores_functions_and_workspace(self):
        original = collision_gjk._tri_affine_coord
        horizon = collision_convex.MJ_MAX_EPAHORIZON
        install = guard.install

        def fail_after_install():
            install()
            raise RuntimeError("injected install failure")

        with patch.object(guard, "install", side_effect=fail_after_install), \
                patch.object(NewtonMJWarpManager, "_create_solver") as create:
            with self.assertRaisesRegex(RuntimeError, "injected install failure"):
                NewtonGuardedMJWarpManager._create_solver(None, GuardedMJWarpSolverCfg())
            create.assert_not_called()
        self.assertIs(collision_gjk._tri_affine_coord, original)
        self.assertEqual(collision_convex.MJ_MAX_EPAHORIZON, horizon)
        self.assertFalse(NewtonGuardedMJWarpManager._contact_guard_installed)

    def test_repeated_install_is_idempotent_and_conflicts_fail(self):
        original = guard.install()
        self.assertIs(guard.install(), original)
        with patch.object(collision_gjk, "_attach_face", object()):
            with self.assertRaisesRegex(RuntimeError, "replaced while active"):
                guard.install()

    def test_shared_tick_checks_gpu_overflow(self):
        data = object()
        for cpu in (False, True):
            with patch.object(NewtonMJWarpManager, "_check_solver_status"), \
                    patch.object(NewtonGuardedMJWarpManager, "_solver",
                                 SimpleNamespace(use_mujoco_cpu=cpu, mjw_data=data)), \
                    patch("robolab.utils.guarded_mjwarp.check_solver_overflow") as check:
                NewtonGuardedMJWarpManager._check_solver_status()
                if cpu:
                    check.assert_not_called()
                else:
                    check.assert_called_once_with(data)

    def test_heightfield_candidate_capacity_restores_types_and_factory(self):
        names = ("MJ_MAXCONPAIR", "vec_maxconpair", "mat_maxconpair", "ccd_hfield_kernel_builder")
        original = {name: getattr(collision_convex, name) for name in names}
        NewtonGuardedMJWarpManager._configure_heightfield_capacity(512)
        self.assertEqual(collision_convex.MJ_MAXCONPAIR, 512)
        self.assertIsNot(collision_convex.ccd_hfield_kernel_builder, original["ccd_hfield_kernel_builder"])
        NewtonGuardedMJWarpManager._restore_heightfield_capacity()
        for name, value in original.items():
            self.assertIs(getattr(collision_convex, name), value)

    def test_heightfield_capacity_rolls_back_on_solver_failure(self):
        original = collision_convex.ccd_hfield_kernel_builder
        with patch.object(NewtonMJWarpManager, "_create_solver", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                NewtonGuardedMJWarpManager._create_solver(None, GuardedMJWarpSolverCfg(heightfield_contact_capacity=512))
        self.assertEqual(collision_convex.MJ_MAXCONPAIR, 50)
        self.assertIs(collision_convex.ccd_hfield_kernel_builder, original)
        self.assertIsNone(guard._original_affine)


if __name__ == "__main__":
    unittest.main()
