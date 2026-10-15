"""Memory limits may change, but solver precision and overflow safety may not."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from robolab.tasks.manager_based.ufo.pufo_env_cfg import RP1PUFOEnvCfg
from robolab.utils.physics_memory import check_solver_overflow, solver_capacity, diagnose_ground_miss


class TestPhysicsCapacity(unittest.TestCase):
    def test_mjwarp_contact_guard_is_process_local_and_restorable(self):
        from mujoco_warp._src import collision_gjk
        from robolab.utils import mjwarp_contact_guard

        mjwarp_contact_guard.restore()
        stock = collision_gjk._tri_affine_coord
        stock_attach = collision_gjk._attach_face
        installed = mjwarp_contact_guard.install()
        self.assertIs(installed, stock)
        self.assertIsNot(collision_gjk._tri_affine_coord, stock)
        self.assertIsNot(collision_gjk._attach_face, stock_attach)
        mjwarp_contact_guard.restore()
        self.assertIs(collision_gjk._tri_affine_coord, stock)
        self.assertIs(collision_gjk._attach_face, stock_attach)

    def test_event_probe_preserves_stateful_event_reset(self):
        from robolab.utils.physics_nan import PhysicsNaNProbe

        original = MagicMock()
        config = SimpleNamespace(func=original)
        env = SimpleNamespace(event_manager=SimpleNamespace(
            _mode_term_names={"reset": ["gains"]}, _mode_term_cfgs={"reset": [config]}))
        probe = PhysicsNaNProbe.__new__(PhysicsNaNProbe)
        probe.originals = []
        probe.check_newton = MagicMock()
        state = object()
        with patch("isaaclab_newton.physics.newton_manager.NewtonManager",
                   SimpleNamespace(backend=SimpleNamespace(state_0=state))):
            probe.install_events(env)
            config.func(env, [0])
            config.func.reset(env_ids=[0])
        original.assert_called_once_with(env, [0])
        original.reset.assert_called_once_with(env_ids=[0])
        self.assertEqual(probe.check_newton.call_count, 2)
        probe.close()
        self.assertIs(config.func, original)

    def test_stage_probe_recaptures_graph_and_is_opt_in(self):
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from robolab.utils.physics_memory import profile_physics

        class Scene(dict):
            pass

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                old_graph = object()
                scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(1, 13))))
                scene.terrain = SimpleNamespace(terrain_levels=torch.zeros(1, dtype=torch.long))
                physics = SimpleNamespace(use_cuda_graph=True, solver_cfg=SimpleNamespace(njmax=4096, nconmax=512))
                env = SimpleNamespace(
                    scene=scene, reset=MagicMock(return_value=({}, {})), device="cpu", num_envs=1,
                    episode_length_buf=torch.zeros(1, dtype=torch.long),
                    cfg=SimpleNamespace(sim=SimpleNamespace(physics=physics)),
                    action_manager=SimpleNamespace(total_action_dim=1),
                )
                original_step = MagicMock()
                env.sim = SimpleNamespace(step=original_step)

                def step(actions):
                    self.assertIs(NewtonManager._graph, None if enabled else old_graph)
                    self.assertEqual(NewtonManager._graph_capture_pending, enabled)
                    env.sim.step()
                    return {"policy": actions}, torch.zeros(1), None, None, None

                env.step = step
                solver = SimpleNamespace(mjw_model=SimpleNamespace(opt=SimpleNamespace(graph_conditional=True)))
                probe = MagicMock(labels={"integration:qvel": 1})
                with patch.object(NewtonManager, "_solver", solver), \
                        patch.object(NewtonManager, "_graph", old_graph), \
                        patch.object(NewtonManager, "_graph_capture_pending", False), \
                        patch("robolab.utils.physics_nan.PhysicsNaNProbe", return_value=probe) as factory, \
                        patch("robolab.utils.physics_memory.memory_snapshot"), \
                        patch("robolab.utils.physics_memory.solver_capacity", return_value={}):
                    profile_physics(env, 1, nan_path="unused.pt" if enabled else None)
                    self.assertIs(env.sim.step, original_step)
                    if enabled:
                        factory.assert_called_once_with(solver, "unused.pt")
                        probe.install.assert_called_once()
                        self.assertEqual(probe.raise_if_failed.call_count, 2)
                        probe.raise_if_failed.assert_called_with(env)
                        probe.close.assert_called_once()
                        self.assertIsNone(NewtonManager._graph)
                        self.assertTrue(NewtonManager._graph_capture_pending)
                    else:
                        factory.assert_not_called()
                        self.assertIs(NewtonManager._graph, old_graph)

    def test_ground_diagnostic_selects_root_misses_and_explicitly_refreshes(self):
        class Scene(dict):
            pass

        local = torch.tensor([[[1., 0., 2.], [0., 0., 2.]]]).expand(2, -1, -1).clone()
        hits = torch.zeros(2, 2, 3)
        hits[0, 0] = torch.inf  # A peripheral miss is not the failure being diagnosed.
        hits[1, 1] = torch.inf
        sensor = SimpleNamespace(
            ray_starts=local, data=SimpleNamespace(ray_hits_w=hits, pos_w=torch.zeros(2, 3)),
            _ray_starts_w=local.clone(), _ray_directions_w=torch.zeros(2, 2, 3),
            cfg=SimpleNamespace(mesh_prim_paths=["terrain"], max_distance=20.), _device="cpu",
            meshes={("terrain", "cpu"): SimpleNamespace(points=torch.zeros(4, 3))},
        )
        masks = []

        def refresh(mask):
            masks.append(mask.clone())
            sensor.data.ray_hits_w[mask, 1] = 7.

        sensor._update_buffers_impl = refresh
        scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(2, 13))))
        scene.sensors = {"terrain_priv": sensor}
        scene.terrain = SimpleNamespace(terrain_levels=torch.tensor([0, 9]), terrain_types=torch.tensor([1, 2]))
        scene.env_origins = torch.zeros(2, 3)
        env = SimpleNamespace(scene=scene, num_envs=2, device="cpu", episode_length_buf=torch.tensor([4, 0]),
                              reset_buf=torch.tensor([False, True]), _pufo_reset_lift=torch.zeros(2))
        with patch("warp.to_torch", side_effect=lambda x: x), patch("warp.from_torch", side_effect=lambda x: x), \
                patch("isaaclab.utils.warp.raycast_mesh", side_effect=lambda starts, *a, **kw: (torch.zeros_like(starts),)):
            result = diagnose_ground_miss(env, [(3, torch.zeros(2, 13), torch.tensor([3, 29]))])
        self.assertEqual(result["root_miss_worlds"], [1])
        self.assertEqual(result["terrain_levels"], [9])
        self.assertEqual(result["refreshed_hits"], [[7., 7., 7.]])
        self.assertEqual(masks[0].tolist(), [False, True])
        self.assertEqual(result["recent"][0]["episode_length"], [29])

    def test_profile_preserves_physics_accuracy_and_graphs(self):
        physics = RP1PUFOEnvCfg().sim.physics
        cfg = physics.solver_cfg
        self.assertTrue(physics.use_cuda_graph)
        self.assertEqual(physics.num_substeps, 2)
        self.assertEqual((cfg.iterations, cfg.ls_iterations, cfg.tolerance), (600, 80, .01))
        self.assertEqual((cfg.nconmax, cfg.njmax), (512, 4096))
        self.assertTrue(physics.class_type.endswith(":NewtonPUFOMJWarpManager"))
        from robolab.tasks.manager_based.ufo.pufo_physics import NewtonPUFOMJWarpManager
        from isaaclab.utils.backend_utils import FactoryBase

        with patch("isaaclab.sim.simulation_context.SimulationContext.instance",
                   return_value=SimpleNamespace(physics_manager=NewtonPUFOMJWarpManager)):
            self.assertEqual(FactoryBase._get_backend(), "newton")

    def test_overflows_are_fatal_including_epa_and_nonzero_ranks(self):
        with patch("warp.to_torch", side_effect=lambda x: x):
            for flag in (1, 2, 4, 8, 256):
                with self.assertRaisesRegex(RuntimeError, "worlds=\\[2\\]"):
                    check_solver_overflow(SimpleNamespace(overflow=torch.tensor([0, 0, flag, 0])))
            data = SimpleNamespace(overflow=torch.zeros(4, dtype=torch.int32),
                                   nacon=torch.tensor([20]), nefc=torch.tensor([0, 8, 11, 3]),
                                   solver_niter=torch.tensor([1, 2, 3, 0]))
            self.assertEqual(solver_capacity(SimpleNamespace(mjw_data=data)),
                             dict(contacts=20, constraints_per_world=11, solver_iterations=3))

    def test_epa_workspace_is_restored_on_teardown_and_failed_creation(self):
        from mujoco_warp._src import collision_convex
        from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
        from robolab.tasks.manager_based.ufo.pufo_physics import NewtonPUFOMJWarpManager, PUFOMJWarpSolverCfg

        original = collision_convex.MJ_MAX_EPAHORIZON
        try:
            with patch.object(NewtonMJWarpManager, "_create_solver", return_value="solver"):
                self.assertEqual(NewtonPUFOMJWarpManager._create_solver(None, PUFOMJWarpSolverCfg()), "solver")
                self.assertEqual(collision_convex.MJ_MAX_EPAHORIZON, max(64, original))
            with patch.object(NewtonMJWarpManager, "clear"):
                NewtonPUFOMJWarpManager.clear()
            self.assertEqual(collision_convex.MJ_MAX_EPAHORIZON, original)
            with patch.object(NewtonMJWarpManager, "_create_solver", side_effect=RuntimeError("test")):
                with self.assertRaisesRegex(RuntimeError, "test"):
                    NewtonPUFOMJWarpManager._create_solver(None, PUFOMJWarpSolverCfg())
            self.assertEqual(collision_convex.MJ_MAX_EPAHORIZON, original)
        finally:
            NewtonPUFOMJWarpManager._restore_epa_horizon()
