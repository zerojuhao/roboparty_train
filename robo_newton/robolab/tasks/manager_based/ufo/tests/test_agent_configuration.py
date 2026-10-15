"""Exercise network and loss overrides after runner configuration serialization."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from robolab.tasks.manager_based.ufo.agents.rp1_ufo_agent_cfg import RP1UFORunnerCfg
from robolab.tasks.manager_based.ufo.agents.rp1_pufo_agent_cfg import RP1PUFORunnerCfg
from rsl_rl.algorithms.fb_cpr_aux import FbCprAuxAgent
from rsl_rl.algorithms.pufo import PufoAgent, PufoRolloutContext
from rsl_rl.algorithms.fb_networks import _FixedAdaptiveAvgPool2d
from rsl_rl.runners.fbcpr_runner import FbCprRunner, UFO_HISTORY_LENGTHS, pufo_obs_dims


class TestAgentConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_agent(
        self,
        visual=False,
        loss_type="bce",
        reward_type="log_odds",
        shared=True,
        estimate=True,
        serialize=True,
        depth_shape=None,
        shared_context=False,
        context_enabled=(True, True),
        device="cpu",
        inference_only=False,
        estimator_updates_encoder=True,
    ):
        cfg = RP1PUFORunnerCfg() if visual else RP1UFORunnerCfg()
        cfg.z_dim = 8
        cfg.actor.hidden_dim, cfg.actor.hidden_layers = 16, 2
        cfg.actor.activation = "elu"
        cfg.forward_map.hidden_dim, cfg.forward_map.hidden_layers = 24, 1
        cfg.forward_map.embedding_layers = 3
        cfg.critic.hidden_dim, cfg.critic.hidden_layers = 32, 2
        cfg.critic.embedding_layers = 2
        cfg.aux_critic.hidden_dim, cfg.aux_critic.hidden_layers = 20, 3
        cfg.aux_critic.embedding_layers = 2
        cfg.backward_map.hidden_dims = [12, 10]
        if shared is not None:
            cfg.shared_value_net = shared
        cfg.discriminator.hidden_dims = [20, 12]
        cfg.discriminator.activation = "elu"
        cfg.discriminator.layer_norm = False
        cfg.discriminator.loss_type = loss_type
        cfg.discriminator.reward_type = reward_type
        cfg.use_fused_adam = cfg.use_bf16_amp = False
        cfg.torch_compile_mode = None
        cfg.batch_size = cfg.z_buffer_size = 8
        dims = dict(state=4, privileged_state=6, last_action=2, history_actor=3)
        if visual:
            if shared_context is not None:
                cfg.shared_context_value_net = shared_context
            cfg.heading_critic_enabled, cfg.velocity_critic_enabled = context_enabled
            if depth_shape is not None:
                cfg.depth_encoder.input_shape = depth_shape
            dims.update(heading=2, velocity_command=3, velocity_target_yaw=3, terrain_priv=5)
            cfg.heading_critic.hidden_dim, cfg.heading_critic.hidden_layers = 18, 1
            cfg.heading_critic.embedding_layers = 2
            cfg.velocity_critic.hidden_dim, cfg.velocity_critic.hidden_layers = 22, 3
            cfg.velocity_critic.embedding_layers = 2
            cfg.depth_encoder.channels = [4, 6]
            cfg.depth_encoder.kernel_sizes = [3, 3]
            cfg.depth_encoder.strides = [1, 2]
            cfg.depth_encoder.paddings = [1, 1]
            cfg.depth_encoder.pool_shape = (3, 2)
            cfg.depth_encoder.hidden_dims = [20, 16]
            cfg.depth_encoder.output_size = 12
            cfg.depth_encoder.gru_layers = 2
            cfg.depth_encoder.activation = "elu"
            cfg.velocity_estimator.hidden_dims = [14, 8]
            cfg.estimated_velocity_enabled = estimate
            cfg.velocity_estimator_updates_encoder = estimator_updates_encoder
        # Match FbCprRunner's top-level-only SimpleNamespace conversion.
        serialized = SimpleNamespace(**cfg.to_dict()) if serialize else cfg
        agent_cls = PufoAgent if visual else FbCprAuxAgent
        return agent_cls(dims, 2, serialized, device=device, inference_only=inference_only)

    def test_tracking_inference_matches_full_agent_without_training_modules(self):
        for visual in (False, True):
            full = self.make_agent(visual=visual, shared_context=True)
            slim = self.make_agent(visual=visual, shared_context=True, inference_only=True)
            slim.load_state_dict(full.tracking_state_dict())
            obs = {key: torch.randn(3, size) for key, size in full.obs_dims.items()}
            if visual:
                obs["depth_image"] = torch.randint(256, (3, *full.depth_shape), dtype=torch.uint8)
            z = torch.randn(3, full.z_dim)
            torch.testing.assert_close(full.act(obs, z, mean=True), slim.act(obs, z, mean=True), rtol=0, atol=0)
            torch.testing.assert_close(full._backward_map(full._b_in(obs)), slim._backward_map(slim._b_in(obs)), rtol=0, atol=0)
            self.assertEqual(set(slim.state_dict()), {"actor", "backward_map", "obs_normalizer"})
            for name in ("_critic_shared", "_target_forward_map", "_discriminator", "actor_optimizer",
                         "_context_critic_shared", "z_buffer", "off_diag"):
                self.assertFalse(hasattr(slim, name), name)
            slim.load_state_dict(full.state_dict())
            torch.testing.assert_close(full.act(obs, z, mean=True), slim.act(obs, z, mean=True), rtol=0, atol=0)

    def test_stop_gradient_agent_update_keeps_estimator_learning_and_diagnostics_finite(self):
        agent = self.make_agent(visual=True, estimator_updates_encoder=False)
        self.assertFalse(agent._actor.velocity_estimator_updates_encoder)
        obs = {key: torch.randn(8, size) for key, size in agent.obs_dims.items()}
        obs["depth_image"] = torch.randint(256, (8, *agent.depth_shape), dtype=torch.uint8)
        parts = agent._norm_parts(obs)
        before = [p.detach().clone() for p in agent._actor.velocity_estimator.parameters()]
        measured = agent.update_actor(parts, agent._full_input(parts), torch.randn(8, agent.z_dim),
                                      depth_gradient_diagnostics=True)
        self.assertTrue(all(torch.isfinite(value).all() for value in measured.values()))
        self.assertEqual(float(measured["depth_latent_velocity_grad_norm"]), 0.)
        self.assertEqual(float(measured["depth_gradient_local/estimator_actor_ratio"]), 0.)
        self.assertGreater(float(measured["depth_latent_actor_grad_norm"]), 0.)
        self.assertTrue(any(not torch.equal(old, new) for old, new in
                            zip(before, agent._actor.velocity_estimator.parameters())))
        for axis in "xyz":
            self.assertIn(f"velocity_estimator/rmse_{axis}", measured)
            self.assertIn(f"velocity_estimator/bias_{axis}", measured)
            self.assertIn(f"velocity_estimator/stopped_mae_{axis}", measured)

    def test_pufo_defaults_to_shared_reward_critics(self):
        self.assertTrue(RP1PUFORunnerCfg().to_dict()["shared_value_net"])
        agent = self.make_agent(visual=True, shared=None)
        self.assertTrue(agent.shared_value_net)
        self.assertFalse(hasattr(agent, "_critic"))
        self.assertFalse(hasattr(agent, "_aux_critic"))
        self.assertFalse(hasattr(agent, "aux_critic_optimizer"))
        critic_parameters = {id(p) for p in agent._critic_shared.parameters()}
        optimizer_parameters = [
            id(p) for group in agent.critic_optimizer.param_groups for p in group["params"]
        ]
        self.assertEqual(len(optimizer_parameters), len(set(optimizer_parameters)))
        self.assertEqual(set(optimizer_parameters), critic_parameters)
        for name in ("_heading_critic", "_velocity_critic"):
            self.assertTrue(critic_parameters.isdisjoint(id(p) for p in getattr(agent, name).parameters()))
        self.assertIn("target_critic_shared", agent.state_dict())
        for name, parameter in agent._critic_shared.state_dict().items():
            torch.testing.assert_close(parameter, agent._target_critic_shared.state_dict()[name])

    def test_pufo_serializes_per_update_logging_without_changing_training_cadence(self):
        cfg = RP1PUFORunnerCfg().to_dict()
        self.assertTrue(cfg["log_every_update"])
        self.assertEqual(cfg["num_agent_updates"], 16)
        self.assertEqual(cfg["update_agent_every_samples"], 1024)
        self.assertEqual(cfg["log_interval_steps"], 384_000)
        self.assertEqual(cfg["eval_interval_steps"], 3_200_000)

    def test_shared_pufo_rejects_separate_critic_checkpoint(self):
        separate = self.make_agent(visual=True, shared=False)
        shared = self.make_agent(visual=True, shared=None)
        with self.assertRaisesRegex(ValueError, "does not contain shared critic weights"):
            shared.load_state_dict(separate.state_dict())

    def test_pufo_defaults_to_two_distinct_shared_critic_groups(self):
        agent = self.make_agent(visual=True, shared=None, shared_context=None)
        self.assertTrue(agent.shared_context_value_net)
        critic = agent._context_critic_shared
        self.assertEqual(critic.head_names, ("heading", "velocity"))
        self.assertEqual(critic.trunk[0].mlp[1].out_features, 18)
        self.assertFalse(hasattr(agent, "_heading_critic"))
        self.assertFalse(hasattr(agent, "_velocity_critic"))
        self.assertFalse(hasattr(agent, "heading_critic_optimizer"))
        self.assertFalse(hasattr(agent, "velocity_critic_optimizer"))
        params = {id(p) for p in critic.parameters()}
        optimizer_params = [id(p) for group in agent.context_critic_optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(optimizer_params), len(set(optimizer_params)))
        self.assertEqual(set(optimizer_params), params)
        self.assertTrue(params.isdisjoint(id(p) for p in agent._critic_shared.parameters()))
        torch.testing.assert_close(critic.state_dict(), agent._target_context_critic_shared.state_dict())

    def test_shared_context_falls_back_when_only_one_head_is_enabled(self):
        for enabled in ((True, False), (False, True)):
            agent = self.make_agent(visual=True, shared_context=True, context_enabled=enabled)
            self.assertFalse(agent.shared_context_value_net)
            self.assertEqual(hasattr(agent, "_heading_critic"), enabled[0])
            self.assertEqual(hasattr(agent, "_velocity_critic"), enabled[1])

    def test_context_checkpoint_layout_mismatch_fails_before_loading(self):
        for shared in (False, True):
            source = self.make_agent(visual=True, shared_context=shared)
            dest = self.make_agent(visual=True, shared_context=not shared)
            before = {key: value.clone() for key, value in dest._actor.state_dict().items()}
            with self.assertRaisesRegex(ValueError, "context critic layout"):
                dest.load_state_dict(source.state_dict())
            torch.testing.assert_close(dest._actor.state_dict(), before)
            with self.assertRaisesRegex(ValueError, "context optimizer layout"):
                dest.load_optimizer_state_dict(source.optimizer_state_dict())

    def test_shared_context_preserves_each_mask_and_bootstrap_discount(self):
        class ConstantHeads(torch.nn.Module):
            def __init__(self, heading, velocity):
                super().__init__()
                self.heading = torch.nn.Parameter(torch.tensor(float(heading)))
                self.velocity = torch.nn.Parameter(torch.tensor(float(velocity)))

            def forward(self, obs, z, action):
                return tuple(p.expand(2, len(z), 1) for p in (self.heading, self.velocity))

        agent = self.make_agent(visual=True, shared_context=True)
        critic = agent._context_critic_shared = ConstantHeads(0, 0)
        agent._target_context_critic_shared = ConstantHeads(2, 4)
        agent.context_critic_optimizer = torch.optim.SGD(critic.parameters(), lr=0.)
        agent._avg_grad = Mock()
        action = torch.zeros(4, 2)
        agent._sample_target_actions = Mock(return_value=(action, action))
        obs, z, next_z = torch.zeros(4, agent.full_obs_dim), torch.zeros(4, 8), torch.ones(4, 8)
        batch = {"aux_rewards": {
            "heading_reward": torch.tensor([1., 100., 3., 100.]),
            "velocity_reward": torch.tensor([100., 2., 100., 4.]),
            "velocity_next": torch.tensor([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 0.]]),
        }}
        with patch.object(agent.context_critic_optimizer, "step", wraps=agent.context_critic_optimizer.step) as step:
            metrics = agent._update_context_critics(
                train_batch=batch, actor_obs={}, next_actor_obs={}, obs=obs, next_obs=obs,
                action=action, z=z, next_z=next_z, discount=torch.tensor([.5, 0., 0., .5]),
                heading_valid=torch.tensor([True, False, True, False]),
                velocity_valid=torch.tensor([False, True, False, True]),
            )
            step.assert_called_once()
        agent._avg_grad.assert_called_once()
        agent._sample_target_actions.assert_called_once_with({}, next_z, count=2)
        self.assertAlmostEqual(metrics["heading_critic_loss"].item(), 6.5)
        self.assertAlmostEqual(metrics["velocity_critic_loss"].item(), 10.)
        self.assertAlmostEqual(metrics["heading_target_Q"].item(), 2.5)
        self.assertAlmostEqual(metrics["velocity_target_Q"].item(), 3.)
        self.assertAlmostEqual(critic.heading.grad.item(), -5.)
        self.assertAlmostEqual(critic.velocity.grad.item(), -6.)
        for valid in (torch.zeros(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool)):
            loss, _, _ = agent._context_critic_loss(torch.zeros(2, 4, 1), torch.ones(4, 1), valid)
            self.assertEqual(loss.item(), float(valid.any()))

    def test_actor_uses_both_shared_context_heads_without_updating_critic(self):
        class ActionHeads(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gain = torch.nn.Parameter(torch.ones(2))
                self.frozen_on_forward = False

            def forward(self, obs, z, action):
                self.frozen_on_forward = not self.gain.requires_grad
                return tuple((action[:, i:i+1] * self.gain[i]).expand(2, -1, -1) for i in range(2))

        class ZeroForward(torch.nn.Module):
            def forward(self, obs, z, action):
                return torch.zeros(2, len(z), z.shape[-1], device=z.device)

        for heading_coef, velocity_coef in ((1., 0.), (0., 1.)):
            with self.subTest(heading=heading_coef, velocity=velocity_coef):
                agent = self.make_agent(visual=True, shared_context=True)
                critic = agent._context_critic_shared = ActionHeads()
                agent._forward_map = ZeroForward()
                agent.cfg.scale_reg = False
                agent.cfg.reg_coeff = agent.cfg.reg_coeff_aux = agent.cfg.velocity_estimator_loss_coef = 0.
                agent.cfg.reg_coeff_heading, agent.cfg.reg_coeff_velocity = heading_coef, velocity_coef
                parts = {name: torch.randn(8, dim) for name, dim in agent.obs_dims.items()}
                parts["depth_image"] = torch.randint(256, (8, *agent.depth_shape), dtype=torch.uint8)
                z = agent.sample_z(8)
                before = [p.detach().clone() for p in agent._actor.parameters()]
                metrics = agent.update_actor(
                    parts, agent._full_input(parts), z,
                    context_actor_obs=parts, context_full_obs=agent._full_input(parts), context_z=z,
                    heading_valid=torch.arange(8) % 2 == 0, velocity_valid=torch.arange(8) % 2 == 1,
                )
                torch.testing.assert_close(
                    metrics["actor_loss"], -heading_coef * metrics["Q_heading"] - velocity_coef * metrics["Q_velocity"]
                )
                self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, agent._actor.parameters())))
                self.assertTrue(critic.frozen_on_forward)
                self.assertTrue(critic.gain.requires_grad)
                self.assertIsNone(critic.gain.grad)
                torch.testing.assert_close(critic.gain, torch.ones(2))

    def test_depth_shape_is_configured_with_the_encoder(self):
        agent = self.make_agent(visual=True, depth_shape=(4, 10, 12))
        self.assertEqual(agent.depth_shape, (4, 10, 12))
        parts = {name: torch.randn(2, dim) for name, dim in agent.obs_dims.items()}
        parts["depth_image"] = torch.randint(0, 256, (2, 4, 10, 12), dtype=torch.uint8)
        with torch.no_grad():
            action = agent.act(parts, agent.sample_z(2), mean=True)
        self.assertEqual(action.shape, (2, 2))

    def test_pufo_uses_four_frame_proprio_history_and_reference_depth_strides(self):
        cfg = RP1PUFORunnerCfg()
        self.assertEqual(set(UFO_HISTORY_LENGTHS.values()), {4})
        self.assertEqual(pufo_obs_dims(10, 5)["history_actor"], 4 * 78)
        self.assertEqual(cfg.depth_encoder.strides, [1, 2, 1])

    def test_sparse_update_metrics_are_averaged_over_present_samples(self):
        runner = object.__new__(FbCprRunner)
        metrics = runner._mean_update_metrics(
            [
                {"actor_loss": torch.tensor(1.0)},
                {"actor_loss": torch.tensor(3.0), "depth_latent_grad_cosine": torch.tensor(-0.5)},
            ]
        )
        self.assertEqual(metrics["actor_loss"].item(), 2.0)
        self.assertEqual(metrics["depth_latent_grad_cosine"].item(), -0.5)

    def test_context_critic_loss_is_normalized_over_valid_samples(self):
        class ConstantCritic(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(float(value)))

            def forward(self, obs, z, action):
                return self.value.expand(2, len(z), 1)

        agent = object.__new__(PufoAgent)
        agent.critic_members = 2
        agent._targets_uncertainty = lambda qs, penalty: (None, None, qs.mean(dim=0))
        agent._avg_grad = lambda params: None
        critic = ConstantCritic(0.0)
        target = ConstantCritic(0.0)
        optimizer = torch.optim.SGD(critic.parameters(), lr=0.0)
        batch_size = 4
        loss, _, _ = agent._context_critic_update(
            critic,
            target,
            optimizer,
            torch.zeros(batch_size, 1),
            torch.zeros(batch_size, 1),
            torch.zeros(batch_size),
            torch.ones(batch_size, 1),
            torch.ones(batch_size, dtype=torch.bool),
            torch.zeros(batch_size, 1),
            torch.zeros(batch_size, 1),
            torch.zeros(batch_size, 1),
            torch.zeros(batch_size, 1),
            0.0,
        )
        self.assertEqual(loss.item(), 1.0)

    def test_serialized_network_and_loss_overrides(self):
        for visual in (False, True):
            with self.subTest(visual=visual):
                agent = self.make_agent(visual, "lsgan", "amp")
                self.assertEqual(agent.discriminator_loss, "lsgan")
                layers = [m for m in agent._discriminator.trunk if isinstance(m, torch.nn.Linear)]
                self.assertEqual([m.out_features for m in layers], [20, 12, 1])
                self.assertEqual(agent._actor.embed_s[0].mlp[1].out_features, 16)
                self.assertIsInstance(agent._actor.embed_s[0].mlp[2], torch.nn.ELU)
                self.assertEqual(agent._forward_map.embed_z[0].mlp[1].out_features, 24)
                self.assertEqual(len(agent._forward_map.embed_z), 3)
                self.assertEqual(agent._critic_shared.trunk[0].mlp[1].out_features, 32)
                backward_layers = [m.out_features for m in agent._backward_map.net if isinstance(m, torch.nn.Linear)]
                self.assertEqual(backward_layers, [12, 10, 8])
                self.assertFalse(any(isinstance(m, torch.nn.LayerNorm) for m in agent._discriminator.trunk))
                parts = {name: torch.randn(4, dim) for name, dim in agent.obs_dims.items()}
                if visual:
                    parts["depth_image"] = torch.randint(0, 256, (4, *agent.depth_shape), dtype=torch.uint8)
                z = torch.randn(4, 8)
                self.assertEqual(agent._actor_forward(parts, z).mean.shape, (4, 2))
                obs = agent._full_input(parts)
                action = torch.randn(4, 2)
                self.assertEqual(agent._forward_map(obs, z, action).shape, (2, 4, 8))
                self.assertEqual(agent._critic_shared(obs, z, action)[0].shape, (2, 4, 1))

    def test_visual_networks_are_independently_configurable(self):
        agent = self.make_agent(visual=True)
        encoder = agent._actor.depth_encoder
        self.assertEqual([m.out_channels for m in encoder.frame_cnn if isinstance(m, torch.nn.Conv2d)], [4, 6])
        self.assertEqual([m.out_features for m in encoder.frame_projection if isinstance(m, torch.nn.Linear)], [20, 16])
        self.assertEqual((encoder.gru.input_size, encoder.gru.hidden_size, encoder.gru.num_layers), (16, 12, 2))
        self.assertEqual([m.out_features for m in agent._actor.velocity_estimator if isinstance(m, torch.nn.Linear)], [14, 8, 3])
        self.assertEqual(agent._actor.fusion.mlp[1].in_features, 16 + 12)
        self.assertEqual(agent._heading_critic.embed_z[0].mlp[1].out_features, 18)
        self.assertEqual(agent._velocity_critic.embed_z[0].mlp[1].out_features, 22)
        self.assertEqual(len(agent._heading_critic.Fs), 2)
        self.assertEqual(len(agent._velocity_critic.Fs), 4)

    def test_velocity_estimator_can_be_disabled(self):
        agent = self.make_agent(visual=True, estimate=False, serialize=False)
        parts = {name: torch.randn(4, dim) for name, dim in agent.obs_dims.items()}
        parts["depth_image"] = torch.randint(0, 256, (4, *agent.depth_shape), dtype=torch.uint8)
        distribution = agent._actor_forward(parts, torch.randn(4, 8))
        self.assertEqual(distribution.mean.shape, (4, 2))
        self.assertFalse(hasattr(agent._actor, "velocity_estimator"))
        self.assertFalse(hasattr(distribution, "velocity_estimate"))

    def test_visual_context_keeps_physical_units(self):
        agent = self.make_agent(visual=True)
        context = {
            "heading": torch.tensor([[0.25, -0.5], [0.0, 0.0]]),
            "velocity_command": torch.tensor([[1.0, -0.2, 1.0], [0.0, 0.0, 0.0]]),
            "velocity_target_yaw": torch.tensor([[0.7, -0.1, 0.0], [0.0, 0.0, 0.2]]),
        }
        normalized = agent._norm_parts(context)
        for name, value in context.items():
            torch.testing.assert_close(normalized[name], value)

    def test_context_relabel_uses_the_z_selection_mask(self):
        agent = self.make_agent(visual=True)
        batch_size = agent.cfg.batch_size
        current = torch.tensor([[1.0, 0.0]]).repeat(batch_size, 1)
        expert_next = torch.tensor([[0.0, 1.0]]).repeat(batch_size, 1)
        original_obs = {
            "heading": torch.full((batch_size, 2), 3.0),
            "velocity_command": torch.full((batch_size, 3), 4.0),
        }
        original_next = {name: value.clone() for name, value in original_obs.items()}
        train_batch = {"aux_rewards": {"root_heading_xy": current, "next_root_heading_xy": current}}
        expert_batch = {
            "heading_forward_xy": current,
            "reference_velocity_world_xy": current,
            "reference_velocity_valid": torch.ones(batch_size, 1, dtype=torch.bool),
            "next": {
                "heading_forward_xy": expert_next,
                "reference_velocity_world_xy": expert_next,
                "reference_velocity_valid": torch.ones(batch_size, 1, dtype=torch.bool),
            },
        }
        mask = (torch.arange(batch_size) < batch_size // 2).unsqueeze(-1)
        relabeled, relabeled_next = agent._relabel_train_observations(
            train_batch,
            expert_batch,
            original_obs,
            original_next,
            agent.sample_z(batch_size),
            torch.ones(batch_size, 1, dtype=torch.long),
            torch.arange(batch_size),
            mask,
        )

        torch.testing.assert_close(relabeled["heading"][: batch_size // 2], torch.zeros(batch_size // 2, 2))
        torch.testing.assert_close(
            relabeled_next["heading"][: batch_size // 2], torch.ones(batch_size // 2, 2)
        )
        torch.testing.assert_close(
            relabeled["velocity_command"][batch_size // 2 :],
            original_obs["velocity_command"][batch_size // 2 :],
        )

    def test_context_transition_masks_source_and_boundary(self):
        agent = self.make_agent(visual=True)
        agent.cfg.rollout_expert_trajectories_length = 2
        batch_size = agent.cfg.batch_size
        expert_ids = torch.tensor([0])
        state = PufoRolloutContext(
            z=agent.sample_z(batch_size),
            heading_target_world=torch.tensor([[1.0, 0.0]]).repeat(batch_size, 1),
            velocity_target_world=torch.zeros(batch_size, 2),
            heading_valid=(torch.arange(batch_size) == 0).unsqueeze(-1),
            velocity_valid=(torch.arange(batch_size) == 0).unsqueeze(-1),
            context_id=torch.zeros(batch_size, 1, dtype=torch.long),
            source_type=(torch.arange(batch_size) == 0).long().unsqueeze(-1),
            expert_env_ids=expert_ids,
            tracking_z=agent.sample_z(2).reshape(1, 2, -1),
            tracking_heading_world=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            tracking_velocity_world=torch.zeros(1, 2, 2),
            tracking_velocity_valid=torch.ones(1, 2, 1, dtype=torch.bool),
        )
        metadata = agent.context_transition(
            state,
            torch.tensor([1, *([0] * (batch_size - 1))]),
            torch.tensor([[1.0, 0.0]]).repeat(batch_size, 1),
            torch.tensor([[1.0, 0.0]]).repeat(batch_size, 1),
            torch.zeros(batch_size, 2),
            torch.zeros(batch_size, dtype=torch.bool),
        )

        self.assertTrue(metadata["velocity_valid"][0])
        self.assertFalse(metadata["velocity_valid"][1:].any())
        self.assertFalse(metadata["heading_context_continues"][0])

    def test_rollout_context_advances_paired_expert_targets(self):
        agent = self.make_agent(visual=True)
        agent.cfg.rollout_expert_trajectories_length = 3
        agent.cfg.rollout_expert_trajectories_percentage = 0.5
        num_envs = agent.cfg.batch_size
        current = F.normalize(torch.randn(num_envs, 2), dim=-1)

        def sample_context(_replay, batch_dim, length):
            z = agent.sample_z(batch_dim * length).reshape(batch_dim, length, -1)
            heading = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
            heading = heading.unsqueeze(0).expand(batch_dim, -1, -1).clone()
            velocity = heading.clone()
            valid = torch.ones(batch_dim, length, 1, dtype=torch.bool)
            return z, heading, velocity, valid

        agent._sample_tracking_context = sample_context
        state = agent.advance_rollout_context(None, torch.zeros(num_envs), current, {})
        expert_ids = state.expert_env_ids
        random_mask = torch.ones(num_envs, dtype=torch.bool)
        random_mask[expert_ids] = False

        self.assertTrue(state.heading_valid[expert_ids].all())
        self.assertFalse(state.heading_valid[random_mask].any())
        torch.testing.assert_close(
            state.heading_target_world[expert_ids], current[expert_ids]
        )
        first = agent.context_observations(state, current)
        torch.testing.assert_close(
            first["heading"][expert_ids], torch.zeros(expert_ids.numel(), 2)
        )
        torch.testing.assert_close(
            first["velocity_command"][expert_ids],
            torch.tensor([1.0, 0.0, 1.0]).expand(expert_ids.numel(), -1),
            atol=1.0e-6,
            rtol=1.0e-6,
        )

        context_ids = state.context_id.clone()
        state = agent.advance_rollout_context(state, torch.ones(num_envs), current, {})
        torch.testing.assert_close(state.z[expert_ids], state.tracking_z[:, 1])
        torch.testing.assert_close(state.context_id, context_ids)
        second = agent.context_observations(state, current)
        torch.testing.assert_close(
            second["heading"][expert_ids],
            torch.tensor([1.0, 1.0]).expand(expert_ids.numel(), -1),
            atol=1.0e-6,
            rtol=1.0e-6,
        )

    def test_configurable_static_pool_matches_adaptive_pool(self):
        for height, width, output in ((5, 9, (2, 4)), (9, 18, (3, 2)), (3, 5, (4, 3))):
            image = torch.randn(2, 6, height, width)
            torch.testing.assert_close(
                _FixedAdaptiveAvgPool2d(height, width, output)(image), F.adaptive_avg_pool2d(image, output),
            )

    def test_all_networks_update_and_checkpoint_roundtrip(self):
        for visual in (False, True):
            for shared, shared_context in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(visual=visual, shared=shared, shared_context=shared_context):
                    agent = self.make_agent(visual=visual, shared=shared, shared_context=shared_context)
                    batch_size = agent.cfg.batch_size
                    parts = {name: torch.randn(batch_size, dim) for name, dim in agent.obs_dims.items()}
                    if visual:
                        parts["depth_image"] = torch.randint(
                            0, 256, (batch_size, *agent.depth_shape), dtype=torch.uint8
                        )
                    batch = {
                        "observation": parts,
                        "next": {
                            "observation": {name: value.clone() for name, value in parts.items()},
                            "terminated": torch.zeros(batch_size, 1, dtype=torch.bool),
                        },
                        "action": torch.randn(batch_size, 2).clamp(-1, 1),
                        "z": agent.sample_z(batch_size),
                        "aux_rewards": {"aux_reward": torch.rand(batch_size),
                                        "action_rate": torch.arange(batch_size, dtype=torch.float32),
                                        "positive": torch.full((batch_size,), 3.)},
                    }
                    if visual:
                        heading_xy = F.normalize(torch.randn(batch_size, 2), dim=-1)
                        batch.update(
                            heading_forward_xy=heading_xy,
                            reference_velocity_world_xy=torch.randn(batch_size, 2),
                            reference_velocity_valid=torch.ones(batch_size, 1, dtype=torch.bool),
                        )
                        batch["next"].update(
                            heading_forward_xy=heading_xy.clone(),
                            reference_velocity_world_xy=torch.randn(batch_size, 2),
                            reference_velocity_valid=torch.ones(batch_size, 1, dtype=torch.bool),
                        )
                        batch["aux_rewards"].update(
                            heading_next=torch.zeros(batch_size, 2),
                            velocity_next=torch.cat(
                                (torch.randn(batch_size, 2), torch.ones(batch_size, 1)), dim=-1
                            ),
                            heading_z_next=batch["z"].clone(),
                            heading_valid=torch.ones(batch_size, 1, dtype=torch.bool),
                            velocity_valid=torch.ones(batch_size, 1, dtype=torch.bool),
                            heading_context_continues=torch.ones(batch_size, 1, dtype=torch.bool),
                            heading_context_id=torch.ones(batch_size, 1, dtype=torch.long),
                            heading_source_type=torch.ones(batch_size, 1, dtype=torch.long),
                            root_heading_xy=heading_xy,
                            next_root_heading_xy=heading_xy.clone(),
                            heading_reward=-torch.rand(batch_size),
                            velocity_reward=-torch.rand(batch_size),
                        )
                    replay = SimpleNamespace(sample=lambda size: batch)
                    names = ["actor", "backward_map", "forward_map", "discriminator"]
                    names += ["critic_shared"] if shared else ["critic", "aux_critic"]
                    if visual:
                        names += ["context_critic_shared"] if shared_context else ["heading_critic", "velocity_critic"]
                    modules = {name: getattr(agent, "_" + name) for name in names}
                    if shared:
                        modules.update(
                            critic_trunk=agent._critic_shared.trunk,
                            critic_head_disc=agent._critic_shared.head_disc,
                            critic_head_aux=agent._critic_shared.head_aux,
                        )
                    if visual:
                        modules.update(depth_encoder=agent._actor.depth_encoder, velocity_estimator=agent._actor.velocity_estimator)
                        if shared_context:
                            modules.update(
                                context_trunk=agent._context_critic_shared.trunk,
                                context_heading_head=agent._context_critic_shared.head_heading,
                                context_velocity_head=agent._context_critic_shared.head_velocity,
                            )
                    before = {name: [p.detach().clone() for p in module.parameters()] for name, module in modules.items()}
                    metrics = agent.update(
                        {"train": replay, "expert_slicer": replay},
                        depth_gradient_diagnostics=visual,
                        reward_term_scales={"action_rate": -.1, "positive": 2.},
                    )
                    self.assertTrue(all(torch.isfinite(value).all() for value in metrics.values()))
                    self.assertEqual(float(metrics["Reward/raw/action_rate"]), (batch_size - 1) / 2)
                    self.assertEqual(float(metrics["Reward/raw/positive"]), 3.)
                    self.assertAlmostEqual(float(metrics["Reward/weighted/total"]),
                                           6. - .1 * (batch_size - 1) / 2, places=5)
                    if visual:
                        self.assertGreater(metrics["depth_latent_actor_grad_norm"], 0.0)
                        self.assertGreater(metrics["depth_latent_velocity_grad_norm"], 0.0)
                        self.assertGreaterEqual(metrics["depth_latent_grad_cosine"], -1.0)
                        for key in (
                            "velocity_estimator/loss",
                            "velocity_estimator/mae_x",
                            "velocity_estimator/mae_y",
                            "velocity_estimator/mae_z",
                            "velocity_estimator/false_motion_when_stopped",
                            "depth_ablation/action_mae_zero",
                            "depth_ablation/action_mae_temporal_reverse",
                            "depth_ablation/velocity_mae_zero",
                            "depth_ablation/velocity_mae_temporal_reverse",
                        ):
                            self.assertIn(key, metrics)
                            self.assertTrue(torch.isfinite(metrics[key]).all())
                        self.assertLessEqual(metrics["depth_latent_grad_cosine"], 1.0)
                    for name, module in modules.items():
                        self.assertTrue(any(
                            not torch.equal(old, new) for old, new in zip(before[name], module.parameters())
                        ), name)
                    restored = self.make_agent(visual=visual, shared=shared, shared_context=shared_context)
                    restored.load_state_dict(agent.state_dict())
                    restored.load_optimizer_state_dict(agent.optimizer_state_dict())
                    for name, state in agent.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[name], state)
                    torch.testing.assert_close(
                        restored.optimizer_state_dict(), agent.optimizer_state_dict()
                    )
                    torch.testing.assert_close(
                        restored.act(parts, batch["z"], mean=True), agent.act(parts, batch["z"], mean=True),
                    )

    def test_loss_formulas_and_gradient_penalty_update(self):
        for loss_type in ("bce", "gan", "lsgan", "wgan"):
            with self.subTest(loss_type=loss_type):
                agent = self.make_agent(loss_type=loss_type)
                real, fake = torch.randn(4, 10), torch.randn(4, 10)
                z = torch.randn(4, 8)
                real_score = agent._discriminator.compute_logits(real, z)
                fake_score = agent._discriminator.compute_logits(fake, z)
                if loss_type in ("bce", "gan"):
                    expected = F.softplus(-real_score).mean() + F.softplus(fake_score).mean()
                    if loss_type == "gan":
                        expected = 0.5 * expected
                elif loss_type == "lsgan":
                    expected = 0.5 * ((real_score - 1).square().mean() + (fake_score + 1).square().mean())
                else:
                    expected = fake_score.mean() - real_score.mean()
                metrics = agent.update_discriminator(real, z, fake, z, None)
                torch.testing.assert_close(metrics["disc_loss"], expected.detach())
                before = [p.detach().clone() for p in agent._discriminator.parameters()]
                metrics = agent.update_discriminator(real, z, fake, z, 10.0)
                self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
                self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, agent._discriminator.parameters())))

    def test_reward_formulas(self):
        agent = self.make_agent()
        logits = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        for mode, expected in (
            ("log_odds", logits),
            ("amp", (1 - 0.25 * (logits - 1).square()).clamp_min(0)),
            ("softplus", F.softplus(logits)),
            ("raw", logits),
        ):
            agent.discriminator_reward_type = mode
            torch.testing.assert_close(agent.discriminator_reward_from_logits(logits), expected)

    def test_invalid_loss_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "discriminator loss"):
            self.make_agent(loss_type="typo")


if __name__ == "__main__":
    unittest.main()
