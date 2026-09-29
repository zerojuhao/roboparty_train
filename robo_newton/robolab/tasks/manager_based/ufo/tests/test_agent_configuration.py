"""Exercise network and loss overrides after runner configuration serialization."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from robolab.tasks.manager_based.ufo.agents.rp1_ufo_agent_cfg import RP1UFORunnerCfg
from robolab.tasks.manager_based.ufo.agents.rp1_pufo_agent_cfg import RP1PUFORunnerCfg
from rsl_rl.algorithms.fb_cpr_aux import FbCprAuxAgent
from rsl_rl.algorithms.pufo import PufoAgent


class TestAgentConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_agent(self, visual=False, loss_type="bce", reward_type="log_odds"):
        cfg = RP1PUFORunnerCfg() if visual else RP1UFORunnerCfg()
        cfg.z_dim = 8
        cfg.actor_hidden_dim, cfg.actor_hidden_layers = 16, 2
        cfg.forward_hidden_dim, cfg.forward_hidden_layers = 24, 2
        cfg.critic_hidden_dim, cfg.critic_hidden_layers = 32, 2
        cfg.backward_hidden_dim = 12
        cfg.context_critic_hidden_dim, cfg.context_critic_layers = 16, 2
        cfg.discriminator.hidden_dims = [20, 12]
        cfg.discriminator.loss_type = loss_type
        cfg.discriminator.reward_type = reward_type
        cfg.use_fused_adam = cfg.use_bf16_amp = False
        cfg.torch_compile_mode = None
        cfg.batch_size = cfg.z_buffer_size = 8
        dims = dict(state=4, privileged_state=6, last_action=2, history_actor=3)
        if visual:
            dims.update(heading=2, velocity_command=3, velocity_target_yaw=3, terrain_priv=5)
        # Match FbCprRunner's top-level-only SimpleNamespace conversion.
        serialized = SimpleNamespace(**cfg.to_dict())
        agent_cls = PufoAgent if visual else FbCprAuxAgent
        return agent_cls(dims, 2, serialized, device="cpu")

    def test_serialized_network_and_loss_overrides(self):
        for visual in (False, True):
            with self.subTest(visual=visual):
                agent = self.make_agent(visual, "lsgan", "amp")
                self.assertEqual(agent.discriminator_loss, "lsgan")
                layers = [m for m in agent._discriminator.trunk if isinstance(m, torch.nn.Linear)]
                self.assertEqual([m.out_features for m in layers], [20, 12, 1])
                self.assertEqual(agent.actor_hidden_dim, 16)
                self.assertEqual(agent.forward_hidden_dim, 24)
                self.assertEqual(agent.critic_hidden_dim, 32)
                parts = {name: torch.randn(4, dim) for name, dim in agent.obs_dims.items()}
                if visual:
                    parts["depth_image"] = torch.rand(4, 8, 18, 32)
                z = torch.randn(4, 8)
                self.assertEqual(agent._actor_forward(parts, z).mean.shape, (4, 2))
                obs = agent._full_input(parts)
                action = torch.randn(4, 2)
                self.assertEqual(agent._forward_map(obs, z, action).shape, (2, 4, 8))
                self.assertEqual(agent._critic_shared(obs, z, action)[0].shape, (2, 4, 1))

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
