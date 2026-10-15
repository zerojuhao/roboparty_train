"""Depth quantization contract shared by Parkour and SSR."""

from types import SimpleNamespace

import torch

from robolab.tasks.manager_based.parkour.agents.rp1_parkour_agent_cfg import RP1ParkourAmpRunnerCfg
from robolab.tasks.manager_based.parkour.agents.rp1_ssr_agent_cfg import RP1SSRAmpRunnerCfg
from robolab.tasks.manager_based.parkour.agents.rpo_parkour_agent_cfg import RPOParkourAmpRunnerCfg
from robolab.tasks.manager_based.parkour.mdp.observations.exteroception import (
    delayed_visualizable_image,
    quantize_normalized_depth,
)
from robolab.tasks.manager_based.parkour.rp1_parkour_env_cfg import RP1ParkourEnvCfg
from robolab.tasks.manager_based.parkour.rp1_ssr_env_cfg import RP1SSREnvCfg


def test_normalized_depth_quantizes_like_pufo() -> None:
    depth = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    actual = quantize_normalized_depth(depth)
    torch.testing.assert_close(actual, torch.tensor([0, 128, 255], dtype=torch.uint8))


def test_delayed_depth_uses_cached_indices_without_mutating_sensor_history() -> None:
    history = torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 1, 2, 1)
    original = history.clone()
    term = object.__new__(delayed_visualizable_image)
    term.sensor = SimpleNamespace(data=SimpleNamespace(output={"depth_history": history}))
    term.data_type = "depth_history"
    term._batch_indices = torch.arange(2).unsqueeze(1)
    term._frame_indices = torch.tensor([[0, 2, 5], [1, 3, 4]])
    term.quantize_uint8 = False

    actual = term(None, data_type="depth_history")
    expected = history.squeeze(-1)[term._batch_indices.expand(-1, 3), term._frame_indices]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(history, original)


def test_parkour_and_ssr_enable_uint8_observations() -> None:
    for cfg_type in (RP1ParkourEnvCfg, RP1SSREnvCfg):
        cfg = cfg_type()
        assert cfg.observations.policy.depth_image.params["quantize_uint8"] is True


def test_parkour_and_ssr_compact_rollout_depth_history() -> None:
    expected_offsets = (35, 30, 25, 20, 15, 10, 5, 0)
    for cfg_type in (RPOParkourAmpRunnerCfg, RP1ParkourAmpRunnerCfg, RP1SSRAmpRunnerCfg):
        cfg = cfg_type()
        assert cfg.compact_depth_history is True
        assert tuple(cfg.depth_history_offsets) == expected_offsets
        runner_cfg = cfg.to_dict()
        assert runner_cfg["compact_depth_history"] is True
        assert tuple(runner_cfg["depth_history_offsets"]) == expected_offsets


def test_parkour_training_speed_configuration() -> None:
    for cfg_type in (RPOParkourAmpRunnerCfg, RP1ParkourAmpRunnerCfg, RP1SSRAmpRunnerCfg):
        cfg = cfg_type()
        assert cfg.check_for_nan_interval == 100
        assert cfg.torch_compile_mode == "default"
        assert cfg.algorithm.use_mixed_precision is True
        assert cfg.algorithm.amp_cfg.disc_num_updates == 4
        serialized = cfg.to_dict()
        assert serialized["check_for_nan_interval"] == 100
        assert serialized["torch_compile_mode"] == "default"
        assert serialized["algorithm"]["use_mixed_precision"] is True
        assert serialized["algorithm"]["amp_cfg"]["disc_num_updates"] == 4

    rpo_cfg = RPOParkourAmpRunnerCfg()
    assert rpo_cfg.algorithm.num_learning_epochs == 3
    assert rpo_cfg.algorithm.num_mini_batches == 4

    ssr_cfg = RP1SSRAmpRunnerCfg()
    assert ssr_cfg.foothold_imagination.batch_size == 2048
    assert ssr_cfg.foothold_imagination.updates_per_iteration == 4
    assert ssr_cfg.foothold_imagination.balanced_candidate_multiplier == 4
    assert ssr_cfg.foothold_imagination.grid.expectation_eval_chunk_size == 256


def test_parkour_agents_enable_velocity_estimation() -> None:
    for cfg_type in (RPOParkourAmpRunnerCfg, RP1ParkourAmpRunnerCfg):
        cfg = cfg_type()
        assert cfg.actor.enable_velocity_estimation is True
        assert cfg.actor.aux_velocity_coef > 0.0
        assert cfg.algorithm.enable_aux_loss is True
        assert cfg.algorithm.aux_loss_coef > 0.0
