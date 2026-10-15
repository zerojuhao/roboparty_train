"""Check task observation noise against PUFO without starting a simulator."""

import importlib

import gymnasium as gym
import pytest

import robolab.tasks  # noqa: F401
from isaaclab.managers import ObservationTermCfg
from robolab.tasks.manager_based.ufo.pufo_env_cfg import PUFOObservationsCfg


TASK_SPECS = [
    spec
    for spec in gym.registry.values()
    if isinstance(spec.kwargs.get("env_cfg_entry_point"), str)
    and spec.kwargs["env_cfg_entry_point"].startswith("robolab.")
]


@pytest.fixture(params=TASK_SPECS, ids=lambda spec: spec.id)
def task_config(request):
    module, name = request.param.kwargs["env_cfg_entry_point"].split(":")
    return request.param.id, getattr(importlib.import_module(module), name)()


def test_proprioceptive_noise_matches_pufo(task_config):
    task_id, cfg = task_config
    reference = PUFOObservationsCfg().policy
    terms = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel")

    if hasattr(cfg, "observations"):
        for name in terms:
            actual = getattr(cfg.observations.policy, name).noise
            expected = getattr(reference, name).noise
            assert actual.to_dict() == expected.to_dict(), (task_id, name)
        if not task_id.endswith(("-Play", "-EVAL")):
            assert cfg.observations.policy.enable_corruption
    else:
        assert cfg.noise.add_noise
        for name in terms:
            direct_name = "ang_vel" if name == "base_ang_vel" else name
            amplitude = getattr(cfg.noise.noise_scales, direct_name)
            expected = getattr(reference, name).noise
            assert (-amplitude, amplitude) == (expected.n_min, expected.n_max), (task_id, name)


@pytest.mark.parametrize("task_id", ["RP1-AMP", "RPO-AMP", "RP1-AMP-Play", "RPO-AMP-Play"])
def test_amp_privileged_observations_remain_noise_free(task_id):
    module, name = gym.spec(task_id).kwargs["env_cfg_entry_point"].split(":")
    cfg = getattr(importlib.import_module(module), name)()
    for name in ("critic", "disc", "upper_disc", "disc_demo", "upper_disc_demo"):
        group = getattr(cfg.observations, name)
        assert not group.enable_corruption
        for term in vars(group).values():
            if isinstance(term, ObservationTermCfg):
                assert term.noise is None
    if task_id.endswith("-Play"):
        assert not cfg.observations.policy.enable_corruption
