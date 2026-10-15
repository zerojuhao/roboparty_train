import pytest

from queue_amp_resume import ROOT, continuation_command


def test_continuation_keeps_overrides_but_replaces_all_resume_targets():
    original = [
        "python", "-m", "torch.distributed.run", "--nproc_per_node=8",
        "robo_newton/scripts/rsl_rl/train.py", "--task", "RP1-AMP-Path",
        "--max_iterations", "36500", "--run_name", "default",
        "--resume", "--load_run", "older", "--checkpoint", "model_30500.pt",
        "--max_iterations=1000", "--run_name=source",
        "agent.algorithm.entropy_coef=0.01", "env.rewards.elbow_position_range_l2.weight=-0.5",
    ]
    result = continuation_command(original, "source", "model_31499.pt", "extended", 4000)
    assert result[:2] == ["bash", str(ROOT / "project_ops/train_amp_proprio.sh")]
    for key, value in {"--load_run": "source", "--checkpoint": "model_31499.pt",
                       "--run_name": "extended", "--max_iterations": "4000"}.items():
        assert result.count(key) == 1
        assert result[result.index(key) + 1] == value
    assert "agent.algorithm.entropy_coef=0.01" in result
    assert "env.rewards.elbow_position_range_l2.weight=-0.5" in result
    assert result.count("--resume") == 1


@pytest.mark.parametrize("iterations", [0, -1])
def test_nonpositive_length_is_rejected(iterations):
    with pytest.raises(ValueError, match="iterations"):
        continuation_command([], "source", "model_0.pt", "next", iterations)


def test_amp_launch_defaults_to_5000_iterations():
    script = (ROOT / "project_ops/train_amp_proprio.sh").read_text()
    assert "--max_iterations 5000" in script


def test_rp1_agent_saves_every_500_iterations():
    from robolab.tasks.manager_based.amp.agents.rp1_amp_agent_cfg import RP1AmpRunnerCfg

    cfg = RP1AmpRunnerCfg()
    assert cfg.max_iterations == 5000
    assert cfg.save_interval == 500
