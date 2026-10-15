"""Parkour and SSR depth must share the uint8 training/deployment contract."""

from __future__ import annotations

import tempfile

import onnx
import onnxruntime as ort
import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models.parkour_model import (
    DepthEncoder,
    NormalizedDepthEncoderOnnxExporter,
    ParkourActorOnnxExporter,
    ParkourModel,
)


def encoder_cfg() -> dict:
    return {
        "channels": [4],
        "kernel_sizes": [3],
        "strides": [1],
        "hidden_sizes": [8],
        "output_size": 6,
        "paddings": [1],
        "nonlinearity": "ReLU",
        "use_maxpool": False,
        "last_activation": "ReLU",
        "input_dtype": "uint8",
        "input_range": (0.0, 255.0),
    }


def observations(batch: int = 3) -> TensorDict:
    policy = TensorDict(
        {
            "proprio_a": torch.randn(batch, 8),
            "proprio_b": torch.randn(batch, 6),
            "depth_image": torch.randint(0, 256, (batch, 2, 4, 5), dtype=torch.uint8),
        },
        batch_size=[batch],
    )
    critic = TensorDict(
        {
            "base_lin_vel": torch.randn(batch, 6),
            "left_foot_height_map": torch.randn(batch, 4),
            "right_foot_height_map": torch.randn(batch, 4),
        },
        batch_size=[batch],
    )
    return TensorDict({"policy": policy, "critic": critic}, batch_size=[batch])


def model(*, ssr: bool = False) -> tuple[ParkourModel, TensorDict]:
    obs = observations()
    net = ParkourModel(
        obs,
        {"actor": ["policy"], "critic": ["critic"]},
        "actor",
        4,
        hidden_dims=[8],
        obs_normalization=True,
        encoder_cfg=encoder_cfg(),
        num_moe_experts=2,
        enable_velocity_estimation=not ssr,
        velocity_estimation_history_length=2,
        velocity_estimation_estimator_hidden_dims=[8],
        enable_ssr_estimation=ssr,
        proprio_history_length=2,
        estimation_estimator_hidden_dims=[8],
        estimation_decoder_hidden_dims=[8],
        estimation_foot_height_actor_dim=3,
    ).eval()
    return net, obs


def test_depth_encoder_requires_uint8_and_dequantizes_before_convolution() -> None:
    net = DepthEncoder((2, 4, 5), encoder_cfg()).eval()
    depth = torch.randint(0, 256, (3, 2, 4, 5), dtype=torch.uint8)
    expected = net.head(net.conv(depth.float().div(255.0)).flatten(1))
    torch.testing.assert_close(net(depth), expected)
    with pytest.raises(TypeError, match="uint8"):
        net(depth.float().div(255.0))


def test_parkour_velocity_estimator_uses_privileged_target_and_reports_metrics() -> None:
    net, obs = model(ssr=False)

    output = net(obs)
    loss = net.get_aux_loss()
    metrics = net.get_aux_metrics()

    assert output.shape == (len(obs), 4)
    assert loss is not None and loss.requires_grad
    assert "Estimation/velocity_rmse" in metrics
    assert torch.isfinite(metrics["Estimation/velocity_rmse"])
    loss.backward()
    assert any(parameter.grad is not None for parameter in net.velocity_estimator.parameters())
    assert any(key.startswith("velocity_estimator.0.") for key in net.state_dict())
    assert not any(key.startswith("velocity_estimator.velocity_estimator.") for key in net.state_dict())


def test_parkour_velocity_estimator_uses_all_three_velocity_components() -> None:
    net, obs = model(ssr=False)
    expected = obs["critic", "base_lin_vel"][:, -3:]

    net(obs)

    torch.testing.assert_close(net.velocity_estimator.last_velocity_target, expected)
    assert net.velocity_estimator.last_velocity_pred.shape == expected.shape


@pytest.mark.parametrize("ssr", [False, True])
def test_export_wrappers_match_parkour_and_ssr_inference(ssr: bool) -> None:
    net, obs = model(ssr=ssr)
    group = obs["policy"]
    depth = group["depth_image"]
    encoder = NormalizedDepthEncoderOnnxExporter(net.depth_encoder).eval()
    latent = encoder(depth.float().div(255.0))
    actor_input = torch.cat(
        (torch.cat([group[name] for name in net.flat_names], dim=-1), latent), dim=-1
    )
    with torch.no_grad():
        expected = net(obs)
        actual = ParkourActorOnnxExporter(net).eval()(actor_input)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("ssr", [False, True])
def test_exported_graphs_preserve_quantization_and_actor_parity(ssr: bool) -> None:
    net, obs = model(ssr=ssr)
    with tempfile.TemporaryDirectory() as directory:
        net.export_as_onnx(obs[:1], directory)
        graph = onnx.load(f"{directory}/depth_encoder.onnx")
        assert graph.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        assert any(node.op_type == "Round" for node in graph.graph.node)
        assert any(
            node.op_type == "Cast"
            and any(attr.name == "to" and attr.i == onnx.TensorProto.UINT8 for attr in node.attribute)
            for node in graph.graph.node
        )

        depth = obs["policy", "depth_image"][:1]
        expected = net.depth_encoder(depth).detach().numpy()
        session = ort.InferenceSession(
            f"{directory}/depth_encoder.onnx", providers=["CPUExecutionProvider"]
        )
        actual = session.run(None, {"depth_normalized": depth.float().div(255.0).numpy()})[0]
        torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected), rtol=1e-5, atol=1e-6)

        group = obs["policy"][:1]
        proprio = torch.cat([group[name] for name in net.flat_names], dim=-1)
        actor_input = torch.cat((proprio, net.depth_encoder(depth)), dim=-1)
        expected_action = ParkourActorOnnxExporter(net).eval()(actor_input).detach().numpy()
        actor_session = ort.InferenceSession(
            f"{directory}/actor.onnx", providers=["CPUExecutionProvider"]
        )
        actual_action = actor_session.run(None, {"input": actor_input.detach().numpy()})[0]
        torch.testing.assert_close(
            torch.from_numpy(actual_action), torch.from_numpy(expected_action), rtol=1e-5, atol=1e-6
        )
