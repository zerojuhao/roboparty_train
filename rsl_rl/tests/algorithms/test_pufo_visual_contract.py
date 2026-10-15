"""Tests for PUFO depth input and context contracts."""

import torch

import pytest

from rsl_rl.algorithms.fb_networks import (
    BackwardContextOnnxExporter,
    BackwardMap,
    DepthEncoder,
    ObsNormalizer,
)


def _encoder() -> DepthEncoder:
    return DepthEncoder(
        (2, 8, 8),
        channels=[4],
        kernel_sizes=[3],
        strides=[1],
        paddings=[1],
        pool_shape=(2, 2),
        hidden_dims=[8],
        activation="relu",
        output_size=4,
        gru_layers=1,
        input_dtype="uint8",
        input_range=(0, 255),
    )


def test_depth_encoder_requires_declared_uint8_contract() -> None:
    """The default deployable encoder rejects implicit float scaling."""
    encoder = _encoder().eval()
    output = encoder(torch.zeros(3, 2, 8, 8, dtype=torch.uint8))
    assert output.shape == (3, 4)
    with pytest.raises(TypeError, match=r"torch\.uint8"):
        encoder(torch.zeros(3, 2, 8, 8, dtype=torch.float32))


def test_depth_encoder_rejects_non_uint8_configuration_range() -> None:
    """A uint8 contract cannot silently declare a different range."""
    with pytest.raises(ValueError, match="input_range"):
        DepthEncoder(
            (2, 8, 8), channels=[4], kernel_sizes=[3], strides=[1], paddings=[1],
            pool_shape=(2, 2), hidden_dims=[8], activation="relu", output_size=4,
            gru_layers=1, input_dtype="uint8", input_range=(0, 1),
        )


def test_context_export_has_explicit_inputs_and_z_output(tmp_path) -> None:
    """The deployable context graph emits the same projected z as PyTorch."""
    torch.manual_seed(7)
    backward = BackwardMap(
        5,
        3,
        hidden_dims=[8],
        activation="relu",
        first_activation="relu",
        layer_norm=False,
        norm=True,
    ).eval()
    normalizer = ObsNormalizer({"state": 2, "privileged_state": 3}).eval()
    exporter = BackwardContextOnnxExporter(
        backward, normalizer, ("state", "privileged_state")
    ).eval()
    inputs = (
        torch.randn(2, 2),
        torch.randn(2, 3),
    )
    expected = exporter(*inputs).detach().numpy()
    output_path = tmp_path / "context.onnx"
    torch.onnx.export(
        exporter,
        inputs,
        output_path,
        opset_version=18,
        input_names=["state", "privileged_state"],
        output_names=["z"],
        dynamic_axes={
            "state": {0: "batch"},
            "privileged_state": {0: "batch"},
            "z": {0: "batch"},
        },
        external_data=False,
        dynamo=False,
    )
    onnx = pytest.importorskip("onnx")
    onnx.checker.check_model(onnx.load(output_path))
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_inputs()] == ["state", "privileged_state"]
    actual = session.run(["z"], {"state": inputs[0].numpy(), "privileged_state": inputs[1].numpy()})[0]
    assert actual.shape == (2, 3)
    assert abs(actual - expected).max() < 1e-5
