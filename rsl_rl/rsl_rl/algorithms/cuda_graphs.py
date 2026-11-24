"""Replay fixed-shape FB-CPR gradients without per-kernel CPU dispatch."""

import torch
from collections.abc import Callable
from torch.utils._pytree import tree_flatten, tree_map
from typing import Any


class StaticGraph:
    """Capture a callable with fixed tensor inputs and reusable output storage."""

    def __init__(self, evaluate: Callable, inputs: dict[str, Any], device: torch.device) -> None:
        """Warm up and capture without advancing the training RNG state.

        Args:
            evaluate: Callable receiving the input tree as keyword arguments.
            inputs: Nested dictionaries, tuples or lists with tensor leaves.
            device: CUDA device with an explicit device index.
        """
        self.inputs = tree_map(lambda value: value.detach().clone(), inputs)
        self.leaves, self.spec = tree_flatten(self.inputs)
        self.graph = torch.cuda.CUDAGraph()

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        # Warmup/capture must not consume the training run's random samples.
        with torch.random.fork_rng(devices=[device.index]):
            with torch.cuda.stream(stream):
                for _ in range(3):
                    evaluate(**self.inputs)
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            with torch.cuda.graph(self.graph, stream=stream):
                self.outputs = evaluate(**self.inputs)

    def matches(self, inputs: dict[str, Any]) -> bool:
        """Check the input tree, tensor shapes, dtypes and devices before replay."""
        leaves, spec = tree_flatten(inputs)
        return spec == self.spec and all(
            (source.shape, source.dtype, source.device) == (target.shape, target.dtype, target.device)
            for source, target in zip(leaves, self.leaves)
        )

    def replay(self, inputs: dict[str, Any]) -> Any:
        """Replay matching inputs; clone outputs that must survive the next replay."""
        leaves, _ = tree_flatten(inputs)
        with torch.no_grad():
            torch._foreach_copy_(self.leaves, leaves)
        self.graph.replay()
        return self.outputs


class FBGraph(StaticGraph):
    """Capture forward/backward-map gradients."""

    def __init__(self, agent, parameters, inputs):
        def evaluate(**inputs):
            loss, metrics = agent._fb_loss(**inputs)
            return metrics, torch.autograd.grad(loss, parameters)

        super().__init__(evaluate, inputs, torch.device(agent.device))


class CriticGraph(StaticGraph):
    """Capture gradients for the shared reward and auxiliary critic."""

    def __init__(self, agent, parameters, inputs):
        def evaluate(**inputs):
            loss, metrics = agent._shared_critic_loss(**inputs)
            return metrics, torch.autograd.grad(loss, parameters)

        super().__init__(evaluate, inputs, torch.device(agent.device))
