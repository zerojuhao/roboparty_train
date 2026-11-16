"""Replay fixed-shape model and FB gradients without per-kernel CPU dispatch."""

import torch
from collections.abc import Callable
from torch.utils._pytree import tree_flatten, tree_map
from typing import Any

from .mbfb_dynamics import rollout_error
from .world_model_mbfb_sequence import sequence_svg_loss, sequence_value_gradients


class WorldModelGraph:
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


class SVGGraph(WorldModelGraph):
    """Capture imagined actor gradients and the sequence-value training batch."""

    def __init__(self, model, agent, parameters, inputs):
        def evaluate(**inputs):
            loss, metrics = sequence_svg_loss(model, agent, **inputs)
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
            gradients = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(parameters, gradients))
            return loss.detach(), metrics, gradients, model._value_batch

        previous_batch = model._value_batch
        try:
            super().__init__(evaluate, inputs, model.device)
        finally:
            model._value_batch = previous_batch


class DynamicsGraph(WorldModelGraph):
    """Capture dynamics fitting errors and parameter gradients."""

    def __init__(self, model, batch):
        parameters = tuple(model.dynamics.parameters())

        def evaluate(batch):
            errors = rollout_error(model.dynamics, batch)
            loss = errors.mean()
            gradients = torch.autograd.grad(loss, parameters)
            return errors.detach(), loss.detach(), gradients

        super().__init__(evaluate, dict(batch=batch), model.device)


class ValueGraph(WorldModelGraph):
    """Capture chunked sequence-value targets and gradients."""

    def __init__(self, model, discount, batch):
        super().__init__(
            lambda batch: sequence_value_gradients(model, discount, batch), dict(batch=batch), model.device
        )


class ForwardGraph(WorldModelGraph):
    """Capture model-based targets for the forward map."""

    def __init__(self, model, agent, inputs):
        super().__init__(lambda **inputs: model.forward_rollout(agent, **inputs), inputs, model.device)


class FBGraph(WorldModelGraph):
    """Capture forward/backward-map gradients with an optional imagined target."""

    def __init__(self, agent, parameters, inputs, weight):
        self.weight = weight

        def evaluate(**inputs):
            if weight is not None:
                inputs["model_target"] = dict(inputs["model_target"], mbfb=True, weight=weight)
            loss, metrics = agent._fb_loss(**inputs, check_model_finite=False)
            return metrics, torch.autograd.grad(loss, parameters)

        super().__init__(evaluate, inputs, torch.device(agent.device))


class CriticGraph(WorldModelGraph):
    """Capture gradients for the shared reward and auxiliary critic."""

    def __init__(self, agent, parameters, inputs):
        def evaluate(**inputs):
            loss, metrics = agent._shared_critic_loss(**inputs)
            return metrics, torch.autograd.grad(loss, parameters)

        super().__init__(evaluate, inputs, torch.device(agent.device))


class ActorGraph(WorldModelGraph):
    """Capture real-data actor gradients, optionally separating auxiliary losses."""

    def __init__(self, agent, inputs, *, split):
        from .world_model_mbfb import native_actor_gradients

        parameters = tuple(p for p in agent._actor.parameters() if p.requires_grad)

        def evaluate(**inputs):
            terms, metrics, _, _ = agent._actor_objectives(**inputs)
            return metrics, native_actor_gradients(terms, parameters, split=split)

        super().__init__(evaluate, inputs, torch.device(agent.device))
