"""Read-only, rank-local parameter gradients for weighted actor loss terms."""

from itertools import combinations
from functools import wraps

import torch


def eager_gradient_diagnostics(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if not kwargs.get('depth_gradient_diagnostics', False):
            return method(self, *args, **kwargs)
        saved = []
        visited = set()
        for value in vars(self).values():
            if not isinstance(value, torch.nn.Module):
                continue
            for module in value.modules():
                if id(module) in visited:
                    continue
                visited.add(id(module))
                forward = module.forward
                original = getattr(forward, '_torchdynamo_orig_callable', None)
                if original is not None:
                    saved.append((module, forward))
                    module.forward = original
        try:
            return method(self, *args, **kwargs)
        finally:
            for module, forward in saved:
                module.forward = forward
    return wrapped


def enable_compiled_gradient_audit():
    # AOT backward must preserve buffers for per-loss autograd.grad calls.
    import torch._functorch.config as config

    config.donated_buffer = False


def parameter_gradient_metrics(terms, parameters):
    parameters = tuple(p for p in parameters if p.requires_grad)
    if not parameters:
        return {}
    vectors = {}
    for name, loss in terms.items():
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        vectors[name] = torch.cat([
            (torch.zeros_like(p) if g is None else g).detach().float().reshape(-1)
            for p, g in zip(parameters, gradients)
        ])
    norms = {name: vector.norm() for name, vector in vectors.items()}
    metrics = {f"depth_gradient_local/norm/{name}": norm for name, norm in norms.items()}
    for left, right in combinations(vectors, 2):
        denom = norms[left] * norms[right]
        metrics[f"depth_gradient_local/cosine/{left}_{right}"] = torch.where(
            denom > 0, vectors[left].dot(vectors[right]) / denom.clamp_min(1e-30),
            torch.zeros_like(denom),
        )
    actor = sum(vector for name, vector in vectors.items() if name != 'estimator')
    actor_norm = actor.norm()
    metrics['depth_gradient_local/norm/actor_total'] = actor_norm
    if 'estimator' in vectors:
        denom = actor_norm * norms['estimator']
        metrics['depth_gradient_local/cosine/actor_total_estimator'] = torch.where(
            denom > 0, actor.dot(vectors['estimator']) / denom.clamp_min(1e-30),
            torch.zeros_like(denom),
        )
        metrics['depth_gradient_local/estimator_actor_ratio'] = norms['estimator'] / actor_norm.clamp_min(1e-30)
    return metrics
