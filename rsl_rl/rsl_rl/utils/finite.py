"""Finite checks with device-side aggregation and failure-only diagnostics."""

import torch


def _floating_leaves(value, name):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _floating_leaves(child, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _floating_leaves(child, f"{name}[{index}]")
    elif torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
        yield name, value


@torch.no_grad()
def nonfinite_flag(value, *, device=None):
    """Return a scalar int32 flag without reading a tensor value on the host.

    Reduce on each source device first, so mixed-device trees transfer only
    scalar flags. Rollouts supply their collective device explicitly.
    """
    groups = {}
    for _, tensor in _floating_leaves(value, ""):
        groups.setdefault(tensor.device, []).append(torch.isfinite(tensor).all())
    target = device if device is not None else next(iter(groups), "cpu")
    flags = [(~torch.stack(checks).all()).to(device=target, dtype=torch.int32)
             for checks in groups.values()]
    if not flags:
        return torch.zeros((), device=target, dtype=torch.int32)
    return torch.stack(flags).amax()


@torch.no_grad()
def raise_nonfinite_details(value, name):
    """Inspect individual leaves only after an aggregate check failed."""
    for path, tensor in _floating_leaves(value, name):
        finite = torch.isfinite(tensor)
        if bool(finite.all()):
            continue
        bad_rows, bad_cols = [], []
        if tensor.ndim:
            bad = (~finite).reshape(tensor.shape[0], -1)
            bad_rows = bad.any(dim=1).nonzero(as_tuple=False).flatten()[:8].tolist()
            bad_cols = bad.any(dim=0).nonzero(as_tuple=False).flatten()[:24].tolist()
        bad_values = tensor.reshape(-1)[~finite.reshape(-1)][:8].detach().cpu().tolist()
        raise FloatingPointError(
            f"non-finite tensor at {path}: shape={tuple(tensor.shape)}, "
            f"bad_rows={bad_rows}, bad_cols={bad_cols}, values={bad_values}"
        )


def check_finite_tree(value, name):
    if nonfinite_flag(value).item():
        raise_nonfinite_details(value, name)
