"""Optional allocator measurements for non-overlapping world-model phases."""

from functools import wraps

import torch


def memory_phase(name):
    def decorate(function):
        @wraps(function)
        def measured(model, *args, **kwargs):
            enabled = getattr(model.cfg, "profile_memory", False)
            if not enabled or model.device.type != "cuda":
                return function(model, *args, **kwargs)
            baseline = torch.cuda.memory_allocated(model.device)
            torch.cuda.reset_peak_memory_stats(model.device)
            metrics = function(model, *args, **kwargs)
            peak = torch.cuda.max_memory_allocated(model.device)
            reserved = torch.cuda.max_memory_reserved(model.device)
            for key, value in (("allocated_mib", peak), ("reserved_mib", reserved),
                               ("extra_mib", max(0, peak - baseline))):
                metrics[f"wm/memory_{name}_{key}"] = torch.tensor(value / 2**20, device=model.device)
            return metrics
        return measured
    return decorate
