"""Lossless actor-history reconstruction with a small ring-buffer prefix."""

import torch


class ReplayActorHistory:
    def __init__(self, spec, num_envs, capacity, device):
        self.spec = tuple(tuple(item) for item in spec)
        if any(len(item) != 4 or not isinstance(item[0], str)
               or any(type(x) is not int for x in item[1:])
               or not 0 <= item[1] < item[2] or item[3] < 1 for item in self.spec):
            raise ValueError("Invalid actor history layout")
        self.length = max(item[3] for item in self.spec)
        if capacity < self.length + 1:
            raise ValueError("Replay capacity must exceed actor history length")
        self.width = sum((stop - start) * length for _, start, stop, length in self.spec)
        self.keys = tuple(dict.fromkeys(item[0] for item in self.spec))
        self.num_envs, self.device = num_envs, device
        self.prefix = {}
        self.new_segment = True

    def compact(self, data, buffer):
        observation = dict(data["observation"])
        history = observation.pop("history_actor")
        n = history.shape[0]
        if history.shape != (n, self.num_envs, self.width):
            raise ValueError("Unexpected actor history shape")
        starts = data["actor_history_start"].to(self.device).clone()
        if starts.shape != (n, self.num_envs, 1) or starts.dtype != torch.bool:
            raise ValueError("Actor history requires explicit boolean reset markers")
        for key, start, stop, _ in self.spec:
            value = observation[key]
            if (value.ndim != 3 or value.shape[:2] != history.shape[:2]
                    or value.shape[-1] < stop or value.dtype != history.dtype):
                raise ValueError("Actor history source shape/dtype mismatch")
        if self.new_segment:
            if bool((history[0] != 0).any()):
                raise ValueError("A new replay segment must start with zero actor history")
            starts[0] = True
        if not self.prefix:
            self.prefix = {
                key: torch.zeros((self.length, *observation[key].shape[1:]),
                                 dtype=observation[key].dtype, device=self.device)
                for key in self.keys
            }
            self.prefix["actor_history_start"] = torch.ones(
                self.length, self.num_envs, 1, dtype=torch.bool, device=self.device)
        # Preserve overwritten frames so even the oldest retained transition
        # keeps its exact history and remains eligible for sampling.
        dropped = max(0, len(buffer) + n - buffer.capacity)
        if dropped:
            oldest = buffer._idx if buffer._is_full else 0
            indices = (oldest + torch.arange(max(0, dropped - self.length), dropped,
                                             device=self.device)) % buffer.capacity
            for key, prefix in self.prefix.items():
                source = (buffer.storage[key] if key == "actor_history_start"
                          else buffer.storage["observation"][key])
                self.prefix[key] = torch.cat((prefix, source[indices]))[-self.length:].clone()
        self.new_segment = False
        return {**data, "observation": observation, "actor_history_start": starts}

    def restore(self, observation, idxs, buffer):
        times, envs = idxs
        age = (times - buffer._idx) % buffer.capacity if buffer._is_full else times
        offsets = torch.arange(self.length + 1, device=self.device)
        source_times = (times[:, None] - offsets) % buffer.capacity
        relative_age = age[:, None] - offsets
        prefix_indices = (self.length + relative_age).clamp(0, self.length - 1)

        def gather(key):
            source = (buffer.storage[key] if key == "actor_history_start"
                      else buffer.storage["observation"][key])
            stored = source[source_times, envs[:, None]]
            prefix = self.prefix[key][prefix_indices, envs[:, None]]
            return torch.where((relative_age < 0).unsqueeze(-1), prefix, stored)

        starts = gather("actor_history_start").squeeze(-1)
        since_reset = torch.where(starts, offsets, self.length).amin(dim=1)
        frames = {key: gather(key) for key in self.keys}
        parts = []
        for key, start, stop, length in self.spec:
            offsets = torch.arange(length, 0, -1, device=self.device)
            values = frames[key][:, offsets, start:stop]
            values = torch.where((offsets <= since_reset[:, None]).unsqueeze(-1), values, 0.0)
            parts.append(values.flatten(1))
        return {**observation, "history_actor": torch.cat(parts, dim=-1)}
