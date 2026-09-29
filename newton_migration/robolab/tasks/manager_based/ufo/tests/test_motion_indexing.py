"""CPU regressions for cached UFO motion frame indices, without simulator imports."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
import torch


def load_motion_module(motion_dir: Path, package_name: str):
    package = types.ModuleType(package_name)
    package.__path__ = [str(motion_dir)]
    sys.modules[package_name] = package
    name = f"{package_name}.motion_lib_npz"
    spec = importlib.util.spec_from_file_location(name, motion_dir / "motion_lib_npz.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


motion = load_motion_module(Path(__file__).resolve().parents[1] / "motion", "_ufo_motion_indexing")


class TestMotionIndexing(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        rng = np.random.default_rng(42)
        self.sources = []
        records = []
        for source_id, length in enumerate((6, 13, 2)):
            arrays = {}
            for field in motion.MOTION_FIELDS:
                shape = (length, 2) if field.startswith("joint") else (length, 3, 3)
                if field == "body_quat_w":
                    shape = (length, 3, 4)
                arrays[field] = rng.standard_normal(shape).astype(np.float32)
                if field == "body_quat_w":
                    arrays[field] /= np.linalg.norm(arrays[field], axis=-1, keepdims=True)
            path = self.root / f"source_{source_id}.npz"
            np.savez(path, fps=np.float32(10), **arrays)
            records.append({"name": f"source_{source_id}", "npz": path.name})
            # Match the loader's one-time quaternion normalization.
            arrays["body_quat_w"] /= np.linalg.norm(arrays["body_quat_w"], axis=-1, keepdims=True)
            self.sources.append({key: torch.from_numpy(value) for key, value in arrays.items()})
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps({"clips": records, "npz": {"joint_order": ["j0", "j1"], "output_fps": 10}}),
            encoding="utf-8",
        )

    def make_library(self, windowed, library_class=None):
        cfg = {"motion_file": str(self.manifest)}
        if windowed:
            cfg.update(clip_seconds=0.4, min_clip_seconds=0.2)
        return (library_class or motion.MotionLibNPZ)(cfg, num_envs=8, device="cpu")

    @staticmethod
    def windows(windowed):
        if windowed:
            return [(0, 0, 4), (0, 4, 2), (1, 0, 4), (1, 4, 4), (1, 8, 4), (2, 0, 2)]
        return [(0, 0, 6), (1, 0, 13), (2, 0, 2)]

    def assert_reference(self, library, windowed, input_kind, with_offset):
        windows = self.windows(windowed)
        ids = torch.tensor([i for i in range(len(windows)) for _ in range(7)])
        times = torch.tensor([
            time
            for _, _, length in windows
            for time in (-1.0, 0.0, 0.025, 0.1, (length - 1) / 10 - 1e-5, (length - 1) / 10, 20.0)
        ], dtype=torch.float32)
        offset = torch.arange(ids.numel() * 3, dtype=torch.float32).reshape(-1, 3) / 10 if with_offset else None
        convert = {"tensor": lambda value: value, "list": lambda value: value.tolist(), "numpy": lambda value: value.numpy()}
        actual = library.get_motion_state(convert[input_kind](ids), convert[input_kind](times), offset=offset)

        lengths = torch.tensor([windows[i][2] for i in ids.tolist()])
        durations = (lengths - 1).float() / 10
        frame0 = (torch.clip(times / durations, 0, 1) * (lengths - 1).float()).long()
        frame1 = torch.minimum(frame0 + 1, lengths - 1)
        dt = torch.full_like(times, 0.1)
        blend = torch.clip((times - frame0.float() * dt) / dt, 0, 1).unsqueeze(-1)
        expected = {}
        output_keys = dict(zip(motion.MOTION_FIELDS, ("dof_pos", "dof_vel", "rg_pos", "rb_rot", "body_vel", "body_ang_vel")))
        for field, output_key in output_keys.items():
            endpoints = []
            for frame in (frame0, frame1):
                endpoints.append(torch.stack([
                    self.sources[windows[i][0]][field][windows[i][1] + int(f)]
                    for i, f in zip(ids.tolist(), frame)
                ]))
            value0, value1 = endpoints
            weight = blend if field.startswith("joint") else blend.unsqueeze(-1)
            if field == "body_quat_w":
                expected[output_key] = motion.slerp(value0, value1, weight)
            else:
                expected[output_key] = (1 - weight) * value0 + weight * value1
            if field == "body_pos_w" and offset is not None:
                expected[output_key] += offset.unsqueeze(-2)
        expected.update(
            root_pos=expected["rg_pos"][:, 0],
            root_rot=expected["rb_rot"][:, 0],
            root_vel=expected["body_vel"][:, 0],
            root_ang_vel=expected["body_ang_vel"][:, 0],
            rg_pos_t=expected["rg_pos"],
            rg_rot_t=expected["rb_rot"],
            body_vel_t=expected["body_vel"],
            body_ang_vel_t=expected["body_ang_vel"],
        )
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_full_clip_indices(self):
        library = self.make_library(False)
        self.assertEqual(library._motion_starts.tolist(), [0, 6, 19])

    def test_window_indices_and_dropped_tail(self):
        library = self.make_library(True)
        self.assertEqual(library._motion_starts.tolist(), [0, 4, 6, 10, 14, 19])
        self.assertEqual(library._motion_num_frames.tolist(), [4, 2, 4, 4, 4, 2])

    def test_full_clip_queries(self):
        library = self.make_library(False)
        for input_kind in ("tensor", "list", "numpy"):
            for with_offset in (False, True):
                with self.subTest(input_kind=input_kind, with_offset=with_offset):
                    self.assert_reference(library, False, input_kind, with_offset)

    def test_windowed_queries(self):
        library = self.make_library(True)
        for input_kind in ("tensor", "list", "numpy"):
            for with_offset in (False, True):
                with self.subTest(input_kind=input_kind, with_offset=with_offset):
                    self.assert_reference(library, True, input_kind, with_offset)


if __name__ == "__main__":
    unittest.main()
