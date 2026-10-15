"""CPU-only regressions for prefix selection and lossless dataset publication."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import trim_lafan_prefix as trim


class TrimPrefixTest(unittest.TestCase):
    def arrays(self):
        arrays = {"fps": np.array([50], dtype=np.int64)}
        for key in trim.FIELDS:
            shape = (200, 2) if key.startswith("joint") else (200, 2, 4 if key == "body_quat_w" else 3)
            arrays[key] = np.zeros(shape, dtype=np.float32)
        arrays["body_quat_w"][..., 3] = 1
        arrays["joint_pos"][:40, 1] = np.deg2rad(20)
        arrays["body_pos_w"][..., 2] = 0.8
        arrays["joint_vel"][:] = np.arange(200)[:, None]
        return arrays

    def test_first_sustained_window_not_brief_crossing(self):
        a = self.arrays()
        a["joint_pos"][10:35] = 0  # Only 0.48 seconds between endpoint samples.
        self.assertEqual(trim.choose_start(a, [0, 1], [0, 1]), 40)

    def test_late_window_requires_manual_review(self):
        a = self.arrays()
        a["joint_pos"][:150, 1] = 0.4
        self.assertIsNone(trim.choose_start(a, [0, 1], [0, 1]))

    def test_tilt_guard_and_neutral_noop(self):
        a = self.arrays()
        a["joint_pos"][:] = 0
        self.assertEqual(trim.choose_start(a, [0, 1], [0, 1]), 0)
        a["body_quat_w"][:50, :, 0] = np.sin(np.deg2rad(15))
        a["body_quat_w"][:50, :, 3] = np.cos(np.deg2rad(15))
        self.assertEqual(trim.choose_start(a, [0, 1], [0, 1]), 50)

    def test_all_fields_exact_suffix_fps_and_dtype_preserved(self):
        a = self.arrays()
        b = trim.trim_arrays(a, 40)
        for key in a:
            np.testing.assert_array_equal(b[key], a[key][40:] if key in trim.FIELDS else a[key])
            self.assertEqual(b[key].dtype, a[key].dtype)
        with self.assertRaises(ValueError):
            trim.trim_arrays(a, 199)

    def make_dataset(self, base):
        root = base / "dataset"
        root.mkdir()
        np.savez(root / "motion.npz", **self.arrays())
        m = {"dataset": "fixture", "clips": [{"name": "motion", "npz": "motion.npz",
             "npz_frames": 200, "npz_sha256": trim.sha256(root / "motion.npz")}],
             "npz": {"joint_order": ["waist_roll_joint", "waist_yaw_joint"],
                     "body_order": ["base_link", "waist_yaw_link"],
                     "body_quaternion_order": "xyzw", "output_fps": 50, "total_frames": 200}}
        (root / "manifest.json").write_text(json.dumps(m))
        return root

    def test_publish_backup_hashes_and_repeated_run_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = self.make_dataset(base)
            m, r = trim.inspect(root, {})
            old_hash = trim.sha256(root / "motion.npz")
            trim.apply(root, base / "backup", m, r)
            self.assertEqual(trim.sha256(base / "backup/motion.npz"), old_hash)
            new = json.loads((root / "manifest.json").read_text())
            self.assertEqual(new["npz"]["total_frames"], 160)
            self.assertEqual(new["clips"][0]["npz_sha256"], trim.sha256(root / "motion.npz"))
            with self.assertRaises(ValueError):
                trim.inspect(root, {})

    def test_checksum_mismatch_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_dataset(Path(tmp))
            a = self.arrays()
            a["joint_pos"][0, 0] = np.nan
            np.savez(root / "motion.npz", **a)
            with self.assertRaises(ValueError):
                trim.inspect(root, {})
            with self.assertRaises(ValueError):
                trim.read_arrays(root / "motion.npz")

    def test_manual_review_file_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = self.make_dataset(base)
            a = self.arrays()
            a["joint_pos"][:150, 1] = 0.4
            np.savez(root / "review.npz", **a)
            m = json.loads((root / "manifest.json").read_text())
            m["clips"].append({"name": "review", "npz": "review.npz", "npz_frames": 200,
                               "npz_sha256": trim.sha256(root / "review.npz")})
            m["npz"]["total_frames"] = 400
            (root / "manifest.json").write_text(json.dumps(m))
            m, r = trim.inspect(root, {})
            self.assertEqual(r["clips"][1]["status"], "manual_review")
            trim.apply(root, base / "backup", m, r)
            self.assertEqual(trim.sha256(root / "review.npz"), trim.sha256(base / "backup/review.npz"))

    def test_absolute_npz_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make_dataset(Path(tmp))
            m = json.loads((root / "manifest.json").read_text())
            m["clips"][0]["npz"] = str(root / "motion.npz")
            (root / "manifest.json").write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                trim.inspect(root, {})

    def test_publication_failure_restores_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = self.make_dataset(base)
            m, r = trim.inspect(root, {})
            old_hash = trim.sha256(root / "manifest.json")
            rename = trim.os.rename
            def fail_publish(src, dst):
                if Path(src).name.startswith(".dataset_trim_stage_"):
                    raise OSError("test publication failure")
                rename(src, dst)
            with patch.object(trim.os, "rename", side_effect=fail_publish):
                with self.assertRaises(OSError):
                    trim.apply(root, base / "backup", m, r)
            self.assertEqual(trim.sha256(root / "manifest.json"), old_hash)
            self.assertFalse((base / "backup").exists())
            self.assertEqual(list(base.iterdir()), [root])

    def test_windows_match_loader_tail_policy(self):
        self.assertEqual(trim.window_count(1049, 50), 2)
        self.assertEqual(trim.window_count(1050, 50), 3)


if __name__ == "__main__":
    unittest.main()
