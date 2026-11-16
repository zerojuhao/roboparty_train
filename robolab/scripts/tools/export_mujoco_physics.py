#!/usr/bin/env python3
"""Export a robot physics asset from a nominal, single-robot Newton MuJoCo model.

No policy or checkpoint is loaded. Supply a nonrandomized model with one plane.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

import mujoco
from robolab.utils.mujoco_physics import export_physics_asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Nominal compiled .mjb model or .xml")
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--physics-dt", type=float, required=True, help="Actual solver substep dt in seconds")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    loader = mujoco.MjModel.from_binary_path if args.model.suffix == ".mjb" else mujoco.MjModel.from_xml_path
    profile = export_physics_asset(loader(str(args.model)), args.urdf, args.output,
                                   data_root=args.data_root, timestep=args.physics_dt)
    print(f"Exported {len(profile['joints'])} joints and {len(profile['geometries'])} geometries: {args.output}")


if __name__ == "__main__":
    main()
