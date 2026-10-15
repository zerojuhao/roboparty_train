#!/usr/bin/env python3
"""Export the H200-2 mjlab rp1_simple training terrain for MuJoCo sim2sim.

Requires the UFO checkout and its venv (mjlab). Does not touch training GPUs.

  /home/zym/UFO/.venv/bin/python robo_newton/scripts/tools/export_pufo_h200_terrain.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

UFO_ROOT = Path("/home/zym/UFO")
OUT_DIR = Path(__file__).resolve().parents[1] / "mujoco" / "pufo_h200_rp1_simple"


def main() -> None:
    sys.path.insert(0, str(UFO_ROOT))
    import mujoco
    from mjlab.terrains.terrain_generator import TerrainGenerator
    from humanoidverse.terrains.rp1_simple import (
        RP1_TERRAIN_BORDER_PATCHES,
        add_rp1_outer_random_patches,
        add_rp1_outer_walls,
        make_rp1_simple_generator_cfg,
        rp1_terrain_outer_extent,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = make_rp1_simple_generator_cfg()
    generator = TerrainGenerator(cfg, device="cpu")
    spec = mujoco.MjSpec()
    generator.compile(spec)
    add_rp1_outer_random_patches(spec, cfg, RP1_TERRAIN_BORDER_PATCHES)
    add_rp1_outer_walls(spec, cfg, RP1_TERRAIN_BORDER_PATCHES)

    for geom in list(spec.body("terrain").geoms):
        geom.material = ""
        geom.rgba[:] = (0.5, 0.5, 0.5, 1.0)
        geom.friction[:] = (0.9, 0.02, 0.002)
        geom.condim = 3
        geom.group = 0
    for material in list(spec.materials):
        spec.delete(material)
    for texture in list(spec.textures):
        spec.delete(texture)

    xml_path = OUT_DIR / "terrain.xml"
    xml_path.write_text(spec.to_xml())
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    columns = list(cfg.sub_terrains.keys())
    stairs_col = columns.index("low_stairs_down")
    level = 5
    meta = {
        "source": "H200-2 /home/zym/UFO/humanoidverse/terrains/rp1_simple.py",
        "generator": (
            "mjlab TerrainGenerator + rp1_simple "
            "(10x7 core grid + 2 outer random patch rings + outer walls)"
        ),
        "seed": cfg.seed,
        "tile_size": list(cfg.size),
        "num_rows": cfg.num_rows,
        "num_cols": len(columns),
        "border_patch_rings": RP1_TERRAIN_BORDER_PATCHES,
        "outer_extent_m": list(rp1_terrain_outer_extent(cfg, RP1_TERRAIN_BORDER_PATCHES)),
        "terrain_types": columns,
        "stairs_reset": {
            "terrain_type": "low_stairs_down",
            "terrain_level": level,
            "origin": generator.terrain_origins[level, stairs_col].tolist(),
        },
        "terrain_origins": generator.terrain_origins.tolist(),
        "ngeom": int(model.ngeom),
        "nhfield": int(model.nhfield),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    size = xml_path.stat().st_size
    print("Wrote", xml_path, "bytes", size)
    print("ngeom", model.ngeom, "nhfield", model.nhfield)
    print("stairs_reset", meta["stairs_reset"])


if __name__ == "__main__":
    main()
