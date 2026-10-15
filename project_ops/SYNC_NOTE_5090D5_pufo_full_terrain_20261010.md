# Sync note for 5090D-5 — PUFO sim2sim full H200 terrain (2026-10-10)

## Changed on 4090 host (this machine)
- `robo_newton/scripts/mujoco/sim2sim_rp1_pufo.py`
  - Stairs panel now attaches full H200-2 mjlab `rp1_simple` map instead of
    flat ground + single 5x5 stairs tile + landing pads.
  - Reset places the robot on `low_stairs_down` level-5 origin from meta.json.
- `robo_newton/scripts/mujoco/pufo_h200_rp1_simple/`
  - `terrain.xml` (~504 KB): full compiled map (2290 geoms, 52 heightfields)
  - `meta.json`: origins, type columns, stairs reset origin, extents
- `robo_newton/scripts/tools/export_pufo_h200_terrain.py`
  - Regenerator using `/home/zym/UFO/.venv` + `humanoidverse.terrains.rp1_simple`
- `project_ops/scripts/test_sim2sim_rp1_pufo.py` + `README_pufo.md` updated

## Not changed / do not touch
- AMP training `RP1-AMP-Path` (`rp1_amp_path_speed4_runloop_fresh_longeval_20261010`) — leave running.

## Sync to 5090D-5
Copy at least:
1. `robo_newton/scripts/mujoco/sim2sim_rp1_pufo.py`
2. `robo_newton/scripts/mujoco/pufo_h200_rp1_simple/` (entire dir)
3. `robo_newton/scripts/tools/export_pufo_h200_terrain.py`
4. `project_ops/scripts/test_sim2sim_rp1_pufo.py`
5. `project_ops/scripts/README_pufo.md`

If 5090D-5 has no UFO checkout at `/home/zym/UFO`, ship the prebuilt
`pufo_h200_rp1_simple/` assets; regenerating requires UFO+mjlab.

## Verify
```bash
OMP_NUM_THREADS=1 /data/conda/envs/isaaclab_zym/bin/python -m unittest \
  discover -s project_ops/scripts -p test_sim2sim_rp1_pufo.py
```
