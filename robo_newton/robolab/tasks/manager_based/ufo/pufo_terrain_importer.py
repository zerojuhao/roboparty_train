"""PUFO guard terrain and stair probes using the shared native importer."""

import numpy as np

from isaaclab.terrains import TerrainImporter as BaseTerrainImporter
from robolab.terrains.native_importer import NativeTerrainImporter, box_bounds


class PUFOTerrainImporter(NativeTerrainImporter):
    def __init__(self, cfg):
        import isaaclab.sim as sim_utils
        from isaaclab_newton.physics import NewtonCfg

        cfg = cfg.copy()
        if isinstance(sim_utils.SimulationContext.instance().cfg.physics, NewtonCfg) and cfg.terrain_generator is not None:
            cfg.terrain_generator.use_cache = False
        super().__init__(cfg)

    def _import_observation_mesh(self, name, mesh):
        # Preserve the original guard mesh, without vertex welding.
        BaseTerrainImporter.import_mesh(self, name, mesh)

    def validate_solver_geometry(self, model):
        """Fail if USD visibility/filtering silently dropped an authored collider."""
        import mujoco

        actual = super().validate_solver_geometry(model)
        # Probe both sides of stair edges, where a coarse heightfield smears treads.
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        errors = []
        names = list(self.cfg.terrain_generator.sub_terrains)
        for name in ("low_stairs_up", "low_stairs_down"):
            if name not in names:
                continue
            center = self.terrain_generator.terrain_origins[
                min(5, self.cfg.terrain_generator.num_rows - 1), names.index(name)]
            for offset in (-1.99, -1.71, -1.69, -1.41, -1.39, -1.11, -1.09, -.81, -.79, 0., .79, .81, 1.69, 1.71):
                xy = center[:2] + (offset, 0.)
                hit = ((xy >= self.authored_boxes[:, 0, :2]) &
                       (xy <= self.authored_boxes[:, 1, :2])).all(-1)
                expected = self.authored_boxes[hit, 1, 2].max()
                origin = np.array([*xy, 10.])
                distance = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]),
                                         None, True, -1, np.empty(1, dtype=np.int32))
                errors.append(abs((origin[2] - distance) - expected) if distance >= 0 else np.inf)
        max_error = max(errors, default=0.)
        if max_error > 1e-4:
            raise RuntimeError(f"PUFO stair collision differs from authored boxes by {max_error} m")
        print(f"[PUFO terrain] solver verified: boxes={actual['box']}, heightfields={actual['heightfield']}, "
              f"stair_probe_max_error={max_error:.3g} m", flush=True)
