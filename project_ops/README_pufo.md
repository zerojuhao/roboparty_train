# RP1 PUFO Sim2Sim Recording

Recommended export entrypoint (following H200-2's export script):

```bash
bash project_ops/scripts/export_pufo_tracking_video.sh
```

Optional environment variables: `CHECKPOINT`, `MOTION_NAME`, `START_SECONDS`,
`DURATION_SECONDS`, `VIDEO_PATH`, `FPS`, and
`CAMERA_DISTANCE/AZIMUTH/ELEVATION`.
The default records the whole motion with a 3.4 m, 55 degree azimuth,
-25 degree elevation camera. H.264 CRF 18 and faststart match the H200 export.
Final videos are published only after successful encoding and full decoding.
The three requested panels are retained instead of H200's four-panel layout.

The corrected exporter uses world angular velocity rotated into `base_link`.
MuJoCo `mj_objectVelocity(mjOBJ_BODY, flg_local=1)` instead uses the inertial
principal axes and must not be fed directly to the actor. Old videos without
the `_linkframe` suffix were recorded with that incorrect observation.

`sim2sim_rp1_pufo.py` loads the newest published checkpoint in the newest
RP1-PUFO run by default. It runs the saved actor, depth encoder, backward
encoder, and observation normalizer on CPU. It does not load replay, modify
the checkpoint, or stop training. MuJoCo EGL rendering still uses a small
amount of GPU memory.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  /data/conda/envs/isaaclab_zym/bin/python \
  robo_newton/scripts/mujoco/sim2sim_rp1_pufo.py
```

ONNX deployment export is handled by `robo_newton/scripts/rsl_rl/play.py`,
independently of video recording. Pass `--export-only` with the usual PUFO
task/checkpoint arguments. The export directory contains `actor.onnx`,
`depth_encoder.onnx`, `backward_encoder.onnx`, and `pufo_onnx.json`.
`backward_encoder.onnx` is the context graph: it emits `z` from the privileged
observations, which is then supplied to `actor.onnx`.

Use `--checkpoint /absolute/path/model_123.pt` to pin a model, `--start 0
--duration 30` for a segment, or `--output /absolute/path/comparison.mp4`
to select the output. Existing videos are not overwritten. By default, the
entire `dance1_subject2` motion is recorded at 25 fps in a 1920x720 video:

1. Kinematic dataset motion (not a policy rollout).
2. MuJoCo policy tracking on flat ground.
3. MuJoCo policy tracking on real staircase collision geometry.

The stairs panel attaches the full H200-2 mjlab `rp1_simple` training
terrain (`pufo_h200_rp1_simple/terrain.xml`): a seeded 10x7 curriculum grid
of 5 m tiles (flat, perlin_rough, low_stairs_up/down, low_platforms,
hf_pyramid_slope_inv, boxes), plus two outer random-patch rings and walls.
The robot is placed on the `low_stairs_down` level-5 spawn origin from that
map. Regenerate the asset with UFO's venv:

```bash
/home/zym/UFO/.venv/bin/python   robo_newton/scripts/tools/export_pufo_h200_terrain.py
```

a reference export. New recordings still use the `_h200stairs` suffix.

The actor receives the same state/history ordering and scaled previous
actions as training. Tracking uses B(s[t+1]); heading and velocity come
from reference frame t. Depth is ray-cast from the waist camera, cropped
to 18x36, blurred, normalized, quantized to uint8, and sampled oldest-first
from the 37-frame history. The deterministic delay is one frame. Control
runs at 50 Hz, PD at 200 Hz, and physics at 400 Hz. Nominal PD gains,
armature, friction, and the motoring/braking torque envelope come from
the saved environment configuration.

Deployment rollouts disable observation noise, random torque, randomized
actuator delay, and automatic resets. Consequently this is a sim2sim test,
not an exact reproduction of the training evaluator's randomized conditions.
The full continuous motion also differs from training's 10-second clips.
Falls and tracking failures are retained in the recording.

Outputs beside the MP4:

- `.json`: checkpoint identity, deployment settings, FK validation, and joint errors.
- `.npz`: actual/reference joint trajectories, actual root positions, and actor actions at 50 Hz.

Joint L2 displayed in the video is instantaneous joint-space error, not EMD.

```bash
OMP_NUM_THREADS=1 /data/conda/envs/isaaclab_zym/bin/python -m unittest \
  discover -s project_ops/scripts -p test_sim2sim_rp1_pufo.py
```
