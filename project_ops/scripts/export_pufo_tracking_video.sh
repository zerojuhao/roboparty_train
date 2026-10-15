#!/usr/bin/env bash
# Three-panel CPU sim2sim export, following H200-2 export_tracking_video.sh.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${PYTHON_BIN:=/data/conda/envs/isaaclab_zym/bin/python}"
: "${MOTION_NAME:=dance1_subject2}"
: "${START_SECONDS:=0}"
: "${FPS:=25}"
: "${CAMERA_DISTANCE:=3.4}"
: "${CAMERA_AZIMUTH:=55}"
: "${CAMERA_ELEVATION:=-25}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

args=(--motion "$MOTION_NAME" --start "$START_SECONDS" --fps "$FPS"
      --camera-distance "$CAMERA_DISTANCE" --camera-azimuth "$CAMERA_AZIMUTH"
      --camera-elevation "$CAMERA_ELEVATION")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "$CHECKPOINT")
[[ -z "${DURATION_SECONDS:-}" ]] || args+=(--duration "$DURATION_SECONDS")
[[ -z "${VIDEO_PATH:-}" ]] || args+=(--output "$VIDEO_PATH")

echo '[export-video] reference | flat | H200 low_stairs_down level 5; no push, DR, observation noise, or automatic reset'

REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
exec "$PYTHON_BIN" -u "$REPO_ROOT/robo_newton/scripts/mujoco/sim2sim_rp1_pufo.py" "${args[@]}" "$@"
