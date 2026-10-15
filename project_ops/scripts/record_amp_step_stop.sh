#!/bin/bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec /data/zym/conda/envs/isaaclab_zym/bin/python \
    "$repo_root/robo_newton/scripts/mujoco/sim2sim_rp1_amp.py" \
    --duration 10.5 --max-speed 4 --abrupt-start-at 0.5 --abrupt-stop-at 5.5 \
    --width 1280 --height 720 "$@"
