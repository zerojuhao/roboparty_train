#!/usr/bin/env bash
set -eo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$OPS_DIR/.."
source "$OPS_DIR/activate_5090d6.sh"
exec python project_ops/scripts/run_pufo_velocity_ab.py "$@"
