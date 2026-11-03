#!/usr/bin/env bash
# The reference build: the bay and its fleet authored as a USD stage by reference/build_scene.py, plus the reference
# fleet controller (reference/controller.py + vehicle.py behind run.sh: the flying robots' hoist and the vehicle's
# program) (runs Isaac Sim headless on the sandbox's GPU, as the agent's account, within the agent budget).
set -euo pipefail
SUB="${ALE_HOME:-/home/user}/submission"
mkdir -p "$SUB"
/opt/venv/bin/python3 "$(dirname "$0")/reference/build_scene.py" --out "$SUB/scene.usda"
cp "$(dirname "$0")/reference/controller.py" "$(dirname "$0")/reference/vehicle.py" "$(dirname "$0")/reference/run.sh" "$SUB/"
echo "[oracle] scene and controller ready in $SUB" >&2
