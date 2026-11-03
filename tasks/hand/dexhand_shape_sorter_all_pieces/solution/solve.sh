#!/usr/bin/env bash
# Privileged reference: replays the plans stored in oracle/ref/plans/. They were produced offline by a ground-truth-state
# planner (scripted Wuji grasp, re-orientation and insertion primitives, servo alignment on the true state and roll-back
# search over grasp variants; no refinement backend, see each plan's `_provenance`), using the TRUE hidden start states
# (piece poses and cube offset), the TRUE hidden physics and the TRUE hidden calibration error of each scene (the plans
# pre-compensate the calibration error). The oracle reads only its own staged folder and writes the submission; it
# needs no network and no simulation.
set -euo pipefail
SUB="${ALE_HOME:-/home/user}/submission"
mkdir -p "$SUB/plans"
cp "$(dirname "$0")"/ref/plans/inst_*.json "$SUB/plans/"
echo "[oracle] wrote $(ls "$SUB/plans" | wc -l) plans" >&2
