#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Two jobs:
#   1. the givens: payload/ into the agent home, owned by the agent, plus the robot model the image holds
#      (/opt/arb-assets/unitree_g1 -> ~/given/robot/unitree_g1);
#   2. the practice lab: lab/ (the lab server and its own copy of the furniture cell) installed at /opt/arb-lab, owned
#      by the lab account `arbcell` with mode 0700 (the agent can read neither the cell's code nor the lab's state),
#      and the lab server started as that account. It serves practice episodes of the cell, never privileged, on the
#      Unix socket /run/furniture-cell/practice.sock; setup waits (at most 60 s) until the socket is up.
# The grader never reads anything set up here: verification is separate (task.toml), and tests/test.sh builds the
# grader's own ROOT-OWNED read-only copy of the givens (/opt/arb) in the fresh verifier sandbox, from
# tests/setup_payload/ (a byte copy of payload/ here: change both together, then tests/given_hashes.json) and the
# image's robot model.
# Dynamic-only: packages and the robot model live in environment/Dockerfile.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
G1_DIR="${ARB_G1_DIR:-/opt/arb-assets/unitree_g1}"
LAB_DIR="${ARB_LAB_DIR:-/opt/arb-lab}"
LAB_USER=arbcell
SOCK_DIR=/run/furniture-cell
SOCK="$SOCK_DIR/practice.sock"
PY=/opt/venv/bin/python3

# ---- 1. the agent's givens --------------------------------------------------------------------------------------------
# cp -r, NOT -a: with -a the `payload/.` form also copies the payload directory's own root ownership onto the agent
# home, locking the agent out of its own workspace. Every top-level entry that came from payload/ is chowned
# (recursively), nothing else: the home itself keeps the ownership the image gave it.
cp -r payload/. "$HOME_DIR/"
for entry in payload/* payload/.[!.]*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/given/robot"
rm -rf "$HOME_DIR/given/robot/unitree_g1"
cp -r "$G1_DIR" "$HOME_DIR/given/robot/unitree_g1"
chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/given"
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"

# ---- 2. the practice lab ----------------------------------------------------------------------------------------------
id -u "$LAB_USER" >/dev/null                         # the account comes with the image
[ -f lab/lab_server.py ] && [ -f lab/lab_config.json ] && [ -f lab/furniture_cell/server.py ] || {
  echo "[setup] lab/ is incomplete: lab_server.py, lab_config.json and furniture_cell/ are required" >&2
  exit 1
}
if [ -d "$LAB_DIR" ]; then                           # a second run of this script: stop the previous lab first
  pkill -TERM -u "$LAB_USER" 2>/dev/null || true
  sleep 2
  pkill -KILL -u "$LAB_USER" 2>/dev/null || true
fi
rm -rf "$LAB_DIR" "$SOCK_DIR"
mkdir -p "$LAB_DIR"
cp -r lab/. "$LAB_DIR/"
find "$LAB_DIR" -name '__pycache__' -type d -prune -exec rm -rf {} +
mkdir -p "$LAB_DIR/state"
touch "$LAB_DIR/lab.log" "$LAB_DIR/lab_server.log"
chown -R "$LAB_USER:$LAB_USER" "$LAB_DIR"
chmod -R u+rwX,go-rwx "$LAB_DIR"
mkdir -p "$SOCK_DIR"
chown "$LAB_USER:$LAB_USER" "$SOCK_DIR"
chmod 0755 "$SOCK_DIR"

# The daemon: its own session (setsid; the stage's process group is killed when the stage times out), a minimal
# environment (env -i: nothing of the stage's ALE_* variables), every std stream redirected (the engine waits for the
# stage's pipes to close; an inherited pipe would hang setup), one BLAS thread (the server forks an episode process
# per episode) and four rasteriser threads for its camera frames.
cd "$LAB_DIR"
setsid nohup runuser -u "$LAB_USER" -- env -i \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin HOME="$LAB_DIR/state" LANG=C.UTF-8 \
    MPLCONFIGDIR="$LAB_DIR/state/mpl" MPLBACKEND=Agg MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    ARB_G1_DIR="$G1_DIR" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=4 \
    "$PY" -s -B "$LAB_DIR/lab_server.py" --socket "$SOCK" --config "$LAB_DIR/lab_config.json" \
        --state "$LAB_DIR/state" --log "$LAB_DIR/lab_server.log" --allow-user "$AGENT_USER" \
    >>"$LAB_DIR/lab.log" 2>&1 </dev/null &
LAB_PID=$!
cd - >/dev/null
for _ in $(seq 1 120); do
  [ -S "$SOCK" ] && break
  kill -0 "$LAB_PID" 2>/dev/null || break
  sleep 0.5
done
if [ ! -S "$SOCK" ]; then
  echo "[setup] the practice lab did not come up; its logs:" >&2
  tail -n 40 "$LAB_DIR/lab_server.log" "$LAB_DIR/lab.log" >&2 || true
  pkill -KILL -u "$LAB_USER" 2>/dev/null || true
  exit 1
fi
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER); practice lab on $SOCK" >&2
