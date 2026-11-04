#!/usr/bin/env bash
# The verify stage's entry point. Runs as root in a SEPARATE verifier sandbox (task.toml
# [verifier] environment_mode = "separate"): a fresh sandbox from the task image, which the
# agent never touched, with only the declared artifact (/home/user/submission) restored.
# While the agent worked this folder was absent; now it holds the hidden grader and anchor.
#
# Root-stage hygiene first (the image sets HOME=/home/user): root's own HOME, no user
# site-packages, a private root-only temp, and an isolated interpreter (-I ignores PYTHON*
# env vars and leaves the cwd off sys.path), so nothing the restored submission carries can
# steer the root grader. verify.py adds its own directory to sys.path explicitly, so the
# grader package and env.py still import under -I.
#
# `$ALE_VERIFY_PYTHON` and nothing else: the engine stages `ale_verify` into the first
# 3.12+ interpreter it finds (`python3`, then `python3.12`) and names it here, so any
# other interpreter would import from a python that package was never installed into.
# Run by hand outside the engine, the variable is unset and `python3` stands in.
set -euo pipefail
umask 077
export HOME=/root PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
STAGE_TMP="$(mktemp -d)"
chmod 700 "$STAGE_TMP"
export TMPDIR="$STAGE_TMP" XDG_CACHE_HOME="$STAGE_TMP/cache" XDG_CONFIG_HOME="$STAGE_TMP/config"
unset PYTHONPATH PYTHONSTARTUP
cd "$(dirname "$0")"
exec "${ALE_VERIFY_PYTHON:-python3}" -I verify.py
