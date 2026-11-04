#!/usr/bin/env bash
# The verify stage, as root, in a separate verifier sandbox (task.toml): a fresh one from the task image, with tests/ at
# /opt/ale/tests and only /home/user/submission restored. setup.sh has not run here (no exploration service, no ledger),
# and nothing else the agent left in its own sandbox is present. grader.py freezes the submission and drives it as the
# agent account on fresh private scenes.
set -euo pipefail
cd "$(dirname "$0")"
exec "${ALE_VERIFY_PYTHON:-python3}" grader.py
