#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "$ALE_VERIFY_PYTHON" grader.py
