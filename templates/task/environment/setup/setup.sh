#!/usr/bin/env bash
# Setup — the engine uploads all of environment/setup/ to /opt/ale/setup once the sandbox
# is up and runs `bash setup.sh` there as trusted root (cwd = that folder, network open),
# before the agent phase, within a fixed 1800 s. This folder is not part of the image's
# build context.
#
# One job here: put the agent-visible material (the practice grader, the agent's own
# editable copy of the environment) into the agent home, owned by the agent. Keep this
# stage dynamic-only: package installation and fixed downloads belong in
# environment/Dockerfile, and lint rejects them here.
#
# Everything under payload/ is staged, whatever it is called — add, rename or remove
# payload files without touching this script. Uploads ignore the modes in git (files
# arrive as 0644, folders as 0755), so a staged file that must be executable gets its
# mode here.
set -euo pipefail

HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"

# cp -r, NOT -a: with -a the `payload/.` form also copies the payload directory's own
# root ownership onto the agent home, locking the agent out of its own workspace.
cp -r payload/. "$HOME_DIR/"

# chown every top-level entry that came from payload/ (recursively) and nothing else: the
# home itself keeps the ownership the image gave it.
for entry in payload/* payload/.[!.]*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
