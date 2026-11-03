#!/usr/bin/env bash
# Setup (root, cwd = this staged folder, before the agent phase): stage the agent-visible payload into the home.
# Nothing hidden is used here: the hidden start states live only in verify/ref.
set -euo pipefail
umask 022
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
# cp -r, NOT -a: with -a the payload directory's root ownership would lock the agent out of its home.
cp -r payload/. "$HOME_DIR/"
for entry in payload/* payload/.[!.]*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
  # fixed, content-independent timestamps (no ordering signal)
  find "$HOME_DIR/$(basename "$entry")" -exec touch -h -d "2026-01-01 00:00:00" {} +
done
# the agent's images, fetched into the image at build time (image/assets.sha256, docs/task-assets.md)
if [ -d /opt/task-assets/payload ]; then
  cp -r /opt/task-assets/payload/. "$HOME_DIR/"
  for entry in /opt/task-assets/payload/*; do
    [ -e "$entry" ] || continue
    chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
    find "$HOME_DIR/$(basename "$entry")" -exec touch -h -d "2026-01-01 00:00:00" {} +
  done
fi
mkdir -p "$HOME_DIR/submission/plans"
chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
