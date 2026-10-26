#!/usr/bin/env bash
# Commit everything with a message, then mirror to the local remote.
# Usage: scripts/save.sh "what changed"
set -euo pipefail
msg="${1:-checkpoint $(date '+%H:%M')}"
git add -A
if git diff --cached --quiet; then echo "nothing to commit"; exit 0; fi
git -c user.name="Aoyi Feng" -c user.email="aofeng@student.ethz.ch" commit -q -m "$msg"
git push -q origin HEAD 2>/dev/null || true
git push -q --tags 2>/dev/null || true
echo "saved: $(git log -1 --format='%h %s')"
