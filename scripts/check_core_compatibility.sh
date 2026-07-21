#!/bin/bash
# Verify the pinned common core used by the battle/submission workflow.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CORE="$REPO/vendor/ptcg-agent-core"

if [ ! -f "$CORE/package.json" ] || [ ! -f "$CORE/docs/kaggle-submission.md" ]; then
  echo "ptcg-agent-core is not initialized; run: git submodule update --init --recursive" >&2
  exit 1
fi

git -C "$CORE" rev-parse --verify HEAD >/dev/null
if [ ! -d "$CORE/node_modules" ]; then
  npm --prefix "$CORE" ci --ignore-scripts
fi
npm --prefix "$CORE" run typecheck
npm --prefix "$CORE" test
echo "ptcg-agent-core compatibility: OK ($(git -C "$CORE" rev-parse --short HEAD))"
