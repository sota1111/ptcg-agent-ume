#!/bin/bash
# Build the Kaggle archive defined by the shared ptcg-agent-core guide.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE="$REPO/submission.tar.gz"

for required in main.py deck.csv agents data cg; do
  if [ ! -e "$REPO/$required" ]; then
    echo "missing required submission path: $required" >&2
    exit 1
  fi
done

tar -C "$REPO" -czf "$ARCHIVE" \
  --exclude='__pycache__' --exclude='*.pyc' \
  main.py deck.csv agents data cg
gzip -t "$ARCHIVE"

listing="$(mktemp)"
trap 'rm -f -- "$listing"' EXIT
tar -tzf "$ARCHIVE" > "$listing"
grep -Fx 'main.py' "$listing" >/dev/null
grep -Fx 'deck.csv' "$listing" >/dev/null
if grep -E '(^|/)(\.env($|\.)|\.git/|vendor/|tests/|eval/|venv/|access_token|kaggle\.json|__pycache__/|.*\.pyc$)' "$listing"; then
  echo "submission contains a forbidden path" >&2
  exit 1
fi

echo "submission archive: $ARCHIVE"
