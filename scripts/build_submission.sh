#!/bin/bash
# Pack the submission entry point and all of its bundled runtime dependencies.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
[ -d cg ] && [ -d agents ] && [ -d data ] && [ -f main.py ] && [ -f deck.csv ] || {
  echo "missing cg/, agents/, data/, main.py, or deck.csv (run setup_engine.sh)"
  exit 1
}
tar --exclude='__pycache__' --exclude='*.pyc' -czf submission.tar.gz main.py deck.csv agents data cg
echo "wrote $REPO/submission.tar.gz"; tar -tzf submission.tar.gz | head
