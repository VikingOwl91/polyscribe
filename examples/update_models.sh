#!/usr/bin/env bash
set -euo pipefail

# Verify/update local models non-interactively (useful in CI)

BIN=${BIN:-./target/release/polyscribe}
MODELS_DIR=${POLYSCRIBE_MODELS_DIR:-$PWD/models}
export POLYSCRIBE_MODELS_DIR="$MODELS_DIR"

mkdir -p "$MODELS_DIR"
"$BIN" --update-models --no-interaction -q

echo "Models updated in $MODELS_DIR"
