#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.
set -euo pipefail

# Verify/update local models non-interactively (useful in CI)

BIN=${BIN:-./target/release/polyscribe}
MODELS_DIR=${POLYSCRIBE_MODELS_DIR:-$PWD/models}
export POLYSCRIBE_MODELS_DIR="$MODELS_DIR"

mkdir -p "$MODELS_DIR"
"$BIN" --update-models --no-interaction -q

echo "Models updated in $MODELS_DIR"
