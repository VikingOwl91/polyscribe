#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.
set -euo pipefail

# Launch the interactive model downloader and select models to install

BIN=${BIN:-./target/release/polyscribe}
MODELS_DIR=${POLYSCRIBE_MODELS_DIR:-$PWD/models}
export POLYSCRIBE_MODELS_DIR="$MODELS_DIR"

mkdir -p "$MODELS_DIR"
"$BIN" --download-models
