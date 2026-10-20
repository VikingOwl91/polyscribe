#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.
set -euo pipefail

# Transcribe an audio/video file to JSON, TOML, and SRT into ./output
# Requires a model and --language; first run may prompt to download.

BIN=${BIN:-./target/release/polyscribe}
INPUT=${1:-samples/example.mp3}
OUTDIR=${OUTDIR:-output}

mkdir -p "$OUTDIR"
"$BIN" -v -o "$OUTDIR" --language "${LANGUAGE:-en}" "$INPUT"
echo "Done. See $OUTDIR for JSON/TOML/SRT files."
