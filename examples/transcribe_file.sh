#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <COPYRIGHT HOLDER>. All rights reserved.
set -euo pipefail

# Transcribe an audio/video file to JSON and SRT into ./output
# Requires a model; first run may prompt to download.

BIN=${BIN:-./target/release/polyscribe}
INPUT=${1:-samples/example.mp3}
OUTDIR=${OUTDIR:-output}

mkdir -p "$OUTDIR"
"$BIN" -v -o "$OUTDIR" "$INPUT"
echo "Done. See $OUTDIR for JSON/SRT files."
