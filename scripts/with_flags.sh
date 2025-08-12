#!/usr/bin/env bash
# Wrapper that ensures --no-interaction -q are present, then delegates to the real BIN (stub by default)
set -euo pipefail

REAL_BIN=${REAL_BIN:-"$(dirname "$0")/bin_stub.sh"}

# Append flags if not already present in args
args=("$@")
need_no_interaction=1
need_quiet=1
for a in "${args[@]}"; do
  [[ "$a" == "--no-interaction" ]] && need_no_interaction=0
  [[ "$a" == "-q" || "$a" == "--quiet" ]] && need_quiet=0
done

if [[ $need_no_interaction -eq 1 ]]; then
  args=("--no-interaction" "${args[@]}")
fi
if [[ $need_quiet -eq 1 ]]; then
  args=("-q" "${args[@]}")
fi

# Never read stdin; prevent accidental blocking by redirecting from /dev/null
# Also advertise offline via env variables commonly checked by the app
export CI=1
export POLYSCRIBE_MODELS_BASE_COPY_DIR="${POLYSCRIBE_MODELS_BASE_COPY_DIR:-}" # leave empty by default

exec "$REAL_BIN" "${args[@]}" </dev/null
