#!/usr/bin/env bash
# Lightweight stub for examples-check: simulates the PolyScribe CLI without I/O or network
# - Accepts any arguments
# - Exits 0
# - Produces no output unless VERBOSE_STUB=1
# - Never performs network operations
# - Never reads from stdin
set -euo pipefail

if [[ "${VERBOSE_STUB:-0}" == "1" ]]; then
  echo "[stub] polyscribe $*" 1>&2
fi

# Behave quietly if -q/--quiet is present by default (no output)
# Honor --help/-h: print minimal usage if verbose requested
if [[ "${VERBOSE_STUB:-0}" == "1" ]]; then
  for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
      echo "PolyScribe stub: no-op (examples-check)" 1>&2
      break
    fi
  done
fi

# Always succeed quietly
exit 0
