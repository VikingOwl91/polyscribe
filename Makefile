# Lightweight examples-check: runs all examples/*.sh with --no-interaction -q and stubbed BIN
# This target does not perform network calls and never prompts for input.

.SHELL := /bin/bash

.PHONY: examples-check
examples-check:
	@set -euo pipefail; \
	shopt -s nullglob; \
	BIN_WRAPPER="$(PWD)/scripts/with_flags.sh"; \
	failed=0; \
	for f in examples/*.sh; do \
		echo "[examples-check] Running $$f"; \
		BIN="$$BIN_WRAPPER" bash "$$f" </dev/null >/dev/null 2>&1 || { \
			echo "[examples-check] FAILED: $$f"; failed=1; \
		}; \
	done; \
	if [[ $$failed -ne 0 ]]; then \
		echo "[examples-check] Some examples failed."; \
		exit 1; \
	else \
		echo "[examples-check] All examples passed (no interaction, quiet)."; \
	fi
