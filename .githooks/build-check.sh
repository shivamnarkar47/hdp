#!/usr/bin/env bash
# Build-check hook: compile all modules, run the full unit-test suite, and
# verify the `kaal` entry point. Shared by pre-commit and pre-push.
# Skip everything with KAAL_SKIP_HOOKS=1; skip with a warning if .venv is absent.
set -euo pipefail

if [ "${KAAL_SKIP_HOOKS:-0}" = "1" ]; then
  echo "hooks: skipped (KAAL_SKIP_HOOKS=1)"
  exit 0
fi

if [ ! -f .venv/bin/python ]; then
  echo "hooks: WARNING .venv not found — skipping build check (commit proceeds)"
  exit 0
fi

# 1. Fast syntax/bytecode gate
./.venv/bin/python -m compileall -q harness tests

# 2. Full unit-test suite — fail the commit/push if it fails.
#    pipefail makes the pipeline status reflect the test run, not `tail`.
if ./.venv/bin/python -m unittest discover -s tests 2>&1 | tail -3; then
  echo "hooks: tests OK"
else
  echo "hooks: tests FAILED" >&2
  exit 1
fi

# 3. Entry point proves the binary builds and reports the right version
test "$(./.venv/bin/kaal --version)" = "kaal 0.1.0"

echo "hooks: build check passed"
