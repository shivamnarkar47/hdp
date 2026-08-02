#!/usr/bin/env bash
# kala installer — Linux/macOS
#
# Overrides (env): KALA_REPO_URL, KALA_INSTALL_DIR, KALA_BIN_DIR
set -euo pipefail

REPO_URL="${KALA_REPO_URL:-https://github.com/shivamnarkar47/hdp.git}"
INSTALL_DIR="${KALA_INSTALL_DIR:-$HOME/.local/share/kala}"
BIN_DIR="${KALA_BIN_DIR:-$HOME/.local/bin}"

# version_ge A B — true if dotted numeric version A >= B
version_ge() {
  local -a a b
  IFS=. read -ra a <<< "$1"
  IFS=. read -ra b <<< "$2"
  local i
  for i in "${!a[@]}"; do
    if (( i >= ${#b[@]} )); then return 0; fi
    if (( a[i] > b[i] )); then return 0; fi
    if (( a[i] < b[i] )); then return 1; fi
  done
  # a exhausted: a >= b iff a has no fewer components
  (( ${#a[@]} >= ${#b[@]} ))
}

# --- Python check (>= 3.12) -------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 was not found on PATH." >&2
  echo "Install Python 3.12 or newer (https://www.python.org/downloads/) and re-run." >&2
  exit 1
fi
PY_VERSION="$(python3 --version 2>&1 | awk '{print $2}')"
if ! version_ge "$PY_VERSION" "3.12"; then
  echo "Error: Python $PY_VERSION is too old; kala requires Python >= 3.12." >&2
  echo "Install Python 3.12 or newer (https://www.python.org/downloads/) and re-run." >&2
  exit 1
fi
echo "Found python3 $PY_VERSION"

# --- Fetch the code ----------------------------------------------------------
mkdir -p "$INSTALL_DIR"
if [[ -d "$INSTALL_DIR/.git" ]]; then
  echo "Updating existing installation at $INSTALL_DIR"
  git -C "$INSTALL_DIR" pull --ff-only
elif command -v git >/dev/null 2>&1; then
  echo "Cloning $REPO_URL into $INSTALL_DIR"
  git clone "$REPO_URL" "$INSTALL_DIR"
else
  echo "git not found; downloading a tarball of the main branch"
  curl -fsSL "${REPO_URL%.git}/archive/refs/heads/main.tar.gz" \
    | tar -xz --strip-components=1 -C "$INSTALL_DIR"
fi

# --- Virtual environment -----------------------------------------------------
cd "$INSTALL_DIR"
if command -v uv >/dev/null 2>&1; then
  if [[ ! -x "$INSTALL_DIR/.venv/bin/python" ]]; then
    echo "Creating virtual environment with uv"
    uv venv "$INSTALL_DIR/.venv"
  fi
  uv pip install --python "$INSTALL_DIR/.venv/bin/python" .
else
  echo "Creating virtual environment with python3 -m venv"
  python3 -m venv "$INSTALL_DIR/.venv"
  "$INSTALL_DIR/.venv/bin/pip" install .
fi

# --- Launcher ----------------------------------------------------------------
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/kala" <<EOF
#!/bin/sh
exec "$INSTALL_DIR/.venv/bin/python" -m harness "\$@"
EOF
chmod +x "$BIN_DIR/kala"

# --- PATH hint ---------------------------------------------------------------
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  echo "NOTE: $BIN_DIR is not on your PATH. Add it with:"
  echo "  export PATH=\"$BIN_DIR:\$PATH\""
fi

# --- Success ----------------------------------------------------------------
echo
echo "kala installed successfully."
echo "  Install dir: $INSTALL_DIR"
echo "  Launcher:    $BIN_DIR/kala"
echo
echo "Try:  kala --help"
echo "API key: set OPENCODE_API_KEY in your environment, or let the harness read the omp auth store."
exit 0
