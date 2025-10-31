#!/usr/bin/env bash
set -euo pipefail

# Config
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR=".venv"

# Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
  echo "[install] Creating virtual environment in $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Upgrade pip and install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "\n[install] Done. Activate with: source $VENV_DIR/bin/activate"
echo "[install] Then run: ./run --mode both"
