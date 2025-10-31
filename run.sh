
set -euo pipefail


MODE="${MODE:-both}"
CONFIG="${CONFIG:-config.yaml}"
PYTHON_BIN="${PYTHON:-python3}"


if [ -d ".venv" ]; then

  source .venv/bin/activate
fi

if [ "$#" -eq 0 ]; then
  exec "$PYTHON_BIN" evaluate_gcs.py --config "$CONFIG" --mode "$MODE"
else
  exec "$PYTHON_BIN" evaluate_gcs.py "$@"
fi
