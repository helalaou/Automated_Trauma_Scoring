
set -euo pipefail


TASK="${TASK:-6bin}"
CONFIG="${CONFIG:-config.yaml}"
PYTHON_BIN="${PYTHON:-python3}"


if [ -d ".venv" ]; then

  source .venv/bin/activate
fi

if [ "$#" -eq 0 ]; then
  exec "$PYTHON_BIN" evaluate_gcs.py --config "$CONFIG" --task "$TASK"
else
  exec "$PYTHON_BIN" evaluate_gcs.py "$@"
fi
