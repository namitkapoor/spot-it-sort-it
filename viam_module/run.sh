#!/usr/bin/env bash
# Local Viam module entrypoint. Builds its own environment on first start (no python3-venv needed: uses a
# standalone uv binary). Everything printed here shows up in the machine logs.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f .venv/.sortit-ready ]; then
  rm -rf .venv  # a half-built env from an earlier failed start
  if [ ! -x bin/uv ]; then
    echo "sortit: installing uv into $PWD/bin" >&2
    mkdir -p bin
    (curl -LsSf https://astral.sh/uv/install.sh || wget -qO- https://astral.sh/uv/install.sh) \
      | env UV_INSTALL_DIR="$PWD/bin" UV_NO_MODIFY_PATH=1 sh >&2
  fi
  echo "sortit: creating environment" >&2
  bin/uv venv --python 3.12 .venv >&2
  VIRTUAL_ENV="$PWD/.venv" bin/uv pip install -q -r requirements.txt >&2
  touch .venv/.sortit-ready
  echo "sortit: environment ready" >&2
fi
VIRTUAL_ENV="$PWD/.venv" bin/uv pip install -q -r requirements.txt >&2  # no-op when already satisfied
exec .venv/bin/python main.py "$@"
