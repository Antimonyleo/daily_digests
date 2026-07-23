#!/usr/bin/env bash
# One-command setup + launch for DailyDigest.
#
# Installs uv (the Python toolchain manager) if it is missing, then starts the
# local web app and opens it in your browser. Works on Linux, macOS, and
# Windows via WSL. After the first run you can just use ./scripts/start.sh.
#
#   bash scripts/install.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (one-time, no admin rights needed)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs to one of these; make it visible to this shell session.
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "" >&2
  echo "uv was installed but is not on PATH in this shell." >&2
  echo "Close this window, open a NEW terminal, and run:  ./scripts/start.sh" >&2
  echo "(or see https://docs.astral.sh/uv/getting-started/installation/)" >&2
  exit 1
fi

echo "Starting DailyDigest..."
exec ./scripts/start.sh
