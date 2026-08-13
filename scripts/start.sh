#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
NO_BROWSER="${NO_BROWSER:-0}"

if ! command -v uv >/dev/null 2>&1; then
  echo "DailyDigest needs uv. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

mkdir -p data

echo "Syncing DailyDigest dependencies..."
uv sync --frozen --no-dev --inexact

if [[ "$NO_BROWSER" == "1" ]]; then
  exec uv run --no-sync dd start --host "$HOST" --port "$PORT" --no-browser
fi

exec uv run --no-sync dd start --host "$HOST" --port "$PORT"
