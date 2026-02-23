#!/usr/bin/env bash
set -e
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
exec python -m uvicorn src.api.app:app --host "$HOST" --port "$PORT"
