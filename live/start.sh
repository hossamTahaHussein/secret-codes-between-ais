#!/usr/bin/env bash
# Start the live demo on http://localhost:8000 (GPU if it has room, otherwise CPU mode).
cd "$(dirname "$0")/.."
exec .venv/bin/python live/server.py "$@"
