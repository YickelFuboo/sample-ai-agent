#!/bin/bash
cd "$(dirname "$0")"
if command -v poetry &>/dev/null; then
    poetry run start
else
    python -m app.main
fi
