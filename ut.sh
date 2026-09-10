#!/bin/sh
# Use the active, configured DeePTB environment.
set -eu
cd "$(dirname "$0")"
exec python tools/test.py "$@"
