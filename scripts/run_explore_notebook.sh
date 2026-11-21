#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
exec uv run jupyter lab --ServerApp.root_dir=. --LabApp.extension_manager=readonly notebooks/01_data_audit.ipynb "$@"
