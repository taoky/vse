#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

export PADDLE_PDX_CACHE_HOME="$project_dir/.cache/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export VSE_VIDEOSUBFINDER_PATH=${VSE_VIDEOSUBFINDER_PATH:-"$project_dir/.local/videosubfinder/VideoSubFinder/VideoSubFinderWXW"}

exec "$project_dir/.venv/bin/python" "$project_dir/gui.py" "$@"
