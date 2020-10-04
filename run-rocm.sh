#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python_bin=${VSE_PYTHON:-"$project_dir/.venv-rocm/bin/python"}

if [ ! -x "$python_bin" ]; then
    echo "ROCm Python environment not found: $python_bin" >&2
    echo "Set VSE_PYTHON to a Python containing a ROCm-enabled PyTorch build." >&2
    exit 1
fi

export PADDLE_PDX_CACHE_HOME="$project_dir/.cache/paddlex"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export VSE_OCR_BACKEND=rapidocr-torch
export VSE_RAPIDOCR_MODEL_DIR=${VSE_RAPIDOCR_MODEL_DIR:-"$project_dir/.cache/rapidocr-safe-test"}
export VSE_VIDEOSUBFINDER_PATH=${VSE_VIDEOSUBFINDER_PATH:-"$project_dir/.local/videosubfinder/VideoSubFinder/VideoSubFinderWXW"}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0}

"$python_bin" -m backend.tools.check_rocm
exec "$python_bin" "$project_dir/gui.py" "$@"
