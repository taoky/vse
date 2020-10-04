#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_dir="$project_dir/third_party/VideoSubFinder"
build_dir="$source_dir/build-cpu"
install_dir="$project_dir/.local/videosubfinder"

cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$install_dir" \
    -DUSE_CUDA=OFF
cmake --build "$build_dir" --parallel "${VSE_BUILD_JOBS:-2}"
cmake --install "$build_dir"

binary="$install_dir/VideoSubFinder/VideoSubFinderWXW"
if [ ! -x "$binary" ]; then
    echo "VideoSubFinder build did not produce: $binary" >&2
    exit 1
fi

echo "Built source-audited CPU VideoSubFinder: $binary"
