#!/usr/bin/env bash
#
# Build script for FHE-BERT-Tiny.
#
# FHE-BERT-Tiny's CKKS backend runs through FIDESlib (GPU-accelerated CKKS,
# see src/FHEController.cpp), which must already be built and installed
# system-wide (found via `find_package(fideslib REQUIRED CONFIG)`, default
# search path /usr/local) along with the CUDA toolkit. This script just
# configures + builds the FHE-BERT-Tiny application against that install.
#
# Usage:
#   ./build.sh              # build FHE-BERT-Tiny
#   ./build.sh --clean      # wipe the build/ dir, then build
#   ./build.sh --jobs N     # override parallel build job count (default: nproc)
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"

JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
CLEAN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=1
            shift
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "$CLEAN" -eq 1 ]]; then
    echo "==> Cleaning previous build"
    rm -rf "$BUILD_DIR"
fi

echo "==> Using $JOBS parallel build jobs"

echo "==> Configuring FHE-BERT-Tiny (against system-installed FIDESlib)"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR"

echo "==> Building FHE-BERT-Tiny"
cmake --build "$BUILD_DIR" -j "$JOBS"

echo
echo "==> Done. Executable: $BUILD_DIR/FHE-BERT-Tiny"
echo "    LD_LIBRARY_PATH must include /usr/lib/x86_64-linux-gnu at run time"
echo "    (works around ldconfig resolving libcuda.so.1 to a broken shim here), e.g.:"
echo "        cd $ROOT_DIR && LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH ./build/FHE-BERT-Tiny --generate_keys"
echo "        cd $ROOT_DIR && LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH ./build/FHE-BERT-Tiny \"I really enjoyed this movie!\" --verbose"
