#!/bin/bash
#
# Build FHE-BERT-Tiny and its FIDESlib dependency in one shot.
#
# FIDESlib is a pinned git submodule (third_party/FIDESlib), added directly to this
# project's build via add_subdirectory -- there is no separate "build FIDESlib, then
# `make install` it, then `find_package` it from FHE-BERT-Tiny" step anymore. One
# `cmake`/`cmake --build` drives both projects as a single incremental dependency
# graph: FIDESLIB_ARCH is auto-detected from the GPU present (see cmake/DetectCudaArch.cmake)
# instead of FIDESlib's default 7-architecture matrix, its test/benchmark suites and LTO are
# left off (see CMakeLists.txt), and ccache + Ninja are wired in when available.
#
# The one piece that still needs a standalone step is OpenFHE: it's a heavy external
# C++ library FIDESlib links against via find_package(OpenFHE CONFIG), not something
# folded into this build graph. This script builds it from patched source once and
# skips rebuilding it on later runs via a stamp file (see third_party/FIDESlib/deps/build.sh).
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIDESLIB_DIR="$ROOT_DIR/third_party/FIDESlib"
JOBS="$(nproc 2>/dev/null || echo 4)"
OPENFHE_INSTALL_PREFIX="${OPENFHE_INSTALL_PREFIX:-/usr/local}"

if [[ ! -f "$FIDESLIB_DIR/CMakeLists.txt" ]]; then
    echo "==> third_party/FIDESlib is empty; initializing submodule"
    git -C "$ROOT_DIR" submodule update --init --recursive
fi

GENERATOR_FLAGS=()
if command -v ninja >/dev/null 2>&1; then
    GENERATOR_FLAGS=(-G Ninja)
else
    echo "==> ninja not found; using CMake's default generator (Unix Makefiles)"
fi

LAUNCHER_FLAGS=()
if command -v ccache >/dev/null 2>&1; then
    LAUNCHER_FLAGS=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
else
    echo "==> ccache not found; building without a compiler cache (install ccache to speed up repeat builds)"
fi

# Skip rebuilding OpenFHE from source when the prefix already has a build that matches
# this FIDESlib checkout's patch (third_party/FIDESlib/deps/build.sh stamps this on install).
OPENFHE_PATCH_HASH="$(sha256sum "$FIDESLIB_DIR/deps/fideslib-ref-1.5.1.1.patch" | cut -d' ' -f1)"
OPENFHE_STAMP="$OPENFHE_INSTALL_PREFIX/share/openfhe/.fideslib_openfhe_stamp"
if [[ -f "$OPENFHE_STAMP" && "$(cat "$OPENFHE_STAMP")" == "$OPENFHE_PATCH_HASH" ]]; then
    echo "==> OpenFHE at $OPENFHE_INSTALL_PREFIX already matches this FIDESlib's patch ($OPENFHE_PATCH_HASH); skipping OpenFHE rebuild"
else
    echo "==> OpenFHE at $OPENFHE_INSTALL_PREFIX missing or stale for patch $OPENFHE_PATCH_HASH; building it from patched source"
    ( cd "$FIDESLIB_DIR/deps" && ./build.sh "$OPENFHE_INSTALL_PREFIX" )
fi

echo "==> Configuring + building FHE-BERT-Tiny (FIDESlib built in-tree via add_subdirectory)"
cd "$ROOT_DIR"
ldconfig
rm -rf build/
mkdir build && cd build
cmake "${GENERATOR_FLAGS[@]}" "${LAUNCHER_FLAGS[@]}" -DOPENFHE_INSTALL_PREFIX="$OPENFHE_INSTALL_PREFIX" ..
cmake --build . -j "$JOBS"
echo "Done building project"
