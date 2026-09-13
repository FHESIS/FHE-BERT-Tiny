# Build Consolidation: What Was Done

Implements the plan in `CONSOLIDATION_REPORT.md`. FIDESlib is now built as an
in-tree dependency of FHE-BERT-Tiny instead of a separate system install.

## Changes

- **FIDESlib vendored as a pinned git submodule** at `third_party/FIDESlib`
  (`.gitmodules` added), instead of an implicit "whatever's in `/usr/local`"
  dependency.
- **`CMakeLists.txt`** now does `add_subdirectory(third_party/FIDESlib)`
  instead of `find_package(fideslib REQUIRED CONFIG)` — one configure, one
  incremental build graph for both projects. It also:
  - auto-detects `FIDESLIB_ARCH` from the GPU present (`cmake/DetectCudaArch.cmake`),
    pinning to one architecture instead of FIDESlib's 7-arch default;
  - forces `FIDESLIB_COMPILE_TESTS`/`FIDESLIB_COMPILE_BENCHMARKS`/`FIDESLIB_ENABLE_LTO`
    off (not needed to consume the library);
  - re-`find_package`s `OpenMP` and `CUDAToolkit` at the top level, since their
    imported targets don't propagate out of the `add_subdirectory` scope.
- **`build_with_deps.sh` simplified**: no more separate "build+install
  FIDESlib, then configure FHE-BERT-Tiny against it" — it initializes the
  submodule if needed, builds OpenFHE only if a patch-hash stamp says it's
  stale, wires in `ccache`/`ninja` when present, then runs one
  `cmake`/`cmake --build`.
- **Two CMake bugs fixed** in FIDESlib's `CMakeLists.txt` (committed locally
  inside the submodule, not pushed upstream): an unconditional
  `set(CMAKE_CXX_COMPILER "g++")` broke when CXX was already enabled by a
  parent `project()`; `FIDESLIB_ENABLE_LTO` (default `OFF`) makes Release-build
  LTO opt-in instead of automatic.
- `ccache` and `ninja-build` installed system-wide.

Nothing in `FHE-BERT-Tiny` or the standalone `/root/FIDESlib` checkout has
been committed — `git status` shows the pending changes; ask if you want them
committed.

## Build and run, single command

```bash
./build_with_deps.sh && LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH ./build/FHE-BERT-Tiny "I really enjoyed this movie!" --verbose
```

- `build_with_deps.sh` builds FIDESlib + FHE-BERT-Tiny in one pass (~7s on
  this machine once OpenFHE and ccache are warm; longer on the very first run,
  which also builds OpenFHE from patched source).
- The `LD_LIBRARY_PATH` override works around `ldconfig` resolving
  `libcuda.so.1` to a broken shim in this environment.
- Keys already exist under `keys/`, so this loads the existing context
  instead of regenerating it. To regenerate from scratch instead:
  `./build/FHE-BERT-Tiny --generate_keys`.

## Verified

- Clean `rm -rf build/` → `./build_with_deps.sh` succeeds end-to-end (single
  CUDA arch, no test/benchmark targets, OpenFHE rebuild skipped via stamp).
- The resulting binary runs and produces correct classifier output.
- A second from-scratch rebuild (still `rm -rf build/`) completed in ~7s wall
  time with ccache hits, on a 32-core box.
