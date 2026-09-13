# Consolidating FHE-BERT-Tiny + FIDESlib for Fast Build & Benchmark Loops

## Summary

FHE-BERT-Tiny's CKKS backend is a thin C++ application (`src/main.cpp`,
`src/FHEController.cpp`, ~54 call sites) that runs entirely through FIDESlib's
`api/` wrapper (`Ptxt`/`Ctxt` = `fideslib::Plaintext`/`fideslib::Ciphertext`),
with a handful of raw `lbcrypto::` calls falling through to the same patched
OpenFHE that FIDESlib itself was built against. Today the two projects are
related only by prose (READMEs, `post-install.sh`) and a `find_package`
lookup against whatever happens to be installed at `/usr/local` — there is no
recorded FIDESlib commit pin, no shared build orchestration, and no build
caching anywhere in the chain. The result: every fresh environment pays the
full cost of building OpenFHE from source, then FIDESlib for **7 CUDA
architectures** it doesn't need, then FHE-BERT-Tiny — with no compiler cache
and no incremental-build relationship between the two repos. This report
documents the current path end-to-end and lays out a consolidation plan to
make that loop fast and reproducible.

## Current code path (confirmed by reading the source)

1. `FHE-BERT-Tiny/src/FHEController.h` includes `<fideslib.hpp>` and OpenFHE
   serialization headers directly, and aliases `Ptxt`/`Ctxt` to FIDESlib's
   API types. `FHEController.cpp` drives the whole BERT circuit (masking,
   matmul, rotations, bootstrap, `eval_exp`) through this wrapper — it is the
   only real consumer of FIDESlib in this repo.
2. `FHE-BERT-Tiny/CMakeLists.txt` requires FIDESlib via
   `find_package(fideslib REQUIRED CONFIG)` — i.e. FIDESlib must already be
   **built and installed system-wide** (default prefix `/usr/local`) before
   FHE-BERT-Tiny can even configure. It also hard-codes
   `/usr/local/include/openfhe` as an extra include path, because a few
   `lbcrypto::` calls (e.g. `FHECKKSRNS::GetBootstrapDepth`) aren't exposed
   through FIDESlib's wrapper.
3. `FIDESlib/CMakeLists.txt` in turn requires a patched OpenFHE 1.5.1
   (`find_package(OpenFHE 1.5.1 CONFIG EXACT REQUIRED ...)`), optionally
   building+installing it from the `deps/openfhe-src` submodule
   (`FIDESLIB_INSTALL_OPENFHE=ON` → runs `deps/build.sh`, which always
   `rm -rf`s any previous OpenFHE build/install and rebuilds from scratch —
   there's no "already built at this commit" short-circuit).
4. `post-install.sh` (the documented "set up everything" path) does this
   whole chain serially on every fresh box: apt installs → clone FIDESlib →
   configure+build+`make install` FIDESlib (with `FIDESLIB_INSTALL_OPENFHE=ON`,
   so OpenFHE is compiled from source too) → clone FHE-BERT-Tiny → configure+build
   it. Nothing is cached or reused between runs.

## What actually makes this slow

1. **CUDA architecture over-targeting.** `FIDESlib/CMakeLists.txt` defaults
   `FIDESLIB_ARCH` to
   `80-real;86-real;89-real;90-real;90-virtual;100-real;120-real` — **7
   targets** — confirmed in the live build's `CMakeCache.txt`. The actual GPU
   in this environment is a single RTX A5000, `compute_cap 8.6`
   (`nvidia-smi` confirms). FIDESlib's `src/` has 28 `.cu` + 37 `.cuh`/`.cpp`
   files compiled with `-rdc=true` (relocatable device code, needed for
   cross-TU device-symbol resolution), which forces per-architecture device
   compilation and a heavier device link step. Building for 7 archs when 1 is
   needed is very likely the single largest lever available here — this cost
   is paid again for `fideslib-test` and `fideslib-bench`, which compile the
   same headers into two more binaries.
2. **Tests and benchmarks are compiled by default, from unpinned Git refs.**
   `FIDESLIB_COMPILE_TESTS` and `FIDESLIB_COMPILE_BENCHMARKS` both default
   `ON` (confirmed in `CMakeCache.txt`), and both use `FetchContent` against
   `GIT_TAG main` for googletest and google/benchmark respectively — a
   floating branch, fetched over the network on every clean `build/`
   directory, with no version pin. FHE-BERT-Tiny only ever links the
   `fideslib` static library, never `fideslib-test`/`fideslib-bench`, so this
   entire cost (two extra dependency fetches + two extra large binaries) is
   pure waste on the path that matters to it.
3. **LTO/IPO is unconditionally forced on for Release builds**
   (`CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE` whenever
   `CMAKE_BUILD_TYPE=Release`, no way to opt out short of overriding
   `CMAKE_BUILD_TYPE`). Cross-TU interprocedural optimization over a large
   templated NTT/CKKS kernel codebase, multiplied across the arch list above,
   is expensive at link time and mostly pays off for FIDESlib's own
   microbenchmarks — not for a downstream consumer that just wants a working
   `fideslib.a` quickly.
4. **No compiler cache, no faster generator.** Neither `ccache` nor `ninja`
   is installed in this environment; CMake falls back to `Unix Makefiles`
   with no `CMAKE_CXX_COMPILER_LAUNCHER` / `CMAKE_CUDA_COMPILER_LAUNCHER`
   configured anywhere. Since the current workflow (`post-install.sh`,
   `build_with_deps.sh`) routinely does `rm -rf build/` and starts over, there
   is no reuse of object files across "fresh environment" runs even when the
   underlying source hasn't changed — `ccache` on `nvcc` is well supported and
   essentially free to add.
5. **OpenFHE is rebuilt from source on every `FIDESLIB_INSTALL_OPENFHE=ON`
   run, unconditionally.** `deps/build.sh` starts with `rm -rf openfhe-install
   openfhe-src`-equivalent cleanup and always recompiles the patched fork
   (`deps/openfhe-ref-1.5.1.1.patch` applied over commit `dd8d374` of
   `openfhe-development`) — a large templated math/lattice-crypto library
   (installed static libs alone: ~4.9M `core`, ~20M `pke`, ~1.4M `binfhe`).
   There is no check for "prefix already has this exact patched commit
   installed," so this multi-minute build re-runs even when nothing about
   OpenFHE changed.
6. **Two independently-versioned repos with no pin between them.**
   FHE-BERT-Tiny has no submodule, lockfile, or recorded commit/tag for the
   FIDESlib it needs — compatibility is whatever's currently sitting in
   `/usr/local`. FIDESlib itself has no version compatibility promise for
   FHE-BERT-Tiny's `api/` usage. This isn't just a build-speed problem: it
   means "rebuild FIDESlib" and "rebuild FHE-BERT-Tiny" can't be reasoned
   about as one incremental build graph, and a benchmark result can't be
   traced back to an exact pair of commits.
7. **Ancillary:** `FIDESlib/docker/Dockerfile` (an `nvidia/cuda:...-devel`
   base meant to shortcut "compiling and using FIDESlib without the burden of
   dependency management") exists but is unused by FHE-BERT-Tiny's own setup
   scripts, which apt-install directly on the host instead. Large generated
   artifacts (`benchmark_results.json` at 75KB, `benchmark_run.log`) are
   currently tracked at the FHE-BERT-Tiny repo root rather than gitignored,
   which will keep bloating repo clones as more benchmark runs are recorded.

## Recommendations

Ordered roughly by impact-per-effort.

### 1. Pin `FIDESLIB_ARCH` to the real target GPU for dev/bench builds
Default builds (and CI) should pass `-DFIDESLIB_ARCH=86-real` (or
auto-detect via `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`)
instead of building the full 7-way portability matrix. Reserve the wide arch
list for an explicit "release/distribute" build profile that isn't on the
inner dev loop. This alone should cut FIDESlib's CUDA compile+device-link
time roughly in proportion to archs dropped (7→1).

### 2. Turn off tests/benchmarks when FIDESlib is built as a dependency
Pass `-DFIDESLIB_COMPILE_TESTS=OFF -DFIDESLIB_COMPILE_BENCHMARKS=OFF` whenever
FIDESlib is being built to be *consumed* (i.e. every FHE-BERT-Tiny build).
This removes two FetchContent network fetches and two extra large binary
targets from the path that FHE-BERT-Tiny actually depends on. Keep them ON
only when someone is actually iterating on FIDESlib itself. When they are
built, pin `GIT_TAG` for googletest/benchmark to a fixed release instead of
`main`, so the fetch is reproducible and cacheable.

### 3. Add `ccache` + switch to Ninja
Install `ccache` and `ninja-build`, and set
`CMAKE_CXX_COMPILER_LAUNCHER=ccache`, `CMAKE_CUDA_COMPILER_LAUNCHER=ccache`
at the top-level configure (both are supported with `nvcc`). Use `-G Ninja`
for better parallel scheduling than `Unix Makefiles`. This turns the current
"always `rm -rf build/` and start cold" workflow into something that can
actually benefit from unchanged translation units — important once builds
happen repeatedly across container restarts of the same disk, or in CI with
a persisted cache dir.

### 4. Make LTO/IPO opt-in, not automatic, for Release
Gate `CMAKE_INTERPROCEDURAL_OPTIMIZATION` behind its own option (e.g.
`FIDESLIB_ENABLE_LTO`, default `OFF`) rather than tying it to
`CMAKE_BUILD_TYPE STREQUAL "Release"`. Enable it only for builds whose
purpose is to measure/publish runtime performance numbers, not for the
everyday "build FIDESlib so FHE-BERT-Tiny links against it" loop.

### 5. Stop treating OpenFHE as always-rebuild-from-source
Before `deps/build.sh` runs, check whether the target prefix already has
OpenFHE installed at the exact patched commit (e.g. stamp a
`<prefix>/share/openfhe/.fideslib-commit` file with the submodule SHA during
install, and compare it on entry). Skip the rebuild when it matches. This
turns `FIDESLIB_INSTALL_OPENFHE=ON` from "always pay the multi-minute OpenFHE
build" into "pay it once per actual OpenFHE/patch change."

### 6. Pin the FIDESlib dependency instead of "whatever's installed"
Record the exact FIDESlib commit FHE-BERT-Tiny is built/tested against —
either as a git submodule under e.g. `third_party/FIDESlib` pinned to a SHA,
or at minimum a `FIDESLIB_COMMIT` file checked by CI. This is what makes
"consolidation" meaningful rather than cosmetic: it lets a single top-level
build graph know when FIDESlib actually needs to be rebuilt versus reused,
and lets a benchmark result be attributed to an exact pair of commits.

### 7. Replace the two-stage `find_package`/system-install flow with `add_subdirectory`
Rather than "configure+build+`make install` FIDESlib to `/usr/local`, then
separately configure+build FHE-BERT-Tiny against the installed package,"
have a top-level `CMakeLists.txt` (in a monorepo layout, or in FHE-BERT-Tiny
pointing at the pinned FIDESlib submodule) do:

```cmake
set(FIDESLIB_COMPILE_TESTS OFF CACHE BOOL "" FORCE)
set(FIDESLIB_COMPILE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(FIDESLIB_INSTALL_OPENFHE OFF CACHE BOOL "" FORCE)  # assume prebuilt/cached OpenFHE
add_subdirectory(third_party/FIDESlib)
add_executable(FHE-BERT-Tiny src/main.cpp src/FHEController.cpp)
target_link_libraries(FHE-BERT-Tiny PRIVATE fideslib::fideslib CUDA::cudart)
```

One `cmake --build` then drives both projects as a single dependency graph:
CMake/Ninja only recompiles the FIDESlib translation units that actually
changed, instead of the current "reinstall FIDESlib, then reconfigure
FHE-BERT-Tiny from a fresh `find_package` lookup" pattern. `find_package`
mode can remain supported (useful for people who only want the library), but
shouldn't be the only path.

### 8. One benchmark entry point, tagged with both commits
Today FIDESlib has its own `bench/` (Google Benchmark, kernel-level: NTT,
rotation, bootstrap, etc. — 14 files) and FHE-BERT-Tiny has its own
end-to-end `src/python/run_benchmark.py` + `build/FHE-BERT-Tiny` binary —
unrelated, with results (`benchmark_results.json`, `benchmark_run.log`)
committed at the repo root instead of gitignored. Add a thin top-level
script that can drive either suite and stamps output with
`(fideslib_commit, fhe-bert-tiny_commit, FIDESLIB_ARCH, GPU model)`, and move
generated result files to a gitignored `results/` directory so repeated
benchmark runs don't bloat the repo.

### 9. Use the existing Docker image as the cached dependency layer
`FIDESlib/docker/Dockerfile` already exists for "compiling and using FIDESlib
without the burden of dependency management" but isn't wired into
FHE-BERT-Tiny's setup scripts at all. Building a shared base image with
patched-OpenFHE + FIDESlib (single-arch, tests/benchmarks off) prebuilt as a
layer — rebuilt only when the pinned commit changes — lets FHE-BERT-Tiny's
own dev loop and CI skip the OpenFHE+FIDESlib compile entirely most of the
time, only compiling the thin application layer on top.

## Suggested target structure

```
FHE-BERT-Tiny/                      (or a new monorepo root)
├── CMakeLists.txt                  # top-level: add_subdirectory(third_party/FIDESlib), builds FHE-BERT-Tiny
├── third_party/
│   └── FIDESlib/                   # git submodule, pinned to a tested commit
├── cmake/
│   └── DetectCudaArch.cmake        # nvidia-smi-based FIDESLIB_ARCH default
├── src/                            # FHEController.{h,cpp}, main.cpp (unchanged)
├── bench/
│   └── run_all.sh                  # drives fideslib-bench + FHE-BERT-Tiny benchmark, tags commits
├── results/                        # gitignored benchmark output
└── docker/
    └── Dockerfile                  # prebuilds OpenFHE+FIDESlib(single-arch) as a cached layer
```

## Expected impact

- **Arch pruning (7→1):** the largest single win — proportionally less CUDA
  device compilation and device-link work across ~28 `.cu` files and 3 CUDA
  binaries (`fideslib`, `fideslib-test`, `fideslib-bench`) whenever those
  extra targets are even built.
- **Skipping tests/benchmarks when consumed:** removes 2 network fetches and
  2 large binary builds from every FHE-BERT-Tiny build.
- **`ccache` + Ninja:** turns repeat builds (common in a dev loop that
  currently `rm -rf build/`s routinely) from "always full cost" into
  "near-zero cost when nothing changed."
- **OpenFHE build-skip + submodule pin + `add_subdirectory`:** removes the
  multi-minute from-scratch OpenFHE rebuild and the redundant
  install-then-reconfigure round trip on every environment setup, and makes
  "did FIDESlib change" a real, checkable question instead of implicit state
  in `/usr/local`.

None of these require touching `FHEController.cpp`'s actual FIDESlib call
sites — this is entirely a build/dependency-orchestration change, orthogonal
to the runtime-performance work already captured in `PERFORMANCE_REPORT.md`.
