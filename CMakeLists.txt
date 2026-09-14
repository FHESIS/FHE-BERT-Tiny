project(FHE-BERT-Tiny)

cmake_minimum_required(VERSION 3.25.2)

# No build type was ever set here, so CMake defaulted to no optimization flags at all (plain -O0)
# for main.cpp/FHEController.cpp -- the plaintext masking/encoding/orchestration code that runs on
# the CPU around every GPU call. Default to Release (-O3 -DNDEBUG) unless the caller overrides it.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(APPLE)
    set(CMAKE_C_COMPILER "/Library/Developer/CommandLineTools/usr/bin/gcc")
    set(CMAKE_CXX_COMPILER "/Library/Developer/CommandLineTools/usr/bin/g++")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

### FIDESlib provides GPU-accelerated CKKS primitives; FHEController runs the whole circuit
### through its api/ wrapper instead of talking to OpenFHE directly. FIDESlib is vendored as a
### pinned git submodule (third_party/FIDESlib) and built in-tree via add_subdirectory: one
### `cmake --build` now drives both projects as a single incremental dependency graph, instead
### of a separate `cmake --install` to a system prefix plus a `find_package(... CONFIG)` lookup.
if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/FIDESlib/CMakeLists.txt")
    message(FATAL_ERROR
            "third_party/FIDESlib is empty -- run: git submodule update --init --recursive")
endif ()

# Building FIDESlib as a dependency: skip its own test/benchmark suites (FHE-BERT-Tiny only
# ever links the fideslib static library) and leave LTO off (its link-time cost isn't worth it
# outside FIDESlib's own runtime benchmarking).
set(FIDESLIB_COMPILE_TESTS OFF CACHE BOOL "" FORCE)
set(FIDESLIB_COMPILE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(FIDESLIB_ENABLE_LTO OFF CACHE BOOL "" FORCE)
# Assume a matching patched OpenFHE is already installed at OPENFHE_INSTALL_PREFIX (default
# /usr/local) -- see third_party/FIDESlib/deps/build.sh and build_with_deps.sh, which builds it
# once and skips rebuilding on later runs via a stamp file.
set(FIDESLIB_INSTALL_OPENFHE OFF CACHE BOOL "" FORCE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(DetectCudaArch)

add_subdirectory(third_party/FIDESlib)

### FHEController still makes a couple of raw lbcrypto:: calls (e.g. FHECKKSRNS::GetBootstrapDepth)
### that aren't exposed through fideslib's wrapper API, so the OpenFHE headers FIDESlib itself was
### built and linked against (system-wide, patched OpenFHE 1.5.1) must also be on the include path.
set(SYSTEM_OPENFHE_INCLUDE "/usr/local/include/openfhe" CACHE STRING "OpenFHE headers matching the FIDESlib install.")
include_directories(${SYSTEM_OPENFHE_INCLUDE})
include_directories(${SYSTEM_OPENFHE_INCLUDE}/third-party/include)
include_directories(${SYSTEM_OPENFHE_INCLUDE}/core)
include_directories(${SYSTEM_OPENFHE_INCLUDE}/pke)
include_directories(${SYSTEM_OPENFHE_INCLUDE}/binfhe)

include_directories(include)

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIR})

# fideslib.a is built with OpenMP (see third_party/FIDESlib/src/parallel_for.cpp); that flag is
# set on CMAKE_CXX_FLAGS inside FIDESlib's own add_subdirectory scope and doesn't propagate back
# up to this directory, so link OpenMP explicitly here too.
find_package(OpenMP REQUIRED)

# CUDA::cudart is created by find_package(CUDAToolkit) inside FIDESlib's add_subdirectory scope;
# CMake's CUDAToolkit imported targets aren't visible outside the directory that found them, so
# call it again here to get CUDA::cudart in this scope too (cheap: already cached by then).
find_package(CUDAToolkit REQUIRED)

add_executable(FHE-BERT-Tiny
        src/client_main.cpp
        src/ServerCircuit.cpp src/ServerCircuit.h
        src/ServerLog.h
        src/FHEController.cpp src/FHEController.h
        src/Utils.h)

target_link_libraries(FHE-BERT-Tiny PRIVATE fideslib CUDA::cudart OpenMP::OpenMP_CXX)

### security_tests/cca_probe_main.cpp: standalone research tool, see its file header. Reuses
### FHEController but never touches the BERT circuit/weights.
add_executable(cca-probe
        security_tests/cca_probe_main.cpp
        src/FHEController.cpp src/FHEController.h
        src/Utils.h)
target_include_directories(cca-probe PRIVATE src)

target_link_libraries(cca-probe PRIVATE fideslib CUDA::cudart OpenMP::OpenMP_CXX)
