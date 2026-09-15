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
### through its api/ wrapper instead of talking to OpenFHE directly. FIDESlib is built and
### installed separately (see build_fideslib.sh, which builds ../FIDESlib and installs it to
### OPENFHE_INSTALL_PREFIX's sibling prefix, default /usr/local) and picked up here via its
### installed CMake package config rather than as an in-tree add_subdirectory.
find_package(fideslib CONFIG REQUIRED)

# fideslib.a is built with OpenMP (see FIDESlib's src/parallel_for.cpp); that flag isn't part
# of fideslib::fideslib's INTERFACE_LINK_LIBRARIES, so link OpenMP explicitly here too.
find_package(OpenMP REQUIRED)

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

add_executable(FHE-BERT-Tiny
        src/client_main.cpp
        src/ServerCircuit.cpp src/ServerCircuit.h
        src/ServerLog.h
        src/FHEController.cpp src/FHEController.h
        src/Utils.h)

target_link_libraries(FHE-BERT-Tiny PRIVATE fideslib::fideslib OpenMP::OpenMP_CXX)

### security_tests/cca_probe_main.cpp: standalone research tool, see its file header. Reuses
### FHEController but never touches the BERT circuit/weights.
add_executable(cca-probe
        security_tests/cca_probe_main.cpp
        src/FHEController.cpp src/FHEController.h
        src/Utils.h)
target_include_directories(cca-probe PRIVATE src)

target_link_libraries(cca-probe PRIVATE fideslib::fideslib OpenMP::OpenMP_CXX)

