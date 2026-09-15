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

### FIDESlib provides GPU-accelerated CKKS primitives; FHEController now runs the whole circuit
### through its api/ wrapper instead of talking to OpenFHE directly.
find_package(fideslib REQUIRED CONFIG)
if (fideslib_FOUND)
    message(STATUS "FOUND PACKAGE FIDESlib")
    message(STATUS "FIDESlib Version: ${fideslib_VERSION}")
    message(STATUS "FIDESlib header files location: ${FIDESLIB_INCLUDE_PATH}")
    message(STATUS "FIDESlib lib files location: ${FIDESLIB_LIBRARIES_PATH}")
else ()
    message(FATAL_ERROR "PACKAGE FIDESlib NOT FOUND")
endif ()

set(CMAKE_CXX_COMPILER ${FIDESLIB_CXX_COMPILER})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")

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

target_link_libraries(FHE-BERT-Tiny PRIVATE fideslib::fideslib CUDA::cudart)
