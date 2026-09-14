#!/usr/bin/env bash

set -e
set -x

#Remove previous installation.
rm -rf openfhe-install
rm -rf openfhe-src

# Target installation directory.
mkdir -p $1
git clone https://github.com/openfheorg/openfhe-development openfhe-src

#Source clone.
cd openfhe-src
git checkout fideslib-ref-v1.5.1.1
#git checkout v1.5.1
#git config user.email "FIDESlib"
#git config user.name "FIDESlib"
git apply ../fideslib-ref-1.5.1.1.patch
#git apply ../openfhe-1.5.1.patch

# Compilation and installation.
mkdir build
cd build
echo "Installing into $1"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$1" ..
make -j12
make install -j12

# Stamp the install with a hash of the patch just applied, so callers building
# FIDESlib as a dependency (see FHE-BERT-Tiny/build_with_deps.sh) can detect
# "already installed, matching patch" and skip re-running this from-scratch
# OpenFHE rebuild.
mkdir -p "$1/share/openfhe"
sha256sum ../../fideslib-ref-1.5.1.1.patch | cut -d' ' -f1 > "$1/share/openfhe/.fideslib_openfhe_stamp"
echo "Stamped OpenFHE install at $1 for patch $(cat "$1/share/openfhe/.fideslib_openfhe_stamp")"