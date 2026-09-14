#!/bin/zsh
echo "Removing build folder"
rm -rf build/
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$HOME/openfhe-install ..
make
echo "Done"
