#!/bin/bash
#cd ../FIDESlib
#CUDA_ARCH=`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`
#rm -rf build/
#mkdir build && cd build
#cmake -DFIDESLIB_ARCH=${CUDA_ARCH//./} -DCMAKE_BUILD_TYPE=Release -DFIDESLIB_INSTALL_OPENFHE=ON ..
#make -j`nproc` && make install
#echo "Done building FIDESlib"
#cd -
ldconfig
rm -rf build/
mkdir build && cd build
cmake ..
make -j4
echo "Done building project"

