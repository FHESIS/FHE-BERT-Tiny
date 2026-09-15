#!/bin/bash
sudo apt update && sudo apt install wget make build-essential cmake git python3-venv libgsl-dev -y
git clone https://github.com/FHESIS/FIDESlib
cd FIDESlib
CUDA_ARCH=`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`
mkdir build && cd build
cmake -DFIDESLIB_ARCH=${CUDA_ARCH//./} -DCMAKE_BUILD_TYPE=Release -DFIDESLIB_INSTALL_OPENFHE=ON ..
make -j`nproc` && make install
git clone https://github.com/FHESIS/FHE-BERT-Tiny
git checkout gpu-build
mkdir build && cd build
cmake ..
make -j`nproc`
curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
cd ../src/python/
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install urllib3
cd -
ldconfig
./build/FHE-BERT-Tiny --generate_keys
./build/FHE-BERT-Tiny "Dune was a quite interesting movie" --verbose
python3 ./src/python/PlainCircuit.py "Dune was a quite interesting movie" --verbose
