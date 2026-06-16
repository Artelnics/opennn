#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
CUDNN=/home/artelnics/benchenv/lib/python3.12/site-packages/nvidia/cudnn
CUDA=/usr/local/cuda-12.9/targets/x86_64-linux
c++ -O3 -DNDEBUG -std=c++20 -flto=auto -fno-fat-lto-objects -march=native -fopenmp \
  -DEIGEN_NO_DEBUG -DHAVE_CUDNN_FRONTEND -DOPENNN_HAS_CUDA \
  -I/home/artelnics/opennn-precision/opennn \
  -I$CUDNN/include \
  -I/home/artelnics/opennn-wsl/build-gpu/_deps/eigen-src \
  -isystem /home/artelnics/opennn-precision/build-gpu129/_deps/opennn_libjpeg_turbo-install/include \
  -isystem /home/artelnics/opennn-wsl/build-gpu/_deps/cudnn_frontend-src/include \
  -isystem $CUDA/include \
  opennn_rosenbrock_trial.cpp -o opennn_rosenbrock_trial \
  -L$CUDA/lib/stubs -L$CUDA/lib \
  -Wl,-rpath,$CUDNN/lib:$CUDA/lib \
  /home/artelnics/opennn-precision/build-gpu129/opennn/libopennn.a \
  $CUDNN/lib/libcudnn.so.9 \
  /home/artelnics/opennn-precision/build-gpu129/_deps/opennn_libjpeg_turbo-install/lib/libjpeg.a \
  /usr/lib/x86_64-linux-gnu/libtbb.so.12.11 \
  $CUDA/lib/libcudart.so $CUDA/lib/libcublas.so $CUDA/lib/libcublasLt.so \
  $CUDA/lib/libnvrtc.so $CUDA/lib/stubs/libcuda.so \
  -lstdc++fs -lz -lgomp -lpthread -ldl -lrt
echo built
