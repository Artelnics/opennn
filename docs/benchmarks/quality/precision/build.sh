#!/usr/bin/env bash
# Build the OpenNN convergence-gate driver against the WSL CUDA build of OpenNN.
set -e
cd "$(dirname "$0")"
CUDNN=/home/artelnics/benchenv/lib/python3.12/site-packages/nvidia/cudnn
CUDA=/usr/local/cuda-12.9/targets/x86_64-linux
OPENNN=/home/artelnics/opennn-precision
BUILD=$OPENNN/build-gpu129
c++ -O3 -DNDEBUG -std=c++20 -flto=auto -fno-fat-lto-objects -march=native -fopenmp \
  -DEIGEN_NO_DEBUG -DHAVE_CUDNN_FRONTEND -DOPENNN_HAS_CUDA \
  -I$OPENNN/opennn \
  -I$CUDNN/include \
  -I/home/artelnics/opennn-wsl/build-gpu/_deps/eigen-src \
  -isystem $BUILD/_deps/opennn_libjpeg_turbo-install/include \
  -isystem /home/artelnics/opennn-wsl/build-gpu/_deps/cudnn_frontend-src/include \
  -isystem $CUDA/include \
  opennn_precision.cpp -o opennn_precision \
  -L$CUDA/lib/stubs -L$CUDA/lib \
  -Wl,-rpath,$CUDNN/lib:$CUDA/lib \
  $BUILD/opennn/libopennn.a \
  $CUDNN/lib/libcudnn.so.9 \
  $BUILD/_deps/opennn_libjpeg_turbo-install/lib/libjpeg.a \
  /usr/lib/x86_64-linux-gnu/libtbb.so.12.11 \
  $CUDA/lib/libcudart.so $CUDA/lib/libcublas.so $CUDA/lib/libcublasLt.so \
  $CUDA/lib/libnvrtc.so $CUDA/lib/stubs/libcuda.so \
  -lstdc++fs -lz -lgomp -lpthread -ldl -lrt
echo built
