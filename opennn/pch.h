#pragma once

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif

#ifndef NDEBUG
#define NDEBUG
#endif

#define EIGEN_MAX_ALIGN_BYTES 32
#define EIGEN_NO_DEBUG
#define NOMINMAX
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#define _CRT_SECURE_NO_WARNINGS
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <algorithm>
#include <string>
#include <cassert>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <time.h>
#include <iterator>
#include <map>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <exception>
#include <memory>
#include <random>
#include <regex>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <omp.h>

#include "../eigen/Eigen/Core"
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../eigen/Eigen/src/Core/util/DisableStupidWarnings.h"

//#define CUDA // Comment this line to disable cuda files

#ifdef CUDA

#include "../opennn/kernel.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <curand.h>
#include <cudnn.h>
#include <nvtx3/nvToolsExt.h>

template <typename T>
void check_cuda_status(T status, const char* file, int line, const char* msg)
{
    if (status != 0)
        throw runtime_error(string(msg) + " Error: " + to_string(static_cast<int>(status)) +
                            " in " + file + ":" + to_string(line));
}

#define CHECK_CUDA(x) check_cuda_status(x, __FILE__, __LINE__, "CUDA")
#define CHECK_CUBLAS(x) check_cuda_status(x, __FILE__, __LINE__, "CuBLAS")
#define CHECK_CUDNN(x) check_cuda_status(x, __FILE__, __LINE__, "cuDNN")

#define CUDA_MALLOC_AND_REPORT(ptr, size)                                         \
    do {                                                                          \
        size_t free_before, free_after, total;                                    \
        CHECK_CUDA(cudaMemGetInfo(&free_before, &total));                         \
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&(ptr)), (size)));         \
        CHECK_CUDA(cudaMemGetInfo(&free_after,  &total));                         \
                                                                                  \
        size_t bytes = free_before - free_after;                                  \
        if (bytes == 0) {                                                         \
            printf("cudaMalloc (%s):   reutilizado (%zu bytes solicitados)\n",    \
                   #ptr, static_cast<size_t>(size));                                         \
        } else {                                                                  \
            printf("cudaMalloc (%s):   %.6f MB  (%zu bytes)\n", #ptr,             \
                   bytes / (1024.0 * 1024.0), bytes);                             \
        }                                                                         \
    } while (0)

#endif

using namespace std;
using namespace Eigen;

using type = float;

namespace opennn {
constexpr type EPSILON = numeric_limits<type>::epsilon();
constexpr type MAX = numeric_limits<type>::max();
}

constexpr int Layout = Eigen::RowMajor;

using MatrixR = Matrix<type, Dynamic, Dynamic, Layout>;
using MatrixI = Matrix<Index, Dynamic, Dynamic, Layout>;
using MatrixB = Matrix<bool, Dynamic, Dynamic, Layout>;

using VectorR = Matrix<type, Dynamic, 1>;
using VectorI = Matrix<Index, Dynamic, 1>;
using VectorB = Matrix<bool, Dynamic, 1>;

using VectorMap = Map<VectorR, AlignedMax>;
using MatrixMap = Map<MatrixR, Layout | AlignedMax>;

using Tensor0 = Tensor<type, 0, Layout | AlignedMax>;
using Tensor1 = Tensor<type, 1, Layout | AlignedMax>;
using Tensor2 = Tensor<type, 2, Layout | AlignedMax>;
using Tensor3 = Tensor<type, 3, Layout | AlignedMax>;
using Tensor4 = Tensor<type, 4, Layout | AlignedMax>;

template <int Rank>
using TensorR = Tensor<type, Rank, Layout | AlignedMax>;

using TensorMap1 = TensorMap<Tensor<type, 1, Layout | AlignedMax>, AlignedMax>;
using TensorMap2 = TensorMap<Tensor<type, 2, Layout | AlignedMax>, AlignedMax>;
using TensorMap3 = TensorMap<Tensor<type, 3, Layout | AlignedMax>, AlignedMax>;
using TensorMap4 = TensorMap<Tensor<type, 4, Layout | AlignedMax>, AlignedMax>;

template <int Rank>
using TensorMapR = TensorMap<Tensor<type, Rank, Layout | AlignedMax>, AlignedMax>;

#include "tinyxml2.h"

using namespace tinyxml2;

template<typename Base, typename T>
inline bool is_instance_of(const T* ptr)
{
    return dynamic_cast<const Base*>(ptr);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
