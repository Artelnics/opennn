#pragma once

// IntelliSense-only define so VS Code does not gray out CUDA code.
// Real builds get OPENNN_WITH_CUDA from CMake (-DOPENNN_WITH_CUDA).
#if defined(__INTELLISENSE__) && !defined(OPENNN_WITH_CUDA)
#define OPENNN_WITH_CUDA
#endif

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif

#define EIGEN_MAX_ALIGN_BYTES 64
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

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/src/Core/util/DisableStupidWarnings.h>

#ifdef OPENNN_WITH_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <cublasLt.h>
#include <curand.h>
#include <cudnn.h>
#include <cuda_bf16.h>
#include <nvtx3/nvToolsExt.h>

#else

// =========================================================================
// CPU-only stubs.
// Goal: every CUDA/cuDNN/cuBLAS *type* that appears in OpenNN's signatures or
// struct members must compile when CUDA is off. Values don't matter — runtime
// GPU paths are gated by Device::is_gpu() and never execute in this build.
// Enums (not type aliases) so scoped references like
// `cudnnPoolingMode_t::CUDNN_POOLING_MAX` keep working.
//
// When adding a new CUDA-only type referenced from non-CUDA-guarded code,
// add the stub here in the matching subsection.
// =========================================================================

// --- Runtime / library handles ---
using cudaStream_t     = void*;
using cudaEvent_t      = void*;
using cublasHandle_t   = void*;
using cublasLtHandle_t = void*;
using cudnnHandle_t    = void*;

// --- Scalar / opaque types ---
struct __nv_bfloat16 {};
struct __half {};

// --- CUDA / cuBLAS enums ---
enum cudaDataType_t                  { CUDA_R_32F = 0, CUDA_R_16F = 2, CUDA_R_8I = 3, CUDA_R_32I = 10, CUDA_R_16BF = 14 };
enum cublasComputeType_t             { CUBLAS_COMPUTE_32F = 0, CUBLAS_COMPUTE_32F_FAST_16BF = 65, CUBLAS_COMPUTE_32F_FAST_TF32 = 68 };
enum cublasOperation_t               { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };

// --- cuDNN enums ---
enum cudnnDataType_t                 { CUDNN_DATA_FLOAT = 0, CUDNN_DATA_HALF = 2, CUDNN_DATA_INT8 = 3, CUDNN_DATA_INT32 = 4, CUDNN_DATA_BFLOAT16 = 14 };
enum cudnnActivationMode_t           { CUDNN_ACTIVATION_IDENTITY = 0, CUDNN_ACTIVATION_SIGMOID = 1, CUDNN_ACTIVATION_RELU = 2, CUDNN_ACTIVATION_TANH = 3, CUDNN_ACTIVATION_ELU = 4 };
enum cudnnPoolingMode_t              { CUDNN_POOLING_MAX = 0 };
enum cudnnBatchNormMode_t            { CUDNN_BATCHNORM_PER_ACTIVATION = 0 };
enum cudnnConvolutionFwdAlgo_t       { CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0 };
enum cudnnConvolutionBwdDataAlgo_t   { CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0 };
enum cudnnConvolutionBwdFilterAlgo_t { CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0 };

// --- cuDNN descriptor handles (opaque pointer-typed) ---
struct cudnnTensorStruct {};
using cudnnTensorDescriptor_t      = cudnnTensorStruct*;
using cudnnFilterDescriptor_t      = void*;
using cudnnConvolutionDescriptor_t = void*;
using cudnnPoolingDescriptor_t     = void*;
using cudnnActivationDescriptor_t  = void*;
using cudnnDropoutDescriptor_t     = void*;
using cudnnOpTensorDescriptor_t    = void*;

#endif

// CUDA-only machinery — functions, kernel declarations, error-check macros.
#ifdef OPENNN_WITH_CUDA

#include "../opennn/kernel.cuh"

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

#endif

using namespace std;
using namespace Eigen;

using type = float;

namespace opennn {
constexpr type EPSILON = numeric_limits<type>::epsilon();
constexpr type MAX = numeric_limits<type>::max();
constexpr type NEG_INFINITY = -numeric_limits<type>::infinity();
constexpr type QUIET_NAN = numeric_limits<type>::quiet_NaN();
constexpr type SOFTMAX_MASK_VALUE = type(-1e9f);

// Generic stream output for std::vector. Lives here so it's available everywhere
// pch.h reaches (which is essentially the whole library).
template <typename T>
ostream& operator<<(ostream& os, const vector<T>& vec)
{
    os << "[ ";
    for(size_t i = 0; i < vec.size(); ++i)
    {
        os << vec[i];
        if (i + 1 < vec.size()) os << "; ";
    }
    os << " ]";
    return os;
}
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "tinyxml2.h"
#pragma GCC diagnostic pop

using namespace tinyxml2;

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
