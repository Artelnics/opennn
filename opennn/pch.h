#pragma once

#if defined(__INTELLISENSE__) && !defined(OPENNN_HAS_CUDA)
#define OPENNN_HAS_CUDA
#endif

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif

#define EIGEN_MAX_ALIGN_BYTES 64
#ifndef EIGEN_NO_DEBUG
#define EIGEN_NO_DEBUG
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#define _CRT_SECURE_NO_WARNINGS
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <algorithm>
#include <ranges>
#include <span>
#include <numbers>
#include <source_location>
#include <execution>
#include <charconv>
#include <format>
#include <string>
#include <string_view>
#include <cassert>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <exception>
#include <memory>
#include <optional>
#include <random>
#include <regex>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <set>
#include <sstream>
#include <omp.h>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/src/Core/util/DisableStupidWarnings.h>

#ifdef OPENNN_HAS_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <cublasLt.h>
#include <curand.h>
#include <cudnn.h>
#include <cuda_bf16.h>
#include <nvtx3/nvToolsExt.h>

#include "../opennn/kernel.cuh"

template <typename T>
void check_cuda_status(T status, const char* msg,
                       std::source_location loc = std::source_location::current())
{
    if (status != 0)
        throw std::runtime_error(std::string(msg) + " Error: " + std::to_string(static_cast<int>(status)) +
                                 " in " + loc.file_name() + ":" + std::to_string(loc.line()));
}

#define CHECK_CUDA(x)   check_cuda_status(x, "CUDA")
#define CHECK_CUBLAS(x) check_cuda_status(x, "CuBLAS")
#define CHECK_CUDNN(x)  check_cuda_status(x, "cuDNN")

#else


using cudaStream_t     = void*;
using cudaEvent_t      = void*;
using cudaGraph_t      = void*;
using cudaGraphExec_t  = void*;
using cublasHandle_t   = void*;
using cublasLtHandle_t = void*;
using cudnnHandle_t    = void*;
using cublasLtMatmulDesc_t   = void*;
using cublasLtMatrixLayout_t = void*;
struct cublasLtMatmulAlgo_t {};

struct __nv_bfloat16 {};
struct __half {};

enum cudaDataType_t                  { CUDA_R_32F = 0, CUDA_R_16F = 2, CUDA_R_8I = 3, CUDA_R_32I = 10, CUDA_R_16BF = 14 };
enum cublasComputeType_t             { CUBLAS_COMPUTE_32F = 0, CUBLAS_COMPUTE_32F_FAST_16BF = 65, CUBLAS_COMPUTE_32F_FAST_TF32 = 68 };
enum cublasOperation_t               { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };
enum cublasLtEpilogue_t              { CUBLASLT_EPILOGUE_DEFAULT = 1, CUBLASLT_EPILOGUE_BIAS = 4, CUBLASLT_EPILOGUE_RELU_BIAS = 132 };

enum cudnnDataType_t                 { CUDNN_DATA_FLOAT = 0, CUDNN_DATA_HALF = 2, CUDNN_DATA_INT8 = 3, CUDNN_DATA_INT32 = 4, CUDNN_DATA_BFLOAT16 = 14 };
enum cudnnActivationMode_t           { CUDNN_ACTIVATION_IDENTITY = 0, CUDNN_ACTIVATION_SIGMOID = 1, CUDNN_ACTIVATION_RELU = 2, CUDNN_ACTIVATION_TANH = 3, CUDNN_ACTIVATION_ELU = 4 };
enum cudnnPoolingMode_t              { CUDNN_POOLING_MAX = 0 };
enum cudnnBatchNormMode_t            { CUDNN_BATCHNORM_PER_ACTIVATION = 0 };
enum cudnnConvolutionFwdAlgo_t       { CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0 };
enum cudnnConvolutionBwdDataAlgo_t   { CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0 };
enum cudnnConvolutionBwdFilterAlgo_t { CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0 };

struct cudnnTensorStruct {};
using cudnnTensorDescriptor_t      = cudnnTensorStruct*;
using cudnnFilterDescriptor_t      = void*;
using cudnnConvolutionDescriptor_t = void*;
using cudnnPoolingDescriptor_t     = void*;
using cudnnActivationDescriptor_t  = void*;
using cudnnDropoutDescriptor_t     = void*;
using cudnnOpTensorDescriptor_t    = void*;
using cudnnRNNDescriptor_t         = void*;
using cudnnRNNDataDescriptor_t     = void*;

#endif

using namespace std;
using namespace Eigen;

namespace opennn {

using bfloat16 = __nv_bfloat16;

inline void throw_if(bool condition, const string& message,
                     const source_location& loc = source_location::current())
{
    if (condition)
        throw runtime_error(std::format("{} [at {}:{}]",
                                        message, loc.file_name(), loc.line()));
}

constexpr float EPSILON = numeric_limits<float>::epsilon();
constexpr float MAX = numeric_limits<float>::max();
constexpr float NEG_INFINITY = -numeric_limits<float>::infinity();
constexpr float QUIET_NAN = numeric_limits<float>::quiet_NaN();
constexpr float SOFTMAX_MASK_VALUE = float(-1e9f);

template <typename T>
ostream& operator<<(ostream& os, const vector<T>& vec)
{
    os << "[ ";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        os << vec[i];
        if (i + 1 < vec.size()) os << "; ";
    }
    os << " ]";
    return os;
}
}

constexpr int Layout = Eigen::RowMajor;

using MatrixR = Matrix<float, Dynamic, Dynamic, Layout>;
using MatrixI = Matrix<Index, Dynamic, Dynamic, Layout>;
using MatrixB = Matrix<bool, Dynamic, Dynamic, Layout>;

using VectorR = Matrix<float, Dynamic, 1>;
using VectorI = Matrix<Index, Dynamic, 1>;
using VectorB = Matrix<bool, Dynamic, 1>;

using VectorMap = Map<VectorR, AlignedMax>;
using MatrixMap = Map<MatrixR, Layout | AlignedMax>;

using Tensor0 = Tensor<float, 0, Layout | AlignedMax>;
using Tensor2 = Tensor<float, 2, Layout | AlignedMax>;
using Tensor3 = Tensor<float, 3, Layout | AlignedMax>;
using Tensor4 = Tensor<float, 4, Layout | AlignedMax>;

template <int Rank>
using TensorR = Tensor<float, Rank, Layout | AlignedMax>;

using TensorMap2 = TensorMap<Tensor<float, 2, Layout | AlignedMax>, AlignedMax>;
using TensorMap3 = TensorMap<Tensor<float, 3, Layout | AlignedMax>, AlignedMax>;
using TensorMap4 = TensorMap<Tensor<float, 4, Layout | AlignedMax>, AlignedMax>;

template <int Rank>
using TensorMapR = TensorMap<Tensor<float, Rank, Layout | AlignedMax>, AlignedMax>;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "json.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
