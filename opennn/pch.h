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
#include <bit>
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
#include <Eigen/Cholesky>
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

struct cudnnTensorStruct {};
using cudnnTensorDescriptor_t      = cudnnTensorStruct*;
using cudnnDropoutDescriptor_t     = void*;
using cudnnOpTensorDescriptor_t    = void*;
using cudnnRNNDescriptor_t         = void*;
using cudnnRNNDataDescriptor_t     = void*;

#endif

using namespace std;
using Eigen::Index;

namespace opennn {

using namespace Eigen;
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

using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Layout>;
using MatrixI = Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic, Layout>;
using MatrixB = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Layout>;

using VectorR = Eigen::Matrix<float, Eigen::Dynamic, 1>;
using VectorI = Eigen::Matrix<Index, Eigen::Dynamic, 1>;
using VectorB = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

using VectorMap = Eigen::Map<VectorR, Eigen::AlignedMax>;
using MatrixMap = Eigen::Map<MatrixR, Eigen::AlignedMax>;

using Tensor0 = Eigen::Tensor<float, 0, Layout | Eigen::AlignedMax>;
using Tensor2 = Eigen::Tensor<float, 2, Layout | Eigen::AlignedMax>;
using Tensor3 = Eigen::Tensor<float, 3, Layout | Eigen::AlignedMax>;
using Tensor4 = Eigen::Tensor<float, 4, Layout | Eigen::AlignedMax>;

template <int Rank>
using TensorR = Eigen::Tensor<float, Rank, Layout | Eigen::AlignedMax>;

using TensorMap2 = Eigen::TensorMap<Eigen::Tensor<float, 2, Layout | Eigen::AlignedMax>, Eigen::AlignedMax>;
using TensorMap3 = Eigen::TensorMap<Eigen::Tensor<float, 3, Layout | Eigen::AlignedMax>, Eigen::AlignedMax>;
using TensorMap4 = Eigen::TensorMap<Eigen::Tensor<float, 4, Layout | Eigen::AlignedMax>, Eigen::AlignedMax>;

template <int Rank>
using TensorMapR = Eigen::TensorMap<Eigen::Tensor<float, Rank, Layout | Eigen::AlignedMax>, Eigen::AlignedMax>;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "json.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
