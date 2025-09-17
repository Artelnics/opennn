#ifndef PCH_H
#define PCH_H

#pragma once

#define NUMERIC_LIMITS_MIN type(0.000001)

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
#include <codecvt>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <iterator>
#include <map>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <exception>
#include <memory>
#include <random>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <stdlib.h>
#include <set>
#include <regex>
#include <sstream>

#include <omp.h>

#define EIGEN_USE_THREADS

#include "../eigen/Eigen/Core"
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../eigen/Eigen/src/Core/util/DisableStupidWarnings.h"

//#define OPENNN_CUDA // Comment this line to disable cuda files

#ifdef OPENNN_CUDA

#include "../opennn/kernel.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <curand.h>
#include <cudnn.h>
#include <nvtx3/nvToolsExt.h>

#define CHECK_CUDA(call) do \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
    { \
        fprintf(stderr, "CUDA Error %s:%d - %s (%d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err), err); \
    } \
} while(0)

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
                   #ptr, (size_t)(size));                                         \
        } else {                                                                  \
            printf("cudaMalloc (%s):   %.6f MB  (%zu bytes)\n", #ptr,             \
                   bytes / (1024.0 * 1024.0), bytes);                             \
        }                                                                         \
    } while (0)

#define CHECK_CUDA_ERROR(message)                                       \
    do {                                                                \
        cudaError_t error = cudaGetLastError();                         \
        if (error != cudaSuccess) {                                     \
            std::cerr << "CUDA Error after " << message << ": "         \
                      << cudaGetErrorString(error) << " (" << error     \
                      << ")" << std::endl;                              \
        }                                                               \
    } while (0)

#endif

using namespace std;
using namespace Eigen;
//using namespace tinyxml2;

using type = float;

using dimensions = vector<Index>;

template<typename Base, typename T>
inline bool is_instance_of(const T* ptr)
{
    return dynamic_cast<const Base*>(ptr);
}

#endif // PCH_H
