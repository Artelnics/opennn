//#define OPENNN_MKL
#ifdef OPENNN_MKL
    #define EIGEN_USE_MKL_ALL
    #include "mkl.h"
#endif

//Eigen includes
#include "../eigen/Eigen/src/Core/util/DisableStupidWarnings.h"

#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

// For numeric limits

#define NOMINMAX

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif

#pragma warning(push, 0)
#include "tinyxml2.h"
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../eigen/unsupported/Eigen/CXX11/ThreadPool"
#pragma warning(pop)

#ifdef OPENNN_CUDA

#include "../../opennn-cuda/opennn_cuda/kernels.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cublasXt.h>
#include <curand.h>

#endif

#include <omp.h>

namespace OpenNN
{
    typedef double type;
}

