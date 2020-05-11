#include "omp.h"

//Eigen includes

#include "../eigen/Eigen/src/Core/util/DisableStupidWarnings.h"

#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

// For numeric limits

#define NOMINMAX

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif

//#define OPENNN_CUDA

#ifdef OPENNN_CUDA

#include "../../opennn-cuda/opennn_cuda/kernels.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cublasXt.h>
#include <curand.h>

#endif


#pragma warning(push, 0)
#include "tinyxml2.h"
#pragma warning(pop)


#pragma warning(push, 0)
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../eigen/unsupported/Eigen/CXX11/ThreadPool"
#pragma warning(pop)


// #define OPENNN_MKL

#ifdef OPENNN_MKL
    #include "mkl.h"
#endif


//#define EIGEN_USE_BLAS

//#define EIGEN_TEST_NO_LONGDOUBLE

//#define EIGEN_TEST_NO_COMPLEX

//#define EIGEN_TEST_FUNC cxx11_tensor_cuda

//#define EIGEN_DEFAULT_DENSE_INDEX_TYPE Index

//#define EIGEN_USE_GPU

//#define EIGEN_MALLOC_ALREADY_ALIGNED 1

//#define EIGEN_UNROLLING_LIMIT 1000

namespace OpenNN
{
    typedef float type;
}



