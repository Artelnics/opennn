#include "omp.h"

//Eigen includes

#include "../eigen/Eigen/src/Core/util/DisableStupidWarnings.h"

#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

// For numeric limits

#define NOMINMAX

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif

#ifdef __OPENNN_CUDA__

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

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



