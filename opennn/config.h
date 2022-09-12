#include "half.hpp"

#define NUMERIC_LIMITS_MIN 0.000001

//#define OPENNN_MKL

#ifdef OPENNN_MKL
    #define EIGEN_USE_MKL_ALL
    #include "mkl.h"
#endif

//Eigen includes

#include "../eigen/Eigen/src/Core/util/DisableStupidWarnings.h"

#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#define _CRT_SECURE_NO_WARNINGS

// For numeric limits

#define NOMINMAX

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif

//#pragma warning(push, 0)
#include "tinyxml2.h"
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../eigen/unsupported/Eigen/CXX11/ThreadPool"
//#pragma warning(pop)

#ifdef OPENNN_CUDA

#include "../../opennn-cuda/opennn-cuda/kernel.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cublasXt.h>
#include <curand.h>

#endif

#include <omp.h>

#define	STRING(x) #x
#define TOSTRING(x) STRING(x)
#define LOG __FILE__ ":" TOSTRING(__LINE__)"\n"

namespace opennn
{
    using namespace std;
    using namespace Eigen;
    using type = float;
}

