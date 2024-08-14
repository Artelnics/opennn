#ifndef OPENNN_CONFIG_H
#define OPENNN_CONFIG_H

#define NUMERIC_LIMITS_MIN type(0.000001)

//Eigen includes

#include "../eigen/Eigen/src/Core/util/DisableStupidWarnings.h"

#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

// For numeric limits

#define NOMINMAX

#define EIGEN_USE_THREADS

//#pragma warning(push, 0)
#include "../eigen/unsupported/Eigen/CXX11/Tensor"

//#define OPENNN_CUDA

#ifdef OPENNN_CUDA

#include "../../opennn_cuda/CudaOpennn/kernel.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cublasXt.h>
#include <curand.h>
#include <cudnn.h>

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

    using dimensions = vector<Index>;
     
    //using execution_policy = std::execution::par;

    template<typename Base, typename T>
    inline bool is_instance_of(const T* ptr)
    {
        return dynamic_cast<const Base*>(ptr) != nullptr;
    }
}

#endif

