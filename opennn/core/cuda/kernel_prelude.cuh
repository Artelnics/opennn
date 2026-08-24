#ifndef KERNEL_PRELUDE_CUH
#define KERNEL_PRELUDE_CUH

#ifdef OPENNN_HAS_CUDA

#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#if defined(__CUDACC__) && defined(_MSC_VER)
__host__ __device__ inline float arg(float x) noexcept
{
    return x < 0.0f ? 3.14159265358979323846f : 0.0f;
}

__host__ __device__ inline double arg(double x) noexcept
{
    return x < 0.0 ? 3.14159265358979323846 : 0.0;
}

template<typename Integer, typename = std::enable_if_t<std::is_integral_v<Integer>>>
__host__ __device__ inline double arg(Integer x) noexcept
{
    if constexpr (std::is_signed_v<Integer>)
        return x < 0 ? 3.14159265358979323846 : 0.0;
    else
        return 0.0;
}
#endif

#include <Eigen/Core>

using Eigen::Index;

#endif

#endif
