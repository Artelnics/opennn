#ifndef KERNEL_COMMON_CUH
#define KERNEL_COMMON_CUH

#include <cstdint>
#include <cfloat>
#include <limits>
#include <stdexcept>

#include "kernel.cuh"
#include "types.h"

namespace opennn::device
{

void check_last_error();
void* allocate(Device, Index);
void deallocate(Device, void*, Index);
void set_zero_async(void*, Index, cudaStream_t);
cudaStream_t get_compute_stream();

}

static constexpr int block_size = 256;

static constexpr int activation_identity  = 0;
static constexpr int activation_sigmoid   = 1;
static constexpr int activation_tanh      = 2;
static constexpr int activation_relu      = 3;
static constexpr int activation_softmax   = 4;
static constexpr int activation_leaky_relu = 5;
static constexpr int activation_gelu      = 6;

static constexpr int class_activation_softmax = 0;
static constexpr int class_activation_sigmoid = 1;

static inline int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

static inline int grid_size_for(int n)
{
    return ceil_div(n, block_size);
}

static inline int checked_int(Index value)
{
    if (value > Index(std::numeric_limits<int>::max())
        || value < Index(std::numeric_limits<int>::min()))
        throw std::runtime_error("CUDA wrapper value exceeds int range.");
    return static_cast<int>(value);
}

static inline void checked_host_condition(bool condition, const char* message)
{
    if (condition) throw std::runtime_error(message);
}

#define OPENNN_CUDA_LAUNCH(...) \
    do {                        \
        __VA_ARGS__;            \
        opennn::device::check_last_error(); \
    } while (false)

static inline int vector_work_size(int total, int n_vec, int vec_width)
{
    const int n_tail = total - n_vec * vec_width;
    return n_vec > n_tail ? n_vec : n_tail;
}

static inline bool is_float4_aligned(const void* ptr)
{
    return (reinterpret_cast<std::uintptr_t>(ptr) & 0xF) == 0;
}

template<typename... Ptrs>
static inline bool are_float4_aligned(const Ptrs*... ptrs)
{
    return (is_float4_aligned(ptrs) && ...);
}

__device__ __forceinline__ float sigmoid_f(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ inline void rnn_activation(int activation_id, float z, float& h, float& dh)
{
    switch (activation_id)
    {
        case activation_sigmoid:
            h  = sigmoid_f(z);
            dh = h * (1.0f - h);
            break;
        case activation_tanh:
            h  = tanhf(z);
            dh = 1.0f - h * h;
            break;
        case activation_relu:
            h  = z > 0.0f ? z : 0.0f;
            dh = z > 0.0f ? 1.0f : 0.0f;
            break;
        case activation_identity:
        case activation_softmax:
        default:
            h  = z;
            dh = 1.0f;
            break;
    }
}

#endif // KERNEL_COMMON_CUH
