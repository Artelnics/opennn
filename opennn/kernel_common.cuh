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

__device__ inline void rnn_activation(int activation_id, float z, float& h, float& dh)
{
    switch (activation_id)
    {
        case 1:
            h  = 1.0f / (1.0f + expf(-z));
            dh = h * (1.0f - h);
            break;
        case 2:
            h  = tanhf(z);
            dh = 1.0f - h * h;
            break;
        case 3:
            h  = z > 0.0f ? z : 0.0f;
            dh = z > 0.0f ? 1.0f : 0.0f;
            break;
        case 0:
        case 4:
        default:
            h  = z;
            dh = 1.0f;
            break;
    }
}

#endif // KERNEL_COMMON_CUH
