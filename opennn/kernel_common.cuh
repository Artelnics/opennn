#ifndef KERNEL_COMMON_CUH
#define KERNEL_COMMON_CUH

#include <cstdint>
#include <cfloat>

#include "tensor_utilities.h"
#include "kernel.cuh"

static constexpr int block_size = 256;

static inline int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

static inline int grid_size_for(int n)
{
    return ceil_div(n, block_size);
}

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

// Per-step RNN activation: computes output h and its derivative dh w.r.t. the
// pre-activation z. Identity (0) and Softmax (4, degenerate per-step) -> identity.
__device__ inline void rnn_activation(int activation_id, float z, float& h, float& dh)
{
    switch (activation_id)
    {
        case 1:  // Sigmoid
            h  = 1.0f / (1.0f + expf(-z));
            dh = h * (1.0f - h);
            break;
        case 2:  // Tanh
            h  = tanhf(z);
            dh = 1.0f - h * h;
            break;
        case 3:  // ReLU
            h  = z > 0.0f ? z : 0.0f;
            dh = z > 0.0f ? 1.0f : 0.0f;
            break;
        case 0:  // Identity
        case 4:  // Softmax (degenerate per-step -> identity)
        default:
            h  = z;
            dh = 1.0f;
            break;
    }
}

#endif // KERNEL_COMMON_CUH
