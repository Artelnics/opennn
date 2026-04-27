#ifndef KERNEL_COMMON_CUH
#define KERNEL_COMMON_CUH

// Shared host-side helpers for the split kernel TUs (kernel_optimizers.cu,
// kernel_losses.cu, kernel_layers.cu). Header-only, internal-linkage so each
// TU gets its own copy without ODR conflicts.

#include <cstdint>

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

// For unified vec+tail kernels: launch enough threads to cover whichever
// loop is larger; each thread grid-strides through both phases.
static inline int vector_work_size(int total, int n_vec, int vec_width)
{
    const int n_tail = total - n_vec * vec_width;
    return n_vec > n_tail ? n_vec : n_tail;
}

// 16-byte alignment is required for float4 loads/stores. cudaMalloc gives
// 256-byte alignment for the arena base, but per-layer offsets within the
// arena can land on any 4-byte boundary, so we must check each pointer.
static inline bool is_float4_aligned(const void* ptr)
{
    return (reinterpret_cast<std::uintptr_t>(ptr) & 0xF) == 0;
}

template<typename... Ptrs>
static inline bool are_float4_aligned(const Ptrs*... ptrs)
{
    return (is_float4_aligned(ptrs) && ...);
}

#endif // KERNEL_COMMON_CUH
