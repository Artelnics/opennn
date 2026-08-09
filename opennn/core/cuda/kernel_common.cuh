#ifndef KERNEL_COMMON_CUH
#define KERNEL_COMMON_CUH

#include <cstdint>
#include <cfloat>
#include <limits>
#include <stdexcept>

#include "opennn/core/cuda/kernel.cuh"
#include "opennn/core/configuration.h"

namespace opennn::device
{

void check_last_error();
void* allocate(Device, Index);
void deallocate(Device, void*, Index);
void set_zero_async(void*, Index, cudaStream_t);
cudaStream_t get_compute_stream();

}

static constexpr int block_size = 256;

static constexpr int activation_identity   = int(opennn::ActivationFunction::Identity);
static constexpr int activation_sigmoid    = int(opennn::ActivationFunction::Sigmoid);
static constexpr int activation_tanh       = int(opennn::ActivationFunction::Tanh);
static constexpr int activation_relu       = int(opennn::ActivationFunction::ReLU);
static constexpr int activation_softmax    = int(opennn::ActivationFunction::Softmax);
static constexpr int activation_leaky_relu = int(opennn::ActivationFunction::LeakyReLU);
static constexpr int activation_gelu       = int(opennn::ActivationFunction::GELU);
static constexpr int activation_gelu_tanh  = int(opennn::ActivationFunction::GELUTanh);
static constexpr int activation_silu       = int(opennn::ActivationFunction::SiLU);

static constexpr int class_activation_softmax = 0;
static constexpr int class_activation_sigmoid = 1;

static constexpr float leaky_relu_slope = opennn::LEAKY_RELU_SLOPE;

static inline int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

static inline int grid_size_for(int n)
{
    return ceil_div(n, block_size);
}

static inline int grid_size_strided_for(int n)
{
    static const int max_blocks = [] {
        int device = 0, sm_count = 0;
        if (cudaGetDevice(&device) == cudaSuccess
            && cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) == cudaSuccess
            && sm_count > 0)
            return 32 * sm_count;
        cudaGetLastError();
        return std::numeric_limits<int>::max();
    }();
    const int blocks = ceil_div(n, block_size);
    return blocks < max_blocks ? blocks : max_blocks;
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

template<typename K, typename... Args>
static inline void launch_elementwise(Index n, K kernel, Args... args)
{
    if (n == 0) return;
    const int total = checked_int(n);
    OPENNN_CUDA_LAUNCH(kernel<<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, args...));
}

template<typename K, typename... Args>
static inline void launch_elementwise_strided(Index n, K kernel, Args... args)
{
    if (n == 0) return;
    const int total = checked_int(n);
    OPENNN_CUDA_LAUNCH(kernel<<<grid_size_strided_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, args...));
}

#define OPENNN_INSTANTIATE_FLOAT_BF16(X) \
    X(float)                             \
    X(__nv_bfloat16)

#define OPENNN_INSTANTIATE_FLOAT_BF16_2(X) \
    X(float, float)                        \
    X(float, __nv_bfloat16)                \
    X(__nv_bfloat16, float)                \
    X(__nv_bfloat16, __nv_bfloat16)

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

static inline bool is_bfloat162_aligned(const void* ptr)
{
    return ptr == nullptr || (reinterpret_cast<std::uintptr_t>(ptr) & 0x3) == 0;
}

template<typename K, typename... Args>
static inline void launch_vec4_on(cudaStream_t stream, Index n, bool aligned, K kernel, Args... args)
{
    if (n == 0) return;
    const int total = checked_int(n);
    const int n_vec = aligned ? total / 4 : 0;
    OPENNN_CUDA_LAUNCH(kernel<<<grid_size_for(vector_work_size(total, n_vec, 4)), block_size, 0,
                       stream>>>(n_vec, total, args...));
}

template<typename K, typename... Args>
static inline void launch_vec4(Index n, bool aligned, K kernel, Args... args)
{
    launch_vec4_on(opennn::device::get_compute_stream(), n, aligned, kernel, args...);
}

__device__ __forceinline__ float sigmoid_f(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static constexpr float padding_epsilon = 1e-7f;

template<typename T>
__device__ __forceinline__ bool token_is_padding(const T* token, int features)
{
    for (int e = 0; e < features; ++e)
        if (fabsf(static_cast<float>(token[e])) > padding_epsilon) return false;
    return true;
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

#endif
