#ifndef KERNEL_COMMON_CUH
#define KERNEL_COMMON_CUH

#ifdef OPENNN_HAS_CUDA

#include <limits>
#include <stdexcept>

#include "opennn/core/cuda/kernel_prelude.cuh"
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

static constexpr int class_activation_sigmoid = 1;

static constexpr float leaky_relu_slope = opennn::LEAKY_RELU_SLOPE;

template<typename I>
static inline I ceil_div(I a, I b)
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

// VecIO<T, VEC>: VEC adjacent elements moved as one aligned raw load/store (up
// to 16 bytes: eight BF16, four FP32 or sixteen bytes) and, for the float
// variants, converted element by element. The caller guarantees the address
// is VEC * sizeof(T) aligned.
template<int BYTES> struct RawBytes;
template<> struct RawBytes<1>  { unsigned char v; };
template<> struct RawBytes<2>  { unsigned short v; };
template<> struct RawBytes<4>  { unsigned int v; };
template<> struct RawBytes<8>  { uint2 v; };
template<> struct RawBytes<16> { uint4 v; };

__device__ static inline float element_to_float(float x) { return x; }
__device__ static inline float element_to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ static inline void element_from_float(float x, float& out) { out = x; }
__device__ static inline void element_from_float(float x, __nv_bfloat16& out) { out = __float2bfloat16(x); }

template<typename T, int VEC> struct VecIO
{
    using Raw = RawBytes<int(sizeof(T)) * VEC>;
    __device__ static void load(const T* p, T* out)
    {
        const Raw raw = *reinterpret_cast<const Raw*>(p);
        const T* e = reinterpret_cast<const T*>(&raw);
        #pragma unroll
        for (int k = 0; k < VEC; ++k) out[k] = e[k];
    }
    __device__ static void store(T* p, const T* v)
    {
        Raw raw;
        T* e = reinterpret_cast<T*>(&raw);
        #pragma unroll
        for (int k = 0; k < VEC; ++k) e[k] = v[k];
        *reinterpret_cast<Raw*>(p) = raw;
    }
    __device__ static void load_float(const T* p, float* out)
    {
        T v[VEC];
        load(p, v);
        #pragma unroll
        for (int k = 0; k < VEC; ++k) out[k] = element_to_float(v[k]);
    }
    __device__ static void store_float(T* p, const float* v)
    {
        T e[VEC];
        #pragma unroll
        for (int k = 0; k < VEC; ++k) element_from_float(v[k], e[k]);
        store(p, e);
    }
};

static inline int vector_work_size(int total, int n_vec, int vec_width)
{
    const int n_tail = total - n_vec * vec_width;
    return n_vec > n_tail ? n_vec : n_tail;
}

// Pointer alignment for vector loads; a null pointer counts as aligned so an
// optional operand does not disable the vector path.
template<int BYTES>
static inline bool is_aligned(const void* ptr)
{
    return ptr == nullptr || (reinterpret_cast<std::uintptr_t>(ptr) & (BYTES - 1)) == 0;
}
template<int BYTES, typename... Ptrs>
static inline bool are_aligned(const Ptrs*... ptrs)
{
    return (is_aligned<BYTES>(ptrs) && ...);
}
// Elements of T in a 16-byte vector: eight BF16, four FP32.
template<typename T> constexpr int vec16 = 16 / int(sizeof(T));

// Launches a kernel of the form (n_vec, n, args...): the first n_vec * VEC
// elements go through the vector path, the tail through the scalar one; with
// `aligned` false everything is scalar.
template<int VEC, typename K, typename... Args>
static inline void launch_vec_on(cudaStream_t stream, Index n, bool aligned, K kernel, Args... args)
{
    if (n == 0) return;
    const int total = checked_int(n);
    const int n_vec = aligned ? total / VEC : 0;
    OPENNN_CUDA_LAUNCH(kernel<<<grid_size_for(vector_work_size(total, n_vec, VEC)), block_size, 0,
                       stream>>>(n_vec, total, args...));
}

// Runtime float/bf16 choice for a templated call: f.template operator()<T>().
template<typename F>
static inline void dispatch_float_bf16(bool bf16, F&& f)
{
    if (bf16) f.template operator()<__nv_bfloat16>();
    else      f.template operator()<float>();
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


// Warp reductions (all 32 lanes must participate). warp_reduce_sum/max use
// xor shuffles and leave the result in every lane; warp_reduce_sum2 folds two
// sums in one pass with down shuffles and leaves them in lane 0.
__device__ __forceinline__ float warp_reduce_sum(float x)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        x += __shfl_xor_sync(0xffffffff, x, offset);
    return x;
}

__device__ __forceinline__ float warp_reduce_max(float x)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset));
    return x;
}

__device__ __forceinline__ void warp_reduce_sum2(float& a, float& b)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        a += __shfl_down_sync(0xffffffff, a, offset);
        b += __shfl_down_sync(0xffffffff, b, offset);
    }
}

// Block-wide reductions; the result is valid in thread 0, which they report.
__device__ __forceinline__ bool block_reduce_sum(float& a)
{
    a = warp_reduce_sum(a);

    __shared__ float warp_a[32];
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_a[warp_id] = a;
    __syncthreads();

    const int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0)
        a = warp_reduce_sum(threadIdx.x < num_warps ? warp_a[threadIdx.x] : 0.0f);
    return threadIdx.x == 0;
}

__device__ __forceinline__ bool block_reduce_sum2(float& a, float& b)
{
    warp_reduce_sum2(a, b);

    __shared__ float warp_a[32];
    __shared__ float warp_b[32];

    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0)
    {
        warp_a[warp_id] = a;
        warp_b[warp_id] = b;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0)
    {
        a = (threadIdx.x < num_warps) ? warp_a[threadIdx.x] : 0.0f;
        b = (threadIdx.x < num_warps) ? warp_b[threadIdx.x] : 0.0f;
        warp_reduce_sum2(a, b);
    }
    return threadIdx.x == 0;
}

// NHWC index arithmetic: flat element index -> (n, h, w, c).
__device__ __forceinline__ void nhwc_decompose(Index i, int channels, int width, int height,
                                               Index& n, int& h, int& w, int& c)
{
    c = int(i % channels); i /= channels;
    w = int(i % width);    i /= width;
    h = int(i % height);
    n = i / height;
}

#endif // OPENNN_HAS_CUDA

#endif
