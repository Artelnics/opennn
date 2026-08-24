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
void deallocate(Device, void*, Index) noexcept;
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

// One thread per element, on a caller-chosen stream. The kernel takes the
// element count as its first parameter. A null stream means the compute stream.
template<typename K, typename... Args>
static inline void launch_elementwise_on(cudaStream_t stream, Index n, K kernel, Args... args)
{
    if (n <= 0) return;
    if (stream == nullptr) stream = opennn::device::get_compute_stream();
    const int total = checked_int(n);
    OPENNN_CUDA_LAUNCH(kernel<<<grid_size_for(total), block_size, 0, stream>>>(total, args...));
}

template<typename K, typename... Args>
static inline void launch_elementwise(Index n, K kernel, Args... args)
{
    launch_elementwise_on(nullptr, n, kernel, args...);
}

// One warp per row: the kernel derives its row from the global warp id and
// takes the row count as its first parameter.
template<typename K, typename... Args>
static inline void launch_warp_rows(cudaStream_t stream, Index rows, K kernel, Args... args)
{
    if (rows <= 0) return;
    if (stream == nullptr) stream = opennn::device::get_compute_stream();
    const int blocks = checked_int(ceil_div(rows * 32, Index(block_size)));
    OPENNN_CUDA_LAUNCH(kernel<<<blocks, block_size, 0, stream>>>(checked_int(rows), args...));
}

// Threads for one row of the given width. A row narrower than the block does
// not need the rest of the block standing idle; below a warp there is nothing
// left to save.
static inline int threads_for_width(int width)
{
    if (width <= 32)  return 32;
    if (width <= 64)  return 64;
    if (width <= 128) return 128;
    return block_size;
}

template<typename K, typename... Args>
static inline void launch_elementwise_strided(Index n, K kernel, Args... args)
{
    if (n == 0) return;
    const int total = checked_int(n);
    OPENNN_CUDA_LAUNCH(kernel<<<grid_size_strided_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, args...));
}

// One-thread kernel (scalar bookkeeping such as folding a metric into a sum).
template<typename K, typename... Args>
static inline void launch_single(cudaStream_t stream, K kernel, Args... args)
{
    if (stream == nullptr) stream = opennn::device::get_compute_stream();
    OPENNN_CUDA_LAUNCH(kernel<<<1, 1, 0, stream>>>(args...));
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

// Elementwise activations by enum value. Here rather than beside the activation
// kernels because a kernel that produces a value can apply this to it before
// the store, which is what fusing an activation into an epilogue means; the
// single-output dense forward does exactly that.
__device__ __forceinline__ float opennn_activation_value(float x, int function)
{
    if (function == activation_sigmoid)    return sigmoid_f(x);
    if (function == activation_tanh)       return tanhf(x);
    if (function == activation_relu)       return fmaxf(x, 0.0f);
    if (function == activation_leaky_relu) return x >= 0.0f ? x : leaky_relu_slope * x;
    if (function == activation_gelu)       return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
    if (function == activation_gelu_tanh)
    {
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
    }
    if (function == activation_silu)       return x * sigmoid_f(x);
    return x;
}

static constexpr float padding_epsilon = 1e-7f;

template<typename T>
__device__ __forceinline__ bool token_is_padding(const T* token, int features)
{
    for (int e = 0; e < features; ++e)
        if (fabsf(static_cast<float>(token[e])) > padding_epsilon) return false;
    return true;
}

// One-thread row softmax over n contiguous elements: max, exp, sum, normalize
// by 1 / (sum + epsilon). dst may alias src. FastExp selects __expf over expf
// so each caller keeps its own numerics.
template<typename T, bool FastExp = false>
__device__ __forceinline__ void row_softmax(const T* src, T* dst, int n, float epsilon)
{
    float max_value = static_cast<float>(src[0]);
    for (int j = 1; j < n; ++j) max_value = fmaxf(max_value, static_cast<float>(src[j]));

    float sum = 0.0f;
    for (int j = 0; j < n; ++j)
    {
        const float x = static_cast<float>(src[j]) - max_value;
        const float e = FastExp ? __expf(x) : expf(x);
        dst[j] = static_cast<T>(e);
        sum += e;
    }
    const float inv_sum = 1.0f / (sum + epsilon);
    for (int j = 0; j < n; ++j) dst[j] = static_cast<T>(static_cast<float>(dst[j]) * inv_sum);
}

// Its backward for one row: dx = y * (dy - <y, dy>) * scale. dx may alias dy.
template<typename T>
__device__ __forceinline__ void row_softmax_backward(const T* y, const T* dy, T* dx, int n, float scale)
{
    float dot = 0.0f;
    for (int j = 0; j < n; ++j) dot += static_cast<float>(y[j]) * static_cast<float>(dy[j]);
    for (int j = 0; j < n; ++j)
        dx[j] = static_cast<T>(static_cast<float>(y[j]) * (static_cast<float>(dy[j]) - dot) * scale);
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
