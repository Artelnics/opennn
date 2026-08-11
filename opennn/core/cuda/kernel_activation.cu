//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A C T I V A T I O N   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// elementwise activations, SwiGLU and dropout

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/core/cuda/kernel_activation.cuh"
#include <curand_kernel.h>

template<typename T>
__global__ void swiglu_forward_kernel(const int n, const T* __restrict__ gate, const T* __restrict__ up, T* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float g = static_cast<float>(gate[i]);
    const float silu = g / (1.0f + expf(-g));
    out[i] = static_cast<T>(silu * static_cast<float>(up[i]));
}

template<typename T>
__global__ void swiglu_backward_kernel(const int n, const T* __restrict__ dout, const T* __restrict__ gate, const T* __restrict__ up, T* __restrict__ dgate, T* __restrict__ dup)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float d   = static_cast<float>(dout[i]);
    const float g   = static_cast<float>(gate[i]);
    const float sig = 1.0f / (1.0f + expf(-g));
    const float silu = g * sig;

    if (dup)   dup[i]   = static_cast<T>(d * silu);
    if (dgate) dgate[i] = static_cast<T>(d * static_cast<float>(up[i]) * sig * (1.0f + g * (1.0f - sig)));
}

template<typename T>
void swiglu_forward_cuda(const int n, const T* gate, const T* up, T* out)
{
    launch_elementwise(n, swiglu_forward_kernel<T>, gate, up, out);
}

template<typename T>
void swiglu_backward_cuda(const int n, const T* dout, const T* gate, const T* up, T* dgate, T* dup)
{
    launch_elementwise(n, swiglu_backward_kernel<T>, dout, gate, up, dgate, dup);
}

__device__ __forceinline__ float opennn_activation_value(float x, int function)
{
    if (function == activation_sigmoid)    return 1.0f / (1.0f + expf(-x));
    if (function == activation_tanh)       return tanhf(x);
    if (function == activation_relu)       return fmaxf(x, 0.0f);
    if (function == activation_leaky_relu) return x >= 0.0f ? x : leaky_relu_slope * x;
    if (function == activation_gelu)       return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
    if (function == activation_gelu_tanh)
    {
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
    }
    if (function == activation_silu)       return x / (1.0f + expf(-x));
    return x;
}

__device__ __forceinline__ float opennn_activation_grad(float y, float d, int function)
{
    if (function == activation_sigmoid)    return d * y * (1.0f - y);
    if (function == activation_tanh)       return d * (1.0f - y * y);
    if (function == activation_relu)       return y > 0.0f ? d : 0.0f;
    if (function == activation_leaky_relu) return y >= 0.0f ? d : leaky_relu_slope * d;
    if (function == activation_gelu)
    {
        const float cdf = 0.5f * (1.0f + erff(y * 0.70710678118654752440f));
        const float pdf = 0.39894228040143267794f * expf(-0.5f * y * y);
        return d * (cdf + y * pdf);
    }
    if (function == activation_gelu_tanh)
    {
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        const float y2 = y * y;
        const float u = sqrt_2_over_pi * (y + 0.044715f * y * y2);
        const float t = tanhf(u);
        const float du = sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * y2);
        return d * (0.5f * (1.0f + t) + 0.5f * y * (1.0f - t * t) * du);
    }
    if (function == activation_silu)
    {

        const float s = 1.0f / (1.0f + expf(-y));
        return d * s * (1.0f + y * (1.0f - s));
    }
    return d;
}

template<typename T>
__global__ void activation_forward_kernel(const int n, T* __restrict__ data, const int function)
{
    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += Index(blockDim.x) * gridDim.x)
        data[idx] = static_cast<T>(opennn_activation_value(static_cast<float>(data[idx]), function));
}

__global__ void activation_forward_kernel_bf162(const int n2, __nv_bfloat162* __restrict__ data, const int function)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n2; idx += blockDim.x * gridDim.x)
    {
        const float2 f = __bfloat1622float2(data[idx]);
        data[idx] = __floats2bfloat162_rn(opennn_activation_value(f.x, function),
                                          opennn_activation_value(f.y, function));
    }
}

__global__ void activation_forward_kernel_f4(const int n_vec, const int n, float* __restrict__ data, const int function)
{
    const Index tid = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = Index(blockDim.x) * gridDim.x;

    float4* __restrict__ const d4 = reinterpret_cast<float4*>(data);
    for (Index i = tid; i < n_vec; i += stride)
    {
        float4 v = d4[i];
        v.x = opennn_activation_value(v.x, function);
        v.y = opennn_activation_value(v.y, function);
        v.z = opennn_activation_value(v.z, function);
        v.w = opennn_activation_value(v.w, function);
        d4[i] = v;
    }

    const int tail_start = n_vec * 4;
    for (Index i = tail_start + tid; i < n; i += stride)
        data[i] = opennn_activation_value(data[i], function);
}

template<typename T>
void activation_forward_cuda(const Index n, T* data, const int function)
{
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
        if ((n & 1) == 0)
        {
            launch_elementwise_strided(n / 2, activation_forward_kernel_bf162, reinterpret_cast<__nv_bfloat162*>(data), function);
            return;
        }

    if constexpr (std::is_same_v<T, float>)
    {
        launch_vec4_on(opennn::device::get_compute_stream(), n, are_float4_aligned(data),
                       activation_forward_kernel_f4, data, function);
        return;
    }

    launch_elementwise_strided(n, activation_forward_kernel<T>, data, function);
}

template<typename T>
__global__ void activation_backward_kernel(const int n, const T* __restrict__ outputs, T* __restrict__ delta, const int function)
{
    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += Index(blockDim.x) * gridDim.x)
        delta[idx] = static_cast<T>(opennn_activation_grad(static_cast<float>(outputs[idx]),
                                                           static_cast<float>(delta[idx]), function));
}

__global__ void activation_backward_kernel_bf162(const int n2, const __nv_bfloat162* __restrict__ outputs,
                                                 __nv_bfloat162* __restrict__ delta, const int function)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n2; idx += blockDim.x * gridDim.x)
    {
        const float2 y = __bfloat1622float2(outputs[idx]);
        const float2 d = __bfloat1622float2(delta[idx]);
        delta[idx] = __floats2bfloat162_rn(opennn_activation_grad(y.x, d.x, function),
                                           opennn_activation_grad(y.y, d.y, function));
    }
}

__global__ void activation_backward_kernel_f4(const int n_vec, const int n,
                                              const float* __restrict__ outputs,
                                              float* __restrict__ delta, const int function)
{
    const Index tid = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = Index(blockDim.x) * gridDim.x;

    const float4* __restrict__ const y4 = reinterpret_cast<const float4*>(outputs);
    float4* __restrict__ const d4 = reinterpret_cast<float4*>(delta);
    for (Index i = tid; i < n_vec; i += stride)
    {
        const float4 y = y4[i];
        float4 d = d4[i];
        d.x = opennn_activation_grad(y.x, d.x, function);
        d.y = opennn_activation_grad(y.y, d.y, function);
        d.z = opennn_activation_grad(y.z, d.z, function);
        d.w = opennn_activation_grad(y.w, d.w, function);
        d4[i] = d;
    }

    const int tail_start = n_vec * 4;
    for (Index i = tail_start + tid; i < n; i += stride)
        delta[i] = opennn_activation_grad(outputs[i], delta[i], function);
}

template<typename T>
void activation_backward_cuda(const Index n, const T* outputs, T* delta, const int function)
{
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
        if ((n & 1) == 0)
        {
            launch_elementwise_strided(n / 2, activation_backward_kernel_bf162,
                               reinterpret_cast<const __nv_bfloat162*>(outputs),
                               reinterpret_cast<__nv_bfloat162*>(delta), function);
            return;
        }

    if constexpr (std::is_same_v<T, float>)
    {
        launch_vec4_on(opennn::device::get_compute_stream(), n, are_float4_aligned(outputs, delta),
                       activation_backward_kernel_f4, outputs, delta, function);
        return;
    }

    launch_elementwise_strided(n, activation_backward_kernel<T>, outputs, delta, function);
}

template<typename T>
__global__ void dropout_forward_kernel(const int n, T* __restrict__ output, uint8_t* __restrict__ mask, const float scale, const float rate, const unsigned long long seed)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    const float r = curand_uniform(&state);

    const uint8_t keep = (r >= rate) ? uint8_t(1) : uint8_t(0);
    mask[idx] = keep;
    output[idx] = static_cast<T>(static_cast<float>(output[idx]) * (keep * scale));
}

template<typename T>
void dropout_forward_cuda(const Index n, T* output, uint8_t* mask, const float rate, const unsigned long long seed)
{
    launch_elementwise(n, dropout_forward_kernel<T>, output, mask, 1.0f / (1.0f - rate), rate, seed);
}

template<typename T>
__global__ void dropout_backward_kernel(const int n, const T* __restrict__ output_delta, T* __restrict__ input_delta, const uint8_t* __restrict__ mask, const float scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float dy = static_cast<float>(output_delta[idx]);
    const float m  = static_cast<float>(mask[idx]) * scale;
    input_delta[idx] = static_cast<T>(dy * m);
}

template<typename T>
void dropout_backward_cuda(const Index n, const T* output_delta, T* input_delta, const uint8_t* mask, const float rate)
{
    launch_elementwise(n, dropout_backward_kernel<T>, output_delta, input_delta, mask, 1.0f / (1.0f - rate));
}

#define INSTANTIATE(T) \
    template void swiglu_forward_cuda<T>(const int, const T*, const T*, T*); \
    template void swiglu_backward_cuda<T>(const int, const T*, const T*, const T*, T*, T*); \
    template void activation_forward_cuda<T>(const Index, T*, const int); \
    template void activation_backward_cuda<T>(const Index, const T*, T*, const int); \
    template void dropout_forward_cuda<T>(const Index, T*, uint8_t*, const float, const unsigned long long); \
    template void dropout_backward_cuda<T>(const Index, const T*, T*, const uint8_t*, const float);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
