//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// layer, RMS and batch normalization

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/core/cuda/kernel_normalization.cuh"

template<typename T, bool FuseResidual, bool HasMean>
__global__ void norm_forward_kernel(const int N, const int D, const T* __restrict__ X, const T* __restrict__ R, T* __restrict__ sum, T* __restrict__ Y, float* __restrict__ means, float* __restrict__ inv_vars, const float* __restrict__ gamma, const float* __restrict__ beta, const float eps)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const T* x_row = X + idx * D;
    T* y_row = Y + idx * D;
    T* s_row = FuseResidual ? sum + idx * D : nullptr;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        float x;
        if constexpr (FuseResidual)
        {
            x = static_cast<float>(x_row[i]) + static_cast<float>(R[idx * D + i]);
            s_row[i] = static_cast<T>(x);
        }
        else
            x = static_cast<float>(x_row[i]);

        if constexpr (HasMean) local_sum += x;
        local_sum_sq += x * x;
    }

    __shared__ float s_mean;
    __shared__ float s_inv_var;

    if (block_reduce_sum2(local_sum, local_sum_sq))
    {
        const float inv_D = 1.0f / static_cast<float>(D);
        if constexpr (HasMean)
        {
            const float mean = local_sum * inv_D;

            const float variance = fmaxf(local_sum_sq * inv_D - mean * mean, 0.0f);
            const float inv_var = rsqrtf(variance + eps);
            s_mean    = mean;
            s_inv_var = inv_var;
            means[idx]    = mean;
            inv_vars[idx] = inv_var;
        }
        else
        {
            const float inv_var = rsqrtf(local_sum_sq * inv_D + eps);
            s_inv_var = inv_var;
            if (inv_vars) inv_vars[idx] = inv_var;
        }
    }
    __syncthreads();

    const float inv_var = s_inv_var;
    float mean = 0.0f;
    if constexpr (HasMean) mean = s_mean;

    const T* src_row = FuseResidual ? s_row : x_row;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        if constexpr (HasMean)
        {
            const float x_hat = (static_cast<float>(src_row[i]) - mean) * inv_var;
            y_row[i] = static_cast<T>(fmaf(gamma[i], x_hat, beta[i]));
        }
        else
        {
            const float x_hat = static_cast<float>(src_row[i]) * inv_var;
            y_row[i] = static_cast<T>(gamma[i] * x_hat);
        }
    }
}

static inline int layernorm_threads(int D)
{
    if (D <= 32) return 32;
    if (D <= 64) return 64;
    if (D <= 128) return 128;
    return 256;
}

template<typename T>
__global__ void batchnorm_inference_kernel(const Index total, const int channels,
                                           const T* __restrict__ x,
                                           const T* __restrict__ residual,
                                           const float* __restrict__ gamma,
                                           const float* __restrict__ beta,
                                           const float* __restrict__ mean,
                                           const float* __restrict__ variance,
                                           const float epsilon,
                                           const int apply_relu,
                                           T* __restrict__ y)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += Index(blockDim.x) * gridDim.x)
    {
        const int c = int(i % channels);
        const float scale = gamma[c] * rsqrtf(variance[c] + epsilon);
        float value = (static_cast<float>(x[i]) - mean[c]) * scale + beta[c];
        if (residual) value += static_cast<float>(residual[i]);
        if (apply_relu) value = fmaxf(value, 0.0f);
        y[i] = static_cast<T>(value);
    }
}

template<typename T>
void batchnorm_inference_cuda(const Index total, const Index channels,
                              const T* x, const T* residual,
                              const float* gamma, const float* beta,
                              const float* mean, const float* variance,
                              const float epsilon, const bool apply_relu, T* y)
{
    if (channels == 0) return;
    launch_elementwise_strided(total, batchnorm_inference_kernel<T>, checked_int(channels),
                       x, residual, gamma, beta, mean, variance,
                       epsilon, apply_relu ? 1 : 0, y);
}

__global__ void conv_bn_fold_kernel(const Index total, const int kernel_size, const int kernels,
                                    const float* __restrict__ weights,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ beta,
                                    const float* __restrict__ mean,
                                    const float* __restrict__ variance,
                                    const float epsilon,
                                    float* __restrict__ folded_weights,
                                    float* __restrict__ folded_bias)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += Index(blockDim.x) * gridDim.x)
    {
        const int k = int(i / kernel_size);
        const int r = int(i % kernel_size);
        const float scale = gamma[k] * rsqrtf(variance[k] + epsilon);
        folded_weights[Index(r) * kernels + k] = weights[i] * scale;
        if (r == 0)
            folded_bias[k] = beta[k] - mean[k] * scale;
    }
}

void conv_bn_fold_cuda(const Index kernels, const Index kernel_size,
                       const float* weights,
                       const float* gamma, const float* beta,
                       const float* mean, const float* variance,
                       const float epsilon,
                       float* folded_weights, float* folded_bias)
{
    launch_elementwise_strided(kernels * kernel_size, conv_bn_fold_kernel,
                       checked_int(kernel_size), checked_int(kernels), weights,
                       gamma, beta, mean, variance, epsilon,
                       folded_weights, folded_bias);
}

__global__ void add_relu_kernel(const Index total,
                                const float* __restrict__ a,
                                const float* __restrict__ b,
                                const int apply_relu,
                                float* __restrict__ y)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += Index(blockDim.x) * gridDim.x)
    {
        const float value = a[i] + b[i];
        y[i] = apply_relu ? fmaxf(value, 0.0f) : value;
    }
}

void add_relu_cuda(const Index total, const float* a, const float* b,
                   const bool apply_relu, float* y)
{
    launch_elementwise_strided(total, add_relu_kernel, a, b, apply_relu ? 1 : 0, y);
}

template<typename T>
void layernorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps)
{
    if (N == 0 || D == 0) return;

    OPENNN_CUDA_LAUNCH((norm_forward_kernel<T, false, true><<<N, layernorm_threads(D), 0, opennn::device::get_compute_stream()>>>(N, D, X, nullptr, nullptr, Y, means, inv_vars, gamma, beta, eps)));
}

template<typename T>
void layernorm_add_forward_cuda(const int N, const int D, const T* X, const T* R, T* sum, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps)
{
    if (N == 0 || D == 0) return;

    OPENNN_CUDA_LAUNCH((norm_forward_kernel<T, true, true><<<N, layernorm_threads(D), 0, opennn::device::get_compute_stream()>>>(N, D, X, R, sum, Y, means, inv_vars, gamma, beta, eps)));
}

template<typename T, bool HasMean>
__global__ void norm_backward_kernel(const int N, const int D, const T* __restrict__ dY, const T* __restrict__ X, const float* __restrict__ means, const float* __restrict__ inv_vars, const float* __restrict__ gamma, T* __restrict__ dX)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const T* dy_row = dY + idx * D;
    const T* x_row = X + idx * D;
    T* dx_row = dX + idx * D;

    float mean = 0.0f;
    if constexpr (HasMean) mean = means[idx];
    const float inv_var = inv_vars[idx];

    float local_sum_D      = 0.0f;
    float local_sum_D_xhat = 0.0f;

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float d     = static_cast<float>(dy_row[i]) * gamma[i];
        float x_hat;
        if constexpr (HasMean) x_hat = (static_cast<float>(x_row[i]) - mean) * inv_var;
        else                   x_hat = static_cast<float>(x_row[i]) * inv_var;
        if constexpr (HasMean) local_sum_D += d;
        local_sum_D_xhat += d * x_hat;
    }

    __shared__ float s_mean_D;
    __shared__ float s_mean_D_xhat;

    if (block_reduce_sum2(local_sum_D, local_sum_D_xhat))
    {
        const float inv_D = 1.0f / static_cast<float>(D);
        if constexpr (HasMean) s_mean_D = local_sum_D * inv_D;
        s_mean_D_xhat = local_sum_D_xhat * inv_D;
    }
    __syncthreads();

    float mean_D = 0.0f;
    if constexpr (HasMean) mean_D = s_mean_D;
    const float mean_D_xhat = s_mean_D_xhat;

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float d     = static_cast<float>(dy_row[i]) * gamma[i];
        float x_hat;
        if constexpr (HasMean) x_hat = (static_cast<float>(x_row[i]) - mean) * inv_var;
        else                   x_hat = static_cast<float>(x_row[i]) * inv_var;
        if constexpr (HasMean)
            dx_row[i] = static_cast<T>((d - mean_D - x_hat * mean_D_xhat) * inv_var);
        else
            dx_row[i] = static_cast<T>((d - x_hat * mean_D_xhat) * inv_var);
    }
}

template<typename T, int NUM_WARPS, bool HasMean>
__global__ void norm_weight_gradient_coalesced_kernel(const int N, const int D,
                                                      const int chunk,
                                                      const T* __restrict__ dY,
                                                      const T* __restrict__ X,
                                                      const float* __restrict__ means,
                                                      const float* __restrict__ inv_vars,
                                                      float* __restrict__ dGamma,
                                                      float* __restrict__ dBeta)
{
    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int d       = blockIdx.x * 32 + lane;
    const bool active = (d < D);
    const int n0      = blockIdx.y * chunk;
    const int n1      = min(N, n0 + chunk);

    float local_gamma = 0.0f;
    float local_beta  = 0.0f;

    if (active)
    {
        for (int n = n0 + warp_id; n < n1; n += NUM_WARPS)
        {
            const float dy    = static_cast<float>(dY[n * D + d]);
            float x_hat;
            if constexpr (HasMean) x_hat = (static_cast<float>(X[n * D + d]) - means[n]) * inv_vars[n];
            else                   x_hat = static_cast<float>(X[n * D + d]) * inv_vars[n];
            local_gamma += dy * x_hat;
            if constexpr (HasMean) local_beta += dy;
        }
    }

    __shared__ float partial_gamma[NUM_WARPS][32];
    __shared__ float partial_beta [HasMean ? NUM_WARPS : 1][32];

    partial_gamma[warp_id][lane] = local_gamma;
    if constexpr (HasMean) partial_beta[warp_id][lane] = local_beta;
    __syncthreads();

    if (warp_id == 0 && active)
    {
        float g = 0.0f;
        float b = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w)
        {
            g += partial_gamma[w][lane];
            if constexpr (HasMean) b += partial_beta[w][lane];
        }
        if (gridDim.y == 1)
        {
            dGamma[d] = g;
            if constexpr (HasMean) dBeta[d] = b;
        }
        else
        {
            atomicAdd(dGamma + d, g);
            if constexpr (HasMean) atomicAdd(dBeta + d, b);
        }
    }
}

template<typename T, bool HasMean>
static void norm_backward_launch(const int N, const int D, const T* dY, const T* X, const float* means, const float* inv_vars, const float* gamma, T* dX, float* dGamma, float* dBeta)
{
    if (dX)
        OPENNN_CUDA_LAUNCH((norm_backward_kernel<T, HasMean><<<N, layernorm_threads(D), 0, opennn::device::get_compute_stream()>>>(N, D, dY, X, means, inv_vars, gamma, dX)));

    constexpr int NUM_WARPS = 8;
    const dim3 block(32, NUM_WARPS);
    const int grid_x = (D + 31) / 32;

    const int desired_chunks = grid_x < 192 ? 192 / grid_x : 1;
    int chunk = ceil_div(N, desired_chunks);
    if (chunk < NUM_WARPS * 8) chunk = NUM_WARPS * 8;
    const int grid_y = ceil_div(N, chunk);
    if (grid_y > 1)
    {
        const cudaStream_t stream = opennn::device::get_compute_stream();
        cudaMemsetAsync(dGamma, 0, size_t(D) * sizeof(float), stream);
        if constexpr (HasMean) cudaMemsetAsync(dBeta, 0, size_t(D) * sizeof(float), stream);
    }
    norm_weight_gradient_coalesced_kernel<T, NUM_WARPS, HasMean><<<dim3(grid_x, grid_y), block, 0,
        opennn::device::get_compute_stream()>>>(N, D, chunk, dY, X, means, inv_vars, dGamma, dBeta);
    opennn::device::check_last_error();
}

template<typename T>
void layernorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* means, const float* inv_vars, const float* gamma, T* dX, float* dGamma, float* dBeta)
{
    if (N == 0 || D == 0) return;

    norm_backward_launch<T, true>(N, D, dY, X, means, inv_vars, gamma, dX, dGamma, dBeta);
}

template<typename T>
void rmsnorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* inv_rms, const float* weight, const float eps)
{
    if (N == 0 || D == 0) return;

    OPENNN_CUDA_LAUNCH((norm_forward_kernel<T, false, false><<<N, layernorm_threads(D), 0, opennn::device::get_compute_stream()>>>(N, D, X, nullptr, nullptr, Y, nullptr, inv_rms, weight, nullptr, eps)));
}

template<typename T>
void rmsnorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* inv_rms, const float* weight, T* dX, float* dWeight)
{
    if (N == 0 || D == 0) return;

    norm_backward_launch<T, false>(N, D, dY, X, nullptr, inv_rms, weight, dX, dWeight, nullptr);
}

#define INSTANTIATE(T) \
    template void batchnorm_inference_cuda<T>(const Index, const Index, const T*, const T*, const float*, const float*, const float*, const float*, const float, const bool, T*); \
    template void layernorm_forward_cuda<T>(const int, const int, const T*, T*, float*, float*, const float*, const float*, const float); \
    template void layernorm_add_forward_cuda<T>(const int, const int, const T*, const T*, T*, T*, float*, float*, const float*, const float*, const float); \
    template void layernorm_backward_cuda<T>(const int, const int, const T*, const T*, const float*, const float*, const float*, T*, float*, float*); \
    template void rmsnorm_forward_cuda<T>(const int, const int, const T*, T*, float*, const float*, const float); \
    template void rmsnorm_backward_cuda<T>(const int, const int, const T*, const T*, const float*, const float*, T*, float*);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
