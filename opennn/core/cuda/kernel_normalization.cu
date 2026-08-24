//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/core/device_backend.h"
#include "opennn/core/cuda/kernel_normalization.cuh"

template<typename T, bool FuseResidual, bool HasMean>
__global__ void norm_forward_kernel(const int N, const int D, const T* __restrict__ X, const T* __restrict__ R, T* __restrict__ sum, T* __restrict__ Y, float* __restrict__ means, float* __restrict__ inv_vars, const float* __restrict__ gamma, const float* __restrict__ beta, const float eps)
{
    const int idx = blockIdx.x;

    const Index row_base = Index(idx) * Index(D);

    const T* x_row = X + row_base;
    T* y_row = Y + row_base;
    T* s_row = FuseResidual ? sum + row_base : nullptr;

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

constexpr int BN_THREADS = 256;
constexpr Index BN_MAX_ROW_BLOCKS = 128;

__device__ static inline unsigned batchnorm_mask_bits(const uint8_t* mask, const Index i, const int c0)
{
    return mask ? unsigned(mask[i / 8]) >> (c0 & 4) : 0xFFu;
}

Index batchnorm_partial_rows(const Index rows)
{
    return rows < BN_MAX_ROW_BLOCKS ? (rows <= 0 ? 1 : rows) : BN_MAX_ROW_BLOCKS;
}

static dim3 batchnorm_reduce_block(const Index channel_groups)
{
    const int lanes = int(channel_groups < 32 ? channel_groups : 32);
    return dim3(unsigned(lanes), unsigned(BN_THREADS / lanes));
}

static Index batchnorm_row_blocks(const Index rows, const dim3& reduce_block)
{
    const Index wanted = rows / (Index(reduce_block.y) * 4);
    const Index cap = batchnorm_partial_rows(rows);
    return wanted < 1 ? 1 : (wanted > cap ? cap : wanted);
}

constexpr int BN_FINALIZE_LANES = 8;

__device__ static bool batchnorm_sum_partials(const int channels, const int row_blocks,
                                              const float* __restrict__ partials,
                                              int& c, float& s1, float& s2)
{
    __shared__ float sh1[BN_FINALIZE_LANES][32];
    __shared__ float sh2[BN_FINALIZE_LANES][32];
    c = blockIdx.x * 32 + threadIdx.x;
    float t1 = 0.0f, t2 = 0.0f;
    if (c < channels)
        for (int b = threadIdx.y; b < row_blocks; b += BN_FINALIZE_LANES)
        {
            const Index slot = (Index(b) * channels + c) * 2;
            t1 += partials[slot];
            t2 += partials[slot + 1];
        }
    sh1[threadIdx.y][threadIdx.x] = t1;
    sh2[threadIdx.y][threadIdx.x] = t2;
    __syncthreads();
    if (threadIdx.y != 0 || c >= channels) return false;
    s1 = s2 = 0.0f;
    #pragma unroll
    for (int j = 0; j < BN_FINALIZE_LANES; ++j)
    {
        s1 += sh1[j][threadIdx.x];
        s2 += sh2[j][threadIdx.x];
    }
    return true;
}

template<int VEC>
__device__ static void batchnorm_store_partials(const int channels, const int c0,
                                                const float* s1, const float* s2,
                                                float* __restrict__ partials)
{
    __shared__ float sh1[BN_THREADS * VEC];
    __shared__ float sh2[BN_THREADS * VEC];
    const int lane = (threadIdx.y * blockDim.x + threadIdx.x) * VEC;
    #pragma unroll
    for (int k = 0; k < VEC; ++k)
    {
        sh1[lane + k] = s1[k];
        sh2[lane + k] = s2[k];
    }
    __syncthreads();

    if (threadIdx.y != 0) return;
    #pragma unroll
    for (int k = 0; k < VEC; ++k)
    {
        const int c = c0 + k;
        if (c >= channels) continue;
        float t1 = 0.0f, t2 = 0.0f;
        for (unsigned j = 0; j < blockDim.y; ++j)
        {
            const int idx = (j * blockDim.x + threadIdx.x) * VEC + k;
            t1 += sh1[idx];
            t2 += sh2[idx];
        }
        const Index slot = (Index(blockIdx.y) * channels + c) * 2;
        partials[slot] = t1;
        partials[slot + 1] = t2;
    }
}

struct BnReduceLaunch
{
    dim3 reduce_grid, reduce_block;
    Index row_blocks, rows_per_block;
    dim3 finalize_grid, finalize_block;
};

static BnReduceLaunch batchnorm_reduce_launch(const Index rows, const Index channels, const int vec)
{
    BnReduceLaunch launch;
    const Index channel_groups = channels / vec;
    launch.reduce_block = batchnorm_reduce_block(channel_groups);
    launch.row_blocks = batchnorm_row_blocks(rows, launch.reduce_block);
    launch.rows_per_block = ceil_div(rows, launch.row_blocks);
    launch.reduce_grid = dim3(unsigned(ceil_div(channel_groups, Index(launch.reduce_block.x))), unsigned(launch.row_blocks));
    launch.finalize_grid = dim3(unsigned(ceil_div(channels, Index(32))));
    launch.finalize_block = dim3(32, BN_FINALIZE_LANES);
    return launch;
}

template<typename T, int VEC>
__global__ void batchnorm_forward_reduce_kernel(const Index rows, const int channels,
                                                const Index rows_per_block,
                                                const T* __restrict__ x,
                                                float* __restrict__ partials)
{
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    const Index row_begin = Index(blockIdx.y) * rows_per_block;
    const Index row_end = min(rows, row_begin + rows_per_block);

    float s1[VEC], s2[VEC];
    #pragma unroll
    for (int k = 0; k < VEC; ++k) s1[k] = s2[k] = 0.0f;

    if (c0 < channels)
    {
        float v[VEC];
        for (Index r = row_begin + threadIdx.y; r < row_end; r += blockDim.y)
        {
            VecIO<T, VEC>::load_float(x + r * channels + c0, v);
            #pragma unroll
            for (int k = 0; k < VEC; ++k)
            {
                s1[k] += v[k];
                s2[k] += v[k] * v[k];
            }
        }
    }

    batchnorm_store_partials<VEC>(channels, c0, s1, s2, partials);
}

__global__ void batchnorm_forward_finalize_kernel(const int channels, const int row_blocks,
                                                  const float inv_rows, const float unbias,
                                                  const float epsilon, const float momentum,
                                                  const float* __restrict__ partials,
                                                  const float* __restrict__ gamma,
                                                  const float* __restrict__ beta,
                                                  float* __restrict__ mean,
                                                  float* __restrict__ inv_var,
                                                  float* __restrict__ running_mean,
                                                  float* __restrict__ running_var,
                                                  float* __restrict__ scale_shift)
{
    int c;
    float s1, s2;
    if (!batchnorm_sum_partials(channels, row_blocks, partials, c, s1, s2)) return;
    const float m = s1 * inv_rows;
    const float var = fmaxf(s2 * inv_rows - m * m, 0.0f);
    const float iv = 1.0f / sqrtf(var + epsilon);
    mean[c] = m;
    inv_var[c] = iv;
    running_mean[c] = running_mean[c] * (1.0f - momentum) + m * momentum;
    running_var[c]  = running_var[c]  * (1.0f - momentum) + var * unbias * momentum;
    const float scale = gamma[c] * iv;
    scale_shift[c] = scale;
    scale_shift[channels + c] = beta[c] - m * scale;
}

template<typename T, int VEC, bool RELU, bool ADD>
__global__ void batchnorm_forward_apply_kernel(const Index groups, const int channels,
                                               const T* __restrict__ x,
                                               const T* __restrict__ residual,
                                               const float* __restrict__ scale_shift,
                                               T* __restrict__ y,
                                               uint8_t* __restrict__ mask)
{
    static_assert(VEC == 8 || VEC == 4, "mask packing needs 8 or 4 channels per lane");
    const int channel_groups = channels / VEC;
    for (Index base = Index(blockIdx.x) * blockDim.x; base < groups;
         base += Index(gridDim.x) * blockDim.x)
    {
        const Index gi = base + threadIdx.x;
        const bool valid = gi < groups;
        unsigned bits = 0;
        if (valid)
        {
            const int c0 = int(gi % channel_groups) * VEC;
            const Index i = (gi / channel_groups) * channels + c0;
            float vx[VEC], vr[VEC], out[VEC];
            VecIO<T, VEC>::load_float(x + i, vx);
            if constexpr (ADD) VecIO<T, VEC>::load_float(residual + i, vr);
            #pragma unroll
            for (int k = 0; k < VEC; ++k)
            {
                const int c = c0 + k;
                float v = vx[k] * scale_shift[c] + scale_shift[channels + c];
                if constexpr (ADD) v += vr[k];
                if constexpr (RELU)
                {
                    v = fmaxf(v, 0.0f);
                    bits |= (v > 0.0f ? 1u : 0u) << k;
                }
                out[k] = v;
            }
            VecIO<T, VEC>::store_float(y + i, out);
        }
        if (RELU && mask)
        {
            if constexpr (VEC == 8) { if (valid) mask[gi] = uint8_t(bits); }
            else
            {
                const unsigned high = __shfl_down_sync(0xffffffffu, bits, 1);
                if (valid && (gi & 1) == 0) mask[gi >> 1] = uint8_t(bits | (high << 4));
            }
        }
    }
}

template<typename T>
void batchnorm_forward_fused_cuda(const Index rows, const Index channels,
                                  const T* x, const T* residual,
                                  const float* gamma, const float* beta,
                                  const float epsilon, const float momentum,
                                  T* y, float* mean, float* inv_var,
                                  float* running_mean, float* running_var,
                                  const bool relu, uint8_t* mask,
                                  float* partials)
{
    if (rows == 0 || channels == 0) return;
    checked_host_condition(channels % 8 != 0, "batchnorm_forward_fused_cuda: channels must be a multiple of 8.");
    constexpr int VEC = vec16<T>;
    cudaStream_t stream = opennn::device::get_compute_stream();

    const BnReduceLaunch g = batchnorm_reduce_launch(rows, channels, VEC);
    OPENNN_CUDA_LAUNCH((batchnorm_forward_reduce_kernel<T, VEC><<<g.reduce_grid, g.reduce_block, 0, stream>>>(
        rows, checked_int(channels), g.rows_per_block, x, partials)));

    float* scale_shift = partials + 2 * batchnorm_partial_rows(rows) * channels;
    const float unbias = rows > 1 ? float(rows) / float(rows - 1) : 1.0f;
    OPENNN_CUDA_LAUNCH((batchnorm_forward_finalize_kernel<<<g.finalize_grid, g.finalize_block, 0, stream>>>(
        checked_int(channels), checked_int(g.row_blocks), 1.0f / static_cast<float>(rows), unbias,
        epsilon, momentum, partials, gamma, beta, mean, inv_var, running_mean, running_var, scale_shift)));

    const bool add = residual != nullptr;
    const Index groups = rows * (channels / VEC);
    if (relu && add)       launch_elementwise_strided(groups, batchnorm_forward_apply_kernel<T, VEC, true,  true >, checked_int(channels), x, residual, scale_shift, y, mask);
    else if (relu)         launch_elementwise_strided(groups, batchnorm_forward_apply_kernel<T, VEC, true,  false>, checked_int(channels), x, residual, scale_shift, y, mask);
    else if (add)          launch_elementwise_strided(groups, batchnorm_forward_apply_kernel<T, VEC, false, true >, checked_int(channels), x, residual, scale_shift, y, mask);
    else                   launch_elementwise_strided(groups, batchnorm_forward_apply_kernel<T, VEC, false, false>, checked_int(channels), x, residual, scale_shift, y, mask);
}

template<typename T, int VEC, bool XHAT_FROM_Y>
__global__ void batchnorm_backward_reduce_kernel(const Index rows, const int channels,
                                                 const Index rows_per_block,
                                                 const T* __restrict__ x,
                                                 const T* __restrict__ dy,
                                                 const T* __restrict__ y,
                                                 const uint8_t* __restrict__ mask,
                                                 const float* __restrict__ gamma,
                                                 const float* __restrict__ beta,
                                                 const float* __restrict__ mean,
                                                 const float* __restrict__ inv_var,
                                                 float* __restrict__ partials)
{
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    const Index row_begin = Index(blockIdx.y) * rows_per_block;
    const Index row_end = min(rows, row_begin + rows_per_block);

    float s1[VEC], s2[VEC], m[VEC], iv[VEC], b[VEC], ig[VEC];
    #pragma unroll
    for (int k = 0; k < VEC; ++k)
    {
        s1[k] = s2[k] = 0.0f;
        const int c = c0 + k;
        const bool ok = c < channels;
        m[k]  = ok ? mean[c] : 0.0f;
        iv[k] = ok ? inv_var[c] : 0.0f;
        b[k]  = (ok && XHAT_FROM_Y) ? beta[c] : 0.0f;
        const float g = (ok && XHAT_FROM_Y) ? gamma[c] : 1.0f;
        ig[k] = fabsf(g) > 1e-30f ? 1.0f / g : 0.0f;
    }

    if (c0 < channels)
    {
        float vdy[VEC], vy[VEC], vx[VEC];
        for (Index r = row_begin + threadIdx.y; r < row_end; r += blockDim.y)
        {
            const Index i = r * channels + c0;
            VecIO<T, VEC>::load_float(dy + i, vdy);
            if (y) VecIO<T, VEC>::load_float(y + i, vy);
            if (!XHAT_FROM_Y) VecIO<T, VEC>::load_float(x + i, vx);
            const unsigned bits = batchnorm_mask_bits(mask, i, c0);
            #pragma unroll
            for (int k = 0; k < VEC; ++k)
            {
                float g = vdy[k];
                if (mask ? !((bits >> k) & 1u) : (y && vy[k] <= 0.0f)) g = 0.0f;
                const float x_hat = XHAT_FROM_Y ? (vy[k] - b[k]) * ig[k] : (vx[k] - m[k]) * iv[k];
                s1[k] += g;
                s2[k] += g * x_hat;
            }
        }
    }

    batchnorm_store_partials<VEC>(channels, c0, s1, s2, partials);
}

__global__ void batchnorm_backward_finalize_kernel(const int channels, const int row_blocks,
                                                   const float* __restrict__ partials,
                                                   float* __restrict__ dgamma,
                                                   float* __restrict__ dbeta)
{
    int c;
    float s1, s2;
    if (!batchnorm_sum_partials(channels, row_blocks, partials, c, s1, s2)) return;
    dbeta[c] = s1;
    dgamma[c] = s2;
}

template<typename T, int VEC>
__global__ void batchnorm_backward_apply_kernel(const Index groups, const int channels,
                                                const float inv_rows,
                                                const T* __restrict__ x,
                                                T* __restrict__ dy_dx,
                                                const T* __restrict__ y,
                                                const uint8_t* __restrict__ mask,
                                                const float* __restrict__ gamma,
                                                const float* __restrict__ mean,
                                                const float* __restrict__ inv_var,
                                                const float* __restrict__ dgamma,
                                                const float* __restrict__ dbeta,
                                                T* __restrict__ dpre)
{
    const int channel_groups = channels / VEC;
    for (Index gi = Index(blockIdx.x) * blockDim.x + threadIdx.x; gi < groups;
         gi += Index(gridDim.x) * blockDim.x)
    {
        const int c0 = int(gi % channel_groups) * VEC;
        const Index i = (gi / channel_groups) * channels + c0;

        float vdy[VEC], vy[VEC], vx[VEC], out[VEC], pre[VEC];
        VecIO<T, VEC>::load_float(dy_dx + i, vdy);
        if (y) VecIO<T, VEC>::load_float(y + i, vy);
        VecIO<T, VEC>::load_float(x + i, vx);
        const unsigned bits = batchnorm_mask_bits(mask, i, c0);

        #pragma unroll
        for (int k = 0; k < VEC; ++k)
        {
            const int c = c0 + k;
            float g = vdy[k];
            if (mask ? !((bits >> k) & 1u) : (y && vy[k] <= 0.0f)) g = 0.0f;
            pre[k] = g;
            const float gm = gamma[c], iv = inv_var[c];
            const float x_hat = (vx[k] - mean[c]) * iv;
            out[k] = gm * iv * (g - dbeta[c] * inv_rows - x_hat * dgamma[c] * inv_rows);
        }
        if (dpre) VecIO<T, VEC>::store_float(dpre + i, pre);
        VecIO<T, VEC>::store_float(dy_dx + i, out);
    }
}

template<typename T, int VEC, bool XHAT_FROM_Y>
static void batchnorm_backward_launch(const Index rows, const Index channels,
                                      const T* x, T* dy_dx, const T* y, const uint8_t* mask,
                                      const float* gamma, const float* beta,
                                      const float* mean, const float* inv_var,
                                      T* dpre, float* dgamma, float* dbeta,
                                      float* partials, cudaStream_t stream)
{
    const BnReduceLaunch g = batchnorm_reduce_launch(rows, channels, VEC);
    OPENNN_CUDA_LAUNCH((batchnorm_backward_reduce_kernel<T, VEC, XHAT_FROM_Y><<<g.reduce_grid, g.reduce_block, 0, stream>>>(
        rows, checked_int(channels), g.rows_per_block, x, dy_dx, y, mask, gamma, beta, mean, inv_var, partials)));

    OPENNN_CUDA_LAUNCH((batchnorm_backward_finalize_kernel<<<g.finalize_grid, g.finalize_block, 0, stream>>>(
        checked_int(channels), checked_int(g.row_blocks), partials, dgamma, dbeta)));

    launch_elementwise_strided(rows * (channels / VEC), batchnorm_backward_apply_kernel<T, VEC>,
                               checked_int(channels), 1.0f / static_cast<float>(rows),
                               x, dy_dx, y, mask, gamma, mean, inv_var, dgamma, dbeta, dpre);
}

template<typename T>
void batchnorm_backward_fused_cuda(const Index rows, const Index channels,
                                   const T* x, T* dy_dx, const T* y, const uint8_t* mask,
                                   const float* gamma, const float* beta,
                                   const float* mean, const float* inv_var,
                                   const bool xhat_from_y,
                                   T* dpre, float* dgamma, float* dbeta,
                                   float* partials)
{
    if (rows == 0 || channels == 0) return;
    cudaStream_t stream = opennn::device::get_compute_stream();

    const bool wide = channels % 8 == 0;
    const bool vec2 = channels % 2 == 0;
    if (!wide) mask = nullptr;
    if (mask) y = nullptr;

    const bool from_y = xhat_from_y && y != nullptr && beta != nullptr;

    const auto launch = [&]<int VEC>()
    {
        if (from_y) batchnorm_backward_launch<T, VEC, true >(rows, channels, x, dy_dx, y, mask, gamma, beta, mean, inv_var, dpre, dgamma, dbeta, partials, stream);
        else        batchnorm_backward_launch<T, VEC, false>(rows, channels, x, dy_dx, y, mask, gamma, beta, mean, inv_var, dpre, dgamma, dbeta, partials, stream);
    };
    if (wide)      launch.template operator()<vec16<T>>();
    else if (vec2) launch.template operator()<2>();
    else           launch.template operator()<1>();
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

static constexpr int norm_warp_rows_per_block = 8;

template<typename T, int ITER>
__device__ __forceinline__ bool norm_warp_shape(const int D)
{
    return D == 32 * vec16<T> * ITER;
}

template<typename T, bool FuseResidual, bool HasMean, int ITER>
__global__ void __launch_bounds__(256)
norm_forward_warp_kernel(const int N, const int D, const T* __restrict__ X, const T* __restrict__ R,
                         T* __restrict__ sum, T* __restrict__ Y, float* __restrict__ means,
                         float* __restrict__ inv_vars, const float* __restrict__ gamma,
                         const float* __restrict__ beta, const float eps)
{
    constexpr int VEC = vec16<T>;
    const int lane = threadIdx.x & 31;
    const int warps = gridDim.x * norm_warp_rows_per_block;
    const float inv_D = 1.0f / static_cast<float>(D);

    for (int row = blockIdx.x * norm_warp_rows_per_block + (threadIdx.x >> 5); row < N; row += warps)
    {
        const Index base = Index(row) * D;
        float x[ITER][VEC];
        float local_sum = 0.0f, local_sum_sq = 0.0f;

        #pragma unroll
        for (int it = 0; it < ITER; ++it)
        {
            const int col = (it * 32 + lane) * VEC;
            VecIO<T, VEC>::load_float(X + base + col, x[it]);
            if constexpr (FuseResidual)
            {
                float r[VEC];
                VecIO<T, VEC>::load_float(R + base + col, r);
                #pragma unroll
                for (int k = 0; k < VEC; ++k) x[it][k] += r[k];
                VecIO<T, VEC>::store_float(sum + base + col, x[it]);
            }
            #pragma unroll
            for (int k = 0; k < VEC; ++k)
            {
                if constexpr (HasMean) local_sum += x[it][k];
                local_sum_sq += x[it][k] * x[it][k];
            }
        }

        if constexpr (HasMean) warp_reduce_sum2(local_sum, local_sum_sq);
        else                   local_sum_sq = warp_reduce_sum(local_sum_sq);

        float mean = 0.0f, inv_var;
        if constexpr (HasMean)
        {
            mean = local_sum * inv_D;
            inv_var = rsqrtf(fmaxf(local_sum_sq * inv_D - mean * mean, 0.0f) + eps);
            if (lane == 0) { means[row] = mean; inv_vars[row] = inv_var; }
        }
        else
        {
            inv_var = rsqrtf(local_sum_sq * inv_D + eps);
            if (lane == 0 && inv_vars) inv_vars[row] = inv_var;
        }

        #pragma unroll
        for (int it = 0; it < ITER; ++it)
        {
            const int col = (it * 32 + lane) * VEC;
            float g[VEC], b[VEC], y[VEC];
            VecIO<float, 4>::load(gamma + col, g);
            if constexpr (VEC == 8) VecIO<float, 4>::load(gamma + col + 4, g + 4);
            if constexpr (HasMean)
            {
                VecIO<float, 4>::load(beta + col, b);
                if constexpr (VEC == 8) VecIO<float, 4>::load(beta + col + 4, b + 4);
            }
            #pragma unroll
            for (int k = 0; k < VEC; ++k)
            {
                if constexpr (HasMean) y[k] = fmaf(g[k], (x[it][k] - mean) * inv_var, b[k]);
                else                   y[k] = g[k] * x[it][k] * inv_var;
            }
            VecIO<T, VEC>::store_float(Y + base + col, y);
        }
    }
}

template<typename T, bool HasMean, int ITER>
__global__ void __launch_bounds__(256)
norm_backward_warp_kernel(const int N, const int D, const T* __restrict__ dY, const T* __restrict__ X,
                          const float* __restrict__ means, const float* __restrict__ inv_vars,
                          const float* __restrict__ gamma, T* __restrict__ dX,
                          T* __restrict__ dX2, float* __restrict__ partials)
{
    constexpr int VEC = vec16<T>;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = gridDim.x * norm_warp_rows_per_block;
    const float inv_D = 1.0f / static_cast<float>(D);

    float g[ITER][VEC];
    float pg[ITER][VEC], pb[ITER][VEC];
    #pragma unroll
    for (int it = 0; it < ITER; ++it)
    {
        const int col = (it * 32 + lane) * VEC;
        VecIO<float, 4>::load(gamma + col, g[it]);
        if constexpr (VEC == 8) VecIO<float, 4>::load(gamma + col + 4, g[it] + 4);
        #pragma unroll
        for (int k = 0; k < VEC; ++k) { pg[it][k] = 0.0f; pb[it][k] = 0.0f; }
    }

    for (int row = blockIdx.x * norm_warp_rows_per_block + warp; row < N; row += warps)
    {
        const Index base = Index(row) * D;
        const float mean = HasMean ? means[row] : 0.0f;
        const float inv_var = inv_vars[row];

        float dy[ITER][VEC], xhat[ITER][VEC];
        float local_sum_d = 0.0f, local_sum_dx = 0.0f;
        #pragma unroll
        for (int it = 0; it < ITER; ++it)
        {
            const int col = (it * 32 + lane) * VEC;
            VecIO<T, VEC>::load_float(dY + base + col, dy[it]);
            VecIO<T, VEC>::load_float(X + base + col, xhat[it]);
            #pragma unroll
            for (int k = 0; k < VEC; ++k)
            {
                xhat[it][k] = (xhat[it][k] - mean) * inv_var;
                const float d = dy[it][k] * g[it][k];
                if constexpr (HasMean) { local_sum_d += d; pb[it][k] += dy[it][k]; }
                local_sum_dx += d * xhat[it][k];
                pg[it][k] += dy[it][k] * xhat[it][k];
            }
        }

        if constexpr (HasMean) warp_reduce_sum2(local_sum_d, local_sum_dx);
        else                   local_sum_dx = warp_reduce_sum(local_sum_dx);
        const float mean_d = local_sum_d * inv_D;
        const float mean_dx = local_sum_dx * inv_D;

        if (dX)
        {
            #pragma unroll
            for (int it = 0; it < ITER; ++it)
            {
                const int col = (it * 32 + lane) * VEC;
                float dx[VEC];
                #pragma unroll
                for (int k = 0; k < VEC; ++k)
                    dx[k] = (dy[it][k] * g[it][k] - mean_d - xhat[it][k] * mean_dx) * inv_var;
                VecIO<T, VEC>::store_float(dX + base + col, dx);
                if (dX2) VecIO<T, VEC>::store_float(dX2 + base + col, dx);
            }
        }
    }

    __shared__ float block_gamma[32 * 8 * 4];
    __shared__ float block_beta [32 * 8 * 4];
    for (int i = threadIdx.x; i < D; i += blockDim.x) { block_gamma[i] = 0.0f; block_beta[i] = 0.0f; }
    __syncthreads();
    for (int w = 0; w < norm_warp_rows_per_block; ++w)
    {
        if (w == warp)
        {
            #pragma unroll
            for (int it = 0; it < ITER; ++it)
            {
                const int col = (it * 32 + lane) * VEC;
                #pragma unroll
                for (int k = 0; k < VEC; ++k)
                {
                    block_gamma[col + k] += pg[it][k];
                    if constexpr (HasMean) block_beta[col + k] += pb[it][k];
                }
            }
        }
        __syncthreads();
    }
    float* out = partials + Index(blockIdx.x) * 2 * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        out[i] = block_gamma[i];
        if constexpr (HasMean) out[D + i] = block_beta[i];
    }
}

template<bool HasMean>
__global__ void norm_weight_gradient_finalize_kernel(const int blocks, const int D,
                                                     const float* __restrict__ partials,
                                                     float* __restrict__ dGamma, float* __restrict__ dBeta)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float g = 0.0f, b = 0.0f;
    for (int i = 0; i < blocks; ++i)
    {
        g += partials[Index(i) * 2 * D + d];
        if constexpr (HasMean) b += partials[Index(i) * 2 * D + D + d];
    }
    dGamma[d] = g;
    if constexpr (HasMean) dBeta[d] = b;
}

static inline int norm_warp_blocks(const int N)
{
    const int needed = ceil_div(N, norm_warp_rows_per_block);
    return needed < 240 ? needed : 240;
}

template<typename T, bool FuseResidual, bool HasMean, int ITER>
static bool norm_forward_warp_try(const int N, const int D, const T* X, const T* R, T* sum, T* Y,
                                  float* means, float* inv_vars, const float* gamma, const float* beta,
                                  const float eps)
{
    if (D != 32 * vec16<T> * ITER) return false;
    OPENNN_CUDA_LAUNCH((norm_forward_warp_kernel<T, FuseResidual, HasMean, ITER>
        <<<ceil_div(N, norm_warp_rows_per_block), 256, 0, opennn::device::get_compute_stream()>>>(
            N, D, X, R, sum, Y, means, inv_vars, gamma, beta, eps)));
    return true;
}

template<typename T, bool HasMean, int ITER>
static bool norm_backward_warp_try(const int N, const int D, const T* dY, const T* X, const float* means,
                                   const float* inv_vars, const float* gamma, T* dX, T* dX2, float* dGamma, float* dBeta)
{
    if (D != 32 * vec16<T> * ITER) return false;
    const int blocks = norm_warp_blocks(N);
    float* partials = opennn::ensure_workspace<float>(opennn::device::GraphWorkspaceKind::NormPartials,
                                                      Index(blocks) * 2 * D);
    const cudaStream_t stream = opennn::device::get_compute_stream();
    OPENNN_CUDA_LAUNCH((norm_backward_warp_kernel<T, HasMean, ITER><<<blocks, 256, 0, stream>>>(
        N, D, dY, X, means, inv_vars, gamma, dX, dX2, partials)));
    OPENNN_CUDA_LAUNCH((norm_weight_gradient_finalize_kernel<HasMean><<<ceil_div(D, 256), 256, 0, stream>>>(
        blocks, D, partials, dGamma, dBeta)));
    return true;
}

template<typename T, bool FuseResidual, bool HasMean>
static void norm_forward_launch(const int N, const int D, const T* X, const T* R, T* sum, T* Y,
                                float* means, float* inv_vars, const float* gamma, const float* beta, const float eps)
{
    if (N == 0 || D == 0) return;

    if (are_aligned<16>(X, R, sum, Y, gamma, beta)
        && (norm_forward_warp_try<T, FuseResidual, HasMean, 1>(N, D, X, R, sum, Y, means, inv_vars, gamma, beta, eps)
         || norm_forward_warp_try<T, FuseResidual, HasMean, 2>(N, D, X, R, sum, Y, means, inv_vars, gamma, beta, eps)
         || norm_forward_warp_try<T, FuseResidual, HasMean, 3>(N, D, X, R, sum, Y, means, inv_vars, gamma, beta, eps)
         || norm_forward_warp_try<T, FuseResidual, HasMean, 4>(N, D, X, R, sum, Y, means, inv_vars, gamma, beta, eps)))
        return;

    OPENNN_CUDA_LAUNCH((norm_forward_kernel<T, FuseResidual, HasMean><<<N, threads_for_width(D), 0, opennn::device::get_compute_stream()>>>(
        N, D, X, R, sum, Y, means, inv_vars, gamma, beta, eps)));
}

template<typename T>
void layernorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps)
{
    norm_forward_launch<T, false, true>(N, D, X, nullptr, nullptr, Y, means, inv_vars, gamma, beta, eps);
}

template<typename T>
void layernorm_add_forward_cuda(const int N, const int D, const T* X, const T* R, T* sum, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps)
{
    norm_forward_launch<T, true, true>(N, D, X, R, sum, Y, means, inv_vars, gamma, beta, eps);
}

template<typename T, bool HasMean>
__global__ void norm_backward_kernel(const int N, const int D, const T* __restrict__ dY, const T* __restrict__ X, const float* __restrict__ means, const float* __restrict__ inv_vars, const float* __restrict__ gamma, T* __restrict__ dX, T* __restrict__ dX2)
{
    const int idx = blockIdx.x;

    const Index row_base = Index(idx) * Index(D);

    const T* dy_row = dY + row_base;
    const T* x_row = X + row_base;
    T* dx_row = dX + row_base;
    T* dx2_row = dX2 ? dX2 + row_base : nullptr;

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
        T dx;
        if constexpr (HasMean)
            dx = static_cast<T>((d - mean_D - x_hat * mean_D_xhat) * inv_var);
        else
            dx = static_cast<T>((d - x_hat * mean_D_xhat) * inv_var);
        dx_row[i] = dx;
        if (dx2_row) dx2_row[i] = dx;
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
            const Index element = Index(n) * Index(D) + Index(d);
            const float dy    = static_cast<float>(dY[element]);
            float x_hat;
            if constexpr (HasMean) x_hat = (static_cast<float>(X[element]) - means[n]) * inv_vars[n];
            else                   x_hat = static_cast<float>(X[element]) * inv_vars[n];
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
static void norm_backward_launch(const int N, const int D, const T* dY, const T* X, const float* means, const float* inv_vars, const float* gamma, T* dX, T* dX2, float* dGamma, float* dBeta)
{
    if (are_aligned<16>(dY, X, dX, dX2, gamma)
        && (norm_backward_warp_try<T, HasMean, 1>(N, D, dY, X, means, inv_vars, gamma, dX, dX2, dGamma, dBeta)
         || norm_backward_warp_try<T, HasMean, 2>(N, D, dY, X, means, inv_vars, gamma, dX, dX2, dGamma, dBeta)
         || norm_backward_warp_try<T, HasMean, 3>(N, D, dY, X, means, inv_vars, gamma, dX, dX2, dGamma, dBeta)
         || norm_backward_warp_try<T, HasMean, 4>(N, D, dY, X, means, inv_vars, gamma, dX, dX2, dGamma, dBeta)))
        return;

    if (dX)
        OPENNN_CUDA_LAUNCH((norm_backward_kernel<T, HasMean><<<N, threads_for_width(D), 0, opennn::device::get_compute_stream()>>>(N, D, dY, X, means, inv_vars, gamma, dX, dX2)));

    constexpr int NUM_WARPS = 8;
    const dim3 block(32, NUM_WARPS);
    const int grid_x = ceil_div(D, 32);

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
    OPENNN_CUDA_LAUNCH((norm_weight_gradient_coalesced_kernel<T, NUM_WARPS, HasMean><<<dim3(grid_x, grid_y), block, 0,
        opennn::device::get_compute_stream()>>>(N, D, chunk, dY, X, means, inv_vars, dGamma, dBeta)));
}

template<typename T>
void layernorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* means, const float* inv_vars, const float* gamma, T* dX, T* dX2, float* dGamma, float* dBeta)
{
    if (N == 0 || D == 0) return;

    norm_backward_launch<T, true>(N, D, dY, X, means, inv_vars, gamma, dX, dX2, dGamma, dBeta);
}

template<typename T>
void rmsnorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* inv_rms, const float* weight, const float eps)
{
    norm_forward_launch<T, false, false>(N, D, X, nullptr, nullptr, Y, nullptr, inv_rms, weight, nullptr, eps);
}

template<typename T>
void rmsnorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* inv_rms, const float* weight, T* dX, float* dWeight)
{
    if (N == 0 || D == 0) return;

    norm_backward_launch<T, false>(N, D, dY, X, nullptr, inv_rms, weight, dX, nullptr, dWeight, nullptr);
}

#define INSTANTIATE(T) \
    template void batchnorm_inference_cuda<T>(const Index, const Index, const T*, const T*, const float*, const float*, const float*, const float*, const float, const bool, T*); \
    template void batchnorm_forward_fused_cuda<T>(const Index, const Index, const T*, const T*, const float*, const float*, const float, const float, T*, float*, float*, float*, float*, const bool, uint8_t*, float*); \
    template void batchnorm_backward_fused_cuda<T>(const Index, const Index, const T*, T*, const T*, const uint8_t*, const float*, const float*, const float*, const float*, const bool, T*, float*, float*, float*); \
    template void layernorm_forward_cuda<T>(const int, const int, const T*, T*, float*, float*, const float*, const float*, const float); \
    template void layernorm_add_forward_cuda<T>(const int, const int, const T*, const T*, T*, T*, float*, float*, const float*, const float*, const float); \
    template void layernorm_backward_cuda<T>(const int, const int, const T*, const T*, const float*, const float*, const float*, T*, T*, float*, float*); \
    template void rmsnorm_forward_cuda<T>(const int, const int, const T*, T*, float*, const float*, const float); \
    template void rmsnorm_backward_cuda<T>(const int, const int, const T*, const T*, const float*, const float*, T*, float*);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
