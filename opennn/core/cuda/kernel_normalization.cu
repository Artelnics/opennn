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

// ---- fused batch-norm training forward and backward (NHWC) -----------------
//
// One thread owns VEC adjacent channels and strides down a slice of rows, so a
// warp reads consecutive channels of one row: coalesced, vector loads along C,
// which is contiguous in NHWC. Reduce blocks hold BN_THREADS threads laid out
// as (channel-group lanes, row lanes); grid (channel-group blocks, row blocks).
// Row-block partials are summed by a second, tiny kernel so the reduction order
// is fixed - no float atomics.
//
// ReLU mask: the forward apply pass packs (y > 0) eight channels per byte,
// mask[(row * channels + c0) / 8] (one lane's eight BF16 channels, or two
// adjacent lanes' four FP32 channels each). The backward gates dY with that
// byte instead of re-reading Y: one bit per element in place of a second
// full-width tensor in each of its two passes.
//
// x_hat: from X as (x - mean) * inv_var. Without a mask, in the reduce pass
// only, and only for a ReLU output that is BN(x) with no residual add
// (XHAT_FROM_Y), it is rebuilt as (y - beta) / gamma instead: where y == 0 the
// masked dY is 0 and x_hat drops out of both sums, so X need not be read there.
// The apply pass always reads X: dX = gamma * inv_var * (g - dbeta/M - x_hat *
// dgamma/M) is non-zero on masked elements too, and there y says nothing about
// x_hat.

constexpr int BN_THREADS = 256;
constexpr Index BN_MAX_ROW_BLOCKS = 128;

// Upper bound on the row blocks a launch uses for `rows`: what the caller's
// partials scratch must hold.
Index batchnorm_partial_rows(const Index rows)
{
    return rows < BN_MAX_ROW_BLOCKS ? (rows <= 0 ? 1 : rows) : BN_MAX_ROW_BLOCKS;
}

// As many channel-group lanes as the tensor has, up to a warp; the rest of the
// block strides down the rows, so narrow layers (64 channels) still fill it.
static dim3 batchnorm_reduce_block(const Index channel_groups)
{
    const int lanes = int(channel_groups < 32 ? channel_groups : 32);
    return dim3(unsigned(lanes), unsigned(BN_THREADS / lanes));
}

// Row blocks for a launch: enough to fill the GPU when there are rows to
// spare, but at least four rows per row lane, so a late ResNet stage at a small
// batch (128-512 rows) is not spread over 128 nearly idle blocks whose partials
// then have to be summed again.
static Index batchnorm_row_blocks(const Index rows, const dim3& reduce_block)
{
    const Index wanted = rows / (Index(reduce_block.y) * 4);
    const Index cap = batchnorm_partial_rows(rows);
    return wanted < 1 ? 1 : (wanted > cap ? cap : wanted);
}

// Sums the row-block partials of a channel: a warp of channels per block, eight
// lanes striding the row blocks, so a 128-block reduction is 16 loads per lane
// and a shared sum, not 128 dependent loads on one thread.
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

// Sums a reduce block's per-thread (s1, s2) down its row lanes and stores one
// (s1, s2) pair per channel for this row block.
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

// ---- forward -----------------------------------------------------------------

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

// Batch statistics from the row-block partials, the running-statistics update,
// and the per-channel scale/shift the apply pass uses. Population variance for
// the batch, as the CPU path; the running variance keeps the sample variance
// cuDNN's forward stores.
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

// Per element: y = relu?(x * scale + shift [+ residual]), and the ReLU bit.
template<int VEC, bool RELU, bool ADD>
__device__ static inline unsigned batchnorm_apply_group(const int channels, const int c0,
                                                        const float* vx, const float* vr,
                                                        const float* __restrict__ scale_shift,
                                                        float* out)
{
    unsigned bits = 0;
    #pragma unroll
    for (int k = 0; k < VEC; ++k)
    {
        const int c = c0 + k;
        float v = vx[k] * scale_shift[c] + scale_shift[channels + c];
        if (ADD) v += vr[k];
        if (RELU)
        {
            v = fmaxf(v, 0.0f);
            bits |= (v > 0.0f ? 1u : 0u) << k;
        }
        out[k] = v;
    }
    return bits;
}

// The mask byte covers eight channels. VEC == 8 (BF16): a lane writes its own
// byte, plain grid-stride loop. VEC == 4 (FP32): two adjacent lanes hold the
// two nibbles - the group index is even for the low nibble, and consecutive
// groups are consecutive lanes of one warp - so the odd lane's bits are
// shuffled down and the even lane writes; every lane of the block runs the
// loop body for the shuffle's sake, lanes past the end just do no work.
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
    float vx[VEC], vr[VEC], out[VEC];

    if constexpr (VEC == 8)
    {
        for (Index gi = Index(blockIdx.x) * blockDim.x + threadIdx.x; gi < groups;
             gi += Index(gridDim.x) * blockDim.x)
        {
            const int c0 = int(gi % channel_groups) * VEC;
            const Index i = (gi / channel_groups) * channels + c0;
            VecIO<T, VEC>::load_float(x + i, vx);
            if (ADD) VecIO<T, VEC>::load_float(residual + i, vr);
            const unsigned bits = batchnorm_apply_group<VEC, RELU, ADD>(channels, c0, vx, vr, scale_shift, out);
            VecIO<T, VEC>::store_float(y + i, out);
            if (RELU && mask) mask[gi] = uint8_t(bits);
        }
    }
    else
    {
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
                VecIO<T, VEC>::load_float(x + i, vx);
                if (ADD) VecIO<T, VEC>::load_float(residual + i, vr);
                bits = batchnorm_apply_group<VEC, RELU, ADD>(channels, c0, vx, vr, scale_shift, out);
                VecIO<T, VEC>::store_float(y + i, out);
            }
            if (RELU && mask)
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
    if (channels % 8 != 0)
        throw std::runtime_error("batchnorm_forward_fused_cuda: channels must be a multiple of 8.");
    // 16-byte channel groups: eight BF16 or four FP32 channels per lane.
    constexpr int VEC = sizeof(T) == 2 ? 8 : 4;
    cudaStream_t stream = opennn::device::get_compute_stream();

    const Index channel_groups = channels / VEC;
    const dim3 reduce_block = batchnorm_reduce_block(channel_groups);
    const Index row_blocks = batchnorm_row_blocks(rows, reduce_block);
    const Index rows_per_block = (rows + row_blocks - 1) / row_blocks;
    const dim3 reduce_grid(unsigned((channel_groups + reduce_block.x - 1) / reduce_block.x),
                           unsigned(row_blocks));
    OPENNN_CUDA_LAUNCH((batchnorm_forward_reduce_kernel<T, VEC><<<reduce_grid, reduce_block, 0, stream>>>(
        rows, checked_int(channels), rows_per_block, x, partials)));

    float* scale_shift = partials + 2 * batchnorm_partial_rows(rows) * channels;
    const float unbias = rows > 1 ? float(rows) / float(rows - 1) : 1.0f;
    const dim3 finalize_block(32, BN_FINALIZE_LANES);
    OPENNN_CUDA_LAUNCH((batchnorm_forward_finalize_kernel<<<
        unsigned((channels + 31) / 32), finalize_block, 0, stream>>>(
        checked_int(channels), checked_int(row_blocks), 1.0f / static_cast<float>(rows), unbias,
        epsilon, momentum, partials, gamma, beta, mean, inv_var, running_mean, running_var, scale_shift)));

    const bool add = residual != nullptr;
    const Index groups = rows * channel_groups;
    if (relu && add)       launch_elementwise_strided(groups, batchnorm_forward_apply_kernel<T, VEC, true,  true >, checked_int(channels), x, residual, scale_shift, y, mask);
    else if (relu)         launch_elementwise_strided(groups, batchnorm_forward_apply_kernel<T, VEC, true,  false>, checked_int(channels), x, residual, scale_shift, y, mask);
    else if (add)          launch_elementwise_strided(groups, batchnorm_forward_apply_kernel<T, VEC, false, true >, checked_int(channels), x, residual, scale_shift, y, mask);
    else                   launch_elementwise_strided(groups, batchnorm_forward_apply_kernel<T, VEC, false, false>, checked_int(channels), x, residual, scale_shift, y, mask);
}

// ---- backward ----------------------------------------------------------------

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
            const unsigned bits = mask ? (unsigned(mask[i / 8]) >> (c0 & 4)) : 0xFFu;
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
        const unsigned bits = mask ? (unsigned(mask[i / 8]) >> (c0 & 4)) : 0xFFu;

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
    const Index channel_groups = channels / VEC;
    const dim3 reduce_block = batchnorm_reduce_block(channel_groups);
    const Index row_blocks = batchnorm_row_blocks(rows, reduce_block);
    const Index rows_per_block = (rows + row_blocks - 1) / row_blocks;
    const dim3 reduce_grid(unsigned((channel_groups + reduce_block.x - 1) / reduce_block.x),
                           unsigned(row_blocks));
    OPENNN_CUDA_LAUNCH((batchnorm_backward_reduce_kernel<T, VEC, XHAT_FROM_Y><<<reduce_grid, reduce_block, 0, stream>>>(
        rows, checked_int(channels), rows_per_block, x, dy_dx, y, mask, gamma, beta, mean, inv_var, partials)));

    const dim3 finalize_block(32, BN_FINALIZE_LANES);
    OPENNN_CUDA_LAUNCH((batchnorm_backward_finalize_kernel<<<
        unsigned((channels + 31) / 32), finalize_block, 0, stream>>>(
        checked_int(channels), checked_int(row_blocks), partials, dgamma, dbeta)));

    launch_elementwise_strided(rows * channel_groups, batchnorm_backward_apply_kernel<T, VEC>,
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

    // 16-byte channel groups where the count allows it: eight BF16 or four
    // FP32 channels per lane (eight FP32 put the reduce kernel at ~90
    // registers and measured slower). The mask packs eight channels per byte
    // and needs one of these two layouts (a VEC == 4 lane reads its nibble);
    // with it Y is not read at all.
    const bool wide = channels % 8 == 0;
    const bool vec2 = channels % 2 == 0;
    if (!wide) mask = nullptr;
    if (mask) y = nullptr;

    // x_hat from Y needs the ReLU output and its parameters.
    const bool from_y = xhat_from_y && y != nullptr && beta != nullptr;

    const auto launch = [&]<int VEC>()
    {
        if (from_y) batchnorm_backward_launch<T, VEC, true >(rows, channels, x, dy_dx, y, mask, gamma, beta, mean, inv_var, dpre, dgamma, dbeta, partials, stream);
        else        batchnorm_backward_launch<T, VEC, false>(rows, channels, x, dy_dx, y, mask, gamma, beta, mean, inv_var, dpre, dgamma, dbeta, partials, stream);
    };
    if (wide)      launch.template operator()<sizeof(T) == 2 ? 8 : 4>();
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
    template void batchnorm_forward_fused_cuda<T>(const Index, const Index, const T*, const T*, const float*, const float*, const float, const float, T*, float*, float*, float*, float*, const bool, uint8_t*, float*); \
    template void batchnorm_backward_fused_cuda<T>(const Index, const Index, const T*, T*, const T*, const uint8_t*, const float*, const float*, const float*, const float*, const bool, T*, float*, float*, float*); \
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
