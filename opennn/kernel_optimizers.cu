#include "kernel_common.cuh"

__device__ __forceinline__ void adam_update_one(
    float& p,
    float& m,
    float& v,
    float g,
    float beta_1,
    float one_minus_beta_1,
    float beta_2,
    float one_minus_beta_2,
    float lr,
    float eps)
{
    m = fmaf(beta_1, m, one_minus_beta_1 * g);
    v = fmaf(beta_2, v, one_minus_beta_2 * g * g);
    p -= lr * m / (sqrtf(v) + eps);
}

__global__ void adam_update_kernel(
    const int n_vec,
    const int n,
    float* __restrict__ parameters,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ gradients,
    __nv_bfloat16* __restrict__ parameters_bf16,
    const float beta_1,
    const float one_minus_beta_1,
    const float beta_2,
    const float one_minus_beta_2,
    const float lr,
    const float eps)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float4* __restrict__ const       p4 = reinterpret_cast<float4*>(parameters);
    float4* __restrict__ const       m4 = reinterpret_cast<float4*>(m);
    float4* __restrict__ const       v4 = reinterpret_cast<float4*>(v);
    const float4* __restrict__ const g4 = reinterpret_cast<const float4*>(gradients);
    __nv_bfloat162* __restrict__ const bf2 = reinterpret_cast<__nv_bfloat162*>(parameters_bf16);

    for (int i = tid; i < n_vec; i += stride)
    {
        float4 P = p4[i];
        float4 M = m4[i];
        float4 V = v4[i];
        const float4 G = g4[i];

        adam_update_one(P.x, M.x, V.x, G.x, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, lr, eps);
        adam_update_one(P.y, M.y, V.y, G.y, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, lr, eps);
        adam_update_one(P.z, M.z, V.z, G.z, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, lr, eps);
        adam_update_one(P.w, M.w, V.w, G.w, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, lr, eps);

        p4[i] = P;
        m4[i] = M;
        v4[i] = V;

        if (bf2)
        {
            bf2[i * 2 + 0] = __floats2bfloat162_rn(P.x, P.y);
            bf2[i * 2 + 1] = __floats2bfloat162_rn(P.z, P.w);
        }
    }

    const int tail_start = n_vec * 4;
    for (int i = tail_start + tid; i < n; i += stride)
    {
        adam_update_one(parameters[i], m[i], v[i], gradients[i],
                        beta_1, one_minus_beta_1, beta_2, one_minus_beta_2,
                        lr, eps);

        if (parameters_bf16)
            parameters_bf16[i] = __float2bfloat16(parameters[i]);
    }
}

void adam_update_cuda(
    const Index n,
    float* parameters,
    float* m,
    float* v,
    const float* gradients,
    const float beta_1,
    const float beta_2,
    const float learning_rate,
    const float epsilon,
    const float bias_correction_1,
    const float bias_correction_2,
    __nv_bfloat16* parameters_bf16)
{
    if (n == 0) return;

    const int total = checked_int(n);
    const float sqrt_bias_correction_2 = sqrtf(bias_correction_2);

    const float effective_lr = learning_rate * sqrt_bias_correction_2 / bias_correction_1;
    const float effective_eps = epsilon * sqrt_bias_correction_2;

    const float one_minus_beta_1 = 1.0f - beta_1;
    const float one_minus_beta_2 = 1.0f - beta_2;

    const bool mirror_aligned = parameters_bf16 == nullptr
        || (reinterpret_cast<std::uintptr_t>(parameters_bf16) & 0x3) == 0;

    const bool aligned = are_float4_aligned(parameters, m, v, gradients) && mirror_aligned;

    const int n_vec = aligned ? (total / 4) : 0;
    const int grid_size = grid_size_for(vector_work_size(total, n_vec, 4));

    OPENNN_CUDA_LAUNCH(adam_update_kernel<<<grid_size, block_size, 0, opennn::device::get_compute_stream()>>>(
        n_vec,
        total,
        parameters,
        m,
        v,
        gradients,
        parameters_bf16,
        beta_1,
        one_minus_beta_1,
        beta_2,
        one_minus_beta_2,
        effective_lr,
        effective_eps));
}


__global__ void adam_prepare_kernel(int* __restrict__ step,
                                    float beta_1, float beta_2,
                                    float learning_rate, float epsilon,
                                    float* __restrict__ effective_lr,
                                    float* __restrict__ effective_eps)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const int t = (*step) + 1;
    *step = t;

    const float bias_correction_1 = 1.0f - powf(beta_1, float(t));
    const float bias_correction_2 = 1.0f - powf(beta_2, float(t));
    const float sqrt_bc2 = sqrtf(bias_correction_2);

    *effective_lr  = learning_rate * sqrt_bc2 / bias_correction_1;
    *effective_eps = epsilon * sqrt_bc2;
}

__global__ void adam_update_capturable_kernel(
    const int n_vec,
    const int n,
    float* __restrict__ parameters,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ gradients,
    __nv_bfloat16* __restrict__ parameters_bf16,
    const float beta_1,
    const float one_minus_beta_1,
    const float beta_2,
    const float one_minus_beta_2,
    const float* __restrict__ effective_lr,
    const float* __restrict__ effective_eps)
{
    const float lr  = *effective_lr;
    const float eps = *effective_eps;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float4* __restrict__ const       p4 = reinterpret_cast<float4*>(parameters);
    float4* __restrict__ const       m4 = reinterpret_cast<float4*>(m);
    float4* __restrict__ const       v4 = reinterpret_cast<float4*>(v);
    const float4* __restrict__ const g4 = reinterpret_cast<const float4*>(gradients);
    __nv_bfloat162* __restrict__ const bf2 = reinterpret_cast<__nv_bfloat162*>(parameters_bf16);

    for (int i = tid; i < n_vec; i += stride)
    {
        float4 P = p4[i];
        float4 M = m4[i];
        float4 V = v4[i];
        const float4 G = g4[i];

        adam_update_one(P.x, M.x, V.x, G.x, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, lr, eps);
        adam_update_one(P.y, M.y, V.y, G.y, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, lr, eps);
        adam_update_one(P.z, M.z, V.z, G.z, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, lr, eps);
        adam_update_one(P.w, M.w, V.w, G.w, beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, lr, eps);

        p4[i] = P;
        m4[i] = M;
        v4[i] = V;

        if (bf2)
        {
            bf2[i * 2 + 0] = __floats2bfloat162_rn(P.x, P.y);
            bf2[i * 2 + 1] = __floats2bfloat162_rn(P.z, P.w);
        }
    }

    const int tail_start = n_vec * 4;
    for (int i = tail_start + tid; i < n; i += stride)
    {
        adam_update_one(parameters[i], m[i], v[i], gradients[i],
                        beta_1, one_minus_beta_1, beta_2, one_minus_beta_2, lr, eps);
        if (parameters_bf16)
            parameters_bf16[i] = __float2bfloat16(parameters[i]);
    }
}

void adam_update_capturable_cuda(
    const Index n,
    float* parameters, float* m, float* v, const float* gradients,
    const float beta_1, const float beta_2,
    const float learning_rate, const float epsilon,
    int* step_device, float* effective_lr_device, float* effective_eps_device,
    __nv_bfloat16* parameters_bf16,
    cudaStream_t stream)
{
    if (n == 0) return;
    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    const int total = checked_int(n);
    const float one_minus_beta_1 = 1.0f - beta_1;
    const float one_minus_beta_2 = 1.0f - beta_2;

    const bool mirror_aligned = parameters_bf16 == nullptr
        || (reinterpret_cast<std::uintptr_t>(parameters_bf16) & 0x3) == 0;
    const bool aligned = are_float4_aligned(parameters, m, v, gradients) && mirror_aligned;
    const int n_vec = aligned ? (total / 4) : 0;
    const int grid_size = grid_size_for(vector_work_size(total, n_vec, 4));

    OPENNN_CUDA_LAUNCH(adam_prepare_kernel<<<1, 1, 0, stream>>>(
        step_device, beta_1, beta_2, learning_rate, epsilon,
        effective_lr_device, effective_eps_device));

    OPENNN_CUDA_LAUNCH(adam_update_capturable_kernel<<<grid_size, block_size, 0, stream>>>(
        n_vec, total, parameters, m, v, gradients, parameters_bf16,
        beta_1, one_minus_beta_1, beta_2, one_minus_beta_2,
        effective_lr_device, effective_eps_device));
}

__device__ __forceinline__ void sgd_update_one(
    float& p,
    float& v,
    float g,
    float lr,
    float momentum,
    bool nesterov)
{
    const float lr_g = lr * g;
    if (momentum <= 0.0f) { p -= lr_g; return; }

    const float v_new = fmaf(momentum, v, -lr_g);
    v = v_new;
    p += nesterov ? fmaf(momentum, v_new, -lr_g) : v_new;
}

__global__ void sgd_update_kernel(
    const int n_vec,
    const int n,
    float* __restrict__ parameters,
    float* __restrict__ velocity,
    const float* __restrict__ gradients,
    __nv_bfloat16* __restrict__ parameters_bf16,
    const float learning_rate,
    const float momentum,
    const bool nesterov)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const bool has_momentum = momentum > 0.0f;

    float4* __restrict__ const       p4 = reinterpret_cast<float4*>(parameters);
    float4* __restrict__ const       v4 = reinterpret_cast<float4*>(velocity);
    const float4* __restrict__ const g4 = reinterpret_cast<const float4*>(gradients);
    __nv_bfloat162* __restrict__ const bf2 = reinterpret_cast<__nv_bfloat162*>(parameters_bf16);

    if (has_momentum)
    {
        for (int i = tid; i < n_vec; i += stride)
        {
            float4 P = p4[i];
            float4 V = v4[i];
            const float4 G = g4[i];

            sgd_update_one(P.x, V.x, G.x, learning_rate, momentum, nesterov);
            sgd_update_one(P.y, V.y, G.y, learning_rate, momentum, nesterov);
            sgd_update_one(P.z, V.z, G.z, learning_rate, momentum, nesterov);
            sgd_update_one(P.w, V.w, G.w, learning_rate, momentum, nesterov);

            p4[i] = P;
            v4[i] = V;

            if (bf2)
            {
                bf2[i * 2 + 0] = __floats2bfloat162_rn(P.x, P.y);
                bf2[i * 2 + 1] = __floats2bfloat162_rn(P.z, P.w);
            }
        }

        const int tail_start = n_vec * 4;
        for (int i = tail_start + tid; i < n; i += stride)
        {
            sgd_update_one(parameters[i], velocity[i], gradients[i],
                           learning_rate, momentum, nesterov);
            if (parameters_bf16)
                parameters_bf16[i] = __float2bfloat16(parameters[i]);
        }
    }
    else
    {
        for (int i = tid; i < n_vec; i += stride)
        {
            float4 P = p4[i];
            const float4 G = g4[i];

            P.x -= learning_rate * G.x;
            P.y -= learning_rate * G.y;
            P.z -= learning_rate * G.z;
            P.w -= learning_rate * G.w;

            p4[i] = P;

            if (bf2)
            {
                bf2[i * 2 + 0] = __floats2bfloat162_rn(P.x, P.y);
                bf2[i * 2 + 1] = __floats2bfloat162_rn(P.z, P.w);
            }
        }

        const int tail_start = n_vec * 4;
        for (int i = tail_start + tid; i < n; i += stride)
        {
            parameters[i] -= learning_rate * gradients[i];
            if (parameters_bf16)
                parameters_bf16[i] = __float2bfloat16(parameters[i]);
        }
    }
}

void sgd_update_cuda(
    const Index n,
    float* parameters,
    float* velocity,
    const float* gradients,
    const float learning_rate,
    const float momentum,
    const bool nesterov,
    __nv_bfloat16* parameters_bf16)
{
    if (n == 0 || learning_rate == 0.0f) return;

    const int total = checked_int(n);

    const bool mirror_aligned = parameters_bf16 == nullptr
        || (reinterpret_cast<std::uintptr_t>(parameters_bf16) & 0x3) == 0;

    const bool velocity_aligned = velocity == nullptr || is_float4_aligned(velocity);
    const bool aligned = are_float4_aligned(parameters, gradients) && velocity_aligned && mirror_aligned;

    const int n_vec = aligned ? (total / 4) : 0;
    const int grid_size = grid_size_for(vector_work_size(total, n_vec, 4));

    OPENNN_CUDA_LAUNCH(sgd_update_kernel<<<grid_size, block_size, 0, opennn::device::get_compute_stream()>>>(
        n_vec,
        total,
        parameters,
        velocity,
        gradients,
        parameters_bf16,
        learning_rate, momentum, nesterov));
}

__global__ void sgd_update_capturable_kernel(
    const int n_vec,
    const int n,
    float* __restrict__ parameters,
    float* __restrict__ velocity,
    const float* __restrict__ gradients,
    __nv_bfloat16* __restrict__ parameters_bf16,
    const float* __restrict__ learning_rate,
    const float momentum,
    const bool nesterov)
{
    const float lr = *learning_rate;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const bool has_momentum = momentum > 0.0f;

    float4* __restrict__ const       p4 = reinterpret_cast<float4*>(parameters);
    float4* __restrict__ const       v4 = reinterpret_cast<float4*>(velocity);
    const float4* __restrict__ const g4 = reinterpret_cast<const float4*>(gradients);
    __nv_bfloat162* __restrict__ const bf2 = reinterpret_cast<__nv_bfloat162*>(parameters_bf16);

    if (has_momentum)
    {
        for (int i = tid; i < n_vec; i += stride)
        {
            float4 P = p4[i];
            float4 V = v4[i];
            const float4 G = g4[i];

            sgd_update_one(P.x, V.x, G.x, lr, momentum, nesterov);
            sgd_update_one(P.y, V.y, G.y, lr, momentum, nesterov);
            sgd_update_one(P.z, V.z, G.z, lr, momentum, nesterov);
            sgd_update_one(P.w, V.w, G.w, lr, momentum, nesterov);

            p4[i] = P;
            v4[i] = V;

            if (bf2)
            {
                bf2[i * 2 + 0] = __floats2bfloat162_rn(P.x, P.y);
                bf2[i * 2 + 1] = __floats2bfloat162_rn(P.z, P.w);
            }
        }

        const int tail_start = n_vec * 4;
        for (int i = tail_start + tid; i < n; i += stride)
        {
            sgd_update_one(parameters[i], velocity[i], gradients[i],
                           lr, momentum, nesterov);
            if (parameters_bf16)
                parameters_bf16[i] = __float2bfloat16(parameters[i]);
        }
    }
    else
    {
        for (int i = tid; i < n_vec; i += stride)
        {
            float4 P = p4[i];
            const float4 G = g4[i];

            P.x -= lr * G.x;
            P.y -= lr * G.y;
            P.z -= lr * G.z;
            P.w -= lr * G.w;

            p4[i] = P;

            if (bf2)
            {
                bf2[i * 2 + 0] = __floats2bfloat162_rn(P.x, P.y);
                bf2[i * 2 + 1] = __floats2bfloat162_rn(P.z, P.w);
            }
        }

        const int tail_start = n_vec * 4;
        for (int i = tail_start + tid; i < n; i += stride)
        {
            parameters[i] -= lr * gradients[i];
            if (parameters_bf16)
                parameters_bf16[i] = __float2bfloat16(parameters[i]);
        }
    }
}

__global__ void set_scalar_kernel(float* __restrict__ dst, const float value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *dst = value;
}

void set_scalar_device_cuda(float* dst, const float value, cudaStream_t stream)
{
    if (stream == nullptr) stream = opennn::device::get_compute_stream();
    OPENNN_CUDA_LAUNCH(set_scalar_kernel<<<1, 1, 0, stream>>>(dst, value));
}

void sgd_update_capturable_cuda(
    const Index n,
    float* parameters,
    float* velocity,
    const float* gradients,
    const float* learning_rate_device,
    const float momentum,
    const bool nesterov,
    __nv_bfloat16* parameters_bf16,
    cudaStream_t stream)
{
    if (n == 0) return;
    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    const int total = checked_int(n);

    const bool mirror_aligned = parameters_bf16 == nullptr
        || (reinterpret_cast<std::uintptr_t>(parameters_bf16) & 0x3) == 0;

    const bool velocity_aligned = velocity == nullptr || is_float4_aligned(velocity);
    const bool aligned = are_float4_aligned(parameters, gradients) && velocity_aligned && mirror_aligned;

    const int n_vec = aligned ? (total / 4) : 0;
    const int grid_size = grid_size_for(vector_work_size(total, n_vec, 4));

    OPENNN_CUDA_LAUNCH(sgd_update_capturable_kernel<<<grid_size, block_size, 0, stream>>>(
        n_vec,
        total,
        parameters,
        velocity,
        gradients,
        parameters_bf16,
        learning_rate_device, momentum, nesterov));
}

__global__ void clip_apply_kernel(const int n,
                                  const float* __restrict__ squared_norm,
                                  const float max_norm,
                                  const float eps,
                                  float* __restrict__ gradient)
{
    const float norm = sqrtf(*squared_norm);
    if (norm <= max_norm) return;
    const float scale = max_norm / (norm + eps);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride)
        gradient[i] *= scale;
}

void clip_gradient_norm_cuda(const Index n,
                             float* gradient,
                             const float* squared_norm,
                             const float max_norm,
                             const float eps)
{
    if (n == 0) return;
    const int total = checked_int(n);
    const int grid = grid_size_for(total);
    OPENNN_CUDA_LAUNCH(clip_apply_kernel<<<grid, block_size, 0, opennn::device::get_compute_stream()>>>(
        total, squared_norm, max_norm, eps, gradient));
}

__global__ void cast_fp32_to_bf16_kernel(const int n_vec,
                                         const int n,
                                         const float* __restrict__ src,
                                         __nv_bfloat16* __restrict__ dst)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const float4* __restrict__ const src4 = reinterpret_cast<const float4*>(src);
    __nv_bfloat162* __restrict__ const dst2 = reinterpret_cast<__nv_bfloat162*>(dst);

    for (int i = tid; i < n_vec; i += stride)
    {
        const float4 in = src4[i];
        dst2[i * 2 + 0] = __floats2bfloat162_rn(in.x, in.y);
        dst2[i * 2 + 1] = __floats2bfloat162_rn(in.z, in.w);
    }

    const int tail_start = n_vec * 4;
    for (int i = tail_start + tid; i < n; i += stride)
        dst[i] = __float2bfloat16(src[i]);
}

void cast_fp32_to_bf16_cuda(const Index n, const float* src, __nv_bfloat16* dst,
                            cudaStream_t stream)
{
    if (n == 0) return;
    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    const int total = checked_int(n);
    const bool dst_aligned = (reinterpret_cast<std::uintptr_t>(dst) & 0x3) == 0;
    const bool aligned = are_float4_aligned(src) && dst_aligned;
    const int n_vec = aligned ? (total / 4) : 0;
    const int grid_size = grid_size_for(vector_work_size(total, n_vec, 4));

    OPENNN_CUDA_LAUNCH(cast_fp32_to_bf16_kernel<<<grid_size, block_size, 0, stream>>>(n_vec, total, src, dst));
}

template<typename TDst>
__device__ __forceinline__ TDst gather_cast(float value) { return value; }
template<>
__device__ __forceinline__ __nv_bfloat16 gather_cast<__nv_bfloat16>(float value) { return __float2bfloat16(value); }

template<typename TDst>
__global__ void gather_rows_kernel(const float* __restrict__ matrix,
                                   const int* __restrict__ row_indices,
                                   TDst* __restrict__ out,
                                   const int n_rows,
                                   const int n_cols,
                                   const int matrix_cols,
                                   const int col_offset)
{
    const int row = blockIdx.x;
    if (row >= n_rows) return;

    const float* __restrict__ src = matrix + size_t(row_indices[row]) * matrix_cols + col_offset;
    TDst* __restrict__ dst = out + size_t(row) * n_cols;

    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        dst[j] = gather_cast<TDst>(src[j]);
}

template<typename TDst>
static void gather_rows_launch(const float* matrix, const int* row_indices, TDst* out,
                               const Index n_rows, const Index n_cols,
                               const Index matrix_cols, const Index col_offset,
                               cudaStream_t stream)
{
    if (n_rows == 0 || n_cols == 0) return;
    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    const int rows = checked_int(n_rows);
    const int cols = checked_int(n_cols);
    const int threads = cols < block_size ? ((cols + 31) / 32) * 32 : block_size;

    OPENNN_CUDA_LAUNCH(gather_rows_kernel<TDst><<<rows, threads > 0 ? threads : 32, 0, stream>>>(
        matrix, row_indices, out, rows, cols, checked_int(matrix_cols), checked_int(col_offset)));
}

void gather_rows_cuda(const float* matrix, const int* row_indices, float* out,
                      const Index n_rows, const Index n_cols,
                      const Index matrix_cols, const Index col_offset,
                      cudaStream_t stream)
{
    gather_rows_launch<float>(matrix, row_indices, out, n_rows, n_cols, matrix_cols, col_offset, stream);
}

void gather_rows_bf16_cuda(const float* matrix, const int* row_indices, __nv_bfloat16* out,
                           const Index n_rows, const Index n_cols,
                           const Index matrix_cols, const Index col_offset,
                           cudaStream_t stream)
{
    gather_rows_launch<__nv_bfloat16>(matrix, row_indices, out, n_rows, n_cols, matrix_cols, col_offset, stream);
}

__global__ void cast_bf16_to_fp32_kernel(const int n,
                                         const __nv_bfloat16* __restrict__ src,
                                         float* __restrict__ dst)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride)
        dst[i] = __bfloat162float(src[i]);
}

void cast_bf16_to_fp32_cuda(const Index n, const __nv_bfloat16* src, float* dst)
{
    if (n == 0) return;
    const int total = checked_int(n);
    const int grid_size = grid_size_for(total);
    OPENNN_CUDA_LAUNCH(cast_bf16_to_fp32_kernel<<<grid_size, block_size, 0, opennn::device::get_compute_stream()>>>(total, src, dst));
}
