//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// recurrent and LSTM steps, time-slice gather/scatter

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/neural_network/layers/kernel_recurrent.cuh"

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

struct RnnCopyParams
{
    RnnCopySpec specs[RNN_COPY_MAX_REGIONS];
    int count;
};

template<typename T, bool Gather>
__global__ void time_slice_kernel(const int n,
                                  const int time_steps,
                                  const int features,
                                  const int t,
                                  const T* __restrict__ src,
                                  T* __restrict__ dst)
{
    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += Index(blockDim.x) * gridDim.x)
    {
        const int b = int(idx) / features;
        const int f = int(idx) - b * features;
        const Index strided = (Index(b) * time_steps + t) * features + f;
        if constexpr (Gather) dst[idx] = src[strided];
        else                  dst[strided] = src[idx];
    }
}

template<typename T>
void gather_time_slice_cuda(const Index batch,
                            const Index time_steps,
                            const Index features,
                            const Index t,
                            const T* src,
                            T* dst)
{
    launch_elementwise(batch * features, time_slice_kernel<T, true>,
                       checked_int(time_steps), checked_int(features), checked_int(t), src, dst);
}

template<typename T>
void scatter_time_slice_cuda(const Index batch,
                             const Index time_steps,
                             const Index features,
                             const Index t,
                             const T* src,
                             T* dst)
{
    launch_elementwise(batch * features, time_slice_kernel<T, false>,
                       checked_int(time_steps), checked_int(features), checked_int(t), src, dst);
}

__global__ void rnn_copy_regions_kernel(const RnnCopyParams params)
{
    const int region = blockIdx.y;
    if (region >= params.count) return;

    const RnnCopySpec spec = params.specs[region];
    const int total = spec.rows * spec.cols;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += gridDim.x * blockDim.x)
    {
        if (spec.transpose)
        {
            const int r = idx / spec.cols;
            const int c = idx - r * spec.cols;
            spec.dst[c * spec.rows + r] = spec.src[idx];
        }
        else
            spec.dst[idx] = spec.src[idx];
    }
}

void rnn_copy_regions_cuda(const RnnCopySpec* specs, int count,
                           cudaStream_t stream)
{
    if (count <= 0) return;
    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    RnnCopyParams params;
    int max_total = 0;
    for (int i = 0; i < count && i < RNN_COPY_MAX_REGIONS; ++i)
    {
        params.specs[i] = specs[i];
        max_total = max(max_total, specs[i].rows * specs[i].cols);
    }
    params.count = min(count, RNN_COPY_MAX_REGIONS);

    const dim3 grid(grid_size_for(max_total), params.count);
    OPENNN_CUDA_LAUNCH(rnn_copy_regions_kernel<<<grid, block_size, 0, stream>>>(params));
}

template<typename T>
__global__ void rnn_step_fused_forward_kernel(const int in_features,
                                              const int out_features,
                                              const T* __restrict__ step_input,
                                              const T* __restrict__ prev_hidden,
                                              const T* __restrict__ W_in,
                                              const T* __restrict__ W_rec,
                                              const T* __restrict__ bias,
                                              T* __restrict__ step_hidden,
                                              T* derivs,
                                              const int activation_id)
{
    extern __shared__ float smem[];
    float* sX = smem;
    float* sH = smem + in_features;

    const int b = blockIdx.x;
    const int j = threadIdx.x;

    for (int i = j; i < in_features; i += blockDim.x)
        sX[i] = static_cast<float>(step_input[b * in_features + i]);

    if (prev_hidden)
        for (int k = j; k < out_features; k += blockDim.x)
            sH[k] = static_cast<float>(prev_hidden[b * out_features + k]);

    __syncthreads();

    float z = static_cast<float>(bias[j]);

    for (int i = 0; i < in_features; ++i)
        z += sX[i] * static_cast<float>(W_in[i * out_features + j]);

    if (prev_hidden)
        for (int k = 0; k < out_features; ++k)
            z += sH[k] * static_cast<float>(W_rec[k * out_features + j]);

    float h_out;
    float dh_out;
    rnn_activation(activation_id, z, h_out, dh_out);

    step_hidden[b * out_features + j] = static_cast<T>(h_out);
    if (derivs) derivs[b * out_features + j] = static_cast<T>(dh_out);
}

template<typename T>
void rnn_step_fused_forward_cuda(const Index batch,
                                 const Index in_features,
                                 const Index out_features,
                                 const T* step_input,
                                 const T* prev_hidden,
                                 const T* W_in,
                                 const T* W_rec,
                                 const T* bias,
                                 T* step_hidden,
                                 T* derivs_or_null,
                                 const int activation_id)
{
    if (batch == 0 || out_features == 0) return;

    const int threads    = checked_int(out_features);
    const int grid_size  = checked_int(batch);
    checked_host_condition(threads > 1024,
                           "rnn_step_fused_forward_cuda: out_features exceeds CUDA max threads per block.");
    const Index shmem_floats = in_features + (prev_hidden ? out_features : Index(0));
    const size_t shmem_bytes = static_cast<size_t>(shmem_floats) * sizeof(float);

    OPENNN_CUDA_LAUNCH(rnn_step_fused_forward_kernel<T><<<grid_size, threads, shmem_bytes,
                                       opennn::device::get_compute_stream()>>>(
        checked_int(in_features),
        checked_int(out_features),
        step_input, prev_hidden, W_in, W_rec, bias,
        step_hidden, derivs_or_null, activation_id));
}

template<typename T>
__global__ void rnn_step_fused_backward_pre_kernel(const int n,
                                                   const int out_features,
                                                   const int time_steps,
                                                   const int t,
                                                   const T* __restrict__ output_delta,
                                                   const T* __restrict__ next_carry,
                                                   const T* __restrict__ activation_derivatives,
                                                   T* __restrict__ delta)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int b = idx / out_features;
    const int j = idx - b * out_features;

    const float dh = output_delta
        ? static_cast<float>(output_delta[idx])
        : static_cast<float>(next_carry[idx]);

    const float sigma_prime = static_cast<float>(
        activation_derivatives[(b * time_steps + t) * out_features + j]);

    const float dz = dh * sigma_prime;

    delta[idx] = static_cast<T>(dz);
}

template<typename T>
void rnn_step_fused_backward_pre_cuda(const Index batch,
                                      const Index out_features,
                                      const Index time_steps,
                                      const Index t,
                                      const T* output_delta,
                                      const T* next_carry,
                                      const T* activation_derivatives,
                                      T* delta)
{
    if (batch == 0 || out_features == 0) return;

    launch_elementwise(batch * out_features, rnn_step_fused_backward_pre_kernel<T>,
                       checked_int(out_features), checked_int(time_steps), checked_int(t),
                       output_delta, next_carry, activation_derivatives, delta);
}

#define INSTANTIATE(T) \
    template void gather_time_slice_cuda<T>(const Index, const Index, const Index, const Index, const T*, T*); \
    template void scatter_time_slice_cuda<T>(const Index, const Index, const Index, const Index, const T*, T*); \
    template void rnn_step_fused_forward_cuda<T>(const Index, const Index, const Index, const T*, const T*, const T*, const T*, const T*, T*, T*, const int); \
    template void rnn_step_fused_backward_pre_cuda<T>(const Index, const Index, const Index, const Index, const T*, const T*, const T*, T*);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
