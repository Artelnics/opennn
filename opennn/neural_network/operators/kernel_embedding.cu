//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// embedding lookup forward and backward

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/neural_network/operators/kernel_embedding.cuh"

template<typename TW, typename T>
__global__ void embedding_forward_kernel(const int n, const float* __restrict__ inputs, const TW* __restrict__ weights, const float* __restrict__ positional_encoding, T* __restrict__ outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    const float scale = scale_embedding ? sqrtf(static_cast<float>(embedding_dimension)) : 1.0f;

    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const int token_index = i / embedding_dimension;
        const int dim_index = i % embedding_dimension;
        const int token_id = static_cast<int>(inputs[token_index]);

        float val = (token_id > 0 && token_id < vocabulary_size)
            ? scale * static_cast<float>(weights[token_id * embedding_dimension + dim_index])
            : 0.0f;

        if (positional_encoding != nullptr && token_id > 0)
        {
            const int seq_index = token_index % sequence_length;
            val += positional_encoding[seq_index * embedding_dimension + dim_index];
        }

        outputs[i] = static_cast<T>(val);
    }
}

template<typename TW, typename T>
void embedding_forward_cuda(const Index n, const float* inputs, const TW* weights, const float* positional_encoding, T* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    launch_elementwise_strided(n, embedding_forward_kernel<TW, T>, inputs, weights, positional_encoding, outputs,
                       sequence_length, embedding_dimension, vocabulary_size, scale_embedding);
}

template<typename T>
__global__ void embedding_backward_kernel(const int n, const float* __restrict__ inputs, const T* __restrict__ output_deltas, float* __restrict__ weight_gradients, float* __restrict__ positional_gradients, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    const float scale = scale_embedding ? sqrtf(static_cast<float>(embedding_dimension)) : 1.0f;

    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const int token_index = i / embedding_dimension;
        const int dim_index = i % embedding_dimension;
        const int token_id = static_cast<int>(inputs[token_index]);

        if (token_id <= 0 || token_id >= vocabulary_size) continue;

        const float delta = static_cast<float>(output_deltas[i]);
        atomicAdd(&weight_gradients[token_id * embedding_dimension + dim_index], scale * delta);

        if (positional_gradients != nullptr)
        {
            const int seq_index = token_index % sequence_length;
            atomicAdd(&positional_gradients[seq_index * embedding_dimension + dim_index], delta);
        }
    }
}

template<typename T>
void embedding_backward_cuda(const Index n, const float* inputs, const T* output_deltas, float* weight_gradients, float* positional_gradients, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    launch_elementwise(n, embedding_backward_kernel<T>, inputs, output_deltas, weight_gradients, positional_gradients,
                       sequence_length, embedding_dimension, vocabulary_size, scale_embedding);
}

// One warp per sample: how many of its tokens are not padding (id 0). Same
// count the CPU path makes, kept on the device so the record can feed the
// attention masks without a host round trip (and inside a CUDA graph).
__global__ void token_valid_lengths_kernel(const int batch_size, const int sequence_length,
                                           const float* __restrict__ token_ids, int* __restrict__ lengths)
{
    const int sample = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (sample >= batch_size) return;

    const float* row = token_ids + Index(sample) * sequence_length;
    int count = 0;
    for (int s = lane; s < sequence_length; s += 32)
        count += static_cast<int>(row[s]) != 0;

    for (int offset = 16; offset > 0; offset >>= 1)
        count += __shfl_xor_sync(0xffffffffu, count, offset);

    if (lane == 0) lengths[sample] = count;
}

void token_valid_lengths_cuda(const Index batch_size, const Index sequence_length,
                              const float* token_ids, int* lengths, cudaStream_t stream)
{
    if (batch_size <= 0) return;
    const int blocks = int((batch_size * 32 + block_size - 1) / block_size);
    OPENNN_CUDA_LAUNCH(token_valid_lengths_kernel<<<blocks, block_size, 0, stream>>>(
        int(batch_size), int(sequence_length), token_ids, lengths));
}

#define INSTANTIATE(T) \
    template void embedding_backward_cuda<T>(const Index, const float*, const T*, float*, float*, const int, const int, const int, const bool);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE

#define INSTANTIATE(TIn, TOut) \
    template void embedding_forward_cuda<TIn, TOut>(const Index, const float*, const TIn*, const float*, TOut*, const int, const int, const int, const bool);

OPENNN_INSTANTIATE_FLOAT_BF16_2(INSTANTIATE)
#undef INSTANTIATE


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
