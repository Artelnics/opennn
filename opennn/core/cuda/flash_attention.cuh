#ifndef FLASH_ATTENTION_CUH
#define FLASH_ATTENTION_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

namespace opennn::flash_attention
{

struct Problem
{
    Index batch = 0;
    Index heads = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;
    Index head_dimension = 0;

    Index query_row_stride = 0,  query_head_stride = 0,  query_batch_stride = 0;
    Index source_row_stride = 0, source_head_stride = 0, source_batch_stride = 0;

    bool causal = false;
    float scale = 0.0f;

    const int* source_lengths = nullptr;
};

bool applies(const Problem&);

void forward(const Problem&,
             const void* query, const void* key, const void* value,
             void* output, float* softmax_lse, cudaStream_t);

void backward(const Problem&,
              const void* query, const void* key, const void* value,
              const void* output, const void* output_delta, const float* softmax_lse,
              void* query_delta, void* key_delta, void* value_delta,
              float* query_delta_accumulator, float* softmax_delta_sum, cudaStream_t);

Index call_count();

Index softmax_lse_elements(const Problem&);
Index softmax_delta_sum_elements(const Problem&);
Index query_delta_accumulator_elements(const Problem&);

}

#endif

#endif
