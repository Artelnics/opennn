#ifndef FLASH_ATTENTION_CUH
#define FLASH_ATTENTION_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

// FlashAttention-2's kernels, behind an interface that names no FA2 type, so
// attention can call them from a plain translation unit. Built only when
// OpenNN_WITH_FLASH_ATTENTION is on (see cmake/flash_attention.cmake);
// everywhere else `applies` answers false and attention keeps cuDNN's graph.

namespace opennn::flash_attention
{

// One attention problem in the layout the caller already holds it in: the
// kernels read the strides, so heads interleaved (B, S, H, D) and heads
// separated (B, H, S, D) are both fine, with no repacking either way.
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

    // How many keys of each sample are real, on the device, or null when every
    // sample fills its slot. Only the key side is masked: a padded query row
    // still produces an output, and its delta arrives zero from a loss that
    // ignores it, which is what keeps it out of every other row's gradient.
    const int* source_lengths = nullptr;
};

// Whether this build can run the problem: FA2's own limits (bf16, head
// dimension 32, 64 or 128, one of the architectures the kernels were built
// for) plus the one this integration adds. A causal mask cannot be combined
// with a padded batch, because FA2 anchors the causal diagonal to the
// bottom-right corner when the key range is shorter than the query range,
// while OpenNN's tokens sit at the top-left; the two disagree on every row.
bool applies(const Problem&);

// The forward, writing the output and the log-sum-exp its backward re-reads.
void forward(const Problem&,
             const void* query, const void* key, const void* value,
             void* output, float* softmax_lse, cudaStream_t);

// The backward. `output` and `softmax_lse` are the forward's, unchanged; the
// two accumulators are scratch the kernels clear themselves.
void backward(const Problem&,
              const void* query, const void* key, const void* value,
              const void* output, const void* output_delta, const float* softmax_lse,
              void* query_delta, void* key_delta, void* value_delta,
              float* query_delta_accumulator, float* softmax_delta_sum, cudaStream_t);

// How many calls have reached the kernels. A rung that quietly stops applying
// leaves every result unchanged and every test passing, so what reads this is
// the parity test: it is how "both rungs agree" is told from "only one ran".
// It counts launches, not steps: a captured CUDA graph records one and replays
// it, so a training run that captured its steps counts far fewer than it ran.
Index call_count();

// The three float buffers above, in elements.
Index softmax_lse_elements(const Problem&);
Index softmax_delta_sum_elements(const Problem&);
Index query_delta_accumulator_elements(const Problem&);

}

#endif

#endif
