//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A S H   A T T E N T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// The call into FlashAttention-2's kernels: fill its parameter struct from the
// tensors attention already holds, and pick the instantiation for the head
// dimension and the mask. Everything FA2 declares stays inside this file.

#include "opennn/core/cuda/flash_attention.cuh"

#ifdef OPENNN_HAS_CUDA

#ifdef OPENNN_HAS_FLASH_ATTENTION
#include "namespace_config.h"
#include "flash.h"
#include <cutlass/numeric_types.h>
#endif

#include <atomic>
#include <stdexcept>
#include <string>

namespace opennn::flash_attention
{

#ifdef OPENNN_HAS_FLASH_ATTENTION

static int round_multiple(Index value, int multiple)
{
    return int((value + multiple - 1) / multiple * multiple);
}

// The head dimensions this build has kernels for; the build sets one macro per
// entry of OpenNN_FLASH_ATTENTION_HEAD_DIMS, and the ones left out are not
// merely refused here, they are not linked in at all.
static bool head_dimension_built(Index head_dimension)
{
    switch (head_dimension)
    {
#ifdef OPENNN_FLASH_ATTENTION_HEAD_DIM_32
    case 32:  return true;
#endif
#ifdef OPENNN_FLASH_ATTENTION_HEAD_DIM_64
    case 64:  return true;
#endif
#ifdef OPENNN_FLASH_ATTENTION_HEAD_DIM_128
    case 128: return true;
#endif
    default:  return false;
    }
}

// Whether a kernel image on this device exists: a cubin built for a capability
// runs on any later minor revision of the same major one, so a build for sm_80
// covers sm_86 and sm_89, and nothing covers a different major.
static bool architecture_built()
{
    static const bool built = []
    {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) return false;

        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) return false;

        const auto covers = [&](int major, int minor)
        { return properties.major == major && properties.minor >= minor; };

        return false
#ifdef OPENNN_FLASH_ATTENTION_SM_80
            || covers(8, 0)
#endif
#ifdef OPENNN_FLASH_ATTENTION_SM_86
            || covers(8, 6)
#endif
#ifdef OPENNN_FLASH_ATTENTION_SM_89
            || covers(8, 9)
#endif
#ifdef OPENNN_FLASH_ATTENTION_SM_90
            || covers(9, 0)
#endif
            ;
    }();

    return built;
}

// Everything the two directions set the same way.
static void set_parameters(FLASH_NAMESPACE::Flash_fwd_params& parameters, const Problem& problem,
                           const void* query, const void* key, const void* value, void* output,
                           float* softmax_lse)
{
    parameters = {};
    parameters.is_bf16 = true;

    parameters.q_ptr = const_cast<void*>(query);
    parameters.k_ptr = const_cast<void*>(key);
    parameters.v_ptr = const_cast<void*>(value);
    parameters.o_ptr = output;

    parameters.q_row_stride   = problem.query_row_stride;
    parameters.q_head_stride  = problem.query_head_stride;
    parameters.q_batch_stride = problem.query_batch_stride;
    parameters.o_row_stride   = problem.query_row_stride;
    parameters.o_head_stride  = problem.query_head_stride;
    parameters.o_batch_stride = problem.query_batch_stride;

    parameters.k_row_stride   = problem.source_row_stride;
    parameters.k_head_stride  = problem.source_head_stride;
    parameters.k_batch_stride = problem.source_batch_stride;
    parameters.v_row_stride   = problem.source_row_stride;
    parameters.v_head_stride  = problem.source_head_stride;
    parameters.v_batch_stride = problem.source_batch_stride;

    parameters.softmax_lse_ptr = softmax_lse;

    parameters.b = int(problem.batch);
    parameters.h = int(problem.heads);
    parameters.h_k = int(problem.heads);
    parameters.h_h_k_ratio = 1;

    parameters.seqlen_q = int(problem.query_sequence_length);
    parameters.seqlen_k = int(problem.source_sequence_length);
    parameters.seqlen_q_rounded = round_multiple(problem.query_sequence_length, 128);
    parameters.seqlen_k_rounded = round_multiple(problem.source_sequence_length, 128);
    parameters.d = int(problem.head_dimension);
    parameters.d_rounded = round_multiple(problem.head_dimension, 32);

    parameters.scale_softmax = problem.scale;
    parameters.scale_softmax_log2 = problem.scale * 1.44269504088896340736f;

    // No dropout: the kernels are compiled without it, and the rung refuses a
    // layer that asks for it.
    parameters.p_dropout = 1.0f;
    parameters.p_dropout_in_uint8_t = 255;
    parameters.rp_dropout = 1.0f;
    parameters.scale_softmax_rp_dropout = problem.scale;

    parameters.is_causal = problem.causal;
    parameters.window_size_left  = problem.causal ? int(problem.source_sequence_length) : -1;
    parameters.window_size_right = problem.causal ? 0 : -1;
    parameters.softcap = 0.0f;

    parameters.unpadded_lse = false;
    parameters.seqlenq_ngroups_swapped = false;

    // A padded batch, without packing the tokens: the tensors keep one
    // sequence-length slot per sample, and cu_seqlens_k carries how many keys
    // of each slot are real rather than where each sample starts, which is what
    // is_seqlens_k_cumulative == false means. That combination is the only one
    // that leaves the offsets dense and still clamps the key range, and it also
    // takes the launch off the even-shape fast path, which is what makes the
    // last block of keys mask itself.
    if (problem.source_lengths)
    {
        parameters.cu_seqlens_k = const_cast<int*>(problem.source_lengths);
        parameters.is_seqlens_k_cumulative = false;
    }
}

template<int HeadDimension>
static void run_forward(FLASH_NAMESPACE::Flash_fwd_params& parameters, bool causal, cudaStream_t stream)
{
    if (causal) FLASH_NAMESPACE::run_mha_fwd_<cutlass::bfloat16_t, HeadDimension, true>(parameters, stream);
    else        FLASH_NAMESPACE::run_mha_fwd_<cutlass::bfloat16_t, HeadDimension, false>(parameters, stream);
}

template<int HeadDimension>
static void run_backward(FLASH_NAMESPACE::Flash_bwd_params& parameters, bool causal, cudaStream_t stream)
{
    if (causal) FLASH_NAMESPACE::run_mha_bwd_<cutlass::bfloat16_t, HeadDimension, true>(parameters, stream);
    else        FLASH_NAMESPACE::run_mha_bwd_<cutlass::bfloat16_t, HeadDimension, false>(parameters, stream);
}

// One case per head dimension the build has, and a throw for anything else:
// applies() answered false for it, so reaching here is a caller's mistake.
#define OPENNN_FLASH_ATTENTION_DISPATCH(run, head_dimension, ...)                        \
    switch (head_dimension)                                                              \
    {                                                                                    \
    OPENNN_FLASH_ATTENTION_CASE_32(run, __VA_ARGS__)                                     \
    OPENNN_FLASH_ATTENTION_CASE_64(run, __VA_ARGS__)                                     \
    OPENNN_FLASH_ATTENTION_CASE_128(run, __VA_ARGS__)                                    \
    default:                                                                             \
        throw std::runtime_error("FlashAttention: no kernel for head dimension "          \
                            + std::to_string(head_dimension) + " in this build.");       \
    }

#ifdef OPENNN_FLASH_ATTENTION_HEAD_DIM_32
#define OPENNN_FLASH_ATTENTION_CASE_32(run, ...)  case 32:  run<32>(__VA_ARGS__);  break;
#else
#define OPENNN_FLASH_ATTENTION_CASE_32(run, ...)
#endif
#ifdef OPENNN_FLASH_ATTENTION_HEAD_DIM_64
#define OPENNN_FLASH_ATTENTION_CASE_64(run, ...)  case 64:  run<64>(__VA_ARGS__);  break;
#else
#define OPENNN_FLASH_ATTENTION_CASE_64(run, ...)
#endif
#ifdef OPENNN_FLASH_ATTENTION_HEAD_DIM_128
#define OPENNN_FLASH_ATTENTION_CASE_128(run, ...) case 128: run<128>(__VA_ARGS__); break;
#else
#define OPENNN_FLASH_ATTENTION_CASE_128(run, ...)
#endif

bool applies(const Problem& problem)
{
    if (!head_dimension_built(problem.head_dimension)) return false;
    if (!architecture_built()) return false;

    // See the header: FA2 would anchor this to the wrong corner.
    if (problem.causal && problem.source_lengths) return false;

    return problem.batch > 0 && problem.heads > 0
        && problem.query_sequence_length > 0 && problem.source_sequence_length > 0;
}

static std::atomic<Index> calls{0};

Index call_count() { return calls.load(std::memory_order_relaxed); }

void forward(const Problem& problem,
             const void* query, const void* key, const void* value,
             void* output, float* softmax_lse, cudaStream_t stream)
{
    calls.fetch_add(1, std::memory_order_relaxed);

    FLASH_NAMESPACE::Flash_fwd_params parameters;
    set_parameters(parameters, problem, query, key, value, output, softmax_lse);

    OPENNN_FLASH_ATTENTION_DISPATCH(run_forward, problem.head_dimension,
                                    parameters, problem.causal, stream)
}

void backward(const Problem& problem,
              const void* query, const void* key, const void* value,
              const void* output, const void* output_delta, const float* softmax_lse,
              void* query_delta, void* key_delta, void* value_delta,
              float* query_delta_accumulator, float* softmax_delta_sum, cudaStream_t stream)
{
    calls.fetch_add(1, std::memory_order_relaxed);

    FLASH_NAMESPACE::Flash_bwd_params parameters;
    set_parameters(parameters, problem, query, key, value,
                   const_cast<void*>(output), const_cast<float*>(softmax_lse));

    parameters.do_ptr = const_cast<void*>(output_delta);
    parameters.do_row_stride   = problem.query_row_stride;
    parameters.do_head_stride  = problem.query_head_stride;
    parameters.do_batch_stride = problem.query_batch_stride;

    parameters.dq_ptr = query_delta;
    parameters.dq_row_stride   = problem.query_row_stride;
    parameters.dq_head_stride  = problem.query_head_stride;
    parameters.dq_batch_stride = problem.query_batch_stride;

    parameters.dk_ptr = key_delta;
    parameters.dv_ptr = value_delta;
    parameters.dk_row_stride   = problem.source_row_stride;
    parameters.dk_head_stride  = problem.source_head_stride;
    parameters.dk_batch_stride = problem.source_batch_stride;
    parameters.dv_row_stride   = problem.source_row_stride;
    parameters.dv_head_stride  = problem.source_head_stride;
    parameters.dv_batch_stride = problem.source_batch_stride;

    parameters.dq_accum_ptr = query_delta_accumulator;
    parameters.dk_accum_ptr = nullptr;
    parameters.dv_accum_ptr = nullptr;
    parameters.dsoftmax_sum = softmax_delta_sum;
    parameters.dq_accum_split_stride = 0;
    parameters.deterministic = false;

    OPENNN_FLASH_ATTENTION_DISPATCH(run_backward, problem.head_dimension,
                                    parameters, problem.causal, stream)
}

Index softmax_lse_elements(const Problem& problem)
{
    return problem.batch * problem.heads * problem.query_sequence_length;
}

Index softmax_delta_sum_elements(const Problem& problem)
{
    return problem.batch * problem.heads * round_multiple(problem.query_sequence_length, 128);
}

Index query_delta_accumulator_elements(const Problem& problem)
{
    return problem.batch * round_multiple(problem.query_sequence_length, 128)
         * problem.heads * round_multiple(problem.head_dimension, 32);
}

#else

bool applies(const Problem&) { return false; }

Index call_count() { return 0; }

[[noreturn]] static void unavailable()
{
    throw std::runtime_error("FlashAttention: this build has no FA2 kernels "
                        "(configure with OpenNN_WITH_FLASH_ATTENTION=ON).");
}

void forward(const Problem&, const void*, const void*, const void*, void*, float*, cudaStream_t)
{ unavailable(); }

void backward(const Problem&, const void*, const void*, const void*, const void*, const void*,
              const float*, void*, void*, void*, float*, float*, cudaStream_t)
{ unavailable(); }

Index softmax_lse_elements(const Problem&) { return 0; }
Index softmax_delta_sum_elements(const Problem&) { return 0; }
Index query_delta_accumulator_elements(const Problem&) { return 0; }

#endif

}

#endif
