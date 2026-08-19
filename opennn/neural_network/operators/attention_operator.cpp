//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A T T E N T I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/operators/attention_operator.h"
#include "opennn/core/profiler.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/operators/dropout_operator.h"
#include "opennn/neural_network/operators/multihead_projection_operator.h"

#ifdef OPENNN_HAS_CUDA
#include "opennn/core/cuda/cudnn_frontend_utilities.h"
#include "opennn/core/cuda/kernel_attention.cuh"
#include "opennn/core/cuda/kernel_cast.cuh"
#endif

namespace opennn
{

// The softmax scale; a zero head dimension (operator not set) gets 1/sqrt(16).
static float attention_scale(Index head_dim)
{
    return head_dim == 0 ? 0.25f : 1.0f / sqrt(float(head_dim));
}

float AttentionOperator::scaling_factor() const
{
    return attention_scale(head_dimension);
}

void AttentionOperator::set(Index new_heads_number, Index new_head_dimension,
                    Index new_query_sequence_length, Index new_source_sequence_length,
                    bool new_use_causal_mask, Type new_compute_dtype)
{
    heads_number = new_heads_number;
    head_dimension = new_head_dimension;
    query_sequence_length = new_query_sequence_length;
    source_sequence_length = new_source_sequence_length;
    use_causal_mask = new_use_causal_mask;
    compute_dtype = new_compute_dtype;

    if (use_causal_mask && query_sequence_length > 0 && source_sequence_length > 0)
        causal_mask = MatrixR::NullaryExpr(query_sequence_length, source_sequence_length,
            [](Index row, Index column) { return column > row ? NEG_INFINITY : 0.0f; });
    else
        causal_mask.resize(0, 0);
}


static bool row_nonzero(const float* row, Index dim)
{
    return Map<const Array<float, Dynamic, 1>>(row, dim).abs().maxCoeff() > EPSILON;
}

bool AttentionOperator::get_contiguous_source_lengths(const TensorView& source_input,
                                                vector<Index>& lengths,
                                                bool& has_padding)
{
    if (source_input.get_shape().get_rank() != 3 || !source_input.is_fp32())
        return false;

    const Index batch_size          = source_input.get_shape()[0];
    const Index sequence_length     = source_input.get_shape()[1];
    const Index embedding_dimension = source_input.get_shape()[2];
    const float* source_data        = source_input.as<float>();

    lengths.assign(batch_size, sequence_length);
    has_padding = false;

    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        const float* batch = source_data + batch_index * sequence_length * embedding_dimension;

        Index valid_length = 0;
        while (valid_length < sequence_length
               && row_nonzero(batch + valid_length * embedding_dimension, embedding_dimension))
            ++valid_length;

        if (valid_length == 0) return false;

        for (Index i = valid_length; i < sequence_length; ++i)
            if (row_nonzero(batch + i * embedding_dimension, embedding_dimension))
                return false;

        if (valid_length < sequence_length) has_padding = true;
        lengths[batch_index] = valid_length;
    }

    return true;
}

void AttentionOperator::softmax_rows_prefix(float* matrix, Index rows, Index cols, Index length)
{
    for (Index row = 0; row < rows; ++row)
    {
        Eigen::Map<Eigen::VectorXf> v(matrix + row * cols, length);
        v = (v.array() - v.maxCoeff()).exp();
        v /= v.sum();
    }
}

Index AttentionOperator::infer_attention_prefix_length(const TensorView& attention_weights,
                                                 Index batch_index)
{
    const auto& shape = attention_weights.get_shape();
    const float* first_row = attention_weights.as<float>()
        + batch_index * shape[1] * shape[2] * shape[3];

    Index length = shape[3];
    while (length > 0 && first_row[length - 1] == 0.0f)
        --length;

    return length;
}

vector<TensorSpec> AttentionOperator::forward_scratch_specs(Index batch_size) const
{
    // SDPA never touches these buffers: dropout runs inside the cuDNN graph
    // (seed/offset in apply_sdpa_forward), and padding is a mask on the graph
    // rather than a scratch matrix to write the mask into.
    if (use_sdpa)
        return vector<TensorSpec>(2, {Shape{}, compute_dtype});

    const Shape attention_shape = {batch_size, heads_number,
                                   query_sequence_length, source_sequence_length};
    const Shape dropout_shape = dropout.active() ? attention_shape : Shape{};

    return {
        {attention_shape, compute_dtype},
        {dropout_shape,   compute_dtype},
    };
}

TensorSpec AttentionOperator::backward_scratch_spec(Index batch_size) const
{
    if (use_sdpa)
        return {Shape{}, compute_dtype};

    return {{batch_size, heads_number, query_sequence_length, source_sequence_length},
            compute_dtype};
}

vector<TensorSpec> AttentionOperator::sdpa_gradient_scratch_specs(Index batch_size) const
{
    if (!use_sdpa || compute_dtype != Type::FP32)
        return vector<TensorSpec>(sdpa_scratch_slots_count, {Shape{}, Type::BF16});

    const Shape query_shape  = {batch_size, heads_number, query_sequence_length,  head_dimension};
    const Shape source_shape = {batch_size, heads_number, source_sequence_length, head_dimension};

    return {
        {query_shape,  Type::BF16},
        {query_shape,  Type::BF16},
        {source_shape, Type::BF16},
        {source_shape, Type::BF16},
        {query_shape,  Type::BF16},
        {source_shape, Type::BF16},
        {source_shape, Type::BF16},
        {query_shape,  Type::BF16},
    };
}

TensorSpec AttentionOperator::sdpa_qkv_pack_spec(Index batch_size) const
{
    if (!use_sdpa || compute_dtype != Type::FP32)
        return {Shape{}, Type::BF16};

    const SdpaBf16Pack pack{
        batch_size * heads_number * query_sequence_length  * head_dimension,
        batch_size * heads_number * source_sequence_length * head_dimension};

    return {{pack.total_elements()}, Type::BF16};
}

vector<TensorSpec> AttentionOperator::sdpa_state_specs(const Index batch_size) const
{
    if (!use_sdpa)
        return vector<TensorSpec>(sdpa_state_slots_count, {Shape{}, Type::FP32});

    return {
        {{batch_size, heads_number, query_sequence_length, 1}, Type::FP32},
        {{batch_size * Index(sizeof(int32_t))}, Type::INT8},
        {{batch_size * Index(sizeof(int32_t))}, Type::INT8},
        {{2 * Index(sizeof(int64_t))}, Type::INT8},
    };
}

bool AttentionOperator::sdpa_supported(Type dtype, Device device)
{
#ifdef OPENNN_HAS_CUDA
    return device == Device::CUDA && is_one_of(dtype, Type::BF16, Type::FP32);
#else
    (void)dtype; (void)device;
    return false;
#endif
}

#ifdef OPENNN_HAS_CUDA

struct AttentionOperator::SDPACache
{
    struct CacheKey
    {
        Index batch_size = 0;
        Index q_seq      = 0;
        Index src_seq    = 0;
        Index heads      = 0;
        Index head_dim   = 0;
        bool  dropout_active = false;
        float dropout_rate   = 0.0f;
        bool  causal         = false;
        bool  is_training    = false;
        bool  interleaved    = false;

        bool operator==(const CacheKey&) const = default;
    };

    struct CacheKeyHash
    {
        size_t operator()(const CacheKey& k) const
        {
            return hash_combine(k.batch_size, k.q_seq, k.src_seq, k.heads, k.head_dim,
                                Index(k.dropout_active), Index(bit_cast<uint32_t>(k.dropout_rate)),
                                Index(k.causal), Index(k.is_training),
                                Index(k.interleaved));
        }
    };

    struct Entry
    {
        shared_ptr<cudnn_frontend::graph::Graph> fwd_graph;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_Q, fwd_K, fwd_V, fwd_O, fwd_Stats;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_SeqLenQ, fwd_SeqLenKV;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_Seed, fwd_Offset;
        int64_t fwd_workspace_bytes = 0;
        bool fwd_autotune_pending = false;

        shared_ptr<cudnn_frontend::graph::Graph> bwd_graph;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_Q, bwd_K, bwd_V, bwd_O, bwd_dO, bwd_Stats;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_dQ, bwd_dK, bwd_dV;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_SeqLenQ, bwd_SeqLenKV;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_Seed, bwd_Offset;
        int64_t bwd_workspace_bytes = 0;
        bool bwd_autotune_pending = false;
    };

    unordered_map<CacheKey, Entry, CacheKeyHash> entries;

    Entry& get_or_create_entry(const CacheKey& key)
    {
        // Entries contain descriptors only. Tensor and workspace addresses
        // are owned by propagation state, so evicting a descriptor cannot
        // invalidate a captured CUDA graph.
        return detail::bounded_cache_entry(
            entries, key, cudnn_frontend::graph_cache_capacity);
    }
};

namespace
{

constexpr size_t sdpa_statistics_state = 0;
constexpr size_t sdpa_query_lengths_state = 1;
constexpr size_t sdpa_source_lengths_state = 2;
constexpr size_t sdpa_dropout_state = 3;

// A (B, H, S, D) tensor as the graph sees it, over memory that is either
// (B, H, S, D) or, interleaved, (B, S, H, D): the layout the projections'
// GEMMs write and the output projection reads.
shared_ptr<cudnn_frontend::graph::Tensor_attributes>
heads_input(cudnn_frontend::graph::Graph& graph, const char* name,
            int64_t B, int64_t H, int64_t S, int64_t D, bool interleaved)
{
    return cudnn_frontend::bhsd_tensor(graph, name, B, H, S, D, interleaved);
}

void heads_output(shared_ptr<cudnn_frontend::graph::Tensor_attributes>& T,
                  int64_t B, int64_t H, int64_t S, int64_t D, bool interleaved)
{
    cudnn_frontend::set_bhsd_output(T, B, H, S, D, interleaved);
}

void refresh_sdpa_sequence_lengths(const AttentionOperator::SDPACache::CacheKey& k,
                                   const TensorView& source_input,
                                   int32_t* query_lengths,
                                   int32_t* source_lengths)
{
    const Shape& shape = source_input.get_shape();
    const bool ok = shape.get_rank() == 3
        && shape[0] == k.batch_size
        && shape[1] == k.src_seq
        && source_input.is_cuda();

    throw_if(!ok,
             "SDPA padding mask: source_input must be a rank-3 CUDA tensor with supported dtype.");

    source_input.dispatch([&]<typename T>() {
        attention_sequence_lengths_cuda<T>(to_int(k.batch_size),
                                           to_int(k.q_seq),
                                           to_int(k.src_seq),
                                           to_int(shape[2]),
                                           source_input.as<T>(),
                                           query_lengths,
                                           source_lengths);
    });
}

}

static void build_sdpa_forward_graph(AttentionOperator::SDPACache::Entry& entry,
                                      const AttentionOperator::SDPACache::CacheKey& k)
{
    const auto graph = cudnn_frontend::new_graph(Type::BF16);

    entry.fwd_Q = heads_input(*graph, "Q", k.batch_size, k.heads, k.q_seq,   k.head_dim, k.interleaved);
    entry.fwd_K = heads_input(*graph, "K", k.batch_size, k.heads, k.src_seq, k.head_dim, k.interleaved);
    entry.fwd_V = heads_input(*graph, "V", k.batch_size, k.heads, k.src_seq, k.head_dim, k.interleaved);
    entry.fwd_SeqLenQ  = cudnn_frontend::seq_len_scalar(*graph, "SeqLenQ",  k.batch_size);
    entry.fwd_SeqLenKV = cudnn_frontend::seq_len_scalar(*graph, "SeqLenKV", k.batch_size);

    auto sdpa_options = cudnn_frontend::graph::SDPA_attributes()
                        .set_name("flash_attn_fwd")
                        .set_generate_stats(k.is_training)
                        .set_padding_mask(true)
                        .set_seq_len_q(entry.fwd_SeqLenQ)
                        .set_seq_len_kv(entry.fwd_SeqLenKV)
                        .set_causal_mask(k.causal)
                        .set_attn_scale(attention_scale(k.head_dim));

    if (k.dropout_active)
    {
        entry.fwd_Seed   = cudnn_frontend::scalar_tensor(*graph, "Seed", cudnn_frontend::DataType_t::INT64);
        entry.fwd_Offset = cudnn_frontend::scalar_tensor(*graph, "Offset", cudnn_frontend::DataType_t::INT64);
        sdpa_options.set_dropout(k.dropout_rate, entry.fwd_Seed, entry.fwd_Offset);
    }

    auto [O, Stats] = graph->sdpa(entry.fwd_Q, entry.fwd_K, entry.fwd_V, sdpa_options);

    heads_output(O, k.batch_size, k.heads, k.q_seq, k.head_dim, k.interleaved);
    entry.fwd_O = O;

    if (k.is_training && Stats)
    {
        Stats->set_output(true)
              .set_data_type(cudnn_frontend::DataType_t::FLOAT)
              .set_dim({k.batch_size, k.heads, k.q_seq, 1})
              .set_stride({k.heads * k.q_seq, k.q_seq, 1, 1});
        entry.fwd_Stats = Stats;
    }

    entry.fwd_autotune_pending = cudnn_frontend::finalize_attention(
        *graph, "sdpa fwd", entry.fwd_workspace_bytes, true);

    entry.fwd_graph = graph;
}

static void build_sdpa_backward_graph(AttentionOperator::SDPACache::Entry& entry,
                                       const AttentionOperator::SDPACache::CacheKey& k)
{
    const auto graph = cudnn_frontend::new_graph(Type::BF16);

    entry.bwd_Q  = heads_input(*graph, "Q_bwd",  k.batch_size, k.heads, k.q_seq,   k.head_dim, k.interleaved);
    entry.bwd_K  = heads_input(*graph, "K_bwd",  k.batch_size, k.heads, k.src_seq, k.head_dim, k.interleaved);
    entry.bwd_V  = heads_input(*graph, "V_bwd",  k.batch_size, k.heads, k.src_seq, k.head_dim, k.interleaved);
    entry.bwd_dO = heads_input(*graph, "dO_bwd", k.batch_size, k.heads, k.q_seq,   k.head_dim, k.interleaved);
    entry.bwd_SeqLenQ  = cudnn_frontend::seq_len_scalar(*graph, "SeqLenQ_bwd",  k.batch_size);
    entry.bwd_SeqLenKV = cudnn_frontend::seq_len_scalar(*graph, "SeqLenKV_bwd", k.batch_size);

    // O is read from the merged (B, S, H*D) attention output whichever layout
    // the head tensors use.
    entry.bwd_O = heads_input(*graph, "O_bwd", k.batch_size, k.heads, k.q_seq, k.head_dim, true);

    entry.bwd_Stats = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                    .set_name("Stats_bwd")
                                    .set_data_type(cudnn_frontend::DataType_t::FLOAT)
                                    .set_dim   ({k.batch_size, k.heads, k.q_seq, 1})
                                    .set_stride({k.heads * k.q_seq, k.q_seq, 1, 1}));

    auto sdpa_bwd_options = cudnn_frontend::graph::SDPA_backward_attributes()
                            .set_name("flash_attn_bwd")
                            .set_padding_mask(true)
                            .set_seq_len_q(entry.bwd_SeqLenQ)
                            .set_seq_len_kv(entry.bwd_SeqLenKV)
                            .set_causal_mask(k.causal)
                            .set_attn_scale(attention_scale(k.head_dim));

    if (k.dropout_active)
    {
        entry.bwd_Seed   = cudnn_frontend::scalar_tensor(*graph, "Seed_bwd", cudnn_frontend::DataType_t::INT64);
        entry.bwd_Offset = cudnn_frontend::scalar_tensor(*graph, "Offset_bwd", cudnn_frontend::DataType_t::INT64);
        sdpa_bwd_options.set_dropout(k.dropout_rate, entry.bwd_Seed, entry.bwd_Offset);
    }

    auto [dQ, dK, dV] = graph->sdpa_backward(entry.bwd_Q, entry.bwd_K, entry.bwd_V,
                                              entry.bwd_O, entry.bwd_dO, entry.bwd_Stats,
                                              sdpa_bwd_options);

    heads_output(dQ, k.batch_size, k.heads, k.q_seq,   k.head_dim, k.interleaved);
    heads_output(dK, k.batch_size, k.heads, k.src_seq, k.head_dim, k.interleaved);
    heads_output(dV, k.batch_size, k.heads, k.src_seq, k.head_dim, k.interleaved);

    entry.bwd_dQ = dQ;
    entry.bwd_dK = dK;
    entry.bwd_dV = dV;

    entry.bwd_autotune_pending = cudnn_frontend::finalize_attention(
        *graph, "sdpa bwd", entry.bwd_workspace_bytes, true);

    entry.bwd_graph = graph;
}

#else

struct AttentionOperator::SDPACache {};

#endif

AttentionOperator::AttentionOperator() = default;
AttentionOperator::~AttentionOperator() = default;
AttentionOperator::AttentionOperator(AttentionOperator&&) noexcept = default;
AttentionOperator& AttentionOperator::operator=(AttentionOperator&&) noexcept = default;

void AttentionOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training)
{
    PROFILE_SCOPE("op:attention_fwd");
    auto& forward_slots = forward_propagation.slots[layer];

    const auto& src_views = get_inputs(forward_propagation, layer);
    const TensorView& source_input = src_views[min(size_t{1}, src_views.size() - 1)];

    const TensorView& query = get_input(forward_propagation, layer);
    const Shape& query_shape = query.get_shape();

    // Interleaved heads: SDPA writes the merged (B, S, H*D) output directly.
    const bool sdpa_interleaved = use_sdpa && interleaved_heads && query.is_cuda();

    TensorView attention_out = sdpa_interleaved
        ? forward_slots[attention_output_slot]
        : forward_slots[scratch_slot].reshape_prefix(
              {forward_propagation.batch_size, query_shape[1], query_shape[2], query_shape[3]});

    // Keys and values come from the last of the layer's inputs: its own, for
    // self-attention, and the encoder's for cross-attention. That is the
    // sequence the mask has to describe, and in an encoder-decoder it is not
    // the one the queries came from.
    const SequenceLengths explicit_lengths =
        forward_propagation.input_sequence_lengths(layer, forward_propagation.inputs[layer].size() - 1);

#ifdef OPENNN_HAS_CUDA
    if (use_sdpa && query.is_cuda())
    {
        throw_if(sdpa_state_slot == 0,
                 "AttentionOperator: use_sdpa is set but the owning layer did not "
                 "assign sdpa_state_slot.");
        apply_sdpa_forward(query, get_input(forward_propagation, layer, 1), get_input(forward_propagation, layer, 2), source_input,
                           attention_out, forward_slots[sdpa_qkv_pack_slot],
                           span<const TensorView>(forward_slots.data() + sdpa_state_slot,
                                                  sdpa_state_slots_count),
                           is_training,
                           explicit_lengths.device);
    }
    else
#endif
    apply_unfused(query, get_input(forward_propagation, layer, 1), get_input(forward_propagation, layer, 2), source_input,
                  get_output(forward_propagation, layer), get_output(forward_propagation, layer, 1),
                  attention_out, forward_slots[dropout_mask_slot],
                  forward_slots[scratch_slot].as<float>(), is_training,
                  explicit_lengths);

    if (!sdpa_interleaved) merge_output_heads(forward_propagation, layer);
}

void AttentionOperator::merge_output_heads(ForwardPropagation& forward_propagation, size_t layer) const
{
    const Index batch_size = forward_propagation.batch_size;
    auto& forward_slots = forward_propagation.slots[layer];

    const TensorView heads = forward_slots[scratch_slot].reshape_prefix(
        {batch_size, heads_number, query_sequence_length, head_dimension});
    TensorView merged = forward_slots[attention_output_slot].reshape_prefix(
        {batch_size, query_sequence_length, heads_number, head_dimension});

    merge_heads(heads, merged);
}

void AttentionOperator::split_output_delta(ForwardPropagation& forward_propagation,
                                           BackPropagation& back_propagation,
                                           size_t layer) const
{
    const Index batch_size = forward_propagation.batch_size;

    const TensorView merged_delta =
        back_propagation.slots[layer][merged_output_delta_slot].reshape_prefix(
            {batch_size, query_sequence_length, heads_number, head_dimension});
    TensorView heads_delta = forward_propagation.slots[layer][scratch_slot].reshape_prefix(
        {batch_size, heads_number, query_sequence_length, head_dimension});

    split_heads(merged_delta, heads_delta);
}

void AttentionOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    PROFILE_SCOPE("op:attention_bwd");
    auto& forward_slots = forward_propagation.slots[layer];

    const TensorView& query             = get_input(forward_propagation, layer);
    const TensorView& key               = get_input(forward_propagation, layer, 1);
    const TensorView& value             = get_input(forward_propagation, layer, 2);
    const TensorView& attention_weights = get_output(forward_propagation, layer);
    const TensorView& attention_weights_dropped = get_output(forward_propagation, layer, 1);
    const TensorView& dropout_mask = forward_slots[dropout_mask_slot];

    // Interleaved heads: the merged (B, S, H*D) output delta is dO as it is.
    const bool sdpa_interleaved = use_sdpa && interleaved_heads && query.is_cuda();
    if (!sdpa_interleaved) split_output_delta(forward_propagation, back_propagation, layer);

    const Shape& query_shape = query.get_shape();
    const TensorView output_delta = sdpa_interleaved
        ? back_propagation.slots[layer][merged_output_delta_slot]
        : forward_slots[scratch_slot].reshape_prefix(
              {forward_propagation.batch_size, query_shape[1], query_shape[2], query_shape[3]});

    TensorView& attention_weight_delta = get_output_delta(back_propagation, layer);
    TensorView& query_delta            = get_output_delta(back_propagation, layer, 1);
    TensorView& key_delta              = get_output_delta(back_propagation, layer, 2);
    TensorView& value_delta            = get_output_delta(back_propagation, layer, 3);

#ifdef OPENNN_HAS_CUDA

    // The backward graph reuses the length tensors the forward filled, so it
    // masks with whatever the forward masked with, derived or exported alike.
    if (output_delta.is_cuda() && use_sdpa)
    {
        throw_if(sdpa_gradient_slot == 0,
                 "AttentionOperator: use_sdpa is set but the owning layer did not "
                 "assign sdpa_gradient_slot (gradient scratch comes from the backward arena).");
        throw_if(sdpa_state_slot == 0,
                 "AttentionOperator: use_sdpa is set but the owning layer did not "
                 "assign sdpa_state_slot.");

        const auto& slots = back_propagation.slots[layer];
        return apply_sdpa_backward(query, key, value, forward_slots[attention_output_slot],
                                   output_delta,
                                   query_delta, key_delta, value_delta,
                                   span<const TensorView>(forward_slots.data() + sdpa_state_slot,
                                                          sdpa_state_slots_count),
                                   span<const TensorView>(slots.data() + sdpa_gradient_slot,
                                                          sdpa_scratch_slots_count));
    }

    if (output_delta.is_cuda())
    {
        apply_delta_unfused(query, key, value,
                            attention_weights, attention_weights_dropped,
                            dropout_mask, output_delta, attention_weight_delta,
                            query_delta, key_delta, value_delta,
            [&]() {
                CHECK_CUDNN(cudnnSoftmaxBackward(device::get_cudnn_handle(),
                                                 CUDNN_SOFTMAX_ACCURATE,
                                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                                 &one,
                                                 attention_weights.get_descriptor(),
                                                 attention_weights.get_data(),
                                                 attention_weight_delta.get_descriptor(),
                                                 attention_weight_delta.get_data(),
                                                 &zero,
                                                 attention_weight_delta.get_descriptor(),
                                                 attention_weight_delta.get_data()));
            });
        return;
    }
#endif
    apply_delta_cpu(query, key, value,
                    attention_weights, attention_weights_dropped,
                    dropout_mask, output_delta,
                    attention_weight_delta,
                    query_delta, key_delta, value_delta);
}

void AttentionOperator::apply_unfused(const TensorView& query,
                              const TensorView& key,
                              const TensorView& value,
                              const TensorView& source_input,
                              TensorView& attention_weights,
                              TensorView& attention_weights_dropped,
                              TensorView& output,
                              TensorView& dropout_mask,
                              [[maybe_unused]] void* scratch,
                              bool is_training,
                              const SequenceLengths explicit_lengths)
{
    const bool use_cpu_fast_path =
        !query.is_cuda()
        && !use_causal_mask
        && !dropout.active()
        && query.is_fp32()
        && key.is_fp32()
        && value.is_fp32()
        && attention_weights.is_fp32()
        && output.is_fp32()
        && source_input.get_shape().get_rank() == 3
        && attention_weights.get_shape().get_rank() == 4;

    if (use_cpu_fast_path)
    {
        vector<Index> valid_lengths;
        bool has_padding = false;
        bool have_lengths = false;

        if (explicit_lengths.host && Index(explicit_lengths.host->size()) == source_input.get_shape()[0])
        {
            valid_lengths = *explicit_lengths.host;
            have_lengths = true;
            has_padding = ranges::any_of(valid_lengths,
                [&](const Index length) { return length < source_input.get_shape()[1]; });
        }
        else
        {
            have_lengths = get_contiguous_source_lengths(source_input, valid_lengths, has_padding);
        }

        if (have_lengths && has_padding)
        {
            const Index batch_size = source_input.get_shape()[0];
            const Index query_length = query.get_shape()[2];
            const Index source_length = key.get_shape()[2];
            const Index batch_heads = batch_size * heads_number;
            const float scale = scaling_factor();

            #pragma omp parallel for
            for (Index batch_head = 0; batch_head < batch_heads; ++batch_head)
            {
                const Index batch_index = batch_head / heads_number;
                const Index valid_length = valid_lengths[batch_index];

                const MatrixMap query_matrix = query.as_matrix(batch_head);
                const MatrixMap key_matrix = key.as_matrix(batch_head);
                const MatrixMap value_matrix = value.as_matrix(batch_head);
                MatrixMap attention_matrix = attention_weights.as_matrix(batch_head);
                MatrixMap output_matrix = output.as_matrix(batch_head);

                const Index query_rows = zero_padded_queries
                    ? min(valid_length, query_length)
                    : query_length;

                auto attention_computed = attention_matrix.topRows(query_rows);
                attention_computed.leftCols(valid_length).noalias() =
                    scale * (query_matrix.topRows(query_rows) * key_matrix.topRows(valid_length).transpose());
                if (valid_length < source_length)
                    attention_computed.rightCols(source_length - valid_length).setZero();
                softmax_rows_prefix(attention_matrix.data(), query_rows, source_length, valid_length);
                output_matrix.topRows(query_rows).noalias() =
                    attention_computed.leftCols(valid_length) * value_matrix.topRows(valid_length);

                if (query_rows < query_length)
                {
                    attention_matrix.bottomRows(query_length - query_rows).setZero();
                    output_matrix.bottomRows(query_length - query_rows).setZero();
                }
            }

            return;
        }
    }

    multiply(query, false, key, true, attention_weights, scaling_factor(), 0.0f);

    const Index batch_size = source_input.get_shape()[0];
    const Index source_length = source_input.get_shape()[1];
    const Index embedding_dimension = source_input.get_shape()[2];
    const Index query_length = attention_weights.get_shape()[2];

#ifdef OPENNN_HAS_CUDA
    if (attention_weights.is_cuda() && explicit_lengths.device)
    {
        attention_weights.dispatch([&]<typename T>() {
            attention_length_masked_softmax_cuda<T>(to_int(batch_size),
                                          to_int(heads_number),
                                          to_int(query_length),
                                          to_int(source_length),
                                          explicit_lengths.device,
                                          attention_weights.as<T>(),
                                          reinterpret_cast<T*>(scratch),
                                          use_causal_mask,
                                          zero_padded_queries);
        });
    }
    else if (attention_weights.is_cuda())
        attention_weights.dispatch([&]<typename T>() {
            attention_masked_softmax_cuda<T>(to_int(batch_size),
                                    to_int(heads_number),
                                    to_int(query_length),
                                    to_int(source_length),
                                    to_int(embedding_dimension),
                                    source_input.as<T>(),
                                    attention_weights.as<T>(),
                                    reinterpret_cast<T*>(scratch),
                                    use_causal_mask,
                                    zero_padded_queries);
        });
    else
#endif
    {
        const Index att_rows_per_batch = heads_number * query_length;

        #pragma omp parallel for
        for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
        {
            const float* source_batch = source_input.as<float>() + batch_index * source_length * embedding_dimension;
            float*       attention_batch = attention_weights.as<float>() + batch_index * att_rows_per_batch * source_length;

            for (Index source_index = 0; source_index < source_length; ++source_index)
            {
                const float* source_row = source_batch + source_index * embedding_dimension;
                if (row_nonzero(source_row, embedding_dimension)) continue;

                for (Index row_index = 0; row_index < att_rows_per_batch; ++row_index)
                    attention_batch[row_index * source_length + source_index] = SOFTMAX_MASK_VALUE;
            }
        }

        if (use_causal_mask)
        {
            const Index batch_heads = batch_size * heads_number;
            MatrixMap attention_flat = attention_weights.as_flat_matrix();
            attention_flat += causal_mask.replicate(batch_heads, 1);
        }
    }

    if (!attention_weights.is_cuda())
    {
        softmax(attention_weights);

        if (zero_padded_queries)
        {
            const Index att_rows_per_batch = heads_number * query_length;

            #pragma omp parallel for
            for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
            {
                const float* source_batch = source_input.as<float>() + batch_index * source_length * embedding_dimension;
                float*       attention_batch = attention_weights.as<float>() + batch_index * att_rows_per_batch * source_length;

                for (Index query_index = 0; query_index < min(query_length, source_length); ++query_index)
                {
                    if (row_nonzero(source_batch + query_index * embedding_dimension, embedding_dimension)) continue;

                    for (Index head_index = 0; head_index < heads_number; ++head_index)
                    {
                        float* row = attention_batch + (head_index * query_length + query_index) * source_length;
                        for (Index k = 0; k < source_length; ++k) row[k] = 0.0f;
                    }
                }
            }
        }
    }

    if (is_training && dropout.active())
    {
        copy(attention_weights, attention_weights_dropped);
        dropout_forward(attention_weights_dropped, dropout_mask, dropout.rate);
        return multiply(attention_weights_dropped, false, value, false, output);
    }

    multiply(attention_weights, false, value, false, output);
}

#ifdef OPENNN_HAS_CUDA

void AttentionOperator::apply_sdpa_forward(const TensorView& query,
                               const TensorView& key,
                               const TensorView& value,
                               const TensorView& source_input,
                               TensorView& output,
                               const TensorView& qkv_pack_bf16,
                               const span<const TensorView> state,
                               bool is_training,
                               const int* explicit_lengths)
{
    throw_if(!sdpa_supported(query.get_type(), query.get_device()),
             "AttentionOperator: SDPA backend selected by the layer "
             "but not supported (build without HAVE_CUDNN_FRONTEND, "
             "unsupported dtype, or CPU runtime).");

    if (!sdpa_cache) sdpa_cache = make_unique<SDPACache>();

    const bool dropout_in_graph = dropout.active() && is_training;

    SDPACache::CacheKey cache_key{
        query.get_shape()[0],
        query.get_shape()[2],
        key.get_shape()[2],
        heads_number,
        head_dimension,
        dropout_in_graph,
        dropout_in_graph ? dropout.rate : 0.0f,
        use_causal_mask,
        is_training,
        interleaved_heads
    };

    auto& entry = sdpa_cache->get_or_create_entry(cache_key);
    if (!entry.fwd_graph)
        build_sdpa_forward_graph(entry, cache_key);

    throw_if(state.size() < sdpa_state_slots_count,
             "SDPA forward: {} state slots were passed, the layout needs {}.",
             state.size(), sdpa_state_slots_count);

    const TensorView& statistics = state[sdpa_statistics_state];
    const TensorView& query_lengths = state[sdpa_query_lengths_state];
    const TensorView& source_lengths = state[sdpa_source_lengths_state];
    const TensorView& dropout_state = state[sdpa_dropout_state];

    throw_if(query_lengths.empty() || source_lengths.empty(),
             "SDPA forward: sequence-length state was not planned.");
    throw_if(is_training && statistics.empty(),
             "SDPA forward: training statistics state was not planned.");
    throw_if(dropout_in_graph && dropout_state.empty(),
             "SDPA forward: dropout state was not planned.");

    int32_t* const query_length_data = query_lengths.as<int32_t>();
    int32_t* const source_length_data = source_lengths.as<int32_t>();

    // Where the lengths come from decides whether padding is visible at all.
    // The scan reads them back out of the activations, which works only while a
    // padded row is still the zero row the Embedding wrote: one normalization
    // downstream turns that row into the normalization's own shift, and every
    // attention layer past the first sees padding it cannot distinguish from
    // data. Lengths an Embedding exports are exact and stay exact, so they are
    // preferred wherever they are available.
    if (explicit_lengths)
        attention_sdpa_lengths_cuda(to_int(cache_key.batch_size), to_int(cache_key.q_seq),
                                    to_int(cache_key.src_seq), explicit_lengths,
                                    query_length_data,
                                    source_length_data);
    else
        refresh_sdpa_sequence_lengths(cache_key, source_input,
                                      query_length_data, source_length_data);

    if (dropout_in_graph)
    {
        const array<int64_t, 2> values{
            static_cast<int64_t>(sdpa_dropout_seed),
            static_cast<int64_t>(sdpa_dropout_offset++)};
        device::copy_async(dropout_state.get_data(), values.data(),
                           Index(sizeof(values)),
                           device::CopyKind::HostToDevice,
                           device::get_compute_stream());
    }

    void* q_ptr = query.get_data();
    void* k_ptr = key.get_data();
    void* v_ptr = value.get_data();
    void* o_ptr = output.get_data();
    const bool fp32_via_bf16 = query.is_fp32();

    bfloat16* output_bf16 = nullptr;

    if (fp32_via_bf16)
    {
        cudaStream_t cstream = device::get_compute_stream();
        const Index q_elems  = query.size();
        const Index kv_elems = key.size();

        throw_if(qkv_pack_bf16.empty(),
                 "SDPA forward: the transient Q/K/V/O BF16 pack was not planned "
                 "(ForwardPropagation::set ran without the SDPA pack spec).");

        const SdpaBf16Pack pack{q_elems, kv_elems};

        throw_if(qkv_pack_bf16.size() < pack.total_elements(),
                 "SDPA forward: the Q/K/V/O BF16 pack holds {} elements, the "
                 "layout needs {}.", qkv_pack_bf16.size(), pack.total_elements());

        const auto [query_bf16, key_bf16, value_bf16, packed_output] =
            pack.over(qkv_pack_bf16.as<bfloat16>());
        output_bf16 = packed_output;

        cast_fp32_to_bf16(q_elems,  query.as<float>(), query_bf16, cstream);
        cast_fp32_to_bf16(kv_elems, key.as<float>(),   key_bf16, cstream);
        cast_fp32_to_bf16(kv_elems, value.as<float>(), value_bf16, cstream);
        q_ptr = query_bf16;
        k_ptr = key_bf16;
        v_ptr = value_bf16;
        o_ptr = output_bf16;
    }
    cudnn_frontend::VariantPack tensor_map;
    tensor_map[entry.fwd_Q] = q_ptr;
    tensor_map[entry.fwd_K] = k_ptr;
    tensor_map[entry.fwd_V] = v_ptr;
    tensor_map[entry.fwd_O] = o_ptr;
    tensor_map[entry.fwd_SeqLenQ]  = query_length_data;
    tensor_map[entry.fwd_SeqLenKV] = source_length_data;
    if (is_training && entry.fwd_Stats)
        tensor_map[entry.fwd_Stats] = statistics.get_data();
    if (dropout_in_graph)
    {
        tensor_map[entry.fwd_Seed]   = dropout_state.as<int64_t>();
        tensor_map[entry.fwd_Offset] = dropout_state.as<int64_t>() + 1;
    }

    if (entry.fwd_autotune_pending)
        cudnn_frontend::autotune_now(entry.fwd_autotune_pending,
                                     *entry.fwd_graph, tensor_map,
                                     entry.fwd_workspace_bytes, "sdpa fwd");
    cudnn_frontend::execute_graph(*entry.fwd_graph,
                                  tensor_map,
                                  cudnn_frontend::shared_workspace(entry.fwd_workspace_bytes),
                                  "SDPA forward execute",
                                  cudnn_frontend::timing_label("sdpa_fwd"));
    if (fp32_via_bf16)
        cast_bf16_to_fp32(output.size(), output_bf16, output.as<float>());
}

#endif

template<typename SoftmaxBwd>
void AttentionOperator::apply_delta_unfused(const TensorView& query,
                                     const TensorView& key,
                                     const TensorView& value,
                                     const TensorView& attention_weights,
                                     const TensorView& attention_weights_dropped,
                                     const TensorView& dropout_mask,
                                     const TensorView& output_delta,
                                     TensorView& attention_weight_delta,
                                     TensorView& query_delta,
                                     TensorView& key_delta,
                                     TensorView& value_delta,
                                     SoftmaxBwd&& softmax_bwd) const
{
    const TensorView& attention_used = dropout.active()
        ? attention_weights_dropped
        : attention_weights;

    multiply(attention_used, true, output_delta, false, value_delta);
    multiply(output_delta, false, value, true, attention_weight_delta);

    if (dropout.active())
        dropout_backward(attention_weight_delta, dropout_mask, dropout.rate);

    if (!attention_weight_delta.empty())
        softmax_bwd();

    const float scale = scaling_factor();
    multiply(attention_weight_delta, false, key,   false, query_delta, scale, 0.0f);
    multiply(attention_weight_delta, true,  query, false, key_delta,   scale, 0.0f);
}

void AttentionOperator::apply_delta_cpu(const TensorView& query,
                                const TensorView& key,
                                const TensorView& value,
                                const TensorView& attention_weights,
                                const TensorView& attention_weights_dropped,
                                const TensorView& dropout_mask,
                                const TensorView& output_delta,
                                TensorView& attention_weight_delta,
                                TensorView& query_delta,
                                TensorView& key_delta,
                                TensorView& value_delta) const
{
    const bool use_cpu_fast_path =
        !query.is_cuda()
        && !use_causal_mask
        && !dropout.active()
        && compute_dtype == Type::FP32
        && attention_weights.get_shape().get_rank() == 4
        && attention_weight_delta.get_shape().get_rank() == 4;

    if (use_cpu_fast_path)
    {
        const Index batch_size = query.get_shape()[0];
        const Index source_length = key.get_shape()[2];
        vector<Index> valid_lengths(batch_size);
        bool has_padding = false;
        bool valid_prefixes = true;

        for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
        {
            const Index valid_length = infer_attention_prefix_length(attention_weights, batch_index);
            if (valid_length <= 0 || valid_length > source_length)
            {
                valid_prefixes = false;
                break;
            }

            valid_lengths[batch_index] = valid_length;
            if (valid_length < source_length)
                has_padding = true;
        }

        if (valid_prefixes && has_padding)
        {
            const Index batch_heads = batch_size * heads_number;
            const float scale = scaling_factor();

            #pragma omp parallel for
            for (Index batch_head = 0; batch_head < batch_heads; ++batch_head)
            {
                const Index batch_index = batch_head / heads_number;
                const Index valid_length = valid_lengths[batch_index];

                const MatrixMap query_matrix = query.as_matrix(batch_head);
                const MatrixMap key_matrix = key.as_matrix(batch_head);
                const MatrixMap value_matrix = value.as_matrix(batch_head);
                const MatrixMap attention_matrix = attention_weights.as_matrix(batch_head);
                const MatrixMap output_delta_matrix = output_delta.as_matrix(batch_head);

                MatrixMap attention_delta_matrix = attention_weight_delta.as_matrix(batch_head);
                MatrixMap query_delta_matrix = query_delta.as_matrix(batch_head);
                MatrixMap key_delta_matrix = key_delta.as_matrix(batch_head);
                MatrixMap value_delta_matrix = value_delta.as_matrix(batch_head);

                const auto attention_valid = attention_matrix.leftCols(valid_length);
                auto attention_delta_valid = attention_delta_matrix.leftCols(valid_length);

                value_delta_matrix.topRows(valid_length).noalias() =
                    attention_valid.transpose() * output_delta_matrix;

                attention_delta_valid.noalias() =
                    output_delta_matrix * value_matrix.topRows(valid_length).transpose();

                const VectorR dot = (attention_valid.array() * attention_delta_valid.array()).rowwise().sum();
                attention_delta_valid.array() =
                    attention_valid.array() * (attention_delta_valid.colwise() - dot).array();

                query_delta_matrix.noalias() =
                    scale * (attention_delta_valid * key_matrix.topRows(valid_length));
                key_delta_matrix.topRows(valid_length).noalias() =
                    scale * (attention_delta_valid.transpose() * query_matrix);

                if (valid_length < source_length)
                {
                    attention_delta_matrix.rightCols(source_length - valid_length).setZero();
                    key_delta_matrix.bottomRows(source_length - valid_length).setZero();
                    value_delta_matrix.bottomRows(source_length - valid_length).setZero();
                }
            }

            return;
        }
    }

    apply_delta_unfused(query, key, value,
                        attention_weights, attention_weights_dropped,
                        dropout_mask, output_delta, attention_weight_delta,
                        query_delta, key_delta, value_delta,
        [&]() {
            const MatrixMap y  = attention_weights.as_flat_matrix();
            MatrixMap       dY = attention_weight_delta.as_flat_matrix();
            const VectorR dot = (y.array() * dY.array()).rowwise().sum();
            dY.array() = y.array() * (dY.colwise() - dot).array();
        });
}

#ifdef OPENNN_HAS_CUDA

void AttentionOperator::apply_sdpa_backward(const TensorView& query,
                                const TensorView& key,
                                const TensorView& value,
                                const TensorView& attention_output,
                                const TensorView& output_delta,
                                TensorView& query_delta,
                                TensorView& key_delta,
                                TensorView& value_delta,
                                const span<const TensorView> state,
                                span<const TensorView> bf16_scratch) const
{
    throw_if(!sdpa_supported(query.get_type(), query.get_device()) || !sdpa_cache,
             "AttentionOperator: SDPA backward called without a live SDPA "
             "forward graph (use_sdpa set inconsistently between fwd/bwd).");

    const bool dropout_in_graph = dropout.active();

    SDPACache::CacheKey cache_key{
        query.get_shape()[0],
        query.get_shape()[2],
        key.get_shape()[2],
        heads_number,
        head_dimension,
        dropout_in_graph,
        dropout_in_graph ? dropout.rate : 0.0f,
        use_causal_mask,
        true,
        interleaved_heads
    };

    auto& entry = sdpa_cache->get_or_create_entry(cache_key);
    if (!entry.bwd_graph)
        build_sdpa_backward_graph(entry, cache_key);

    throw_if(state.size() < sdpa_state_slots_count,
             "SDPA backward: {} state slots were passed, the layout needs {}.",
             state.size(), sdpa_state_slots_count);

    const TensorView& statistics = state[sdpa_statistics_state];
    const TensorView& query_lengths = state[sdpa_query_lengths_state];
    const TensorView& source_lengths = state[sdpa_source_lengths_state];
    const TensorView& dropout_state = state[sdpa_dropout_state];

    throw_if(statistics.empty() || query_lengths.empty() || source_lengths.empty(),
             "SDPA backward: forward state was not preserved.");
    throw_if(dropout_in_graph && dropout_state.empty(),
             "SDPA backward: dropout state was not preserved.");

    void* bq  = query.get_data();
    void* bk  = key.get_data();
    void* bv  = value.get_data();

    void* bo  = attention_output.get_data();
    void* bdo = output_delta.get_data();
    void* bdq = query_delta.get_data();
    void* bdk = key_delta.get_data();
    void* bdv = value_delta.get_data();
    const bool fp32_via_bf16 = query.is_fp32();

    // Indexed positionally below, so the count is a precondition rather than
    // something the caller can get away with shortening.
    throw_if(bf16_scratch.size() < sdpa_scratch_slots_count,
             "SDPA backward: {} BF16 scratch slots were passed, the layout needs {}.",
             bf16_scratch.size(), sdpa_scratch_slots_count);

    const TensorView& output_gradient_bf16 = bf16_scratch[0];
    const TensorView& query_gradient_bf16  = bf16_scratch[1];
    const TensorView& key_gradient_bf16    = bf16_scratch[2];
    const TensorView& value_gradient_bf16  = bf16_scratch[3];
    if (fp32_via_bf16)
    {
        const TensorView& query_bf16  = bf16_scratch[4];
        const TensorView& key_bf16    = bf16_scratch[5];
        const TensorView& value_bf16  = bf16_scratch[6];
        const TensorView& output_bf16 = bf16_scratch[7];

        throw_if(output_gradient_bf16.empty() || query_gradient_bf16.empty()
                 || key_gradient_bf16.empty() || value_gradient_bf16.empty()
                 || query_bf16.empty() || key_bf16.empty() || value_bf16.empty()
                 || output_bf16.empty(),
                 "SDPA backward: BF16 scratch views were not planned "
                 "(BackPropagation::set ran without the SDPA backward specs).");

        cudaStream_t cstream = device::get_compute_stream();
        cast_fp32_to_bf16(query.size(), query.as<float>(), query_bf16.as<bfloat16>(), cstream);
        cast_fp32_to_bf16(key.size(),   key.as<float>(),   key_bf16.as<bfloat16>(), cstream);
        cast_fp32_to_bf16(value.size(), value.as<float>(), value_bf16.as<bfloat16>(), cstream);
        cast_fp32_to_bf16(attention_output.size(), attention_output.as<float>(),
                          output_bf16.as<bfloat16>(), cstream);
        bq  = query_bf16.get_data();
        bk  = key_bf16.get_data();
        bv  = value_bf16.get_data();
        bo  = output_bf16.get_data();
        cast_fp32_to_bf16(output_delta.size(), output_delta.as<float>(),
                          output_gradient_bf16.as<bfloat16>(), cstream);
        bdo = output_gradient_bf16.get_data();
        bdq = query_gradient_bf16.get_data();
        bdk = key_gradient_bf16.get_data();
        bdv = value_gradient_bf16.get_data();
    }
    cudnn_frontend::VariantPack tensor_map;
    tensor_map[entry.bwd_Q]     = bq;
    tensor_map[entry.bwd_K]     = bk;
    tensor_map[entry.bwd_V]     = bv;
    tensor_map[entry.bwd_O]     = bo;
    tensor_map[entry.bwd_dO]    = bdo;
    tensor_map[entry.bwd_Stats] = statistics.get_data();
    tensor_map[entry.bwd_dQ]    = bdq;
    tensor_map[entry.bwd_dK]    = bdk;
    tensor_map[entry.bwd_dV]    = bdv;
    tensor_map[entry.bwd_SeqLenQ]  = query_lengths.get_data();
    tensor_map[entry.bwd_SeqLenKV] = source_lengths.get_data();
    if (dropout_in_graph)
    {
        tensor_map[entry.bwd_Seed]   = dropout_state.as<int64_t>();
        tensor_map[entry.bwd_Offset] = dropout_state.as<int64_t>() + 1;
    }

    if (entry.bwd_autotune_pending)
        cudnn_frontend::autotune_now(entry.bwd_autotune_pending,
                                     *entry.bwd_graph, tensor_map,
                                     entry.bwd_workspace_bytes, "sdpa bwd");
    cudnn_frontend::execute_graph(*entry.bwd_graph,
                                  tensor_map,
                                  cudnn_frontend::shared_workspace(entry.bwd_workspace_bytes),
                                  "SDPA backward execute",
                                  cudnn_frontend::timing_label("sdpa_bwd"));
    if (fp32_via_bf16)
    {
        cast_bf16_to_fp32(query.size(), query_gradient_bf16.as<bfloat16>(), query_delta.as<float>());
        cast_bf16_to_fp32(key.size(),   key_gradient_bf16.as<bfloat16>(), key_delta.as<float>());
        cast_bf16_to_fp32(value.size(), value_gradient_bf16.as<bfloat16>(), value_delta.as<float>());
    }
}

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
