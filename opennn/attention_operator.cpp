//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A T T E N T I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifdef HAVE_CUDNN_FRONTEND
#include <cudnn_frontend.h>
#endif

#include "attention_operator.h"
#include "device_backend.h"
#include "kernel.cuh"
#include "json.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"

namespace opennn
{

namespace
{

#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)
constexpr bool cudnn_frontend_available = true;
#else
constexpr bool cudnn_frontend_available = false;
#endif

}


void AttentionOp::set(Index new_heads_number, Index new_head_dimension,
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
    {
        causal_mask = MatrixR::NullaryExpr(query_sequence_length, source_sequence_length,
            [](Index row, Index column) { return column > row ? NEG_INFINITY : 0.0f; });
    }
    else
    {
        causal_mask.resize(0, 0);
    }
}

float AttentionOp::scaling_factor() const
{
    return (head_dimension == 0) ? 0.25f : 1.0f / float(sqrt(head_dimension));
}

bool AttentionOp::get_contiguous_source_lengths(const TensorView& source_input,
                                                vector<Index>& lengths,
                                                bool& has_padding)
{
    if (source_input.shape.rank != 3 || source_input.type != Type::FP32)
        return false;

    const Index batch_size          = source_input.shape[0];
    const Index sequence_length     = source_input.shape[1];
    const Index embedding_dimension = source_input.shape[2];
    const float* source_data        = source_input.as<float>();

    auto row_nonzero = [embedding_dimension](const float* row) {
        for (Index j = 0; j < embedding_dimension; ++j)
            if (abs(row[j]) > EPSILON) return true;
        return false;
    };

    lengths.assign(batch_size, sequence_length);
    has_padding = false;

    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        const float* batch = source_data + batch_index * sequence_length * embedding_dimension;

        Index valid_length = 0;
        while (valid_length < sequence_length
               && row_nonzero(batch + valid_length * embedding_dimension))
            ++valid_length;

        if (valid_length == 0) return false;

        for (Index i = valid_length; i < sequence_length; ++i)
            if (row_nonzero(batch + i * embedding_dimension))
                return false;

        if (valid_length < sequence_length) has_padding = true;
        lengths[batch_index] = valid_length;
    }

    return true;
}

void AttentionOp::softmax_rows_prefix(float* matrix, Index rows, Index cols, Index length)
{
    for (Index row = 0; row < rows; ++row)
    {
        Eigen::Map<Eigen::VectorXf> v(matrix + row * cols, length);
        v = (v.array() - v.maxCoeff()).exp();
        v /= v.sum();
    }
}

Index AttentionOp::infer_attention_prefix_length(const TensorView& attention_weights,
                                                 Index batch_index)
{
    const auto& shape = attention_weights.shape;
    const float* first_row = attention_weights.as<float>()
        + batch_index * shape[1] * shape[2] * shape[3];

    Index length = shape[3];
    while (length > 0 && first_row[length - 1] == 0.0f)
        --length;

    return length;
}

vector<TensorSpec> AttentionOp::forward_scratch_specs(Index batch_size) const
{
    if (use_sdpa && !dropout.active())
        return vector<TensorSpec>(2, {Shape{}, compute_dtype});

    const Shape attention_shape = {batch_size, heads_number,
                                   query_sequence_length, source_sequence_length};
    const Shape dropout_shape = dropout.active() ? attention_shape : Shape{};

    return {
        {attention_shape, compute_dtype},
        {dropout_shape,   compute_dtype},
    };
}

TensorSpec AttentionOp::backward_scratch_spec(Index batch_size) const
{
    if (use_sdpa)
        return {Shape{}, compute_dtype};

    return {{batch_size, heads_number, query_sequence_length, source_sequence_length},
            compute_dtype};
}

bool AttentionOp::sdpa_supported(Type dtype, Device device)
{
    return cudnn_frontend_available && device == Device::CUDA
        && (dtype == Type::BF16 || dtype == Type::FP32);
}

#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)

namespace
{

cudnn_frontend::DataType_t to_cudnn_frontend_dtype(Type t)
{
    using enum Type;
    switch (t)
    {
        case FP32: return cudnn_frontend::DataType_t::FLOAT;
        case BF16: return cudnn_frontend::DataType_t::BFLOAT16;
        default:         return cudnn_frontend::DataType_t::FLOAT;
    }
}

}  // namespace

#endif

struct AttentionOp::SDPACache
{
    struct CacheKey
    {
        Index batch_size = 0;
        Index q_seq      = 0;
        Index src_seq    = 0;
        Index heads      = 0;
        Index head_dim   = 0;
        Type  dtype      = Type::FP32;
        bool  dropout_active = false;
        bool  causal         = false;
        bool  is_training    = false;

        bool operator==(const CacheKey&) const = default;
    };

    struct CacheKeyHash
    {
        size_t operator()(const CacheKey& k) const
        {
            return hash_combine(k.batch_size, k.q_seq, k.src_seq, k.heads, k.head_dim,
                                Index(k.dtype),
                                Index(k.dropout_active), Index(k.causal), Index(k.is_training));
        }
    };

#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)
    struct Entry
    {
        shared_ptr<cudnn_frontend::graph::Graph> fwd_graph;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_Q, fwd_K, fwd_V, fwd_O, fwd_Stats;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_SeqLenQ, fwd_SeqLenKV;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_Seed, fwd_Offset;
        void* fwd_workspace_buf = nullptr;

        shared_ptr<cudnn_frontend::graph::Graph> bwd_graph;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_Q, bwd_K, bwd_V, bwd_O, bwd_dO, bwd_Stats;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_dQ, bwd_dK, bwd_dV;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_SeqLenQ, bwd_SeqLenKV;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_Seed, bwd_Offset;
        void* bwd_workspace_buf = nullptr;

        void* stats_buf = nullptr;

        void* seed_buf   = nullptr;
        void* offset_buf = nullptr;

        void* seq_len_q_buf  = nullptr;
        void* seq_len_kv_buf = nullptr;
        // bf16 scratch for the fp32-input path (cuDNN flash-attn is bf16-only):
        // Q/K/V cast down, O cast back up.
        void* q_bf16_buf = nullptr;
        void* k_bf16_buf = nullptr;
        void* v_bf16_buf = nullptr;
        void* o_bf16_buf = nullptr;
        Index q_bf16_elems = 0, kv_bf16_elems = 0, o_bf16_elems = 0;
        // backward bf16 scratch for the fp32 path: dO cast down; dQ/dK/dV
        // come out of the bf16 graph and are cast back up to fp32.
        void* do_bf16_buf = nullptr;
        void* dq_bf16_buf = nullptr;
        void* dk_bf16_buf = nullptr;
        void* dv_bf16_buf = nullptr;
        Index do_bf16_elems = 0, dq_bf16_elems = 0, dkv_bf16_elems = 0;
    };

    unordered_map<CacheKey, Entry, CacheKeyHash> entries;

    mutable Entry*   last_entry_ = nullptr;
    mutable CacheKey last_key_;
    mutable bool     last_valid_ = false;

    Entry& get_or_create_entry(const CacheKey& key)
    {
        if (last_valid_ && key == last_key_) return *last_entry_;
        Entry& e = entries[key];
        last_entry_ = &e;
        last_key_   = key;
        last_valid_ = true;
        return e;
    }

    Entry* find_entry(const CacheKey& key) const
    {
        if (last_valid_ && key == last_key_) return last_entry_;
        const auto it = entries.find(key);
        if (it == entries.end()) return nullptr;
        last_entry_ = const_cast<Entry*>(&it->second);
        last_key_   = key;
        last_valid_ = true;
        return last_entry_;
    }

    ~SDPACache()
    {
        for (auto& [_, e] : entries)
        {
            device::deallocate(Device::CUDA, e.fwd_workspace_buf, 0);
            device::deallocate(Device::CUDA, e.q_bf16_buf, 0);
            device::deallocate(Device::CUDA, e.k_bf16_buf, 0);
            device::deallocate(Device::CUDA, e.v_bf16_buf, 0);
            device::deallocate(Device::CUDA, e.o_bf16_buf, 0);
            device::deallocate(Device::CUDA, e.do_bf16_buf, 0);
            device::deallocate(Device::CUDA, e.dq_bf16_buf, 0);
            device::deallocate(Device::CUDA, e.dk_bf16_buf, 0);
            device::deallocate(Device::CUDA, e.dv_bf16_buf, 0);
            device::deallocate(Device::CUDA, e.bwd_workspace_buf, 0);
            device::deallocate(Device::CUDA, e.stats_buf, 0);
            device::deallocate(Device::CUDA, e.seed_buf, 0);
            device::deallocate(Device::CUDA, e.offset_buf, 0);
            device::deallocate(Device::CUDA, e.seq_len_q_buf, 0);
            device::deallocate(Device::CUDA, e.seq_len_kv_buf, 0);
        }
    }
#endif  // OPENNN_HAS_CUDA && HAVE_CUDNN_FRONTEND
};

#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)

namespace
{

float attention_scale(Index head_dim) { return 1.0f / sqrt(float(head_dim)); }

auto sdpa_check = [](auto s, const string& what) {
    throw_if(s.is_bad(),
             format("SDPA {}: {}", what, s.get_message()));
};

shared_ptr<cudnn_frontend::graph::Tensor_attributes>
bhsd_input(cudnn_frontend::graph::Graph& graph, const char* name, int64_t B, int64_t H, int64_t S, int64_t D)
{
    return graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim   ({B, H, S, D})
                        .set_stride({H * S * D, S * D, D, 1}));
}

shared_ptr<cudnn_frontend::graph::Tensor_attributes>
seq_len_input(cudnn_frontend::graph::Graph& graph, const char* name, int64_t batch_size)
{
    return graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim({batch_size, 1, 1, 1})
                        .set_stride({1, 1, 1, 1})
                        .set_data_type(cudnn_frontend::DataType_t::INT32));
}

void bhsd_output(shared_ptr<cudnn_frontend::graph::Tensor_attributes>& T,
                 int64_t B, int64_t H, int64_t S, int64_t D)
{
    T->set_output(true).set_dim({B, H, S, D}).set_stride({H * S * D, S * D, D, 1});
}

void build_sdpa_graph_common(cudnn_frontend::graph::Graph& graph, Type dtype)
{
    graph.set_io_data_type(to_cudnn_frontend_dtype(dtype))
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
}
void finalize_sdpa_graph(cudnn_frontend::graph::Graph& graph, cudnnHandle_t handle, const string& tag)
{
    sdpa_check(graph.validate(),                                                tag + " validate");
    sdpa_check(graph.build_operation_graph(handle),                             tag + " build_operation_graph");
    sdpa_check(graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}),               tag + " create_execution_plans");
    sdpa_check(graph.build_plans(handle, cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_CHOICE), tag + " build_plans");
}

void refresh_sdpa_sequence_lengths(AttentionOp::SDPACache::Entry& entry,
                                   const AttentionOp::SDPACache::CacheKey& k,
                                   const TensorView& source_input)
{
    const bool ok = source_input.shape.rank == 3 && source_input.shape[0] == k.batch_size && source_input.shape[1] == k.src_seq
        && source_input.is_cuda();

    if (ok)
        source_input.dispatch([&](auto tag) {
            using T = decltype(tag);
            attention_sequence_lengths_cuda<T>(to_int(k.batch_size),
                                               to_int(k.q_seq),
                                               to_int(k.src_seq),
                                               to_int(source_input.shape[2]),
                                               source_input.as<T>(),
                                               static_cast<int32_t*>(entry.seq_len_q_buf),
                                               static_cast<int32_t*>(entry.seq_len_kv_buf));
        });

    throw_if(!ok,
             "SDPA padding mask: source_input must be a rank-3 CUDA tensor with supported dtype.");
}

}  // namespace

static void build_sdpa_forward_graph(AttentionOp::SDPACache::Entry& entry,
                                      const AttentionOp::SDPACache::CacheKey& k,
                                      cudnnHandle_t handle,
                                      float dropout_rate)
{
    const auto graph = make_shared<cudnn_frontend::graph::Graph>();
    build_sdpa_graph_common(*graph, k.dtype);

    entry.fwd_Q = bhsd_input(*graph, "Q", k.batch_size, k.heads, k.q_seq,   k.head_dim);
    entry.fwd_K = bhsd_input(*graph, "K", k.batch_size, k.heads, k.src_seq, k.head_dim);
    entry.fwd_V = bhsd_input(*graph, "V", k.batch_size, k.heads, k.src_seq, k.head_dim);
    entry.fwd_SeqLenQ  = seq_len_input(*graph, "SeqLenQ",  k.batch_size);
    entry.fwd_SeqLenKV = seq_len_input(*graph, "SeqLenKV", k.batch_size);

    auto sdpa_options = cudnn_frontend::graph::SDPA_attributes()
                        .set_name("flash_attn_fwd")
                        .set_is_inference(!k.is_training)
                        .set_padding_mask(true)
                        .set_seq_len_q(entry.fwd_SeqLenQ)
                        .set_seq_len_kv(entry.fwd_SeqLenKV)
                        .set_causal_mask(k.causal)
                        .set_attn_scale(attention_scale(k.head_dim));

    if (!entry.seq_len_q_buf)
        entry.seq_len_q_buf = device::allocate(Device::CUDA,
                                               Index(size_t(k.batch_size) * sizeof(int32_t)));
    if (!entry.seq_len_kv_buf)
        entry.seq_len_kv_buf = device::allocate(Device::CUDA,
                                                Index(size_t(k.batch_size) * sizeof(int32_t)));

    if (k.dropout_active)
    {
        entry.fwd_Seed   = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                         .set_name("Seed").set_dim({1,1,1,1}).set_stride({1,1,1,1})
                                         .set_data_type(cudnn_frontend::DataType_t::INT64));
        entry.fwd_Offset = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                         .set_name("Offset").set_dim({1,1,1,1}).set_stride({1,1,1,1})
                                         .set_data_type(cudnn_frontend::DataType_t::INT64));
        sdpa_options.set_dropout(dropout_rate, entry.fwd_Seed, entry.fwd_Offset);

        if (!entry.seed_buf)
            entry.seed_buf = device::allocate(Device::CUDA, Index(sizeof(int64_t)));
        if (!entry.offset_buf)
            entry.offset_buf = device::allocate(Device::CUDA, Index(sizeof(int64_t)));
    }

    auto [O, Stats] = graph->sdpa(entry.fwd_Q, entry.fwd_K, entry.fwd_V, sdpa_options);

    bhsd_output(O, k.batch_size, k.heads, k.q_seq, k.head_dim);
    entry.fwd_O = O;

    if (k.is_training && Stats)
    {
        Stats->set_output(true)
              .set_data_type(cudnn_frontend::DataType_t::FLOAT)
              .set_dim({k.batch_size, k.heads, k.q_seq, 1})
              .set_stride({k.heads * k.q_seq, k.q_seq, 1, 1});
        entry.fwd_Stats = Stats;
    }

    finalize_sdpa_graph(*graph, handle, "fwd");

    int64_t ws = 0;
    graph->get_workspace_size(ws);
    if (ws > 0)
        entry.fwd_workspace_buf = device::allocate(Device::CUDA, Index(ws));

    if (k.is_training)
    {
        const size_t stats_bytes = size_t(k.batch_size * k.heads * k.q_seq) * sizeof(float);
        entry.stats_buf = device::allocate(Device::CUDA, Index(stats_bytes));
    }

    entry.fwd_graph = graph;
}

static void build_sdpa_backward_graph(AttentionOp::SDPACache::Entry& entry,
                                       const AttentionOp::SDPACache::CacheKey& k,
                                       cudnnHandle_t handle,
                                       float dropout_rate)
{
    const auto graph = make_shared<cudnn_frontend::graph::Graph>();
    build_sdpa_graph_common(*graph, k.dtype);

    entry.bwd_Q  = bhsd_input(*graph, "Q_bwd",  k.batch_size, k.heads, k.q_seq,   k.head_dim);
    entry.bwd_K  = bhsd_input(*graph, "K_bwd",  k.batch_size, k.heads, k.src_seq, k.head_dim);
    entry.bwd_V  = bhsd_input(*graph, "V_bwd",  k.batch_size, k.heads, k.src_seq, k.head_dim);
    entry.bwd_dO = bhsd_input(*graph, "dO_bwd", k.batch_size, k.heads, k.q_seq,   k.head_dim);
    entry.bwd_SeqLenQ  = seq_len_input(*graph, "SeqLenQ_bwd",  k.batch_size);
    entry.bwd_SeqLenKV = seq_len_input(*graph, "SeqLenKV_bwd", k.batch_size);

    entry.bwd_O = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                .set_name("O_bwd")
                                .set_dim({k.batch_size, k.heads, k.q_seq, k.head_dim})
                                .set_stride({k.heads * k.q_seq * k.head_dim,
                                             k.q_seq * k.head_dim,
                                             k.head_dim,
                                             1}));

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
        entry.bwd_Seed   = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                         .set_name("Seed_bwd").set_dim({1,1,1,1}).set_stride({1,1,1,1})
                                         .set_data_type(cudnn_frontend::DataType_t::INT64));
        entry.bwd_Offset = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                         .set_name("Offset_bwd").set_dim({1,1,1,1}).set_stride({1,1,1,1})
                                         .set_data_type(cudnn_frontend::DataType_t::INT64));
        sdpa_bwd_options.set_dropout(dropout_rate, entry.bwd_Seed, entry.bwd_Offset);
    }

    auto [dQ, dK, dV] = graph->sdpa_backward(entry.bwd_Q, entry.bwd_K, entry.bwd_V,
                                              entry.bwd_O, entry.bwd_dO, entry.bwd_Stats,
                                              sdpa_bwd_options);

    bhsd_output(dQ, k.batch_size, k.heads, k.q_seq,   k.head_dim);
    bhsd_output(dK, k.batch_size, k.heads, k.src_seq, k.head_dim);
    bhsd_output(dV, k.batch_size, k.heads, k.src_seq, k.head_dim);

    entry.bwd_dQ = dQ;
    entry.bwd_dK = dK;
    entry.bwd_dV = dV;

    finalize_sdpa_graph(*graph, handle, "bwd");

    int64_t ws = 0;
    graph->get_workspace_size(ws);
    if (ws > 0)
        entry.bwd_workspace_buf = device::allocate(Device::CUDA, Index(ws));

    entry.bwd_graph = graph;
}

#endif  // OPENNN_HAS_CUDA && HAVE_CUDNN_FRONTEND

AttentionOp::AttentionOp() = default;
AttentionOp::~AttentionOp() = default;
AttentionOp::AttentionOp(AttentionOp&&) noexcept = default;
AttentionOp& AttentionOp::operator=(AttentionOp&&) noexcept = default;

void AttentionOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training)
{
    auto& forward_slots = fp.forward_slots[layer];

    const auto& src_views = get_inputs(fp, layer, 3);
    const TensorView& source_input = src_views[min(source_view_index, src_views.size() - 1)];

    const TensorView& query = get_input(fp, layer);

    TensorView attention_out = forward_slots[scratch_slots[0]].reshape(
        {fp.batch_size, query.shape[1], query.shape[2], query.shape[3]});

    if (query.is_cuda())
    {
        apply_gpu(query, get_input(fp, layer, 1), get_input(fp, layer, 2), source_input,
                  get_output(fp, layer), get_output(fp, layer, 1),
                  attention_out, forward_slots[scratch_slots[0]].as<float>(), is_training);
        return;
    }
    apply_cpu(query, get_input(fp, layer, 1), get_input(fp, layer, 2), source_input,
              get_output(fp, layer), get_output(fp, layer, 1),
              attention_out, forward_slots[scratch_slots[0]].as<float>(), is_training);
}

void AttentionOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const
{
    auto& forward_slots = fp.forward_slots[layer];

    const TensorView& query             = get_input(fp, layer);
    const TensorView& key               = get_input(fp, layer, 1);
    const TensorView& value             = get_input(fp, layer, 2);
    const TensorView& attention_output  = forward_slots[attention_output_slots[0]];
    const TensorView& attention_weights = get_output(fp, layer);
    const TensorView& attention_weights_dropped = get_output(fp, layer, 1);

    const TensorView output_delta = forward_slots[scratch_slots[0]]
        .reshape({fp.batch_size, query.shape[1], query.shape[2], query.shape[3]});

    TensorView& attention_weight_delta = get_output_delta(bp, layer);
    TensorView& query_delta            = get_output_delta(bp, layer, 1);
    TensorView& key_delta              = get_output_delta(bp, layer, 2);
    TensorView& value_delta            = get_output_delta(bp, layer, 3);

    if (output_delta.is_cuda())
    {
        apply_delta_gpu(query, key, value, attention_output,
                        attention_weights, attention_weights_dropped,
                        output_delta,
                        attention_weight_delta,
                        query_delta, key_delta, value_delta);
        return;
    }
    apply_delta_cpu(query, key, value, attention_output,
                    attention_weights, attention_weights_dropped,
                    output_delta,
                    attention_weight_delta,
                    query_delta, key_delta, value_delta);
}

void AttentionOp::apply_cpu(const TensorView& query,
                          const TensorView& key,
                          const TensorView& value,
                          const TensorView& source_input,
                          TensorView& attention_weights,
                          TensorView& attention_weights_dropped,
                          TensorView& output,
                          [[maybe_unused]] void* scratch,
                          bool is_training)
{
    if (query.device != Device::CUDA
        && !use_causal_mask
        && !dropout.active()
        && compute_dtype == Type::FP32
        && query.type == Type::FP32
        && key.type == Type::FP32
        && value.type == Type::FP32
        && attention_weights.type == Type::FP32
        && output.type == Type::FP32
        && source_input.shape.rank == 3
        && attention_weights.shape.rank == 4)
    {
        vector<Index> valid_lengths;
        bool has_padding = false;

        if (get_contiguous_source_lengths(source_input, valid_lengths, has_padding) && has_padding)
        {
            const Index batch_size = source_input.shape[0];
            const Index query_sequence_length = query.shape[2];
            const Index source_sequence_length = key.shape[2];
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

                auto attention_valid = attention_matrix.leftCols(valid_length);
                attention_valid.noalias() = scale * (query_matrix * key_matrix.topRows(valid_length).transpose());
                if (valid_length < source_sequence_length)
                    attention_matrix.rightCols(source_sequence_length - valid_length).setZero();
                softmax_rows_prefix(attention_matrix.data(), query_sequence_length, source_sequence_length, valid_length);
                output_matrix.noalias() = attention_valid * value_matrix.topRows(valid_length);
            }

            return;
        }
    }

    multiply(query, false, key, true, attention_weights, scaling_factor(), 0.0f);

    {
        const Index batch_size = source_input.shape[0];
        const Index source_sequence_length = source_input.shape[1];
        const Index embedding_dimension = source_input.shape[2];
        const Index query_sequence_length = attention_weights.shape[2];

#ifdef OPENNN_HAS_CUDA
        if (attention_weights.is_cuda())
            attention_weights.dispatch([&](auto tag) {
                using T = decltype(tag);
                attention_masks_cuda<T>(to_int(batch_size),
                                        to_int(heads_number),
                                        to_int(query_sequence_length),
                                        to_int(source_sequence_length),
                                        to_int(embedding_dimension),
                                        source_input.as<T>(),
                                        attention_weights.as<T>(),
                                        reinterpret_cast<T*>(scratch),
                                        use_causal_mask);
            });
        else
#endif
        {
            const Index att_rows_per_batch = heads_number * query_sequence_length;

            #pragma omp parallel for
            for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
            {
                const float* source_batch = source_input.as<float>() + batch_index * source_sequence_length * embedding_dimension;
                float*       attention_batch = attention_weights.as<float>() + batch_index * att_rows_per_batch * source_sequence_length;

                for (Index source_index = 0; source_index < source_sequence_length; ++source_index)
                {
                    const float* source_row = source_batch + source_index * embedding_dimension;
                    const float max_abs = Map<const Array<float, Dynamic, 1>>(source_row, embedding_dimension).abs().maxCoeff();
                    if (max_abs > EPSILON) continue;

                    for (Index row_index = 0; row_index < att_rows_per_batch; ++row_index)
                        attention_batch[row_index * source_sequence_length + source_index] = SOFTMAX_MASK_VALUE;
                }
            }

            if (use_causal_mask)
            {
                const Index batch_heads = batch_size * heads_number;
                MatrixMap attention_flat = attention_weights.as_flat_matrix();
                attention_flat += causal_mask.replicate(batch_heads, 1);
            }
        }
    }

    softmax(attention_weights);

    const bool apply_dropout = is_training && dropout.active();
    TensorView& used = apply_dropout ? attention_weights_dropped : attention_weights;

    if (apply_dropout)
    {
        copy(attention_weights, used);
        dropout_forward(used, dropout.mask, dropout.rate);
    }

    multiply(used, false, value, false, output);
}

void AttentionOp::apply_gpu(const TensorView& query,
                          const TensorView& key,
                          const TensorView& value,
                          const TensorView& source_input,
                          TensorView& attention_weights,
                          TensorView& attention_weights_dropped,
                          TensorView& output,
                          void* scratch,
                          bool is_training)
{
#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)
    if (!use_sdpa)
    {
        apply_cpu(query, key, value, source_input,
                  attention_weights, attention_weights_dropped,
                  output, scratch, is_training);
        return;
    }

    throw_if(!sdpa_supported(query.type, query.device),
             "AttentionOp: SDPA backend selected by the layer "
             "but not supported (build without HAVE_CUDNN_FRONTEND, "
             "unsupported dtype, or CPU runtime).");

    if (!sdpa_cache) sdpa_cache = make_unique<SDPACache>();

    const bool dropout_in_graph = dropout.active() && is_training;

    const bool fp32_via_bf16 = (query.type == Type::FP32);
    const Type graph_dtype = fp32_via_bf16 ? Type::BF16 : query.type;
    SDPACache::CacheKey ck{
        query.shape[0],
        query.shape[2],
        key.shape[2],
        heads_number,
        head_dimension,
        graph_dtype,
        dropout_in_graph,
        use_causal_mask,
        is_training
    };

    auto& entry = sdpa_cache->get_or_create_entry(ck);
    if (!entry.fwd_graph)
        build_sdpa_forward_graph(entry, ck, Backend::get_cudnn_handle(), dropout.rate);

    refresh_sdpa_sequence_lengths(entry, ck, source_input);

    if (dropout_in_graph)
    {
        sdpa_last_used_offset = sdpa_dropout_offset;
        const int64_t seed_value   = static_cast<int64_t>(sdpa_dropout_seed);
        const int64_t offset_value = static_cast<int64_t>(sdpa_last_used_offset);
        device::copy_async(entry.seed_buf, &seed_value, Index(sizeof(int64_t)),
                           device::CopyKind::HostToDevice,
                           Backend::get_compute_stream());
        device::copy_async(entry.offset_buf, &offset_value, Index(sizeof(int64_t)),
                           device::CopyKind::HostToDevice,
                           Backend::get_compute_stream());
        ++sdpa_dropout_offset;
    }

    void* q_ptr = query.data;
    void* k_ptr = key.data;
    void* v_ptr = value.data;
    void* o_ptr = output.data;
    if (fp32_via_bf16)
    {
        cudaStream_t cstream = Backend::get_compute_stream();
        const Index q_elems  = query.size();
        const Index kv_elems = key.size();
        const Index o_elems  = output.size();
        if (q_elems > entry.q_bf16_elems)
        { device::deallocate(Device::CUDA, entry.q_bf16_buf, 0);
          entry.q_bf16_buf = device::allocate(Device::CUDA, q_elems * Index(sizeof(bfloat16)));
          entry.q_bf16_elems = q_elems; }
        if (kv_elems > entry.kv_bf16_elems)
        { device::deallocate(Device::CUDA, entry.k_bf16_buf, 0);
          device::deallocate(Device::CUDA, entry.v_bf16_buf, 0);
          entry.k_bf16_buf = device::allocate(Device::CUDA, kv_elems * Index(sizeof(bfloat16)));
          entry.v_bf16_buf = device::allocate(Device::CUDA, kv_elems * Index(sizeof(bfloat16)));
          entry.kv_bf16_elems = kv_elems; }
        if (o_elems > entry.o_bf16_elems)
        { device::deallocate(Device::CUDA, entry.o_bf16_buf, 0);
          entry.o_bf16_buf = device::allocate(Device::CUDA, o_elems * Index(sizeof(bfloat16)));
          entry.o_bf16_elems = o_elems; }
        cast_fp32_to_bf16_cuda(q_elems,  query.as<float>(), static_cast<bfloat16*>(entry.q_bf16_buf), cstream);
        cast_fp32_to_bf16_cuda(kv_elems, key.as<float>(),   static_cast<bfloat16*>(entry.k_bf16_buf), cstream);
        cast_fp32_to_bf16_cuda(kv_elems, value.as<float>(), static_cast<bfloat16*>(entry.v_bf16_buf), cstream);
        q_ptr = entry.q_bf16_buf; k_ptr = entry.k_bf16_buf; v_ptr = entry.v_bf16_buf; o_ptr = entry.o_bf16_buf;
    }
    unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tp;
    tp[entry.fwd_Q] = q_ptr;
    tp[entry.fwd_K] = k_ptr;
    tp[entry.fwd_V] = v_ptr;
    tp[entry.fwd_O] = o_ptr;
    tp[entry.fwd_SeqLenQ]  = entry.seq_len_q_buf;
    tp[entry.fwd_SeqLenKV] = entry.seq_len_kv_buf;
    if (is_training && entry.fwd_Stats) tp[entry.fwd_Stats] = entry.stats_buf;
    if (dropout_in_graph)
    {
        tp[entry.fwd_Seed]   = entry.seed_buf;
        tp[entry.fwd_Offset] = entry.offset_buf;
    }

    auto status = entry.fwd_graph->execute(Backend::get_cudnn_handle(), tp, entry.fwd_workspace_buf);
    throw_if(status.is_bad(),
             format("SDPA forward execute: {}", status.get_message()));
    if (fp32_via_bf16)
        cast_bf16_to_fp32_cuda(output.size(), static_cast<const bfloat16*>(entry.o_bf16_buf),
                               output.as<float>());
#else
    apply_cpu(query, key, value, source_input,
              attention_weights, attention_weights_dropped,
              output, scratch, is_training);
#endif
}

template<typename SoftmaxBwd>
void AttentionOp::apply_delta_unfused(const TensorView& query,
                                     const TensorView& key,
                                     const TensorView& value,
                                     const TensorView& attention_weights,
                                     const TensorView& attention_weights_dropped,
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
        dropout_backward(attention_weight_delta, dropout.mask, dropout.rate);

    if (!attention_weight_delta.empty())
        softmax_bwd();

    const float scale = scaling_factor();
    multiply(attention_weight_delta, false, key,   false, query_delta, scale, 0.0f);
    multiply(attention_weight_delta, true,  query, false, key_delta,   scale, 0.0f);
}

void AttentionOp::apply_delta_cpu(const TensorView& query,
                                const TensorView& key,
                                const TensorView& value,
                                const TensorView& /*attention_output*/,
                                const TensorView& attention_weights,
                                const TensorView& attention_weights_dropped,
                                const TensorView& output_delta,
                                TensorView& attention_weight_delta,
                                TensorView& query_delta,
                                TensorView& key_delta,
                                TensorView& value_delta) const
{
    if (query.device != Device::CUDA
        && !use_causal_mask
        && !dropout.active()
        && compute_dtype == Type::FP32
        && query.type == Type::FP32
        && key.type == Type::FP32
        && value.type == Type::FP32
        && attention_weights.type == Type::FP32
        && output_delta.type == Type::FP32
        && attention_weight_delta.type == Type::FP32
        && query_delta.type == Type::FP32
        && key_delta.type == Type::FP32
        && value_delta.type == Type::FP32
        && attention_weights.shape.rank == 4
        && attention_weight_delta.shape.rank == 4)
    {
        const Index batch_size = query.shape[0];
        const Index source_sequence_length = key.shape[2];
        vector<Index> valid_lengths(batch_size);
        bool has_padding = false;
        bool valid_prefixes = true;

        for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
        {
            const Index valid_length = infer_attention_prefix_length(attention_weights, batch_index);
            if (valid_length <= 0 || valid_length > source_sequence_length)
            {
                valid_prefixes = false;
                break;
            }

            valid_lengths[batch_index] = valid_length;
            if (valid_length < source_sequence_length)
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

                if (valid_length < source_sequence_length)
                {
                    attention_delta_matrix.rightCols(source_sequence_length - valid_length).setZero();
                    key_delta_matrix.bottomRows(source_sequence_length - valid_length).setZero();
                    value_delta_matrix.bottomRows(source_sequence_length - valid_length).setZero();
                }
            }

            return;
        }
    }

    apply_delta_unfused(query, key, value,
                        attention_weights, attention_weights_dropped,
                        output_delta, attention_weight_delta,
                        query_delta, key_delta, value_delta,
        [&]() {
            const MatrixMap y  = attention_weights.as_flat_matrix();
            MatrixMap       dY = attention_weight_delta.as_flat_matrix();
            const VectorR dot = (y.array() * dY.array()).rowwise().sum();
            dY.array() = y.array() * (dY.colwise() - dot).array();
        });
}

#ifdef OPENNN_HAS_CUDA

void AttentionOp::apply_delta_gpu_unfused(const TensorView& query,
                                        const TensorView& key,
                                        const TensorView& value,
                                        const TensorView& attention_weights,
                                        const TensorView& attention_weights_dropped,
                                        const TensorView& output_delta,
                                        TensorView& attention_weight_delta,
                                        TensorView& query_delta,
                                        TensorView& key_delta,
                                        TensorView& value_delta) const
{
    apply_delta_unfused(query, key, value,
                        attention_weights, attention_weights_dropped,
                        output_delta, attention_weight_delta,
                        query_delta, key_delta, value_delta,
        [&]() {
            CHECK_CUDNN(cudnnSoftmaxBackward(Backend::get_cudnn_handle(),
                                             CUDNN_SOFTMAX_ACCURATE,
                                             CUDNN_SOFTMAX_MODE_CHANNEL,
                                             &one,
                                             attention_weights.get_descriptor(),         attention_weights.data,
                                             attention_weight_delta.get_descriptor(), attention_weight_delta.data,
                                             &zero,
                                             attention_weight_delta.get_descriptor(), attention_weight_delta.data));
        });
}

#endif

void AttentionOp::apply_delta_gpu(const TensorView& query,
                                const TensorView& key,
                                const TensorView& value,
                                const TensorView& attention_output,
                                const TensorView& attention_weights,
                                const TensorView& attention_weights_dropped,
                                const TensorView& output_delta,
                                TensorView& attention_weight_delta,
                                TensorView& query_delta,
                                TensorView& key_delta,
                                TensorView& value_delta) const
{
#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)
    if (!use_sdpa)
    {
        apply_delta_gpu_unfused(query, key, value,
                                attention_weights, attention_weights_dropped,
                                output_delta, attention_weight_delta,
                                query_delta, key_delta, value_delta);
        return;
    }

    throw_if(!sdpa_supported(query.type, query.device) || !sdpa_cache,
             "AttentionOp: SDPA backward called without a live SDPA "
             "forward graph (use_sdpa set inconsistently between fwd/bwd).");

    const bool dropout_in_graph = dropout.active();

    const bool fp32_via_bf16 = (query.type == Type::FP32);
    const Type graph_dtype = fp32_via_bf16 ? Type::BF16 : query.type;
    SDPACache::CacheKey ck{
        query.shape[0],
        query.shape[2],
        key.shape[2],
        heads_number,
        head_dimension,
        graph_dtype,
        dropout_in_graph,
        use_causal_mask,
        true
    };

    SDPACache::Entry* entry_ptr = sdpa_cache->find_entry(ck);
    throw_if(!entry_ptr || !entry_ptr->fwd_graph,
             "AttentionOp::apply_delta_gpu: SDPA forward did not populate "
             "a cache entry for this shape. Cache key drifted between "
             "forward and backward (likely batch size changing across "
             "iterations under use_sdpa).");

    auto& entry = *entry_ptr;
    if (!entry.bwd_graph)
        build_sdpa_backward_graph(entry, ck, Backend::get_cudnn_handle(), dropout.rate);

    if (dropout_in_graph)
    {
        const int64_t seed_value   = static_cast<int64_t>(sdpa_dropout_seed);
        const int64_t offset_value = static_cast<int64_t>(sdpa_last_used_offset);
        device::copy_async(entry.seed_buf, &seed_value, Index(sizeof(int64_t)),
                           device::CopyKind::HostToDevice,
                           Backend::get_compute_stream());
        device::copy_async(entry.offset_buf, &offset_value, Index(sizeof(int64_t)),
                           device::CopyKind::HostToDevice,
                           Backend::get_compute_stream());
    }

    void* bq = const_cast<float*>(query.as<float>());
    void* bk = const_cast<float*>(key.as<float>());
    void* bv = const_cast<float*>(value.as<float>());
    void* bo = const_cast<float*>(attention_output.as<float>());
    void* bdo = const_cast<float*>(output_delta.as<float>());
    void* bdq = query_delta.data;
    void* bdk = key_delta.data;
    void* bdv = value_delta.data;
    if (fp32_via_bf16)
    {
        cudaStream_t cstream = Backend::get_compute_stream();
        const Index q_elems  = query.size();
        const Index kv_elems = key.size();
        const Index do_elems = output_delta.size();
        // reuse the forward's cast Q/K/V/O (same iteration's forward populated them)
        bq = entry.q_bf16_buf; bk = entry.k_bf16_buf;
        bv = entry.v_bf16_buf; bo = entry.o_bf16_buf;
        if (do_elems > entry.do_bf16_elems)
        { device::deallocate(Device::CUDA, entry.do_bf16_buf, 0);
          entry.do_bf16_buf = device::allocate(Device::CUDA, do_elems * Index(sizeof(bfloat16)));
          entry.do_bf16_elems = do_elems; }
        if (q_elems > entry.dq_bf16_elems)
        { device::deallocate(Device::CUDA, entry.dq_bf16_buf, 0);
          entry.dq_bf16_buf = device::allocate(Device::CUDA, q_elems * Index(sizeof(bfloat16)));
          entry.dq_bf16_elems = q_elems; }
        if (kv_elems > entry.dkv_bf16_elems)
        { device::deallocate(Device::CUDA, entry.dk_bf16_buf, 0);
          device::deallocate(Device::CUDA, entry.dv_bf16_buf, 0);
          entry.dk_bf16_buf = device::allocate(Device::CUDA, kv_elems * Index(sizeof(bfloat16)));
          entry.dv_bf16_buf = device::allocate(Device::CUDA, kv_elems * Index(sizeof(bfloat16)));
          entry.dkv_bf16_elems = kv_elems; }
        cast_fp32_to_bf16_cuda(do_elems, output_delta.as<float>(), static_cast<bfloat16*>(entry.do_bf16_buf), cstream);
        bdo = entry.do_bf16_buf;
        bdq = entry.dq_bf16_buf; bdk = entry.dk_bf16_buf; bdv = entry.dv_bf16_buf;
    }
    unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tp;
    tp[entry.bwd_Q]     = bq;
    tp[entry.bwd_K]     = bk;
    tp[entry.bwd_V]     = bv;
    tp[entry.bwd_O]     = bo;
    tp[entry.bwd_dO]    = bdo;
    tp[entry.bwd_Stats] = entry.stats_buf;
    tp[entry.bwd_dQ]    = bdq;
    tp[entry.bwd_dK]    = bdk;
    tp[entry.bwd_dV]    = bdv;
    tp[entry.bwd_SeqLenQ]  = entry.seq_len_q_buf;
    tp[entry.bwd_SeqLenKV] = entry.seq_len_kv_buf;
    if (dropout_in_graph)
    {
        tp[entry.bwd_Seed]   = entry.seed_buf;
        tp[entry.bwd_Offset] = entry.offset_buf;
    }

    auto status = entry.bwd_graph->execute(Backend::get_cudnn_handle(), tp, entry.bwd_workspace_buf);
    throw_if(status.is_bad(),
             format("SDPA backward execute: {}", status.get_message()));
    if (fp32_via_bf16)
    {
        cast_bf16_to_fp32_cuda(query.size(), static_cast<const bfloat16*>(entry.dq_bf16_buf), query_delta.as<float>());
        cast_bf16_to_fp32_cuda(key.size(),   static_cast<const bfloat16*>(entry.dk_bf16_buf), key_delta.as<float>());
        cast_bf16_to_fp32_cuda(value.size(), static_cast<const bfloat16*>(entry.dv_bf16_buf), value_delta.as<float>());
    }
#elif defined(OPENNN_HAS_CUDA)
    apply_delta_gpu_unfused(query, key, value,
                            attention_weights, attention_weights_dropped,
                            output_delta, attention_weight_delta,
                            query_delta, key_delta, value_delta);
#else
    apply_delta_cpu(query, key, value, attention_output,
                    attention_weights, attention_weights_dropped,
                    output_delta, attention_weight_delta,
                    query_delta, key_delta, value_delta);
#endif
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
