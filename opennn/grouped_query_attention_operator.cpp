//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O U P E D   Q U E R Y   A T T E N T I O N   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cmath>
#include <cstring>
#include <vector>

#include "grouped_query_attention_operator.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#ifdef OPENNN_HAS_CUDA
#include "device_backend.h"
#include "cudnn_frontend_utilities.h"
#endif

namespace opennn
{

void GroupedQueryAttentionOperator::set(Index new_sequence_length, Index new_hidden,
                                        Index new_q_heads, Index new_kv_heads, Index new_head_dim,
                                        float new_rope_theta, float new_rms_epsilon, bool new_use_qk_norm)
{
    sequence_length = new_sequence_length;
    hidden          = new_hidden;
    q_heads         = new_q_heads;
    kv_heads        = new_kv_heads;
    head_dim        = new_head_dim;
    rope_theta      = new_rope_theta;
    rms_epsilon     = new_rms_epsilon;
    use_qk_norm     = new_use_qk_norm;

}

vector<TensorSpec> GroupedQueryAttentionOperator::parameter_specs() const
{

    vector<TensorSpec> specs = {
        {Shape{q_dim(),  hidden},   weights_dtype},
        {Shape{kv_dim(), hidden},   weights_dtype},
        {Shape{kv_dim(), hidden},   weights_dtype},
        {Shape{hidden,   q_dim()},  weights_dtype},
    };

    if (use_qk_norm)
    {
        specs.push_back({Shape{head_dim}, Type::FP32});
        specs.push_back({Shape{head_dim}, Type::FP32});
    }

    return specs;
}

vector<Operator::SlotQuantization> GroupedQueryAttentionOperator::parameter_quantization() const
{
    return {{q_dim(), 0}, {kv_dim(), 0}, {kv_dim(), 0}, {hidden, 0}};
}

void GroupedQueryAttentionOperator::link_parameters(span<const TensorView> views)
{
    if (!link_views(views, {&q_proj, &k_proj, &v_proj, &o_proj})) return;

    const Index elem = Index(type_bytes(q_proj.type));
    qkv_fused = q_proj.type == k_proj.type && k_proj.type == v_proj.type
        && static_cast<const char*>(k_proj.data) == static_cast<const char*>(q_proj.data) + q_proj.size() * elem
        && static_cast<const char*>(v_proj.data) == static_cast<const char*>(k_proj.data) + k_proj.size() * elem;

    if (use_qk_norm && views.size() >= 6)
    {
        q_norm = views[4];
        k_norm = views[5];
    }
    else
    {
        q_norm = {};
        k_norm = {};
    }
}

void GroupedQueryAttentionOperator::link_parameter_scales(span<const TensorView> views)
{
    if (views.size() < 4) return;
    q_scale = views[0];
    k_scale = views[1];
    v_scale = views[2];
    o_scale = views[3];

    const bool scales_fused = q_scale.data && k_scale.data && v_scale.data
        && k_scale.as<const float>() == q_scale.as<const float>() + q_scale.size()
        && v_scale.as<const float>() == k_scale.as<const float>() + k_scale.size();

    qkv_scale = scales_fused
        ? TensorView(q_scale.data, Shape{q_dim() + 2 * kv_dim()}, Type::FP32, q_scale.device)
        : TensorView{};

    // The fused single-token decode needs one contiguous scale vector.
    if (q_proj.is_int8() && !scales_fused)
        qkv_fused = false;
}

void GroupedQueryAttentionOperator::set_parameters_random()
{

    if (q_norm.data) q_norm.as_vector().setOnes();
    if (k_norm.data) k_norm.as_vector().setOnes();
}

void GroupedQueryAttentionOperator::back_propagate(ForwardPropagation&, BackPropagation&, size_t) const
{

    throw runtime_error("GroupedQueryAttention is inference-only: back-propagation is not implemented.");
}

namespace
{

struct GroupedAttentionCpuScratch
{
    std::vector<float> cos, sin;
    std::vector<float> q, k, v, qr, kr, attn;
    Index table_len = -1, head_dim = 0;
    float theta = 0.0f;

    void build_tables(Index new_table_len, Index new_head_dim, float new_theta)
    {
        if (table_len == new_table_len && head_dim == new_head_dim && theta == new_theta) return;
        cos.resize(size_t(new_table_len) * new_head_dim);
        sin.resize(size_t(new_table_len) * new_head_dim);
        TensorView cos_v(cos.data(), {new_table_len, new_head_dim});
        TensorView sin_v(sin.data(), {new_table_len, new_head_dim});
        rotary_build_tables(cos_v, sin_v, new_table_len, new_head_dim, new_theta);
        table_len = new_table_len; head_dim = new_head_dim; theta = new_theta;
    }
};

GroupedAttentionCpuScratch& gqa_cpu_scratch()
{
    thread_local GroupedAttentionCpuScratch scratch;
    return scratch;
}

float* grown(std::vector<float>& buffer, size_t n)
{
    if (buffer.size() < n) buffer.resize(n);
    return buffer.data();
}

}

void GroupedQueryAttentionOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool  )
{
    TensorView& input  = get_input(forward_propagation, layer);
    TensorView& output = get_output(forward_propagation, layer);

    const Index batch = forward_propagation.batch_size;

#ifdef OPENNN_HAS_CUDA
    if (input.is_cuda())
    {
        forward_gpu(input, output, batch, forward_propagation.past_length,
                    forward_propagation.get_sequence_capacity(),
                    static_cast<const int*>(forward_propagation.position_device.data));
        return;
    }
#endif

    const Index seq   = input.shape[1];
    const Index qd    = q_dim();
    const Index kd    = kv_dim();
    const float scale = 1.0f / std::sqrt(float(head_dim));

    const Index table_len = sequence_length;
    throw_if(seq < 1 || forward_propagation.past_length < 0
             || forward_propagation.past_length + seq > table_len,
             "GroupedQueryAttentionOperator: query [{}, {}) exceeds the "
             "{}-token KV cache.",
             forward_propagation.past_length,
             forward_propagation.past_length + seq, table_len);
    auto& scratch = gqa_cpu_scratch();
    scratch.build_tables(table_len, head_dim, rope_theta);
    TensorView cos_v(scratch.cos.data(), {table_len, head_dim}), sin_v(scratch.sin.data(), {table_len, head_dim});

    float* x_all = input.as<float>();
    float* o_all = output.as<float>();

    if (batch == 1)
    {
        const Index past  = forward_propagation.past_length;
        const Index total = past + seq;

        const Index capacity_bytes = table_len * kd * Index(sizeof(float));
        if (cache_capacity != table_len || kv_key.device_type != Device::CPU)
        {
            kv_key.resize_bytes(capacity_bytes, Device::CPU);
            kv_value.resize_bytes(capacity_bytes, Device::CPU);
            cache_capacity = table_len;
        }
        float* kcache = kv_key.as<float>();
        float* vcache = kv_value.as<float>();

        float* q    = grown(scratch.q,    size_t(seq) * qd);
        float* k    = grown(scratch.k,    size_t(seq) * kd);
        float* qr   = grown(scratch.qr,   size_t(seq) * qd);
        float* attn = grown(scratch.attn, size_t(seq) * qd);

        TensorView x_b(x_all, {1, seq, hidden});
        TensorView q_v(q, {1, seq, qd}), k_v(k, {1, seq, kd});
        TensorView v_slot(vcache + size_t(past) * kd, {1, seq, kd});
        TensorView k_slot(kcache + size_t(past) * kd, {1, seq, kd});

        tied_lm_head_forward(x_b, q_proj, q_v);
        tied_lm_head_forward(x_b, k_proj, k_v);
        tied_lm_head_forward(x_b, v_proj, v_slot);

        if (use_qk_norm)
        {
            qk_norm_forward(q_v, q_norm, q_v, head_dim, rms_epsilon);
            qk_norm_forward(k_v, k_norm, k_v, head_dim, rms_epsilon);
        }

        TensorView qr_v(qr, {1, seq, qd});
        rotary_forward(q_v, cos_v, sin_v, qr_v,   head_dim, head_dim, past);
        rotary_forward(k_v, cos_v, sin_v, k_slot, head_dim, head_dim, past);

        TensorView key_all(kcache, {1, total, kd}), val_all(vcache, {1, total, kd});
        TensorView attn_v(attn, {1, seq, qd});
        grouped_attention_forward(qr_v, key_all, val_all, attn_v, q_heads, kv_heads, head_dim, true, scale, past);

        TensorView o_b(o_all, {1, seq, hidden});
        tied_lm_head_forward(attn_v, o_proj, o_b);
        return;
    }

    throw_if(forward_propagation.past_length != 0,
             "GroupedQueryAttentionOperator: KV-cache decoding requires batch size 1.");

    float* q    = grown(scratch.q,    size_t(seq) * qd);
    float* k    = grown(scratch.k,    size_t(seq) * kd);
    float* v    = grown(scratch.v,    size_t(seq) * kd);
    float* qr   = grown(scratch.qr,   size_t(seq) * qd);
    float* kr   = grown(scratch.kr,   size_t(seq) * kd);
    float* attn = grown(scratch.attn, size_t(seq) * qd);

    for (Index b = 0; b < batch; ++b)
    {
        TensorView x_b(x_all + size_t(b) * seq * hidden, {1, seq, hidden});
        TensorView q_v(q, {1, seq, qd}), k_v(k, {1, seq, kd}), v_v(v, {1, seq, kd});

        tied_lm_head_forward(x_b, q_proj, q_v);
        tied_lm_head_forward(x_b, k_proj, k_v);
        tied_lm_head_forward(x_b, v_proj, v_v);

        if (use_qk_norm)
        {
            qk_norm_forward(q_v, q_norm, q_v, head_dim, rms_epsilon);
            qk_norm_forward(k_v, k_norm, k_v, head_dim, rms_epsilon);
        }

        TensorView qr_v(qr, {1, seq, qd}), kr_v(kr, {1, seq, kd});
        rotary_forward(q_v, cos_v, sin_v, qr_v, head_dim, head_dim, 0);
        rotary_forward(k_v, cos_v, sin_v, kr_v, head_dim, head_dim, 0);

        TensorView attn_v(attn, {1, seq, qd});
        grouped_attention_forward(qr_v, kr_v, v_v, attn_v, q_heads, kv_heads, head_dim, true, scale, 0);

        TensorView o_b(o_all + size_t(b) * seq * hidden, {1, seq, hidden});
        tied_lm_head_forward(attn_v, o_proj, o_b);
    }
}

#ifdef OPENNN_HAS_CUDA

namespace
{

struct GroupedAttentionScratch
{
    Buffer cos{Device::CUDA}, sin{Device::CUDA};
    Buffer q{Device::CUDA}, k{Device::CUDA}, v{Device::CUDA};
    Buffer qr{Device::CUDA}, kr{Device::CUDA}, attn{Device::CUDA};
    Buffer qkv{Device::CUDA}, partials{Device::CUDA};
    Index sequence = -1;
    Index query_capacity = 0;
    Index q_dim = 0, kv_dim = 0, head_dim = 0;
    float theta = 0.0f;
    Type dtype = Type::FP32;
};

// One scratch per attention geometry: networks with different dimensions
// (e.g. a speculative draft next to its main model) alternate on the same
// thread, and a single shared scratch would be reallocated on every switch —
// invalidating the pointers a captured CUDA graph holds.
GroupedAttentionScratch& gqa_scratch(Index sequence, Index q_dim, Index kv_dim,
                                     Index head_dim, float theta, Type dtype)
{
    thread_local map<tuple<Index, Index, Index, Index, float, int>,
                     GroupedAttentionScratch> scratches;
    return scratches[{sequence, q_dim, kv_dim, head_dim, theta, int(dtype)}];
}

struct GroupedAttentionSDPA
{
    shared_ptr<cudnn_frontend::graph::Graph> graph;
    shared_ptr<cudnn_frontend::graph::Tensor_attributes> Q, K, V, O, SeqQ, SeqKV;
    unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensors;
    void* workspace = nullptr;
    int32_t* seq_device = nullptr;
    int32_t* seq_pinned = nullptr;
    Index max_q = 0, max_kv = 0;
    Index q_heads = 0, kv_heads = 0, head_dim = 0;
    bool failed = false;

    ~GroupedAttentionSDPA()
    {
        device::deallocate(Device::CUDA, workspace, 0);
        device::deallocate(Device::CUDA, seq_device, 0);
        if (seq_pinned) device::deallocate_pinned_host(seq_pinned);
    }
};

GroupedAttentionSDPA& gqa_sdpa(Index max_q, Index max_kv,
                               Index q_heads, Index kv_heads, Index head_dim)
{
    thread_local map<tuple<Index, Index, Index, Index, Index>,
                     GroupedAttentionSDPA> graphs;
    return graphs[{max_q, max_kv, q_heads, kv_heads, head_dim}];
}

shared_ptr<cudnn_frontend::graph::Tensor_attributes>
gqa_bshd_tensor(cudnn_frontend::graph::Graph& graph, const char* name,
                int64_t heads, int64_t max_seq, int64_t head_dim)
{
    return graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim   ({1, heads, max_seq, head_dim})
                        .set_stride({heads * max_seq * head_dim, head_dim, heads * head_dim, 1}));
}

void gqa_sdpa_build(GroupedAttentionSDPA& s, Index max_q, Index max_kv,
                    Index q_heads, Index kv_heads, Index head_dim, float scale)
{
    auto graph = cudnn_frontend::new_graph(Type::BF16);

    s.Q = gqa_bshd_tensor(*graph, "Q", q_heads,  max_q,  head_dim);
    s.K = gqa_bshd_tensor(*graph, "K", kv_heads, max_kv, head_dim);
    s.V = gqa_bshd_tensor(*graph, "V", kv_heads, max_kv, head_dim);

    s.SeqQ  = cudnn_frontend::seq_len_scalar(*graph, "SeqQ");
    s.SeqKV = cudnn_frontend::seq_len_scalar(*graph, "SeqKV");

    auto options = cudnn_frontend::graph::SDPA_attributes()
                   .set_name("gqa_prefill")
                   .set_is_inference(true)
                   .set_padding_mask(true)
                   .set_seq_len_q(s.SeqQ)
                   .set_seq_len_kv(s.SeqKV)
                   .set_causal_mask_bottom_right(true)
                   .set_attn_scale(scale);

    auto [O, stats] = graph->sdpa(s.Q, s.K, s.V, options);
    (void)stats;
    O->set_output(true)
      .set_dim   ({1, q_heads, max_q, head_dim})
      .set_stride({q_heads * max_q * head_dim, head_dim, q_heads * head_dim, 1});
    s.O = O;

    cudnn_frontend::finalize_attention(*graph, "gqa sdpa");

    int64_t workspace_bytes = 0;
    graph->get_workspace_size(workspace_bytes);
    device::deallocate(Device::CUDA, s.workspace, 0);
    s.workspace = workspace_bytes > 0 ? device::allocate(Device::CUDA, Index(workspace_bytes)) : nullptr;

    if (!s.seq_device) s.seq_device = static_cast<int32_t*>(device::allocate(Device::CUDA, Index(2 * sizeof(int32_t))));
    if (!s.seq_pinned) s.seq_pinned = static_cast<int32_t*>(device::allocate_pinned_host(Index(2 * sizeof(int32_t))));

    s.graph = move(graph);
    s.tensors.clear();
    s.tensors.reserve(6);
    s.max_q = max_q;
    s.max_kv = max_kv;
    s.q_heads = q_heads;
    s.kv_heads = kv_heads;
    s.head_dim = head_dim;
}

}

void GroupedQueryAttentionOperator::forward_gpu(TensorView& input, TensorView& output, Index batch, Index past,
                                                Index query_capacity,
                                                const int* position_device)
{
    const Index seq = input.shape[1];
    const Index qd  = q_dim();
    const Index kd  = kv_dim();
    const float scale = 1.0f / std::sqrt(float(head_dim));
    cudaStream_t stream = device::get_compute_stream();

    const Type  act  = input.type;
    const Index elem = Index(type_bytes(act));

    const Index table_len = sequence_length;
    throw_if(seq < 1 || query_capacity < seq,
             "GroupedQueryAttentionOperator: query length {} exceeds its "
             "temporary capacity {}.", seq, query_capacity);
    throw_if(past < 0 || past + seq > table_len,
             "GroupedQueryAttentionOperator: query [{}, {}) exceeds the "
             "{}-token KV cache.", past, past + seq, table_len);
    auto& s = gqa_scratch(table_len, qd, kd, head_dim, rope_theta, act);
    {
        const bool geometry_changed =
            s.sequence != table_len || s.dtype != act
            || s.q_dim != qd || s.kv_dim != kd
            || s.head_dim != head_dim || s.theta != rope_theta;
        if (geometry_changed)
        {
            std::vector<float> cos_h(size_t(table_len) * head_dim), sin_h(size_t(table_len) * head_dim);
            { TensorView cv(cos_h.data(), {table_len, head_dim}), sv(sin_h.data(), {table_len, head_dim});
              rotary_build_tables(cv, sv, table_len, head_dim, rope_theta); }

            auto upload = [&](const std::vector<float>& host) {
                Buffer b(Device::CPU);
                b.resize_bytes(Index(host.size()) * Index(sizeof(float)), Device::CPU);
                std::memcpy(b.data, host.data(), host.size() * sizeof(float));
                b.migrate_to(Device::CUDA, stream);
                return b;
            };
            s.cos = upload(cos_h);
            s.sin = upload(sin_h);
            s.query_capacity = 0;
            s.partials.resize_bytes(grouped_attention_decode_scratch_floats(q_heads, head_dim)
                                    * Index(sizeof(float)), Device::CUDA);
            s.sequence = table_len;
            s.q_dim = qd; s.kv_dim = kd; s.head_dim = head_dim;
            s.theta = rope_theta;
            s.dtype = act;
        }

        if (s.query_capacity < query_capacity)
        {
            const auto grow = [&](Index n, Buffer& b)
            {
                b.grow_to(n * elem);
            };
            grow(query_capacity * qd, s.q);
            grow(query_capacity * kd, s.k);
            grow(query_capacity * kd, s.v);
            grow(query_capacity * qd, s.qr);
            grow(query_capacity * kd, s.kr);
            grow(query_capacity * qd, s.attn);
            grow(qd + 2 * kd, s.qkv);
            s.query_capacity = query_capacity;
        }

        if (cache_capacity != table_len || cache_dtype != act || kv_key.device_type != Device::CUDA)
        {
            kv_key.resize_bytes(table_len * kd * elem, Device::CUDA);
            kv_value.resize_bytes(table_len * kd * elem, Device::CUDA);
            cache_capacity = table_len;
            cache_dtype = act;
        }
    }

    TensorView cos_v(s.cos.data, {table_len, head_dim}, Type::FP32, Device::CUDA);
    TensorView sin_v(s.sin.data, {table_len, head_dim}, Type::FP32, Device::CUDA);

    if (batch == 1)
    {
        const Index total = past + seq;
        TensorView x_b(input.data,  {1, seq, hidden}, act, Device::CUDA);
        TensorView o_b(output.data, {1, seq, hidden}, act, Device::CUDA);
        TensorView q_v(s.q.data,  {1, seq, qd}, act, Device::CUDA);
        TensorView k_v(s.k.data,  {1, seq, kd}, act, Device::CUDA);
        TensorView qr_v(s.qr.data, {1, seq, qd}, act, Device::CUDA);
        TensorView attn_v(s.attn.data, {1, seq, qd}, act, Device::CUDA);

        char* v_at = static_cast<char*>(kv_value.data) + size_t(past) * kd * elem;
        char* k_at = static_cast<char*>(kv_key.data)   + size_t(past) * kd * elem;
        TensorView v_slot(v_at, {1, seq, kd}, act, Device::CUDA);
        TensorView k_slot(k_at, {1, seq, kd}, act, Device::CUDA);

        if (seq == 1 && qkv_fused && position_device && use_qk_norm)
        {
            TensorView qkv_row(s.qkv.data, {1, 1, qd + 2 * kd}, act, Device::CUDA);
            {
                TensorView qkv_w(q_proj.data, {qd + 2 * kd, hidden}, q_proj.type, Device::CUDA);
                tied_lm_head_forward(x_b, qkv_w, qkv_row, qkv_scale);
            }

            TensorView key_cache(kv_key.data,   {1, table_len, kd}, act, Device::CUDA);
            TensorView val_cache(kv_value.data, {1, table_len, kd}, act, Device::CUDA);
            {
                qk_rope_cache_append(qkv_row, q_norm, k_norm, cos_v, sin_v, qr_v, key_cache, val_cache,
                                     q_heads, kv_heads, head_dim, rms_epsilon, position_device);
            }
            {
                grouped_attention_forward(qr_v, key_cache, val_cache, attn_v, q_heads, kv_heads, head_dim,
                                          true, scale, past,
                                          static_cast<float*>(s.partials.data), position_device);
            }
            {
                tied_lm_head_forward(attn_v, o_proj, o_b, o_scale);
            }
            return;
        }

        if (seq == 1 && qkv_fused)
        {
            TensorView qkv_row(s.qkv.data, {1, 1, qd + 2 * kd}, act, Device::CUDA);
            TensorView qkv_w(q_proj.data, {qd + 2 * kd, hidden}, q_proj.type, Device::CUDA);
            tied_lm_head_forward(x_b, qkv_w, qkv_row, qkv_scale);
            q_v = TensorView(s.qkv.data, {1, 1, qd}, act, Device::CUDA);
            k_v = TensorView(static_cast<char*>(s.qkv.data) + size_t(qd) * elem, {1, 1, kd}, act, Device::CUDA);
            device::copy_async(v_at, static_cast<char*>(s.qkv.data) + size_t(qd + kd) * elem,
                               kd * elem, device::CopyKind::DeviceToDevice, stream);
        }
        else
        {
            tied_lm_head_forward(x_b, q_proj, q_v, q_scale);
            tied_lm_head_forward(x_b, k_proj, k_v, k_scale);
            tied_lm_head_forward(x_b, v_proj, v_slot, v_scale);
        }

        {
            if (use_qk_norm)
            {
                qk_norm_forward(q_v, q_norm, q_v, head_dim, rms_epsilon);
                qk_norm_forward(k_v, k_norm, k_v, head_dim, rms_epsilon);
            }

            rotary_forward(q_v, cos_v, sin_v, qr_v,   head_dim, head_dim, past);
            rotary_forward(k_v, cos_v, sin_v, k_slot, head_dim, head_dim, past);
        }

        auto& sdpa = gqa_sdpa(query_capacity, table_len,
                              q_heads, kv_heads, head_dim);
        if (seq > 1 && act == Type::BF16 && !sdpa.failed)
        {
            if (!sdpa.graph || sdpa.max_q != query_capacity
                || sdpa.max_kv != table_len || sdpa.q_heads != q_heads
                || sdpa.kv_heads != kv_heads || sdpa.head_dim != head_dim)
            {
                try
                {
                    gqa_sdpa_build(sdpa, query_capacity, table_len,
                                   q_heads, kv_heads, head_dim, scale);
                }
                catch (const exception& e)
                {
                    sdpa.failed = true;
                    cerr << "GroupedQueryAttention: cuDNN flash-attention prefill unavailable ("
                         << e.what() << "); using the generic kernel.\n";
                }
            }

            if (!sdpa.failed)
            {
                {
                    sdpa.seq_pinned[0] = int32_t(seq);
                    sdpa.seq_pinned[1] = int32_t(total);
                    device::copy_async(sdpa.seq_device, sdpa.seq_pinned, Index(2 * sizeof(int32_t)),
                                       device::CopyKind::HostToDevice, stream);

                    sdpa.tensors[sdpa.Q]     = s.qr.data;
                    sdpa.tensors[sdpa.K]     = kv_key.data;
                    sdpa.tensors[sdpa.V]     = kv_value.data;
                    sdpa.tensors[sdpa.O]     = s.attn.data;
                    sdpa.tensors[sdpa.SeqQ]  = sdpa.seq_device;
                    sdpa.tensors[sdpa.SeqKV] = sdpa.seq_device + 1;
                    cudnn_frontend::check_status(
                        sdpa.graph->execute(Backend::get_cudnn_handle(), sdpa.tensors, sdpa.workspace),
                        "gqa sdpa execute");
                }
                {
                    tied_lm_head_forward(attn_v, o_proj, o_b, o_scale);
                }
                return;
            }
        }

        TensorView key_all(kv_key.data,   {1, total, kd}, act, Device::CUDA);
        TensorView val_all(kv_value.data, {1, total, kd}, act, Device::CUDA);
        {
            grouped_attention_forward(qr_v, key_all, val_all, attn_v, q_heads, kv_heads, head_dim, true, scale, past,
                                      static_cast<float*>(s.partials.data));
        }
        {
            tied_lm_head_forward(attn_v, o_proj, o_b, o_scale);
        }
        return;
    }

    throw_if(past != 0, "GroupedQueryAttentionOperator: KV-cache decoding requires batch size 1.");

    TensorView q_v (s.q.data,    {1, seq, qd}, act, Device::CUDA);
    TensorView k_v (s.k.data,    {1, seq, kd}, act, Device::CUDA);
    TensorView v_v (s.v.data,    {1, seq, kd}, act, Device::CUDA);
    TensorView qr_v(s.qr.data,   {1, seq, qd}, act, Device::CUDA);
    TensorView kr_v(s.kr.data,   {1, seq, kd}, act, Device::CUDA);
    TensorView attn_v(s.attn.data, {1, seq, qd}, act, Device::CUDA);

    for (Index b = 0; b < batch; ++b)
    {
        char* in_b  = static_cast<char*>(input.data)  + size_t(b) * seq * hidden * elem;
        char* out_b = static_cast<char*>(output.data) + size_t(b) * seq * hidden * elem;
        TensorView x_b(in_b,  {1, seq, hidden}, act, Device::CUDA);
        TensorView o_b(out_b, {1, seq, hidden}, act, Device::CUDA);

        tied_lm_head_forward(x_b, q_proj, q_v, q_scale);
        tied_lm_head_forward(x_b, k_proj, k_v, k_scale);
        tied_lm_head_forward(x_b, v_proj, v_v, v_scale);

        if (use_qk_norm)
        {
            qk_norm_forward(q_v, q_norm, q_v, head_dim, rms_epsilon);
            qk_norm_forward(k_v, k_norm, k_v, head_dim, rms_epsilon);
        }

        rotary_forward(q_v, cos_v, sin_v, qr_v, head_dim, head_dim, 0);
        rotary_forward(k_v, cos_v, sin_v, kr_v, head_dim, head_dim, 0);

        grouped_attention_forward(qr_v, kr_v, v_v, attn_v, q_heads, kv_heads, head_dim, true, scale, 0);

        tied_lm_head_forward(attn_v, o_proj, o_b, o_scale);
    }
}

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
