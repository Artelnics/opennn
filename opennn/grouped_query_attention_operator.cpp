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

// Parameter order (defines the byte layout in the network's .bin): q, k, v, o
// projections (bias-free, stored [out, in] like the HF checkpoint) then, when
// QK-Norm is enabled, the two norm weights of size head_dim.
vector<TensorSpec> GroupedQueryAttentionOperator::parameter_specs() const
{
    // Projections follow compute_dtype (bf16 mirror on a CUDA/BF16 build); the
    // QK-Norm weights stay FP32. The fp32 master (.bin) layout is unchanged.
    vector<TensorSpec> specs = {
        {Shape{q_dim(),  hidden},   compute_dtype},   // q_proj
        {Shape{kv_dim(), hidden},   compute_dtype},   // k_proj
        {Shape{kv_dim(), hidden},   compute_dtype},   // v_proj
        {Shape{hidden,   q_dim()},  compute_dtype},   // o_proj
    };

    if (use_qk_norm)
    {
        specs.push_back({Shape{head_dim}, Type::FP32});   // q_norm
        specs.push_back({Shape{head_dim}, Type::FP32});   // k_norm
    }

    return specs;
}

void GroupedQueryAttentionOperator::link_parameters(span<const TensorView> views)
{
    if (views.size() < 4) return;
    q_proj = views[0];
    k_proj = views[1];
    v_proj = views[2];
    o_proj = views[3];

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

void GroupedQueryAttentionOperator::set_parameters_random()
{
    // Pretrained weights are loaded afterwards; QK-Norm scales at 1 keep an
    // unloaded layer a no-op rather than a zero.
    if (q_norm.data) q_norm.as_vector().setOnes();
    if (k_norm.data) k_norm.as_vector().setOnes();
}

void GroupedQueryAttentionOperator::back_propagate(ForwardPropagation&, BackPropagation&, size_t) const
{
    // The base class no-op would silently kill the gradient flow mid-network.
    throw runtime_error("GroupedQueryAttention is inference-only: back-propagation is not implemented.");
}

void GroupedQueryAttentionOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool /*is_training*/)
{
    TensorView& input  = get_input(forward_propagation, layer);    // [batch, seq, hidden]
    TensorView& output = get_output(forward_propagation, layer);   // [batch, seq, hidden]

    const Index batch = forward_propagation.batch_size;

#ifdef OPENNN_HAS_CUDA
    if (input.is_cuda())
    {
        forward_gpu(input, output, batch, forward_propagation.past_length,
                    static_cast<const int*>(forward_propagation.position_device.data));
        return;
    }
#endif

    const Index seq   = input.shape[1];   // tokens this pass
    const Index qd    = q_dim();
    const Index kd    = kv_dim();
    const float scale = 1.0f / std::sqrt(float(head_dim));

    // Tables span the compiled window so decode can index positions past..past+seq-1.
    const Index table_len = sequence_length;
    std::vector<float> cos_tab(size_t(table_len) * head_dim), sin_tab(size_t(table_len) * head_dim);
    TensorView cos_v(cos_tab.data(), {table_len, head_dim}), sin_v(sin_tab.data(), {table_len, head_dim});
    rotary_build_tables(cos_v, sin_v, table_len, head_dim, rope_theta);

    float* x_all = input.as<float>();
    float* o_all = output.as<float>();

    if (batch == 1)   // incremental path (past == 0 prefill; past > 0 decode)
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

        std::vector<float> q(size_t(seq) * qd), k(size_t(seq) * kd), qr(size_t(seq) * qd), attn(size_t(seq) * qd);

        TensorView x_b(x_all, {1, seq, hidden});
        TensorView q_v(q.data(), {1, seq, qd}), k_v(k.data(), {1, seq, kd});
        TensorView v_slot(vcache + size_t(past) * kd, {1, seq, kd});   // raw v into cache
        TensorView k_slot(kcache + size_t(past) * kd, {1, seq, kd});   // post-RoPE k into cache

        tied_lm_head_forward(x_b, q_proj, q_v);
        tied_lm_head_forward(x_b, k_proj, k_v);
        tied_lm_head_forward(x_b, v_proj, v_slot);

        if (use_qk_norm)
        {
            qk_norm_forward(q_v, q_norm, q_v, head_dim, rms_epsilon);   // QK-Norm, before RoPE
            qk_norm_forward(k_v, k_norm, k_v, head_dim, rms_epsilon);
        }

        TensorView qr_v(qr.data(), {1, seq, qd});
        rotary_forward(q_v, cos_v, sin_v, qr_v,   head_dim, head_dim, past);
        rotary_forward(k_v, cos_v, sin_v, k_slot, head_dim, head_dim, past);

        TensorView key_all(kcache, {1, total, kd}), val_all(vcache, {1, total, kd});
        TensorView attn_v(attn.data(), {1, seq, qd});
        grouped_attention_forward(qr_v, key_all, val_all, attn_v, q_heads, kv_heads, head_dim, true, scale, past);

        TensorView o_b(o_all, {1, seq, hidden});
        tied_lm_head_forward(attn_v, o_proj, o_b);
        return;
    }

    throw_if(forward_propagation.past_length != 0,
             "GroupedQueryAttentionOperator: KV-cache decoding requires batch size 1.");

    std::vector<float> q(size_t(seq) * qd), k(size_t(seq) * kd), v(size_t(seq) * kd);
    std::vector<float> qr(size_t(seq) * qd), kr(size_t(seq) * kd), attn(size_t(seq) * qd);

    for (Index b = 0; b < batch; ++b)
    {
        TensorView x_b(x_all + size_t(b) * seq * hidden, {1, seq, hidden});
        TensorView q_v(q.data(), {1, seq, qd}), k_v(k.data(), {1, seq, kd}), v_v(v.data(), {1, seq, kd});

        tied_lm_head_forward(x_b, q_proj, q_v);
        tied_lm_head_forward(x_b, k_proj, k_v);
        tied_lm_head_forward(x_b, v_proj, v_v);

        if (use_qk_norm)
        {
            qk_norm_forward(q_v, q_norm, q_v, head_dim, rms_epsilon);   // QK-Norm, before RoPE
            qk_norm_forward(k_v, k_norm, k_v, head_dim, rms_epsilon);
        }

        TensorView qr_v(qr.data(), {1, seq, qd}), kr_v(kr.data(), {1, seq, kd});
        rotary_forward(q_v, cos_v, sin_v, qr_v, head_dim, head_dim, 0);
        rotary_forward(k_v, cos_v, sin_v, kr_v, head_dim, head_dim, 0);

        TensorView attn_v(attn.data(), {1, seq, qd});
        grouped_attention_forward(qr_v, kr_v, v_v, attn_v, q_heads, kv_heads, head_dim, true, scale, 0);

        TensorView o_b(o_all + size_t(b) * seq * hidden, {1, seq, hidden});
        tied_lm_head_forward(attn_v, o_proj, o_b);
    }
}

#ifdef OPENNN_HAS_CUDA

namespace
{

// Per-pass scratch, RoPE tables and decode buffers carry no state across
// operator invocations and the layers run serially, so every GQA layer shares
// one per-thread pool instead of 36 private copies (~1.1 GB on Qwen3-4B).
// Follows the device_backend shared-workspace precedent, with the same caveat:
// one GQA model shape per thread at a time — a rebuild moves the addresses a
// captured graph baked in. K/V caches stay per-layer (the only cross-pass state).
struct GroupedAttentionScratch
{
    Buffer cos{Device::CUDA}, sin{Device::CUDA};
    Buffer q{Device::CUDA}, k{Device::CUDA}, v{Device::CUDA};
    Buffer qr{Device::CUDA}, kr{Device::CUDA}, attn{Device::CUDA};
    Buffer qkv{Device::CUDA}, partials{Device::CUDA};
    Index sequence = -1;
    Index q_dim = 0, kv_dim = 0, head_dim = 0;
    float theta = 0.0f;
    Type dtype = Type::FP32;
};

GroupedAttentionScratch& gqa_scratch()
{
    thread_local GroupedAttentionScratch scratch;
    return scratch;
}

// cuDNN-frontend flash-attention graph for the prefill (seq > 1, BF16). One
// graph built at the compiled max dims with a padding mask: the actual query
// and key counts live in two device int32 scalars, and the bottom-right causal
// diagonal aligns to them, so suffix prefill (kv longer than q after prefix
// caching) masks correctly without per-shape rebuilds. A build failure falls
// back to the generic kernel for good.
struct GroupedAttentionSDPA
{
    shared_ptr<cudnn_frontend::graph::Graph> graph;
    shared_ptr<cudnn_frontend::graph::Tensor_attributes> Q, K, V, O, SeqQ, SeqKV;
    void* workspace = nullptr;
    int32_t* seq_device = nullptr;   // [seq_q, seq_kv]
    int32_t* seq_pinned = nullptr;
    Index max_seq = 0, q_heads = 0, kv_heads = 0, head_dim = 0;
    bool failed = false;

    ~GroupedAttentionSDPA()
    {
        device::deallocate(Device::CUDA, workspace, 0);
        device::deallocate(Device::CUDA, seq_device, 0);
        if (seq_pinned) device::deallocate_pinned_host(seq_pinned);
    }
};

GroupedAttentionSDPA& gqa_sdpa()
{
    thread_local GroupedAttentionSDPA sdpa;
    return sdpa;
}

// BSHD tensor ({batch 1, heads, max_seq, head_dim} over rows of heads*head_dim).
shared_ptr<cudnn_frontend::graph::Tensor_attributes>
gqa_bshd_tensor(cudnn_frontend::graph::Graph& graph, const char* name,
                int64_t heads, int64_t max_seq, int64_t head_dim)
{
    return graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim   ({1, heads, max_seq, head_dim})
                        .set_stride({heads * max_seq * head_dim, head_dim, heads * head_dim, 1}));
}

void gqa_sdpa_build(GroupedAttentionSDPA& s, Index max_seq, Index q_heads, Index kv_heads,
                    Index head_dim, float scale)
{
    auto graph = make_shared<cudnn_frontend::graph::Graph>();
    graph->set_io_data_type(cudnn_frontend::DataType_t::BFLOAT16)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    s.Q = gqa_bshd_tensor(*graph, "Q", q_heads,  max_seq, head_dim);
    s.K = gqa_bshd_tensor(*graph, "K", kv_heads, max_seq, head_dim);
    s.V = gqa_bshd_tensor(*graph, "V", kv_heads, max_seq, head_dim);

    auto seq_scalar = [&](const char* name) {
        return graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name(name).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1})
                             .set_data_type(cudnn_frontend::DataType_t::INT32));
    };
    s.SeqQ  = seq_scalar("SeqQ");
    s.SeqKV = seq_scalar("SeqKV");

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
      .set_dim   ({1, q_heads, max_seq, head_dim})
      .set_stride({q_heads * max_seq * head_dim, head_dim, q_heads * head_dim, 1});
    s.O = O;

    cudnnHandle_t handle = Backend::get_cudnn_handle();
    cudnn_frontend::check_status(graph->validate(), "gqa sdpa validate");
    cudnn_frontend::check_status(graph->build_operation_graph(handle), "gqa sdpa build_operation_graph");
    cudnn_frontend::check_status(graph->create_execution_plans({cudnn_frontend::HeurMode_t::A}), "gqa sdpa plans");
    cudnn_frontend::check_status(graph->build_plans(handle, cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_CHOICE), "gqa sdpa build_plans");

    int64_t workspace_bytes = 0;
    graph->get_workspace_size(workspace_bytes);
    device::deallocate(Device::CUDA, s.workspace, 0);
    s.workspace = workspace_bytes > 0 ? device::allocate(Device::CUDA, Index(workspace_bytes)) : nullptr;

    if (!s.seq_device) s.seq_device = static_cast<int32_t*>(device::allocate(Device::CUDA, Index(2 * sizeof(int32_t))));
    if (!s.seq_pinned) s.seq_pinned = static_cast<int32_t*>(device::allocate_pinned_host(Index(2 * sizeof(int32_t))));

    s.graph = move(graph);
    s.max_seq = max_seq; s.q_heads = q_heads; s.kv_heads = kv_heads; s.head_dim = head_dim;
}

}

void GroupedQueryAttentionOperator::forward_gpu(TensorView& input, TensorView& output, Index batch, Index past,
                                                const int* position_device)
{
    const Index seq = input.shape[1];   // tokens this pass
    const Index qd  = q_dim();
    const Index kd  = kv_dim();
    const float scale = 1.0f / std::sqrt(float(head_dim));
    cudaStream_t stream = device::get_compute_stream();

    // Activations follow the network dtype; cos/sin and QK-Norm weights stay FP32.
    const Type  act  = input.type;
    const Index elem = Index(type_bytes(act));

    // Shared scratch and tables sized to the compiled max seq; rebuilt only when
    // the shape, dtype or RoPE base changes.
    const Index table_len = sequence_length;
    auto& s = gqa_scratch();
    if (s.sequence != table_len || s.dtype != act || s.q_dim != qd || s.kv_dim != kd
        || s.head_dim != head_dim || s.theta != rope_theta)
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
        auto alloc = [&](Index n, Buffer& b) { b.resize_bytes(n * elem, Device::CUDA); };

        s.cos = upload(cos_h);   // fp32
        s.sin = upload(sin_h);   // fp32
        alloc(table_len * qd, s.q);   alloc(table_len * kd, s.k);   alloc(table_len * kd, s.v);
        alloc(table_len * qd, s.qr);  alloc(table_len * kd, s.kr);  alloc(table_len * qd, s.attn);
        alloc(qd + 2 * kd, s.qkv);   // single fused-projection row (decode only)
        s.partials.resize_bytes(grouped_attention_decode_scratch_floats(q_heads, head_dim)
                                * Index(sizeof(float)), Device::CUDA);
        s.sequence = table_len;
        s.q_dim = qd; s.kv_dim = kd; s.head_dim = head_dim;
        s.theta = rope_theta;
        s.dtype = act;
    }

    if (cache_capacity != table_len || cache_dtype != act || kv_key.device_type != Device::CUDA)
    {
        kv_key.resize_bytes(table_len * kd * elem, Device::CUDA);
        kv_value.resize_bytes(table_len * kd * elem, Device::CUDA);
        cache_capacity = table_len;
        cache_dtype = act;
    }

    TensorView cos_v(s.cos.data, {table_len, head_dim}, Type::FP32, Device::CUDA);
    TensorView sin_v(s.sin.data, {table_len, head_dim}, Type::FP32, Device::CUDA);

    if (batch == 1)   // incremental path (past == 0 prefill; past > 0 decode)
    {
        const Index total = past + seq;
        TensorView x_b(input.data,  {1, seq, hidden}, act, Device::CUDA);
        TensorView o_b(output.data, {1, seq, hidden}, act, Device::CUDA);
        TensorView q_v(s.q.data,  {1, seq, qd}, act, Device::CUDA);
        TensorView k_v(s.k.data,  {1, seq, kd}, act, Device::CUDA);
        TensorView qr_v(s.qr.data, {1, seq, qd}, act, Device::CUDA);
        TensorView attn_v(s.attn.data, {1, seq, qd}, act, Device::CUDA);

        // raw v and post-RoPE k written into the cache at the append offset.
        char* v_at = static_cast<char*>(kv_value.data) + size_t(past) * kd * elem;
        char* k_at = static_cast<char*>(kv_key.data)   + size_t(past) * kd * elem;
        TensorView v_slot(v_at, {1, seq, kd}, act, Device::CUDA);
        TensorView k_slot(k_at, {1, seq, kd}, act, Device::CUDA);

        if (seq == 1 && qkv_fused && position_device && use_qk_norm)
        {
            // Fully device-positioned decode step (CUDA-graph capturable): one
            // fused projection row, then a single kernel doing QK-Norm + RoPE +
            // cache append at *position_device, then split-KV attention whose
            // valid length also comes from the device position.
            TensorView qkv_row(s.qkv.data, {1, 1, qd + 2 * kd}, act, Device::CUDA);
            TensorView qkv_w(q_proj.data, {qd + 2 * kd, hidden}, q_proj.type, Device::CUDA);
            tied_lm_head_forward(x_b, qkv_w, qkv_row);

            TensorView key_cache(kv_key.data,   {1, table_len, kd}, act, Device::CUDA);
            TensorView val_cache(kv_value.data, {1, table_len, kd}, act, Device::CUDA);
            qk_rope_cache_append(qkv_row, q_norm, k_norm, cos_v, sin_v, qr_v, key_cache, val_cache,
                                 q_heads, kv_heads, head_dim, rms_epsilon, position_device);

            grouped_attention_forward(qr_v, key_cache, val_cache, attn_v, q_heads, kv_heads, head_dim,
                                      true, scale, past,
                                      static_cast<float*>(s.partials.data), position_device);

            tied_lm_head_forward(attn_v, o_proj, o_b);
            return;
        }

        if (seq == 1 && qkv_fused)
        {
            // One projection for the whole [q | k | v] row; the single-row case
            // keeps q/k/v contiguous, so the per-piece views below just offset it.
            TensorView qkv_row(s.qkv.data, {1, 1, qd + 2 * kd}, act, Device::CUDA);
            TensorView qkv_w(q_proj.data, {qd + 2 * kd, hidden}, q_proj.type, Device::CUDA);
            tied_lm_head_forward(x_b, qkv_w, qkv_row);

            q_v = TensorView(s.qkv.data, {1, 1, qd}, act, Device::CUDA);
            k_v = TensorView(static_cast<char*>(s.qkv.data) + size_t(qd) * elem, {1, 1, kd}, act, Device::CUDA);
            device::copy_async(v_at, static_cast<char*>(s.qkv.data) + size_t(qd + kd) * elem,
                               kd * elem, device::CopyKind::DeviceToDevice, stream);
        }
        else
        {
            tied_lm_head_forward(x_b, q_proj, q_v);
            tied_lm_head_forward(x_b, k_proj, k_v);
            tied_lm_head_forward(x_b, v_proj, v_slot);
        }

        if (use_qk_norm)
        {
            qk_norm_forward(q_v, q_norm, q_v, head_dim, rms_epsilon);   // QK-Norm, before RoPE
            qk_norm_forward(k_v, k_norm, k_v, head_dim, rms_epsilon);
        }

        rotary_forward(q_v, cos_v, sin_v, qr_v,   head_dim, head_dim, past);
        rotary_forward(k_v, cos_v, sin_v, k_slot, head_dim, head_dim, past);

        auto& sdpa = gqa_sdpa();
        if (seq > 1 && act == Type::BF16 && !sdpa.failed)
        {
            if (!sdpa.graph || sdpa.max_seq != table_len || sdpa.q_heads != q_heads
                || sdpa.kv_heads != kv_heads || sdpa.head_dim != head_dim)
            {
                try { gqa_sdpa_build(sdpa, table_len, q_heads, kv_heads, head_dim, scale); }
                catch (const exception& e)
                {
                    sdpa.failed = true;
                    cerr << "GroupedQueryAttention: cuDNN flash-attention prefill unavailable ("
                         << e.what() << "); using the generic kernel.\n";
                }
            }

            if (!sdpa.failed)
            {
                sdpa.seq_pinned[0] = int32_t(seq);
                sdpa.seq_pinned[1] = int32_t(total);
                device::copy_async(sdpa.seq_device, sdpa.seq_pinned, Index(2 * sizeof(int32_t)),
                                   device::CopyKind::HostToDevice, stream);

                std::unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensors;
                tensors[sdpa.Q]     = s.qr.data;
                tensors[sdpa.K]     = kv_key.data;
                tensors[sdpa.V]     = kv_value.data;
                tensors[sdpa.O]     = s.attn.data;
                tensors[sdpa.SeqQ]  = sdpa.seq_device;
                tensors[sdpa.SeqKV] = sdpa.seq_device + 1;
                cudnn_frontend::check_status(
                    sdpa.graph->execute(Backend::get_cudnn_handle(), tensors, sdpa.workspace),
                    "gqa sdpa execute");

                tied_lm_head_forward(attn_v, o_proj, o_b);
                return;
            }
        }

        TensorView key_all(kv_key.data,   {1, total, kd}, act, Device::CUDA);
        TensorView val_all(kv_value.data, {1, total, kd}, act, Device::CUDA);
        grouped_attention_forward(qr_v, key_all, val_all, attn_v, q_heads, kv_heads, head_dim, true, scale, past,
                                  static_cast<float*>(s.partials.data));

        tied_lm_head_forward(attn_v, o_proj, o_b);
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

        tied_lm_head_forward(x_b, q_proj, q_v);
        tied_lm_head_forward(x_b, k_proj, k_v);
        tied_lm_head_forward(x_b, v_proj, v_v);

        if (use_qk_norm)
        {
            qk_norm_forward(q_v, q_norm, q_v, head_dim, rms_epsilon);   // QK-Norm, before RoPE
            qk_norm_forward(k_v, k_norm, k_v, head_dim, rms_epsilon);
        }

        rotary_forward(q_v, cos_v, sin_v, qr_v, head_dim, head_dim, 0);
        rotary_forward(k_v, cos_v, sin_v, kr_v, head_dim, head_dim, 0);

        grouped_attention_forward(qr_v, kr_v, v_v, attn_v, q_heads, kv_heads, head_dim, true, scale, 0);

        tied_lm_head_forward(attn_v, o_proj, o_b);
    }
}

#endif  // OPENNN_HAS_CUDA

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
