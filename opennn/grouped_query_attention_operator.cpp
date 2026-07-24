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

    // Scratch, tables and cache sized to the compiled max seq; rebuilt only when
    // that length or the dtype changes.
    const Index table_len = sequence_length;
    if (gpu_sequence != table_len || gpu_dtype != act)
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

        d_cos = upload(cos_h);   // fp32
        d_sin = upload(sin_h);   // fp32
        alloc(table_len * qd, d_q);   alloc(table_len * kd, d_k);   alloc(table_len * kd, d_v);
        alloc(table_len * qd, d_qr);  alloc(table_len * kd, d_kr);  alloc(table_len * qd, d_attn);
        alloc(table_len * kd, kv_key);  alloc(table_len * kd, kv_value);
        alloc(qd + 2 * kd, d_qkv);   // single fused-projection row (decode only)
        d_attn_partials.resize_bytes(grouped_attention_decode_scratch_floats(q_heads, head_dim)
                                     * Index(sizeof(float)), Device::CUDA);
        gpu_sequence = table_len;
        gpu_dtype = act;
        cache_capacity = table_len;
        cache_dtype = act;
    }

    TensorView cos_v(d_cos.data, {table_len, head_dim}, Type::FP32, Device::CUDA);
    TensorView sin_v(d_sin.data, {table_len, head_dim}, Type::FP32, Device::CUDA);

    if (batch == 1)   // incremental path (past == 0 prefill; past > 0 decode)
    {
        const Index total = past + seq;
        TensorView x_b(input.data,  {1, seq, hidden}, act, Device::CUDA);
        TensorView o_b(output.data, {1, seq, hidden}, act, Device::CUDA);
        TensorView q_v(d_q.data,  {1, seq, qd}, act, Device::CUDA);
        TensorView k_v(d_k.data,  {1, seq, kd}, act, Device::CUDA);
        TensorView qr_v(d_qr.data, {1, seq, qd}, act, Device::CUDA);
        TensorView attn_v(d_attn.data, {1, seq, qd}, act, Device::CUDA);

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
            TensorView qkv_row(d_qkv.data, {1, 1, qd + 2 * kd}, act, Device::CUDA);
            TensorView qkv_w(q_proj.data, {qd + 2 * kd, hidden}, q_proj.type, Device::CUDA);
            tied_lm_head_forward(x_b, qkv_w, qkv_row);

            TensorView key_cache(kv_key.data,   {1, table_len, kd}, act, Device::CUDA);
            TensorView val_cache(kv_value.data, {1, table_len, kd}, act, Device::CUDA);
            qk_rope_cache_append(qkv_row, q_norm, k_norm, cos_v, sin_v, qr_v, key_cache, val_cache,
                                 q_heads, kv_heads, head_dim, rms_epsilon, position_device);

            grouped_attention_forward(qr_v, key_cache, val_cache, attn_v, q_heads, kv_heads, head_dim,
                                      true, scale, past,
                                      static_cast<float*>(d_attn_partials.data), position_device);

            tied_lm_head_forward(attn_v, o_proj, o_b);
            return;
        }

        if (seq == 1 && qkv_fused)
        {
            // One projection for the whole [q | k | v] row; the single-row case
            // keeps q/k/v contiguous, so the per-piece views below just offset it.
            TensorView qkv_row(d_qkv.data, {1, 1, qd + 2 * kd}, act, Device::CUDA);
            TensorView qkv_w(q_proj.data, {qd + 2 * kd, hidden}, q_proj.type, Device::CUDA);
            tied_lm_head_forward(x_b, qkv_w, qkv_row);

            q_v = TensorView(d_qkv.data, {1, 1, qd}, act, Device::CUDA);
            k_v = TensorView(static_cast<char*>(d_qkv.data) + size_t(qd) * elem, {1, 1, kd}, act, Device::CUDA);
            device::copy_async(v_at, static_cast<char*>(d_qkv.data) + size_t(qd + kd) * elem,
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

        TensorView key_all(kv_key.data,   {1, total, kd}, act, Device::CUDA);
        TensorView val_all(kv_value.data, {1, total, kd}, act, Device::CUDA);
        grouped_attention_forward(qr_v, key_all, val_all, attn_v, q_heads, kv_heads, head_dim, true, scale, past,
                                  static_cast<float*>(d_attn_partials.data));

        tied_lm_head_forward(attn_v, o_proj, o_b);
        return;
    }

    throw_if(past != 0, "GroupedQueryAttentionOperator: KV-cache decoding requires batch size 1.");

    TensorView q_v (d_q.data,    {1, seq, qd}, act, Device::CUDA);
    TensorView k_v (d_k.data,    {1, seq, kd}, act, Device::CUDA);
    TensorView v_v (d_v.data,    {1, seq, kd}, act, Device::CUDA);
    TensorView qr_v(d_qr.data,   {1, seq, qd}, act, Device::CUDA);
    TensorView kr_v(d_kr.data,   {1, seq, kd}, act, Device::CUDA);
    TensorView attn_v(d_attn.data, {1, seq, qd}, act, Device::CUDA);

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
