//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O U P E D   Q U E R Y   A T T E N T I O N   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

// Causal grouped-query self-attention (Ainslie et al., 2023): bias-free q/k/v/o
// projections, optional per-head QK-Norm before RoPE, decoupled head_dim.
// Inference forward only: back_propagate throws.
// Batch-1 supports KV-cache decoding: prefill (past_length == 0) fills the cache,
// each single-token pass (past_length == cache length) appends and attends it.
struct GroupedQueryAttentionOperator : Operator
{
    Index sequence_length = 0;
    Index hidden          = 0;
    Index q_heads         = 0;
    Index kv_heads        = 0;
    Index head_dim        = 0;
    float rope_theta      = 1000000.0f;
    float rms_epsilon     = 1.0e-6f;

    // Per-head RMSNorm of q/k before RoPE (Qwen3 on, LLaMA/Mistral off).
    // Changes the parameter layout, so it must be chosen before compile().
    bool use_qk_norm = true;

    TensorView q_proj, k_proj, v_proj, o_proj, q_norm, k_norm;

    // The q/k/v parameter blocks are usually contiguous (specs 0-2, aligned
    // sizes); when they are, decode runs one fused [q_dim + 2*kv_dim, hidden]
    // projection instead of three.
    bool qkv_fused = false;

    void set(Index new_sequence_length, Index new_hidden,
             Index new_q_heads, Index new_kv_heads, Index new_head_dim,
             float new_rope_theta, float new_rms_epsilon, bool new_use_qk_norm);

    Index q_dim()  const { return q_heads  * head_dim; }
    Index kv_dim() const { return kv_heads * head_dim; }

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView>) override;
    void set_parameters_random() override;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    // GPU forward; past is ForwardPropagation::past_length (0 restarts the cache).
    // position_device mirrors past on device (ForwardPropagation::stage_position);
    // single-token decode then runs the fused QK-Norm+RoPE+append kernel and the
    // attention kernel with device-side positions, so the whole step is
    // CUDA-graph capturable.
    void forward_gpu(TensorView& input, TensorView& output, Index batch, Index past,
                     const int* position_device);

    // Scratch + RoPE tables, sized to the compiled max sequence.
    // d_attn_partials is the fp32 split-KV scratch of the decode attention kernel;
    // d_qkv holds fused-projection rows [q | k | v].
    mutable Buffer d_cos, d_sin, d_q, d_k, d_v, d_qr, d_kr, d_attn, d_attn_partials, d_qkv;
    mutable Index gpu_sequence = -1;
    mutable Type gpu_dtype = Type::FP32;

    // K/V cache [max_seq, kv_dim]: post-RoPE keys, raw values.
    mutable Buffer kv_key, kv_value;
    mutable Index cache_capacity = 0;
    mutable Type cache_dtype = Type::FP32;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
