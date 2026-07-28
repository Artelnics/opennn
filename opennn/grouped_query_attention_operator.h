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

struct GroupedQueryAttentionOperator : Operator
{
    Index sequence_length = 0;
    Index hidden          = 0;
    Index q_heads         = 0;
    Index kv_heads        = 0;
    Index head_dim        = 0;
    float rope_theta      = 1000000.0f;
    float rms_epsilon     = 1.0e-6f;

    bool use_qk_norm = true;

    TensorView q_proj, k_proj, v_proj, o_proj, q_norm, k_norm;
    TensorView q_scale, k_scale, v_scale, o_scale, qkv_scale;

    bool qkv_fused = false;

    void set(Index new_sequence_length, Index new_hidden,
             Index new_q_heads, Index new_kv_heads, Index new_head_dim,
             float new_rope_theta, float new_rms_epsilon, bool new_use_qk_norm);

    Index q_dim()  const { return q_heads  * head_dim; }
    Index kv_dim() const { return kv_heads * head_dim; }

    vector<TensorSpec> parameter_specs() const override;
    vector<SlotQuantization> parameter_quantization() const override;
    void link_parameters(span<const TensorView>) override;
    void link_parameter_scales(span<const TensorView>) override;
    void set_parameters_random() override;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void forward_gpu(TensorView& input, TensorView& output, Index batch, Index past,
                     Index query_capacity, const int* position_device) const;

    mutable Buffer kv_key, kv_value;
    mutable Index cache_capacity = 0;
    mutable Type cache_dtype = Type::FP32;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
