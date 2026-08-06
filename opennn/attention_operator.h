//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A T T E N T I O N   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"
#include "dropout_operator.h"

namespace opennn
{

struct AttentionOperator : Operator
{
    Index heads_number = 0;
    Index head_dimension = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;
    bool  use_causal_mask = false;
    bool use_sdpa = false;

    bool zero_padded_queries = false;

    MatrixR causal_mask;

    DropoutOperator dropout;

    void set(Index, Index,
             Index, Index,
             bool, Type);

    static bool sdpa_supported(Type, Device);

    vector<TensorSpec> forward_scratch_specs(Index) const;

    TensorSpec backward_scratch_spec(Index) const;

    // Backward BF16 scratch slots: dO, dQ, dK, dV, then rematerialized Q, K, V, O.
    static constexpr size_t sdpa_scratch_slots_count = 8;

    vector<TensorSpec> sdpa_gradient_scratch_specs(Index) const;

    TensorSpec sdpa_qkv_pack_spec(Index) const;

    size_t scratch_slot = 0;
    size_t attention_output_slot = 0;

    // First of the sdpa_scratch_slots_count consecutive backward slots planned
    // by the owning layer via sdpa_gradient_scratch_specs.
    size_t sdpa_gradient_slot = 0;

    // Forward-transient slot holding the packed BF16 Q/K/V casts (sdpa_qkv_pack_spec).
    size_t sdpa_qkv_pack_slot = 0;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    AttentionOperator();
    ~AttentionOperator() override;
    AttentionOperator(AttentionOperator&&) noexcept;
    AttentionOperator& operator=(AttentionOperator&&) noexcept;
    AttentionOperator(const AttentionOperator&) = delete;
    AttentionOperator& operator=(const AttentionOperator&) = delete;

    struct SDPACache;

private:
    float scaling_factor() const;

    void apply_unfused(const TensorView&,
                       const TensorView&,
                       const TensorView&,
                       const TensorView&,
                       TensorView&,
                       TensorView&,
                       TensorView&,
                       void*,
                       bool,
                       const vector<Index>* explicit_lengths = nullptr);

#ifdef OPENNN_HAS_CUDA
    void apply_sdpa_forward(const TensorView&,
                            const TensorView&,
                            const TensorView&,
                            const TensorView&,
                            TensorView&,
                            const TensorView&,
                            bool);
#endif

    void apply_delta_cpu(const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         TensorView&,
                         TensorView&,
                         TensorView&,
                         TensorView&) const;

#ifdef OPENNN_HAS_CUDA
    void apply_sdpa_backward(const TensorView&,
                             const TensorView&,
                             const TensorView&,
                             const TensorView&,
                             const TensorView&,
                             TensorView&,
                             TensorView&,
                             TensorView&,
                             span<const TensorView>) const;
#endif

    static bool get_contiguous_source_lengths(const TensorView&,
                                              vector<Index>&,
                                              bool&);
    static void softmax_rows_prefix(float*, Index, Index, Index);
    static Index infer_attention_prefix_length(const TensorView&,
                                               Index);

    template<typename SoftmaxBwd>
    void apply_delta_unfused(const TensorView&,
                              const TensorView&,
                              const TensorView&,
                              const TensorView&,
                              const TensorView&,
                              const TensorView&,
                              TensorView&,
                              TensorView&,
                              TensorView&,
                              TensorView&,
                              SoftmaxBwd&&) const;

    mutable unique_ptr<SDPACache> sdpa_cache;

    uint64_t sdpa_dropout_seed   = 0x9E3779B97F4A7C15ULL;
    uint64_t sdpa_dropout_offset = 0;
    mutable uint64_t sdpa_last_used_offset = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
