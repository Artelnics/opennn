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

    MatrixR causal_mask;

    DropoutOperator dropout;

    void set(Index, Index,
             Index, Index,
             bool, Type);

    static bool sdpa_supported(Type, Device);

    vector<TensorSpec> forward_scratch_specs(Index) const;

    TensorSpec backward_scratch_spec(Index) const;

    size_t source_view_index = 1;

    vector<size_t> scratch_slots;
    vector<size_t> attention_output_slots;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    AttentionOperator();
    ~AttentionOperator() override;
    AttentionOperator(AttentionOperator&&) noexcept;
    AttentionOperator& operator=(AttentionOperator&&) noexcept;
    AttentionOperator(const AttentionOperator&) = delete;
    AttentionOperator& operator=(const AttentionOperator&) = delete;

#ifdef OPENNN_HAS_CUDA
    struct SDPACache;
#endif

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
                       bool);

    void apply_sdpa_forward(const TensorView&,
                            const TensorView&,
                            const TensorView&,
                            const TensorView&,
                            TensorView&,
                            bool);

    void apply_delta_cpu(const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         TensorView&,
                         TensorView&,
                         TensorView&,
                         TensorView&) const;

    void apply_sdpa_backward(const TensorView&,
                             const TensorView&,
                             const TensorView&,
                             const TensorView&,
                             const TensorView&,
                             TensorView&,
                             TensorView&,
                             TensorView&) const;

    void apply_delta_gpu_unfused(const TensorView&,
                                 const TensorView&,
                                 const TensorView&,
                                 const TensorView&,
                                 const TensorView&,
                                 const TensorView&,
                                 TensorView&,
                                 TensorView&,
                                 TensorView&,
                                 TensorView&) const;

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

#ifdef OPENNN_HAS_CUDA
    mutable unique_ptr<SDPACache> sdpa_cache;
#endif

    uint64_t sdpa_dropout_seed   = 0x9E3779B97F4A7C15ULL;
    uint64_t sdpa_dropout_offset = 0;
    mutable uint64_t sdpa_last_used_offset = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
