//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A T T E N T I O N   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/operators/operator.h"
#include "opennn/neural_network/operators/dropout_operator.h"

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

    bool interleaved_heads = false;

    bool zero_padded_queries = false;

    MatrixR causal_mask;

    DropoutOperator dropout;

    void set(Index, Index,
             Index, Index,
             bool, Type);

    static bool sdpa_supported(Type, Device);

    vector<TensorSpec> forward_scratch_specs(Index) const;

    TensorSpec backward_scratch_spec(Index) const;

    static constexpr size_t sdpa_scratch_slots_count = 8;

    vector<TensorSpec> sdpa_gradient_scratch_specs(Index) const;

    struct SdpaBf16Pack
    {
        Index query_elements = 0;
        Index source_elements = 0;

        static Index slot_elements(Index elements)
        {
            return get_aligned_bytes(elements * Index(sizeof(bfloat16))) / Index(sizeof(bfloat16));
        }

        Index total_elements() const
        {
            return 2 * slot_elements(query_elements) + 2 * slot_elements(source_elements);
        }

        struct Pointers { bfloat16* query; bfloat16* key; bfloat16* value; bfloat16* output; };

        Pointers over(bfloat16* base) const
        {
            bfloat16* const key   = base + slot_elements(query_elements);
            bfloat16* const value = key  + slot_elements(source_elements);
            return {base, key, value, value + slot_elements(source_elements)};
        }
    };

    TensorSpec sdpa_qkv_pack_spec(Index) const;
    static constexpr size_t sdpa_state_slots_count = 4;
    vector<TensorSpec> sdpa_state_specs(Index) const;

    size_t scratch_slot = 0;
    size_t attention_output_slot = 0;

    size_t concatenated_output_delta_slot = 0;

    size_t sdpa_gradient_slot = 0;

    size_t sdpa_qkv_pack_slot = 0;
    size_t sdpa_state_slot = 0;
    size_t dropout_mask_slot = 0;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void concatenate_output_heads(ForwardPropagation&, size_t) const;
    void split_output_delta(ForwardPropagation&, BackPropagation&, size_t) const;

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
                       TensorView&,
                       void*,
                       bool,
                       SequenceLengths explicit_lengths = {});

#ifdef OPENNN_HAS_CUDA
    void apply_sdpa_forward(const TensorView&,
                            const TensorView&,
                            const TensorView&,
                            const TensorView&,
                            TensorView&,
                            const TensorView&,
                            span<const TensorView>,
                            bool,
                            const int* explicit_lengths = nullptr);
#endif

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
                         TensorView&,
                         const vector<Index>* = nullptr) const;

#ifdef OPENNN_HAS_CUDA
    void apply_sdpa_backward(const TensorView&,
                             const TensorView&,
                             const TensorView&,
                             const TensorView&,
                             const TensorView&,
                             TensorView&,
                             TensorView&,
                             TensorView&,
                             span<const TensorView>,
                             span<const TensorView>) const;
#endif

    static bool get_contiguous_source_lengths(const TensorView&,
                                              vector<Index>&,
                                              bool&);
    static void softmax_rows_prefix(float*, Index, Index, Index);

    void apply_delta_unfused(const TensorView&,
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

    unique_ptr<SDPACache> sdpa_cache;

#ifdef OPENNN_HAS_CUDA
    static constexpr uint64_t sdpa_dropout_seed = 0x9E3779B97F4A7C15ULL;
    uint64_t sdpa_dropout_offset = 0;
#endif
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
