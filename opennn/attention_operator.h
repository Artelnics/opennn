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

struct AttentionOp : Operator
{
    Index heads_number = 0;
    Index head_dimension = 0;
    Index query_sequence_length = 0;
    Index source_sequence_length = 0;
    bool  use_causal_mask = false;
    Type  compute_dtype = Type::FP32;

    bool use_sdpa = false;

    MatrixR causal_mask;

    DropoutOp dropout;

    void set(Index heads_number, Index head_dimension,
             Index query_sequence_length, Index source_sequence_length,
             bool use_causal_mask, Type compute_dtype);

    static bool sdpa_supported(Type dtype, Device device);

    void set_dropout_rate(float rate) { dropout.set_rate(rate); }

    vector<TensorSpec> forward_scratch_specs(Index batch_size) const;

    TensorSpec backward_scratch_spec(Index batch_size) const;

    size_t source_view_index = 1;

    vector<size_t> scratch_slots;
    vector<size_t> attention_output_slots;

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

    void destroy_cuda() override;

    AttentionOp();
    ~AttentionOp() override;
    AttentionOp(AttentionOp&&) noexcept;
    AttentionOp& operator=(AttentionOp&&) noexcept;
    AttentionOp(const AttentionOp&) = delete;
    AttentionOp& operator=(const AttentionOp&) = delete;

    struct SDPACache;

private:
    float scaling_factor() const;

    void apply_cpu(const TensorView& query,
                   const TensorView& key,
                   const TensorView& value,
                   const TensorView& source_input,
                   TensorView& attention_weights,
                   TensorView& attention_weights_dropped,
                   TensorView& output,
                   void* scratch,
                   bool is_training);

    void apply_gpu(const TensorView& query,
                   const TensorView& key,
                   const TensorView& value,
                   const TensorView& source_input,
                   TensorView& attention_weights,
                   TensorView& attention_weights_dropped,
                   TensorView& output,
                   void* scratch,
                   bool is_training);

    void apply_delta_cpu(const TensorView& query,
                         const TensorView& key,
                         const TensorView& value,
                         const TensorView& attention_output,
                         const TensorView& attention_weights,
                         const TensorView& attention_weights_dropped,
                         const TensorView& output_delta,
                         TensorView& attention_weight_delta,
                         TensorView& query_delta,
                         TensorView& key_delta,
                         TensorView& value_delta) const;

    void apply_delta_gpu(const TensorView& query,
                         const TensorView& key,
                         const TensorView& value,
                         const TensorView& attention_output,
                         const TensorView& attention_weights,
                         const TensorView& attention_weights_dropped,
                         const TensorView& output_delta,
                         TensorView& attention_weight_delta,
                         TensorView& query_delta,
                         TensorView& key_delta,
                         TensorView& value_delta) const;

    void apply_delta_gpu_unfused(const TensorView& query,
                                 const TensorView& key,
                                 const TensorView& value,
                                 const TensorView& attention_weights,
                                 const TensorView& attention_weights_dropped,
                                 const TensorView& output_delta,
                                 TensorView& attention_weight_delta,
                                 TensorView& query_delta,
                                 TensorView& key_delta,
                                 TensorView& value_delta) const;

    static bool get_contiguous_source_lengths(const TensorView& source_input,
                                              vector<Index>& lengths,
                                              bool& has_padding);
    static void softmax_rows_prefix(float* matrix, Index rows, Index cols, Index length);
    static Index infer_attention_prefix_length(const TensorView& attention_weights,
                                               Index batch_index);

    template<typename SoftmaxBwd>
    void apply_delta_unfused(const TensorView& query,
                              const TensorView& key,
                              const TensorView& value,
                              const TensorView& attention_weights,
                              const TensorView& attention_weights_dropped,
                              const TensorView& output_delta,
                              TensorView& attention_weight_delta,
                              TensorView& query_delta,
                              TensorView& key_delta,
                              TensorView& value_delta,
                              SoftmaxBwd&& softmax_bwd) const;

    mutable unique_ptr<SDPACache> sdpa_cache;

    uint64_t sdpa_dropout_seed   = 0x9E3779B97F4A7C15ULL;
    uint64_t sdpa_dropout_offset = 0;
    mutable uint64_t sdpa_last_used_offset = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
