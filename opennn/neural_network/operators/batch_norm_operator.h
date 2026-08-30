//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   N O R M   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/operators/operator.h"

namespace opennn
{

// Every path that normalizes - the operator's own forward, the cuDNN graphs, and the
// fold of batch norm into convolution weights for inference - must use this same value.
// It lived as a file-local constant in batch_norm_operator.cpp, so the fold in
// convolutional_layer.cpp could not see it and reached for the generic EPSILON
// (numeric_limits<float>::epsilon(), ~84x smaller), which made folded inference
// disagree with every other path.
inline constexpr float BN_EPSILON = 1e-5f;

struct BatchNormalizationOperator : Operator
{
    Index features = 0;
    float momentum = 0.1f;
    bool fuse_relu = false;
    bool fuse_add = false;
    size_t residual_delta_slot = 0;

    TensorView gamma;
    TensorView beta;
    TensorView running_mean;
    TensorView running_variance;

    TensorView gamma_gradient;
    TensorView beta_gradient;

    bool active() const { return features > 0; }

    BatchNormalizationOperator();
    ~BatchNormalizationOperator() override;

    struct BatchNormalizationGraphCache;

    unique_ptr<BatchNormalizationGraphCache> bn_graph_cache;

    void set(Index, float new_momentum = 0.1f);

    vector<TensorSpec> parameter_specs() const override;
    vector<TensorSpec> state_specs() const override { return parameter_specs(); }
    vector<ParameterSlot> parameter_slots() override;
    void link_parameters(span<const TensorView>) override;
    void link_states    (span<const TensorView>) override;

    void set_parameters_random() override { init_defaults(); }
    void set_parameters_glorot() override { init_defaults(); }
    void initialize_states() override;

    void init_defaults();

    void forward_propagate(ForwardPropagation&, size_t, ForwardPropagationMode) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

    void to_JSON(JsonWriter&) const override;
    void from_JSON(const Json*) override;
    void load_state_from_JSON(const Json*) override;

    void invalidate_inference_cache() { inference_cache_dirty = true; }

private:
    VectorR inference_scale;
    VectorR inference_shift;
    bool    inference_cache_dirty = true;

    void update_inference_cache();

    void apply_inference_cpu(const TensorView&, TensorView&);
    void apply_inference_gpu(const TensorView&, TensorView&,
                             const TensorView&);

    void apply_training_cpu (const TensorView&,
                             TensorView&, TensorView&,
                             TensorView&);
    void apply_training_gpu (const TensorView&,
                             TensorView&, TensorView&,
                             TensorView&,
                             const TensorView&,
                             TensorView&);

    void apply_delta_cpu(const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         TensorView&) const;
    void apply_delta_gpu(const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         const TensorView&,
                         TensorView&,
                         TensorView&) const;

    bool own_forward_kernel(const TensorView& mask) const noexcept;

    TensorView& relu_mask(ForwardPropagation&, size_t layer) const noexcept;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
