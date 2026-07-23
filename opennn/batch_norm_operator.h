//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   N O R M   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

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
    // Always present so the class layout is identical in CPU and CUDA builds;
    // stays null on CPU. The cache type is completed in the .cpp.
    mutable unique_ptr<BatchNormalizationGraphCache> bn_graph_cache;

    void set(Index, float new_momentum = 0.1f);

    vector<TensorSpec> parameter_specs() const override;
    vector<TensorSpec> state_specs()     const override;
    void link_parameters(span<const TensorView>) override;
    void link_gradients (span<const TensorView>) override;
    void link_states    (span<const TensorView>) override;

    void set_parameters_random() override { init_defaults(); }
    void set_parameters_glorot() override { init_defaults(); }

    void init_defaults();

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
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

    mutable VectorR delta_scale_scratch;

    void apply_inference_cpu(const TensorView&, TensorView&);
    void apply_inference_gpu(const TensorView&, TensorView&,
                             const TensorView&);

    void apply_training_cpu (const TensorView&,
                             TensorView&, TensorView&,
                             TensorView&);
    void apply_training_gpu (const TensorView&,
                             TensorView&, TensorView&,
                             TensorView&,
                             const TensorView&);

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
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
