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

struct BatchNormOp : Operator
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

    BatchNormOp();
    ~BatchNormOp() override;

    struct BnGraphCache;
    mutable unique_ptr<BnGraphCache> bn_graph_cache;

    void destroy_cuda() override;

    void set(Index new_features, float new_momentum = 0.1f);

    vector<TensorSpec> parameter_specs() const override;
    vector<TensorSpec> state_specs()     const override;
    void link_parameters(span<const TensorView> views) override;
    void link_gradients (span<const TensorView> views) override;
    void link_states    (span<const TensorView> views) override;

    void set_parameters_random() override { init_defaults(); }
    void set_parameters_glorot() override { init_defaults(); }

    void init_defaults();

    void forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) override;
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const override;

    void update_inference_cache();
    void invalidate_inference_cache() { inference_cache_dirty = true; }

    void to_JSON(JsonWriter& w) const override;
    void from_JSON(const Json* parent) override;
    void load_state_from_JSON(const Json* parent) override;

private:
    VectorR inference_scale;
    VectorR inference_shift;
    bool    inference_cache_dirty = true;

    mutable VectorR delta_scale_scratch;

    void apply_inference_cpu(const TensorView& input, TensorView& output);
    void apply_inference_gpu(const TensorView& input, TensorView& output,
                             const TensorView& residual);

    void apply_training_cpu (const TensorView& input,
                             TensorView& mean, TensorView& inverse_variance,
                             TensorView& output);
    void apply_training_gpu (const TensorView& input,
                             TensorView& mean, TensorView& inverse_variance,
                             TensorView& output,
                             const TensorView& residual);

    void apply_delta_cpu(const TensorView& input,
                         const TensorView& mean,
                         const TensorView& inverse_variance,
                         TensorView& delta) const;
    void apply_delta_gpu(const TensorView& input,
                         const TensorView& output,
                         const TensorView& residual,
                         const TensorView& mean,
                         const TensorView& inverse_variance,
                         TensorView& delta,
                         TensorView& residual_delta) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
