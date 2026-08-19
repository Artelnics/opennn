//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/scaling.h"
#include "opennn/neural_network/layers/layer.h"
#include "opennn/neural_network/operators/operator.h"

namespace opennn
{

void scale(const TensorView&,
           const TensorView&, const TensorView&,
           const TensorView&, const TensorView&,
           const TensorView&,
           float, float,
           TensorView&);

void unscale(const TensorView&,
             const TensorView&, const TensorView&,
             const TensorView&, const TensorView&,
             const TensorView&,
             float, float,
             TensorView&);

struct ScaleOperator : Operator
{
    bool invert = false;

    float min_range = -1.0f;
    float max_range = 1.0f;

    TensorView minimums;
    TensorView maximums;
    TensorView means;
    TensorView standard_deviations;
    TensorView scalers;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
};

class Scaling : public Layer, public FeatureScalingEndpoint
{
public:

    Scaling(const Shape& = {});

    Shape get_output_shape() const noexcept override { return input_shape; }

    const vector<Descriptives>& get_descriptives() const noexcept { return descriptives; }
    const vector<ScalerMethod>& get_scalers()      const noexcept { return scalers; }

    VectorR get_minimums()            const;
    VectorR get_maximums()            const;
    VectorR get_means()               const;
    VectorR get_standard_deviations() const;

    float get_min_range() const noexcept { return min_range; }
    float get_max_range() const noexcept { return max_range; }
    bool is_inverse() const noexcept { return scale_op.invert; }
    VariableRole get_scaling_role() const noexcept override
    {
        return is_inverse() ? VariableRole::Target : VariableRole::Input;
    }

    FeatureScaling get_feature_scaling() const override
    {
        return {descriptives, scalers, min_range, max_range};
    }

    void set(const Shape& = {});
    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 1, 2, 3); }
    bool skip_for_pre_scaled_input() const noexcept override { return true; }

    void apply_input_shape(const Shape&) override;

    void set_descriptives(const vector<Descriptives>&);
    void set_scalers(const vector<string>&);
    void set_scalers(const string&);
    void set_feature_scaling(const FeatureScaling&) override;

    bool is_passthrough() const;

    vector<TensorSpec> get_forward_specs(Index) const override;
    void forward_propagate(ForwardPropagation&, size_t, bool) override;

    float* link_states(float*, Device) override;

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

    string write_expression(const vector<string>&,
                            const vector<string>&) const override;

protected:

    Scaling(LayerType, bool invert);

    vector<Descriptives> descriptives;
    vector<ScalerMethod> scalers;
    float min_range = -1.0f;
    float max_range = 1.0f;

    // Device mirror of the configured feature statistics and scaler methods.
    // It belongs to the model; propagation contexts own only transient data.
    Buffer op_storage;
    bool   op_storage_dirty = true;

    ScaleOperator scale_op;

    void refresh_op_storage(Device);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
