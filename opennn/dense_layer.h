//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

class Dense final : public Layer
{
public:

    Dense(const Shape& = {},
          const Shape& = {},
          const string& = "Tanh",
          bool = false,
          const string& = "dense_layer");

    Shape get_input_shape() const override { return input_shape; }
    Shape get_output_shape() const override;

    Index get_input_features() const { return input_shape.empty() ? 0 : input_shape.back(); }
    Index get_sequence_length() const { return (input_shape.rank == 2) ? input_shape[0] : Index(1); }

    const Activation::Function& get_activation_function() const { return activation.function; }
    Activation::Function get_output_activation() const override { return activation.function; }

    bool get_batch_normalization() const { return batch_norm.active(); }
    float get_momentum() const { return batch_norm.momentum; }

    vector<Operator*> get_operators() override;
    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;
    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override;

    void set(const Shape& = {},
             const Shape& = {},
             const string& = "Tanh",
             bool = false,
             const string& = "dense_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;
    void set_compute_dtype(Type new_compute_dtype) override
    {
        Layer::set_compute_dtype(new_compute_dtype);
        configure_operators();
    }

    void set_activation_function(const string&);
    void set_batch_normalization(bool enable);
    void set_dropout_rate(float new_dropout_rate) { dropout.set_rate(new_dropout_rate); }
    void set_momentum(float new_momentum);

    void set_parameters_glorot() override;
    void set_parameters_random() override;

    void forward_propagate(ForwardPropagation&, size_t layer, bool is_training) noexcept override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t layer) const noexcept override;

    void from_JSON(const JsonDocument&) override;
    void load_state_from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

private:

    Shape input_shape;
    Index output_features = 0;

    Combination combination;
    Activation  activation;
    BatchNorm   batch_norm;
    Dropout     dropout;

    enum Parameters {Bias, Weight, Gamma, Beta};
    enum States {RunningMean, RunningVariance};
    enum Forward {Input, CombinationView, BatchNormMean, BatchNormInverseVariance, ActivationView, Output};
    enum Backward {OutputDelta, InputDelta};

    void configure_operators();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
