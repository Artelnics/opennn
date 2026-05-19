//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"

namespace opennn
{

/// @brief Basic recurrent (RNN) layer that unrolls over time steps with a shared activation.
class Recurrent final : public Layer
{
public:

    /// @brief Constructs a recurrent layer with input and output shapes.
    /// @param input_shape Input shape as (time_steps, input_features).
    /// @param output_shape Output shape (number of hidden units).
    Recurrent(const Shape& = {0, 0}, const Shape& = {0});

    /// @brief Returns the input tensor shape (time_steps, input_features).
    Shape get_input_shape() const override { return {time_steps, input_features}; }

    /// @brief Returns the output tensor shape (hidden units).
    Shape get_output_shape() const override;

    const TensorView& get_biases() const { return biases; }
    const TensorView& get_input_weights() const { return input_weights; }
    const TensorView& get_recurrent_weights() const { return recurrent_weights; }
    const string& get_activation_function() const { return activation_function; }

    /// @brief Returns the tensor specifications for the trainable parameters.
    vector<TensorSpec> get_parameter_specs() const override;

    /// @brief Returns the tensor specifications used during forward propagation.
    vector<TensorSpec> get_forward_specs(Index batch_size) const override;

    /// @brief Returns the tensor specifications used during back propagation.
    vector<TensorSpec> get_backward_specs(Index batch_size) const override;

    /// @brief Reconfigures the layer with new input and output shapes.
    void set(const Shape& = {}, const Shape& = {});

    /// @brief Updates the layer for a new input shape.
    void set_input_shape(const Shape&) override;

    /// @brief Updates the layer for a new output shape.
    void set_output_shape(const Shape&) override;

    /// @brief Sets the activation function by name (e.g. "Tanh", "ReLU").
    void set_activation_function(const string&);

    /// @brief Runs the forward pass, unrolling the recurrence across time steps.
    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    /// @brief Runs the backward pass (backpropagation through time).
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    /// @brief Reads the layer configuration from a JSON node.
    void read_JSON_body(const Json*) override;

    /// @brief Writes the layer configuration to a JSON writer.
    void write_JSON_body(JsonWriter&) const override;

    /// @brief Returns a symbolic expression of the layer for export.
    string write_expression(const vector<string>& input_names,
                            const vector<string>& output_names) const override;

private:

    Index time_steps = 0;
    Index input_features = 0;

    TensorView biases;
    TensorView input_weights;
    TensorView recurrent_weights;

    Tensor2 empty_2;  // sentinel passed to calculate_activations<2>(...) when no derivative output is wanted

    string activation_function = "Tanh";
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
