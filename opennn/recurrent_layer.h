//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file recurrent_layer.h
 * @brief Declares the Recurrent layer: a vanilla RNN cell unrolled over a
 *        fixed number of time steps.
 */

#pragma once

#include "layer.h"

namespace opennn
{

/**
 * @class Recurrent
 * @brief Plain (Elman-style) recurrent layer over fixed-length sequences.
 *
 * Inputs are rank-2 tensors (time_steps, input_features). At each step the
 * hidden state is updated as h_t = activation(W_x * x_t + W_h * h_{t-1} + b),
 * with shared weights and biases across all time steps.
 *
 * The layer owns three parameter tensors (input weights, recurrent
 * weights, biases) and applies a configurable elementwise activation.
 */
class Recurrent final : public Layer
{
public:

    /**
     * @brief Constructs a Recurrent layer.
     * @param input_shape Per-sample input shape (time_steps, input_features).
     * @param output_shape Per-sample output shape (number of hidden units).
     */
    Recurrent(const Shape& input_shape = {0, 0}, const Shape& output_shape = {0});

    /** @brief Returns the per-sample input shape (time_steps, input_features). */
    Shape get_input_shape() const override { return {time_steps, input_features}; }
    /**
     * @brief Returns the per-sample output shape.
     * @return Shape with the number of hidden units.
     */
    Shape get_output_shape() const override;

    /** @brief Read-only access to the bias TensorView. */
    const TensorView& get_biases() const { return biases; }
    /** @brief Read-only access to the input-to-hidden weight TensorView. */
    const TensorView& get_input_weights() const { return input_weights; }
    /** @brief Read-only access to the hidden-to-hidden weight TensorView. */
    const TensorView& get_recurrent_weights() const { return recurrent_weights; }
    /** @brief Name of the activation function applied to the hidden state. */
    const string& get_activation_function() const { return activation_function; }

    /**
     * @brief Specifications of the parameter tensors.
     * @return Specs for input weights, recurrent weights and biases.
     */
    vector<pair<Shape, Type>> get_parameter_specs() const override;
    /**
     * @brief Specifications of the forward intermediate buffers.
     * @param batch_size Batch size used for sizing.
     * @return Per-step combination and activation buffers plus the output.
     */
    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;
    /**
     * @brief Specifications of the backward intermediate buffers.
     * @param batch_size Batch size used for sizing.
     * @return Buffers used for time-unrolled gradient accumulation.
     */
    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override;

    /**
     * @brief Re-initializes the layer.
     * @param input_shape Per-sample input shape (time_steps, input_features).
     * @param output_shape Per-sample output shape (hidden units).
     */
    void set(const Shape& input_shape = {}, const Shape& output_shape = {});

    /** @brief Updates the input shape and re-shapes weight tensors accordingly. */
    void set_input_shape(const Shape&) override;
    /** @brief Updates the output shape and re-shapes weight tensors accordingly. */
    void set_output_shape(const Shape&) override;

    /**
     * @brief Sets the activation function applied to the hidden state.
     *
     * Receives the activation name; see Activation::Function for the
     * supported set.
     */
    void set_activation_function(const string&);

    /**
     * @brief Forward pass: unrolls the recurrence over time_steps steps.
     *
     * Receives the ForwardPropagation buffer slice, this layer's index and
     * the training flag.
     */
    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;
    /**
     * @brief Backward pass through time (BPTT) over the unrolled recurrence.
     *
     * Receives the forward intermediates, the BackPropagation buffer and
     * this layer's index inside the network.
     */
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    /**
     * @brief Reads the layer-specific JSON body (shapes, activation name).
     */
    void read_JSON_body(const Json*) override;
    /**
     * @brief Writes the layer-specific JSON body (shapes, activation name).
     */
    void write_JSON_body(JsonWriter&) const override;

private:

    /** @brief Number of time steps in the input sequence. */
    Index time_steps = 0;
    /** @brief Number of features per time step. */
    Index input_features = 0;

    /** @brief Bias tensor view (one bias per hidden unit). */
    TensorView biases;
    /** @brief Input-to-hidden weight tensor view. */
    TensorView input_weights;
    /** @brief Hidden-to-hidden (recurrent) weight tensor view. */
    TensorView recurrent_weights;

    /** @brief Sentinel passed to calculate_activations<2>() when no derivative output is wanted. */
    Tensor2 empty_2;

    /** @brief Name of the activation function (default "Tanh"). */
    string activation_function = "Tanh";
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
