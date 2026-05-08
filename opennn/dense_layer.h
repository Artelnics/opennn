//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file dense_layer.h
 * @brief Declares the Dense layer: a fully-connected linear projection
 *        followed by an optional batch normalization, an activation
 *        function and an optional dropout.
 */

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

/**
 * @class Dense
 * @brief Fully-connected layer: y = activation(BN(x * W + b)) with optional dropout.
 *
 * The layer is a thin Layer wrapper around four reusable Operator pieces:
 * Combination (linear projection), BatchNorm (optional), Activation and
 * Dropout. Forward and backward passes traverse this fixed pipeline using
 * the slot order encoded in the private Forward / Backward enums.
 *
 * Inputs may be 1D (per-sample feature vectors) or 2D (sequence of feature
 * vectors). When 2D, the first dimension is treated as a sequence length and
 * the same weights are applied to every position.
 */
class Dense final : public Layer
{
public:

    /**
     * @brief Constructs a Dense layer.
     * @param input_shape Per-sample input shape; empty means "set later".
     * @param output_shape Per-sample output shape; the trailing dimension
     *                     is the number of output features.
     * @param activation_name Name of the activation function ("Tanh" by
     *                        default; see Activation for the list).
     * @param batch_normalization Enables BatchNorm between the linear
     *                            projection and the activation.
     * @param label Human-readable label assigned to this layer.
     */
    Dense(const Shape& input_shape = {},
          const Shape& output_shape = {},
          const string& activation_name = "Tanh",
          bool batch_normalization = false,
          const string& label = "dense_layer");

    /** @brief Returns the per-sample input shape. */
    Shape get_input_shape() const override { return input_shape; }

    /**
     * @brief Returns the per-sample output shape.
     * @return Shape with the same leading dimensions as the input and the
     *         configured number of output features as its trailing dimension.
     */
    Shape get_output_shape() const override;

    /**
     * @brief Number of input features (last dimension of the input shape).
     * @return 0 if the input shape is empty; the trailing dimension otherwise.
     */
    Index get_input_features() const { return input_shape.empty() ? 0 : input_shape.back(); }

    /**
     * @brief Sequence length when the input is 2D, 1 otherwise.
     * @return Leading dimension of the input shape if rank-2, else 1.
     */
    Index get_sequence_length() const { return (input_shape.rank == 2) ? input_shape[0] : Index(1); }

    /** @brief Reference to the activation function applied at this layer's output. */
    const Activation::Function& get_activation_function() const { return activation.function; }

    /** @brief Activation function fused at the end of this layer. */
    Activation::Function get_output_activation() const override { return activation.function; }

    /** @brief Whether batch normalization is enabled in the pipeline. */
    bool get_batch_normalization() const { return batch_norm.active(); }

    /** @brief BatchNorm momentum used when updating running statistics. */
    float get_momentum() const { return batch_norm.momentum; }

    /**
     * @brief Returns the active operators in pipeline order.
     * @return Pointers to Combination, then BatchNorm if enabled, then
     *         Activation, then Dropout.
     */
    vector<Operator*> get_operators() override;

    /**
     * @brief Specifications of the forward intermediate buffers.
     * @param batch_size Batch size used for sizing.
     * @return One spec per slot in the Forward enum.
     */
    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;

    /**
     * @brief Re-initializes the layer; same arguments as the constructor.
     * @param input_shape Per-sample input shape.
     * @param output_shape Per-sample output shape.
     * @param activation_name Name of the activation function.
     * @param batch_normalization Enables BatchNorm.
     * @param label Human-readable label.
     */
    void set(const Shape& input_shape = {},
             const Shape& output_shape = {},
             const string& activation_name = "Tanh",
             bool batch_normalization = false,
             const string& label = "dense_layer");

    /** @brief Updates the input shape and re-shapes weight tensors accordingly. */
    void set_input_shape(const Shape&) override;

    /** @brief Updates the output features and re-shapes weight tensors accordingly. */
    void set_output_shape(const Shape&) override;

    /** @brief Reconfigures inner operators when the compute dtype changes. */
    void on_compute_dtype_changed() override { configure_operators(); }

    /**
     * @brief Sets the activation function by name.
     *
     * Receives the name of the activation function, e.g. "Tanh", "ReLU",
     * "Logistic"; see Activation::Function for the supported set.
     */
    void set_activation_function(const string&);

    /**
     * @brief Enables or disables BatchNorm between the linear projection and
     *        the activation.
     * @param enable True to enable BatchNorm.
     */
    void set_batch_normalization(bool enable);

    /**
     * @brief Sets the dropout rate applied at the layer output.
     * @param new_dropout_rate Probability of dropping each unit (0 disables dropout).
     */
    void set_dropout_rate(float new_dropout_rate) { dropout.set_rate(new_dropout_rate); }

    /**
     * @brief Sets the momentum used by BatchNorm to update running statistics.
     * @param new_momentum Momentum value in [0, 1].
     */
    void set_momentum(float new_momentum);

    /**
     * @brief Backward pass through the Dropout, Activation, BatchNorm and
     *        Combination operators in reverse order.
     * @param fp Forward intermediates from the matching forward pass.
     * @param bp BackPropagation buffer in which to accumulate gradients.
     * @param layer Index of this layer inside the network.
     */
    void back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept override;

    /**
     * @brief Reads the layer-specific JSON body (activation name, BN flag,
     *        dropout rate, output features) from the given JSON node.
     */
    void read_JSON_body(const Json*) override;

private:

    /** @brief Per-sample input shape. */
    Shape input_shape;

    /** @brief Number of output features (last dimension of the output shape). */
    Index output_features = 0;

    /** @brief Linear projection y = x * W + b. */
    Combination combination;
    /** @brief Pointwise activation function. */
    Activation  activation;
    /** @brief Optional batch normalization between projection and activation. */
    BatchNorm   batch_norm;
    /** @brief Optional dropout applied at the layer output during training. */
    Dropout     dropout;

    /** @brief Slot ordering in ForwardPropagation::views[layer]. */
    enum Forward {Input, CombinationView, BatchNormMean, BatchNormInverseVariance, ActivationView, Output};
    /** @brief Slot ordering in BackPropagation::backward_views[layer]. */
    enum Backward {OutputDelta, InputDelta};

    /** @brief Reconfigures inner operators after a shape or dtype change. */
    void configure_operators();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
