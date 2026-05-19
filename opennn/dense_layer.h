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

/// @brief Fully-connected layer with configurable activation, optional batch normalization and dropout.
class Dense final : public Layer
{
public:

    /// @brief Constructs a dense layer with the given input/output shapes, activation, and batch-norm flag.
    Dense(const Shape& = {},
          const Shape& = {},
          const string& = "Tanh",
          bool = false,
          const string& = "dense_layer");

    /// @brief Returns the layer output shape, derived from input shape and configured output features.
    Shape get_output_shape() const override;

    Index get_input_features() const { return input_shape.empty() ? 0 : input_shape.back(); }
    Index get_sequence_length() const { return (input_shape.rank == 2) ? input_shape[0] : Index(1); }

    const ActivationOp::Function& get_activation_function() const { return activation.function; }
    ActivationOp::Function get_output_activation() const override { return activation.function; }

    bool get_batch_normalization() const { return batch_norm.active(); }
    float get_momentum() const { return batch_norm.momentum; }

    /// @brief Returns the tensor specs of intermediate forward buffers for a given batch size.
    vector<TensorSpec> get_forward_specs(Index batch_size) const override;

    /// @brief Reconfigures the layer with new shapes, activation, batch-normalization flag and label.
    void set(const Shape& = {},
             const Shape& = {},
             const string& = "Tanh",
             bool = false,
             const string& = "dense_layer");

    /// @copydoc Layer::set_input_shape
    void set_input_shape(const Shape&) override;
    /// @copydoc Layer::set_output_shape
    void set_output_shape(const Shape&) override;
    void on_compute_dtype_changed() override { configure_operators(); }

    /// @brief Sets the activation function from its string name (e.g. "Tanh", "Logistic", "Linear").
    void set_activation_function(const string&);
    /// @brief Enables or disables batch normalization on the layer combination output.
    void set_batch_normalization(bool enable);
    void set_dropout_rate(float new_dropout_rate) { dropout.set_rate(new_dropout_rate); }
    /// @brief Sets the running-statistics momentum used by batch normalization.
    void set_momentum(float new_momentum);

    /// @copydoc Layer::read_JSON_body
    void read_JSON_body(const Json*) override;

    /// @copydoc Layer::write_expression
    string write_expression(const vector<string>& input_names,
                            const vector<string>& output_names) const override;

private:

    Index output_features = 0;

    CombinationOp combination;
    ActivationOp  activation;
    BatchNormOp   batch_norm;
    DropoutOp     dropout;

    enum Forward {Input, CombinationView, BatchNormMean, BatchNormInverseVariance, ActivationView, Output};

    void configure_operators();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
