//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   R E L U   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

// Dense + ReLU fused into a single forward op (cuBLASLt RELU_BIAS epilogue
// on GPU; ReLU baked into Combination::apply_cpu when epilogue is RELU_BIAS).
// No batch-norm, no dropout, activation hard-wired to ReLU — keeps
// forward_propagate branch-free for CUDA Graph capture.
class DenseRelu final : public Layer
{
public:

    DenseRelu(const Shape& = {},
              const Shape& = {},
              const string& = "dense_relu_layer");

    Shape get_input_shape() const override { return input_shape; }
    Shape get_output_shape() const override;

    Index get_input_features() const { return input_shape.empty() ? 0 : input_shape.back(); }
    Index get_sequence_length() const { return (input_shape.rank == 2) ? input_shape[0] : Index(1); }

    Activation::Function get_output_activation() const override { return Activation::Function::ReLU; }

    vector<Operator*> get_operators() override { return {&combination}; }

    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;

    void set(const Shape& = {},
             const Shape& = {},
             const string& = "dense_relu_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;
    void on_compute_dtype_changed() override { configure_operators(); }

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t layer) const noexcept override;


private:

    Shape input_shape;
    Index output_features = 0;

    Combination combination;
    Activation  activation;

    enum Forward {Input, Output};
    enum Backward {OutputDelta, InputDelta};

    void configure_operators();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
