//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A C T I V A T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"

namespace opennn
{

class Activation final : public Layer
{
public:

    Activation(const Shape& = {},
               const string& = "ReLU",
               const string& = "activation_layer");

    Shape get_output_shape() const override { return input_shape; }
    ActivationOp::Function get_output_activation() const override { return activation.function; }

    void set(const Shape&, const string&, const string&);
    void set_input_shape(const Shape& new_input_shape) override
    {
        set(new_input_shape, ActivationOp::to_string(activation.function), label);
    }

    void set_function(const string& name) { activation.set_function(name); }

private:

    ActivationOp activation;

    enum Forward {Input, Output};
    enum Backward {OutputDelta, InputDelta};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
