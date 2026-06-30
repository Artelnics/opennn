//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A C T I V A T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "activation_operator.h"

namespace opennn
{

class Activation final : public Layer
{
public:

    Activation(const Shape& = {},
               const string& = "ReLU",
               const string& = "activation_layer");

    Shape get_output_shape() const override { return input_shape; }
    ActivationFunction get_output_activation() const override { return activation_operator.activation_function; }

    void set(const Shape&, const string&, const string&);
    void set_input_shape(const Shape&) override;

private:

    ActivationOperator activation_operator;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
