//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   R E L U   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dense_relu_layer.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "random_utilities.h"

namespace opennn
{

DenseRelu::DenseRelu(const Shape& new_input_shape,
                     const Shape& new_output_shape,
                     const string& new_label)
    : Layer(LayerType::DenseRelu)
{
    operators = {&combination_relu};
    set(new_input_shape, new_output_shape, new_label);
}

Shape DenseRelu::get_output_shape() const
{
    if (input_shape.empty()) return {output_features};
    Shape output_shape = input_shape;
    output_shape.back() = output_features;
    return output_shape;
}

void DenseRelu::configure_operators()
{
    combination_relu.set(get_input_features(), output_features, compute_dtype);
}

void DenseRelu::set(const Shape& new_input_shape,
                    const Shape& new_output_shape,
                    const string& new_label)
{
    if (new_input_shape.empty() && new_output_shape.empty())
    {
        input_shape = {};
        output_features = 0;
        return;
    }

    check_rank(new_input_shape, {1, 2}, "DenseRelu", "input");
    check_rank(new_output_shape, {1}, "DenseRelu", "output");

    input_shape = new_input_shape;
    output_features = new_output_shape.back();

    set_label(new_label);

    configure_operators();
}

void DenseRelu::set_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {1, 2}, "DenseRelu", "input");
    input_shape = new_input_shape;
    configure_operators();
}

void DenseRelu::set_output_shape(const Shape& new_output_shape)
{
    output_features = new_output_shape.back();
    configure_operators();
}

REGISTER(Layer, DenseRelu, "DenseRelu")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
