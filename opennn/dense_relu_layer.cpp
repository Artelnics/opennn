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
{
    set(new_input_shape, new_output_shape, new_label);
}

Shape DenseRelu::get_output_shape() const
{
    if (input_shape.empty()) return {output_features};
    Shape output_shape = input_shape;
    output_shape.back() = output_features;
    return output_shape;
}

vector<pair<Shape, Type>> DenseRelu::get_forward_specs(Index batch_size) const
{
    return {{Shape{batch_size}.append(get_output_shape()), compute_dtype}};
}

void DenseRelu::configure_operators()
{
    combination.set(get_input_features(), output_features, compute_dtype);

    combination.epilogue     = CUBLASLT_EPILOGUE_RELU_BIAS;
    combination.input_slots  = {Input};
    combination.output_slots = {Output};

    activation.set_function(Activation::Function::ReLU);
    activation.input_slots  = {Output};
    activation.output_slots = {Output};
}

void DenseRelu::set(const Shape& new_input_shape,
                    const Shape& new_output_shape,
                    const string& new_label)
{
    is_trainable = true;
    layer_type = LayerType::DenseRelu;
    name = "DenseRelu";

    if (new_input_shape.empty() && new_output_shape.empty())
    {
        input_shape = {};
        output_features = 0;
        return;
    }

    if (new_input_shape.rank != 1 && new_input_shape.rank != 2)
        throw runtime_error("DenseRelu input shape rank must be 1 or 2 (got "
                            + to_string(new_input_shape.rank) + ").");

    if (new_output_shape.rank != 1)
        throw runtime_error("DenseRelu output shape rank must be 1.");

    input_shape = new_input_shape;
    output_features = new_output_shape.back();

    set_label(new_label);

    configure_operators();
}

void DenseRelu::set_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.rank != 1 && new_input_shape.rank != 2)
        throw runtime_error("DenseRelu input shape rank must be 1 or 2.");

    input_shape = new_input_shape;
    configure_operators();
}

void DenseRelu::set_output_shape(const Shape& new_output_shape)
{
    output_features = new_output_shape.back();
    configure_operators();
}

void DenseRelu::back_propagate(ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation,
                               size_t layer) const noexcept
{
    auto& forward_views   = forward_propagation.views[layer];
    auto& delta_views     = back_propagation.delta_views[layer];

    const TensorView& input  = forward_views[Input][0];
    const TensorView& output = forward_views[Output][0];

    TensorView& output_delta = delta_views[OutputDelta][0];

    activation.apply_delta(output, output_delta);

    const Index total_rows = input.size() / input.shape.back();

    TensorView output_delta_2d = output_delta.reshape({total_rows, output_delta.shape.back()});
    TensorView input_2d        = input.reshape({total_rows, input.shape.back()});

    TensorView input_delta_2d;
    if (!is_first_layer)
    {
        TensorView& input_delta = delta_views[InputDelta][0];
        input_delta_2d = input_delta.reshape({total_rows, input_delta.shape.back()});
    }

    combination.apply_delta(output_delta_2d,
                            input_2d,
                            input_delta_2d,
                            false);
}

REGISTER(Layer, DenseRelu, "DenseRelu")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
