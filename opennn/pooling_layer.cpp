//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "pooling_layer.h"
#include "loss.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

Pooling::Pooling(const Shape& new_input_shape,
                 const Shape& new_pool_dimensions,
                 const Shape& new_stride_shape,
                 const Shape& new_padding_dimensions,
                 const string& new_pooling_method,
                 const string& new_name) : Layer()
{
    name = "Pooling";
    layer_type = LayerType::Pooling;

    set(new_input_shape,
        new_pool_dimensions,
        new_stride_shape,
        new_padding_dimensions,
        new_pooling_method,
        new_name);
}

Shape Pooling::get_output_shape() const
{
    const Index rows_number = get_output_height();
    const Index columns_number = get_output_width();
    const Index channels = input_shape[2];

    return { rows_number, columns_number, channels };
}

Index Pooling::get_output_height() const
{
    return (get_input_height() - pool_height + 2 * padding_height) / row_stride + 1;
}

Index Pooling::get_output_width() const
{
    return (get_input_width() - pool_width + 2 * padding_width) / column_stride + 1;
}

void Pooling::set(const Shape& new_input_shape,
                  const Shape& new_pool_dimensions,
                  const Shape& new_stride_shape,
                  const Shape& new_padding_dimensions,
                  const string& new_pooling_method,
                  const string& new_label)
{
    if(new_pool_dimensions.rank() != 2)
        throw runtime_error("Pool shape must be 2");

    if (new_stride_shape.rank() != 2)
        throw runtime_error("Stride shape must be 2");

    if (new_padding_dimensions.rank() != 2)
        throw runtime_error("Padding shape must be 2");

    if (new_pool_dimensions[0] > new_input_shape[0] || new_pool_dimensions[1] > new_input_shape[1])
        throw runtime_error("Pool shape cannot be bigger than input shape");

    if (new_stride_shape[0] <= 0 || new_stride_shape[1] <= 0)
        throw runtime_error("Stride shape cannot be 0 or lower");

    if (new_stride_shape[0] > new_input_shape[0] || new_stride_shape[1] > new_input_shape[1])
        throw runtime_error("Stride shape cannot be bigger than input shape");

    if (new_padding_dimensions[0] < 0 || new_padding_dimensions[1] < 0)
        throw runtime_error("Padding shape cannot be lower than 0");

    input_shape = new_input_shape;

    set_pool_size(new_pool_dimensions[0], new_pool_dimensions[1]);

    set_row_stride(new_stride_shape[0]);
    set_column_stride(new_stride_shape[1]);

    set_padding_height(new_padding_dimensions[0]);
    set_padding_width(new_padding_dimensions[1]);

    set_pooling_method(new_pooling_method);

    set_label(new_label);

    label = "pooling_layer";

#ifdef OPENNN_WITH_CUDA

    // Pooling descriptor

    cudnnCreatePoolingDescriptor(&pooling_descriptor);

    cudnnSetPooling2dDescriptor(pooling_descriptor,
                                pooling_mode,
                                CUDNN_PROPAGATE_NAN,
                                pool_height, pool_width,
                                padding_height, padding_width,
                                row_stride, column_stride);

#endif

    cached_pool_args.pool_dimensions = {pool_height, pool_width};
    cached_pool_args.stride_shape = {row_stride, column_stride};
    cached_pool_args.padding_shape = {padding_height, padding_width};
#ifdef OPENNN_WITH_CUDA
    cached_pool_args.pooling_descriptor = pooling_descriptor;
#endif
}

void Pooling::set_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.rank() != 3)
        throw runtime_error("Input shape must be 3");

    input_shape = new_input_shape;
}

void Pooling::set_pool_size(const Index new_pool_rows_number,
                            Index new_pool_columns_number)
{
    pool_height = new_pool_rows_number;
    pool_width = new_pool_columns_number;
}

void Pooling::set_pooling_method(const string& new_pooling_method)
{
    pooling_method = string_to_pooling_method(new_pooling_method);

#ifdef OPENNN_WITH_CUDA
    if (pooling_method == PoolingMethod::MaxPooling)
        pooling_mode = CUDNN_POOLING_MAX;
    else
        pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
#endif
}

void Pooling::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    const TensorView& input = forward_views[Inputs][0];
    TensorView& output = forward_views[Outputs][0];

    if(pooling_method == PoolingMethod::MaxPooling)
        max_pooling(input, output, forward_views[MaximalIndices][0], cached_pool_args, is_training);
    else
        average_pooling(input, output, cached_pool_args);
}

void Pooling::back_propagate(ForwardPropagation& forward_propagation,
                             BackPropagation& back_propagation,
                             size_t layer) const noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    auto& backward_views = back_propagation.backward_views[layer];

    const TensorView& input = forward_views[Inputs][0];
    const TensorView& output = forward_views[Outputs][0];
    const TensorView& output_gradient = backward_views[OutputGradients][0];
    TensorView& input_gradient = backward_views[InputGradients][0];

    if(pooling_method == PoolingMethod::MaxPooling)
        max_pooling_backward(input, output, output_gradient, forward_views[MaximalIndices][0], input_gradient, cached_pool_args);
    else
        average_pooling_backward(input, output, output_gradient, input_gradient, cached_pool_args);
}

void Pooling::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Pooling");

    write_xml_properties(printer, {
        {"Label", label},
        {"InputDimensions", shape_to_string(get_input_shape())},
        {"PoolHeight", to_string(get_pool_height())},
        {"PoolWidth", to_string(get_pool_width())},
        {"PoolingMethod", pooling_method_to_string(pooling_method)},
        {"ColumnStride", to_string(get_column_stride())},
        {"RowStride", to_string(get_row_stride())},
        {"PaddingHeight", to_string(get_padding_height())},
        {"PaddingWidth", to_string(get_padding_width())}
    });

    printer.close_element();
}

void Pooling::from_XML(const XmlDocument& document)
{
    const XmlElement* pooling_layer_element = get_xml_root(document, "Pooling");

    set_label(read_xml_string(pooling_layer_element, "Label"));
    set_input_shape(string_to_shape(read_xml_string(pooling_layer_element, "InputDimensions")));
    set_pool_size(read_xml_index(pooling_layer_element, "PoolHeight"), read_xml_index(pooling_layer_element, "PoolWidth"));
    set_pooling_method(read_xml_string(pooling_layer_element, "PoolingMethod"));
    set_column_stride(read_xml_index(pooling_layer_element, "ColumnStride"));
    set_row_stride(read_xml_index(pooling_layer_element, "RowStride"));
    set_padding_height(read_xml_index(pooling_layer_element, "PaddingHeight"));
    set_padding_width(read_xml_index(pooling_layer_element, "PaddingWidth"));
}

// CUDA forward/backward handled via unified ForwardPropagation/BackPropagation views
// and operators in math_utilities.h

REGISTER(Layer, Pooling, "Pooling")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
