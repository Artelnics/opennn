//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   R E L U   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "convolutional_relu_layer.h"
#include "neural_network.h"
#include "loss.h"

namespace opennn
{

ConvolutionalRelu::ConvolutionalRelu(const Shape& new_input_shape,
                                     const Shape& new_kernel_shape,
                                     const Shape& new_stride_shape,
                                     const string& new_convolution_type,
                                     const string& new_label)
    : Layer("ConvolutionalRelu", LayerType::ConvolutionalRelu)
{
    operators = {&convolution_relu};
    set(new_input_shape,
        new_kernel_shape,
        new_stride_shape,
        new_convolution_type,
        new_label);
}

Shape ConvolutionalRelu::get_output_shape() const
{
    return { get_output_height(), get_output_width(), kernels_number };
}

Index ConvolutionalRelu::get_output_height() const
{
    return use_padding
        ? (input_height + row_stride - 1) / row_stride
        : (input_height - kernel_height) / row_stride + 1;
}

Index ConvolutionalRelu::get_output_width() const
{
    return use_padding
        ? (input_width + column_stride - 1) / column_stride
        : (input_width - kernel_width) / column_stride + 1;
}

Index ConvolutionalRelu::get_padding_height() const
{
    if (!use_padding) return 0;
    const Index total_padding = (get_output_height() - 1) * row_stride + kernel_height - input_height;
    return total_padding / 2;
}

Index ConvolutionalRelu::get_padding_width() const
{
    if (!use_padding) return 0;
    const Index total_padding = (get_output_width() - 1) * column_stride + kernel_width - input_width;
    return total_padding / 2;
}

vector<pair<Shape, Type>> ConvolutionalRelu::get_forward_specs(Index batch_size) const
{
    return {{{batch_size, get_output_height(), get_output_width(), kernels_number}, compute_dtype}};
}

void ConvolutionalRelu::update_convolution_operator()
{
    convolution_relu.set(input_height, input_width,
                         kernels_number, kernel_height, kernel_width, kernel_channels,
                         row_stride, column_stride,
                         get_padding_height(), get_padding_width(),
                         compute_dtype);

    convolution_relu.input_slots  = {Input};
    convolution_relu.output_slots = {Output};

    convolution_relu.output_delta_slots = {OutputDelta};
    convolution_relu.input_delta_slots  = {InputDelta};
}

void ConvolutionalRelu::set(const Shape& new_input_shape,
                            const Shape& new_kernel_shape,
                            const Shape& new_stride_shape,
                            const string& new_convolution_type,
                            const string& new_label)
{
    if (new_kernel_shape.rank != 4)
        throw runtime_error("Kernel shape must be 4");

    if (new_stride_shape.rank != 2)
        throw runtime_error("Stride shape must be 2");

    if (new_kernel_shape[0] > new_input_shape[0] || new_kernel_shape[1] > new_input_shape[1])
        throw runtime_error("kernel shape cannot be bigger than input shape");

    if (new_kernel_shape[2] != new_input_shape[2])
        throw runtime_error("kernel_channels must match input_channels dimension");

    if (new_stride_shape[0] > new_input_shape[0] || new_stride_shape[1] > new_input_shape[1])
        throw runtime_error("Stride shape cannot be bigger than input shape");

    if (new_stride_shape[0] <= 0 || new_stride_shape[1] <= 0)
        throw runtime_error("Stride must be positive.");

    if (new_convolution_type != "Valid" && new_convolution_type != "Same")
        throw runtime_error("Convolution type must be 'Valid' or 'Same'.");

    if (new_convolution_type == "Same"
        && (new_kernel_shape[0] % 2 == 0 || new_kernel_shape[1] % 2 == 0))
        throw runtime_error("Kernel shape (height and width) must be odd (3x3,5x5 etc) when using 'Same' padding mode to ensure symmetric padding.");

    input_height    = new_input_shape[0];
    input_width     = new_input_shape[1];
    input_channels  = new_input_shape[2];

    kernel_height   = new_kernel_shape[0];
    kernel_width    = new_kernel_shape[1];
    kernel_channels = new_kernel_shape[2];
    kernels_number  = new_kernel_shape[3];

    row_stride      = new_stride_shape[0];
    column_stride   = new_stride_shape[1];

    use_padding     = (new_convolution_type == "Same");

    set_label(new_label);

    update_convolution_operator();
}

void ConvolutionalRelu::set_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.rank != 3)
        throw runtime_error("Input shape rank must be 3");

    input_height = new_input_shape[0];
    input_width = new_input_shape[1];
    input_channels = new_input_shape[2];

    update_convolution_operator();
}

void ConvolutionalRelu::set_row_stride(const Index new_stride_row)
{
    if (new_stride_row <= 0)
        throw runtime_error("Row stride must be positive.");
    row_stride = new_stride_row;
    update_convolution_operator();
}

void ConvolutionalRelu::set_column_stride(const Index new_stride_column)
{
    if (new_stride_column <= 0)
        throw runtime_error("Column stride must be positive.");
    column_stride = new_stride_column;
    update_convolution_operator();
}

void ConvolutionalRelu::set_convolution_type(const string& new_convolution_type)
{
    if (new_convolution_type != "Valid" && new_convolution_type != "Same")
        throw runtime_error("Convolution type must be 'Valid' or 'Same'.");
    use_padding = (new_convolution_type == "Same");
    update_convolution_operator();
}

void ConvolutionalRelu::read_JSON_body(const Json* element)
{
    kernel_height   = read_json_index(element, "KernelsHeight");
    kernel_width    = read_json_index(element, "KernelsWidth");
    kernel_channels = read_json_index(element, "KernelsChannels");
    kernels_number  = read_json_index(element, "KernelsNumber");

    const Shape stride_shape = string_to_shape(read_json_string(element, "StrideDimensions"));
    row_stride      = stride_shape[0];
    column_stride   = stride_shape[1];

    const string convolution_type = read_json_string(element, "Convolution");
    if (convolution_type != "Valid" && convolution_type != "Same")
        throw runtime_error("Convolution type must be 'Valid' or 'Same'.");
    use_padding = (convolution_type == "Same");

    update_convolution_operator();
}

void ConvolutionalRelu::write_JSON_body(JsonWriter& printer) const
{
    write_json(printer, {
        {"KernelsNumber", to_string(get_kernels_number())},
        {"KernelsHeight", to_string(get_kernel_height())},
        {"KernelsWidth", to_string(get_kernel_width())},
        {"KernelsChannels", to_string(get_kernel_channels())},
        {"StrideDimensions", shape_to_string({get_row_stride(), get_column_stride()})},
        {"Convolution", use_padding ? "Same" : "Valid"}
    });
}

REGISTER(Layer, ConvolutionalRelu, "ConvolutionalRelu")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
