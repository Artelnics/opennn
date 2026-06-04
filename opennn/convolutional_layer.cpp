//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "convolutional_layer.h"
#include "neural_network.h"
#include "loss.h"

namespace opennn
{

Convolutional::Convolutional(const Shape& new_input_shape,
                             const Shape& new_kernel_shape,
                             const string& new_activation_function,
                             const Shape& new_stride_shape,
                             const string& new_convolution_type,
                             bool new_batch_normalization,
                             const string& new_label)
    : Layer(LayerType::Convolutional)
{
    operators = {&convolution, &batch_norm, &activation};

    set(new_input_shape,
        new_kernel_shape,
        new_activation_function,
        new_stride_shape,
        new_convolution_type,
        new_batch_normalization,
        new_label);
}

Shape Convolutional::get_output_shape() const
{
    return { get_output_height(), get_output_width(), kernels_number };
}

Index Convolutional::get_output_height() const
{
    return use_padding
        ? (input_height + row_stride - 1) / row_stride
        : (input_height - kernel_height) / row_stride + 1;
}

Index Convolutional::get_output_width() const
{
    return use_padding
        ? (input_width + column_stride - 1) / column_stride
        : (input_width - kernel_width) / column_stride + 1;
}

Index Convolutional::get_padding_height() const
{
    if (!use_padding) return 0;

    const Index total_padding = (get_output_height() - 1) * row_stride + kernel_height - input_height;

    return (total_padding + 1) / 2;
}

Index Convolutional::get_padding_width() const
{
    if (!use_padding) return 0;

    const Index total_padding = (get_output_width() - 1) * column_stride + kernel_width - input_width;

    return (total_padding + 1) / 2;
}

Index Convolutional::get_input_height() const { return input_height; }

Index Convolutional::get_input_width() const { return input_width; }

Index Convolutional::get_input_channels() const { return input_channels; }

vector<TensorSpec> Convolutional::get_forward_specs(Index batch_size) const
{
    const Shape output_shape = {batch_size, get_output_height(), get_output_width(), kernels_number};
    const Type act = compute_dtype;

    const Shape convolution_view_shape = batch_norm.active() ? output_shape          : Shape{};
    const Shape bn_stat_shape          = batch_norm.active() ? Shape{kernels_number} : Shape{};

    return {
        /*ConvolutionView*/          {convolution_view_shape, act},
        /*BatchNormMean*/            {bn_stat_shape,          Type::FP32},
        /*BatchNormInverseVariance*/ {bn_stat_shape,          Type::FP32},
        /*Output*/                   {output_shape,           act},
    };
}

void Convolutional::update_convolution_operator()
{
    convolution.set(input_height, input_width,
                    kernels_number, kernel_height, kernel_width, kernel_channels,
                    row_stride, column_stride,
                    get_padding_height(), get_padding_width(),
                    compute_dtype);

    convolution.output_slots = batch_norm.active() 
        ? vector<size_t>{ConvolutionView}
        : vector<size_t>{Output};

    if (batch_norm.active())
    {
        batch_norm.input_slots  = {ConvolutionView};
        batch_norm.output_slots = {Output, BatchNormMean, BatchNormInverseVariance};
    }

    activation.input_slots  = {Output};
    activation.output_slots = {Output};

    const bool fuse_relu = (activation.function == ActivationOp::Function::ReLU)
                           && !batch_norm.active();
    convolution.fused_activation = fuse_relu ? activation.descriptor : nullptr;
    activation.forward_fused     = fuse_relu;
}

void Convolutional::set(const Shape& new_input_shape,
                        const Shape& new_kernel_shape,
                        const string& new_activation_function,
                        const Shape& new_stride_shape,
                        const string& new_convolution_type,
                        bool new_batch_normalization,
                        const string& new_label)
{
    throw_if(new_kernel_shape.rank != 4, "Kernel shape must be 4");

    throw_if(new_stride_shape.rank != 2, "Stride shape must be 2");

    throw_if(new_kernel_shape[0] > new_input_shape[0] || new_kernel_shape[1] > new_input_shape[1],
             "kernel shape cannot be bigger than input shape");

    throw_if(new_kernel_shape[2] != new_input_shape[2],
             "kernel_channels must match input_channels dimension");

    throw_if(new_stride_shape[0] > new_input_shape[0] || new_stride_shape[1] > new_input_shape[1],
             "Stride shape cannot be bigger than input shape");

    throw_if(new_stride_shape[0] <= 0 || new_stride_shape[1] <= 0, "Stride must be positive.");

    throw_if(new_convolution_type != "Valid" && new_convolution_type != "Same",
             "Convolution type must be 'Valid' or 'Same'.");

    throw_if(new_convolution_type == "Same"
             && (new_kernel_shape[0] % 2 == 0 || new_kernel_shape[1] % 2 == 0),
             "Kernel shape (height and width) must be odd (3x3,5x5 etc) when using 'Same' padding mode to ensure symmetric padding.");

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

    set_activation_function(new_activation_function);
    set_batch_normalization(new_batch_normalization);

    update_convolution_operator();
}

void Convolutional::set_input_shape(const Shape& new_input_shape)
{
    throw_if(new_input_shape.rank != 3, "Input shape rank must be 3.");

    input_height = new_input_shape[0];
    input_width = new_input_shape[1];
    input_channels = new_input_shape[2];

    update_convolution_operator();
}

void Convolutional::set_row_stride(const Index new_stride_row)
{
    throw_if(new_stride_row <= 0, "Row stride must be positive.");

    row_stride = new_stride_row;

    update_convolution_operator();
}

void Convolutional::set_column_stride(const Index new_stride_column)
{
    throw_if(new_stride_column <= 0, "Column stride must be positive.");

    column_stride = new_stride_column;

    update_convolution_operator();
}

void Convolutional::set_convolution_type(const string& new_convolution_type)
{
    throw_if(new_convolution_type != "Valid" && new_convolution_type != "Same",
             "Convolution type must be 'Valid' or 'Same'.");

    use_padding = (new_convolution_type == "Same");

    update_convolution_operator();
}

void Convolutional::set_activation_function(const string& new_activation_function)
{
    const ActivationOp::Function function = ActivationOp::from_string(new_activation_function);

    throw_if(function == ActivationOp::Function::Softmax,
             "Softmax is not a valid activation for a convolutional layer.");

    activation.set_function(function);
}

void Convolutional::set_batch_normalization(bool new_batch_normalization)
{
    if (new_batch_normalization && kernels_number > 0)
        batch_norm.set(kernels_number, batch_norm.momentum);
    else
        batch_norm.features = 0;
}

void Convolutional::read_JSON_body(const Json* convolutional_layer_element)
{
    kernel_height   = read_json_index(convolutional_layer_element, "KernelsHeight");
    kernel_width    = read_json_index(convolutional_layer_element, "KernelsWidth");
    kernel_channels = read_json_index(convolutional_layer_element, "KernelsChannels");
    kernels_number  = read_json_index(convolutional_layer_element, "KernelsNumber");

    const Shape stride_shape = string_to_shape(read_json_string(convolutional_layer_element, "StrideDimensions"));
    row_stride      = stride_shape[0];
    column_stride   = stride_shape[1];

    const string convolution_type = read_json_string(convolutional_layer_element, "Convolution");
    throw_if(convolution_type != "Valid" && convolution_type != "Same",
             "Convolution type must be 'Valid' or 'Same'.");
    use_padding = (convolution_type == "Same");

    const bool has_batch_norm = read_json_bool(convolutional_layer_element, "BatchNormalization");
    if (has_batch_norm && kernels_number > 0)
        batch_norm.set(kernels_number, batch_norm.momentum);

    update_convolution_operator();
}

void Convolutional::write_JSON_body(JsonWriter& printer) const
{
    write_json(printer, {
        {"KernelsNumber", to_string(get_kernels_number())},
        {"KernelsHeight", to_string(get_kernel_height())},
        {"KernelsWidth", to_string(get_kernel_width())},
        {"KernelsChannels", to_string(get_kernel_channels())},
        {"StrideDimensions", shape_to_string({get_row_stride(), get_column_stride()})},
        {"Convolution", use_padding ? "Same" : "Valid"},
        {"BatchNormalization", to_string(batch_norm.active())}
    });
}

void Convolutional::from_JSON(const JsonDocument& document)
{
    Layer::from_JSON(document);

    update_convolution_operator();
}

REGISTER(Layer, Convolutional, "Convolutional")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
