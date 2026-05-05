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
                             const string& new_label) : Layer()
{
    name = "Convolutional";
    layer_type = LayerType::Convolutional;

    set(new_input_shape,
        new_kernel_shape,
        new_activation_function,
        new_stride_shape,
        new_convolution_type,
        new_batch_normalization,
        new_label);
}

// Getters

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

pair<Index, Index> Convolutional::get_padding() const
{
    return { get_padding_height(), get_padding_width() };
}

Index Convolutional::get_padding_height() const
{
    if (!use_padding) return 0;

    const Index output_height = (input_height + row_stride - 1) / row_stride;
    const Index total_padding = (output_height - 1) * row_stride + kernel_height - input_height;

    return total_padding / 2;
}

Index Convolutional::get_padding_width() const
{
    if (!use_padding) return 0;

    const Index output_width = (input_width + column_stride - 1) / column_stride;
    const Index total_padding = (output_width - 1) * column_stride + kernel_width - input_width;

    return total_padding / 2;
}

Index Convolutional::get_input_height() const { return input_height; }

Index Convolutional::get_input_width() const { return input_width; }

Index Convolutional::get_input_channels() const { return input_channels; }

vector<Operator*> Convolutional::get_operators()
{
    vector<Operator*> ops = {&convolution};
    if (batch_norm.active()) ops.push_back(&batch_norm);
    return ops;
}

vector<pair<Shape, Type>> Convolutional::get_forward_specs(Index batch_size) const
{
    const Shape output_shape = {batch_size, get_output_height(), get_output_width(), kernels_number};
    const Shape padded_shape = Configuration::instance().is_gpu()
        ? Shape{}
        : Shape{batch_size,
                input_height + 2 * get_padding_height(),
                input_width + 2 * get_padding_width(),
                input_channels};
    const Type act = compute_dtype;

    const Shape convolution_view_shape = batch_norm.active() ? output_shape          : Shape{};
    const Shape bn_stat_shape          = batch_norm.active() ? Shape{kernels_number} : Shape{};

    return {
        /*PaddedInput*/              {padded_shape,           act},
        /*ConvolutionView*/          {convolution_view_shape, act},
        /*BatchNormMean*/            {bn_stat_shape,          Type::FP32},
        /*BatchNormInverseVariance*/ {bn_stat_shape,          Type::FP32},
        /*Output*/                   {output_shape,           act},
    };
}

vector<pair<Shape, Type>> Convolutional::get_backward_specs(Index batch_size) const
{
    return {{{batch_size, input_height, input_width, input_channels}, compute_dtype}};
}

void Convolutional::update_convolution_operator()
{
    convolution.set(input_height, input_width,
                    kernels_number, kernel_height, kernel_width, kernel_channels,
                    row_stride, column_stride,
                    get_padding_height(), get_padding_width(),
                    compute_dtype);
}

// Setters

void Convolutional::set(const Shape& new_input_shape,
                        const Shape& new_kernel_shape,
                        const string& new_activation_function,
                        const Shape& new_stride_shape,
                        const string& new_convolution_type,
                        bool new_batch_normalization,
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

    set_activation_function(new_activation_function);
    set_batch_normalization(new_batch_normalization);
}

void Convolutional::set_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.rank != 3)
        throw runtime_error("Input new_input_shape.rank must be 3");

    input_height = new_input_shape[0];
    input_width = new_input_shape[1];
    input_channels = new_input_shape[2];

    update_convolution_operator();
}

void Convolutional::set_row_stride(const Index new_stride_row)
{
    if (new_stride_row <= 0)
        throw runtime_error("Row stride must be positive.");

    row_stride = new_stride_row;

    update_convolution_operator();
}

void Convolutional::set_column_stride(const Index new_stride_column)
{
    if (new_stride_column <= 0)
        throw runtime_error("Column stride must be positive.");

    column_stride = new_stride_column;

    update_convolution_operator();
}

void Convolutional::set_convolution_type(const string& new_convolution_type)
{
    if (new_convolution_type != "Valid" && new_convolution_type != "Same")
        throw runtime_error("Convolution type must be 'Valid' or 'Same'.");

    use_padding = (new_convolution_type == "Same");

    update_convolution_operator();
}

void Convolutional::set_activation_function(const string& new_activation_function)
{
    const Activation::Function function = Activation::from_string(new_activation_function);

    if (function == Activation::Function::Softmax)
        throw runtime_error("Softmax is not a valid activation for a convolutional layer.");

    activation.set_function(function);
}

void Convolutional::set_batch_normalization(bool new_batch_normalization)
{
    if (new_batch_normalization && kernels_number > 0)
        batch_norm.set(kernels_number, batch_norm.momentum);
    else
        batch_norm.features = 0;
}

void Convolutional::set_parameters_glorot()
{
    const Index kernel_area = kernel_height * kernel_width;
    const Index fan_in  = kernel_area * kernel_channels;
    const Index fan_out = kernel_area * kernels_number;
    const float limit = sqrt(6.0f / static_cast<float>(fan_in + fan_out));

    set_random_uniform(parameters[Weight].as_vector(), -limit, limit);
    parameters[Bias].fill(0.0f);
    if (batch_norm.active()) batch_norm.init_defaults();
}

void Convolutional::set_parameters_random()
{
    set_random_uniform(parameters[Weight].as_vector());
    parameters[Bias].fill(0.0f);
    if (batch_norm.active()) batch_norm.init_defaults();
}

// Forward / back propagation

void Convolutional::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    const TensorView& input = forward_views[Input][0];
    TensorView& padded_input = forward_views[PaddedInput][0];
    TensorView& output = forward_views[Output][0];

#ifdef OPENNN_WITH_CUDA
    const bool is_gpu = Configuration::instance().is_gpu();
#else
    constexpr bool is_gpu = false;
#endif

    if (!is_gpu)
    {
        if (use_padding)
            padding(input, padded_input);
        else
            copy(input, padded_input);
    }

    const TensorView& conv_input = is_gpu ? input : padded_input;

    if (batch_norm.active())
    {
        TensorView& combination_output = forward_views[ConvolutionView][0];
        convolution.apply(conv_input, combination_output);

        if (is_training)
            batch_norm.apply_training(combination_output,
                                      forward_views[BatchNormMean][0],
                                      forward_views[BatchNormInverseVariance][0],
                                      output);
        else
            batch_norm.apply_inference(combination_output, output);

        activation.apply(output);
        return;
    }

    const bool fuse_activation = is_gpu && activation.function == Activation::Function::ReLU;

    convolution.apply(conv_input, output, fuse_activation ? activation.descriptor : nullptr);
    if (!fuse_activation) activation.apply(output);
}

void Convolutional::back_propagate(ForwardPropagation& forward_propagation,
                                   BackPropagation& back_propagation,
                                   size_t layer) const noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    auto& delta_views = back_propagation.delta_views[layer];
    auto& gradient_views = back_propagation.gradient_views[layer];

    const TensorView& output = forward_views[Output][0];
    TensorView& output_delta = delta_views[OutputDelta][0];

#ifdef OPENNN_WITH_CUDA
    const bool is_gpu = Configuration::instance().is_gpu();
#else
    constexpr bool is_gpu = false;
#endif

    activation.apply_delta(output, output_delta);

    if (batch_norm.active())
        batch_norm.apply_delta(forward_views[ConvolutionView][0],
                               forward_views[BatchNormMean][0],
                               forward_views[BatchNormInverseVariance][0],
                               gradient_views[Gamma],
                               gradient_views[Beta],
                               output_delta);

    const TensorView& conv_input = is_gpu ? forward_views[Input][0] : forward_views[PaddedInput][0];

    TensorView empty_input_delta;
    TensorView& input_delta_arg = is_first_layer ? empty_input_delta : delta_views[InputDelta][0];

    convolution.apply_delta(conv_input,
                            output_delta,
                            gradient_views[Weight],
                            gradient_views[Bias],
                            input_delta_arg);
}

// Serialization

void Convolutional::from_JSON(const JsonDocument& document)
{
    const Json* convolutional_layer_element = get_json_root(document, "Convolutional");

    set_label(read_json_string(convolutional_layer_element, "Label"));

    const Shape input_shape = string_to_shape(read_json_string(convolutional_layer_element, "InputDimensions"));
    if (input_shape.rank != 3)
        throw runtime_error("Input shape rank must be 3");

    input_height    = input_shape[0];
    input_width     = input_shape[1];
    input_channels  = input_shape[2];

    kernel_height   = read_json_index(convolutional_layer_element, "KernelsHeight");
    kernel_width    = read_json_index(convolutional_layer_element, "KernelsWidth");
    kernel_channels = read_json_index(convolutional_layer_element, "KernelsChannels");
    kernels_number  = read_json_index(convolutional_layer_element, "KernelsNumber");

    const Shape stride_shape = string_to_shape(read_json_string(convolutional_layer_element, "StrideDimensions"));
    row_stride      = stride_shape[0];
    column_stride   = stride_shape[1];

    const string convolution_type = read_json_string(convolutional_layer_element, "Convolution");
    if (convolution_type != "Valid" && convolution_type != "Same")
        throw runtime_error("Convolution type must be 'Valid' or 'Same'.");
    use_padding = (convolution_type == "Same");

    const bool has_batch_norm = read_json_bool(convolutional_layer_element, "BatchNormalization");

    activation.from_JSON(convolutional_layer_element);
    if (has_batch_norm) batch_norm.from_JSON(convolutional_layer_element);

    update_convolution_operator();
    if (has_batch_norm && kernels_number > 0)
        batch_norm.set(kernels_number, batch_norm.momentum);
}

void Convolutional::load_state_from_JSON(const JsonDocument& document)
{
    if (!batch_norm.active()) return;

    const Json* convolutional_layer_element = get_json_root(document, "Convolutional");

    batch_norm.load_state_from_JSON(convolutional_layer_element);
}

void Convolutional::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Convolutional");

    write_json(printer, {
        {"Label", label},
        {"InputDimensions", shape_to_string(get_input_shape())},
        {"KernelsNumber", to_string(get_kernels_number())},
        {"KernelsHeight", to_string(get_kernel_height())},
        {"KernelsWidth", to_string(get_kernel_width())},
        {"KernelsChannels", to_string(get_kernel_channels())},
        {"StrideDimensions", shape_to_string({get_row_stride(), get_column_stride()})},
        {"Convolution", use_padding ? "Same" : "Valid"},
        {"BatchNormalization", to_string(batch_norm.active())}
    });

    activation.to_JSON(printer);
    if (batch_norm.active()) batch_norm.to_JSON(printer);

    printer.close_element();
}

REGISTER(Layer, Convolutional, "Convolutional")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
