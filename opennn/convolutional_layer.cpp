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
    return (convolution_type == ConvolutionType::Same)
        ? (input_height + row_stride - 1) / row_stride
        : (input_height - kernel_height) / row_stride + 1;
}

Index Convolutional::get_output_width() const
{
    return (convolution_type == ConvolutionType::Same)
        ? (input_width + column_stride - 1) / column_stride
        : (input_width - kernel_width) / column_stride + 1;
}

pair<Index, Index> Convolutional::get_padding() const
{
    return { get_padding_height(), get_padding_width() };
}

Index Convolutional::get_padding_height() const
{
    if (convolution_type == ConvolutionType::Valid)
        return 0;

    const Index output_height = (input_height + row_stride - 1) / row_stride;
    const Index total_padding = (output_height - 1) * row_stride + kernel_height - input_height;

    return total_padding / 2;
}

Index Convolutional::get_padding_width() const
{
    if (convolution_type == ConvolutionType::Valid)
        return 0;

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
    if (batch_normalization) ops.push_back(&batch_norm);
    return ops;
}

void Convolutional::configure_operators()
{
    convolution.set(input_height, input_width, input_channels,
                    kernels_number, kernel_height, kernel_width, kernel_channels,
                    row_stride, column_stride,
                    get_padding_height(), get_padding_width(),
                    activation_dtype);

    if (batch_normalization && kernels_number > 0)
        batch_norm.set(kernels_number, momentum);
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

    if (string_to_convolution_type(new_convolution_type) == ConvolutionType::Same
        && (new_kernel_shape[0] % 2 == 0 || new_kernel_shape[1] % 2 == 0))
        throw runtime_error("Kernel shape (height and width) must be odd (3x3,5x5 etc) when using 'Same' padding mode to ensure symmetric padding.");

    input_height = new_input_shape[0];
    input_width = new_input_shape[1];
    input_channels = new_input_shape[2];

    kernel_height = new_kernel_shape[0];
    kernel_width = new_kernel_shape[1];
    kernel_channels = new_kernel_shape[2];
    kernels_number = new_kernel_shape[3];

    set_row_stride(new_stride_shape[0]);
    set_column_stride(new_stride_shape[1]);

    set_convolution_type(new_convolution_type);

    set_activation_function(new_activation_function);

    set_batch_normalization(new_batch_normalization);

    set_label(new_label);

    configure_operators();
}

void Convolutional::set_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.rank != 3)
        throw runtime_error("Input new_input_shape.rank must be 3");

    input_height = new_input_shape[0];
    input_width = new_input_shape[1];
    input_channels = new_input_shape[2];

    configure_operators();
}

void Convolutional::set_row_stride(const Index new_stride_row)
{
    if (new_stride_row <= 0)
        throw runtime_error("Row stride must be positive.");

    row_stride = new_stride_row;
}

void Convolutional::set_column_stride(const Index new_stride_column)
{
    if (new_stride_column <= 0)
        throw runtime_error("Column stride must be positive.");

    column_stride = new_stride_column;
}

void Convolutional::set_convolution_type(const string& new_convolution_type)
{
    convolution_type = string_to_convolution_type(new_convolution_type);
    use_padding = (convolution_type == ConvolutionType::Same);
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
    batch_normalization = new_batch_normalization;
}

void Convolutional::set_parameters_glorot()
{
    const Index kernel_area = kernel_height * kernel_width;
    const Index fan_in  = kernel_area * kernel_channels;
    const Index fan_out = kernel_area * kernels_number;
    const float limit = sqrt(6.0f / static_cast<float>(fan_in + fan_out));

    set_random_uniform(parameters[Weight].as_vector(), -limit, limit);
    parameters[Bias].fill(0.0f);
    if (batch_normalization) batch_norm.init_defaults();
}

void Convolutional::set_parameters_random()
{
    set_random_uniform(parameters[Weight].as_vector());
    parameters[Bias].fill(0.0f);
    if (batch_normalization) batch_norm.init_defaults();
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

    if (batch_normalization)
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

    if (batch_normalization)
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

    set_input_shape(string_to_shape(read_json_string(convolutional_layer_element, "InputDimensions")));

    kernel_height   = read_json_index(convolutional_layer_element, "KernelsHeight");
    kernel_width    = read_json_index(convolutional_layer_element, "KernelsWidth");
    kernel_channels = read_json_index(convolutional_layer_element, "KernelsChannels");
    kernels_number  = read_json_index(convolutional_layer_element, "KernelsNumber");

    const Shape stride_shape = string_to_shape(read_json_string(convolutional_layer_element, "StrideDimensions"));
    set_row_stride(stride_shape[0]);
    set_column_stride(stride_shape[1]);

    set_convolution_type(read_json_string(convolutional_layer_element, "Convolution"));
    set_batch_normalization(read_json_bool(convolutional_layer_element, "BatchNormalization"));

    activation.from_JSON(convolutional_layer_element);
    if (batch_normalization)
    {
        batch_norm.from_JSON(convolutional_layer_element);
        momentum = batch_norm.momentum;
    }
}

void Convolutional::load_state_from_JSON(const JsonDocument& document)
{
    if (!batch_normalization) return;

    const Json* convolutional_layer_element = get_json_root(document, "Convolutional");

    VectorR tmp;
    string_to_vector(read_json_string(convolutional_layer_element, "RunningMeans"), tmp);
    if (tmp.size() == states[RunningMean].size() && states[RunningMean].data)
        states[RunningMean].as_vector() = tmp;

    string_to_vector(read_json_string(convolutional_layer_element, "RunningVariances"), tmp);
    if (tmp.size() == states[RunningVariance].size() && states[RunningVariance].data)
        states[RunningVariance].as_vector() = tmp;
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
        {"Convolution", convolution_type_to_string(convolution_type)},
        {"BatchNormalization", to_string(batch_normalization)}
    });

    activation.to_JSON(printer);
    if (batch_normalization) batch_norm.to_JSON(printer);

    if (batch_normalization)
        write_json(printer, {
            {"RunningMeans", vector_to_string(states[RunningMean].as_vector())},
            {"RunningVariances", vector_to_string(states[RunningVariance].as_vector())}
        });

    printer.close_element();
}

REGISTER(Layer, Convolutional, "Convolutional")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
