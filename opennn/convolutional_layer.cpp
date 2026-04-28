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
}

void Convolutional::set_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.rank != 3)
        throw runtime_error("Input new_input_shape.rank must be 3");

    input_height = new_input_shape[0];
    input_width = new_input_shape[1];
    input_channels = new_input_shape[2];
}

void Convolutional::set_row_stride(const Index new_stride_row)
{
    if (new_stride_row <= 0)
        throw runtime_error("EXCEPTION: new_stride_row must be a positive number");

    row_stride = new_stride_row;
}

void Convolutional::set_column_stride(const Index new_stride_column)
{
    if (new_stride_column <= 0)
        throw runtime_error("EXCEPTION: new_stride_column must be a positive number");

    column_stride = new_stride_column;
}

void Convolutional::set_convolution_type(const string& new_convolution_type)
{
    convolution_type = string_to_convolution_type(new_convolution_type);
    use_padding = (convolution_type == ConvolutionType::Same);
}

void Convolutional::set_activation_function(const string& new_activation_function)
{
    const ActivationFunction function = string_to_activation(new_activation_function);

    if (function == ActivationFunction::Softmax)
        throw runtime_error("Softmax is not a valid activation for a convolutional layer.");

    activation_arguments.activation_function = function;

#ifdef OPENNN_WITH_CUDA
    if (activation_arguments.activation_descriptor)
        cudnnSetActivationDescriptor(activation_arguments.activation_descriptor,
                                     to_cudnn_activation_mode(function),
                                     CUDNN_PROPAGATE_NAN, 0.0);
#endif
}

void Convolutional::set_batch_normalization(bool new_batch_normalization)
{
    batch_normalization = new_batch_normalization;
}

// Parameter initialization

void Convolutional::init_conv_norm_defaults()
{
    parameters[Bias].fill(0.0f);
    parameters[Gamma].fill(1.0f);
    parameters[Beta].fill(0.0f);
    if (batch_normalization && ssize(states) > RunningVariance)
    {
        states[RunningMean].fill(0.0f);
        states[RunningVariance].fill(1.0f);
    }
}

void Convolutional::set_parameters_glorot()
{
    const Index kernel_area = kernel_height * kernel_width;
    const Index fan_in  = kernel_area * kernel_channels;
    const Index fan_out = kernel_area * kernels_number;
    const type limit = sqrt(6.0f / static_cast<type>(fan_in + fan_out));

    set_random_uniform(VectorMap(parameters[Weight].as<float>(), parameters[Weight].size()), -limit, limit);
    init_conv_norm_defaults();
}

void Convolutional::set_parameters_random()
{
    set_random_uniform(VectorMap(parameters[Weight].as<float>(), parameters[Weight].size()));
    init_conv_norm_defaults();
}

#ifdef OPENNN_WITH_CUDA

void Convolutional::init_cuda(Index batch_size)
{
    // Filter + convolution descriptors

    if (!kernel_descriptor)
        cudnnCreateFilterDescriptor(&kernel_descriptor);

    cudnnSetFilter4dDescriptor(kernel_descriptor,
                               CUDNN_ACTIVATION_DTYPE,
                               CUDNN_TENSOR_NHWC,
                               kernels_number, kernel_channels, kernel_height, kernel_width);

    if (!convolution_descriptor)
        cudnnCreateConvolutionDescriptor(&convolution_descriptor);

    cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                    get_padding_height(), get_padding_width(),
                                    row_stride, column_stride,
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_ACTIVATION_DTYPE);

    cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH);

    convolution_arguments.convolution_descriptor = convolution_descriptor;
    convolution_arguments.kernel_descriptor = kernel_descriptor;

    // Activation descriptor

    cudnnActivationDescriptor_t& activation_descriptor = activation_arguments.activation_descriptor;
    if (!activation_descriptor)
        cudnnCreateActivationDescriptor(&activation_descriptor);
    cudnnSetActivationDescriptor(activation_descriptor,
                                 to_cudnn_activation_mode(activation_arguments.activation_function),
                                 CUDNN_PROPAGATE_NAN, 0.0);

    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);

    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NHWC, CUDNN_ACTIVATION_DTYPE,
                               static_cast<int>(batch_size),
                               static_cast<int>(kernel_channels),
                               static_cast<int>(input_height),
                               static_cast<int>(input_width));

    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);

    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NHWC, CUDNN_ACTIVATION_DTYPE,
                               static_cast<int>(batch_size),
                               static_cast<int>(kernels_number),
                               static_cast<int>(get_output_height()),
                               static_cast<int>(get_output_width()));

    int returned_count;

    cudnnConvolutionFwdAlgoPerf_t fwd_perf;
    cudnnFindConvolutionForwardAlgorithm(Device::get_cudnn_handle(),
                                         input_desc, kernel_descriptor, convolution_descriptor, output_desc,
                                         1, &returned_count, &fwd_perf);
    convolution_algorithm = fwd_perf.algo;

    cudnnGetConvolutionForwardWorkspaceSize(Device::get_cudnn_handle(),
                                            input_desc, kernel_descriptor, convolution_descriptor, output_desc,
                                            convolution_algorithm, &cuda_workspace_size);

    cudnnConvolutionBwdDataAlgoPerf_t data_perf;
    cudnnFindConvolutionBackwardDataAlgorithm(Device::get_cudnn_handle(),
                                              kernel_descriptor, output_desc, convolution_descriptor, input_desc,
                                              1, &returned_count, &data_perf);
    algo_data = data_perf.algo;

    cudnnConvolutionBwdFilterAlgoPerf_t filter_perf;
    cudnnFindConvolutionBackwardFilterAlgorithm(Device::get_cudnn_handle(),
                                                input_desc, output_desc, convolution_descriptor, kernel_descriptor,
                                                1, &returned_count, &filter_perf);
    algo_filter = filter_perf.algo;

    size_t bwd_data_ws = 0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(Device::get_cudnn_handle(),
                                                 kernel_descriptor, output_desc, convolution_descriptor, input_desc,
                                                 algo_data, &bwd_data_ws);

    cudnnGetConvolutionBackwardFilterWorkspaceSize(Device::get_cudnn_handle(),
                                                   input_desc, output_desc, convolution_descriptor, kernel_descriptor,
                                                   algo_filter, &cuda_backward_filter_workspace_size);

    cuda_workspace_size = max(cuda_workspace_size, bwd_data_ws);

    if (cuda_workspace) cudaFree(cuda_workspace);
    if (cuda_workspace_size > 0)
        CHECK_CUDA(cudaMalloc(&cuda_workspace, cuda_workspace_size));

    if (cuda_backward_filter_workspace) cudaFree(cuda_backward_filter_workspace);
    if (cuda_backward_filter_workspace_size > 0)
        CHECK_CUDA(cudaMalloc(&cuda_backward_filter_workspace, cuda_backward_filter_workspace_size));

    convolution_arguments.algorithm_forward = convolution_algorithm;
    convolution_arguments.algorithm_data = algo_data;
    convolution_arguments.algorithm_filter = algo_filter;
    convolution_arguments.workspace = cuda_workspace;
    convolution_arguments.workspace_size = cuda_workspace_size;
    convolution_arguments.backward_filter_workspace = cuda_backward_filter_workspace;
    convolution_arguments.backward_filter_workspace_size = cuda_backward_filter_workspace_size;

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
}

void Convolutional::destroy_cuda()
{
    if (activation_arguments.activation_descriptor) cudnnDestroyActivationDescriptor(activation_arguments.activation_descriptor);
    if (kernel_descriptor) cudnnDestroyFilterDescriptor(kernel_descriptor);
    if (convolution_descriptor) cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    if (cuda_workspace) cudaFree(cuda_workspace);
    if (cuda_backward_filter_workspace) cudaFree(cuda_backward_filter_workspace);
}

#endif

// Forward / back propagation

void Convolutional::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training) noexcept
{
    auto& forward_views = forward_propagation.views[layer];

    const TensorView& input = forward_views[Input][0];
    TensorView& padded_input = forward_views[PaddedInput][0];
    TensorView& output = forward_views[Output][0];

    const TensorView& weights = parameters[Weight];
    const TensorView& biases = parameters[Bias];
    const TensorView& gammas = parameters[Gamma];
    const TensorView& betas = parameters[Beta];

#ifdef OPENNN_WITH_CUDA
    const bool is_gpu = Device::instance().is_gpu();
#else
    constexpr bool is_gpu = false;
#endif

    if (!is_gpu)
    {
        if(use_padding)
            padding(input, padded_input);
        else
            copy(input, padded_input);
    }

    const TensorView& conv_input = is_gpu ? input : padded_input;

    if (batch_normalization)
    {
        TensorView& combination_output = forward_views[Convolution][0];
        convolution(conv_input, weights, biases, combination_output, convolution_arguments);

        if(is_training)
            batch_normalization_training(combination_output, gammas, betas,
                                         states[RunningMean], states[RunningVariance],
                                         forward_views[BatchNormMean][0], forward_views[BatchNormInverseVariance][0],
                                         output, momentum);
        else
            batch_normalization_inference(combination_output, gammas, betas,
                                          states[RunningMean], states[RunningVariance],
                                          output);

        activation(output, activation_arguments);
    }
    else
    {
        const ActivationFunction func = activation_arguments.activation_function;
        const bool can_fuse_gpu = is_gpu && func != ActivationFunction::Linear;

        if (can_fuse_gpu)
        {
            convolution_activation(conv_input, weights, biases, output, convolution_arguments, activation_arguments);
        }
        else
        {
            convolution(conv_input, weights, biases, output, convolution_arguments);
            activation(output, activation_arguments);
        }
    }
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
    const bool is_gpu = Device::instance().is_gpu();
#else
    constexpr bool is_gpu = false;
#endif

    activation_delta(output, output_delta, output_delta, activation_arguments);

    if (batch_normalization)
        batch_normalization_backward(forward_views[Convolution][0], output, output_delta,
                                     forward_views[BatchNormMean][0], forward_views[BatchNormInverseVariance][0],
                                     parameters[Gamma], gradient_views[Gamma], gradient_views[Beta],
                                     output_delta);

    const TensorView& conv_input = is_gpu ? forward_views[Input][0] : forward_views[PaddedInput][0];

    convolution_backward_weights(conv_input,
                                 output_delta,
                                 gradient_views[Weight],
                                 gradient_views[Bias],
                                 convolution_arguments);

    if (!is_first_layer)
        convolution_backward_data(output_delta,
                                  parameters[Weight],
                                  delta_views[InputDelta][0],
                                  convolution_arguments);
}

// Serialization

void Convolutional::from_XML(const XmlDocument& document)
{
    const XmlElement* convolutional_layer_element = get_xml_root(document, "Convolutional");

    set_label(read_xml_string(convolutional_layer_element, "Label"));

    set_input_shape(string_to_shape(read_xml_string(convolutional_layer_element, "InputDimensions")));

    kernel_height = read_xml_index(convolutional_layer_element, "KernelsHeight");
    kernel_width = read_xml_index(convolutional_layer_element, "KernelsWidth");
    kernel_channels = read_xml_index(convolutional_layer_element, "KernelsChannels");
    kernels_number = read_xml_index(convolutional_layer_element, "KernelsNumber");

    set_activation_function(read_xml_string(convolutional_layer_element, "Activation"));

    const Shape stride_shape = string_to_shape(read_xml_string(convolutional_layer_element, "StrideDimensions"));
    set_row_stride(stride_shape[0]);
    set_column_stride(stride_shape[1]);

    set_convolution_type(read_xml_string(convolutional_layer_element, "Convolution"));

    const XmlElement* batch_normalization_element = convolutional_layer_element->first_child_element("BatchNormalization");
    const bool use_batch_normalization = batch_normalization_element && batch_normalization_element->get_text()
                                         && string(batch_normalization_element->get_text()) == "true";
    set_batch_normalization(use_batch_normalization);

}

// Phase 2: runs after NN::compile(), so states[] is allocated. Parses BN running
// statistics directly into the arena — no staging required.
void Convolutional::load_state_from_XML(const XmlDocument& document)
{
    if(!batch_normalization) return;

    const XmlElement* convolutional_layer_element = get_xml_root(document, "Convolutional");

    VectorR tmp;
    string_to_vector(read_xml_string(convolutional_layer_element, "RunningMeans"), tmp);
    if(tmp.size() == states[RunningMean].size() && states[RunningMean].data)
        VectorMap(states[RunningMean].as<float>(), states[RunningMean].size()) = tmp;

    string_to_vector(read_xml_string(convolutional_layer_element, "RunningVariances"), tmp);
    if(tmp.size() == states[RunningVariance].size() && states[RunningVariance].data)
        VectorMap(states[RunningVariance].as<float>(), states[RunningVariance].size()) = tmp;
}

void Convolutional::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Convolutional");

    write_xml(printer, {
        {"Label", label},
        {"InputDimensions", shape_to_string(get_input_shape())},
        {"KernelsNumber", to_string(get_kernels_number())},
        {"KernelsHeight", to_string(get_kernel_height())},
        {"KernelsWidth", to_string(get_kernel_width())},
        {"KernelsChannels", to_string(get_kernel_channels())},
        {"Activation", activation_to_string(activation_arguments.activation_function)},
        {"StrideDimensions", shape_to_string({get_row_stride(), get_column_stride()})},
        {"Convolution", convolution_type_to_string(convolution_type)},
        {"BatchNormalization", batch_normalization ? "true" : "false"}
    });

    if (batch_normalization)
        write_xml(printer, {
            {"RunningMeans", vector_to_string(states[RunningMean].as_vector())},
            {"RunningVariances", vector_to_string(states[RunningVariance].as_vector())}
        });

    printer.close_element();
}

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

REGISTER(Layer, Convolutional, "Convolutional")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
