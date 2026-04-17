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

void Convolutional::set(const Shape& new_input_shape,
                        const Shape& new_kernel_shape,
                        const string& new_activation_function,
                        const Shape& new_stride_shape,
                        const string& new_convolution_type,
                        bool new_batch_normalization,
                        const string& new_label)
{
    if (new_kernel_shape.rank() != 4)
        throw runtime_error("Kernel shape must be 4");

    if (new_stride_shape.rank() != 2)
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

#ifdef OPENNN_WITH_CUDA

    cudnnCreateFilterDescriptor(&kernel_descriptor);

    cudnnSetFilter4dDescriptor(kernel_descriptor,
                               CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NHWC,
                               kernels_number,
                               kernel_channels,
                               kernel_height,
                               kernel_width);

    cudnnCreateConvolutionDescriptor(&convolution_descriptor);

    cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                    get_padding_height(), get_padding_width(),
                                    get_row_stride(), get_column_stride(),
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);

    convolution_arguments.convolution_descriptor = convolution_descriptor;
    convolution_arguments.kernel_descriptor = kernel_descriptor;
#endif
}

void Convolutional::set_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.rank() != 3)
        throw runtime_error("Input new_input_shape.rank() must be 3");

    input_height = new_input_shape[0];
    input_width = new_input_shape[1];
    input_channels = new_input_shape[2];
}

void Convolutional::set_batch_normalization(bool new_batch_normalization)
{
    batch_normalization = new_batch_normalization;
}

void Convolutional::set_activation_function(const string& new_activation_function)
{
    const ActivationFunction function = string_to_activation(new_activation_function);

    if (function == ActivationFunction::Softmax)
        throw runtime_error("Softmax is not a valid activation for a convolutional layer.");

    activation_arguments.activation_function = function;

#ifdef OPENNN_WITH_CUDA
    cudnnActivationDescriptor_t& descriptor = activation_arguments.activation_descriptor;

    if (!descriptor)
        cudnnCreateActivationDescriptor(&descriptor);

    cudnnActivationMode_t activation_mode = CUDNN_ACTIVATION_IDENTITY;

    switch (function)
    {
    case ActivationFunction::Sigmoid:                 activation_mode = CUDNN_ACTIVATION_SIGMOID; break;
    case ActivationFunction::HyperbolicTangent:       activation_mode = CUDNN_ACTIVATION_TANH;    break;
    case ActivationFunction::RectifiedLinear:         activation_mode = CUDNN_ACTIVATION_RELU;    break;
    case ActivationFunction::ScaledExponentialLinear: activation_mode = CUDNN_ACTIVATION_ELU;     break;
    default: break;
    }

    cudnnSetActivationDescriptor(descriptor, activation_mode, CUDNN_PROPAGATE_NAN, 0.0);
#endif
}

void Convolutional::set_convolution_type(const string& new_convolution_type)
{
    convolution_type = string_to_convolution_type(new_convolution_type);
    use_padding = (convolution_type == ConvolutionType::Same);
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

void Convolutional::set_parameters_glorot()
{
    const Index kernel_area = get_kernel_height() * get_kernel_width();
    const Index fan_in = kernel_area * get_kernel_channels();
    const Index fan_out = kernel_area * get_kernels_number();

    const type limit = sqrt(6.0f / static_cast<type>(fan_in + fan_out));

    VectorMap(parameters[Bias].data, parameters[Bias].size()).setZero();

    set_random_uniform(VectorMap(parameters[Weight].data, parameters[Weight].size()), -limit, limit);

    VectorMap(parameters[Gamma].data, parameters[Gamma].size()).setConstant(1.0);

    VectorMap(parameters[Beta].data, parameters[Beta].size()).setZero();

    if (batch_normalization)
    {
        VectorMap(states[RunningMean].data, states[RunningMean].size()).setZero();
        VectorMap(states[RunningVariance].data, states[RunningVariance].size()).setOnes();
    }
}

void Convolutional::set_parameters_random()
{
    VectorMap(parameters[Bias].data, parameters[Bias].size()).setZero();

    set_random_uniform(VectorMap(parameters[Weight].data, parameters[Weight].size()));

    VectorMap(parameters[Gamma].data, parameters[Gamma].size()).setConstant(1.0);

    VectorMap(parameters[Beta].data, parameters[Beta].size()).setZero();

    if (batch_normalization)
    {
        VectorMap(states[RunningMean].data, states[RunningMean].size()).setZero();
        VectorMap(states[RunningVariance].data, states[RunningVariance].size()).setOnes();
    }
}

#ifdef OPENNN_WITH_CUDA

void Convolutional::init_cuda(Index batch_size)
{
    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);

    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
                               static_cast<int>(batch_size),
                               static_cast<int>(kernel_channels),
                               static_cast<int>(get_input_height()),
                               static_cast<int>(get_input_width()));

    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);

    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
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

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
}

#endif

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

    const ActivationFunction func = activation_arguments.activation_function;

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        convolution_arguments.algorithm_forward = convolution_algorithm;
        convolution_arguments.workspace = cuda_workspace;
        convolution_arguments.workspace_size = cuda_workspace_size;

        if (batch_normalization)
        {
            TensorView& combination_output = forward_views[Convolution][0];
            convolution(input, weights, biases, combination_output, convolution_arguments);

            if (is_training)
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
            if (func != ActivationFunction::Linear)
            {
                convolution_activation(input, weights, biases, output, convolution_arguments, activation_arguments);
            }
            else
            {
                convolution(input, weights, biases, output, convolution_arguments);
                activation(output, activation_arguments);
            }
        }
        return;
    }
#endif

    use_padding
        ? padding(input, padded_input)
        : copy(input, padded_input);

    if (batch_normalization)
    {
        TensorView& combination_output = forward_views[Convolution][0];
        convolution(padded_input, weights, biases, combination_output, convolution_arguments);

        is_training
            ? batch_normalization_training(combination_output, gammas, betas,
                                           states[RunningMean], states[RunningVariance],
                                           forward_views[BatchNormMean][0], forward_views[BatchNormInverseVariance][0],
                                           output, momentum)        
            : batch_normalization_inference(combination_output, gammas, betas,
                                          states[RunningMean], states[RunningVariance],
                                          output);
    }
    else
    {
        convolution(padded_input, weights, biases, output, convolution_arguments);
    }

    activation(output, activation_arguments);
}

void Convolutional::back_propagate(ForwardPropagation& forward_propagation,
                                   BackPropagation& back_propagation,
                                   size_t layer) const noexcept
{
    auto& forward_views = forward_propagation.views[layer];
    auto& backward_views = back_propagation.backward_views[layer];
    auto& gradient_views = back_propagation.gradient_views[layer];

    const TensorView& output = forward_views[Output][0];
    TensorView& delta = backward_views[OutputGradient][0];

    ConvolutionArguments bwd_args = convolution_arguments;

    activation_gradient(output, delta, delta, activation_arguments);

    if (batch_normalization)
        batch_normalization_backward(forward_views[Convolution][0], output, delta,
                                     forward_views[BatchNormMean][0], forward_views[BatchNormInverseVariance][0],
                                     parameters[Gamma], gradient_views[Gamma], gradient_views[Beta],
                                     delta);

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        bwd_args.algorithm_filter = algo_filter;
        bwd_args.algorithm_data = algo_data;
        bwd_args.workspace = cuda_workspace;
        bwd_args.workspace_size = cuda_workspace_size;
        bwd_args.backward_filter_workspace = cuda_backward_filter_workspace;
        bwd_args.backward_filter_workspace_size = cuda_backward_filter_workspace_size;

        convolution_backward_weights(forward_views[Input][0],
                                     delta,
                                     gradient_views[Weight],
                                     gradient_views[Bias],
                                     bwd_args);

        if (!is_first_layer)
            convolution_backward_data(delta,
                                      parameters[Weight],
                                      backward_views[InputGradient][0],
                                      backward_views[InputGradient][0],
                                      bwd_args);
        return;
    }
#endif

    convolution_backward_weights(forward_views[PaddedInput][0],
                                 delta,
                                 gradient_views[Weight],
                                 gradient_views[Bias],
                                 bwd_args);

    if (!is_first_layer)
        convolution_backward_data(delta,
                                  parameters[Weight],
                                  backward_views[InputGradient][0],
                                  backward_views[InputGradient][0],
                                  bwd_args);
}

void Convolutional::from_XML(const XmlDocument& document)
{
    const XmlElement* convolutional_layer_element = get_xml_root(document, "Convolutional");

    set_label(read_xml_string(convolutional_layer_element, "Label"));

    set_input_shape(string_to_shape(read_xml_string(convolutional_layer_element, "InputDimensions")));

    set_activation_function(read_xml_string(convolutional_layer_element, "Activation"));

    const Shape stride_shape = string_to_shape(read_xml_string(convolutional_layer_element, "StrideDimensions"));
    set_row_stride(stride_shape[0]);
    set_column_stride(stride_shape[1]);

    set_convolution_type(read_xml_string(convolutional_layer_element, "Convolution"));

    const XmlElement* batch_normalization_element = convolutional_layer_element->first_child_element("BatchNormalization");
    const bool use_batch_normalization = batch_normalization_element && batch_normalization_element->get_text()
                                         && string(batch_normalization_element->get_text()) == "true";
    set_batch_normalization(use_batch_normalization);

    if (batch_normalization)
    {
        VectorR tmp;

        string_to_vector(read_xml_string(convolutional_layer_element, "RunningMeans"), tmp);
        VectorMap(states[RunningMean].data, states[RunningMean].size()) = tmp;

        string_to_vector(read_xml_string(convolutional_layer_element, "RunningVariances"), tmp);
        VectorMap(states[RunningVariance].data, states[RunningVariance].size()) = tmp;
    }
}

void Convolutional::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Convolutional");

    write_xml_properties(printer, {
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
        write_xml_properties(printer, {
            {"RunningMeans", vector_to_string(states[RunningMean].as_vector())},
            {"RunningVariances", vector_to_string(states[RunningVariance].as_vector())}
        });

    printer.close_element();
}

Shape Convolutional::get_output_shape() const
{
    return { get_output_height(), get_output_width(), get_kernels_number() };
}

Index Convolutional::get_output_height() const
{
    const Index input_height = get_input_height();
    const Index stride = get_row_stride();

    return (convolution_type == ConvolutionType::Same)
        ? (input_height + stride - 1) / stride
        : (input_height - get_kernel_height()) / stride + 1;
}

Index Convolutional::get_output_width() const
{
    const Index input_width = get_input_width();
    const Index stride = get_column_stride();

    return (convolution_type == ConvolutionType::Same)
        ? (input_width + stride - 1) / stride
        : (input_width - get_kernel_width()) / stride + 1;
}

pair<Index, Index> Convolutional::get_padding() const
{
    return { get_padding_height(), get_padding_width() };
}

Index Convolutional::get_padding_height() const
{
    if (convolution_type == ConvolutionType::Valid)
        return 0;

    const Index input_height = get_input_height();
    const Index stride = get_row_stride();
    const Index output_height = (input_height + stride - 1) / stride;
    const Index total_padding = (output_height - 1) * stride + get_kernel_height() - input_height;

    return total_padding / 2;
}

Index Convolutional::get_padding_width() const
{
    if (convolution_type == ConvolutionType::Valid)
        return 0;

    const Index input_width = get_input_width();
    const Index stride = get_column_stride();
    const Index output_width = (input_width + stride - 1) / stride;
    const Index total_padding = (output_width - 1) * stride + get_kernel_width() - input_width;

    return total_padding / 2;
}

Index Convolutional::get_input_height() const { return input_height; }

Index Convolutional::get_input_width() const { return input_width; }

Index Convolutional::get_input_channels() const { return input_channels; }

vector<Shape> Convolutional::get_forward_shapes(const Index batch_size) const
{
    const Index input_height = get_input_height();
    const Index input_width = get_input_width();
    const Index input_channels = get_input_channels();

    const Index output_height = get_output_height();
    const Index output_width = get_output_width();

    const Index padding_height = get_padding_height();
    const Index padding_width = get_padding_width();

    const Shape output_shape = {batch_size, output_height, output_width, kernels_number};
    const Shape padded_shape = {batch_size,
                                input_height + 2 * padding_height,
                                input_width + 2 * padding_width,
                                input_channels};

    if (batch_normalization)
        return {padded_shape,             // PaddedInput
                output_shape,             // Convolution
                Shape{kernels_number},    // BatchNormMean
                Shape{kernels_number},    // BatchNormInverseVariance
                output_shape};            // Output

    return {padded_shape,                 // PaddedInput
            Shape{},                      // Convolution (unused)
            Shape{},                      // BatchNormMean (unused)
            Shape{},                      // BatchNormInverseVariance (unused)
            output_shape};                // Output
}

REGISTER(Layer, Convolutional, "Convolutional")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
