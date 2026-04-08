//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
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
                             bool new_batch_normaliztion,
                             const string& new_name) : Layer()
{
    name = "Convolutional";

    set(new_input_shape,
        new_kernel_shape,
        new_activation_function,
        new_stride_shape,
        new_convolution_type,
        new_batch_normaliztion,
        new_name);
}

void Convolutional::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training)
{
    const TensorView& input = forward_propagation.views[layer][Inputs][0];
    TensorView& padded_input = forward_propagation.views[layer][PaddedInputs][0];

    convolution_type == "Same"
        ? padding(input, padded_input)
        : copy(input, padded_input);

    TensorView& output = forward_propagation.views[layer].back()[0];

    convolution(padded_input, parameters[Weights], parameters[Biases], output);

    activation(output, {activation_function});
}

void Convolutional::back_propagate(ForwardPropagation& forward_propagation,
                                   BackPropagation& back_propagation,
                                   size_t layer) const
{
    const TensorView& output = forward_propagation.views[layer].back()[0];
    TensorView& delta = back_propagation.backward_views[layer][OutputGradients][0];

    activation_gradient(output, delta, delta, activation_function);

    convolution_backward_weights(forward_propagation.views[layer][PaddedInputs][0],
                                 delta,
                                 back_propagation.gradient_views[layer][Weights],
                                 back_propagation.gradient_views[layer][Biases]);

    if(!is_first_layer)
        convolution_backward_data(delta,
                                  parameters[Weights],
                                  back_propagation.backward_views[layer][InputGradients][0],
                                  back_propagation.backward_views[layer][InputGradients + 1][0]);
}

Index Convolutional::get_output_height() const
{
    return (convolution_type == "Same")
        ? (get_input_height() + get_row_stride() - 1) / get_row_stride()
        : (get_input_height() - get_kernel_height()) / get_row_stride() + 1;
}

Index Convolutional::get_output_width() const
{
    return (convolution_type == "Same")
        ? (get_input_width() + get_column_stride() - 1) / get_column_stride()
        : (get_input_width() - get_kernel_width()) / get_column_stride() + 1;
}

Shape Convolutional::get_output_shape() const
{
    return { get_output_height(), get_output_width(), get_kernels_number() };
}

Index Convolutional::get_padding_height() const
{
    if (convolution_type == "Valid")
        return 0;

    if (convolution_type == "Same")
    {
        const Index output_height = (get_input_height() + get_row_stride() - 1) / get_row_stride();

        const Index total_padding = (output_height - 1) * get_row_stride() + get_kernel_height() - get_input_height();

        return total_padding / 2;
    }

    throw runtime_error("Unknown convolution type");
}

Index Convolutional::get_padding_width() const
{
    if (convolution_type == "Valid")
        return 0;

    if (convolution_type == "Same")
    {
        const Index output_width = (get_input_width() + get_column_stride() - 1) / get_column_stride();

        const Index total_padding = (output_width - 1) * get_column_stride() + get_kernel_width() - get_input_width();

        return total_padding / 2;
    }

    throw runtime_error("Unknown convolution type");
}

void Convolutional::set(const Shape& new_input_shape,
                        const Shape& new_kernel_shape,
                        const string& new_activation_function,
                        const Shape& new_stride_shape,
                        const string& new_convolution_type,
                        bool new_batch_normalization,
                        const string& new_label)
{
    if(new_kernel_shape.rank != 4)
        throw runtime_error("Kernel shape must be 4");

    if (new_stride_shape.rank != 2)
        throw runtime_error("Stride shape must be 2");

    if (new_kernel_shape[0] > new_input_shape[0] || new_kernel_shape[1] > new_input_shape[1])
        throw runtime_error("kernel shape cannot be bigger than input shape");

    if (new_kernel_shape[2] != new_input_shape[2])
        throw runtime_error("kernel_channels must match input_channels dimension");

    if (new_stride_shape[0] > new_input_shape[0] || new_stride_shape[1] > new_input_shape[1])
        throw runtime_error("Stride shape cannot be bigger than input shape");

    if (new_convolution_type == "Same" && (new_kernel_shape[0] % 2 == 0 || new_kernel_shape[1] % 2 == 0))
        throw runtime_error("Kernel shape (height and width) must be odd (3x3,5x5 etc) when using 'Same' padding mode to ensure symmetric padding.");

    input_shape = new_input_shape;

    kernel_height = new_kernel_shape[0];
    kernel_width = new_kernel_shape[1];
    kernel_channels = new_kernel_shape[2];
    kernels_number = new_kernel_shape[3];

    set_row_stride(new_stride_shape[0]);
    set_column_stride(new_stride_shape[1]);

    set_activation_function(new_activation_function);

    set_convolution_type(new_convolution_type);

    batch_normalization = new_batch_normalization;

    if (batch_normalization)
    {
        running_means.resize(kernels_number);
        running_variances.resize(kernels_number);
    }

    set_label(new_label);

#ifdef CUDA

    if (batch_normalization)
    {
        const Shape batch_normalization_shape = { kernels_number };

        gammas_device.set_descriptor(batch_normalization_shape);
        betas_device.set_descriptor(batch_normalization_shape);
        running_means_device.resize(batch_normalization_shape);
        running_variances_device.resize(batch_normalization_shape);
    }

    cudnnCreateActivationDescriptor(&activation_descriptor);

    cudnnActivationMode_t activation = CUDNN_ACTIVATION_IDENTITY;

    if(activation_function == "Linear")
        activation = CUDNN_ACTIVATION_IDENTITY;
    else if(activation_function == "Sigmoid")
        activation = CUDNN_ACTIVATION_SIGMOID;
    else if (activation_function == "HyperbolicTangent")
        activation = CUDNN_ACTIVATION_TANH;
    else if(activation_function == "RectifiedLinear")
        activation = CUDNN_ACTIVATION_RELU;
    else if(activation_function == "ScaledExponentialLinear")
        activation = CUDNN_ACTIVATION_ELU;
    else if (activation_function == "ClippedRelu")
        activation = CUDNN_ACTIVATION_CLIPPED_RELU;
    else if (activation_function == "Swish")
        activation = CUDNN_ACTIVATION_SWISH;

    cudnnSetActivationDescriptor(activation_descriptor, activation, CUDNN_PROPAGATE_NAN, 0.0);

    // Kernel

    cudnnCreateFilterDescriptor(&kernel_descriptor);

    cudnnSetFilter4dDescriptor(kernel_descriptor,
                               CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NHWC,
                               kernels_number,
                               kernel_channels,
                               kernel_height,
                               kernel_width);

    // Convolution

    cudnnCreateConvolutionDescriptor(&convolution_descriptor);

    cudnnSetConvolution2dDescriptor(convolution_descriptor,
        get_padding_height(), get_padding_width(),
        get_row_stride(), get_column_stride(),
        1, 1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT);

#endif
}

void Convolutional::set_activation_function(const string& new_activation_function)
{
    activation_function = string_to_activation(new_activation_function);
}

void Convolutional::set_batch_normalization(bool new_batch_normalization)
{
    batch_normalization = new_batch_normalization;
}

void Convolutional::set_convolution_type(const string& new_convolution_type)
{
    if(new_convolution_type != "Valid" && new_convolution_type != "Same")
        throw runtime_error("Unknown convolution type: " + new_convolution_type + ".\n");

    convolution_type = new_convolution_type;
}

void Convolutional::set_row_stride(const Index new_stride_row)
{
    if(new_stride_row <= 0)
        throw runtime_error("EXCEPTION: new_stride_row must be a positive number");

    row_stride = new_stride_row;
}

void Convolutional::set_column_stride(const Index new_stride_column)
{
    if(new_stride_column <= 0)
        throw runtime_error("EXCEPTION: new_stride_column must be a positive number");

    column_stride = new_stride_column;
}

void Convolutional::set_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.rank != 3)
        throw runtime_error("Input new_input_shape.rank must be 3");

    input_shape = new_input_shape;
}

void Convolutional::set_parameters_glorot()
{
    const Index fan_in = get_kernel_height() * get_kernel_width() * get_kernel_channels();
    const Index fan_out = get_kernel_height() * get_kernel_width() * get_kernels_number();

    const type limit = sqrt(6.0f / static_cast<type>(fan_in + fan_out));

    VectorMap biases = vector_map(parameters[Biases]);
    biases.setZero();

    VectorMap const weights = vector_map(parameters[Weights]);
    set_random_uniform(weights, -limit, limit);

    if (batch_normalization)
    {
        VectorMap gammas = vector_map(parameters[Gammas]);
        gammas.setConstant(1.0);

        VectorMap betas = vector_map(parameters[Betas]);
        betas.setZero();
    }
}

void Convolutional::set_parameters_random()
{
    VectorMap biases = vector_map(parameters[Biases]);
    biases.setZero();

    VectorMap const weights = vector_map(parameters[Weights]);
    set_random_uniform(weights);

    if (batch_normalization)
    {
        if(parameters[Gammas].size() > 0)
            VectorMap(parameters[Gammas].data, parameters[Gammas].size()).setConstant(1.0);

        if(parameters[Betas].size() > 0)
            VectorMap(parameters[Betas].data, parameters[Betas].size()).setZero();
    }
}

pair<Index, Index> Convolutional::get_padding() const
{
    return { get_padding_height(), get_padding_width() };
}

Index Convolutional::get_input_height() const
{
    return input_shape[0];
}

Index Convolutional::get_input_width() const
{
    return input_shape[1];
}

Index Convolutional::get_input_channels() const
{
    return input_shape[2];
}

void Convolutional::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Convolutional");

    write_xml_properties(printer, {
        {"Label", label},
        {"InputDimensions", shape_to_string(input_shape)},
        {"KernelsNumber", to_string(get_kernels_number())},
        {"KernelsHeight", to_string(get_kernel_height())},
        {"KernelsWidth", to_string(get_kernel_width())},
        {"KernelsChannels", to_string(get_kernel_channels())}
    });
/*
    add_xml_element(printer, "Activation", activation_function);
    add_xml_element(printer, "StrideDimensions", shape_to_string({ get_column_stride(), get_row_stride() }));
    add_xml_element(printer, "Convolution", convolution_type);
    add_xml_element(printer, "BatchNormalization", to_string(batch_normalization));

    if (batch_normalization)
    {
        add_xml_element(printer, "Scales", tensor_to_string<type, 1>(gammas));
        add_xml_element(printer, "Offsets", tensor_to_string<type, 1>(betas));
        add_xml_element(printer, "MovingMeans", tensor_to_string<type, 1>(running_means));
        add_xml_element(printer, "MovingStandardDeviations", tensor_to_string<type, 1>(running_variances));
    }
*/
    printer.CloseElement();
}

void Convolutional::from_XML(const XMLDocument& document)
{
    const XMLElement* convolutional_layer_element = get_xml_root(document, "Convolutional");

    set_label(read_xml_string(convolutional_layer_element, "Label"));

    set_input_shape(string_to_shape(read_xml_string(convolutional_layer_element, "InputDimensions")));

    set_activation_function(read_xml_string(convolutional_layer_element, "Activation"));

    const Shape stride_shape = string_to_shape(read_xml_string(convolutional_layer_element, "StrideDimensions"));
    set_column_stride(stride_shape[0]);
    set_row_stride(stride_shape[1]);

    set_convolution_type(read_xml_string(convolutional_layer_element, "Convolution"));

    bool use_batch_normalization = false;
    const XMLElement* bn_element = convolutional_layer_element->FirstChildElement("BatchNormalization");
    if (bn_element && bn_element->GetText())
        use_batch_normalization = (string(bn_element->GetText()) == "true");

    set_batch_normalization(use_batch_normalization);
/*
    if (batch_normalization)
    {
        const Index kernel_height = read_xml_index(convolutional_layer_element, "KernelsHeight");
        const Index kernel_width = read_xml_index(convolutional_layer_element, "KernelsWidth");
        const Index kernel_channels = read_xml_index(convolutional_layer_element, "KernelsChannels");
        const Index kernels_number = read_xml_index(convolutional_layer_element, "KernelsNumber");

        gammas.resize(kernels_number);
        betas.resize(kernels_number);
        running_means.resize(kernels_number);
        running_variances.resize(kernels_number);

        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "Scales"), gammas);
        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "Offsets"), betas);
        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "MovingMeans"), running_means);
        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "MovingStandardDeviations"), running_variances);
    }
*/
}

vector<Shape> Convolutional::get_forward_shapes(const Index batch_size) const
{    
    const Index input_height = get_input_height();
    const Index input_width = get_input_width();
    const Index input_channels = get_input_channels();

    const Index output_height = get_output_height();
    const Index output_width = get_output_width();

    const Index padding_height = get_padding_height();
    const Index padding_width = get_padding_width();

    vector<Shape> shapes;

    // Padded Inputs
    shapes.push_back({ batch_size, input_height + 2*padding_height, input_width + 2*padding_width, input_channels});

    if (batch_normalization)
    {
        shapes.push_back({ kernels_number }); // Means
        shapes.push_back({ kernels_number }); // StandardDeviations
    }

    // Outputs (must be last — wiring convention)
    shapes.push_back({ batch_size, output_height, output_width, kernels_number });

    return shapes;
}

#ifdef CUDA

void ConvolutionalForwardPropagationCuda::initialize()
{
    const bool use_convolutions = convolutional_layer->use_convolutions();

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index channels = convolutional_layer->get_input_channels();

    const Index output_height = convolutional_layer->get_output_height();
    const Index output_width = convolutional_layer->get_output_width();

    string layer_label = convolutional_layer->get_label();

    // Inputs

    cudnnSetTensor4dDescriptor(input_tensor_descriptor,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_FLOAT,
                               batch_size, channels, input_height, input_width );

    // Outputs

    outputs.set_descriptor({batch_size, output_height, output_width, kernels_number});

    if (use_convolutions)
        convolutions.resize({batch_size, output_height, output_width, kernels_number});

    // Convolution Workspace

    convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    
    cudnnConvolutionFwdAlgoPerf_t fwd_perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];

    int returnedAlgoCount;

    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
        get_cudnn_handle(),
        input_tensor_descriptor,
        convolutional_layer->get_kernel_descriptor(),
        convolutional_layer->get_convolution_descriptor(),
        outputs.get_descriptor(),
        CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
        &returnedAlgoCount,
        fwd_perf_results));

    convolution_algorithm = fwd_perf_results[0].algo;
    
    cudnnGetConvolutionForwardWorkspaceSize(
        get_cudnn_handle(),
        input_tensor_descriptor, convolutional_layer->get_kernel_descriptor(),
        convolutional_layer->get_convolution_descriptor(), outputs.get_descriptor(),
        convolution_algorithm, &workspace_size);

    if (workspace_size > 0)
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

    // Batch Normalization

    if (convolutional_layer->get_batch_normalization())
    {
        Shape batch_normalization_shape = { kernels_number };

        means.resize(batch_normalization_shape);
        inverse_variance.resize(batch_normalization_shape);
    }
}

void ConvolutionalForwardPropagationCuda::print() const
{
    const Shape output_shape = layer->get_output_shape();
    const Index output_height = output_shape[0];
    const Index output_width = output_shape[1];
    const Index kernels_number = output_shape[2];

    cout << layer->get_name() + " forward propagation CUDA" << endl;
    cout << "Batch size: " << batch_size << endl;
    cout << "Outputs dimensions: " << output_height << "x" << output_width << "x" << kernels_number << endl;

    cout << "Outputs data:" << endl;
    if (outputs.data)
        cout << tensor4_from_device(outputs.data, batch_size, output_height, output_width, kernels_number) << endl;
    else
        cout << "Empty (nullptr)" << endl;
}

void ConvolutionalForwardPropagationCuda::free()
{
    cudnnDestroyTensorDescriptor(input_tensor_descriptor);
}

void ConvolutionalBackPropagationCuda::initialize()
{
    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index channels = convolutional_layer->get_input_channels();

    const Index output_height = convolutional_layer->get_output_height();
    const Index output_width = convolutional_layer->get_output_width();

    // Input Deltas

    input_gradients = {TensorView({batch_size, input_height, input_width, channels})};

    // Deltas

    cudnnSetTensor4dDescriptor(gradients_tensor_descriptor,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_FLOAT,
                               batch_size,
                               kernels_number,
                               output_height,
                               output_width);

    // Biases derivatives

    bias_gradients.set_descriptor({ kernels_number });

    // Weight derivatives

    weight_gradients.set_descriptor({ kernels_number, kernel_height, kernel_width, channels });

    // Workspace

    int returned_algo_count;
    cudnnConvolutionBwdDataAlgoPerf_t data_perf;
    cudnnConvolutionBwdFilterAlgoPerf_t filter_perf;

    CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
        get_cudnn_handle(),
        convolutional_layer->get_kernel_descriptor(),
        gradients_tensor_descriptor,
        convolutional_layer->get_convolution_descriptor(),
        input_gradients[0].get_descriptor(),
        1, &returned_algo_count, &data_perf));
    
    algo_data = data_perf.algo;

    CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
        get_cudnn_handle(),
        input_gradients[0].get_descriptor(),
        gradients_tensor_descriptor,
        convolutional_layer->get_convolution_descriptor(),
        convolutional_layer->get_kernel_descriptor(),
        1, &returned_algo_count, &filter_perf));
    
    algo_filter = filter_perf.algo;

    cudnnGetConvolutionBackwardDataWorkspaceSize(get_cudnn_handle(),
                                                 convolutional_layer->get_kernel_descriptor(),
                                                 gradients_tensor_descriptor,
                                                 convolutional_layer->get_convolution_descriptor(),
                                                 input_gradients[0].get_descriptor(),
                                                 algo_data,
                                                 &workspace_size);

    cudnnGetConvolutionBackwardFilterWorkspaceSize(get_cudnn_handle(),
                                                   input_gradients[0].get_descriptor(),
                                                   gradients_tensor_descriptor,
                                                   convolutional_layer->get_convolution_descriptor(),
                                                   convolutional_layer->get_kernel_descriptor(),
                                                   algo_filter,
                                                   &backward_filter_workspace_bytes);

    // Workspace memory

    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

    CHECK_CUDA(cudaMalloc(&backward_filter_workspace, backward_filter_workspace_bytes));

    // Batch Normalization

    if (convolutional_layer->get_batch_normalization())
    {
        beta_gradients.set_descriptor({ kernels_number });
        gamma_gradients.set_descriptor({ kernels_number });
    }
}

#endif

REGISTER(Layer, Convolutional, "Convolutional")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
