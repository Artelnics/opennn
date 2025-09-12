//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "convolutional_layer.h"

namespace opennn
{

Convolutional::Convolutional(const dimensions& new_input_dimensions,
                             const dimensions& new_kernel_dimensions,
                             const string& new_activation_function,
                             const dimensions& new_stride_dimensions,
                             const Convolutional::Convolution& new_convolution_type,
                             const bool& new_batch_normaliztion,
                             const string new_name) : Layer()
{
    name = "Convolutional";

    set(new_input_dimensions,
        new_kernel_dimensions,
        new_activation_function,
        new_stride_dimensions,
        new_convolution_type,
        new_batch_normaliztion,
        new_name);
}


bool Convolutional::get_batch_normalization() const
{
    return batch_normalization;
}

Tensor<type, 1> Convolutional::get_scales() const
{
    return scales;
}

Tensor<type, 1> Convolutional::get_offsets() const
{
    return offsets;
}


void Convolutional::preprocess_inputs(const Tensor<type, 4>& inputs,
                                      Tensor<type, 4>& preprocessed_inputs) const
{
    if (convolution_type == Convolution::Same)
        preprocessed_inputs = inputs.pad(get_paddings());
    else
        preprocessed_inputs.device(*thread_pool_device) = inputs;
}


void Convolutional::calculate_convolutions(const Tensor<type, 4>& inputs,
                                           Tensor<type, 4>& convolutions) const
{
    const Index kernels_number = get_kernels_number();

    for (Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
        const TensorMap<Tensor<type, 3>> kernel_weights = tensor_map(weights, kernel_index);
        TensorMap<Tensor<type, 3>> kernel_convolutions = tensor_map(convolutions, kernel_index);

        kernel_convolutions.device(*thread_pool_device) =
            (inputs.convolve(kernel_weights, array_3( 1, 2, 3)))
                .reshape(kernel_convolutions.dimensions()) + biases(kernel_index);
    }
}


void Convolutional::apply_batch_normalization(unique_ptr<LayerForwardPropagation>& layer_forward_propagation, const bool& is_training)
{
    ConvolutionalForwardPropagation* this_forward_propagation =
        static_cast<ConvolutionalForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 4>& outputs = this_forward_propagation->outputs;
    const Index kernels_number = get_kernels_number();

    const array<Index, 4> reshape_dims = { 1, 1, 1, kernels_number };
    const array<Index, 4> broadcast_dims = { outputs.dimension(0), outputs.dimension(1), outputs.dimension(2), 1 };

    if (is_training)
    {
        Tensor<type, 1>& means = this_forward_propagation->means;
        Tensor<type, 1>& standard_deviations = this_forward_propagation->standard_deviations;

        const array<Index, 3> reduction_axes = { 0, 1, 2 };
        means.device(*thread_pool_device) = outputs.mean(reduction_axes);

        Tensor<type, 4> centered_outputs = outputs - means.reshape(reshape_dims).broadcast(broadcast_dims);
        Tensor<type, 1> variances = centered_outputs.square().mean(reduction_axes);
        standard_deviations.device(*thread_pool_device) = variances.sqrt();

        outputs.device(*thread_pool_device) = centered_outputs / (standard_deviations.reshape(reshape_dims).broadcast(broadcast_dims) + epsilon);

        moving_means.device(*thread_pool_device) = moving_means * momentum + means * (type(1) - momentum);
        moving_standard_deviations.device(*thread_pool_device) = moving_standard_deviations * momentum + standard_deviations * (type(1) - momentum);
    }
    else
    {
        outputs.device(*thread_pool_device) = (outputs - moving_means.reshape(reshape_dims).broadcast(broadcast_dims)) /
                                              (moving_standard_deviations.reshape(reshape_dims).broadcast(broadcast_dims) + epsilon);
    }

    outputs.device(*thread_pool_device) = outputs * scales.reshape(reshape_dims).broadcast(broadcast_dims) +
                                          offsets.reshape(reshape_dims).broadcast(broadcast_dims);
}


void Convolutional::forward_propagate(const vector<TensorView>& input_views,
                                      unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                      const bool& is_training)
{
    const TensorMap<Tensor<type, 4>> inputs = tensor_map<4>(input_views[0]);

    ConvolutionalForwardPropagation* this_forward_propagation =
        static_cast<ConvolutionalForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 4>& preprocessed_inputs = this_forward_propagation->preprocessed_inputs;
    Tensor<type, 4>& outputs = this_forward_propagation->outputs;
    Tensor<type, 4>& activation_derivatives = this_forward_propagation->activation_derivatives;

    preprocess_inputs(inputs, preprocessed_inputs);

    calculate_convolutions(preprocessed_inputs, outputs);

    if(batch_normalization)
        apply_batch_normalization(layer_forward_propagation, is_training);

    is_training
        ? calculate_activations(activation_function, outputs, activation_derivatives)
        : calculate_activations(activation_function, outputs, empty_4);
}

/*
void Convolutional::back_propagate(const vector<TensorView>& input_views,
                                   const vector<TensorView>& delta_views,
                                   unique_ptr<LayerForwardPropagation>& forward_propagation,
                                   unique_ptr<LayerBackPropagation>& back_propagation) const
{
    // --- 1. SETUP ---
    const TensorMap<Tensor<type, 4>> inputs = tensor_map<4>(input_views[0]);
    TensorMap<Tensor<type, 4>> deltas = tensor_map<4>(delta_views[0]);

    const ConvolutionalForwardPropagation* this_forward_propagation =
        static_cast<const ConvolutionalForwardPropagation*>(forward_propagation.get());

    ConvolutionalBackPropagation* convolutional_back_propagation =
        static_cast<ConvolutionalBackPropagation*>(back_propagation.get());

    const Tensor<type, 4>& preprocessed_inputs = this_forward_propagation->preprocessed_inputs;
    const Tensor<type, 4>& activation_derivatives = this_forward_propagation->activation_derivatives;

    Tensor<type, 1>& bias_deltas = convolutional_back_propagation->bias_deltas;
    Tensor<type, 4>& weight_deltas = convolutional_back_propagation->weight_deltas;
    Tensor<type, 4>& input_deltas = convolutional_back_propagation->input_deltas;

    const Index batch_size = inputs.dimension(0);

    // --- 2. APPLY ACTIVATION DERIVATIVE ---
    deltas.device(*thread_pool_device) = deltas * activation_derivatives;

    // --- 3. CALCULATE BIAS GRADIENTS ---
    bias_deltas.device(*thread_pool_device) = deltas.sum(array<Index, 3>({0, 1, 2}));

    // --- 4. CALCULATE WEIGHT GRADIENTS (REFACTORED) ---
    const array<Index, 2> deltas_matrix_dims = {
        batch_size * get_output_height() * get_output_width(),
        get_kernels_number()
    };
    Tensor<type, 2> deltas_as_matrix = deltas.reshape(deltas_matrix_dims);

    Tensor<type, 2> patches_as_matrix = preprocessed_inputs.extract_image_patches(
                                                               get_kernel_height(), get_kernel_width(),
                                                               row_stride, column_stride,
                                                               1, 1,
                                                               Eigen::PADDING_VALID
                                                               ).reshape(
                                                array<Index, 2>{
                                                    batch_size * get_output_height() * get_output_width(),
                                                    get_kernel_height() * get_kernel_width() * get_input_channels()
                                                }
                                                );

    const array<IndexPair<Index>, 1> contract_axes = { IndexPair<Index>(0, 0) };
    Tensor<type, 2> weight_deltas_matrix = patches_as_matrix.contract(deltas_as_matrix, contract_axes);

    convolutional_back_propagation->weight_deltas = weight_deltas_matrix.reshape(
        array<Index, 4>{get_kernel_height(), get_kernel_width(), get_input_channels(), get_kernels_number()}
        );

    // --- 5. CALCULATE INPUT GRADIENTS (REFACTORED) ---
    const array<bool, 4> reverse_spatial_dims = {true, true, false, false};
    const array<int, 4>  shuffle_channel_dims = {0, 1, 3, 2};
    Tensor<type, 4> weights_transformed = weights.reverse(reverse_spatial_dims).shuffle(shuffle_channel_dims);

    const array<pair<Index, Index>, 4> full_paddings = {
        make_pair(0, 0),
        make_pair(get_kernel_height() - 1, get_kernel_height() - 1),
        make_pair(get_kernel_width() - 1, get_kernel_width() - 1),
        make_pair(0, 0)
    };
    Tensor<type, 4> padded_deltas = deltas.pad(full_paddings);

    const array<Index, 3> convolution_axes = {1, 2, 3};
    input_deltas.device(*thread_pool_device) = padded_deltas.convolve(weights_transformed, convolution_axes);
}
*/

void Convolutional::back_propagate(const vector<TensorView>& input_views,
                                   const vector<TensorView>& delta_views,
                                   unique_ptr<LayerForwardPropagation>& forward_propagation,
                                   unique_ptr<LayerBackPropagation>& back_propagation) const
{

    // Convolutional layer

    const Index batch_size = back_propagation->batch_size;
    const Index input_height = get_input_height();
    const Index input_width = get_input_width();
    const Index input_channels = get_input_channels();

    const Index kernels_number = get_kernels_number();
    const Index kernel_height = get_kernel_height();
    const Index kernel_width = get_kernel_width();
    const Index kernel_channels = get_kernel_channels();
    const Index kernel_size = kernel_height * kernel_width * kernel_channels;

    const TensorMap<Tensor<type, 4>> inputs = tensor_map<4>(input_views[0]);
    TensorMap<Tensor<type, 4>> deltas = tensor_map<4>(delta_views[0]);

    // Forward propagation

    ConvolutionalForwardPropagation* this_forward_propagation =
        static_cast<ConvolutionalForwardPropagation*>(forward_propagation.get());

    Tensor<type, 4>& preprocessed_inputs = this_forward_propagation->preprocessed_inputs;

    const Tensor<type, 4>& activation_derivatives = this_forward_propagation->activation_derivatives;

    // Back propagation

    ConvolutionalBackPropagation* convolutional_back_propagation =
        static_cast<ConvolutionalBackPropagation*>(back_propagation.get());

    Tensor<type, 1>& bias_deltas = convolutional_back_propagation->bias_deltas;

    type* weight_deltas_data = convolutional_back_propagation->weight_deltas.data();

    Tensor<type, 4>& rotated_weights = convolutional_back_propagation->rotated_weights;

    vector<vector<Tensor<type, 2>>> precomputed_rotated_slices(kernels_number, vector<Tensor<type, 2>>(input_channels));
    precomputed_rotated_slices.resize(kernels_number);

    Tensor<type, 4>& input_deltas = convolutional_back_propagation->input_deltas;
    input_deltas.setZero();

    const Index pad_height = (input_height + kernel_height - 1) - get_output_height();
    const Index pad_width = (input_width + kernel_width - 1) - get_output_width();
    const Index pad_top = pad_height / 2;
    const Index pad_bottom = pad_height - pad_top;
    const Index pad_left = pad_width / 2;
    const Index pad_right = pad_width - pad_left;

    const array<pair<Index, Index>, 2> paddings
        = { make_pair(pad_top, pad_bottom), make_pair(pad_left, pad_right) };

    // Inputs (for padding same)

    preprocess_inputs(inputs, preprocessed_inputs);

    // Convolution deltas

    deltas.device(*thread_pool_device) = deltas*activation_derivatives;

    // Biases derivatives

    bias_deltas.device(*thread_pool_device) = deltas.sum(array<Index, 3>({0, 1, 2}));

    // Weigth derivatives

#pragma omp parallel for
    for (Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
        const TensorMap<Tensor<type, 3>> kernel_convolution_deltas = tensor_map_(deltas, kernel_index);

        TensorMap<Tensor<type, 4>> kernel_weight_deltas(weight_deltas_data + kernel_index*kernel_size,
                                                        1, kernel_height,kernel_width, kernel_channels);

        kernel_weight_deltas = preprocessed_inputs.convolve(kernel_convolution_deltas, array<Index, 3>({0, 1, 2}));
    }

    // Input derivatives

    rotated_weights.device(*thread_pool_device) = weights.reverse(array<Index, 4>({1, 1, 0, 0}));

#pragma omp parallel for //schedule(static)
    for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const TensorMap<Tensor<type, 3>> kernel_rotated_weights = tensor_map(rotated_weights,kernel_index);

        for (Index channel_index = 0; channel_index < input_channels; ++channel_index)
            precomputed_rotated_slices[kernel_index][channel_index] = kernel_rotated_weights.chip(channel_index, 2);
    }

    const array<Index, 2> convolution_dimensions_2d = {0, 1};

    for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const TensorMap<Tensor<type, 3>> kernel_convolution_deltas = tensor_map_(deltas, kernel_index);

#pragma omp parallel for
        for (Index image_index = 0; image_index < batch_size; ++image_index)
        {
            const Tensor<type, 2> image_kernel_convolutions_derivatives_padded = kernel_convolution_deltas.chip(image_index, 0).pad(paddings);

            for (Index channel_index = 0; channel_index < input_channels; ++channel_index)
            {
                const Tensor<type, 2> convolution_result = image_kernel_convolutions_derivatives_padded
                                                               .convolve(precomputed_rotated_slices[kernel_index][channel_index], convolution_dimensions_2d);

                for (Index h = 0; h < input_height; ++h)
                    for (Index w = 0; w < input_width; ++w)
                        input_deltas(image_index, h, w, channel_index) += convolution_result(h, w);
            }
        }
    }
}


const string& Convolutional::get_activation_function() const
{
    return activation_function;
}


Index Convolutional::get_output_height() const
{
    if (convolution_type == Convolution::Same)
        return (get_input_height() + get_row_stride() - 1) / get_row_stride();
    else
        return (get_input_height() - get_kernel_height()) / get_row_stride() + 1;
}


Index Convolutional::get_output_width() const
{
    if (convolution_type == Convolution::Same)
        return (get_input_width() + get_column_stride() - 1) / get_column_stride();
    else
        return (get_input_width() - get_kernel_width()) / get_column_stride() + 1;
}


dimensions Convolutional::get_input_dimensions() const
{
    return input_dimensions;
}


dimensions Convolutional::get_output_dimensions() const
{
    return { get_output_height(), get_output_width(), get_kernels_number() };
}


Convolutional::Convolution Convolutional::get_convolution_type() const
{
    return convolution_type;
}


string Convolutional::write_convolution_type() const
{
    switch(convolution_type)
    {
    case Convolution::Valid: return "Valid";
    case Convolution::Same: return "Same";
    }

    return string();
}


Index Convolutional::get_column_stride() const
{
    return column_stride;
}


Index Convolutional::get_row_stride() const
{
    return row_stride;
}


Index  Convolutional::get_kernel_height() const
{
    return weights.dimension(0);
}


Index Convolutional::get_kernel_width() const
{
    return weights.dimension(1);
}


Index Convolutional::get_kernel_channels() const
{
    return weights.dimension(2);
}


Index Convolutional::get_kernels_number() const
{
    return weights.dimension(3);
}


Index Convolutional::get_padding_height() const
{
    if (convolution_type == Convolution::Valid)
        return 0;

    if (convolution_type == Convolution::Same)
    {
        const Index output_height = (get_input_height() + get_row_stride() - 1) / get_row_stride();

        const Index total_padding = (output_height - 1) * get_row_stride() + get_kernel_height() - get_input_height();

        return total_padding / 2;
    }

    throw runtime_error("Unknown convolution type");
}


Index Convolutional::get_padding_width() const
{
    if (convolution_type == Convolution::Valid)
        return 0;

    if (convolution_type == Convolution::Same)
    {
        const Index output_width = (get_input_width() + get_column_stride() - 1) / get_column_stride();

        const Index total_padding = (output_width - 1) * get_column_stride() + get_kernel_width() - get_input_width();

        return total_padding / 2;
    }

    throw runtime_error("Unknown convolution type");
}


void Convolutional::set(const dimensions& new_input_dimensions,
                        const dimensions& new_kernel_dimensions,
                        const string& new_activation_function,
                        const dimensions& new_stride_dimensions,
                        const Convolution& new_convolution_type,
                        const bool& new_batch_normalization,
                        const string new_label)
{
    if(new_kernel_dimensions.size() != 4)
        throw runtime_error("Kernel dimensions must be 4");

    if (new_stride_dimensions.size() != 2)
        throw runtime_error("Stride dimensions must be 2");

    if (new_kernel_dimensions[0] > new_input_dimensions[0] || new_kernel_dimensions[1] > new_input_dimensions[1])
        throw runtime_error("kernel dimensions cannot be bigger than input dimensions");

    if (new_kernel_dimensions[2] != new_input_dimensions[2])
        throw runtime_error("kernel_channels must match input_channels dimension");

    if (new_stride_dimensions[0] > new_input_dimensions[0] || new_stride_dimensions[1] > new_input_dimensions[1])
        throw runtime_error("Stride dimensions cannot be bigger than input dimensions");

    if (new_convolution_type == Convolution::Same && (new_kernel_dimensions[0] % 2 == 0 || new_kernel_dimensions[1] % 2 == 0))
        throw runtime_error("Kernel dimensions (height and width) must be odd (3x3,5x5 etc) when using 'Same' padding mode to ensure symmetric padding.");

    input_dimensions = new_input_dimensions;

    const Index kernel_height = new_kernel_dimensions[0];
    const Index kernel_width = new_kernel_dimensions[1];
    const Index kernel_channels = new_kernel_dimensions[2];
    const Index kernels_number = new_kernel_dimensions[3];

    set_row_stride(new_stride_dimensions[0]);
    set_column_stride(new_stride_dimensions[1]);

    set_activation_function(new_activation_function);

    set_convolution_type(new_convolution_type);

    biases.resize(kernels_number);
    weights.resize(kernel_height, kernel_width, kernel_channels, kernels_number);

    set_parameters_random();

    set_batch_normalization(new_batch_normalization);

    if (batch_normalization)
    {
        moving_means.resize(kernels_number);
        moving_means.setZero();
        moving_standard_deviations.resize(kernels_number);
        moving_standard_deviations.setZero();

        scales.resize(kernels_number);
        scales.setConstant(1.0);
        offsets.resize(kernels_number);
        offsets.setZero();
    }

    set_label(new_label);

#ifdef OPENNN_CUDA

    if (batch_normalization)
    {
        cudnnCreateTensorDescriptor(&bn_tensor_descriptor);

        cudnnSetTensor4dDescriptor(bn_tensor_descriptor,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   1, kernels_number, 1, 1);
    }

    cudnnCreateActivationDescriptor(&activation_descriptor);

    cudnnActivationMode_t activation = CUDNN_ACTIVATION_IDENTITY;

    if(activation_function == "Linear")
    {
        activation = CUDNN_ACTIVATION_IDENTITY;
        use_convolutions = false;
    }
    else if(activation_function == "Logistic")
    {
        activation = CUDNN_ACTIVATION_SIGMOID;
        use_convolutions = false;
    }
    else if (activation_function == "HyperbolicTangent")
    {
        activation = CUDNN_ACTIVATION_TANH;
        use_convolutions = false;
    }
    else if(activation_function == "RectifiedLinear")
    {
        activation = CUDNN_ACTIVATION_RELU;
        use_convolutions = false;
    }
    else if(activation_function == "ExponentialLinear")
    {
        activation = CUDNN_ACTIVATION_ELU;
        use_convolutions = true;
    }
    else if (activation_function == "ClippedRelu")
    {
        activation = CUDNN_ACTIVATION_CLIPPED_RELU;
        use_convolutions = true;
    }
    else if (activation_function == "Swish")
    {
        activation = CUDNN_ACTIVATION_SWISH;
        use_convolutions = true;
    }

    cudnnSetActivationDescriptor(activation_descriptor, activation, CUDNN_PROPAGATE_NAN, 0.0);

#endif

}


void Convolutional::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "Logistic"
        || new_activation_function == "HyperbolicTangent"
        || new_activation_function == "Linear"
        || new_activation_function == "RectifiedLinear"
        || new_activation_function == "ExponentialLinear")
        activation_function = new_activation_function;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function);
}


void Convolutional::set_batch_normalization(const bool& new_batch_normalization)
{
    batch_normalization = new_batch_normalization;
}


void Convolutional::set_convolution_type(const Convolutional::Convolution& new_convolution_type)
{
    convolution_type = new_convolution_type;
}


void Convolutional::set_convolution_type(const string& new_convolution_type)
{
    if(new_convolution_type == "Valid")
        convolution_type = Convolution::Valid;
    else if(new_convolution_type == "Same")
        convolution_type = Convolution::Same;
    else
        throw runtime_error("Unknown convolution type: " + new_convolution_type + ".\n");
}


void Convolutional::set_row_stride(const Index& new_stride_row)
{
    if(new_stride_row <= 0)
        throw runtime_error("EXCEPTION: new_stride_row must be a positive number");

    row_stride = new_stride_row;
}


void Convolutional::set_column_stride(const Index& new_stride_column)
{
    if(new_stride_column <= 0)
        throw runtime_error("EXCEPTION: new_stride_column must be a positive number");

    column_stride = new_stride_column;
}


void Convolutional::set_input_dimensions(const dimensions& new_input_dimensions)
{
    if (new_input_dimensions.size() != 3)
        throw runtime_error("Input new_input_dimensions.size() must be 3");

    input_dimensions = new_input_dimensions;
}


pair<Index, Index> Convolutional::get_padding() const
{
    return { get_padding_height(), get_padding_width() };
}


array<pair<Index, Index>, 4> Convolutional::get_paddings() const
{
    const Index pad_rows = get_padding().first;
    const Index pad_columns = get_padding().second;

    const array<pair<Index, Index>, 4> paddings =
        { make_pair(0, 0),
         make_pair(pad_rows, pad_rows),
         make_pair(pad_columns, pad_columns),
         make_pair(0, 0) };

    return paddings;
}


Index Convolutional::get_input_height() const
{
    return input_dimensions[0];
}


Index Convolutional::get_input_width() const
{
    return input_dimensions[1];
}


vector<ParameterView > Convolutional::get_parameter_views() const
{
    vector<ParameterView> parameter_views =
        {
            {(type*)(biases.data()), biases.size()},
            {(type*)(weights.data()), weights.size()}
        };

    if (batch_normalization)
    {
        parameter_views.push_back({ const_cast<type*>(scales.data()), scales.size() });
        parameter_views.push_back({ const_cast<type*>(offsets.data()), offsets.size() });
    }

    return parameter_views;
}


Index Convolutional::get_input_channels() const
{
    return input_dimensions[2];
}


void Convolutional::print() const
{
    cout << "Convolutional layer" << endl;
    cout << "Input dimensions: " << endl;
    print_vector(input_dimensions);
    cout << "Output dimensions: " << endl;
    print_vector(get_output_dimensions());
    cout << "Biases dimensions: " << biases.dimensions() << endl;
    cout << "Weights dimensions: " << weights.dimensions() << endl;
    cout << "biases:" << endl;
    //cout << biases << endl;
    cout << "Weights:" << endl;
    //cout << weights << endl;
}


void Convolutional::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Convolutional");

    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "InputDimensions", dimensions_to_string(input_dimensions));
    add_xml_element(printer, "KernelsNumber", to_string(get_kernels_number()));
    add_xml_element(printer, "KernelsHeight", to_string(get_kernel_height()));
    add_xml_element(printer, "KernelsWidth", to_string(get_kernel_width()));
    add_xml_element(printer, "KernelsChannels", to_string(get_kernel_channels()));
    add_xml_element(printer, "Activation", activation_function);
    add_xml_element(printer, "StrideDimensions", dimensions_to_string({ get_column_stride(), get_row_stride() }));
    add_xml_element(printer, "Convolution", write_convolution_type());
    add_xml_element(printer, "BatchNormalization", to_string(batch_normalization));
    if (batch_normalization)
    {
        add_xml_element(printer, "Scales", tensor_to_string<type, 1>(scales));
        add_xml_element(printer, "Offsets", tensor_to_string<type, 1>(offsets));
        add_xml_element(printer, "MovingMeans", tensor_to_string<type, 1>(moving_means));
        add_xml_element(printer, "MovingStandardDeviations", tensor_to_string<type, 1>(moving_standard_deviations));
    }
    add_xml_element(printer, "Biases", tensor_to_string<type, 1>(biases));
    add_xml_element(printer, "Weights", tensor_to_string<type, 4>(weights));

    printer.CloseElement();
}


void Convolutional::from_XML(const XMLDocument& document)
{
    const XMLElement* convolutional_layer_element = document.FirstChildElement("Convolutional");

    if (!convolutional_layer_element)
        throw runtime_error("Convolutional layer element is nullptr.\n");

    set_label(read_xml_string(convolutional_layer_element, "Label"));

    set_input_dimensions(string_to_dimensions(read_xml_string(convolutional_layer_element, "InputDimensions")));

    const Index kernel_height = read_xml_index(convolutional_layer_element, "KernelsHeight");
    const Index kernel_width = read_xml_index(convolutional_layer_element, "KernelsWidth");
    const Index kernel_channels = read_xml_index(convolutional_layer_element, "KernelsChannels");
    const Index kernels_number = read_xml_index(convolutional_layer_element, "KernelsNumber");

    biases.resize(kernels_number);

    weights.resize(kernel_height, kernel_width, kernel_channels, kernels_number);

    set_activation_function(read_xml_string(convolutional_layer_element, "Activation"));

    const dimensions stride_dimensions = string_to_dimensions(read_xml_string(convolutional_layer_element, "StrideDimensions"));
    set_column_stride(stride_dimensions[0]);
    set_row_stride(stride_dimensions[1]);

    set_convolution_type(read_xml_string(convolutional_layer_element, "Convolution"));

    bool use_batch_normalization = false;
    const XMLElement* bn_element = convolutional_layer_element->FirstChildElement("BatchNormalization");
    if (bn_element && bn_element->GetText())
        use_batch_normalization = (string(bn_element->GetText()) == "true");

    set_batch_normalization(use_batch_normalization);

    if (batch_normalization)
    {
        scales.resize(kernels_number);
        offsets.resize(kernels_number);
        moving_means.resize(kernels_number);
        moving_standard_deviations.resize(kernels_number);

        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "Scales"), scales);
        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "Offsets"), offsets);
        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "MovingMeans"), moving_means);
        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "MovingStandardDeviations"), moving_standard_deviations);
    }

    string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "Biases"), biases);
    string_to_tensor<type, 4>(read_xml_string(convolutional_layer_element, "Weights"), weights);
}


ConvolutionalForwardPropagation::ConvolutionalForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


TensorView ConvolutionalForwardPropagation::get_output_pair() const
{
    const Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const Index output_height = convolutional_layer->get_output_height();
    const Index output_width = convolutional_layer->get_output_width();
    const Index kernels_number = convolutional_layer->get_kernels_number();

    return {(type*)outputs.data(), {batch_size, output_height, output_width, kernels_number}};
}


void ConvolutionalForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    batch_size = new_batch_size;

    layer = new_layer;

    const Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index input_channels = convolutional_layer->get_input_channels();

    const Index kernels_number = convolutional_layer->get_kernels_number();

    const Index output_height = convolutional_layer->get_output_height();
    const Index output_width = convolutional_layer->get_output_width();

    const Index padding_height = convolutional_layer->get_padding_height();
    const Index padding_width = convolutional_layer->get_padding_width();

    preprocessed_inputs.resize(batch_size,
                               input_height + (padding_height*2),
                               input_width + (padding_width*2),
                               input_channels);

    outputs.resize(batch_size,
                   output_height,
                   output_width,
                   kernels_number);

    means.resize(kernels_number);

    standard_deviations.resize(kernels_number);

    activation_derivatives.resize(batch_size,
                                  output_height,
                                  output_width,
                                  kernels_number);
}


void ConvolutionalForwardPropagation::print() const
{
    cout << "Convolutional layer" << endl
         << "Outputs:" << endl
         << outputs << endl
         << "Activation derivatives:" << endl
         << activation_derivatives << endl;
}


ConvolutionalBackPropagation::ConvolutionalBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


void ConvolutionalBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    batch_size = new_batch_size;

    layer = new_layer;

    const Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index channels = convolutional_layer->get_input_channels();

    const Index kernel_height = convolutional_layer->get_kernel_height();
    const Index kernel_width = convolutional_layer->get_kernel_width();
    const Index kernel_channels = convolutional_layer->get_kernel_channels();
    const Index kernels_number = convolutional_layer->get_kernels_number();

    bias_deltas.resize(kernels_number);

    weight_deltas.resize(kernels_number,
                         kernel_height,
                         kernel_width,
                         kernel_channels);

    rotated_weights.resize(kernel_height,
                           kernel_width,
                           kernel_channels,
                           kernels_number);

    input_deltas.resize(batch_size,
                        input_height,
                        input_width,
                        channels);

    // Batch Normalization

    if (convolutional_layer->get_batch_normalization())
    {
        bn_scale_deltas.resize(kernels_number);
        bn_offset_deltas.resize(kernels_number);
    }
}


vector<TensorView> ConvolutionalBackPropagation::get_input_derivative_views() const
{
    const Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index channels = convolutional_layer->get_input_channels();

    convolutional_layer->get_input_dimensions();

    return {{(type*)input_deltas.data(), {batch_size, input_height, input_width, channels}}};
}


vector<ParameterView> ConvolutionalBackPropagation::get_parameter_delta_views() const
{
    const auto* convolutional_layer = static_cast<const Convolutional*>(layer);

    vector<ParameterView> delta_views =
        {
            {const_cast<type*>(bias_deltas.data()), bias_deltas.size()},
            {const_cast<type*>(weight_deltas.data()), weight_deltas.size()}
        };

    if (convolutional_layer->get_batch_normalization())
    {
        delta_views.push_back({ const_cast<type*>(bn_scale_deltas.data()), bn_scale_deltas.size() });
        delta_views.push_back({ const_cast<type*>(bn_offset_deltas.data()), bn_offset_deltas.size() });
    }

    return delta_views;
}


void ConvolutionalBackPropagation::print() const
{
    cout << "Convolutional layer back propagation" << endl
         << "Biases derivatives:\n" << endl
         << bias_deltas << endl
         << "Synaptic weights derivatives:\n" << endl
         << weight_deltas << endl;
}


#ifdef OPENNN_CUDA

void Convolutional::forward_propagate_cuda(const vector<float*>& inputs_device,
                                           unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                           const bool& is_training)
{
    // Inputs

    const Index height = get_input_height();
    const Index width = get_input_width();
    const Index channels = get_input_channels();

    const float* input_device = inputs_device[0];

    // Forward propagation

    ConvolutionalForwardPropagationCuda* convolutional_layer_forward_propagation_cuda
        = static_cast<ConvolutionalForwardPropagationCuda*>(forward_propagation_cuda.get());

    Convolutional* convolutional_layer = static_cast<Convolutional*>(convolutional_layer_forward_propagation_cuda->layer);

    const Index batch_size = convolutional_layer_forward_propagation_cuda->batch_size;

    float* convolutions = convolutional_layer_forward_propagation_cuda->convolutions;
    float* outputs = convolutional_layer_forward_propagation_cuda->outputs;

    float* outputs_buffer = use_convolutions ? convolutions : outputs;

    void* workspace = convolutional_layer_forward_propagation_cuda->workspace;
    size_t workspace_bytes = convolutional_layer_forward_propagation_cuda->workspace_bytes;

    const cudnnTensorDescriptor_t& input_tensor_descriptor = convolutional_layer_forward_propagation_cuda->input_tensor_descriptor;
    const cudnnTensorDescriptor_t& output_tensor_descriptor = convolutional_layer_forward_propagation_cuda->output_tensor_descriptor;
    const cudnnTensorDescriptor_t& biases_tensor_descriptor = convolutional_layer_forward_propagation_cuda->biases_tensor_descriptor;

    const cudnnFilterDescriptor_t& kernel_descriptor = convolutional_layer_forward_propagation_cuda->kernel_descriptor;
    const cudnnConvolutionDescriptor_t& convolution_descriptor = convolutional_layer_forward_propagation_cuda->convolution_descriptor;
    const cudnnConvolutionFwdAlgo_t& convolution_algorithm = convolutional_layer_forward_propagation_cuda->convolution_algorithm;

    if (convolutional_layer_forward_propagation_cuda->is_first_layer)
    {
        type* reordered_inputs_device = convolutional_layer_forward_propagation_cuda->reordered_inputs_device;

        reorder_inputs_cuda(input_device, reordered_inputs_device, batch_size, channels, height, width);

        input_device = reordered_inputs_device;
    }

    cudnnStatus_t status = cudnnConvolutionForward(cudnn_handle,
                                                   &alpha,
                                                   input_tensor_descriptor,
                                                   input_device,
                                                   kernel_descriptor,
                                                   weights_device,
                                                   convolution_descriptor,
                                                   convolution_algorithm,
                                                   workspace, workspace_bytes,
                                                   &beta,
                                                   output_tensor_descriptor,
                                                   outputs_buffer);

    if (status != CUDNN_STATUS_SUCCESS)
        cout << "cudnnConvolutionForward failed: " << cudnnGetErrorString(status) << endl;

    // Biases

    cudnnAddTensor(cudnn_handle,
                   &alpha,
                   biases_tensor_descriptor,
                   biases_device,
                   &alpha,
                   output_tensor_descriptor,
                   outputs_buffer);

    // Batch Normalization

    if (batch_normalization)
    {
        if (is_training)
        {
            status = cudnnBatchNormalizationForwardTraining(
                cudnn_handle,
                CUDNN_BATCHNORM_SPATIAL,
                &alpha, &beta,
                convolutional_layer_forward_propagation_cuda->output_tensor_descriptor,
                outputs_buffer,
                convolutional_layer_forward_propagation_cuda->output_tensor_descriptor,
                outputs_buffer,
                bn_tensor_descriptor,
                bn_scale_device,
                bn_offset_device,
                momentum,
                bn_running_mean_device,
                bn_running_variance_device,
                epsilon,
                convolutional_layer_forward_propagation_cuda->bn_saved_mean,
                convolutional_layer_forward_propagation_cuda->bn_saved_inv_variance
                );

            if (status != CUDNN_STATUS_SUCCESS)
                cout << "cudnnBatchNormalizationForwardTraining failed: " << cudnnGetErrorString(status) << endl;
        }
        else
        {
            status = cudnnBatchNormalizationForwardInference(
                cudnn_handle,
                CUDNN_BATCHNORM_SPATIAL,
                &alpha, &beta,
                convolutional_layer_forward_propagation_cuda->output_tensor_descriptor,
                outputs_buffer,
                convolutional_layer_forward_propagation_cuda->output_tensor_descriptor,
                outputs_buffer,
                bn_tensor_descriptor,
                bn_scale_device,
                bn_offset_device,
                bn_running_mean_device,
                bn_running_variance_device,
                epsilon
                );

            if (status != CUDNN_STATUS_SUCCESS)
                cout << "cudnnBatchNormalizationForwardInference failed: " << cudnnGetErrorString(status) << endl;
        }
    }

    // Activations

    if (convolutional_layer->get_activation_function() != "Linear")
    {
        cudnnStatus_t activationStatus = cudnnActivationForward(cudnn_handle,
                                                                activation_descriptor,
                                                                &alpha,
                                                                output_tensor_descriptor,
                                                                outputs_buffer,
                                                                &beta,
                                                                output_tensor_descriptor,
                                                                outputs);

        if (activationStatus != CUDNN_STATUS_SUCCESS)
            cout << "cudnnActivationForward failed: " << cudnnGetErrorString(activationStatus) << endl;
    }
    else
    {
        const Index outputs_number = get_outputs_number();

        cudaMemcpy(outputs, convolutions, batch_size * outputs_number * sizeof(type), cudaMemcpyDeviceToDevice);
    }
}


void Convolutional::back_propagate_cuda(const vector<float*>& inputs_device,
                                        const vector<float*>& deltas_device,
                                        unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                        unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda) const
{
    // Forward propagation

    ConvolutionalForwardPropagationCuda* convolutional_layer_forward_propagation_cuda
        = static_cast<ConvolutionalForwardPropagationCuda*>(forward_propagation_cuda.get());

    Convolutional* convolutional_layer = static_cast<Convolutional*>(convolutional_layer_forward_propagation_cuda->layer);

//    const Index batch_size = convolutional_layer_forward_propagation_cuda->batch_size;

    const type* convolutions = convolutional_layer_forward_propagation_cuda->convolutions;
    const type* outputs = convolutional_layer_forward_propagation_cuda->outputs;

    // Back propagation

    ConvolutionalBackPropagationCuda* convolutional_layer_back_propagation_cuda
        = static_cast<ConvolutionalBackPropagationCuda*>(back_propagation_cuda.get());

    void* backward_data_workspace = convolutional_layer_back_propagation_cuda->backward_data_workspace;
    void* backward_filter_workspace = convolutional_layer_back_propagation_cuda->backward_filter_workspace;

    size_t backward_data_workspace_bytes = convolutional_layer_back_propagation_cuda->backward_data_workspace_bytes;
    size_t backward_filter_workspace_bytes = convolutional_layer_back_propagation_cuda->backward_filter_workspace_bytes;

    type* weight_deltas_device = convolutional_layer_back_propagation_cuda->weight_deltas_device;
    type* bias_deltas_device = convolutional_layer_back_propagation_cuda->bias_deltas_device;
    type* input_deltas = convolutional_layer_back_propagation_cuda->input_deltas;

    const cudnnTensorDescriptor_t& input_tensor_descriptor = convolutional_layer_back_propagation_cuda->input_tensor_descriptor;
    const cudnnTensorDescriptor_t& deltas_tensor_descriptor = convolutional_layer_back_propagation_cuda->deltas_tensor_descriptor;

    const cudnnFilterDescriptor_t& kernel_descriptor = convolutional_layer_back_propagation_cuda->kernel_descriptor;

    const cudnnConvolutionDescriptor_t& convolution_descriptor = convolutional_layer_back_propagation_cuda->convolution_descriptor;

    // Error combinations derivatives

    if (convolutional_layer->get_activation_function() != "Linear")
    {
        if (use_convolutions && convolutions != nullptr)
        {
            cudnnStatus_t status = cudnnActivationBackward(cudnn_handle,
                                                           activation_descriptor,
                                                           &alpha,
                                                           deltas_tensor_descriptor,
                                                           outputs,
                                                           deltas_tensor_descriptor,
                                                           deltas_device[0],
                                                           deltas_tensor_descriptor,
                                                           convolutions,
                                                           &beta,
                                                           deltas_tensor_descriptor,
                                                           deltas_device[0]);

            if (status != CUDNN_STATUS_SUCCESS)
                cout << "cudnnActivationBackward failed: " << cudnnGetErrorString(status) << endl;
        }
        else
        {
            cudnnStatus_t status = cudnnActivationBackward(cudnn_handle,
                                                           activation_descriptor,
                                                           &alpha,
                                                           deltas_tensor_descriptor,
                                                           outputs,
                                                           deltas_tensor_descriptor,
                                                           deltas_device[0],
                                                           deltas_tensor_descriptor,
                                                           outputs,
                                                           &beta,
                                                           deltas_tensor_descriptor,
                                                           deltas_device[0]);

            if (status != CUDNN_STATUS_SUCCESS)
                cout << "cudnnActivationBackward failed: " << cudnnGetErrorString(status) << endl;
        }
    }

    // Batch Normalization

    if (batch_normalization)
    {
        cudnnStatus_t status = cudnnBatchNormalizationBackward(
            cudnn_handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            &alpha, &alpha,
            convolutional_layer_forward_propagation_cuda->output_tensor_descriptor,
            use_convolutions ? convolutional_layer_forward_propagation_cuda->convolutions : convolutional_layer_forward_propagation_cuda->outputs,
            deltas_tensor_descriptor,
            deltas_device[0],
            deltas_tensor_descriptor,
            deltas_device[0],
            bn_tensor_descriptor,
            bn_scale_device,
            convolutional_layer_back_propagation_cuda->bn_scale_deltas_device,
            convolutional_layer_back_propagation_cuda->bn_offset_deltas_device,
            epsilon,
            convolutional_layer_forward_propagation_cuda->bn_saved_mean,
            convolutional_layer_forward_propagation_cuda->bn_saved_inv_variance
            );

        if (status != CUDNN_STATUS_SUCCESS)
            cout << "cudnnBatchNormalizationBackward failed: " << cudnnGetErrorString(status) << endl;
    }


    // Convolution backwards for weights derivatives

    cudnnConvolutionBackwardFilter(cudnn_handle,
                                   &alpha,
                                   input_tensor_descriptor,
                                   inputs_device[0],
                                   deltas_tensor_descriptor,
                                   deltas_device[0],
                                   convolution_descriptor,
                                   CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                   backward_filter_workspace, backward_filter_workspace_bytes,
                                   &beta,
                                   kernel_descriptor, weight_deltas_device);

    // Biases gradients

    cudnnConvolutionBackwardBias(cudnn_handle,
                                 &alpha,
                                 deltas_tensor_descriptor,
                                 deltas_device[0],
                                 &beta,
                                 biases_tensor_descriptor,
                                 bias_deltas_device);

    // Convolution backwards for input derivatives

    cudnnConvolutionBackwardData(cudnn_handle,
                                 &alpha,
                                 kernel_descriptor,
                                 weights_device,
                                 deltas_tensor_descriptor,
                                 deltas_device[0],
                                 convolution_descriptor,
                                 CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                 backward_data_workspace, backward_data_workspace_bytes,
                                 &beta,
                                 input_tensor_descriptor, input_deltas);
}


vector<ParameterView> Convolutional::get_parameter_views_device() const
{
    vector<ParameterView> parameter_views =
        {
            {biases_device, biases.size()},
            {weights_device, weights.size()}
        };

    if (batch_normalization) {
        parameter_views.push_back({ bn_scale_device, scales.size() });
        parameter_views.push_back({ bn_offset_device, offsets.size() });
    }

    return parameter_views;
}


void Convolutional::allocate_parameters_device()
{
    cout << "Convolutional allocate_parameters_device:" << endl;
    const Index C = get_input_channels();
    const Index R = get_kernel_height();
    const Index S = get_kernel_width();
    const Index K = get_kernels_number();

    if (batch_normalization)
    {
        CUDA_MALLOC_AND_REPORT(bn_scale_device, K * sizeof(float));
        CUDA_MALLOC_AND_REPORT(bn_offset_device, K * sizeof(float));
        CUDA_MALLOC_AND_REPORT(bn_running_mean_device, K * sizeof(float));
        CUDA_MALLOC_AND_REPORT(bn_running_variance_device, K * sizeof(float));
    }

    //CHECK_CUDA(cudaMalloc(&biases_device, K * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(biases_device, K * sizeof(float));

    const size_t weights_size = static_cast<size_t>(R) * S * C * K;

    //CHECK_CUDA(cudaMalloc(&weights_device, weights_size * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(weights_device, weights_size * sizeof(float));
}


void Convolutional::free_parameters_device()
{
    cudaFree(biases_device);
    cudaFree(weights_device);

    biases_device = nullptr;
    weights_device = nullptr;

    if (batch_normalization)
    {
        cudaFree(bn_scale_device);
        cudaFree(bn_offset_device);
        cudaFree(bn_running_mean_device);
        cudaFree(bn_running_variance_device);

        bn_scale_device = nullptr;
        bn_offset_device = nullptr;
        bn_running_mean_device = nullptr;
        bn_running_variance_device = nullptr;
    }
}


void Convolutional::copy_parameters_device()
{
    if (!biases_device)
        cout << "Biases device pointer is null" << endl;

    if (!weights_device)
        cout << "Weights device pointer is null" << endl;

    if (!biases.data())
        cout << "CPU biases data is null" << endl;

    if (!weights.data())
        cout << "CPU weights data is null" << endl;

    CHECK_CUDA(cudaMemcpy(biases_device, biases.data(), biases.size() * sizeof(type), cudaMemcpyHostToDevice));

    const Index kernel_height = weights.dimension(0);
    const Index kernel_width = weights.dimension(1);
    const Index channels = weights.dimension(2);
    const Index kernels_number = weights.dimension(3);

    Tensor<type, 4> weights_for_cudnn_layout(kernel_width, kernel_height, channels, kernels_number);

    for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
        for (Index channel_index = 0; channel_index < channels; ++channel_index)
            for (Index kernel_height_index = 0; kernel_height_index < kernel_height; ++kernel_height_index)
                for (Index kernel_width_index = 0; kernel_width_index < kernel_width; ++kernel_width_index)
                    weights_for_cudnn_layout(kernel_width_index, kernel_height_index, channel_index, kernel_index)
                        = weights(kernel_height_index, kernel_width_index, channel_index, kernel_index);

    CHECK_CUDA(cudaMemcpy(weights_device, weights_for_cudnn_layout.data(), weights_for_cudnn_layout.size() * sizeof(type), cudaMemcpyHostToDevice));

    if (batch_normalization)
    {
        CHECK_CUDA(cudaMemcpy(bn_scale_device, scales.data(), scales.size() * sizeof(type), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(bn_offset_device, offsets.data(), offsets.size() * sizeof(type), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(bn_running_mean_device, moving_means.data(), moving_means.size() * sizeof(type), cudaMemcpyHostToDevice));
        Tensor<type, 1> moving_variances = moving_standard_deviations.square();
        CHECK_CUDA(cudaMemcpy(bn_running_variance_device, moving_variances.data(), moving_variances.size() * sizeof(type), cudaMemcpyHostToDevice));
    }
}


void Convolutional::copy_parameters_host()
{
    if (!biases_device) cout << "Biases is null" << endl;

    if (!weights_device) cout << "Weights is null" << endl;

    CHECK_CUDA(cudaMemcpy(biases.data(), biases_device, biases.size() * sizeof(type), cudaMemcpyDeviceToHost));

    const Index kernel_height = weights.dimension(0);
    const Index kernel_width = weights.dimension(1);
    const Index channels = weights.dimension(2);
    const Index kernels_number = weights.dimension(3);

    Tensor<type, 4> weights_cudnn_layout(kernel_width, kernel_height, channels, kernels_number);

    CHECK_CUDA(cudaMemcpy(weights_cudnn_layout.data(), weights_device, weights_cudnn_layout.size() * sizeof(type), cudaMemcpyDeviceToHost));

    for (Index k = 0; k < kernels_number; ++k)
        for (Index c = 0; c < channels; ++c)
            for (Index h = 0; h < kernel_height; ++h)
                for (Index w = 0; w < kernel_width; ++w)
                    weights(h, w, c, k) = weights_cudnn_layout(w, h, c, k);

    if (batch_normalization)
    {
        CHECK_CUDA(cudaMemcpy(scales.data(), bn_scale_device, scales.size() * sizeof(type), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(offsets.data(), bn_offset_device, offsets.size() * sizeof(type), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(moving_means.data(), bn_running_mean_device, moving_means.size() * sizeof(type), cudaMemcpyDeviceToHost));
        Tensor<type, 1> moving_variances(moving_standard_deviations.size());
        CHECK_CUDA(cudaMemcpy(moving_variances.data(), bn_running_variance_device, moving_variances.size() * sizeof(type), cudaMemcpyDeviceToHost));
        moving_standard_deviations = moving_variances.sqrt();
    }
}


ConvolutionalForwardPropagationCuda::ConvolutionalForwardPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void ConvolutionalForwardPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    cout << "ConvolutionalForwardPropagationCuda set:" << endl;

    batch_size = new_batch_size;

    layer = new_layer;

    Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const bool use_convolutions = convolutional_layer->use_convolutions;

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index channels = convolutional_layer->get_input_channels();

    const Index kernels_number = convolutional_layer->get_kernels_number();
    const Index kernel_height = convolutional_layer->get_kernel_height();
    const Index kernel_width = convolutional_layer->get_kernel_width();

    const Index pad_height = convolutional_layer->get_padding_height();
    const Index pad_width = convolutional_layer->get_padding_width();

    const Index stride_height = convolutional_layer->get_row_stride();
    const Index stride_width = convolutional_layer->get_column_stride();

    string layer_label = convolutional_layer->get_label();

    if (!layer_label.empty() && layer_label.substr(layer_label.length() - 2) == "_1")
        is_first_layer = true;

    if (is_first_layer)
    {
        //CHECK_CUDA(cudaMalloc(&reordered_inputs_device, batch_size * input_height * input_width * channels * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(reordered_inputs_device, batch_size * input_height * input_width * channels * sizeof(float));
    }

    // Inputs

    cudnnCreateTensorDescriptor(&input_tensor_descriptor);

    cudnnSetTensor4dDescriptor(input_tensor_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               batch_size, channels, input_height, input_width );

    // Biases

    cudnnCreateTensorDescriptor(&biases_tensor_descriptor);

    cudnnSetTensor4dDescriptor(biases_tensor_descriptor,
                               CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, kernels_number, 1, 1);

    // Kernels

    cudnnCreateFilterDescriptor(&kernel_descriptor);

    cudnnSetFilter4dDescriptor(kernel_descriptor,
                               CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               kernels_number, channels, kernel_height, kernel_width );

    // Convolution

    cudnnCreateConvolutionDescriptor(&convolution_descriptor);

    cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                    pad_height, pad_width,
                                    stride_height, stride_width,
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT );

    // Outputs

    cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                          input_tensor_descriptor, kernel_descriptor,
                                          &output_batch_size, &output_channels, &output_height, &output_width );

    cudnnCreateTensorDescriptor(&output_tensor_descriptor);

    cudnnSetTensor4dDescriptor(output_tensor_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               output_batch_size, output_channels, output_height, output_width );

    //CHECK_CUDA(cudaMalloc(&outputs, output_batch_size * output_height * output_width * output_channels * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(outputs, output_batch_size * output_height * output_width * output_channels * sizeof(float));

    if (use_convolutions)
    {
        //CHECK_CUDA(cudaMalloc(&convolutions, output_batch_size * output_height * output_width * output_channels * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(convolutions, output_batch_size * output_height * output_width * output_channels * sizeof(float));
    }

    // Workspace

    convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    cudnnGetConvolutionForwardWorkspaceSize(
        convolutional_layer->get_cudnn_handle(),
        input_tensor_descriptor, kernel_descriptor,
        convolution_descriptor, output_tensor_descriptor,
        convolution_algorithm, &workspace_bytes
        );

    if (workspace_bytes > 0)
        CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));

    // Batch Normalization

    if (convolutional_layer->get_batch_normalization())
    {
        //CHECK_CUDA(cudaMalloc(&bn_saved_mean, kernels_number * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(bn_saved_mean, kernels_number * sizeof(float));
        //CHECK_CUDA(cudaMalloc(&bn_saved_inv_variance, kernels_number * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(bn_saved_inv_variance, kernels_number * sizeof(float));
    }
}


void ConvolutionalForwardPropagationCuda::print() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    cout << layer->get_name() + " forward propagation" << endl;

    cout << "Outputs:" << endl;
    cout << matrix_4d_from_device(outputs, batch_size, output_dimensions[0], output_dimensions[1], output_dimensions[2]) << endl;
}


void ConvolutionalForwardPropagationCuda::free()
{
    cudaFree(outputs);
    if (convolutions != nullptr) cudaFree(convolutions);
    cudaFree(workspace);
    cudaFree(reordered_inputs_device);

    outputs = nullptr;
    convolutions = nullptr;
    workspace = nullptr;
    reordered_inputs_device = nullptr;

    cudnnDestroyTensorDescriptor(input_tensor_descriptor);
    cudnnDestroyTensorDescriptor(output_tensor_descriptor);
    cudnnDestroyTensorDescriptor(biases_tensor_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    if (bn_saved_mean) cudaFree(bn_saved_mean);
    if (bn_saved_inv_variance) cudaFree(bn_saved_inv_variance);

    bn_saved_mean = nullptr;
    bn_saved_inv_variance = nullptr;
}


ConvolutionalBackPropagationCuda::ConvolutionalBackPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagationCuda()
{
    set(new_batch_size, new_layer);
}


vector<ParameterView> ConvolutionalBackPropagationCuda::get_parameter_delta_views_device() const
{
    const auto* convolutional_layer = static_cast<const Convolutional*>(layer);

    const Index weight_deltas_size = convolutional_layer->get_kernels_number() *
                                     convolutional_layer->get_kernel_channels() *
                                     convolutional_layer->get_kernel_height() *
                                     convolutional_layer->get_kernel_width();

    const Index bias_deltas_size = convolutional_layer->get_kernels_number();

    vector<ParameterView> delta_views =
        {
            {bias_deltas_device, bias_deltas_size},
            {weight_deltas_device, weight_deltas_size}
        };

    if (convolutional_layer->get_batch_normalization())
    {
        delta_views.push_back({ bn_scale_deltas_device, convolutional_layer->get_scales().size()});
        delta_views.push_back({ bn_offset_deltas_device, convolutional_layer->get_offsets().size()});
    }

    return delta_views;
}


void ConvolutionalBackPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    cout << "ConvolutionalBackPropagationCuda set:" << endl;

    batch_size = new_batch_size;

    layer = new_layer;

    Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index channels = convolutional_layer->get_input_channels();

    const Index kernels_number = convolutional_layer->get_kernels_number();
    const Index kernel_height = convolutional_layer->get_kernel_height();
    const Index kernel_width = convolutional_layer->get_kernel_width();

    const Index output_height = convolutional_layer->get_output_height();
    const Index output_width = convolutional_layer->get_output_width();

    const Index pad_height = convolutional_layer->get_padding_height();
    const Index pad_width = convolutional_layer->get_padding_width();

    const Index stride_height = convolutional_layer->get_row_stride();
    const Index stride_width = convolutional_layer->get_column_stride();

    const size_t input_size = batch_size * channels * input_height * input_width;
    const size_t kernel_size = kernels_number * channels * kernel_height * kernel_width;

    // Inputs

    //CHECK_CUDA(cudaMalloc(&input_deltas, input_size * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(input_deltas, input_size * sizeof(float));

    cudnnCreateTensorDescriptor(&input_tensor_descriptor);

    cudnnSetTensor4dDescriptor(input_tensor_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               batch_size,
                               channels,
                               input_height,
                               input_width);

    // Deltas

    cudnnCreateTensorDescriptor(&deltas_tensor_descriptor);

    cudnnSetTensor4dDescriptor(deltas_tensor_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               batch_size,
                               kernels_number,
                               output_height,
                               output_width);

    // Biases

    //CHECK_CUDA(cudaMalloc(&bias_deltas_device, kernels_number * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(bias_deltas_device, kernels_number * sizeof(float));

    // Kernel descriptor

    cudnnCreateFilterDescriptor(&kernel_descriptor);

    cudnnSetFilter4dDescriptor(kernel_descriptor,
                               CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NCHW,
                               kernels_number,
                               channels,
                               kernel_height,
                               kernel_width);

    // Kernel derivatives

    //CHECK_CUDA(cudaMalloc(&weight_deltas_device, kernel_size * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(weight_deltas_device, kernel_size * sizeof(float));

    cudnnCreateFilterDescriptor(&weight_deltas_tensor_descriptor);

    cudnnSetFilter4dDescriptor(weight_deltas_tensor_descriptor,
                               CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NCHW,
                               kernels_number,
                               channels,
                               kernel_height,
                               kernel_width);

    // Convolution descriptor

    cudnnCreateConvolutionDescriptor(&convolution_descriptor);

    cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                    pad_height, pad_width,
                                    stride_height, stride_width,
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);

    // Workspace

    cudnnGetConvolutionBackwardDataWorkspaceSize(convolutional_layer->get_cudnn_handle(),
                                                 kernel_descriptor,
                                                 input_derivatives_tensor_descriptor,
                                                 convolution_descriptor,
                                                 input_tensor_descriptor,
                                                 CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                                 &backward_data_workspace_bytes);

    cudnnGetConvolutionBackwardFilterWorkspaceSize(convolutional_layer->get_cudnn_handle(),
                                                   input_tensor_descriptor,
                                                   input_derivatives_tensor_descriptor,
                                                   convolution_descriptor,
                                                   weight_deltas_tensor_descriptor,
                                                   CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                                   &backward_filter_workspace_bytes);

    // Workspace memory

    CHECK_CUDA(cudaMalloc(&backward_data_workspace, backward_data_workspace_bytes));

    CHECK_CUDA(cudaMalloc(&backward_filter_workspace, backward_filter_workspace_bytes));

    // Batch Normalization

    if (convolutional_layer->get_batch_normalization())
    {
        //CHECK_CUDA(cudaMalloc(&bn_scale_deltas_device, kernels_number * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(bn_scale_deltas_device, kernels_number * sizeof(float));
        //CHECK_CUDA(cudaMalloc(&bn_offset_deltas_device, kernels_number * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(bn_offset_deltas_device, kernels_number * sizeof(float));
    }
}


void ConvolutionalBackPropagationCuda::print() const
{
    const dimensions input_dimensions = layer->get_input_dimensions();
    const dimensions output_dimensions = layer->get_output_dimensions();

    cout << layer->get_name() + " back propagation" << endl;

    cout << "bias_deltas_device" << endl;
    //vector_from_device(bias_deltas,);

    cout << "weight_deltas_device" << endl;
    //matrix_from_device(weight_deltas,);

    cout << "inputs derivatives" << endl;
    matrix_4d_from_device(input_deltas, batch_size, input_dimensions[0], input_dimensions[1], input_dimensions[2]);
}


void ConvolutionalBackPropagationCuda::free()
{
    cudaFree(input_deltas);
    cudaFree(bias_deltas_device);
    cudaFree(weight_deltas_device);
    cudaFree(backward_data_workspace);
    cudaFree(backward_filter_workspace);

    input_deltas = nullptr;
    bias_deltas_device = nullptr;
    weight_deltas_device = nullptr;
    backward_data_workspace = nullptr;
    backward_filter_workspace = nullptr;

    cudnnDestroyTensorDescriptor(deltas_tensor_descriptor);
    cudnnDestroyTensorDescriptor(input_tensor_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyFilterDescriptor(weight_deltas_tensor_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    if (bn_scale_deltas_device) cudaFree(bn_scale_deltas_device);
    if (bn_offset_deltas_device) cudaFree(bn_offset_deltas_device);

    bn_scale_deltas_device = nullptr;
    bn_offset_deltas_device = nullptr;
}

REGISTER(LayerForwardPropagationCuda, ConvolutionalForwardPropagationCuda, "Convolutional")
REGISTER(LayerBackPropagationCuda, ConvolutionalBackPropagationCuda, "Convolutional")

#endif

REGISTER(Layer, Convolutional, "Convolutional")
REGISTER(LayerForwardPropagation, ConvolutionalForwardPropagation, "Convolutional")
REGISTER(LayerBackPropagation, ConvolutionalBackPropagation, "Convolutional")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
