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


bool Convolutional::get_batch_normalization() const
{
    return batch_normalization;
}


void Convolutional::reorder_weights_for_cudnn()
{
    const Index kernel_height = get_kernel_height();
    const Index kernel_width = get_kernel_width();
    const Index channels = get_kernel_channels();
    const Index kernels_number = get_kernels_number();

    TensorMap4 current_weights_map(weights.data, kernel_height, kernel_width, channels, kernels_number);
    Tensor4 weights_for_cudnn_layout(kernel_width, kernel_height, channels, kernels_number);
    for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
        for (Index channel_index = 0; channel_index < channels; ++channel_index)
            for (Index kernel_height_index = 0; kernel_height_index < kernel_height; ++kernel_height_index)
                for (Index kernel_width_index = 0; kernel_width_index < kernel_width; ++kernel_width_index)
                    weights_for_cudnn_layout(kernel_width_index, kernel_height_index, channel_index, kernel_index)
                        = current_weights_map(kernel_height_index, kernel_width_index, channel_index, kernel_index);

    memcpy(weights.data, weights_for_cudnn_layout.data(), weights.size() * sizeof(type));
}


void Convolutional::preprocess_inputs(const Tensor4& inputs,
                                      Tensor4& preprocessed_inputs) const
{
    preprocessed_inputs = (convolution_type == "Same")
        ? inputs.pad(get_paddings())
        : inputs;
}


void Convolutional::calculate_convolutions(const Tensor4& inputs, TensorMap4 convolutions) const
{  
    const Index kernels_number = get_kernels_number();

    const TensorMap4 weights_map = tensor_map<4>(weights);
    const VectorMap biases_map = vector_map(biases);

    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
        const TensorMap3 kernel_weights = tensor_map_(weights_map, kernel_index);
        TensorMap3 kernel_convolutions = tensor_map_(convolutions, kernel_index);

        kernel_convolutions.device(get_device()) =
            (inputs.convolve(kernel_weights, array_3( 1, 2, 3)))
                .reshape(kernel_convolutions.dimensions()) + biases_map(kernel_index);
    }
}


void Convolutional::forward_propagate(const vector<TensorView>& input_views,
                                      unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                      bool is_training)
{
    const TensorMap4 inputs = tensor_map<4>(input_views[0]);

    ConvolutionalForwardPropagation* convolutional_forward_propagation =
        static_cast<ConvolutionalForwardPropagation*>(layer_forward_propagation.get());

    TensorMap4 outputs = tensor_map<4>(convolutional_forward_propagation->outputs);

    Tensor4& preprocessed_inputs = convolutional_forward_propagation->preprocessed_inputs;

    TensorMap4 activation_derivatives = tensor_map<4>(convolutional_forward_propagation->activation_derivatives);

    preprocess_inputs(inputs, preprocessed_inputs);

    calculate_convolutions(preprocessed_inputs, outputs);
    
    if(batch_normalization)
        normalize_batch<4>(
            outputs,
            outputs,
            vector_map(convolutional_forward_propagation->means),
            vector_map(convolutional_forward_propagation->standard_deviations),
            running_means,
            running_standard_deviations,
            vector_map(gammas),
            vector_map(betas),
            is_training);
    

    if(is_training)
        calculate_activations<4>(activation_function, outputs, activation_derivatives);
    else
        calculate_activations<4>(activation_function, outputs, TensorMap4(empty_4.data(), empty_4.dimensions()));
}


void Convolutional::back_propagate(const vector<TensorView>& input_views,
                                   const vector<TensorView>& output_gradient_views,
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

    const TensorMap4 inputs = tensor_map<4>(input_views[0]);
    TensorMap4 output_gradients = tensor_map<4>(output_gradient_views[0]);

    // Forward propagation

    ConvolutionalForwardPropagation* convolutional_forward_propagation =
        static_cast<ConvolutionalForwardPropagation*>(forward_propagation.get());

    Tensor4& preprocessed_inputs = convolutional_forward_propagation->preprocessed_inputs;

    const TensorMap4 activation_derivatives = tensor_map<4>(convolutional_forward_propagation->activation_derivatives);

    // Back propagation

    TensorMap4 input_gradients = tensor_map<4>(back_propagation->input_gradients[0]);

    ConvolutionalBackPropagation* convolutional_back_propagation =
        static_cast<ConvolutionalBackPropagation*>(back_propagation.get());

    VectorMap bias_gradients = vector_map(convolutional_back_propagation->bias_gradients);

    type* weight_gradients_data = convolutional_back_propagation->weight_gradients.data;

    Tensor4& rotated_weights = convolutional_back_propagation->rotated_weights;

    vector<vector<Tensor2>> precomputed_rotated_slices(kernels_number, vector<Tensor2>(input_channels));
    precomputed_rotated_slices.resize(kernels_number);

    input_gradients.setZero();

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

    output_gradients.device(get_device()) = output_gradients*activation_derivatives;
    
    // Bias derivatives

    const Index features = bias_gradients.size();
    const Index total_elements_to_sum = output_gradients.size() / features;

    MatrixMap output_grads_mat(output_gradients.data(), total_elements_to_sum, features);

    bias_gradients.noalias() = output_grads_mat.colwise().sum();

    // Weights derivatives

#pragma omp parallel for
    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
        const TensorMap3 kernel_convolution_gradients = tensor_map_(output_gradients, kernel_index);

        // @todo check this. If it does not work aligned put TensorMap<Tensor<type, 4>, RowMajor | Unaligned>

//        TensorMap<Tensor<type, 4>, Unaligned> kernel_weight_gradients(weight_gradients_data + kernel_index*kernel_size,
//                                                                   1, kernel_height, kernel_width, kernel_channels);

        TensorMap4 kernel_weight_gradients(weight_gradients_data + kernel_index*kernel_size,
                                           1, kernel_height, kernel_width, kernel_channels);

        kernel_weight_gradients = preprocessed_inputs.convolve(kernel_convolution_gradients, array<Index, 3>({0, 1, 2}));
    }

    // Input derivatives

    rotated_weights.device(get_device()) = tensor_map<4>(weights).reverse(array<Index, 4>({1, 1, 0, 0}));


#pragma omp parallel for //schedule(static)
    for(Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const TensorMap3 kernel_rotated_weights = tensor_map(rotated_weights,kernel_index);

        for(Index channel_index = 0; channel_index < input_channels; ++channel_index)
            precomputed_rotated_slices[kernel_index][channel_index] = kernel_rotated_weights.chip(channel_index, 2);
    }

    const array<Index, 2> convolution_dimensions_2d = {0, 1};

    for(Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const TensorMap3 kernel_convolution_gradients = tensor_map_(output_gradients, kernel_index);

        #pragma omp parallel for
        for(Index image_index = 0; image_index < batch_size; ++image_index)
        {
            const Tensor2 image_kernel_convolutions_derivatives_padded = kernel_convolution_gradients.chip(image_index, 0).pad(paddings);

            for(Index channel_index = 0; channel_index < input_channels; ++channel_index)
            {
                const Tensor2 convolution_result = image_kernel_convolutions_derivatives_padded
                .convolve(precomputed_rotated_slices[kernel_index][channel_index], convolution_dimensions_2d);

                for(Index h = 0; h < input_height; ++h)
                    for(Index w = 0; w < input_width; ++w)
                        input_gradients(image_index, h, w, channel_index) += convolution_result(h, w);
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
    return (convolution_type == "Same")
        ? (get_input_height() + get_row_stride() - 1) / get_row_stride()
        : (get_input_height() - get_kernel_height()) / get_row_stride() + 1;}


Index Convolutional::get_output_width() const
{
    return (convolution_type == "Same")
        ? (get_input_width() + get_column_stride() - 1) / get_column_stride()
        : (get_input_width() - get_kernel_width()) / get_column_stride() + 1;
}


Shape Convolutional::get_input_shape() const
{
    return input_shape;
}


Shape Convolutional::get_output_shape() const
{
    return { get_output_height(), get_output_width(), get_kernels_number() };
}


string Convolutional::get_convolution_type() const
{
    return convolution_type;
}


Index Convolutional::get_column_stride() const
{
    return column_stride;
}


Index Convolutional::get_row_stride() const
{
    return row_stride;
}


Index Convolutional::get_kernel_height() const
{
    return weights.shape[0];
}


Index Convolutional::get_kernel_width() const
{
    return weights.shape[1];
}


Index Convolutional::get_kernel_channels() const
{
    return weights.shape[2];
}


Index Convolutional::get_kernels_number() const
{
    return weights.shape[3];
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
    if(new_kernel_shape.size() != 4)
        throw runtime_error("Kernel shape must be 4");

    if (new_stride_shape.size() != 2)
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

    const Index kernel_height = new_kernel_shape[0];
    const Index kernel_width = new_kernel_shape[1];
    const Index kernel_channels = new_kernel_shape[2];
    const Index kernels_number = new_kernel_shape[3];

    set_row_stride(new_stride_shape[0]);
    set_column_stride(new_stride_shape[1]);

    set_activation_function(new_activation_function);

    set_convolution_type(new_convolution_type);

    biases.shape = {kernels_number};
    weights.shape = {kernel_height, kernel_width, kernel_channels, kernels_number};

    batch_normalization = new_batch_normalization;

    if (batch_normalization)
    {
        running_means.resize(kernels_number);
        running_standard_deviations.resize(kernels_number);

        gammas.shape = {kernels_number};
        betas.shape = {kernels_number};
    }
    else
    {
        gammas.shape.clear();
        betas.shape.clear();
    }

    set_label(new_label);

#ifdef OPENNN_CUDA

    biases_device.set_descriptor({1, kernels_number, 1, 1});
    weights_device.set_descriptor({kernels_number, kernel_channels, kernel_height, kernel_width});

    if (batch_normalization)
    {
        const Shape batch_normalization_dims = { 1, kernels_number, 1, 1 };

        gammas_device.set_descriptor(batch_normalization_dims);
        betas_device.set_descriptor(batch_normalization_dims);
        running_means_device.resize(batch_normalization_dims);
        running_variances_device.resize(batch_normalization_dims);
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
#endif
}


void Convolutional::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "Sigmoid"
    || new_activation_function == "HyperbolicTangent"
    || new_activation_function == "Linear"
    || new_activation_function == "RectifiedLinear"
    || new_activation_function == "ScaledExponentialLinear")
        activation_function = new_activation_function;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function);
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
    if (new_input_shape.size() != 3)
        throw runtime_error("Input new_input_shape.size() must be 3");

    input_shape = new_input_shape;
}


void Convolutional::set_parameters_glorot()
{
    const Index fan_in = get_kernel_height() * get_kernel_width() * get_kernel_channels();
    const Index fan_out = get_kernel_height() * get_kernel_width() * get_kernels_number();

    const type limit = sqrt(6.0f / static_cast<type>(fan_in + fan_out));

    if (biases.size() > 0)
    {
        VectorMap biases_map(biases.data, biases.size());
        biases_map.setZero();
    }

    if (weights.size() > 0)
    {
        VectorMap weights_map(weights.data, weights.size());
        set_random_uniform(weights_map, -limit, limit);
    }

    if (batch_normalization)
    {
        if (gammas.size() > 0)
        {
            VectorMap scales_map(gammas.data, gammas.size());
            scales_map.setConstant(1.0);
        }
        if (betas.size() > 0)
        {
            VectorMap offsets_map(betas.data, betas.size());
            offsets_map.setZero();
        }
    }
}


void Convolutional::set_parameters_random()
{
    if (biases.size() > 0)
    {
        VectorMap biases_map(biases.data, biases.size());
        biases_map.setZero();
    }

    if (weights.size() > 0)
    {
        VectorMap weights_map(weights.data, weights.size());
        set_random_uniform(weights_map);
    }

    if (batch_normalization)
    {
        if (gammas.size() > 0)
            VectorMap(gammas.data, gammas.size()).setConstant(1.0);

        if (betas.size() > 0)
            VectorMap(betas.data, betas.size()).setZero();
    }
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
    return input_shape[0];
}


Index Convolutional::get_input_width() const
{
    return input_shape[1];
}


vector<TensorView*> Convolutional::get_parameter_views()
{
    vector<TensorView*> views = {&biases, &weights};

    if (batch_normalization)
        views.insert(views.end(), {&gammas, &betas});

    return views;
}


Index Convolutional::get_input_channels() const
{
    return input_shape[2];
}


void Convolutional::print() const
{

    cout << "Convolutional layer" << endl
         << "Input shape: " << input_shape << endl
         << "Output shape: " << get_output_shape() << endl
         << "Biases shape: " << biases.shape << endl
         << "Weights shape: " << weights.shape << endl
         << "biases:" << endl;
    //cout << biases << endl;
    cout << "Weights:" << endl;
    //cout << weights << endl;
}


void Convolutional::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Convolutional");

    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "InputDimensions", shape_to_string(input_shape));
    add_xml_element(printer, "KernelsNumber", to_string(get_kernels_number()));
    add_xml_element(printer, "KernelsHeight", to_string(get_kernel_height()));
    add_xml_element(printer, "KernelsWidth", to_string(get_kernel_width()));
    add_xml_element(printer, "KernelsChannels", to_string(get_kernel_channels()));
    add_xml_element(printer, "Activation", activation_function);
    add_xml_element(printer, "StrideDimensions", shape_to_string({ get_column_stride(), get_row_stride() }));
    add_xml_element(printer, "Convolution", convolution_type);
    add_xml_element(printer, "BatchNormalization", to_string(batch_normalization));
/*
    if (batch_normalization)
    {
        add_xml_element(printer, "Scales", tensor_to_string<type, 1>(gammas));
        add_xml_element(printer, "Offsets", tensor_to_string<type, 1>(betas));
        add_xml_element(printer, "MovingMeans", tensor_to_string<type, 1>(running_means));
        add_xml_element(printer, "MovingStandardDeviations", tensor_to_string<type, 1>(running_standard_deviations));
    }
*/
    printer.CloseElement();
}


void Convolutional::from_XML(const XMLDocument& document)
{
    const XMLElement* convolutional_layer_element = document.FirstChildElement("Convolutional");

    if(!convolutional_layer_element)
        throw runtime_error("Convolutional layer element is nullptr.\n");

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
        running_standard_deviations.resize(kernels_number);

        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "Scales"), gammas);
        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "Offsets"), betas);
        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "MovingMeans"), running_means);
        string_to_tensor<type, 1>(read_xml_string(convolutional_layer_element, "MovingStandardDeviations"), running_standard_deviations);
    }
*/
}


ConvolutionalForwardPropagation::ConvolutionalForwardPropagation(const Index new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


void ConvolutionalForwardPropagation::initialize()
{
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

    outputs.shape = {batch_size, output_height, output_width, kernels_number};

    activation_derivatives.shape = { batch_size, output_height, output_width, kernels_number };

    // Batch Normalization

    if (convolutional_layer->get_batch_normalization())
    {
        means.shape = { kernels_number };
        standard_deviations.shape = { kernels_number };
    } 
}


vector<TensorView*> ConvolutionalForwardPropagation::get_workspace_views()
{
    const auto* convolutional_layer = static_cast<const Convolutional*>(layer);

    vector<TensorView*> views = { &outputs, &activation_derivatives };

    if (convolutional_layer->get_batch_normalization())
        views.insert(views.end(), { &means, &standard_deviations });

    return views;
}


void ConvolutionalForwardPropagation::print() const
{
    cout << "Convolutional layer shape" << endl
         << "Outputs:" << endl
         << outputs.shape << endl
         << "Activation derivatives:" << endl
         << activation_derivatives.shape << endl;
}


ConvolutionalBackPropagation::ConvolutionalBackPropagation(const Index new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


void ConvolutionalBackPropagation::initialize()
{
    const Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index channels = convolutional_layer->get_input_channels();

    const Index kernel_height = convolutional_layer->get_kernel_height();
    const Index kernel_width = convolutional_layer->get_kernel_width();
    const Index kernel_channels = convolutional_layer->get_kernel_channels();
    const Index kernels_number = convolutional_layer->get_kernels_number();

    bias_gradients.shape = {kernels_number};

    weight_gradients.shape = { kernel_height, kernel_width, kernel_channels, kernels_number };

    rotated_weights.resize(kernel_height,
                           kernel_width,
                           kernel_channels,
                           kernels_number);

    input_gradients_memory.resize(1);
    input_gradients_memory[0].resize(Shape({ batch_size, input_height, input_width, channels }).count());
    input_gradients.resize(1);
    input_gradients[0].data = input_gradients_memory[0].data();
    input_gradients[0].shape = { batch_size, input_height, input_width, channels };

    // Batch Normalization

    if (convolutional_layer->get_batch_normalization())
    {
        gamma_gradients.shape = {kernels_number};
        beta_gradients.shape = {kernels_number};
    }
}


vector<TensorView*> ConvolutionalBackPropagation::get_gradient_views()
{
    const auto* convolutional_layer = static_cast<const Convolutional*>(layer);

    vector<TensorView*> views = {&bias_gradients, &weight_gradients};

    if (convolutional_layer->get_batch_normalization())
        views.insert(views.end(), {&gamma_gradients, &beta_gradients});

    return views;
}


void ConvolutionalBackPropagation::print() const
{
    cout << "Convolutional layer back propagation shape" << endl
         << "Biases derivatives:\n" << endl
         << bias_gradients.shape << endl
         << "Synaptic weights derivatives:\n" << endl
         << weight_gradients.shape << endl;
}


#ifdef OPENNN_CUDA

void Convolutional::forward_propagate(const vector<TensorViewCuda>& inputs,
                                           unique_ptr<LayerForwardPropagationCuda>& forward_propagation,
                                           bool is_training)
{
    // Forward propagation

    const Index batch_size = forward_propagation->batch_size;

    TensorViewCuda& outputs = forward_propagation->outputs;

    ConvolutionalForwardPropagationCuda* convolutional_forward_propagation
        = static_cast<ConvolutionalForwardPropagationCuda*>(forward_propagation.get());

    TensorCuda& convolutions = convolutional_forward_propagation->convolutions;

    const cudnnConvolutionDescriptor_t convolution_descriptor = convolutional_forward_propagation->convolution_descriptor;
    const cudnnFilterDescriptor_t kernel_descriptor = convolutional_forward_propagation->kernel_descriptor;
    const cudnnTensorDescriptor_t input_tensor_descriptor = convolutional_forward_propagation->input_tensor_descriptor;

    const float* input_data = inputs[0].data;
    float* outputs_buffer = use_convolutions() ? convolutions.data : outputs.data;
    cudnnTensorDescriptor_t current_output_descriptor = use_convolutions() ? convolutions.get_descriptor() : outputs.get_descriptor();

    if (convolutional_forward_propagation->is_first_layer)
    {
        type* reordered_inputs_data = convolutional_forward_propagation->reordered_inputs_device.data;

        reorder_inputs_cuda(inputs[0].data, reordered_inputs_data, batch_size, get_input_channels(), get_input_height(), get_input_width());

        input_data = reordered_inputs_data;
    }
    
    if (!batch_normalization && activation_function != "Softmax" && activation_function != "Linear" && !use_convolutions())
    {
        CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
            get_cudnn_handle(),
            &alpha,
            input_tensor_descriptor,
            input_data,
            kernel_descriptor,
            weights_device.data,
            convolution_descriptor,
            convolutional_forward_propagation->convolution_algorithm,
            convolutional_forward_propagation->workspace,
            convolutional_forward_propagation->workspace_bytes,
            &beta,
            current_output_descriptor,
            outputs.data,
            biases_device.get_descriptor(),
            biases_device.data,
            activation_descriptor,
            current_output_descriptor,
            outputs.data));
    }
    else
    {
        CHECK_CUDNN(cudnnConvolutionForward(get_cudnn_handle(),
            &alpha,
            input_tensor_descriptor,
            input_data,
            kernel_descriptor,
            weights_device.data,
            convolution_descriptor,
            convolutional_forward_propagation->convolution_algorithm,
            convolutional_forward_propagation->workspace,
            convolutional_forward_propagation->workspace_bytes,
            &beta,
            current_output_descriptor,
            outputs_buffer));

        // Biases

        CHECK_CUDNN(cudnnAddTensor(get_cudnn_handle(),
                                   &alpha,
                                   biases_device.get_descriptor(),
                                   biases_device.data,
                                   &alpha,
                                   current_output_descriptor,
                                   outputs_buffer));

        // Batch Normalization

        if (batch_normalization && is_training)
            CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
                get_cudnn_handle(),
                CUDNN_BATCHNORM_SPATIAL,
                &alpha, &beta,
                current_output_descriptor,
                outputs_buffer,
                current_output_descriptor,
                outputs_buffer,
                gammas_device.get_descriptor(),
                gammas_device.data,
                betas_device.data,
                momentum,
                running_means_device.data,
                running_variances_device.data,
                CUDNN_BN_MIN_EPSILON,
                convolutional_forward_propagation->batch_means.data,
                convolutional_forward_propagation->bn_saved_inv_variance.data));
        else if (batch_normalization && !is_training)
            CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
                get_cudnn_handle(),
                CUDNN_BATCHNORM_SPATIAL,
                &alpha, &beta,
                current_output_descriptor,
                outputs_buffer,
                current_output_descriptor,
                outputs_buffer,
                gammas_device.get_descriptor(),
                gammas_device.data,
                betas_device.data,
                running_means_device.data,
                running_variances_device.data,
                CUDNN_BN_MIN_EPSILON));

        // Activations

        if (activation_function != "Linear")
            CHECK_CUDNN(cudnnActivationForward(get_cudnn_handle(),
                activation_descriptor,
                &alpha,
                current_output_descriptor,
                outputs_buffer,
                &beta,
                current_output_descriptor,
                outputs.data));
        else if (use_convolutions())
            cudaMemcpy(outputs.data, outputs_buffer, batch_size * get_outputs_number() * sizeof(type), cudaMemcpyDeviceToDevice);
    }
}


void Convolutional::back_propagate(const vector<TensorViewCuda>& inputs,
                                   const vector<TensorViewCuda>& output_gradients,
                                   unique_ptr<LayerForwardPropagationCuda>& forward_propagation,
                                   unique_ptr<LayerBackPropagationCuda>& back_propagation) const
{
    // Forward propagation

    const TensorViewCuda outputs_view = forward_propagation->outputs;

    ConvolutionalForwardPropagationCuda* convolutional_forward_propagation
        = static_cast<ConvolutionalForwardPropagationCuda*>(forward_propagation.get());

    const type* const convolutions = convolutional_forward_propagation->convolutions.data;

    // Back propagation

    const cudnnTensorDescriptor_t input_tensor_descriptor = back_propagation->input_gradients[0].get_descriptor();

    TensorCuda& input_gradients = back_propagation->input_gradients[0];

    ConvolutionalBackPropagationCuda* convolutional_back_propagation
        = static_cast<ConvolutionalBackPropagationCuda*>(back_propagation.get());

    void* backward_data_workspace = convolutional_back_propagation->backward_data_workspace;
    void* backward_filter_workspace = convolutional_back_propagation->backward_filter_workspace;

    const size_t backward_data_workspace_bytes = convolutional_back_propagation->backward_data_workspace_bytes;
    const size_t backward_filter_workspace_bytes = convolutional_back_propagation->backward_filter_workspace_bytes;

    type* weight_gradients = convolutional_back_propagation->weight_gradients.data;
    type* bias_gradients = convolutional_back_propagation->bias_gradients.data;

    const cudnnTensorDescriptor_t gradients_tensor_descriptor = convolutional_back_propagation->gradients_tensor_descriptor;

    const cudnnFilterDescriptor_t kernel_descriptor = convolutional_back_propagation->kernel_descriptor;

    const cudnnConvolutionDescriptor_t convolution_descriptor = convolutional_back_propagation->convolution_descriptor;

    // Error combinations derivatives

    if (activation_function != "Linear")
        CHECK_CUDNN(cudnnActivationBackward(get_cudnn_handle(),
            activation_descriptor,
            &alpha,
            gradients_tensor_descriptor,
            outputs_view.data,
            gradients_tensor_descriptor,
            output_gradients[0].data,
            gradients_tensor_descriptor,
            (use_convolutions() && convolutions) ? convolutions : outputs_view.data,
            &beta,
            gradients_tensor_descriptor,
            output_gradients[0].data));

    // Batch Normalization

    if (batch_normalization)
        CHECK_CUDNN(cudnnBatchNormalizationBackward(
            get_cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            &alpha, &alpha,
            outputs_view.get_descriptor(),
            use_convolutions() ? convolutions : outputs_view.data,
            gradients_tensor_descriptor,
            output_gradients[0].data,
            gradients_tensor_descriptor,
            output_gradients[0].data,
            gammas_device.get_descriptor(),
            gammas_device.data,
            convolutional_back_propagation->gamma_gradients.data,
            convolutional_back_propagation->beta_gradients.data,
            CUDNN_BN_MIN_EPSILON,
            convolutional_forward_propagation->batch_means.data,
            convolutional_forward_propagation->bn_saved_inv_variance.data));

    // Convolution backwards for Weights derivatives

    cudnnConvolutionBackwardFilter(get_cudnn_handle(),
                                   &alpha,
                                   input_tensor_descriptor,
                                   inputs[0].data,
                                   gradients_tensor_descriptor,
                                   output_gradients[0].data,
                                   convolution_descriptor,
                                   convolutional_back_propagation->algo_filter,
                                   backward_filter_workspace,
                                   backward_filter_workspace_bytes,
                                   &beta,
                                   kernel_descriptor, weight_gradients);

    // Convolution backwards for Bias derivatives

    cudnnConvolutionBackwardBias(get_cudnn_handle(),
                                 &alpha,
                                 gradients_tensor_descriptor,
                                 output_gradients[0].data,
                                 &beta,
                                 biases_device.get_descriptor(),
                                 bias_gradients);

    // Convolution backwards for Input derivatives

    cudnnConvolutionBackwardData(get_cudnn_handle(),
                                 &alpha,
                                 kernel_descriptor,
                                 weights_device.data,
                                 gradients_tensor_descriptor,
                                 output_gradients[0].data,
                                 convolution_descriptor,
                                 convolutional_back_propagation->algo_data,
                                 backward_data_workspace, backward_data_workspace_bytes,
                                 &beta,
                                 input_tensor_descriptor, input_gradients.data);
}


vector<TensorViewCuda*> Convolutional::get_parameter_views_device()
{
    vector<TensorViewCuda*> views_device = { &biases_device, &weights_device };

    if (batch_normalization)
        views_device.insert(views_device.end(), {&gammas_device, &betas_device});

    return views_device;
}


void Convolutional::copy_parameters_device()
{
    if (batch_normalization)
    {
        CHECK_CUDA(cudaMemcpy(running_means_device.data, running_means.data(), running_means.size() * sizeof(type), cudaMemcpyHostToDevice));
        VectorR moving_variances = running_standard_deviations.square();
        CHECK_CUDA(cudaMemcpy(running_variances_device.data, moving_variances.data(), moving_variances.size() * sizeof(type), cudaMemcpyHostToDevice));
    }
}


ConvolutionalForwardPropagationCuda::ConvolutionalForwardPropagationCuda(const Index new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void ConvolutionalForwardPropagationCuda::initialize()
{
    const Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const bool use_convolutions = convolutional_layer->use_convolutions();

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

    if(!layer_label.empty() && layer_label.substr(layer_label.length() - 2) == "_1")
        is_first_layer = true;

    // Kernels

    cudnnCreateFilterDescriptor(&kernel_descriptor);

    cudnnSetFilter4dDescriptor(kernel_descriptor,
                               CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               kernels_number, channels, kernel_height, kernel_width );

    // Inputs

    if (is_first_layer)
        reordered_inputs_device.resize({ batch_size, channels, input_height, input_width });

    cudnnCreateTensorDescriptor(&input_tensor_descriptor);

    cudnnSetTensor4dDescriptor(input_tensor_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               batch_size, channels, input_height, input_width );

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

    outputs.set_descriptor({output_batch_size, output_channels, output_height, output_width});

    if (use_convolutions)
        convolutions.resize({output_batch_size, output_channels, output_height, output_width});

    // Convolution Workspace

    convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    
    cudnnConvolutionFwdAlgoPerf_t fwd_perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];

    int returnedAlgoCount;

    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
        get_cudnn_handle(),
        input_tensor_descriptor,
        kernel_descriptor,
        convolution_descriptor,
        outputs.get_descriptor(),
        CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
        &returnedAlgoCount,
        fwd_perf_results));

    convolution_algorithm = fwd_perf_results[0].algo;
    
    cudnnGetConvolutionForwardWorkspaceSize(
        get_cudnn_handle(),
        input_tensor_descriptor, kernel_descriptor,
        convolution_descriptor, outputs.get_descriptor(),
        convolution_algorithm, &workspace_bytes);

    if (workspace_bytes > 0)
        CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));

    // Batch Normalization

    if (convolutional_layer->get_batch_normalization())
    {
        Shape batch_normalization_dims = { 1, kernels_number, 1, 1 };

        batch_means.resize(batch_normalization_dims);
        bn_saved_inv_variance.resize(batch_normalization_dims);
    }
}


void ConvolutionalForwardPropagationCuda::print() const
{
    const Shape output_shape = layer->get_output_shape();

    cout << layer->get_name() + " forward propagation" << endl
         << "Outputs:" << endl
         << matrix_4d_from_device(outputs.data, batch_size, output_shape[0], output_shape[1], output_shape[2]) << endl;
}


void ConvolutionalForwardPropagationCuda::free()
{
    cudaFree(workspace);
    workspace = nullptr;

    cudnnDestroyTensorDescriptor(input_tensor_descriptor);

    cudnnDestroyFilterDescriptor(kernel_descriptor);

    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
}


ConvolutionalBackPropagationCuda::ConvolutionalBackPropagationCuda(const Index new_batch_size, Layer* new_layer)
    : LayerBackPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void ConvolutionalBackPropagationCuda::initialize()
{
    const Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

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

    // Input Deltas

    input_gradients.resize(1);
    input_gradients[0].resize({ batch_size, channels, input_height,  input_width });

    // Deltas

    cudnnCreateTensorDescriptor(&gradients_tensor_descriptor);

    cudnnSetTensor4dDescriptor(gradients_tensor_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               batch_size,
                               kernels_number,
                               output_height,
                               output_width);

    // Biases

    bias_gradients.set_descriptor({ kernels_number });

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

    weight_gradients.set_descriptor({ kernels_number, channels, kernel_height, kernel_width });

    cudnnCreateFilterDescriptor(&weight_gradients_filter_descriptor);

    cudnnSetFilter4dDescriptor(weight_gradients_filter_descriptor,
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

    int returned_algo_count;
    cudnnConvolutionBwdDataAlgoPerf_t data_perf;
    cudnnConvolutionBwdFilterAlgoPerf_t filter_perf;

    CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
        get_cudnn_handle(),
        kernel_descriptor,
        gradients_tensor_descriptor,
        convolution_descriptor,
        input_gradients[0].get_descriptor(),
        1, &returned_algo_count, &data_perf));
    
    algo_data = data_perf.algo;

    CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
        get_cudnn_handle(),
        input_gradients[0].get_descriptor(),
        gradients_tensor_descriptor,
        convolution_descriptor,
        weight_gradients_filter_descriptor,
        1, &returned_algo_count, &filter_perf));
    
    algo_filter = filter_perf.algo;

    cudnnGetConvolutionBackwardDataWorkspaceSize(get_cudnn_handle(),
                                                 kernel_descriptor,
                                                 gradients_tensor_descriptor,
                                                 convolution_descriptor,
                                                 input_gradients[0].get_descriptor(),
                                                 algo_data,
                                                 &backward_data_workspace_bytes);

    cudnnGetConvolutionBackwardFilterWorkspaceSize(get_cudnn_handle(),
                                                   input_gradients[0].get_descriptor(),
                                                   gradients_tensor_descriptor,
                                                   convolution_descriptor,
                                                   weight_gradients_filter_descriptor,
                                                   algo_filter,
                                                   &backward_filter_workspace_bytes);

    // Workspace memory

    CHECK_CUDA(cudaMalloc(&backward_data_workspace, backward_data_workspace_bytes));

    CHECK_CUDA(cudaMalloc(&backward_filter_workspace, backward_filter_workspace_bytes));

    // Batch Normalization

    if (convolutional_layer->get_batch_normalization())
    {
        beta_gradients.set_descriptor({ 1, kernels_number, 1, 1 });
        gamma_gradients.set_descriptor({ 1, kernels_number,1,1 });
    }
}


vector<TensorViewCuda*> ConvolutionalBackPropagationCuda::get_workspace_views()
{
    vector<TensorViewCuda*> views = { &bias_gradients, &weight_gradients };

    const auto* convolutional_layer = static_cast<const Convolutional*>(layer);

    if (convolutional_layer && convolutional_layer->get_batch_normalization())
        views.insert(views.end(), { &gamma_gradients, &beta_gradients });

    return views;
}


void ConvolutionalBackPropagationCuda::print() const
{
    const Shape input_shape = layer->get_input_shape();

    const auto* convolutional_layer = static_cast<const Convolutional*>(layer);

    cout << layer->get_name() + " back propagation" << endl;

    cout << "bias_gradients" << endl;
    vector_from_device(bias_gradients.data, bias_gradients.size());

    cout << "weight_gradients" << endl;
    matrix_4d_from_device(weight_gradients.data, convolutional_layer->get_kernels_number(),
                                                     convolutional_layer->get_kernel_channels(), 
                                                     convolutional_layer->get_kernel_height(), 
                                                     convolutional_layer->get_kernel_width());

    cout << "inputs derivatives" << endl;
    matrix_4d_from_device(input_gradients[0].data, batch_size, input_shape[0], input_shape[1], input_shape[2]);
}


void ConvolutionalBackPropagationCuda::free()
{
    cudaFree(backward_data_workspace);
    cudaFree(backward_filter_workspace);

    backward_data_workspace = nullptr;
    backward_filter_workspace = nullptr;

    cudnnDestroyTensorDescriptor(gradients_tensor_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyFilterDescriptor(weight_gradients_filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
}

REGISTER(LayerForwardPropagationCuda, ConvolutionalForwardPropagationCuda, "Convolutional")
REGISTER(LayerBackPropagationCuda, ConvolutionalBackPropagationCuda, "Convolutional")

#endif

REGISTER(Layer, Convolutional, "Convolutional")
REGISTER(LayerForwardPropagation, ConvolutionalForwardPropagation, "Convolutional")
REGISTER(LayerBackPropagation, ConvolutionalBackPropagation, "Convolutional")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
