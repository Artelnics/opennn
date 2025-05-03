//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "convolutional_layer.h"
#include "strings_utilities.h"
#include "tensors.h"

namespace opennn
{

Convolutional::Convolutional(const dimensions& new_input_dimensions,
                             const dimensions& new_kernel_dimensions,
                             const Convolutional::Activation& new_activation_function,
                             const dimensions& new_stride_dimensions,
                             const Convolutional::Convolution& new_convolution_type,
                             const string new_name) : Layer()
{
    layer_type = Layer::Type::Convolutional;

    set(new_input_dimensions, 
        new_kernel_dimensions, 
        new_activation_function, 
        new_stride_dimensions, 
        new_convolution_type, 
        new_name);
}


bool Convolutional::get_batch_normalization() const
{
    return batch_normalization;
}


void Convolutional::preprocess_inputs(const Tensor<type, 4>& inputs,
                                      Tensor<type, 4>& preprocessed_inputs) const
{
    if (convolution_type == Convolution::Same)
        if (row_stride != 1 || column_stride != 1)
            preprocessed_inputs.device(*thread_pool_device) = inputs.pad(get_paddings()).stride(get_strides());
        else
            preprocessed_inputs.device(*thread_pool_device) = inputs.pad(get_paddings());
    else
        if (row_stride != 1 || column_stride != 1)
            preprocessed_inputs.device(*thread_pool_device) = inputs.stride(get_strides());
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

        kernel_convolutions.device(*thread_pool_device) = inputs.convolve(kernel_weights, array<Index, 3>({1, 2, 3}))
                                                          + biases(kernel_index);
    }
}


void Convolutional::normalize(unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                              const bool& is_training)
{
    // @todo This method does not work

    ConvolutionalForwardPropagation* this_forward_propagation =
        static_cast<ConvolutionalForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 4>& outputs = this_forward_propagation->outputs;

    if (is_training)
    {
        Tensor<type, 1>& means = this_forward_propagation->means;
        Tensor<type, 1>& standard_deviations = this_forward_propagation->standard_deviations;

        means.device(*thread_pool_device) = outputs.mean(array<Index, 3>({0, 1, 2}));
        standard_deviations.device(*thread_pool_device) = (outputs - means).square().mean().sqrt();

        outputs.device(*thread_pool_device)
            = (outputs - means) / (standard_deviations + epsilon);
    }

    const Index kernels_number = get_kernels_number();

    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
        TensorMap<Tensor<type, 3>> kernel_output = tensor_map(outputs, kernel_index);

        if(is_training)
        {
            //TensorMap<Tensor<type, 0>> mean(&means(kernel_index));

            //TensorMap<Tensor<type, 0>> standard_deviation(&standard_deviations(kernel_index));

            //mean.device(*thread_pool_device) = kernel_output.mean();

            //standard_deviation.device(*thread_pool_device) = (kernel_output - mean(0)).square().mean().sqrt();

            //kernel_output.device(*thread_pool_device)
            //    = (kernel_output - means(kernel_index))/(standard_deviations(kernel_index) + epsilon);
        }
        else
        {
            kernel_output.device(*thread_pool_device) 
                = (kernel_output - moving_means(kernel_index)) / (moving_standard_deviations(kernel_index) + epsilon);
        }
    }

    if(is_training)
    {
        //moving_means.device(*thread_pool_device)
        //    = momentum * moving_means + (type(1) - momentum) * means;

        //moving_standard_deviations.device(*thread_pool_device)
        //    = momentum * moving_standard_deviations + (type(1) - momentum) * standard_deviations;
    }
    else
    {
        outputs.device(*thread_pool_device)
            = (outputs - moving_means)/(moving_standard_deviations + epsilon);
    }
}


void Convolutional::shift(unique_ptr<LayerForwardPropagation>& layer_forward_propagation)
{
    ConvolutionalForwardPropagation* this_forward_propagation =
        static_cast<ConvolutionalForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 4>& outputs = this_forward_propagation->outputs;

    const Index kernels_number = get_kernels_number();

    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
        TensorMap<Tensor<type, 3>> kernel_output = tensor_map(outputs, kernel_index);

        kernel_output.device(*thread_pool_device) = kernel_output * scales(kernel_index) + offsets(kernel_index);
    }
}


void Convolutional::calculate_activations(Tensor<type, 4>& activations, Tensor<type, 4>& activation_derivatives) const
{
    switch(activation_function)
    {
    case Activation::Linear: linear(activations, activation_derivatives); return;
    case Activation::Logistic: logistic(activations, activation_derivatives); return;
    case Activation::HyperbolicTangent: hyperbolic_tangent(activations, activation_derivatives); return;
    case Activation::RectifiedLinear: rectified_linear(activations, activation_derivatives); return;
    case Activation::ScaledExponentialLinear: scaled_exponential_linear(activations, activation_derivatives); return;
    case Activation::SoftPlus: soft_plus(activations, activation_derivatives); return;
    case Activation::SoftSign: soft_sign(activations, activation_derivatives); return;
    case Activation::HardSigmoid: hard_sigmoid(activations, activation_derivatives); return;
    case Activation::ExponentialLinear: exponential_linear(activations, activation_derivatives); return;
    default: return;
    }
}


void Convolutional::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                      unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                      const bool& is_training)
{
    const TensorMap<Tensor<type, 4>> inputs = tensor_map_4(input_pairs[0]);

    ConvolutionalForwardPropagation* this_forward_propagation =
        static_cast<ConvolutionalForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 4>& preprocessed_inputs = this_forward_propagation->preprocessed_inputs;
    Tensor<type, 4>& outputs = this_forward_propagation->outputs;
    Tensor<type, 4>& activation_derivatives = this_forward_propagation->activation_derivatives;

    preprocess_inputs(inputs, preprocessed_inputs);

    calculate_convolutions(preprocessed_inputs, outputs);

    // Linear dump cpu
    const size_t cpu_output_size = outputs.size(); 
    vector<float> host_cpu_combinations(cpu_output_size);
    memcpy(host_cpu_combinations.data(), outputs.data(), cpu_output_size * sizeof(float));
    cout << "\n--- CPU Convolutions + biases (Linear Dump) ---" << endl;
    cout << "[";
    for (size_t i = 0; i < host_cpu_combinations.size(); ++i)
        cout << fixed << setprecision(6) << host_cpu_combinations[i] << ", ";
    cout << "]" << endl;
 
    if(batch_normalization)
    {
        normalize(layer_forward_propagation, is_training);

        shift(layer_forward_propagation);
    }

    is_training
        ? calculate_activations(outputs, activation_derivatives)
        : calculate_activations(outputs, empty_4);

    system("pause");
}


void Convolutional::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                   const vector<pair<type*, dimensions>>& delta_pairs,
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

    const TensorMap<Tensor<type, 4>> inputs = tensor_map_4(input_pairs[0]);
    TensorMap<Tensor<type, 4>> deltas = tensor_map_4(delta_pairs[0]);

    // Forward propagation

    ConvolutionalForwardPropagation* this_forward_propagation =
            static_cast<ConvolutionalForwardPropagation*>(forward_propagation.get());

    Tensor<type, 4>& preprocessed_inputs = this_forward_propagation->preprocessed_inputs;

    const Tensor<type, 4>& activation_derivatives = this_forward_propagation->activation_derivatives;

    // Back propagation

    ConvolutionalBackPropagation* convolutional_back_propagation =
            static_cast<ConvolutionalBackPropagation*>(back_propagation.get());

    Tensor<type, 1>& bias_derivatives = convolutional_back_propagation->bias_derivatives;

    type* weight_derivatives_data = convolutional_back_propagation->weight_derivatives.data();

    Tensor<type, 4>& rotated_weights = convolutional_back_propagation->rotated_weights;

    vector<vector<Tensor<type, 2>>> precomputed_rotated_slices(kernels_number, vector<Tensor<type, 2>>(input_channels));
    precomputed_rotated_slices.resize(kernels_number);

    Tensor<type, 4>& input_derivatives = convolutional_back_propagation->input_derivatives;
    input_derivatives.setZero();
    
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

    bias_derivatives.device(*thread_pool_device) = deltas.sum(array<Index, 3>({0, 1, 2}));

    // Weigth derivatives

    #pragma omp parallel for
    for (Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
        const TensorMap<Tensor<type, 3>> kernel_convolution_deltas = tensor_map_(deltas, kernel_index);

        TensorMap<Tensor<type, 4>> kernel_weight_derivatives(weight_derivatives_data+ kernel_index*kernel_size,
                                   1, kernel_height,kernel_width, kernel_channels);

        kernel_weight_derivatives = preprocessed_inputs.convolve(kernel_convolution_deltas, array<Index, 3>({0, 1, 2}));
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
                        input_derivatives(image_index, h, w, channel_index) += convolution_result(h, w);
            }
        }
    }
}


void Convolutional::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                         Index& index,
                                         Tensor<type, 1>& gradient) const
{
    ConvolutionalBackPropagation* convolutional_back_propagation =
        static_cast<ConvolutionalBackPropagation*>(back_propagation.get());

    copy_to_vector(gradient, convolutional_back_propagation->weight_derivatives, index);
    copy_to_vector(gradient, convolutional_back_propagation->bias_derivatives, index);
}


Convolutional::Activation Convolutional::get_activation_function() const
{
    return activation_function;
}


string Convolutional::get_activation_function_string() const
{
    switch(activation_function)
    {
    case Activation::Logistic: return "Logistic";
    case Activation::HyperbolicTangent: return "HyperbolicTangent";
    case Activation::Linear: return "Linear";
    case Activation::RectifiedLinear: return "RectifiedLinear";
    case Activation::ScaledExponentialLinear: return "ScaledExponentialLinear";
    case Activation::SoftPlus: return "SoftPlus";
    case Activation::SoftSign: return "SoftSign";
    case Activation::HardSigmoid: return "HardSigmoid";
    case Activation::ExponentialLinear: return "ExponentialLinear";
    }

    return string();
}


Index Convolutional::get_output_height() const
{
    const Index input_height = get_input_height();
    const Index kernel_height = get_kernel_height();
    const Index strides = get_row_stride();

    const pair<Index, Index> padding = get_padding();

    return floor((input_height - kernel_height + padding.first)/strides) + 1;
}


Index Convolutional::get_output_width() const
{
    const Index input_width = get_input_width();
    const Index kernel_width = get_kernel_width();
    const Index strides = get_column_stride();

    const pair<Index, Index> padding = get_padding();

    return floor((input_width - kernel_width + padding.second)/strides) + 1;
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
    switch (convolution_type)
    {
    case Convolution::Valid:
        return 0;

    case Convolution::Same:
        return row_stride * (input_dimensions[1] - 1) - input_dimensions[1] + get_kernel_height();
    }

    throw runtime_error("Unknown convolution type");
}


Index Convolutional::get_padding_width() const
{
    switch(convolution_type)
    {
    case Convolution::Valid:
        return 0;

    case Convolution::Same:
        return column_stride*(input_dimensions[2] - 1) - input_dimensions[2] + get_kernel_width();
    }

    throw runtime_error("Unknown convolution type");
}


Tensor<type, 1> Convolutional::get_parameters() const
{
    Tensor<type, 1> parameters(get_parameters_number());

    Index index = 0;

    copy_to_vector(parameters, weights, index);
    copy_to_vector(parameters, biases, index);

// @todo add scales and offsets

    return parameters;
}


Index Convolutional::get_parameters_number() const
{
    return weights.size() + biases.size();
}


void Convolutional::set(const dimensions& new_input_dimensions,
                        const dimensions& new_kernel_dimensions,
                        const Convolutional::Activation& new_activation_function,
                        const dimensions& new_stride_dimensions,
                        const Convolution& new_convolution_type,
                        const string new_name)
{
    if(new_kernel_dimensions.size() != 4)
        throw runtime_error("Kernel dimensions must be 4");

    if (new_stride_dimensions.size() != 2)
        throw runtime_error("Stride dimensions must be 2");

    if (new_kernel_dimensions[0] > new_input_dimensions[0] || new_kernel_dimensions[1] > new_input_dimensions[1])
        throw runtime_error("kernel dimensions cannot be bigger than input dimensions");

    if (new_kernel_dimensions[2] != new_input_dimensions[2])
        throw runtime_error("kernel_channels must match input_channels dimension");

    if (new_stride_dimensions[0] > new_input_dimensions[0] || new_stride_dimensions[1] > new_input_dimensions[0])
        throw runtime_error("Stride dimensions cannot be bigger than input dimensions");
    
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
    biases.setConstant(1.0f);
    weights.resize(kernel_height, kernel_width, kernel_channels, kernels_number);
    weights.setRandom();

    //set_parameters_random();

    moving_means.resize(kernels_number);
    moving_standard_deviations.resize(kernels_number);

    scales.resize(kernels_number);
    offsets.resize(kernels_number);

    set_name(new_name);
}


void Convolutional::set_parameters_constant(const type& value)
{
    biases.setConstant(value);
    weights.setConstant(value);
}


void Convolutional::set_parameters_random()
{
    set_random(biases);
    set_random(weights);
}


void Convolutional::set_activation_function(const Convolutional::Activation& new_activation_function)
{
    activation_function = new_activation_function;
}


void Convolutional::set_activation_function(const string& new_activation_function_name)
{
    if(new_activation_function_name == "Logistic")
        activation_function = Activation::Logistic;
    else if(new_activation_function_name == "HyperbolicTangent")
        activation_function = Activation::HyperbolicTangent;
    else if(new_activation_function_name == "Linear")
        activation_function = Activation::Linear;
    else if(new_activation_function_name == "RectifiedLinear")
        activation_function = Activation::RectifiedLinear;
    else if(new_activation_function_name == "ScaledExponentialLinear")
        activation_function = Activation::ScaledExponentialLinear;
    else if(new_activation_function_name == "SoftPlus")
        activation_function = Activation::SoftPlus;
    else if(new_activation_function_name == "SoftSign")
        activation_function = Activation::SoftSign;
    else if(new_activation_function_name == "HardSigmoid")
        activation_function = Activation::HardSigmoid;
    else if(new_activation_function_name == "ExponentialLinear")
        activation_function = Activation::ExponentialLinear;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function_name + ".\n");
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


void Convolutional::set_parameters(const Tensor<type, 1>& new_parameters, Index& index)
{
    copy_from_vector(weights, new_parameters, index);    
    copy_from_vector(biases, new_parameters, index);
}


pair<Index, Index> Convolutional::get_padding() const
{
    switch (convolution_type)
    {
    case Convolution::Valid:
        return make_pair(0, 0);
    case Convolution::Same:
    {
        const Index input_height = get_input_height();
        const Index input_width = get_input_width();

        const Index kernel_height = get_kernel_height();
        const Index kernel_width = get_kernel_width();

        const Index row_stride = get_row_stride();
        const Index column_stride = get_column_stride();

        const Index output_height = (input_height + row_stride - 1) / row_stride;
        const Index output_width = (input_width + column_stride - 1) / column_stride;

        const Index pad_rows_total = max<Index>(0, (output_height - 1) * row_stride + kernel_height - input_height);
        const Index pad_columns_total = max<Index>(0, (output_width - 1) * column_stride + kernel_width - input_width);

        const Index pad_rows = (pad_rows_total + 1) / 2;
        const Index pad_columns = (pad_columns_total + 1) / 2;

        return make_pair(pad_rows, pad_columns);
    }
    default:
        throw runtime_error("Unknown convolution type.");
    }
}


array<pair<Index, Index>, 4> Convolutional::get_paddings() const
{
    const Index pad_rows = get_padding().first;
    const Index pad_columns = get_padding().second;

    const array<std::pair<Index, Index>, 4> paddings =
        { make_pair(0, 0),
          make_pair(pad_rows, pad_rows),
          make_pair(pad_columns, pad_columns),
          make_pair(0, 0) };

    return paddings;
}


array<Index, 4> Convolutional::get_strides() const
{   
    return array<Index, 4>({1, row_stride, column_stride, 1});
}


Index Convolutional::get_input_height() const
{
    return input_dimensions[0];
}


Index Convolutional::get_input_width() const
{
    return input_dimensions[1];
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
    cout << "Synaptic weights dimensions: " << weights.dimensions() << endl;
    cout << "biases:" << endl;
    cout << biases << endl;
    cout << "Synaptic weights:" << endl;
    cout << weights << endl;
}


void Convolutional::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Convolutional");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "InputDimensions", dimensions_to_string(input_dimensions));
    //add_xml_element(printer, "OutputDimensions", dimensions_to_string(get_output_dimensions()));
    add_xml_element(printer, "KernelsNumber", to_string(get_kernels_number()));
    add_xml_element(printer, "KernelsHeight", to_string(get_kernel_height()));
    add_xml_element(printer, "KernelsWidth", to_string(get_kernel_width()));
    add_xml_element(printer, "KernelsChannels", to_string(get_kernel_channels()));
    add_xml_element(printer, "Activation", get_activation_function_string());
    add_xml_element(printer, "StrideDimensions", dimensions_to_string({ get_column_stride(), get_row_stride() }));
    add_xml_element(printer, "Convolution", write_convolution_type());
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();
}


void Convolutional::from_XML(const XMLDocument& document)
{
    const XMLElement* convolutional_layer_element = document.FirstChildElement("Convolutional");

    if (!convolutional_layer_element) 
        throw runtime_error("Convolutional layer element is nullptr.\n");

    set_name(read_xml_string(convolutional_layer_element, "Name"));

    set_input_dimensions(string_to_dimensions(read_xml_string(convolutional_layer_element, "InputDimensions")));

    //set_output_dimensions(string_to_dimensions(read_xml_string(convolutional_layer_element, "OutputDimensions")));

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

    Index index = 0;

    set_parameters(to_type_vector(read_xml_string(convolutional_layer_element, "Parameters"), " "), index);
}


ConvolutionalForwardPropagation::ConvolutionalForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> ConvolutionalForwardPropagation::get_outputs_pair() const
{
    const Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const Index output_height = convolutional_layer->get_output_height();
    const Index output_width = convolutional_layer->get_output_width();
    const Index kernels_number = convolutional_layer->get_kernels_number();

    return {(type*)outputs.data(), {batch_size, output_height, output_width, kernels_number}};
}


void ConvolutionalForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
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
                               input_height + padding_height,
                               input_width + padding_width,
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

    bias_derivatives.resize(kernels_number);

    weight_derivatives.resize(kernels_number,
                              kernel_height,
                              kernel_width,
                              kernel_channels);

    rotated_weights.resize(kernel_height,
                           kernel_width,
                           kernel_channels,
                           kernels_number);

    input_derivatives.resize(batch_size,
                             input_height,
                             input_width,
                             channels);
}


vector<pair<type*, dimensions>> ConvolutionalBackPropagation::get_input_derivative_pairs() const
{
    const Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index channels = convolutional_layer->get_input_channels();

    convolutional_layer->get_input_dimensions();

    return {{(type*)input_derivatives.data(), {batch_size, input_height, input_width, channels}}};
}


void ConvolutionalBackPropagation::print() const
{
    cout << "Convolutional layer back propagation" << endl
         << "Biases derivatives:\n" << endl
         << bias_derivatives << endl
         << "Synaptic weights derivatives:\n" << endl
         << weight_derivatives << endl;
}


#ifdef OPENNN_CUDA

void Convolutional::forward_propagate_cuda(const vector<pair<type*, dimensions>>& inputs_pair_device,
                                           unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                           const bool& is_training)
{
    // Inputs

    const Index batch_samples_number = inputs_pair_device[0].second[0];
    const Index height = inputs_pair_device[0].second[1];
    const Index width = inputs_pair_device[0].second[2];
    const Index channels = inputs_pair_device[0].second[3];

    const type* inputs_device = inputs_pair_device[0].first;

    // Forward propagation

    ConvolutionalLayerForwardPropagationCuda* convolutional_layer_forward_propagation_cuda
        = static_cast<ConvolutionalLayerForwardPropagationCuda*>(forward_propagation_cuda.get());

    Convolutional* convolutional_layer = static_cast<Convolutional*>(convolutional_layer_forward_propagation_cuda->layer);

    type* convolutions = convolutional_layer_forward_propagation_cuda->convolutions;
    type* outputs = convolutional_layer_forward_propagation_cuda->outputs;

    void* workspace = convolutional_layer_forward_propagation_cuda->workspace;
    size_t workspace_bytes = convolutional_layer_forward_propagation_cuda->workspace_bytes;

    const cudnnActivationDescriptor_t& activation_descriptor = convolutional_layer_forward_propagation_cuda->activation_descriptor;
    const cudnnTensorDescriptor_t& inputs_tensor_descriptor = convolutional_layer_forward_propagation_cuda->inputs_tensor_descriptor;
    const cudnnTensorDescriptor_t& outputs_tensor_descriptor = convolutional_layer_forward_propagation_cuda->outputs_tensor_descriptor;
    const cudnnTensorDescriptor_t& biases_tensor_descriptor = convolutional_layer_forward_propagation_cuda->biases_tensor_descriptor;
    const cudnnFilterDescriptor_t& kernel_descriptor = convolutional_layer_forward_propagation_cuda->kernel_descriptor;
    const cudnnConvolutionDescriptor_t& convolution_descriptor = convolutional_layer_forward_propagation_cuda->convolution_descriptor;
    const cudnnConvolutionFwdAlgo_t& convolution_algorithm = convolutional_layer_forward_propagation_cuda->convolution_algorithm;

    //cudnnConvolutionForward

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudnnStatus_t status = cudnnConvolutionForward(cudnn_handle,
        &alpha,
        inputs_tensor_descriptor,
        inputs_device,
        kernel_descriptor,
        weights_device,
        convolution_descriptor,
        convolution_algorithm,
        workspace, workspace_bytes,
        &beta,
        outputs_tensor_descriptor,
        convolutions);

    if (status != CUDNN_STATUS_SUCCESS)
        cout << "cudnnConvolutionForward failed: " << cudnnGetErrorString(status) << endl;

    // Biases

    cudnnAddTensor(cudnn_handle,
        &alpha,
        biases_tensor_descriptor,
        biases_device,
        &alpha,
        outputs_tensor_descriptor,
        convolutions);

    // Linear dump
    cudaDeviceSynchronize();
    const size_t output_size = static_cast<size_t>(batch_samples_number) * get_output_height() * get_output_width() * get_kernels_number();
    vector<float> host_gpu_convolutions_bias(output_size);
    cudaError_t err_copy = cudaMemcpy(host_gpu_convolutions_bias.data(), convolutions, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err_copy != cudaSuccess) {
        cerr << "ERROR copying GPU convolutions to host: " << cudaGetErrorString(err_copy) << endl;
    }
    else {
        cout << "\n--- GPU Convolutions + biases (Linear Dump) ---" << endl;
        cout << "[";
        for (size_t i = 0; i < host_gpu_convolutions_bias.size(); ++i)
            cout << fixed << setprecision(6) << host_gpu_convolutions_bias[i] << ", ";
        cout << "]" << endl;
    }

    // Activations

    if (convolutional_layer->get_activation_function() != Activation::Linear)
    {
        cudnnStatus_t activationStatus = cudnnActivationForward(cudnn_handle,
            activation_descriptor,
            &alpha,
            outputs_tensor_descriptor,
            convolutions,
            &beta,
            outputs_tensor_descriptor,
            outputs);

        if (activationStatus != CUDNN_STATUS_SUCCESS)
            cout << "cudnnActivationForward failed: " << cudnnGetErrorString(activationStatus) << endl;
    }
    else
    {  
        const Index outputs_number = get_outputs_number();

        cudaMemcpy(outputs, convolutions, batch_samples_number * outputs_number * sizeof(type), cudaMemcpyDeviceToDevice);
    }

    system("pause");
}


void Convolutional::back_propagate_cuda(const vector<pair<type*, dimensions>>& inputs_pair_device,
                                        const vector<pair<type*, dimensions>>& deltas_pair_device,
                                        unique_ptr<LayerForwardPropagationCuda>& forward_propagation_cuda,
                                        unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda) const
{
    // Inputs
    const Index batch_samples_number = inputs_pair_device[0].second[0];

    const type* inputs_device = inputs_pair_device[0].first;
    const type* deltas_device = deltas_pair_device[0].first;

    // Forward propagation

    ConvolutionalLayerForwardPropagationCuda* convolutional_layer_forward_propagation_cuda
        = static_cast<ConvolutionalLayerForwardPropagationCuda*>(forward_propagation_cuda.get());

    Convolutional* convolutional_layer = static_cast<Convolutional*>(convolutional_layer_forward_propagation_cuda->layer);

    const type* convolutions = convolutional_layer_forward_propagation_cuda->convolutions;
    const type* outputs = convolutional_layer_forward_propagation_cuda->outputs;

    const cudnnActivationDescriptor_t& activation_descriptor = convolutional_layer_forward_propagation_cuda->activation_descriptor;
    const cudnnTensorDescriptor_t& biases_tensor_descriptor = convolutional_layer_forward_propagation_cuda->biases_tensor_descriptor;

    // Back propagation

    ConvolutionalLayerBackPropagationCuda* convolutional_layer_back_propagation_cuda
        = static_cast<ConvolutionalLayerBackPropagationCuda*>(back_propagation_cuda.get());

    void* backward_data_workspace = convolutional_layer_back_propagation_cuda->backward_data_workspace;
    void* backward_filter_workspace = convolutional_layer_back_propagation_cuda->backward_filter_workspace;
    size_t backward_data_workspace_bytes = convolutional_layer_back_propagation_cuda->backward_data_workspace_bytes;
    size_t backward_filter_workspace_bytes = convolutional_layer_back_propagation_cuda->backward_filter_workspace_bytes;

    type* error_combinations_derivatives_device = convolutional_layer_back_propagation_cuda->error_combinations_derivatives_device;
    type* weights_derivatives_device = convolutional_layer_back_propagation_cuda->weights_derivatives_device;
    type* biases_derivatives_device = convolutional_layer_back_propagation_cuda->biases_derivatives_device;
    type* input_derivatives = convolutional_layer_back_propagation_cuda->input_derivatives;

    const cudnnTensorDescriptor_t& deltas_device_tensor_descriptor = convolutional_layer_back_propagation_cuda->deltas_device_tensor_descriptor;
    const cudnnTensorDescriptor_t& error_combinations_derivatives_tensor_descriptor = convolutional_layer_back_propagation_cuda->error_combinations_derivatives_tensor_descriptor;
    const cudnnTensorDescriptor_t& inputs_tensor_descriptor = convolutional_layer_back_propagation_cuda->inputs_tensor_descriptor;

    const cudnnFilterDescriptor_t& kernel_descriptor = convolutional_layer_back_propagation_cuda->kernel_descriptor;

    const cudnnConvolutionDescriptor_t& convolution_descriptor = convolutional_layer_back_propagation_cuda->convolution_descriptor;

    // Error combinations derivatives

    const float alpha = 1.0f, beta = 0.0f;

    if (convolutional_layer->get_activation_function() != Activation::Linear)
    {
        cudnnActivationBackward(cudnn_handle,
            activation_descriptor,
            &alpha,
            error_combinations_derivatives_tensor_descriptor,
            outputs,
            deltas_device_tensor_descriptor,
            deltas_device,
            error_combinations_derivatives_tensor_descriptor,
            convolutions,
            &beta,
            error_combinations_derivatives_tensor_descriptor,
            error_combinations_derivatives_device);

    }
    else
    {
        const Index outputs_number = get_outputs_number();

        cudaMemcpy(error_combinations_derivatives_device, deltas_device, batch_samples_number * outputs_number * sizeof(type), cudaMemcpyDeviceToDevice);
    }

    // Convolution backwards for weights derivatives

    cudnnConvolutionBackwardFilter(cudnn_handle,
        &alpha,
        inputs_tensor_descriptor,
        inputs_device,
        error_combinations_derivatives_tensor_descriptor,
        error_combinations_derivatives_device,
        convolution_descriptor,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        backward_filter_workspace, backward_filter_workspace_bytes,
        &beta,
        kernel_descriptor, weights_derivatives_device);

    // Biases gradients

    cudnnConvolutionBackwardBias(cudnn_handle,
        &alpha,
        error_combinations_derivatives_tensor_descriptor,
        error_combinations_derivatives_device,
        &beta,
        biases_tensor_descriptor,
        biases_derivatives_device);

    //cout << "kernel_synaptic_weights_derivatives:\n" << matrix_3d_from_device(kernel_synaptic_weights_derivatives, 2, 2, 1) << endl;
    //cout << "biases_derivatives:\n" << matrix_3d_from_device(biases_derivatives, 1, 1, 1) << endl;
    //system("pause");

    // Convolution backwards for input derivatives

    cudnnConvolutionBackwardData(cudnn_handle,
        &alpha,
        kernel_descriptor,
        weights_device,
        error_combinations_derivatives_tensor_descriptor,
        error_combinations_derivatives_device,
        convolution_descriptor,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        backward_data_workspace, backward_data_workspace_bytes,
        &beta,
        inputs_tensor_descriptor, input_derivatives);
}


void Convolutional::insert_gradient_cuda(unique_ptr<LayerBackPropagationCuda>& back_propagation_cuda,
                                         Index& index,
                                         float* gradient) const
{
    ConvolutionalLayerBackPropagationCuda* convolutional_layer_back_propagation =
        static_cast<ConvolutionalLayerBackPropagationCuda*>(back_propagation_cuda.get());

    copy_to_vector_cuda(gradient, convolutional_layer_back_propagation->weights_derivatives_device, weights.size(), index);
    copy_to_vector_cuda(gradient, convolutional_layer_back_propagation->biases_derivatives_device, biases.size(), index);
}


void Convolutional::set_parameters_cuda(const float* new_parameters, Index& index)
{
    copy_from_vector_cuda(weights_device, new_parameters, weights.size(), index);
    copy_from_vector_cuda(biases_device, new_parameters, biases.size(), index);
}


void Convolutional::get_parameters_cuda(const Tensor<type, 1>& new_parameters, const Index& index)
{
    // @todo
}


void Convolutional::allocate_parameters_device()
{
    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    if (cudaMalloc(&biases_device, outputs_number * sizeof(float)) != cudaSuccess)
        cout << "Biases allocation error" << endl;

    if (cudaMalloc(&weights_device, inputs_number * outputs_number * sizeof(float)) != cudaSuccess)
        cout << "Synaptic weights allocation error" << endl;
}


void Convolutional::free_parameters_device()
{
    cudaFree(biases_device);
    cudaFree(weights_device);

    biases_device = nullptr;
    weights_device = nullptr;
}


void Convolutional::copy_parameters_device()
{
    if (biases_device == nullptr)
        cout << "ERROR: biases_device is null before copy." << endl;
    else if (biases.size() > 0) 
        cudaMemcpy(biases_device, biases.data(), biases.size() * sizeof(type), cudaMemcpyHostToDevice);

    if (weights_device == nullptr)
        cout << "ERROR: weights_device is null before copy." << endl;

    const Index kernels_number = get_kernels_number();    // Dimension 3 in Eigen (RSCK) -> Dimension 0 in KCRS
    const Index kernel_channels = get_kernel_channels();  // Dimension 2 in Eigen (RSCK) -> Dimension 1 in KCRS
    const Index kernel_height = get_kernel_height();      // Dimension 0 in Eigen (RSCK) -> Dimension 2 in KCRS
    const Index kernel_width = get_kernel_width();        // Dimension 1 in Eigen (RSCK) -> Dimension 3 in KCRS
    const size_t total_weights = static_cast<size_t>(kernels_number) * kernel_channels * kernel_height * kernel_width;

    vector<type> weights_kcr_linear(total_weights);

    size_t destination_index = 0;

    for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
        for (Index channel_index = 0; channel_index < kernel_channels; ++channel_index)
            for (Index height_index = 0; height_index < kernel_height; ++height_index)
                for (Index width_index = 0; width_index < kernel_width; ++width_index)
                {
                    weights_kcr_linear[destination_index] = weights(height_index, width_index, channel_index, kernel_index);
                    destination_index++;
                }

    if (destination_index != total_weights)
        cout << "ERROR: Mismatch in element count during weight reordering. Expected " << total_weights << ", processed " << destination_index << endl;

    cudaMemcpy(weights_device, weights_kcr_linear.data(), total_weights * sizeof(type), cudaMemcpyHostToDevice);
}



void Convolutional::copy_parameters_host() // @todo change order
{
    if (biases_device == nullptr)
        cout << "Biases is null" << endl;

    if (weights_device == nullptr)
        cout << "Synaptic weights is null" << endl;

    if (cudaMemcpy(biases.data(), biases_device, biases.size() * sizeof(type), cudaMemcpyDeviceToHost) != cudaSuccess)
        cout << "Biases host copy error" << endl;

    if (cudaMemcpy(weights.data(), weights_device, weights.size() * sizeof(type), cudaMemcpyDeviceToHost) != cudaSuccess)
        cout << "Weights host copy error" << endl;
}


float* Convolutional::get_weights_device() const
{
    return weights_device;
}

float* Convolutional::get_biases_device() const
{
    return biases_device;
}


// CUDA structs

ConvolutionalLayerForwardPropagationCuda::ConvolutionalLayerForwardPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void ConvolutionalLayerForwardPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
    if (new_batch_size == 0) return;

    batch_size = new_batch_size;
    layer = new_layer;

    Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index channels = convolutional_layer->get_input_channels();

    const Index kernel_height = convolutional_layer->get_kernel_height();
    const Index kernel_width = convolutional_layer->get_kernel_width();
    const Index kernels_number = convolutional_layer->get_kernels_number();
    
    const Index pad_height = convolutional_layer->get_padding_height();
    const Index pad_width = convolutional_layer->get_padding_width();

    const Index stride_row = convolutional_layer->get_row_stride();
    const Index stride_column = convolutional_layer->get_column_stride();

    // Inputs

    cudnnCreateTensorDescriptor(&inputs_tensor_descriptor);

    cudnnSetTensor4dDescriptorEx(inputs_tensor_descriptor, CUDNN_DATA_FLOAT,
                                 batch_size, channels, input_height, input_width, 
                                 input_height * input_width * channels,
                                 1,
                                 input_width * channels,
                                 channels);

    // Biases

    cudnnCreateTensorDescriptor(&biases_tensor_descriptor);

    cudnnSetTensor4dDescriptor(biases_tensor_descriptor,
                               CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
                               1, 1, 1, kernels_number);

    // Kernel descriptor

    cudnnCreateFilterDescriptor(&kernel_descriptor);

    cudnnSetFilter4dDescriptor(kernel_descriptor,
                               CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC,
                               kernels_number, channels, kernel_height, kernel_width);

    // Convolution descriptor

    cudnnCreateConvolutionDescriptor(&convolution_descriptor);
   
    cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                    pad_height, pad_width, 
                                    stride_row, stride_column,
                                    1, 1,
                                    CUDNN_CONVOLUTION,
                                    CUDNN_DATA_FLOAT);

    // Output

    cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                          inputs_tensor_descriptor, kernel_descriptor,
                                          &output_batch_size, &output_channels, &output_height, &output_width);

    cudnnCreateTensorDescriptor(&outputs_tensor_descriptor);

    cudnnSetTensor4dDescriptorEx(outputs_tensor_descriptor, CUDNN_DATA_FLOAT,
                                 output_batch_size, output_channels, output_height, output_width,
                                 output_height * output_width * output_channels,
                                 1,
                                 output_width * output_channels,
                                 output_channels);

    // Mallocs

    const size_t output_size = static_cast<size_t>(output_batch_size) * output_height * output_width * output_channels;

    if (cudaMalloc(&outputs, output_size * sizeof(float)) != cudaSuccess)
        cout << "outputs allocation error" << endl;

    if (cudaMalloc(&convolutions, output_size * sizeof(float)) != cudaSuccess)
        cout << "outputs allocation error" << endl;

    // Activations

    cudnnCreateActivationDescriptor(&activation_descriptor);

    cudnnActivationMode_t activation = CUDNN_ACTIVATION_IDENTITY;

    switch (convolutional_layer->get_activation_function()) {
    case Convolutional::Activation::Linear: activation = CUDNN_ACTIVATION_IDENTITY; break;
    case Convolutional::Activation::Logistic: activation = CUDNN_ACTIVATION_SIGMOID; break;
    case Convolutional::Activation::HyperbolicTangent: activation = CUDNN_ACTIVATION_TANH; break;
    case Convolutional::Activation::RectifiedLinear: activation = CUDNN_ACTIVATION_RELU; break;
    case Convolutional::Activation::ExponentialLinear: activation = CUDNN_ACTIVATION_ELU; break;
    default: break;
    }
    cudnnSetActivationDescriptor(activation_descriptor, activation, CUDNN_PROPAGATE_NAN, 0.0);

    // Convolution algorithm

    perfResults.resize(CUDNN_CONVOLUTION_FWD_ALGO_COUNT);

    cudnnFindConvolutionForwardAlgorithm(convolutional_layer->get_cudnn_handle(),
                                         inputs_tensor_descriptor, kernel_descriptor, convolution_descriptor, outputs_tensor_descriptor,
                                         CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returnedAlgoCount, perfResults.data());

    if (returnedAlgoCount > 0 && perfResults[0].status == CUDNN_STATUS_SUCCESS) {
        convolution_algorithm = perfResults[0].algo;
    }
    else {
        convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    }

    // Workspace

    cudnnGetConvolutionForwardWorkspaceSize(convolutional_layer->get_cudnn_handle(),
                                            inputs_tensor_descriptor, kernel_descriptor, convolution_descriptor, outputs_tensor_descriptor,
                                            convolution_algorithm, &workspace_bytes);

    if (workspace_bytes > 0)
        cudaMalloc(&workspace, workspace_bytes);
    else
        workspace = nullptr;
}


void ConvolutionalLayerForwardPropagationCuda::print() const
{
    // @todo
}


void ConvolutionalLayerForwardPropagationCuda::free()
{
    cudaFree(outputs);
    cudaFree(convolutions);
    cudaFree(workspace);

    cudnnDestroyActivationDescriptor(activation_descriptor);
    cudnnDestroyTensorDescriptor(inputs_tensor_descriptor);
    cudnnDestroyTensorDescriptor(outputs_tensor_descriptor);
    cudnnDestroyTensorDescriptor(biases_tensor_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
}


pair<type*, dimensions> ConvolutionalLayerForwardPropagationCuda::get_outputs_pair_device() const
{
    const Convolutional* convolutional_layer = static_cast<Convolutional*>(layer);

    const Index output_height = convolutional_layer->get_output_height();
    const Index output_width = convolutional_layer->get_output_width();
    const Index kernels_number = convolutional_layer->get_kernels_number();

    return { outputs, {batch_size, output_height, output_width, kernels_number} };
}


ConvolutionalLayerBackPropagationCuda::ConvolutionalLayerBackPropagationCuda(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void ConvolutionalLayerBackPropagationCuda::set(const Index& new_batch_size, Layer* new_layer)
{
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

    if (cudaMalloc(&input_derivatives, input_size * sizeof(float)) != cudaSuccess)
        cout << "input derivatives allocation error" << endl;

    cudnnCreateTensorDescriptor(&inputs_tensor_descriptor);

    cudnnSetTensor4dDescriptor(inputs_tensor_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        channels,
        input_height,
        input_width);

    // Deltas

    cudnnCreateTensorDescriptor(&deltas_device_tensor_descriptor);

    cudnnSetTensor4dDescriptor(deltas_device_tensor_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        kernels_number,
        output_height,
        output_width);

    // Error combinations derivatives

    if (cudaMalloc(&error_combinations_derivatives_device, batch_size * output_height * output_width * kernels_number * sizeof(float)) != cudaSuccess)
        cout << "error_combinations_derivatives allocation error" << endl;

    cudnnCreateTensorDescriptor(&error_combinations_derivatives_tensor_descriptor);

    cudnnSetTensor4dDescriptor(error_combinations_derivatives_tensor_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        kernels_number,
        output_height,
        output_width);

    // Biases

    if (cudaMalloc(&biases_derivatives_device, kernels_number * sizeof(float)) != cudaSuccess)
        cout << "biases_derivatives allocation error" << endl;

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

    if (cudaMalloc(&weights_derivatives_device, kernel_size * sizeof(float)) != cudaSuccess)
        cout << "kernel derivatives allocation error" << endl;

    cudnnCreateFilterDescriptor(&weights_derivatives_tensor_descriptor);

    cudnnSetFilter4dDescriptor(weights_derivatives_tensor_descriptor,
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
        inputs_derivatives_tensor_descriptor,
        convolution_descriptor,
        inputs_tensor_descriptor,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        &backward_data_workspace_bytes);

    cudnnGetConvolutionBackwardFilterWorkspaceSize(convolutional_layer->get_cudnn_handle(),
        inputs_tensor_descriptor,
        inputs_derivatives_tensor_descriptor,
        convolution_descriptor,
        weights_derivatives_tensor_descriptor,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        &backward_filter_workspace_bytes);

    // Workspace memory

    if (cudaMalloc(&backward_data_workspace, backward_data_workspace_bytes) != cudaSuccess)
        cout << "backward data workspace allocation error" << endl;

    if (cudaMalloc(&backward_filter_workspace, backward_filter_workspace_bytes) != cudaSuccess)
        cout << "backward filter workspace allocation error" << endl;
}


vector<pair<type*, dimensions>> ConvolutionalLayerBackPropagationCuda::get_input_derivative_pairs_device() const
{
    const dimensions input_dimensions = layer->get_input_dimensions();

    return { {input_derivatives, {batch_size, input_dimensions[0], input_dimensions[1], input_dimensions[2]}}};
}


void ConvolutionalLayerBackPropagationCuda::print() const
{
    // @todo
}


void ConvolutionalLayerBackPropagationCuda::free()
{
    cudaFree(input_derivatives);
    cudaFree(error_combinations_derivatives_device);
    cudaFree(biases_derivatives_device);
    cudaFree(weights_derivatives_device);
    cudaFree(backward_data_workspace);
    cudaFree(backward_filter_workspace);

    cudnnDestroyTensorDescriptor(deltas_device_tensor_descriptor);
    cudnnDestroyTensorDescriptor(error_combinations_derivatives_tensor_descriptor);
    cudnnDestroyTensorDescriptor(inputs_tensor_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyFilterDescriptor(weights_derivatives_tensor_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
}

#endif

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
