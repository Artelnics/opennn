//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"

#include "convolutional_layer.h"
#include "tensors.h"

namespace opennn
{

ConvolutionalLayer::ConvolutionalLayer(const dimensions& new_input_dimensions,
                                       const dimensions& new_kernel_dimensions,
                                       const ConvolutionalLayer::ActivationFunction& new_activation_function,
                                       const dimensions& new_stride_dimensions,
                                       const ConvolutionalLayer::ConvolutionType& new_convolution_type,
                                       const string new_name) : Layer()
{
    layer_type = Layer::Type::Convolutional;

    set(new_input_dimensions, new_kernel_dimensions, new_activation_function, new_stride_dimensions, new_convolution_type, new_name);
}


bool ConvolutionalLayer::get_batch_normalization() const
{
    return batch_normalization;
}


void ConvolutionalLayer::preprocess_inputs(const Tensor<type, 4>& inputs,
                                           Tensor<type, 4>& preprocessed_inputs) const
{
    if (convolution_type == ConvolutionType::Same)
        preprocessed_inputs.device(*thread_pool_device) = inputs.pad(get_paddings());
    else
        preprocessed_inputs.device(*thread_pool_device) = inputs;

    if (row_stride != 1 || column_stride != 1)
        preprocessed_inputs.device(*thread_pool_device) = preprocessed_inputs.stride(get_strides());
}


void ConvolutionalLayer::calculate_convolutions(const Tensor<type, 4>& inputs,
                                                Tensor<type, 4>& convolutions) const
{
    type* convolutions_data = convolutions.data();

    type* synaptic_weights_data = (type*)synaptic_weights.data();

    // Convolutional layer

    const Index kernels_number = get_kernels_number();
    const Index kernel_height = get_kernel_height();
    const Index kernel_width = get_kernel_width();
    const Index kernel_channels = get_kernel_channels();

    const Index kernel_size = kernel_channels*kernel_height*kernel_width;

    const Index batch_samples_number = inputs.dimension(0);
    const Index output_height = get_output_height();
    const Index output_width = get_output_width();

    const Index output_size = batch_samples_number*output_height*output_width;

    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
        const TensorMap<Tensor<type, 3>> kernel(synaptic_weights_data + kernel_index*kernel_size,
                                                kernel_height,
                                                kernel_width,
                                                kernel_channels);

        TensorMap<Tensor<type, 4>> convolution(convolutions_data + kernel_index*output_size,
                                               batch_samples_number,
                                               output_height,
                                               output_width,
                                               1);

        convolution.device(*thread_pool_device) = inputs.convolve(kernel, convolutions_dimensions) + biases(kernel_index);
    }
}


void ConvolutionalLayer::normalize(unique_ptr<LayerForwardPropagation> layer_forward_propagation,
                                   const bool& is_training)
{
    ConvolutionalLayerForwardPropagation* convolutional_layer_forward_propagation =
        static_cast<ConvolutionalLayerForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 4>& outputs = convolutional_layer_forward_propagation->outputs;
    type* outputs_data = outputs.data();

    Tensor<type, 1>& means = convolutional_layer_forward_propagation->means;
    Tensor<type, 1>& standard_deviations = convolutional_layer_forward_propagation->standard_deviations;

    if(is_training)
        means.device(*thread_pool_device) = outputs.mean(means_dimensions);

    const Index batch_samples_number = convolutional_layer_forward_propagation->batch_samples_number;
    const Index output_height = get_output_height();
    const Index output_width = get_output_width();
    const Index single_output_size = batch_samples_number*output_height*output_width;

    const Index kernels_number = get_kernels_number();

    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
        TensorMap<Tensor<type, 4>> kernel_output(outputs_data + kernel_index*single_output_size,
                                                 batch_samples_number,
                                                 output_height,
                                                 output_width,
                                                 1);

        if(is_training)
        {
            TensorMap<Tensor<type, 0>> mean(&means(kernel_index));

            TensorMap<Tensor<type, 0>> standard_deviation(&standard_deviations(kernel_index));

            mean.device(*thread_pool_device) = kernel_output.mean();

            standard_deviation.device(*thread_pool_device) = (kernel_output - mean(0)).square().mean().sqrt();

            kernel_output.device(*thread_pool_device)
                = (kernel_output - means(kernel_index))/(standard_deviations(kernel_index) + epsilon);
        }
        else
        {
            kernel_output.device(*thread_pool_device) 
                = (kernel_output - moving_means(kernel_index))
                    / (moving_standard_deviations(kernel_index) + epsilon);
        }
    }

    if(is_training)
    {
        moving_means.device(*thread_pool_device)
            = momentum*moving_means + (type(1) - momentum)*means;

        moving_standard_deviations.device(*thread_pool_device)
            = momentum*moving_standard_deviations + (type(1) - momentum)*standard_deviations;
    }
}

/*
void ConvolutionalLayer::shift(LayerForwardPropagation* layer_forward_propagation)
{
    ConvolutionalLayerForwardPropagation convolutional_layer_forward_propagation =
            static_cast<ConvolutionalLayerForwardPropagation*>(layer_forward_propagation.get());

    type* outputs_data = convolutional_layer_forward_propagation.outputs.data();

    const Index batch_samples_number = convolutional_layer_forward_propagation.batch_samples_number;
    const Index output_height = get_output_height();
    const Index output_width = get_output_width();
    const Index kernels_number = get_kernels_number();
    const Index single_output_size = batch_samples_number * output_height * output_width;

    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
        TensorMap<Tensor<type, 4>> kernel_output(outputs_data + kernel_index*single_output_size,
                                                 batch_samples_number,
                                                 output_height,
                                                 output_width,
                                                 1);

        kernel_output.device(*thread_pool_device) 
            = kernel_output * scales(kernel_index) + offsets(kernel_index);
    }
}
*/

void ConvolutionalLayer::calculate_activations(Tensor<type, 4>& activations, Tensor<type, 4>& activation_derivatives) const
{
    switch(activation_function)
    {
    case ActivationFunction::Linear: linear(activations, activation_derivatives); return;

    case ActivationFunction::Logistic: logistic(activations, activation_derivatives); return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(activations, activation_derivatives); return;

    case ActivationFunction::RectifiedLinear: rectified_linear(activations, activation_derivatives); return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(activations, activation_derivatives); return;

    case ActivationFunction::SoftPlus: soft_plus(activations, activation_derivatives); return;

    case ActivationFunction::SoftSign: soft_sign(activations, activation_derivatives); return;

    case ActivationFunction::HardSigmoid: hard_sigmoid(activations, activation_derivatives); return;

    case ActivationFunction::ExponentialLinear: exponential_linear(activations, activation_derivatives); return;

    default: return;
    }
}


void ConvolutionalLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                           unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                           const bool& is_training)
{
    const TensorMap<Tensor<type, 4>> inputs = tensor_map_4(input_pairs[0]);

    ConvolutionalLayerForwardPropagation* convolutional_layer_forward_propagation =
        static_cast<ConvolutionalLayerForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 4>& outputs = convolutional_layer_forward_propagation->outputs;

    Tensor<type, 4>& preprocessed_inputs = convolutional_layer_forward_propagation->preprocessed_inputs;

    Tensor<type, 4>& activation_derivatives = convolutional_layer_forward_propagation->activation_derivatives;

    preprocess_inputs(inputs, preprocessed_inputs); 
    
    calculate_convolutions(preprocessed_inputs, outputs);

    if(batch_normalization)
    {
/*
        normalize(layer_forward_propagation,
                  is_training);

        shift(layer_forward_propagation);
*/
    }

    if(is_training)
        calculate_activations(outputs, activation_derivatives);
    else
        calculate_activations(outputs, empty);
}


void ConvolutionalLayer::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                        const vector<pair<type*, dimensions>>& delta_pairs,
                                        unique_ptr<LayerForwardPropagation>& forward_propagation,
                                        unique_ptr<LayerBackPropagation>& back_propagation) const
{
    // Convolutional layer

    const Index batch_samples_number = back_propagation->batch_samples_number;

    const Index input_height = get_input_height();
    const Index input_width = get_input_width();
    const Index input_channels = get_input_channels();

    const Index kernels_number = get_kernels_number();
    const Index kernel_height = get_kernel_height();
    const Index kernel_width = get_kernel_width();
    const Index kernel_channels = get_kernel_channels();

    const Index output_height = get_output_height();
    const Index output_width = get_output_width();

    const TensorMap<Tensor<type, 4>> inputs = tensor_map_4(input_pairs[0]);

    const TensorMap<Tensor<type, 4>> deltas = tensor_map_4(delta_pairs[0]);

    // Forward propagation

    ConvolutionalLayerForwardPropagation* convolutional_layer_forward_propagation =
            static_cast<ConvolutionalLayerForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 4>& activation_derivatives = convolutional_layer_forward_propagation->activation_derivatives;

    // Back propagation

    ConvolutionalLayerBackPropagation* convolutional_layer_back_propagation =
            static_cast<ConvolutionalLayerBackPropagation*>(back_propagation.get());

    Tensor<type, 4>& convolutions_derivatives =
        convolutional_layer_back_propagation->convolutions_derivatives;

    Tensor<type, 1>& biases_derivatives = convolutional_layer_back_propagation->biases_derivatives;

    type* synaptic_weights_derivatives_data = convolutional_layer_back_propagation->synaptic_weights_derivatives.data();

    type* synaptic_weights_data = (type*)synaptic_weights.data();

    Tensor<type, 4>& kernel_synaptic_weights_derivatives = convolutional_layer_back_propagation->kernel_synaptic_weights_derivatives;

    Tensor<type, 4>& input_derivatives = convolutional_layer_back_propagation->input_derivatives;

    const Index kernel_synaptic_weights_number = kernel_channels*kernel_height*kernel_width;

    Tensor<type, 0> biases_derivatives_sum;

    Tensor<type, 3> rotated_kernel_synaptic_weights;
    Tensor<type, 2> rotated_kernel_slice;
    Tensor<type, 2> image_kernel_convolutions_derivatives_padded;
    Tensor<type, 2> channel_convolution(input_height, input_width);

    const Index pad_height = (input_height + kernel_height - 1) - output_height;
    const Index pad_width = (input_width + kernel_width - 1) - output_width;
    const Index pad_top = pad_height / 2;
    const Index pad_bottom = pad_height - pad_top;
    const Index pad_left = pad_width / 2;
    const Index pad_right = pad_width - pad_left;

    const Eigen::array<pair<Index, Index>, 2> paddings 
        = { make_pair(pad_top, pad_bottom), make_pair(pad_left, pad_right) };

    // Convolutions derivatives

    convolutions_derivatives.device(*thread_pool_device) = deltas*activation_derivatives;

    // Biases synaptic weights and input derivatives

    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {       
        TensorMap<Tensor<type, 3>> kernel_synaptic_weights(
                                   synaptic_weights_data + kernel_index * kernel_synaptic_weights_number,
                                   kernel_height,
                                   kernel_width,
                                   kernel_channels);

        TensorMap<Tensor<type, 3>> kernel_convolutions_derivatives(
                                   convolutions_derivatives.data() + kernel_index * batch_samples_number * output_height * output_width,
                                   batch_samples_number,
                                   output_height,
                                   output_width);
        
        // Biases derivatives

        biases_derivatives_sum.device(*thread_pool_device) = kernel_convolutions_derivatives.sum();

        biases_derivatives(kernel_index) = biases_derivatives_sum();

        // Synaptic weights derivatives

        kernel_synaptic_weights_derivatives = inputs.convolve(kernel_convolutions_derivatives, convolutions_dimensions_3d);

        memcpy(synaptic_weights_derivatives_data + kernel_synaptic_weights_number * kernel_index,
               kernel_synaptic_weights_derivatives.data(),
               kernel_synaptic_weights_number * sizeof(type));

        // Input derivatives

        rotated_kernel_synaptic_weights = kernel_synaptic_weights.reverse(reverse_dimensions);
        
        for (Index image_index = 0; image_index < batch_samples_number; image_index++)
        {
            image_kernel_convolutions_derivatives_padded 
                = kernel_convolutions_derivatives.chip(image_index, 0).pad(paddings);

            for (Index channel_index = 0; channel_index < input_channels; ++channel_index)
            {
                rotated_kernel_slice = rotated_kernel_synaptic_weights.chip(channel_index, 2);

                channel_convolution.device(*thread_pool_device) 
                    = image_kernel_convolutions_derivatives_padded.convolve(rotated_kernel_slice, convolution_dimensions_2d);

                #pragma omp parallel for
                for(Index x = 0; x < input_height; ++x)
                    for(Index y = 0; y < input_width; ++y)
                        input_derivatives(image_index, x, y, channel_index) += channel_convolution(x, y);
            }
        }
    }
    cout << "biases derivatives: " << biases_derivatives << endl;
    cout << "kernel_synaptic_weights_derivatives " << kernel_synaptic_weights_derivatives << endl;
}


void ConvolutionalLayer::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                         const Index& index,
                                         Tensor<type, 1>& gradient) const
{
    // Convolutional layer

    const Index synaptic_weights_number = synaptic_weights.size();
    const Index biases_number = biases.size();

    // Back-propagation

    ConvolutionalLayerBackPropagation* convolutional_layer_back_propagation =
        static_cast<ConvolutionalLayerBackPropagation*>(back_propagation.get());

    const type* synaptic_weights_derivatives_data = convolutional_layer_back_propagation->synaptic_weights_derivatives.data();
    const type* biases_derivatives_data = convolutional_layer_back_propagation->biases_derivatives.data();

    #pragma omp parallel sections
    {
        #pragma omp section
        memcpy(gradient.data() + index, synaptic_weights_derivatives_data, synaptic_weights_number * sizeof(type));

        #pragma omp section
        memcpy(gradient.data() + index + synaptic_weights_number, biases_derivatives_data, biases_number * sizeof(type));
    }
}


ConvolutionalLayer::ActivationFunction ConvolutionalLayer::get_activation_function() const
{
    return activation_function;
}


string ConvolutionalLayer::get_activation_function_string() const
{
    switch(activation_function)
    {
    case ActivationFunction::Logistic:
        return "Logistic";

    case ActivationFunction::HyperbolicTangent:
        return "HyperbolicTangent";

    case ActivationFunction::Linear:
        return "Linear";

    case ActivationFunction::RectifiedLinear:
        return "RectifiedLinear";

    case ActivationFunction::ScaledExponentialLinear:
        return "ScaledExponentialLinear";

    case ActivationFunction::SoftPlus:
        return "SoftPlus";

    case ActivationFunction::SoftSign:
        return "SoftSign";

    case ActivationFunction::HardSigmoid:
        return "HardSigmoid";

    case ActivationFunction::ExponentialLinear:
        return "ExponentialLinear";
    }

    return string();
}


Index ConvolutionalLayer::get_output_height() const
{
    const Index input_height = get_input_height();
    const Index kernel_height = get_kernel_height();
    const Index strides = get_row_stride();

    const pair<Index, Index> padding = get_padding();

    return floor((input_height - kernel_height + 2*padding.first)/strides) + 1;
}


Index ConvolutionalLayer::get_output_width() const
{
    const Index input_width = get_input_width();
    const Index kernel_width = get_kernel_width();
    const Index strides = get_column_stride();

    const pair<Index, Index> padding = get_padding();

    return floor((input_width - kernel_width + 2*padding.second)/strides) + 1;
}


dimensions ConvolutionalLayer::get_input_dimensions() const
{
    return input_dimensions;
}


dimensions ConvolutionalLayer::get_output_dimensions() const
{
    const Index rows_number = get_output_height();
    const Index columns_number = get_output_width();
    const Index kernels_number = get_kernels_number();

    return { rows_number, columns_number, kernels_number };
}


ConvolutionalLayer::ConvolutionType ConvolutionalLayer::get_convolution_type() const
{
    return convolution_type;
}


string ConvolutionalLayer::write_convolution_type() const
{
    switch(convolution_type)
    {
    case ConvolutionType::Valid:
        return "Valid";

    case ConvolutionType::Same:
        return "Same";
    }

    return string();
}


Index ConvolutionalLayer::get_column_stride() const
{
    return column_stride;
}


Index ConvolutionalLayer::get_row_stride() const
{
    return row_stride;
}


Index  ConvolutionalLayer::get_kernel_height() const
{
    return synaptic_weights.dimension(0);
}


Index ConvolutionalLayer::get_kernel_width() const
{
    return synaptic_weights.dimension(1);
}


Index ConvolutionalLayer::get_kernel_channels() const
{
    return synaptic_weights.dimension(2);
}


Index ConvolutionalLayer::get_kernels_number() const
{
    return synaptic_weights.dimension(3);
}


Index ConvolutionalLayer::get_padding_height() const
{
    switch (convolution_type)
    {
    case ConvolutionType::Valid:
        return 0;

    case ConvolutionType::Same:
        return row_stride * (input_dimensions[1] - 1) - input_dimensions[1] + get_kernel_height();
    }

    throw runtime_error("Unknown convolution type");
}


Index ConvolutionalLayer::get_padding_width() const
{
    switch(convolution_type)
    {
    case ConvolutionType::Valid:
        return 0;

    case ConvolutionType::Same:
        return column_stride*(input_dimensions[2] - 1) - input_dimensions[2] + get_kernel_width();
    }

    throw runtime_error("Unknown convolution type");
}


Tensor<type, 1> ConvolutionalLayer::get_parameters() const
{
    Tensor<type, 1> parameters(get_parameters_number());

    memcpy(parameters.data(), synaptic_weights.data(), synaptic_weights.size()*sizeof(type));

    memcpy(parameters.data() + synaptic_weights.size(), biases.data(), biases.size()*sizeof(type));

// @todo add scales and offsets

    return parameters;
}


Index ConvolutionalLayer::get_parameters_number() const
{
    return synaptic_weights.size() + biases.size();
}


void ConvolutionalLayer::set(const dimensions& new_input_dimensions,
                             const dimensions& new_kernel_dimensions,
                             const ConvolutionalLayer::ActivationFunction& new_activation_function,
                             const dimensions& new_stride_dimensions,
                             const ConvolutionType& new_convolution_type,
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
    
    set_input_dimensions(new_input_dimensions);

    const Index kernel_height = new_kernel_dimensions[0];
    const Index kernel_width = new_kernel_dimensions[1];
    const Index kernel_channels = new_kernel_dimensions[2];
    const Index kernels_number = new_kernel_dimensions[3];

    set_row_stride(new_stride_dimensions[0]);
    set_column_stride(new_stride_dimensions[1]);

    set_activation_function(new_activation_function);

    set_convolution_type(new_convolution_type);

    biases.resize(kernels_number);
    set_random(biases);

    synaptic_weights.resize(kernel_height,
                            kernel_width,
                            kernel_channels,
                            kernels_number);

    set_random(synaptic_weights);

    moving_means.resize(kernels_number);
    moving_standard_deviations.resize(kernels_number);

    scales.resize(kernels_number);
    offsets.resize(kernels_number);

    set_name(new_name);
}


void ConvolutionalLayer::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    synaptic_weights.setConstant(value);
}


void ConvolutionalLayer::set_parameters_random()
{
    set_random(biases);

    set_random(synaptic_weights);
}


void ConvolutionalLayer::set_activation_function(const ConvolutionalLayer::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


void ConvolutionalLayer::set_activation_function(const string& new_activation_function_name)
{
    if(new_activation_function_name == "Logistic")
        activation_function = ActivationFunction::Logistic;
    else if(new_activation_function_name == "HyperbolicTangent")
        activation_function = ActivationFunction::HyperbolicTangent;
    else if(new_activation_function_name == "Linear")
        activation_function = ActivationFunction::Linear;
    else if(new_activation_function_name == "RectifiedLinear")
        activation_function = ActivationFunction::RectifiedLinear;
    else if(new_activation_function_name == "ScaledExponentialLinear")
        activation_function = ActivationFunction::ScaledExponentialLinear;
    else if(new_activation_function_name == "SoftPlus")
        activation_function = ActivationFunction::SoftPlus;
    else if(new_activation_function_name == "SoftSign")
        activation_function = ActivationFunction::SoftSign;
    else if(new_activation_function_name == "HardSigmoid")
        activation_function = ActivationFunction::HardSigmoid;
    else if(new_activation_function_name == "ExponentialLinear")
        activation_function = ActivationFunction::ExponentialLinear;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function_name + ".\n");
}


void ConvolutionalLayer::set_batch_normalization(const bool& new_batch_normalization)
{
    batch_normalization = new_batch_normalization;
}


void ConvolutionalLayer::set_convolution_type(const ConvolutionalLayer::ConvolutionType& new_convolution_type)
{
    convolution_type = new_convolution_type;
}


void ConvolutionalLayer::set_convolution_type(const string& new_convolution_type)
{
    if(new_convolution_type == "Valid")
        convolution_type = ConvolutionType::Valid;
    else if(new_convolution_type == "Same")
        convolution_type = ConvolutionType::Same;
    else
        throw runtime_error("Unknown convolution type: " + new_convolution_type + ".\n");
}


void ConvolutionalLayer::set_row_stride(const Index& new_stride_row)
{
    if(new_stride_row <= 0)
        throw runtime_error("EXCEPTION: new_stride_row must be a positive number");

    row_stride = new_stride_row;
}


void ConvolutionalLayer::set_column_stride(const Index& new_stride_column)
{
    if(new_stride_column <= 0)
        throw runtime_error("EXCEPTION: new_stride_column must be a positive number");

    column_stride = new_stride_column;
}


void ConvolutionalLayer::set_input_dimensions(const dimensions& new_input_dimensions)
{
    if (new_input_dimensions.size() != 3)
        throw runtime_error("Input new_input_dimensions.size() must be 3");

    input_dimensions = new_input_dimensions;
}


void ConvolutionalLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    #pragma omp parallel sections
    {
    #pragma omp section
    memcpy(synaptic_weights.data(), new_parameters.data() + index, synaptic_weights.size()*sizeof(type));
    
    #pragma omp section
    memcpy(biases.data(), new_parameters.data() + index + synaptic_weights.size(), biases.size()*sizeof(type));
    }
}


pair<Index, Index> ConvolutionalLayer::get_padding() const
{
    switch(convolution_type)
    {
    case ConvolutionType::Valid:
        return make_pair(0, 0);

    case ConvolutionType::Same:
    {
        const Index input_height = get_input_height();
        const Index input_width = get_input_width();

        const Index kernel_height = get_kernel_height();
        const Index kernel_width = get_kernel_width();

        const Index row_stride = get_row_stride();
        const Index column_stride = get_column_stride();

        const Index pad_rows = std::max<Index>(0, ((static_cast<float>(input_height) / row_stride) - 1) * row_stride + kernel_height - input_height) / 2;
        const Index pad_columns = std::max<Index>(0, ((static_cast<float>(input_width) / column_stride) - 1) * column_stride + kernel_width - input_width) / 2;

        return make_pair(pad_rows, pad_columns);
    }
    default:
        throw runtime_error("Unknown convolution type.\n");
    }
}


Eigen::array<pair<Index, Index>, 4> ConvolutionalLayer::get_paddings() const
{
    const Index pad_rows = get_padding().first;
    const Index pad_columns = get_padding().second;

    const Eigen::array<std::pair<Index, Index>, 4> paddings =
        { make_pair(0, 0),
          make_pair(pad_rows, pad_rows),
          make_pair(pad_columns, pad_columns),
          make_pair(0, 0) };

    return paddings;
}


Eigen::array<ptrdiff_t, 4> ConvolutionalLayer::get_strides() const
{   
    return Eigen::array<ptrdiff_t, 4>({1, row_stride, column_stride, 1});
}


Index ConvolutionalLayer::get_input_height() const
{
    return input_dimensions[0];
}


Index ConvolutionalLayer::get_input_width() const
{
    return input_dimensions[1];
}


Index ConvolutionalLayer::get_input_channels() const
{
    return input_dimensions[2];
}


void ConvolutionalLayer::print() const
{
    cout << "Convolutional layer" << endl;
    cout << "Input dimensions: " << endl;
    print_vector(input_dimensions);
    cout << "Output dimensions: " << endl;
    print_vector(get_output_dimensions());
}


void ConvolutionalLayer::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Convolutional");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "InputDimensions", dimensions_to_string(input_dimensions));
    //add_xml_element(printer, "OutputDimensions", dimensions_to_string(get_output_dimensions()));
    add_xml_element(printer, "KernelsNumber", to_string(get_kernels_number()));
    add_xml_element(printer, "KernelsHeight", to_string(get_kernel_height()));
    add_xml_element(printer, "KernelsWidth", to_string(get_kernel_width()));
    add_xml_element(printer, "KernelsChannels", to_string(get_kernel_channels()));
    add_xml_element(printer, "ActivationFunction", get_activation_function_string());
    add_xml_element(printer, "StrideDimensions", dimensions_to_string({ get_column_stride(), get_row_stride() }));
    add_xml_element(printer, "ConvolutionType", write_convolution_type());
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();
}


void ConvolutionalLayer::from_XML(const XMLDocument& document)
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

    synaptic_weights.resize(kernel_height,
        kernel_width,
        kernel_channels,
        kernels_number);

    set_activation_function(read_xml_string(convolutional_layer_element, "ActivationFunction"));

    const dimensions stride_dimensions = string_to_dimensions(read_xml_string(convolutional_layer_element, "StrideDimensions"));
    set_column_stride(stride_dimensions[0]);
    set_row_stride(stride_dimensions[1]);

    set_convolution_type(read_xml_string(convolutional_layer_element, "ConvolutionType"));

    set_parameters(string_to_tensor(read_xml_string(convolutional_layer_element, "Parameters")));
}


ConvolutionalLayerForwardPropagation::ConvolutionalLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


pair<type*, dimensions> ConvolutionalLayerForwardPropagation::get_outputs_pair() const
{
    const ConvolutionalLayer* convolutional_layer = static_cast<ConvolutionalLayer*>(layer);

    const Index output_height = convolutional_layer->get_output_height();
    const Index output_width = convolutional_layer->get_output_width();
    const Index kernels_number = convolutional_layer->get_kernels_number();

    return {(type*)outputs.data(), {batch_samples_number, output_height, output_width, kernels_number}};
}


void ConvolutionalLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_samples_number = new_batch_samples_number;
   
    layer = new_layer;

    const ConvolutionalLayer* convolutional_layer = static_cast<ConvolutionalLayer*>(layer);

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index input_channels = convolutional_layer->get_input_channels();

    const Index kernels_number = convolutional_layer->get_kernels_number();

    const Index output_height = convolutional_layer->get_output_height();
    const Index output_width = convolutional_layer->get_output_width();

    const Index padding_height = convolutional_layer->get_padding_height();
    const Index padding_width = convolutional_layer->get_padding_width();
    
    preprocessed_inputs.resize(batch_samples_number,
                               input_height + padding_height,
                               input_width + padding_width,
                               input_channels);

    outputs.resize(batch_samples_number,
                   output_height,
                   output_width,
                   kernels_number);

    means.resize(kernels_number);

    standard_deviations.resize(kernels_number);

    activation_derivatives.resize(batch_samples_number,
                                   output_height,
                                   output_width,
                                   kernels_number);
}


void ConvolutionalLayerForwardPropagation::print() const
{
    cout << "Convolutional layer" << endl
         << "Outputs:" << endl
         << outputs << endl
         << "Activation derivatives:" << endl
         << activation_derivatives << endl;
}


ConvolutionalLayerBackPropagation::ConvolutionalLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer);
}


void ConvolutionalLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{

    batch_samples_number = new_batch_samples_number;

    layer = new_layer;

    const ConvolutionalLayer* convolutional_layer = static_cast<ConvolutionalLayer*>(layer);

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index channels = convolutional_layer->get_input_channels();

    const Index kernel_height = convolutional_layer->get_kernel_height();
    const Index kernel_width = convolutional_layer->get_kernel_width();
    const Index kernel_channels = convolutional_layer->get_kernel_channels();
    const Index kernels_number = convolutional_layer->get_kernels_number();

    const Index output_height = convolutional_layer->get_output_height();
    const Index output_width = convolutional_layer->get_output_width();

    convolutions_derivatives.resize(batch_samples_number,
                                    output_height,
                                    output_width,
                                    kernels_number);

    biases_derivatives.resize(kernels_number);

    synaptic_weights_derivatives.resize(kernels_number,
                                        kernel_height,
                                        kernel_width,
                                        kernel_channels);

    kernel_synaptic_weights_derivatives.resize(1,
                                               kernel_height,
                                               kernel_width,
                                               kernel_channels);

    //image_convolutions_derivatives.resize(output_height,
    //                                      output_width,
    //                                      1);

    input_derivatives.resize(batch_samples_number,
                             input_height,
                             input_width,
                             channels);
}


vector<pair<type*, dimensions>> ConvolutionalLayerBackPropagation::get_input_derivative_pairs() const
{
    const ConvolutionalLayer* convolutional_layer = static_cast<ConvolutionalLayer*>(layer);

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index channels = convolutional_layer->get_input_channels();

    convolutional_layer->get_input_dimensions();

    return {{(type*)input_derivatives.data(), {batch_samples_number, input_height, input_width, channels}}};
}


void ConvolutionalLayerBackPropagation::print() const
{
    cout << "Convolutional layer back propagation" << endl
         << "Biases derivatives:\n" << endl
         << biases_derivatives << endl
         << "Synaptic weights derivatives:\n" << endl
         << synaptic_weights_derivatives << endl;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
