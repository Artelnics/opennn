//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "convolutional_layer.h"
#include "tensors.h"

namespace opennn
{

/// Default constructor.
/// It creates an empty ConvolutionalLayer object.

ConvolutionalLayer::ConvolutionalLayer() : Layer()
{
    layer_type = Layer::Type::Convolutional;
}


/// Inputs' dimensions modifier constructor.
/// After setting new dimensions for the inputs, it creates and initializes a ConvolutionalLayer object
/// with a number of kernels of a given size.
/// The initialization values are random values from a normal distribution.
/// @param new_input_dimensions A vector containing the new inputs' dimensions.
/// @param kernels_dimensions A vector containing the number, kernel rows, columns and channels.

ConvolutionalLayer::ConvolutionalLayer(const dimensions& new_input_dimensions,
                                       const dimensions& new_kernel_dimensions) : Layer()
{
    layer_type = Layer::Type::Convolutional;

    set(new_input_dimensions, new_kernel_dimensions);
}


/// Returns a boolean, true if convolutional layer is empty and false otherwise.

bool ConvolutionalLayer::is_empty() const
{
    if(biases.size() == 0 && synaptic_weights.size() == 0)
    {
        return true;
    }

    return false;
}


/// Returns the layer's biases.

const Tensor<type, 1>& ConvolutionalLayer::get_biases() const
{
    return biases;
}


/// Returns the layer's synaptic weights.

const Tensor<type, 4>& ConvolutionalLayer::get_synaptic_weights() const
{
    return synaptic_weights;
}


bool ConvolutionalLayer::get_batch_normalization() const
{
    return batch_normalization;
}


/// Inserts padding to the input tensor.
/// @param input Tensor containing the inputs.
/// @param padded_output input tensor padded.

// void ConvolutionalLayer::insert_padding(const Tensor<type, 4>& inputs, Tensor<type, 4>& padded_output) const
// {
//     switch(convolution_type)
//     {
//     case ConvolutionType::Valid:

//         padded_output.device(*thread_pool_device) = inputs;

//         return;

//     case ConvolutionType::Same:

//         const Index pad_rows = get_padding().first;
//         const Index pad_columns = get_padding().second;

//         Eigen::array<pair<Index, Index>, 4> paddings;
//         paddings[0] = make_pair(0, 0);
//         paddings[1] = make_pair(pad_rows, pad_rows);
//         paddings[2] = make_pair(pad_columns, pad_columns);
//         paddings[3] = make_pair(0, 0);

//         padded_output.device(*thread_pool_device) = inputs.pad(paddings);

//         return;
//     }
// }


void ConvolutionalLayer::preprocess_inputs(const Tensor<type, 4>& inputs,
                                           Tensor<type, 4>& preprocessed_inputs) const
{
    if(convolution_type == ConvolutionType::Same)
    {
        const Eigen::array<pair<Index, Index>, 4> paddings = get_paddings();

        preprocessed_inputs.device(*thread_pool_device) = inputs.pad(paddings);
    }
    else
    {
        preprocessed_inputs.device(*thread_pool_device) = inputs;
    }

    if(row_stride != 1 || column_stride != 1)
    {
        const Eigen::array<ptrdiff_t, 4> strides = get_strides();

        preprocessed_inputs.device(*thread_pool_device) = preprocessed_inputs.stride(strides);
    }
}


/// Calculate convolutions

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

        convolution.device(*thread_pool_device)
            = inputs.convolve(kernel, convolutions_dimensions)
            + biases(kernel_index);
    }
}


// Batch normalization

void ConvolutionalLayer::normalize(LayerForwardPropagation* layer_forward_propagation,
                                   const bool& is_training)
{
    ConvolutionalLayerForwardPropagation* convolutional_layer_forward_propagation
            = static_cast<ConvolutionalLayerForwardPropagation*>(layer_forward_propagation);

    type* outputs_data = convolutional_layer_forward_propagation->outputs.data();

    Tensor<type, 4>& outputs = convolutional_layer_forward_propagation->outputs;

    Tensor<type, 1>& means = convolutional_layer_forward_propagation->means;
    Tensor<type, 1>& standard_deviations = convolutional_layer_forward_propagation->standard_deviations;

    if(is_training)
    {
        means.device(*thread_pool_device) = outputs.mean(means_dimensions);
    }

    const Index batch_samples_number = convolutional_layer_forward_propagation->batch_samples_number;
    const Index output_height = get_output_height();
    const Index output_width = get_output_width();
    const Index kernels_number = get_kernels_number();
    const Index single_output_size = batch_samples_number*output_height*output_width;

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
            kernel_output.device(*thread_pool_device) = (kernel_output - moving_means(kernel_index))
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


void ConvolutionalLayer::shift(LayerForwardPropagation* layer_forward_propagation)
{
    ConvolutionalLayerForwardPropagation* convolutional_layer_forward_propagation
            = static_cast<ConvolutionalLayerForwardPropagation*>(layer_forward_propagation);

    type* outputs_data = convolutional_layer_forward_propagation->outputs.data();

    const Index batch_samples_number = convolutional_layer_forward_propagation->batch_samples_number;
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

        kernel_output.device(*thread_pool_device) = kernel_output * scales(kernel_index) + offsets(kernel_index);
    }
}


/// Calculate activations

void ConvolutionalLayer::calculate_activations(const Tensor<type, 4>& convolutions,
                                               Tensor<type, 4>& activations) const
{
    switch(activation_function)
    {
    case ActivationFunction::Linear: linear(convolutions, activations); return;

    case ActivationFunction::Logistic: logistic(convolutions, activations); return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(convolutions, activations); return;

    case ActivationFunction::RectifiedLinear: rectified_linear(convolutions, activations); return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(convolutions, activations); return;

    case ActivationFunction::SoftPlus: soft_plus(convolutions, activations); return;

    case ActivationFunction::SoftSign: soft_sign(convolutions, activations); return;

    case ActivationFunction::HardSigmoid: hard_sigmoid(convolutions, activations); return;

    case ActivationFunction::ExponentialLinear: exponential_linear(convolutions, activations); return;

    default: return;
    }
}


/// Calculates activations derivatives

void ConvolutionalLayer::calculate_activations_derivatives(const Tensor<type, 4>& convolutions,
                                                           Tensor<type, 4>& activations,
                                                           Tensor<type, 4>& activations_derivatives) const
{
    switch(activation_function)
    {
    case ActivationFunction::Linear: linear_derivatives(convolutions,
                                                        activations,
                                                        activations_derivatives);
        return;

    case ActivationFunction::Logistic: logistic_derivatives(convolutions,
                                                            activations,
                                                            activations_derivatives);
        return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent_derivatives(convolutions,
                                                                               activations,
                                                                               activations_derivatives);
        return;

    case ActivationFunction::RectifiedLinear: rectified_linear_derivatives(convolutions,
                                                                           activations,
                                                                           activations_derivatives);
        return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear_derivatives(convolutions,
                                                                                            activations,
                                                                                            activations_derivatives);
        return;

    case ActivationFunction::SoftPlus: soft_plus_derivatives(convolutions,
                                                             activations,
                                                             activations_derivatives);
        return;

    case ActivationFunction::SoftSign: soft_sign_derivatives(convolutions,
                                                             activations,
                                                             activations_derivatives);
        return;

    case ActivationFunction::HardSigmoid: hard_sigmoid_derivatives(convolutions,
                                                                   activations,
                                                                   activations_derivatives);
        return;

    case ActivationFunction::ExponentialLinear: exponential_linear_derivatives(convolutions,
                                                                               activations,
                                                                               activations_derivatives);
        return;

    default:

        return;
    }
}


void ConvolutionalLayer::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                           LayerForwardPropagation* layer_forward_propagation,
                                           const bool& is_training)
{
    const TensorMap<Tensor<type, 4>> inputs(inputs_pair(0).first,
                                            inputs_pair(0).second[0],
                                            inputs_pair(0).second[1],
                                            inputs_pair(0).second[2],
                                            inputs_pair(0).second[3]);

    ConvolutionalLayerForwardPropagation* convolutional_layer_forward_propagation =
            static_cast<ConvolutionalLayerForwardPropagation*>(layer_forward_propagation);

    Tensor<type, 4>& outputs = convolutional_layer_forward_propagation->outputs;

    Tensor<type, 4>& preprocessed_inputs = convolutional_layer_forward_propagation->preprocessed_inputs;

    Tensor<type, 4>& activations_derivatives = convolutional_layer_forward_propagation->activations_derivatives;

    preprocess_inputs(inputs, preprocessed_inputs);

    calculate_convolutions(preprocessed_inputs, outputs);

    if(batch_normalization)
    {
        normalize(layer_forward_propagation,
                  is_training);

        shift(layer_forward_propagation);
    }

    if(is_training)
    {
        calculate_activations_derivatives(outputs,
                                          outputs,
                                          activations_derivatives);
    }
    else
    {
        calculate_activations(outputs,
                              outputs);
    }
}


/* Do this at the end of back_propagate(), with input_derivatives instead of deltas

void ConvolutionalLayer::calculate_hidden_delta(ConvolutionalLayerForwardPropagation* next_convolutional_layer_forward_propagation,
                                                ConvolutionalLayerBackPropagation* next_convolutional_layer_back_propagation,
                                                ConvolutionalLayerBackPropagation* this_convolutional_layer_back_propagation) const
{
    const Tensor<type, 4> next_deltas = next_convolutional_layer_back_propagation->deltas;

    // (next deltas * activations derivatives) ¿convolve? kernels

    const Index kernels_number = get_kernels_number();

    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {

    }

    next_deltas * next_convolutional_layer_forward_propagation->activations_derivatives;
}
*/


void ConvolutionalLayer::back_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                        const Tensor<pair<type*, dimensions>, 1>& deltas_pair,
                                        LayerForwardPropagation* forward_propagation,
                                        LayerBackPropagation* back_propagation) const
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

    const TensorMap<Tensor<type, 4>> inputs(inputs_pair(0).first,
                                            inputs_pair(0).second[0],
                                            inputs_pair(0).second[1],
                                            inputs_pair(0).second[2],
                                            inputs_pair(0).second[3]);

    const TensorMap<Tensor<type, 4>> deltas(deltas_pair(0).first,
                                            deltas_pair(0).second[0],
                                            deltas_pair(0).second[1],
                                            deltas_pair(0).second[2],
                                            deltas_pair(0).second[3]);

    // Forward propagation

    ConvolutionalLayerForwardPropagation* convolutional_layer_forward_propagation =
            static_cast<ConvolutionalLayerForwardPropagation*>(forward_propagation);

    const Tensor<type, 4>& activations_derivatives = convolutional_layer_forward_propagation->activations_derivatives;

    // Back propagation

    ConvolutionalLayerBackPropagation* convolutional_layer_back_propagation =
            static_cast<ConvolutionalLayerBackPropagation*>(back_propagation);

    Tensor<type, 4>& error_convolutions_derivatives =
        convolutional_layer_back_propagation->error_convolutions_derivatives;

    Tensor<type, 1>& biases_derivatives = convolutional_layer_back_propagation->biases_derivatives;

    Eigen::array<Index, 4> offsets;
    Eigen::array<Index, 4> extents;

    const Eigen::array<ptrdiff_t, 3> convolutions_dimensions = {0, 1, 2};

    Tensor<type, 4> delta_slice;
    Tensor<type, 4> image_slice;

    type* synaptic_weights_derivatives_data = convolutional_layer_back_propagation->synaptic_weights_derivatives.data();

    const Index kernel_synaptic_weights_number = kernel_channels*kernel_height*kernel_width;

    Tensor<type, 0> current_sum;

/*
    error_convolutions_derivatives.device(*thread_pool_device) = deltas * activations_derivatives;
*/
    // Synaptic weights derivatives
    
    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
        current_sum.setZero();
        
        TensorMap<Tensor<type, 3>> kernel_synaptic_weights_derivatives(synaptic_weights_derivatives_data + kernel_index*kernel_synaptic_weights_number,
                                                                       kernel_height,
                                                                       kernel_width,
                                                                       kernel_channels);

        kernel_synaptic_weights_derivatives.setZero();

        for(Index image_index = 0; image_index < batch_samples_number; image_index++)
        {
/*
            offsets = {image_index, 0, 0, kernel_index};

            extents = {1, output_height, output_width, 1};

            delta_slice
                = error_convolutions_derivatives.slice(offsets, extents); // device(*thread_pool_device) does not work here

            offsets = {image_index, 0, 0, 0};
            extents = {1, input_height, input_width, input_channels};

            image_slice = inputs.slice(offsets, extents); // device(*thread_pool_device) does not work here

            const TensorMap<Tensor<type, 3>> image(image_slice.data(),
                                                   input_height,
                                                   input_width,
                                                   input_channels);

            const TensorMap<Tensor<type, 3>> delta_reshape(delta_slice.data(),
                                                           output_height,
                                                           output_width,
                                                           1);

            current_sum.device(*thread_pool_device) = delta_slice.sum();

            biases_derivatives(kernel_index) += current_sum();

            kernel_synaptic_weights_derivatives.device(*thread_pool_device)
                += image.convolve(delta_reshape, convolutions_dimensions);
*/
        }

        copy(kernel_synaptic_weights_derivatives.data(),
             kernel_synaptic_weights_derivatives.data() + kernel_synaptic_weights_number,
             synaptic_weights_derivatives_data + kernel_synaptic_weights_number * kernel_index);
    }

    Tensor<type, 4>& input_derivatives = convolutional_layer_back_propagation->input_derivatives;
    input_derivatives.setZero();

    /*
//    input_derivatives = error_convolutions_derivatives.convolve(synaptic_weights,convolutions_dimensions);

    for(int image_index = 0; image_index < batch_samples_number; image_index++)
    {
        for(int d = 0; d < deltas_pair(0).second[2]; ++d)
        {
            for(int row = 0; row < deltas_pair(0).second[0]; row++)
            {
                for(int column = 0; column < deltas_pair(0).second[1]; column++)
                {
                     input_derivatives.chip(image_index, 0).chip(d, 0).slice(std::array<Index, 2>({row, column}), std::array<Index, 2>({kernel_height, kernel_width})) +=
                             error_convolutions_derivatives(row, column, d, image_index) * synaptic_weights.chip(d, 0);
                }
            }
        }
    }
    */
}


void ConvolutionalLayer::insert_gradient(LayerBackPropagation* back_propagation,
                                         const Index& index,
                                         Tensor<type, 1>& gradient) const
{
    type* gradient_data = gradient.data();

    // Convolutional layer

    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    // Back-propagation

    ConvolutionalLayerBackPropagation* convolutional_layer_back_propagation =
            static_cast<ConvolutionalLayerBackPropagation*>(back_propagation);

    const type* biases_derivatives_data = convolutional_layer_back_propagation->biases_derivatives.data();

    const type* synaptic_weights_derivatives_data = convolutional_layer_back_propagation->synaptic_weights_derivatives.data();

    // Copy from back propagation to gradient

    copy(/*execution::par,*/ 
         synaptic_weights_derivatives_data,
         synaptic_weights_derivatives_data + synaptic_weights_number,
         gradient_data + index);

    copy(/*execution::par,*/ 
         biases_derivatives_data,
         biases_derivatives_data + biases_number,
         gradient_data + index + synaptic_weights_number);
}


/// Returns the convolutional layer's activation function.

ConvolutionalLayer::ActivationFunction ConvolutionalLayer::get_activation_function() const
{
    return activation_function;
}


/// Returns a string with the name of the layer activation function.
/// This can be Logistic, HyperbolicTangent, Linear, RectifiedLinear, ScaledExponentialLinear.

string ConvolutionalLayer::write_activation_function() const
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


/// Returns the number of rows the result of applying the layer's kernels to an image will have.

Index ConvolutionalLayer::get_output_height() const
{
    const Index input_height = get_input_height();
    const Index kernel_height = get_kernel_height();
    const Index strides = get_row_stride();

    const pair<Index, Index> padding = get_padding();

    return floor((input_height - kernel_height + 2*padding.first)/strides) + 1;
}


/// Returns the number of columns the result of applying the layer's kernels to an image will have.

Index ConvolutionalLayer::get_output_width() const
{
    const Index input_width = get_input_width();
    const Index kernel_width = get_kernel_width();
    const Index strides = get_column_stride();

    const pair<Index, Index> padding = get_padding();

    return floor((input_width - kernel_width + 2*padding.second)/strides) + 1;
}


/// Returns the dimension of the input variables

dimensions ConvolutionalLayer::get_inputs_dimensions() const
{
    return input_dimensions;
}


/// Returns a vector containing the number of channels, rows and columns of the result of applying the layer's kernels to an image.

dimensions ConvolutionalLayer::get_output_dimensions() const
{
    Index rows_number = get_output_height();
    Index columns_number = get_output_width();
    Index kernels_number = get_kernels_number();

    return { rows_number, columns_number, kernels_number };
}


/// Returns the padding option.

ConvolutionalLayer::ConvolutionType ConvolutionalLayer::get_convolution_type() const
{
    return convolution_type;
}


/// Returns a string with the name of the convolution type.
/// This can be Valid and Same.

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


/// Returns the raw_variable stride.

Index ConvolutionalLayer::get_column_stride() const
{
    return column_stride;
}


/// Returns the row stride.

Index ConvolutionalLayer::get_row_stride() const
{
    return row_stride;
}


/// Returns the number of rows of the layer's kernels.

Index  ConvolutionalLayer::get_kernel_height() const
{
    return synaptic_weights.dimension(0);
}


/// Returns the number of columns of the layer's kernels.

Index ConvolutionalLayer::get_kernel_width() const
{
    return synaptic_weights.dimension(1);
}


/// Returns the number of channels of the layer's kernels.

Index ConvolutionalLayer::get_kernel_channels() const
{
    return synaptic_weights.dimension(2);
}


///Returns the number of kernels of the layer.

Index ConvolutionalLayer::get_kernels_number() const
{
    return synaptic_weights.dimension(3);
}


/// Returns the total number of columns of zeroes to be added to an image before applying a kernel, which depends on the padding option set.

Index ConvolutionalLayer::get_padding_width() const
{
    switch(convolution_type)
    {
    case ConvolutionType::Valid:
    {
        return 0;
    }

    case ConvolutionType::Same:
    {
        return column_stride*(input_dimensions[2] - 1) - input_dimensions[2] + get_kernel_width();
    }
    }

    return 0;
}


/// Returns the total number of rows of zeros to be added to an image before applying a kernel, which depends on the padding option set.

Index ConvolutionalLayer::get_padding_height() const
{
    switch(convolution_type)
    {
    case ConvolutionType::Valid:
    {
        return 0;
    }

    case ConvolutionType::Same:
    {
        return row_stride*(input_dimensions[1] - 1) - input_dimensions[1] + get_kernel_height();
    }
    }

    return 0;
}


/// Returns the number of inputs

Index ConvolutionalLayer::get_inputs_number() const
{
    return get_input_channels() * get_input_height() * get_input_width();
}


/// Returns the number of neurons

Index ConvolutionalLayer::get_neurons_number() const
{
    const Index kernels_number = get_kernels_number();
    const Index kernel_height = get_kernel_height();
    const Index kernel_width = get_kernel_width();

    return kernels_number * kernel_height * kernel_width;
}


/// Returns the layer's parameters in the form of a vector.

Tensor<type, 1> ConvolutionalLayer::get_parameters() const
{
    Tensor<type, 1> parameters(get_parameters_number());

    copy(/*execution::par,*/ 
         synaptic_weights.data(),
         synaptic_weights.data() + synaptic_weights.size(),
         parameters.data());

    copy(/*execution::par,*/ 
         biases.data(),
         biases.data() + biases.size(),
         parameters.data() + synaptic_weights.size());

/// @todo add scales and offsets

    return parameters;
}


/// Returns the number of parameters of the layer.

Index ConvolutionalLayer::get_parameters_number() const
{
    return synaptic_weights.size() + biases.size();
}


/// Sets and initializes the layer's parameters in accordance with the dimensions taken as input.
/// The initialization values are random values from a normal distribution.
/// @param new_input_dimensions A vector containing the desired inputs' dimensions (number of images, number of channels, rows number, columns number).
/// @param new_kernel_dimensions A vector containing the desired kernels' dimensions (number of kernels, number of channels, rows number, columns number).

void ConvolutionalLayer::set(const dimensions& new_input_dimensions,
                             const dimensions& new_kernel_dimensions)
{
    if(new_input_dimensions.size() != 3)
        throw runtime_error("Input dimensions must be 3");

    if(new_kernel_dimensions.size() != 4)
        throw runtime_error("Kernel dimensions must be 4");
    
    const Index kernel_height = new_kernel_dimensions[0];
    const Index kernel_width = new_kernel_dimensions[1];
    const Index kernel_channels = new_kernel_dimensions[2];
    const Index kernels_number = new_kernel_dimensions[3];

    biases.resize(kernels_number);
    biases.setRandom();

    synaptic_weights.resize(kernel_height,
                            kernel_width,
                            kernel_channels,
                            kernels_number);

    synaptic_weights.setRandom();

    moving_means.resize(kernels_number);
    moving_standard_deviations.resize(kernels_number);

    scales.resize(kernels_number);
    offsets.resize(kernels_number);

    input_dimensions = new_input_dimensions;
}


void ConvolutionalLayer::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}


/// Initializes the layer's biases to a given value.
/// @param value The desired value.

void ConvolutionalLayer::set_biases_constant(const type& value)
{
    biases.setConstant(value);
}


/// Initializes the layer's synaptic weights to a given value.
/// @param value The desired value.

void ConvolutionalLayer::set_synaptic_weights_constant(const type& value)
{
    synaptic_weights.setConstant(value);
}


/// Initializes the layer's parameters to a given value.
/// @param value The desired value.

void ConvolutionalLayer::set_parameters_constant(const type& value)
{
    set_biases_constant(value);

    set_synaptic_weights_constant(value);
}


/// Sets the parameters to random numbers using Eigen's setRandom.

void ConvolutionalLayer::set_parameters_random()
{
    biases.setRandom();

    synaptic_weights.setRandom();
}


/// Sets the layer's activation function.
/// @param new_activation_function The desired activation function.

void ConvolutionalLayer::set_activation_function(const ConvolutionalLayer::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


/// Sets a new activation(or transfer) function in a single layer.
/// The second argument is a string containing the name of the function("Logistic", "HyperbolicTangent", etc).
/// @param new_activation_function Activation function for that layer.

void ConvolutionalLayer::set_activation_function(const string& new_activation_function_name)
{
    if(new_activation_function_name == "Logistic")
    {
        activation_function = ActivationFunction::Logistic;
    }
    else if(new_activation_function_name == "HyperbolicTangent")
    {
        activation_function = ActivationFunction::HyperbolicTangent;
    }
    else if(new_activation_function_name == "Linear")
    {
        activation_function = ActivationFunction::Linear;
    }
    else if(new_activation_function_name == "RectifiedLinear")
    {
        activation_function = ActivationFunction::RectifiedLinear;
    }
    else if(new_activation_function_name == "ScaledExponentialLinear")
    {
        activation_function = ActivationFunction::ScaledExponentialLinear;
    }
    else if(new_activation_function_name == "SoftPlus")
    {
        activation_function = ActivationFunction::SoftPlus;
    }
    else if(new_activation_function_name == "SoftSign")
    {
        activation_function = ActivationFunction::SoftSign;
    }
    else if(new_activation_function_name == "HardSigmoid")
    {
        activation_function = ActivationFunction::HardSigmoid;
    }
    else if(new_activation_function_name == "ExponentialLinear")
    {
        activation_function = ActivationFunction::ExponentialLinear;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void set_activation_function(const string&) method.\n"
               << "Unknown activation function: " << new_activation_function_name << ".\n";

        throw runtime_error(buffer.str());
    }
}


/// Sets the layer's biases.
/// @param new_biases The desired biases.

void ConvolutionalLayer::set_biases(const Tensor<type, 1>& new_biases)
{
    biases = new_biases;
}


/// Sets the layer's synaptic weights.
/// @param new_synaptic_weights The desired synaptic weights.

void ConvolutionalLayer::set_synaptic_weights(const Tensor<type, 4>& new_synaptic_weights)
{
    synaptic_weights = new_synaptic_weights;
}


void ConvolutionalLayer::set_batch_normalization(const bool& new_batch_normalization)
{
    batch_normalization = new_batch_normalization;
}


/// Sets the padding option.
/// @param new_convolution_type The desired convolution type.

void ConvolutionalLayer::set_convolution_type(const ConvolutionalLayer::ConvolutionType& new_convolution_type)
{
    convolution_type = new_convolution_type;
}


/// Sets the padding option.
/// @param new_convolution_type The desired convolution type.

void ConvolutionalLayer::set_convolution_type(const string& new_convolution_type)
{
    if(new_convolution_type == "Valid")
    {
        convolution_type = ConvolutionType::Valid;
    }
    else if(new_convolution_type == "Same")
    {
        convolution_type = ConvolutionType::Same;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void set_convolution_type(const string&) method.\n"
               << "Unknown convolution type: " << new_convolution_type << ".\n";

        throw runtime_error(buffer.str());
    }
}

/// Sets the kernels' row stride.
/// @param new_stride_row The desired row stride.

void ConvolutionalLayer::set_row_stride(const Index& new_stride_row)
{
    if(new_stride_row <= 0)
    {
        throw ("EXCEPTION: new_stride_row must be a positive number");
    }

    row_stride = new_stride_row;
}


/// Sets the kernels' raw_variable stride.
/// @param new_stride_row The desired raw_variable stride.

void ConvolutionalLayer::set_column_stride(const Index& new_stride_column)
{
    if(new_stride_column <= 0)
    {
        throw ("EXCEPTION: new_stride_column must be a positive number");
    }

    column_stride = new_stride_column;
}


void ConvolutionalLayer::set_inputs_dimensions(const dimensions& new_input_dimensions)
{
    input_dimensions = new_input_dimensions;
}


/// Sets the synaptic weights and biases to the given values.
/// @param new_parameters A vector containing the synaptic weights and biases, in this order.

void ConvolutionalLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    const Index kernel_height = get_kernel_height();
    const Index kernel_width = get_kernel_width();
    const Index kernel_channels = get_kernel_channels();
    const Index kernels_number = get_kernels_number();

    synaptic_weights.resize(kernel_height,
                            kernel_width,
                            kernel_channels,
                            kernels_number);

    biases.resize(kernels_number);

    copy(/*execution::par,*/ 
         new_parameters.data() + index, 
         new_parameters.data() + index + synaptic_weights.size(), 
         synaptic_weights.data());

    copy(/*execution::par,*/ 
         new_parameters.data() + index + synaptic_weights.size(),
         new_parameters.data() + index + synaptic_weights.size() + biases.size(),
         biases.data());
}


/// Returns the number of biases in the layer.

Index ConvolutionalLayer::get_biases_number() const
{
    return biases.size();
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

        const Index pad_rows = max<Index>(0, (ceil((type)input_height/row_stride) - 1) * row_stride + kernel_height - input_height) / 2;

        const Index pad_columns = max<Index>(0, (ceil((type)input_width/column_stride) - 1) * column_stride + kernel_width - input_width) / 2;

        return make_pair(pad_rows, pad_columns);
    }
    default:
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "pair<Index, Index> get_padding() const method.\n"
               << "Unknown convolution type.\n";

        throw runtime_error(buffer.str());
    }
    }
}


Eigen::array<pair<Index, Index>, 4> ConvolutionalLayer::get_paddings() const
{
    Eigen::array<pair<Index, Index>, 4> paddings;

    const Index pad_rows = get_padding().first;
    const Index pad_columns = get_padding().second;

    paddings[0] = make_pair(0, 0);
    paddings[1] = make_pair(pad_rows, pad_rows);
    paddings[2] = make_pair(pad_columns, pad_columns);
    paddings[3] = make_pair(0, 0);

    return paddings;
}


Eigen::array<ptrdiff_t, 4> ConvolutionalLayer::get_strides() const
{   
    return Eigen::array<ptrdiff_t, 4>({1, row_stride, column_stride, 1});
}


/// Returns the number of synaptic weights in the layer.

Index ConvolutionalLayer::get_synaptic_weights_number() const
{
    return synaptic_weights.size();
}


/// Returns the number of rows of the input.

Index ConvolutionalLayer::get_input_height() const
{
    return input_dimensions[0];
}


/// Returns the number of columns of the input.

Index ConvolutionalLayer::get_input_width() const
{
    return input_dimensions[1];
}


/// Returns the number of channels of the input.

Index ConvolutionalLayer::get_input_channels() const
{
    return input_dimensions[2];
}

//void ConvolutionalLayer::calculate_standard_deviations(LayerForwardPropagation* layer_forward_propagation) const
//{
//    ConvolutionalLayerForwardPropagation* convolutional_layer_forward_propagation
//            = static_cast<ConvolutionalLayerForwardPropagation*>(layer_forward_propagation);

//    const Index batch_samples_number = convolutional_layer_forward_propagation->batch_samples_number;
//    const Index output_height = get_output_height();
//    const Index outputs_raw_variables_number = get_outputs_raw_variables_number();
//    const Index kernels_number = get_kernels_number();
//    const Index single_output_size = batch_samples_number * output_height * outputs_raw_variables_number;

//    Tensor<type, 1>& means = convolutional_layer_forward_propagation->means;

//    Tensor<type, 1>& variaces = convolutional_layer_forward_propagation->standard_deviations;

//    type* outputs_data = convolutional_layer_forward_propagation->outputs_data;

//    Tensor<type, 0> standard_deviation;

//    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
//    {
//        const TensorMap<Tensor<type, 1>> single_kernel_output(outputs_data + kernel_index * single_output_size,
//                                                               single_output_size);

//        standard_deviation.device(*thread_pool_device) = (single_kernel_output - means(kernel_index)).square().mean().sqrt();

//        standard_deviations(kernel_index) = standard_deviation();
//    }
//}


/*
void ConvolutionalLayer::forward(const Tensor<type, 4>& inputs, bool is_training)
{
    const Index batch_samples_number = inputs.dimension(0);
    const Index channels_number = get_kernels_number();

    if(is_training)
    {
        calculate_means(inputs);

        calculate_standard_deviations(inputs, current_means);

        normalize_and_shift(inputs, is_training);

        moving_means = moving_means * momentum + current_means * (1 - momentum);
        moving_standard_deviations = moving_standard_deviations * momentum + current_standard_deviations * (1 - momentum);
    }
    else
    {
        normalize_and_shift(inputs, is_training);
    }
}
*/

/// Serializes the convolutional layer object into an XML document of the TinyXML.
/// See the OpenNN manual for more information about the format of this document.

void ConvolutionalLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Convolutional layer

    file_stream.OpenElement("ConvolutionalLayer");

    // Layer name

    file_stream.OpenElement("LayerName");

    buffer.str("");
    buffer << layer_name;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Image size

    file_stream.OpenElement("InputsVariablesDimensions");

    buffer.str("");

    for(Index i = 0; i < Index(input_dimensions.size()); i++)
    {
        buffer << input_dimensions[i];
        if(i != input_dimensions.size() - 1) buffer << " x ";
//        if(i != input_dimensions.size() - 1) buffer << " ";
    }

    cout << "buffer (INPUT): (XML)" << buffer.str() << endl;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Outputs

    file_stream.OpenElement("OutputsVariablesDimensions");

    buffer.str("");

    for(Index i = 0; i < Index(input_dimensions.size()); i++)
    {
        buffer << get_output_dimensions()[i];
        if(i != input_dimensions.size() - 1) buffer << " x ";
    }

    cout << "buffer (OUTPUT): (XML)" << buffer.str() << endl;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Filters number

    file_stream.OpenElement("FiltersNumber");

    buffer.str("");
    buffer << get_kernels_number();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Filters size

    file_stream.OpenElement("FiltersSize");

    buffer.str("");
    buffer << get_kernel_width();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Activation function

    file_stream.OpenElement("ActivationFunction");

    file_stream.PushText(write_activation_function().c_str());

    file_stream.CloseElement();

    // Stride

    file_stream.OpenElement("Stride");

    buffer.str("");
    buffer << get_row_stride();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Convolution Type

    file_stream.OpenElement("ConvolutionType");

    file_stream.PushText(write_convolution_type().c_str());


    file_stream.CloseElement();

    cout << "filter_size(openn): " << get_kernels_number() << endl;
    cout << "filter_number (openn): " << get_kernel_width() << endl;
    cout << "activation_functin (openn): " << write_activation_function() << endl;
    cout << "stride (opennn): " << get_row_stride() << endl;
    cout << "convolution_type (opennn): " << write_convolution_type() << endl;

    // Parameters

    file_stream.OpenElement("Parameters");

    buffer.str("");
    buffer << get_parameters();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this convolutional layer object.
/// @param document TinyXML document containing the member data.

void ConvolutionalLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Convolution layer

    const tinyxml2::XMLElement* convolutional_layer_element = document.FirstChildElement("ConvolutionalLayer");

    if(!convolutional_layer_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional layer element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Convolutional layer name element

    const tinyxml2::XMLElement* convolution_name_element = convolutional_layer_element->FirstChildElement("LayerName");

    if(!convolution_name_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolution type element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string convolution_name_string = convolution_name_element->GetText();

    set_convolution_type(convolution_name_string);

//    set_convolution_type("Valid");

    // Input variables dimensions element

    const tinyxml2::XMLElement* input_variables_dimensions_element = convolutional_layer_element->FirstChildElement("InputsVariablesDimensions");

    if(!input_variables_dimensions_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional input variables dimensions element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string input_variables_dimensions_string = input_variables_dimensions_element->GetText();

    //    set_input_variables_dimenisons(Index(stoi(input_variables_dimensions_string));


    //Outputs variables dimensions element

    const tinyxml2::XMLElement* outputs_variables_dimensions_element = convolutional_layer_element->FirstChildElement("OutputsVariablesDimensions");

    if(!outputs_variables_dimensions_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional outputs variables dimensions element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string outputs_variables_dimensions_string = outputs_variables_dimensions_element->GetText();

    // Filters Number element

    const tinyxml2::XMLElement* filters_number_element = convolutional_layer_element->FirstChildElement("FiltersNumber");

    if(!filters_number_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional filters number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string filters_number_string = filters_number_element->GetText();

    //    set_input_variables_dimenisons(Index(stoi(input_variables_dimensions_string));

    // Filters Size

    const tinyxml2::XMLElement* filters_size_element = convolutional_layer_element->FirstChildElement("FiltersSize");

    if(!filters_size_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional filters size element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string filters_size_string = filters_size_element->GetText();

    //    set_column_stride(Index(stoi(filters_size_element_string)));

    // Activation Function

    const tinyxml2::XMLElement* activation_function_element = convolutional_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional activation function element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string activation_function_string = activation_function_element->GetText();

    set_activation_function(activation_function_string);

    // Stride

    const tinyxml2::XMLElement* stride_element = convolutional_layer_element->FirstChildElement("Stride");

    if(!stride_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional stride element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string stride_string = stride_element->GetText();

    set_column_stride(stoi(stride_string));
    set_row_stride(stoi(stride_string));

    // Convolution type

    const tinyxml2::XMLElement* convolution_type_element = convolutional_layer_element->FirstChildElement("ConvolutionType");

    if(!convolution_type_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional type element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string convolution_type_string = convolution_type_element->GetText();

    cout << " outputs_variables_dimensions_string:" << outputs_variables_dimensions_string << endl;
    cout << "input_variables_dimensions_string: " << input_variables_dimensions_string << endl;
    cout << "Activation Function == 0: " << activation_function_string << endl;
    cout << "Filters Number == 1: " << filters_number_string << endl;
    cout << "Filters Size == 2: " << filters_number_string << endl;
    cout << "Convolution Type == 3: " << convolution_type_string << endl;
    cout << "Stride == 4: " << stride_string << endl;

    // Parameters

    const tinyxml2::XMLElement* parameters_element = convolutional_layer_element->FirstChildElement("Parameters");

    if(!parameters_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Parameters element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(parameters_element->GetText())
    {
        const string parameters_string = parameters_element->GetText();

        set_parameters(to_type_vector(parameters_string, " "));
    }
}


ConvolutionalLayerForwardPropagation::ConvolutionalLayerForwardPropagation()
    : LayerForwardPropagation()
{
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

    return pair<type*, dimensions>(outputs_data, {batch_samples_number, output_height, output_width, kernels_number});
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

    preprocessed_inputs.resize(batch_samples_number,
                               input_height,
                               input_width,
                               input_channels);

    outputs.resize(batch_samples_number,
                   output_height,
                   output_width,
                   kernels_number);

    means.resize(kernels_number);

    standard_deviations.resize(kernels_number);

    activations_derivatives.resize(batch_samples_number,
                                   input_height,
                                   input_width,
                                   input_channels);

    outputs_data = outputs.data();
}


void ConvolutionalLayerForwardPropagation::print() const
{
    cout << "Convolutional layer" << endl;

    cout << "Outputs:" << endl;
    cout << outputs << endl;

    cout << "Activations derivatives:" << endl;
    cout << activations_derivatives << endl;
}


ConvolutionalLayerBackPropagation::ConvolutionalLayerBackPropagation() : LayerBackPropagation()
{
}


ConvolutionalLayerBackPropagation::ConvolutionalLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer);
}


ConvolutionalLayerBackPropagation::~ConvolutionalLayerBackPropagation()
{
}


void ConvolutionalLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_samples_number = new_batch_samples_number;

    layer = new_layer;

    const ConvolutionalLayer* convolutional_layer = static_cast<ConvolutionalLayer*>(layer);

    const Index input_height = convolutional_layer->get_input_height();
    const Index input_width = convolutional_layer->get_input_width();
    const Index input_channels = convolutional_layer->get_input_channels();

    const Index kernels_number = convolutional_layer->get_kernels_number();
    const Index kernel_height = convolutional_layer->get_kernel_height();
    const Index kernel_width = convolutional_layer->get_kernel_width();
    const Index kernel_channels = convolutional_layer->get_kernel_channels();

    error_convolutions_derivatives.resize(kernels_number,
                                          kernel_height,
                                          kernel_width,
                                          kernel_channels);

    biases_derivatives.resize(kernels_number);

    synaptic_weights_derivatives.resize(kernels_number,
                                        kernel_height,
                                        kernel_width,
                                        kernel_channels);

    input_derivatives.resize(batch_samples_number,
                             input_height,
                             input_width,
                             input_channels);

    inputs_derivatives.resize(1);
    inputs_derivatives(0).first = input_derivatives.data();
    inputs_derivatives(0).second = { batch_samples_number, input_height, input_width, input_channels };
}

void ConvolutionalLayerBackPropagation::print() const
{
    cout << "Biases derivatives:" << endl;
    cout << biases_derivatives << endl;

    cout << "Synaptic weights derivatives:" << endl;
    cout << synaptic_weights_derivatives << endl;
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
