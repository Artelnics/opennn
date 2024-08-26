//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>

#include "strings_utilities.h"
#include "convolutional_layer.h"
#include "tensors.h"

namespace opennn
{

ConvolutionalLayer::ConvolutionalLayer() : Layer()
{
    layer_type = Layer::Type::Convolutional;
    name = "convolutional_layer";
}


ConvolutionalLayer::ConvolutionalLayer(const dimensions& new_input_dimensions,
                                       const dimensions& new_kernel_dimensions) : Layer()
{
    layer_type = Layer::Type::Convolutional;
    name = "convolutional_layer";

    set(new_input_dimensions, new_kernel_dimensions);
}


bool ConvolutionalLayer::is_empty() const
{
    if(biases.size() == 0 && synaptic_weights.size() == 0)
        return true;

    return false;
}


const Tensor<type, 1>& ConvolutionalLayer::get_biases() const
{
    return biases;
}


const Tensor<type, 4>& ConvolutionalLayer::get_synaptic_weights() const
{
    return synaptic_weights;
}


bool ConvolutionalLayer::get_batch_normalization() const
{
    return batch_normalization;
}


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
            = inputs.convolve(kernel, convolutions_dimensions);
            //+ biases(kernel_index);
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
    /*
    cout << "batch_samples_number: " << batch_samples_number << endl;

    cout << "input_height: " << input_height << endl;
    cout << "input_width: " << input_width << endl;
    cout << "input_channels: " << input_channels << endl;

    cout << "kernels_number: " << kernels_number << endl;
    cout << "kernel_height: " << kernel_height << endl;
    cout << "kernel_width: " << kernel_width << endl;
    cout << "kernel_channels: " << kernel_channels << endl;

    cout << "output_height: " << output_height << endl;
    cout << "output_width: " << output_width << endl;

    cout << "inputs_pair(0).second[0]: " << inputs_pair(0).second[0] << endl;
    cout << "inputs_pair(0).second[1]: " << inputs_pair(0).second[1] << endl;
    cout << "inputs_pair(0).second[2]: " << inputs_pair(0).second[2] << endl;
    cout << "inputs_pair(0).second[3]: " << inputs_pair(0).second[3] << endl;

    cout << "deltas_pair(0).second[0]: " << deltas_pair(0).second[0] << endl;
    cout << "deltas_pair(0).second[1]: " << deltas_pair(0).second[1] << endl;
    cout << "deltas_pair(0).second[2]: " << deltas_pair(0).second[2] << endl;
    cout << "deltas_pair(0).second[3]: " << deltas_pair(0).second[3] << endl;
    */

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

    type* synaptic_weights_derivatives_data = convolutional_layer_back_propagation->synaptic_weights_derivatives.data();

    Tensor<type, 4>& input_derivatives = convolutional_layer_back_propagation->input_derivatives;

    Eigen::array<Index, 4> offsets;
    Eigen::array<Index, 4> extents;

    const Eigen::array<ptrdiff_t, 3> convolutions_dimensions = {0, 1, 2};

    Tensor<type, 4> delta_slice;
    Tensor<type, 4> image_slice;

    const Index kernel_synaptic_weights_number = kernel_channels*kernel_height*kernel_width;

    Tensor<type, 0> current_sum;

    error_convolutions_derivatives.device(*thread_pool_device) = deltas * activations_derivatives;

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
            offsets = {image_index, 0, 0, kernel_index};

            extents = {1, output_height, output_width, 1};

            delta_slice = error_convolutions_derivatives.slice(offsets, extents); // device(*thread_pool_device) does not work here

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
        }
        
        copy(kernel_synaptic_weights_derivatives.data(),
             kernel_synaptic_weights_derivatives.data() + kernel_synaptic_weights_number,
             synaptic_weights_derivatives_data + kernel_synaptic_weights_number * kernel_index);
    }

    // Inputs derivatives

    input_derivatives.setZero();

    extents = { 1, output_height, output_width, 1 };

    for (Index image_index = 0; image_index < batch_samples_number; image_index++)
    {
        for (Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
        {
            offsets = { image_index, 0, 0, kernel_index };

            delta_slice = error_convolutions_derivatives.slice(offsets, extents);
            
            const TensorMap<Tensor<type, 3>> delta_reshape(delta_slice.data(),
                                                           output_height,
                                                           output_width,
                                                           1);

            TensorMap<Tensor<type, 3>> kernel_weights(synaptic_weights_derivatives_data + kernel_index * kernel_synaptic_weights_number,
                                                      kernel_height,
                                                      kernel_width,
                                                      kernel_channels);

            input_derivatives.chip(image_index, 0).device(*thread_pool_device) +=
                delta_reshape.convolve(kernel_weights.reverse(Eigen::array<Index, 3>({ 0, 1, 2 })), convolutions_dimensions);
        }
    }

    /* OLD INPUT DERIVATIVES
    input_derivatives = error_convolutions_derivatives.convolve(synaptic_weights,convolutions_dimensions);
    
    for(int image_index = 0; image_index < batch_samples_number; image_index++)
    {
        for(int row = 0; row < deltas_pair(0).second[1]; ++row)
        {
            for(int column = 0; column < deltas_pair(0).second[2]; ++column)
            {
                for(int channel = 0; channel < deltas_pair(0).second[3]; ++channel)
                {
                    input_derivatives.chip(image_index, 0)
                        .chip(row, 0)
                        .chip(column, 0)
                        .slice(std::array<Index, 2>({ row, column }),
                               std::array<Index, 2>({ kernel_height, kernel_width })) +=
                        error_convolutions_derivatives(image_index, row, column, channel) *
                        synaptic_weights.chip(channel, 0);

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

    const Index synaptic_weights_number = get_synaptic_weights_number();
    const Index biases_number = get_biases_number();

    // Back-propagation

    ConvolutionalLayerBackPropagation* convolutional_layer_back_propagation =
            static_cast<ConvolutionalLayerBackPropagation*>(back_propagation);

    const type* synaptic_weights_derivatives_data = convolutional_layer_back_propagation->synaptic_weights_derivatives.data();

    const type* biases_derivatives_data = convolutional_layer_back_propagation->biases_derivatives.data();

    // Copy from back propagation to gradient

    copy(synaptic_weights_derivatives_data,
         synaptic_weights_derivatives_data + synaptic_weights_number,
         gradient_data + index);

    copy(biases_derivatives_data,
         biases_derivatives_data + biases_number,
         gradient_data + index + synaptic_weights_number);
}


ConvolutionalLayer::ActivationFunction ConvolutionalLayer::get_activation_function() const
{
    return activation_function;
}


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
    Index rows_number = get_output_height();
    Index columns_number = get_output_width();
    Index kernels_number = get_kernels_number();

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


Index ConvolutionalLayer::get_inputs_number() const
{
    return get_input_channels() * get_input_height() * get_input_width();
}


Index ConvolutionalLayer::get_neurons_number() const
{
    const Index kernels_number = get_kernels_number();
    const Index kernel_height = get_kernel_height();
    const Index kernel_width = get_kernel_width();

    return kernels_number * kernel_height * kernel_width;
}


Tensor<type, 1> ConvolutionalLayer::get_parameters() const
{
    Tensor<type, 1> parameters(get_parameters_number());

    copy(synaptic_weights.data(),
         synaptic_weights.data() + synaptic_weights.size(),
         parameters.data());

    copy(biases.data(),
         biases.data() + biases.size(),
         parameters.data() + synaptic_weights.size());

// @todo add scales and offsets

    return parameters;
}


Index ConvolutionalLayer::get_parameters_number() const
{
    return synaptic_weights.size() + biases.size();
}


void ConvolutionalLayer::set(const dimensions& new_input_dimensions,
                             const dimensions& new_kernel_dimensions)
{
    if(new_input_dimensions.size() != 3)
        throw runtime_error("Input dimensions must be 3");

    if(new_kernel_dimensions.size() != 4)
        throw runtime_error("Kernel dimensions must be 4");

    if (new_kernel_dimensions.size() != 4)
        throw runtime_error("Kernel dimensions must be 4");
    
    const Index kernel_height = new_kernel_dimensions[0];
    const Index kernel_width = new_kernel_dimensions[1];
    const Index kernel_channels = new_kernel_dimensions[2];
    const Index kernels_number = new_kernel_dimensions[3];

    if (kernel_height > new_input_dimensions[0])
        throw runtime_error("kernel_height cannot be bigger than input dimensions");

    if (kernel_width > new_input_dimensions[1])
        throw runtime_error("kernel_width cannot be bigger than input dimensions");

    if (kernel_channels != new_input_dimensions[2])
        throw runtime_error("kernel_channels must match input_channels dimension");

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

    input_dimensions = new_input_dimensions;
}


void ConvolutionalLayer::set_name(const string& new_layer_name)
{
    name = new_layer_name;
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
        throw runtime_error("Unknown activation function: " + new_activation_function_name + ".\n");
    }
}


void ConvolutionalLayer::set_biases(const Tensor<type, 1>& new_biases)
{
    biases = new_biases;
}


void ConvolutionalLayer::set_synaptic_weights(const Tensor<type, 4>& new_synaptic_weights)
{
    synaptic_weights = new_synaptic_weights;
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
    {
        convolution_type = ConvolutionType::Valid;
    }
    else if(new_convolution_type == "Same")
    {
        convolution_type = ConvolutionType::Same;
    }
    else
    {
        throw runtime_error("Unknown convolution type: " + new_convolution_type + ".\n");
    }
}


void ConvolutionalLayer::set_row_stride(const Index& new_stride_row)
{
    if(new_stride_row <= 0)
    {
        throw runtime_error("EXCEPTION: new_stride_row must be a positive number");
    }

    row_stride = new_stride_row;
}


void ConvolutionalLayer::set_column_stride(const Index& new_stride_column)
{
    if(new_stride_column <= 0)
    {
        throw runtime_error("EXCEPTION: new_stride_column must be a positive number");
    }

    column_stride = new_stride_column;
}


void ConvolutionalLayer::set_inputs_dimensions(const dimensions& new_input_dimensions)
{
    input_dimensions = new_input_dimensions;
}


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

    copy(new_parameters.data() + index,
         new_parameters.data() + index + synaptic_weights.size(), 
         synaptic_weights.data());

    copy(new_parameters.data() + index + synaptic_weights.size(),
         new_parameters.data() + index + synaptic_weights.size() + biases.size(),
         biases.data());
}


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
        throw runtime_error("Unknown convolution type.\n");
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


Index ConvolutionalLayer::get_synaptic_weights_number() const
{
    return synaptic_weights.size();
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
    const Index channels = get_kernels_number();

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

void ConvolutionalLayer::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Convolutional layer

    file_stream.OpenElement("ConvolutionalLayer");

    // Layer name

    file_stream.OpenElement("LayerName");
    file_stream.PushText(name.c_str());
    file_stream.CloseElement();

    // Image size

    file_stream.OpenElement("InputDimensions");
    file_stream.PushText(dimensions_to_string(input_dimensions).c_str());
    file_stream.CloseElement();

    // Outputs

    file_stream.OpenElement("OutputDimensions");
    file_stream.PushText(dimensions_to_string(get_output_dimensions()).c_str());

    file_stream.CloseElement();

    // Filters number

    file_stream.OpenElement("FiltersNumber");
    file_stream.PushText(to_string(get_kernels_number()).c_str());
    file_stream.CloseElement();

    // Filters size

    file_stream.OpenElement("FiltersSize");
    file_stream.PushText(to_string(get_kernel_width()).c_str());
    file_stream.CloseElement();

    // Activation function

    file_stream.OpenElement("ActivationFunction");
    file_stream.PushText(write_activation_function().c_str());
    file_stream.CloseElement();

    // Stride

    file_stream.OpenElement("Stride");
    file_stream.PushText(to_string(get_row_stride()).c_str());
    file_stream.CloseElement();

    // Convolution Type

    file_stream.OpenElement("ConvolutionType");
    file_stream.PushText(write_convolution_type().c_str());
    file_stream.CloseElement();

    // Parameters

    file_stream.OpenElement("Parameters");
    file_stream.PushText(tensor_to_string(get_parameters()).c_str());
    file_stream.CloseElement();

    file_stream.CloseElement();
}


void ConvolutionalLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    // Convolution layer

    const tinyxml2::XMLElement* convolutional_layer_element = document.FirstChildElement("ConvolutionalLayer");

    if(!convolutional_layer_element)
        throw runtime_error("Convolutional layer element is nullptr.\n");

    // Convolutional layer name element

    const tinyxml2::XMLElement* convolution_name_element = convolutional_layer_element->FirstChildElement("LayerName");

    if(!convolution_name_element)
        throw runtime_error("Convolution type element is nullptr.\n");

    const string convolution_name_string = convolution_name_element->GetText();

    set_convolution_type(convolution_name_string);

//    set_convolution_type("Valid");

    // Input variables dimensions element

    const tinyxml2::XMLElement* input_variables_dimensions_element = convolutional_layer_element->FirstChildElement("InputDimensions");

    if(!input_variables_dimensions_element)
        throw runtime_error("Convolutional input variables dimensions element is nullptr.\n");

    const string input_variables_dimensions_string = input_variables_dimensions_element->GetText();

    //    set_input_variables_dimenisons(Index(stoi(input_variables_dimensions_string));

    // Outputs variables dimensions element

    const tinyxml2::XMLElement* outputs_variables_dimensions_element = convolutional_layer_element->FirstChildElement("OutputDimensions");

    if(!outputs_variables_dimensions_element)
        throw runtime_error("Convolutional outputs variables dimensions element is nullptr.\n");

    const string outputs_variables_dimensions_string = outputs_variables_dimensions_element->GetText();

    // Filters Number element

    const tinyxml2::XMLElement* filters_number_element = convolutional_layer_element->FirstChildElement("FiltersNumber");

    if(!filters_number_element)
        throw runtime_error("Convolutional filters number element is nullptr.\n");

    const string filters_number_string = filters_number_element->GetText();

    //    set_input_variables_dimenisons(Index(stoi(input_variables_dimensions_string));

    // Filters Size

    const tinyxml2::XMLElement* filters_size_element = convolutional_layer_element->FirstChildElement("FiltersSize");

    if(!filters_size_element)
        throw runtime_error("Convolutional filters size element is nullptr.\n");

    const string filters_size_string = filters_size_element->GetText();

    //    set_column_stride(Index(stoi(filters_size_element_string)));

    // Activation Function

    const tinyxml2::XMLElement* activation_function_element = convolutional_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
        throw runtime_error("Convolutional activation function element is nullptr.\n");

    const string activation_function_string = activation_function_element->GetText();

    set_activation_function(activation_function_string);

    // Stride

    const tinyxml2::XMLElement* stride_element = convolutional_layer_element->FirstChildElement("Stride");

    if(!stride_element)
        throw runtime_error("Convolutional stride element is nullptr.\n");

    const string stride_string = stride_element->GetText();

    set_column_stride(stoi(stride_string));
    set_row_stride(stoi(stride_string));

    // Convolution type

    const tinyxml2::XMLElement* convolution_type_element = convolutional_layer_element->FirstChildElement("ConvolutionType");

    if(!convolution_type_element)
        throw runtime_error("Convolutional type element is nullptr.\n");

    const string convolution_type_string = convolution_type_element->GetText();

    // Parameters

    const tinyxml2::XMLElement* parameters_element = convolutional_layer_element->FirstChildElement("Parameters");

    if(!parameters_element)
        throw runtime_error("Parameters element is nullptr.\n");

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

    const Index kernel_height = convolutional_layer->get_kernel_height();
    const Index kernel_width = convolutional_layer->get_kernel_width();
    const Index kernel_channels = convolutional_layer->get_kernel_channels();
    const Index kernels_number = convolutional_layer->get_kernels_number();

    error_convolutions_derivatives.resize(batch_samples_number,
                                          input_height,
                                          input_width,
                                          input_channels);

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
