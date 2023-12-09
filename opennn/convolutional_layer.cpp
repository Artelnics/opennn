//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "convolutional_layer.h"
#include "pooling_layer.h"
#include "perceptron_layer.h"
#include "probabilistic_layer.h"

#include "numerical_differentiation.h"

namespace opennn
{

/// Default constructor.
/// It creates an empty ConvolutionalLayer object.

ConvolutionalLayer::ConvolutionalLayer() : Layer()
{
    layer_type = Layer::Type::Convolutional;
}

ConvolutionalLayer::ConvolutionalLayer(const Index& new_inputs_number,
                                       const Index& new_outputs_number,
                                       const ConvolutionalLayer::ActivationFunction& new_activation_function) : Layer()
{
    layer_type = Layer::Type::Convolutional;
}


/// Inputs' dimensions modifier constructor.
/// After setting new dimensions for the inputs, it creates and initializes a ConvolutionalLayer object
/// with a number of kernels of a given size.
/// The initialization values are random values from a normal distribution.
/// @param new_inputs_dimensions A vector containing the new inputs' dimensions.
/// @param kernels_dimensions A vector containing the kernel rows, columns channels and number.

ConvolutionalLayer::ConvolutionalLayer(const Tensor<Index, 1>& new_inputs_dimensions,
                                       const Tensor<Index, 1>& new_kernels_dimensions) : Layer()
{
    layer_type = Layer::Type::Convolutional;

    set(new_inputs_dimensions, new_kernels_dimensions);
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


/// Inserts padding to the input tensor.
/// @param input Tensor containing the inputs.
/// @param padded_output input tensor padded.

void ConvolutionalLayer::insert_padding(const Tensor<type, 4>& inputs, Tensor<type, 4>& padded_output)
{
    switch(convolution_type)
    {
    case ConvolutionType::Valid: padded_output = inputs; return;

    case ConvolutionType::Same:
    {
        const Index input_rows_number = inputs.dimension(0);
        const Index input_cols_number = inputs.dimension(1);

        const Index kernel_rows_number = get_kernels_rows_number();
        const Index kernel_cols_number = get_kernels_columns_number();

        Eigen::array<pair<int, int>, 4> paddings;

        const int pad_rows = int(0.5 *(input_rows_number*(row_stride - 1) - row_stride + kernel_rows_number));
        const int pad_cols = int(0.5 *(input_cols_number*(column_stride - 1) - column_stride + kernel_cols_number));

        paddings[0] = make_pair(pad_rows, pad_rows);
        paddings[1] = make_pair(pad_cols, pad_cols);
        paddings[2] = make_pair(0, 0);
        paddings[3] = make_pair(0, 0);

        padded_output = inputs.pad(paddings);
        return;
    }
    }
}


/// Calculate convolutions

void ConvolutionalLayer::calculate_convolutions(const Tensor<type, 4>& inputs,
                                                type* combinations) const
{
    const Index inputs_rows_number = inputs.dimension(0);
    const Index inputs_columns_number = inputs.dimension(1);
    const Index inputs_channels_number = inputs.dimension(2);
    const Index images_number = inputs.dimension(3);

    const Index kernels_rows_number = synaptic_weights.dimension(0);
    const Index kernels_columns_number = synaptic_weights.dimension(1);
    const Index kernels_channels_number = synaptic_weights.dimension(2);
    const Index kernels_number = synaptic_weights.dimension(3);

#ifdef OPENNN_DEBUG
/*
    const Index output_channel_number = combinations.dimension(2);
    const Index output_images_number = combinations.dimension(3);

    if(output_channel_number != kernels_number)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "ConvolutionalLayer::calculate_convolutions.\n"
               << "output_channel_number" <<output_channel_number <<"must me equal to" << kernels_number<<".\n";

        throw invalid_argument(buffer.str());
    }

    if(output_images_number != images_number)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "ConvolutionalLayer::calculate_convolutions.\n"
               << "output_images_number" <<output_images_number <<"must me equal to" << images_number<<".\n";

        throw invalid_argument(buffer.str());
    }
*/
#endif

    const Index next_image = inputs_rows_number*inputs_columns_number*kernels_channels_number;
    const Index next_kernel = kernels_rows_number*kernels_columns_number*kernels_channels_number;

    const Index output_size_rows_columns = get_outputs_rows_number() * get_outputs_columns_number();

    const Eigen::array<ptrdiff_t, 3> dims = {0, 1, 2};

    const Tensor<type, 4>& inputs_pointer = inputs;
    const Tensor<type, 4>& synaptic_weights_pointer = synaptic_weights; // ??

    const Eigen::array<Index, 3> strides{get_row_stride(), get_column_stride(), 1};

#pragma omp parallel for
    for(int i = 0; i < images_number ;i++)
    {
        const TensorMap<const Tensor<type, 3>> single_image(inputs_pointer.data()+i*next_image,
                                                      inputs_rows_number,
                                                      inputs_columns_number,
                                                      inputs_channels_number);

        for(int j = 0; j<kernels_number; j++)
        {
            const TensorMap<const Tensor<type, 3>> single_kernel(synaptic_weights_pointer.data()+j*next_kernel,
                                                           kernels_rows_number,
                                                           kernels_columns_number,
                                                           kernels_channels_number);

            Tensor<type, 3> tmp_result = single_image.convolve(single_kernel, dims).stride(strides) + biases(j);

            memcpy(combinations +j*output_size_rows_columns +i*output_size_rows_columns*kernels_number,
                   tmp_result.data(), static_cast<size_t>(output_size_rows_columns)*sizeof(float));
        }
    }
}


void ConvolutionalLayer::calculate_convolutions(const Tensor<type, 4>& inputs,
                                                const Tensor<type, 2>& potential_biases,
                                                const Tensor<type, 4>& potential_synaptic_weights,
                                                Tensor<type, 4>& convolutions) const // old version
{
    const Index inputs_rows_number = inputs.dimension(0);
    const Index inputs_columns_number = inputs.dimension(1);
    const Index inputs_channels_number = inputs.dimension(2);
    const Index images_number = inputs.dimension(3);

    const Index kernels_rows_number = potential_synaptic_weights.dimension(0);
    const Index kernels_columns_number = potential_synaptic_weights.dimension(1);
    const Index kernels_channels_number = potential_synaptic_weights.dimension(2);
    const Index kernels_number = potential_synaptic_weights.dimension(3);

    const Index next_image = inputs_rows_number*inputs_columns_number*inputs_channels_number;
    const Index next_kernel = kernels_rows_number*kernels_columns_number*kernels_channels_number;

    const Index output_size_rows_columns = ((inputs_rows_number-kernels_rows_number)+1)*((inputs_columns_number-kernels_columns_number)+1);

    const Eigen::array<ptrdiff_t, 3> dims = {0, 1, 2};

    Tensor<type, 4> inputs_pointer = inputs;
    Tensor<type, 4> synaptic_weights_pointer = potential_synaptic_weights; // ??

#pragma omp parallel for
    for(int i = 0; i < images_number ;i++)
    {
        const TensorMap<Tensor<type, 3>> single_image(inputs_pointer.data()+i*next_image,
                                                      inputs_rows_number,
                                                      inputs_columns_number,
                                                      inputs_channels_number);
        for(int j = 0; j<kernels_number; j++)
        {
            const TensorMap<Tensor<type, 3>> single_kernel(synaptic_weights_pointer.data()+j*next_kernel,
                                                           kernels_rows_number,
                                                           kernels_columns_number,
                                                           kernels_channels_number);

            Tensor<type, 3> tmp_result = single_image.convolve(single_kernel, dims) + potential_biases(j,0);

            memcpy(convolutions.data() +j*output_size_rows_columns +i*output_size_rows_columns*kernels_number,
                   tmp_result.data(), static_cast<size_t>(output_size_rows_columns)*sizeof(float));
        }
    }
}


/// Calculates activations

void ConvolutionalLayer::calculate_activations(type* combinations, const Tensor<Index, 1>& combinations_dimensions,
                                               type* activations, const Tensor<Index, 1>& activations_dimensions) const
{
    //@todo debug checks

    switch(activation_function)
    {
    case ActivationFunction::Linear: linear(combinations, combinations_dimensions, activations, activations_dimensions); return;

    case ActivationFunction::Logistic: logistic(combinations, combinations_dimensions, activations, activations_dimensions); return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(combinations, combinations_dimensions, activations, activations_dimensions); return;

    case ActivationFunction::Threshold: threshold(combinations, combinations_dimensions, activations, activations_dimensions); return;

    case ActivationFunction::SymmetricThreshold: symmetric_threshold(combinations, combinations_dimensions, activations, activations_dimensions); return;

    case ActivationFunction::RectifiedLinear: rectified_linear(combinations, combinations_dimensions, activations, activations_dimensions); return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(combinations, combinations_dimensions, activations, activations_dimensions); return;

    case ActivationFunction::SoftPlus: soft_plus(combinations, combinations_dimensions, activations, activations_dimensions); return;

    case ActivationFunction::SoftSign: soft_sign(combinations, combinations_dimensions, activations, activations_dimensions); return;

    case ActivationFunction::HardSigmoid: hard_sigmoid(combinations, combinations_dimensions, activations, activations_dimensions); return;

    case ActivationFunction::ExponentialLinear: exponential_linear(combinations, combinations_dimensions, activations, activations_dimensions); return;

    default: return;
    }

}



/// Calculates activations derivatives

void ConvolutionalLayer::calculate_activations_derivatives(type* combinations, const Tensor<Index, 1>& combinations_dimensions,
                                                           type* activations, const Tensor<Index, 1>& activations_dimensions,
                                                           type* activations_derivatives, const Tensor<Index, 1>& activations_derivatives_dimensions) const
{
    // @todo debug checks

    switch(activation_function)
    {
    case ActivationFunction::Linear: linear_derivatives(combinations, combinations_dimensions, activations, activations_dimensions, activations_derivatives, activations_derivatives_dimensions); return;

    case ActivationFunction::Logistic: logistic_derivatives(combinations, combinations_dimensions, activations, activations_dimensions, activations_derivatives, activations_derivatives_dimensions); return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent_derivatives(combinations, combinations_dimensions, activations, activations_dimensions, activations_derivatives, activations_derivatives_dimensions); return;

    case ActivationFunction::Threshold: threshold_derivatives(combinations, combinations_dimensions, activations, activations_dimensions, activations_derivatives, activations_derivatives_dimensions); return;

    case ActivationFunction::SymmetricThreshold: symmetric_threshold_derivatives(combinations, combinations_dimensions, activations, activations_dimensions, activations_derivatives, activations_derivatives_dimensions); return;

    case ActivationFunction::RectifiedLinear: rectified_linear_derivatives(combinations, combinations_dimensions, activations, activations_dimensions, activations_derivatives, activations_derivatives_dimensions); return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear_derivatives(combinations, combinations_dimensions, activations, activations_dimensions, activations_derivatives, activations_derivatives_dimensions); return;

    case ActivationFunction::SoftPlus: soft_plus_derivatives(combinations, combinations_dimensions, activations, activations_dimensions, activations_derivatives, activations_derivatives_dimensions); return;

    case ActivationFunction::SoftSign: soft_sign_derivatives(combinations, combinations_dimensions, activations, activations_dimensions, activations_derivatives, activations_derivatives_dimensions); return;

    case ActivationFunction::HardSigmoid: hard_sigmoid_derivatives(combinations, combinations_dimensions, activations, activations_dimensions, activations_derivatives, activations_derivatives_dimensions); return;

    case ActivationFunction::ExponentialLinear: exponential_linear_derivatives(combinations, combinations_dimensions, activations, activations_dimensions, activations_derivatives, activations_derivatives_dimensions); return;

    default: return;
    }
}


void ConvolutionalLayer::forward_propagate(type* inputs_data,
                                           const Tensor<Index,1>& inputs_dimensions,
                                           LayerForwardPropagation* forward_propagation,
                                           bool& switch_train)
{
    ConvolutionalLayerForwardPropagation* convolutional_layer_forward_propagation
            = static_cast<ConvolutionalLayerForwardPropagation*>(forward_propagation);

    const TensorMap<Tensor<type, 4>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1), inputs_dimensions(2), inputs_dimensions(3));
    type* combinations_data = convolutional_layer_forward_propagation->get_combinations_data();

    calculate_convolutions(inputs,
                           combinations_data);

    const Tensor<Index, 1> combinations_dimensions = get_dimensions(convolutional_layer_forward_propagation->combinations);
    const Tensor<Index, 1> outputs_dimensions = convolutional_layer_forward_propagation->outputs_dimensions;

    if(switch_train) // Perform training
    {
        const Tensor<Index, 1> activations_derivatives_dimensions = get_dimensions(convolutional_layer_forward_propagation->activations_derivatives);

        calculate_activations_derivatives(combinations_data, combinations_dimensions,
                                          convolutional_layer_forward_propagation->outputs_data, outputs_dimensions,
                                          convolutional_layer_forward_propagation->get_activations_derivatives_data(), activations_derivatives_dimensions);
    }
    else // Perform deployment
    {
        calculate_activations(combinations_data, combinations_dimensions,
                              convolutional_layer_forward_propagation->outputs_data, outputs_dimensions);
    }
}

void ConvolutionalLayer::calculate_hidden_delta(ConvolutionalLayerForwardPropagation* next_layer_forward_propagation,
                                                ConvolutionalLayerBackPropagation* next_layer_back_propagation,
                                                LayerBackPropagation* layer_back_propagation) const
{

    const ConvolutionalLayer* next_layer = static_cast<const ConvolutionalLayer*>(next_layer_forward_propagation->layer_pointer);

    const Index kernel_numbers = next_layer->get_kernels_number();
    
    const Index next_delta_rows_number = next_layer_back_propagation->deltas_dimensions(0);
    const Index next_delta_cols_number = next_layer_back_propagation->deltas_dimensions(1);
    const Index next_delta_channels_number = next_layer_back_propagation->deltas_dimensions(2);

    const Index images_number = next_layer_back_propagation->deltas_dimensions(3);

    TensorMap<Tensor<type, 4>> next_delta(
        next_layer_back_propagation->deltas_data, 
        next_delta_rows_number,
        next_delta_cols_number,
        next_delta_channels_number,
        images_number);


    const Index current_delta_rows_number = layer_back_propagation->deltas_dimensions(0);
    const Index current_delta_cols_number = layer_back_propagation->deltas_dimensions(1);
    const Index current_delta_channels_number = layer_back_propagation->deltas_dimensions(2);

    TensorMap<Tensor<type, 4>> current_delta(
        layer_back_propagation->deltas_data, 
        current_delta_rows_number,
        current_delta_cols_number,
        current_delta_channels_number,
        images_number);


    const Tensor<type, 4>& next_synaptic_weights = next_layer->get_synaptic_weights();

    auto flip_tensor_horizontal_and_vertical = [](const Tensor<type, 4>& tensor){
        const Eigen::array<bool, 4> dimensions_to_flip({true, true, false, false});
        return tensor.reverse(dimensions_to_flip);
    };

    Tensor<type, 4> next_synaptic_weights_flipped = flip_tensor_horizontal_and_vertical(next_synaptic_weights);

    Tensor<type, 4> next_delta_times_activation_delta(
        next_delta_rows_number,
        next_delta_cols_number,
        next_delta_channels_number,
        images_number); 
    
    next_delta_times_activation_delta.device(*thread_pool_device) = (next_delta * next_layer_forward_propagation->activations_derivatives);

    const Index next_kernel_rows_number = next_layer->get_kernels_rows_number();
    const Index next_kernel_cols_number = next_layer->get_kernels_columns_number();

    const Index rows_padding = (current_delta_rows_number - next_delta_rows_number + next_kernel_rows_number - 1) / 2;
    const Index cols_padding = (current_delta_cols_number - next_delta_cols_number + next_kernel_cols_number - 1) / 2;

    const Index next_delta_with_zeros_padded_rows_number = current_delta_rows_number + next_kernel_rows_number - 1;
    const Index next_delta_with_zeros_padded_cols_number = current_delta_cols_number + next_kernel_cols_number - 1;

    
    Tensor<type, 3> next_delta_with_zeros_padded(
        next_delta_with_zeros_padded_rows_number,
        next_delta_with_zeros_padded_cols_number,
        next_delta_channels_number);
    next_delta_with_zeros_padded.setZero();
    
    const Eigen::array<Index, 3> offsets{rows_padding, cols_padding, 0};
    const Eigen::array<Index, 3> extends{
        next_delta_with_zeros_padded_rows_number - 2 * rows_padding, 
        next_delta_with_zeros_padded_cols_number - 2 * cols_padding, 
        next_delta_channels_number};

    const Index row_stride = next_layer->get_row_stride();
    const Index column_stride = next_layer->get_column_stride();
    const Eigen::array<Index, 3> strides{row_stride, column_stride, 1};

    const Eigen::array<Index, 2> convolution_dimensions = {0, 1};
    
    current_delta.setZero();


    #pragma omp parallel for
    for(Index image_idx = 0; image_idx < images_number; image_idx++)
    {
        const auto& image = next_delta_times_activation_delta.chip(image_idx, 3);
        next_delta_with_zeros_padded.slice(offsets, extends).stride(strides) = image;

        for(Index channel_idx = 0; channel_idx < current_delta_channels_number; channel_idx++)
        {
            for(Index kernel_idx = 0; kernel_idx < kernel_numbers; kernel_idx++)
            {
                current_delta.chip(image_idx, 3).chip(channel_idx, 2) += next_delta_with_zeros_padded.chip(kernel_idx, 2).convolve(
                    next_synaptic_weights_flipped.chip(kernel_idx, 3).chip(channel_idx, 2),
                    convolution_dimensions);
            }
        }
    }

}


void ConvolutionalLayer::calculate_hidden_delta(LayerForwardPropagation* next_layer_forward_propagation,
                                                LayerBackPropagation* next_layer_back_propagation,
                                                LayerBackPropagation* layer_back_propagation) const
{
    switch(next_layer_back_propagation->layer_pointer->get_type())
    {
    case Type::Perceptron:
    {
        PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation =
                static_cast<PerceptronLayerForwardPropagation*>(next_layer_forward_propagation);

        PerceptronLayerBackPropagation* perceptron_layer_back_propagation =
                static_cast<PerceptronLayerBackPropagation*>(next_layer_back_propagation);

        calculate_hidden_delta(perceptron_layer_forward_propagation,
                               perceptron_layer_back_propagation,
                               layer_back_propagation);
    }
        break;
    case Type::Convolutional:
    {
        ConvolutionalLayerForwardPropagation* next_layer_convolutional_forward_propagation =
                static_cast<ConvolutionalLayerForwardPropagation*>(next_layer_forward_propagation);

        ConvolutionalLayerBackPropagation* next_layer_convolutional_back_propagation =
                static_cast<ConvolutionalLayerBackPropagation*>(next_layer_back_propagation);

        calculate_hidden_delta(next_layer_convolutional_forward_propagation,
                               next_layer_convolutional_back_propagation,
                               layer_back_propagation);
    }
        break;
    default:
    {
        //Forwarding
        next_layer_back_propagation->layer_pointer->calculate_hidden_delta(
            next_layer_forward_propagation,
            next_layer_back_propagation,
            layer_back_propagation);
        //cout << "Neural network structure not implemented: " << next_layer_back_propagation->layer_pointer->get_type_string() << endl;
        return;
    }
    }
}


void ConvolutionalLayer::calculate_hidden_delta(PerceptronLayerForwardPropagation* next_forward_propagation,
                                                PerceptronLayerBackPropagation* next_back_propagation,
                                                ConvolutionalLayerBackPropagation* back_propagation) const
{
    const Tensor<type, 2>& next_synaptic_weights = static_cast<PerceptronLayer*>(next_back_propagation->layer_pointer)->get_synaptic_weights();

    const TensorMap<Tensor<type, 2>> next_deltas(next_back_propagation->deltas_data, next_back_propagation->deltas_dimensions(0), next_back_propagation->deltas_dimensions(1));

    TensorMap<Tensor<type, 2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));

    Tensor<type, 2> output_deltas(deltas);

    if(has_NAN(output_deltas))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: perceptron layer class.\n"
               << "void calculate_hidden_delta(const DataSetBatch&, NeuralNetworkForwardPropagation&,LossIndexBackPropagation&) method.\n"
               << "NAN values found in deltas.";

        throw invalid_argument(buffer.str());
    }

    deltas.device(*thread_pool_device) = (next_deltas*next_forward_propagation->activations_derivatives).contract(next_synaptic_weights, A_BT);
}


void ConvolutionalLayer::calculate_error_gradient(type* input_data,
                                                  LayerForwardPropagation* forward_propagation,
                                                  LayerBackPropagation* back_propagation) const
{
    const Index inputs_rows_number = get_inputs_rows_number();
    const Index inputs_columns_number = get_inputs_columns_number();
    const Index batch_samples_number = back_propagation->batch_samples_number;

    const Index channels_number = get_inputs_channels_number();

    const Index kernels_rows_number = get_kernels_rows_number();
    const Index kernels_columns_number = get_kernels_columns_number();
    const Index kernels_number = get_kernels_number();

    const ConvolutionalLayerForwardPropagation* convolutional_layer_forward_propagation =
            static_cast<ConvolutionalLayerForwardPropagation*>(forward_propagation);

    ConvolutionalLayerBackPropagation* convolutional_layer_back_propagation =
            static_cast<ConvolutionalLayerBackPropagation*>(back_propagation);

    TensorMap<Tensor<type,4>> inputs(input_data, inputs_rows_number, inputs_columns_number, channels_number, 
        batch_samples_number);


    const TensorMap<Tensor<type, 4>> deltas(
        back_propagation->deltas_data, 
        back_propagation->deltas_dimensions(0), 
        back_propagation->deltas_dimensions(1),
        back_propagation->deltas_dimensions(2),
        back_propagation->deltas_dimensions(3));

    const Tensor<type, 4>& deltas_times_derivatives = deltas * convolutional_layer_forward_propagation->activations_derivatives;

    const Eigen::array<type, 3> reduction_dimensions({0, 1, 3});

    // Biases gradient
    convolutional_layer_back_propagation->biases_derivatives = deltas_times_derivatives.sum(reduction_dimensions);

    // Weights gradient
    convolutional_layer_back_propagation->synaptic_weights_derivatives.setZero();
    const Eigen::array<Index, 2> convolution_dims = {0, 1};

    #pragma omp parallel for
    for(Index kernel_index = 0; kernel_index < kernels_number; kernel_index++)
    {
        for(Index channel_index = 0; channel_index < channels_number; channel_index++)
        {
            for(Index image_index = 0; image_index < batch_samples_number; image_index++)
            {
                convolutional_layer_back_propagation->synaptic_weights_derivatives.chip(kernel_index, 3).chip(channel_index, 2) += inputs.chip(image_index, 3).chip(
                    channel_index, 2).convolve(
                        deltas_times_derivatives.chip(image_index, 3).chip(kernel_index, 2), convolution_dims);
            }
        }
    }
}


void ConvolutionalLayer::insert_gradient(LayerBackPropagation* back_propagation, const Index& index, Tensor<type, 1>& gradient) const
{
    ConvolutionalLayerBackPropagation* convolutional_layer_back_propagation =
            static_cast<ConvolutionalLayerBackPropagation*>(back_propagation);

    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    copy(convolutional_layer_back_propagation->biases_derivatives.data(),
         convolutional_layer_back_propagation->biases_derivatives.data() + biases_number,
         gradient.data() + index);

    copy(convolutional_layer_back_propagation->synaptic_weights_derivatives.data(),
         convolutional_layer_back_propagation->synaptic_weights_derivatives.data() + synaptic_weights_number,
         gradient.data() + index + biases_number);
}


/// Returns the convolutional layer's activation function.

ConvolutionalLayer::ActivationFunction ConvolutionalLayer::get_activation_function() const
{
    return activation_function;
}


/// Returns a string with the name of the layer activation function.
/// This can be Logistic, HyperbolicTangent, Threshold, SymmetricThreshold, Linear, RectifiedLinear, ScaledExponentialLinear.

string ConvolutionalLayer::write_activation_function() const
{
    switch(activation_function)
    {
    case ActivationFunction::Logistic:
        return "Logistic";

    case ActivationFunction::HyperbolicTangent:
        return "HyperbolicTangent";

    case ActivationFunction::Threshold:
        return "Threshold";

    case ActivationFunction::SymmetricThreshold:
        return "SymmetricThreshold";

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

Index ConvolutionalLayer::get_outputs_rows_number() const
{
    return (get_inputs_rows_number() + 2*get_padding_height() - get_kernels_rows_number()) / row_stride + 1;

    ///@todo padding

    //    const Index kernels_rows_number = get_kernels_rows_number();

    //    const Index padding_height = get_padding_height();

    //    return ((input_variables_dimensions(2) - kernels_rows_number + 2 * padding_height)/row_stride) + 1;
}


/// Returns the number of columns the result of applying the layer's kernels to an image will have.

Index ConvolutionalLayer::get_outputs_columns_number() const
{
    return (get_inputs_columns_number() + 2*get_padding_width() - get_kernels_columns_number()) / column_stride + 1;

    ///@todo padding

    //    const Index kernels_columns_number = get_kernels_columns_number();

    //    const Index padding_width = get_padding_width();

    //    return ((input_variables_dimensions(3) - kernels_columns_number + 2 * padding_width)/column_stride) + 1;
}


/// Returns a vector containing the number of channels, rows and columns of the result of applying the layer's kernels to an image.

Tensor<Index, 1> ConvolutionalLayer::get_outputs_dimensions() const
{
    Tensor<Index, 1> outputs_dimensions(4);

    outputs_dimensions(0) = get_outputs_rows_number(); // Rows
    outputs_dimensions(1) = get_outputs_columns_number(); // Cols
    outputs_dimensions(2) = get_kernels_number(); // Number of kernels (Channels)
    outputs_dimensions(3) = input_variables_dimensions(3); // Number of images

    return outputs_dimensions;
}


/// Returns the dimension of the input variables

Tensor<Index, 1> ConvolutionalLayer::get_input_variables_dimensions() const
{
    return input_variables_dimensions;
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


/// Returns the column stride.

Index ConvolutionalLayer::get_column_stride() const
{
    return column_stride;
}


/// Returns the row stride.

Index ConvolutionalLayer::get_row_stride() const
{
    return row_stride;
}


///Returns the number of kernels of the layer.

Index ConvolutionalLayer::get_kernels_number() const
{
    return synaptic_weights.dimension(3);
}


/// Returns the number of channels of the layer's kernels.

Index ConvolutionalLayer::get_kernels_channels_number() const
{
    return synaptic_weights.dimension(2);
}


/// Returns the number of rows of the layer's kernels.

Index  ConvolutionalLayer::get_kernels_rows_number() const
{
    return synaptic_weights.dimension(0);
}


/// Returns the number of columns of the layer's kernels.

Index ConvolutionalLayer::get_kernels_columns_number() const
{
    return synaptic_weights.dimension(1);
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
        const Index input_cols_number = get_inputs_columns_number();
        return (column_stride*(input_cols_number - 1) - input_cols_number + get_kernels_columns_number()) / 2;
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
        const Index input_rows_number = get_inputs_rows_number();
        return (row_stride*(input_rows_number - 1) - input_rows_number + get_kernels_rows_number()) / 2;
    }
    }

    return 0;
}


/// Returns the number of inputs

Index ConvolutionalLayer::get_inputs_number() const
{
    return get_inputs_channels_number() * get_inputs_rows_number() * get_inputs_columns_number();
}


/// Returns the number of neurons

Index ConvolutionalLayer::get_neurons_number() const
{
    const Index kernels_number = get_kernels_number();
    const Index kernels_rows_number = get_kernels_rows_number();
    const Index kernels_columns_number = get_kernels_columns_number();

    return kernels_number*kernels_rows_number*kernels_columns_number;
}


/// Returns the layer's parameters in the form of a vector.

Tensor<type, 1> ConvolutionalLayer::get_parameters() const
{
    Tensor<type, 1> parameters(get_parameters_number());

    const Index kernels_number = get_kernels_number();
    const Index kernels_channels_number = get_kernels_channels_number();
    const Index kernels_rows_number = get_kernels_rows_number();
    const Index kernels_columns_number = get_kernels_columns_number();

    memcpy(parameters.data(),
           biases.data(), static_cast<size_t>(biases.size())*sizeof(float));
    /*
    memcpy(parameters.data() + biases.size(),
           synaptic_weights.data(), static_cast<size_t>(synaptic_weights.size())*sizeof(float));
    */

    Index element_index = kernels_number;

    #pragma omp parallel for
    for(Index kernel_indx = 0; kernel_indx < kernels_number; kernel_indx++)
    {
        for(Index row_indx = 0; row_indx < kernels_rows_number; row_indx++)
        {
            for(Index column_indx = 0; column_indx < kernels_columns_number; column_indx++)
            {
                for(Index channel_indx = 0; channel_indx < kernels_channels_number; channel_indx++)
                {
                    parameters(element_index) = synaptic_weights(row_indx, column_indx, channel_indx, kernel_indx); 
                    element_index ++;
                }
            }
        }
    }

    return parameters;
}


/// Returns the number of parameters of the layer.

Index ConvolutionalLayer::get_parameters_number() const
{
    return synaptic_weights.size() + biases.size();
}


/// Sets and initializes the layer's parameters in accordance with the dimensions taken as input.
/// The initialization values are random values from a normal distribution.
/// @todo change to memcpy approach
/// @param new_inputs_dimensions A vector containing the desired inputs' dimensions (number of images, number of channels, rows number, columns number).
/// @param new_kernels_dimensions A vector containing the desired kernels' dimensions (number of kernels, number of channels, rows number, columns number).

void ConvolutionalLayer::set(const Tensor<Index, 1>& new_inputs_dimensions, const Tensor<Index, 1>& new_kernels_dimensions)
{
#ifdef OPENNN_DEBUG

    const Index inputs_dimensions_number = new_inputs_dimensions.size();

    if(inputs_dimensions_number != 4)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "ConvolutionalLayer(const Tensor<Index, 1>&) constructor.\n"
               << "Number of inputs dimensions (" << inputs_dimensions_number << ") must be 4 (number of images, channels, rows, columns).\n";

        throw invalid_argument(buffer.str());
    }

#endif

#ifdef OPENNN_DEBUG

    const Index kernels_dimensions_number = new_kernels_dimensions.size();

    if(kernels_dimensions_number != 4)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void set(const Tensor<Index, 1>&) method.\n"
               << "Number of kernels dimensions (" << kernels_dimensions_number << ") must be 4 (number of images, kernels, rows, columns).\n";

        throw invalid_argument(buffer.str());
    }

#endif

    const Index kernels_number = new_kernels_dimensions[3];
    const Index kernels_channels_number = new_kernels_dimensions[2];
    const Index kernels_columns_number = new_kernels_dimensions[1];
    const Index kernels_rows_number = new_kernels_dimensions[0];

    biases.resize(kernels_number);
    biases.setRandom();

    synaptic_weights.resize(kernels_rows_number, kernels_columns_number, kernels_channels_number, kernels_number);
    synaptic_weights.setRandom();

    input_variables_dimensions = new_inputs_dimensions;
}


/// Sets and initializes the layer's parameters in accordance with the dimensions taken as input.
/// The initialization values are random values from a normal distribution.
/// @param new_inputs Layer inputs.
/// @param new_kernels Layer synaptic weights.
/// @param new_biases Layer biases.

void ConvolutionalLayer::set(const Tensor<type, 4>& new_inputs, const Tensor<type, 4>& new_kernels, const Tensor<type, 1>& new_biases)
{
#ifdef OPENNN_DEBUG

    if(new_kernels.dimension(3) != new_biases.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void set(const Tensor<type, 4>& , const Tensor<type, 4>& , const Tensor<type, 1>& ) method.\n"
               << "Biases size must be equal to number of kernels.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<Index, 1> new_inputs_dimensions(4);

    new_inputs_dimensions(0) = new_inputs.dimension(0);
    new_inputs_dimensions(1) = new_inputs.dimension(1);
    new_inputs_dimensions(2) = new_inputs.dimension(2);
    new_inputs_dimensions(3) = new_inputs.dimension(3);

    synaptic_weights = new_kernels;

    biases = new_biases;

    input_variables_dimensions = new_inputs_dimensions;
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
/// The second argument is a string containing the name of the function("Logistic", "HyperbolicTangent", "Threshold", etc).
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
    else if(new_activation_function_name == "Threshold")
    {
        activation_function = ActivationFunction::Threshold;
    }
    else if(new_activation_function_name == "SymmetricThreshold")
    {
        activation_function = ActivationFunction::SymmetricThreshold;
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

        throw invalid_argument(buffer.str());
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

        throw invalid_argument(buffer.str());
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


/// Sets the kernels' column stride.
/// @param new_stride_row The desired column stride.

void ConvolutionalLayer::set_column_stride(const Index& new_stride_column)
{
    if(new_stride_column <= 0)
    {
        throw ("EXCEPTION: new_stride_column must be a positive number");
    }

    column_stride = new_stride_column;
}

void ConvolutionalLayer::set_input_variables_dimenisons(const Tensor<Index,1>& new_input_variables_dimensions)
{
    input_variables_dimensions = new_input_variables_dimensions;
}


/// Sets the synaptic weights and biases to the given values.
/// @param new_parameters A vector containing the synaptic weights and biases, in this order.

void ConvolutionalLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& )
{
    const Index kernels_number = get_kernels_number();
    const Index kernels_channels_number = get_kernels_channels_number();
    const Index kernels_rows_number = get_kernels_rows_number();
    const Index kernels_columns_number = get_kernels_columns_number();

    synaptic_weights.resize(kernels_rows_number, kernels_columns_number, kernels_channels_number, kernels_number);
    biases.resize(kernels_number);

    memcpy(biases.data(),
           new_parameters.data(),
           static_cast<size_t>(kernels_number)*sizeof(type));

    /*
    //Doesnt work like expected. Tensors are by default columnwise aligned in memory. 
    //If this should work, change to rowwise alignment and change dimension to NxHxWxC.
    memcpy(synaptic_weights.data(),
           new_parameters.data()+ biases.size(),
           static_cast<size_t>(synaptic_weights.size())*sizeof(type));
    */



    Index element_index = kernels_number;

    #pragma omp parallel for
    for(Index kernel_indx = 0; kernel_indx < kernels_number; kernel_indx++)
    {
        for(Index row_indx = 0; row_indx < kernels_rows_number; row_indx++)
        {
            for(Index column_indx = 0; column_indx < kernels_columns_number; column_indx++)
            {
                for(Index channel_indx = 0; channel_indx < kernels_channels_number; channel_indx++)
                {
                    synaptic_weights(row_indx, column_indx, channel_indx, kernel_indx) = new_parameters(element_index);
                    element_index ++;
                }
            }
        }
    }
    
}

/// Returns the number of biases in the layer.

Index ConvolutionalLayer::get_biases_number() const
{
    return biases.size();
}


/// Returns the number of synaptic weights in the layer.

Index ConvolutionalLayer::get_synaptic_weights_number() const
{
    return synaptic_weights.size();
}


/// Returns the number of images of the input.

Index ConvolutionalLayer::get_inputs_images_number() const
{
    return input_variables_dimensions[3];
}


/// Returns the number of channels of the input.

Index ConvolutionalLayer::get_inputs_channels_number() const
{
    return input_variables_dimensions[2];
}


/// Returns the number of rows of the input.

Index ConvolutionalLayer::get_inputs_rows_number() const
{
    return input_variables_dimensions[0];
}


/// Returns the number of columns of the input.

Index ConvolutionalLayer::get_inputs_columns_number() const
{
    return input_variables_dimensions[1];
}


void ConvolutionalLayer::to_2d(const Tensor<type, 4>& input_4d, Tensor<type, 2>& output_2d) const
{
    Eigen::array<Index, 2> dimensions =
    {Eigen::array<Index, 2>({input_4d.dimension(0), input_4d.dimension(1) * input_4d.dimension(2) * input_4d.dimension(3)})};

    output_2d = input_4d.reshape(dimensions);
}


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

    const Tensor<Index, 1> inputs_variables_dimensions = get_input_variables_dimensions();

    for(Index i = 0; i < inputs_variables_dimensions.size(); i++)
    {
        buffer << inputs_variables_dimensions(i);

        if(i != (inputs_variables_dimensions.size() - 1)) buffer << " ";
    }

    cout << "Input variables dimensions string: " << buffer.str().c_str() << endl;

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
    buffer << get_kernels_columns_number();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Activation function

    file_stream.OpenElement("ActivationFunction");

    file_stream.PushText(write_activation_function().c_str());

    file_stream.CloseElement();

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

        throw invalid_argument(buffer.str());
    }

    // Convolutional layer name element

    const tinyxml2::XMLElement* convolution_type_element = convolutional_layer_element->FirstChildElement("LayerName");

    if(!convolution_type_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolution type element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const string convolution_type_string = convolution_type_element->GetText();

    set_convolution_type(convolution_type_string);

    // Input variables dimensions element

    const tinyxml2::XMLElement* input_variables_dimensions_element = convolutional_layer_element->FirstChildElement("InputVariablesDimensions");

    if(!input_variables_dimensions_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional input variables dimensions element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const string input_variables_dimensions_string = input_variables_dimensions_element->GetText();

    //    set_input_variables_dimenisons(static_cast<Index>(stoi(input_variables_dimensions_string));

    // Filters Number element

    const tinyxml2::XMLElement* filters_number_element = input_variables_dimensions_element->FirstChildElement("FiltersNumber");

    if(!filters_number_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional filters number element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const string filters_number_element_string = filters_number_element->GetText();

    //    set_input_variables_dimenisons(static_cast<Index>(stoi(input_variables_dimensions_string));

    // Column stride

    const tinyxml2::XMLElement* filters_size_element = convolutional_layer_element->FirstChildElement("FiltersSize");

    if(!filters_size_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional filters size element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const string filters_size_element_string = filters_size_element->GetText();

    //    set_column_stride(static_cast<Index>(stoi(filters_size_element_string)));

    // Row stride

    const tinyxml2::XMLElement* activation_function_element = convolutional_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional activation function element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const string activation_function_string = activation_function_element->GetText();

    set_activation_function(activation_function_string);

    // Parameters

    const tinyxml2::XMLElement* parameters_element = convolutional_layer_element->FirstChildElement("Parameters");

    if(!parameters_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Parameters element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(parameters_element->GetText())
    {
        const string parameters_string = parameters_element->GetText();

        set_parameters(to_type_vector(parameters_string, ' '));
    }
}

ConvolutionalLayerForwardPropagation::ConvolutionalLayerForwardPropagation()
        : LayerForwardPropagation()
{
}


ConvolutionalLayerForwardPropagation::ConvolutionalLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer_pointer);
}

ConvolutionalLayerForwardPropagation::~ConvolutionalLayerForwardPropagation()
{
    outputs_data = nullptr;
}


void ConvolutionalLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
{
    layer_pointer = new_layer_pointer;

    const Index kernels_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_kernels_number();
    const Index outputs_rows_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_outputs_rows_number();
    const Index outputs_columns_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_outputs_columns_number();

    batch_samples_number = new_batch_samples_number;

    combinations.resize(outputs_rows_number, outputs_columns_number, kernels_number, batch_samples_number);
    outputs.resize(outputs_rows_number, outputs_columns_number, kernels_number, batch_samples_number);
    activations_derivatives.resize(outputs_rows_number, outputs_columns_number, kernels_number, batch_samples_number);

    combinations.setZero();
    outputs.setZero();
    activations_derivatives.setZero();

    outputs_data = outputs.data();

    outputs_dimensions = get_dimensions(outputs);
}

void ConvolutionalLayerForwardPropagation::print() const
{
    cout << "Combinations:" << endl;
    cout << combinations << endl;

    cout << "Outputs:" << endl;
    cout << outputs << endl;

    cout << "Activations derivatives:" << endl;
    cout << activations_derivatives << endl;
}

type* ConvolutionalLayerForwardPropagation::get_combinations_data() 
{
    return combinations.data();
}

type* ConvolutionalLayerForwardPropagation::get_activations_derivatives_data() 
{
    return activations_derivatives.data();
}

ConvolutionalLayerBackPropagation::ConvolutionalLayerBackPropagation() : LayerBackPropagation()
{
}

ConvolutionalLayerBackPropagation::ConvolutionalLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer_pointer);
}

ConvolutionalLayerBackPropagation::~ConvolutionalLayerBackPropagation()
{
}

void ConvolutionalLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
{
    layer_pointer = new_layer_pointer;

    batch_samples_number = new_batch_samples_number;

    const Index kernels_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_kernels_number();
    const Index outputs_rows_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_outputs_rows_number();
    const Index outputs_columns_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_outputs_columns_number();
    const Index synaptic_weights_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_synaptic_weights_number();

//        delta.resize(batch_samples_number, kernels_number*outputs_rows_number*outputs_columns_number); --> old
    deltas_dimensions.resize(4);
    deltas_dimensions.setValues({outputs_rows_number, outputs_columns_number, kernels_number, batch_samples_number});

    //delete deltas_data;
    deltas_data = (type*)malloc(static_cast<size_t>(batch_samples_number*kernels_number*outputs_rows_number*outputs_columns_number*sizeof(type)));

    convolutional_delta.resize(outputs_rows_number, outputs_columns_number, kernels_number, batch_samples_number);

    biases_derivatives.resize(kernels_number);

    const Index kernel_rows_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_kernels_rows_number();
    const Index kernel_columns_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_kernels_columns_number();
    const Index kernel_channels_number = static_cast<ConvolutionalLayer*>(layer_pointer)->get_kernels_channels_number();

    synaptic_weights_derivatives.resize(kernel_rows_number, kernel_columns_number, kernel_channels_number, kernels_number);
}

void ConvolutionalLayerBackPropagation::print() const
{
    cout << "Deltas:" << endl;
    //cout << deltas << endl;

//        cout << "Biases derivatives:" << endl;
//        cout << biases_derivatives << endl;

//        cout << "Synaptic weights derivatives:" << endl;
//        cout << synaptic_weights_derivatives << endl;

}


}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
