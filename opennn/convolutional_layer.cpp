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

void ConvolutionalLayer::calculate_convolutions(const Tensor<type, 4>& inputs, Tensor<type, 4> & combinations) const
{

    const Index images_number = inputs.dimension(3);
    const Index rows_input = inputs.dimension(0);
    const Index cols_input = inputs.dimension(1);

    const Index kernels_number = synaptic_weights.dimension(3);
    const Index kernel_rows = synaptic_weights.dimension(0);
    const Index kernel_cols = synaptic_weights.dimension(1);

#ifdef OPENNN_DEBUG

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

#endif

    const Index next_image = inputs.dimension(0)*inputs.dimension(1)*inputs.dimension(2);
    const Index next_kernel = synaptic_weights.dimension(0)*synaptic_weights.dimension(1)*synaptic_weights.dimension(2);

    const Index output_size_rows_cols = ((rows_input-kernel_rows)+1)*((cols_input-kernel_cols)+1);

    const Eigen::array<ptrdiff_t, 3> dims = {0, 1, 2};

    Tensor<type, 4> inputs_pointer = inputs;
    Tensor<type, 4> synaptic_weights_pointer = synaptic_weights;

    #pragma omp parallel for
    for(int i =0; i<images_number ;i++)
    {
        const TensorMap<Tensor<type, 3>> single_image(inputs_pointer.data()+i*next_image,
                                                      inputs.dimension(0),
                                                      inputs.dimension(1),
                                                      inputs.dimension(2));
        for(int j = 0; j<kernels_number; j++)
        {
            const TensorMap<Tensor<type, 3>> single_kernel(synaptic_weights_pointer.data()+j*next_kernel,
                                                           synaptic_weights.dimension(0),
                                                           synaptic_weights.dimension(1),
                                                           synaptic_weights.dimension(2));

            Tensor<type, 3> tmp_result = single_image.convolve(single_kernel, dims) + biases(j);

            memcpy(combinations.data() +j*output_size_rows_cols +i*output_size_rows_cols*kernels_number,
                   tmp_result.data(), static_cast<size_t>(output_size_rows_cols)*sizeof(type));
         }
    }
}


void ConvolutionalLayer::calculate_convolutions(const Tensor<type, 4>& inputs,
                                                const Tensor<type, 2>& potential_biases,
                                                const Tensor<type, 4>& potential_synaptic_weights,
                                                Tensor<type, 4>& convolutions) const
{
}



/// Calculates activations

void ConvolutionalLayer::calculate_activations(Tensor<type, 4>& inputs, Tensor<type, 4>& activations) const
{
    Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);
    Tensor<Index, 1> activations_dimensions = get_dimensions(activations);

    switch(activation_function)
    {
        case ActivationFunction::Linear: linear(inputs.data(), inputs_dimensions, activations.data(), activations_dimensions); return;

        case ActivationFunction::Logistic: logistic(inputs.data(), inputs_dimensions, activations.data(), activations_dimensions); return;

        case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(inputs.data(), inputs_dimensions, activations.data(), activations_dimensions); return;

        case ActivationFunction::Threshold: threshold(inputs.data(), inputs_dimensions, activations.data(), activations_dimensions); return;

        case ActivationFunction::SymmetricThreshold: symmetric_threshold(inputs.data(), inputs_dimensions, activations.data(), activations_dimensions); return;

        case ActivationFunction::RectifiedLinear: rectified_linear(inputs.data(), inputs_dimensions, activations.data(), activations_dimensions); return;

        case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(inputs.data(), inputs_dimensions, activations.data(), activations_dimensions); return;

        case ActivationFunction::SoftPlus: soft_plus(inputs.data(), inputs_dimensions, activations.data(), activations_dimensions); return;

        case ActivationFunction::SoftSign: soft_sign(inputs.data(), inputs_dimensions, activations.data(), activations_dimensions); return;

        case ActivationFunction::HardSigmoid: hard_sigmoid(inputs.data(), inputs_dimensions, activations.data(), activations_dimensions); return;

        case ActivationFunction::ExponentialLinear: exponential_linear(inputs.data(), inputs_dimensions, activations.data(), activations_dimensions); return;
    }
}


void ConvolutionalLayer::calculate_activations_derivatives(Tensor<type, 4>& combinations_4d,
                                                           Tensor<type, 4>& activations,
                                                           Tensor<type, 4>& activations_derivatives) const
{
    Tensor<Index, 1> combinations_dimensions = get_dimensions(combinations_4d);
    Tensor<Index, 1> activations_dimensions = get_dimensions(activations);
    Tensor<Index, 1> activations_derivatives_dimensions = get_dimensions(activations_derivatives);

    switch(activation_function)
    {
        case ActivationFunction::Linear: linear_derivatives(combinations_4d.data(), combinations_dimensions, activations.data(), activations_dimensions, activations_derivatives.data(), activations_derivatives_dimensions); return;

        case ActivationFunction::Logistic: logistic_derivatives(combinations_4d.data(), combinations_dimensions, activations.data(), activations_dimensions, activations_derivatives.data(), activations_derivatives_dimensions); return;

        case ActivationFunction::HyperbolicTangent: hyperbolic_tangent_derivatives(combinations_4d.data(), combinations_dimensions, activations.data(), activations_dimensions, activations_derivatives.data(), activations_derivatives_dimensions); return;

        case ActivationFunction::Threshold: threshold_derivatives(combinations_4d.data(), combinations_dimensions, activations.data(), activations_dimensions, activations_derivatives.data(), activations_derivatives_dimensions); return;

        case ActivationFunction::SymmetricThreshold: symmetric_threshold_derivatives(combinations_4d.data(), combinations_dimensions, activations.data(), activations_dimensions, activations_derivatives.data(), activations_derivatives_dimensions); return;

        case ActivationFunction::RectifiedLinear: rectified_linear_derivatives(combinations_4d.data(), combinations_dimensions, activations.data(), activations_dimensions, activations_derivatives.data(), activations_derivatives_dimensions); return;

        case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear_derivatives(combinations_4d.data(), combinations_dimensions, activations.data(), activations_dimensions, activations_derivatives.data(), activations_derivatives_dimensions); return;

        case ActivationFunction::SoftPlus: soft_plus_derivatives(combinations_4d.data(), combinations_dimensions, activations.data(), activations_dimensions, activations_derivatives.data(), activations_derivatives_dimensions); return;

        case ActivationFunction::SoftSign: soft_sign_derivatives(combinations_4d.data(), combinations_dimensions, activations.data(), activations_dimensions, activations_derivatives.data(), activations_derivatives_dimensions); return;

        case ActivationFunction::HardSigmoid: hard_sigmoid_derivatives(combinations_4d.data(), combinations_dimensions, activations.data(), activations_dimensions, activations_derivatives.data(), activations_derivatives_dimensions); return;

        case ActivationFunction::ExponentialLinear: exponential_linear_derivatives(combinations_4d.data(), combinations_dimensions, activations.data(), activations_dimensions, activations_derivatives.data(), activations_derivatives_dimensions); return;
    }
}


void ConvolutionalLayer::forward_propagate(const Tensor<type, 4> &inputs, LayerForwardPropagation* forward_propagation)
{
    ConvolutionalLayerForwardPropagation* convolutional_layer_forward_propagation
            = static_cast<ConvolutionalLayerForwardPropagation*>(forward_propagation);

#ifdef OPENNN_DEBUG

    const Tensor<Index, 1> outputs_dimensions = get_outputs_dimensions();

    if(outputs_dimensions[0] != convolutional_layer_forward_propagation->combinations.dimension(0))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "ConvolutionalLayer::forward_propagate.\n"
               << "outputs_dimensions[0]" <<outputs_dimensions[0] <<"must be equal to" << convolutional_layer_forward_propagation->combinations.dimension(0)<<".\n";

        throw invalid_argument(buffer.str());
    }

    if(outputs_dimensions[1] != convolutional_layer_forward_propagation->combinations.dimension(1))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "ConvolutionalLayer::forward_propagate.\n"
               << "outputs_dimensions[1]" <<outputs_dimensions[1] <<"must be equal to" << convolutional_layer_forward_propagation->combinations.dimension(1)<<".\n";

        throw invalid_argument(buffer.str());
    }

    if(outputs_dimensions[2] != convolutional_layer_forward_propagation->combinations.dimension(2))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "ConvolutionalLayer::forward_propagate.\n"
               << "outputs_dimensions[2]" <<outputs_dimensions[2] <<"must be equal to" << convolutional_layer_forward_propagation->combinations.dimension(2)<<".\n";

        throw invalid_argument(buffer.str());
    }

    if(outputs_dimensions[3] != convolutional_layer_forward_propagation->combinations.dimension(3))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "ConvolutionalLayer::forward_propagate.\n"
               << "outputs_dimensions[3]" <<outputs_dimensions[3] <<"must be equal to" << convolutional_layer_forward_propagation->combinations.dimension(3)<<".\n";

        throw invalid_argument(buffer.str());
    }

#endif


    calculate_convolutions(inputs,
                           convolutional_layer_forward_propagation->combinations);

    calculate_activations_derivatives(convolutional_layer_forward_propagation->combinations,
                                      convolutional_layer_forward_propagation->activations,
                                      convolutional_layer_forward_propagation->activations_derivatives);

/// @todo check here next layer
//    to_2d(convolutional_layer_forward_propagation->combinations_4d, convolutional_layer_forward_propagation->combinations);
//    to_2d(convolutional_layer_forward_propagation->activations_4d, convolutional_layer_forward_propagation->activations);
//    to_2d(convolutional_layer_forward_propagation->activations_derivatives_4d, convolutional_layer_forward_propagation->activations_derivatives_2d);
}


void ConvolutionalLayer::calculate_hidden_delta_perceptron(PerceptronLayerForwardPropagation* next_forward_propagation,
                                                           PerceptronLayerBackPropagation* next_back_propagation,
                                                           ConvolutionalLayerBackPropagation* back_propagation) const
{
    // Current layer's values

    ConvolutionalLayer* convolutional_layer = static_cast<ConvolutionalLayer*>(back_propagation->layer_pointer);

    const Index images_number = back_propagation->batch_samples_number;
    const Index kernels_number = convolutional_layer->get_kernels_number();
    const Index output_rows_number = convolutional_layer->get_outputs_rows_number();
    const Index output_columns_number = convolutional_layer->get_outputs_columns_number();

    const TensorMap<Tensor<type, 2>> next_deltas(next_back_propagation->deltas_data, next_back_propagation->deltas_dimensions(0), next_back_propagation->deltas_dimensions(1));;
    TensorMap<Tensor<type, 2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));

    const Index size = deltas.size();

    // Next layer's values

    const Index neurons_perceptron = next_forward_propagation->layer_pointer->get_neurons_number();

    const Tensor<type,2> synaptic_weights_perceptron = static_cast<PerceptronLayer*>(next_back_propagation->layer_pointer)->get_synaptic_weights();

    Index hidden_input = 0;

    for(Index tensor_index = 0; tensor_index < size; tensor_index++)
    {
        const Index image_index = tensor_index / (kernels_number * output_rows_number * output_columns_number);
        const Index channel_index = (tensor_index / (output_rows_number * output_columns_number)) % kernels_number;

        const Index column_index  = (tensor_index / output_rows_number) % output_columns_number ;
        const Index row_index = tensor_index % output_rows_number;

        hidden_input =  row_index + column_index*output_rows_number + channel_index*output_rows_number*output_columns_number;

        type sum = 0.0;

        for(Index sum_index = 0; sum_index < neurons_perceptron; sum_index++)
        {
            const type delta_element = next_deltas(image_index, sum_index);

            const type weight = synaptic_weights_perceptron (hidden_input, sum_index);
            sum += delta_element * weight;

        }

        deltas(image_index,
                                row_index +
                                column_index*output_rows_number +
                                channel_index*output_rows_number*output_columns_number) = sum;
        }

    //    back_propagation->delta.device(*thread_pool_device) = back_propagation->delta * next_forward_propagation->activations_derivatives;

    //    const Tensor<type, 2>& next_synaptic_weights = static_cast<PerceptronLayer*>(next_back_propagation->layer_pointer)->get_synaptic_weights();

    //    back_propagation->delta.device(*thread_pool_device) =
    //            (next_back_propagation->delta*next_forward_propagation->activations_derivatives).contract(next_synaptic_weights, A_BT);
     }



//void ConvolutionalLayer::forward_propagate(const Tensor<type, 2>&inputs, LayerForwardPropagation* forward_propagation)
//{
//    const Eigen::array<Eigen::Index, 4> four_dims = {input_variables_dimensions(3), // columns number
//                                                     input_variables_dimensions(2), // rows number
//                                                     input_variables_dimensions(1), // channels number
//                                                     inputs.dimension(0)}; // images number
//    const Eigen::array<Eigen::Index, 2> shuffle_dims_2D = {1, 0};
//    const Eigen::array<Eigen::Index, 4> shuffle_dims_4D = {3, 2, 1, 0};

//    const Tensor<type, 4> inputs_4d = inputs.shuffle(shuffle_dims_2D).reshape(four_dims).shuffle(shuffle_dims_4D);

//    forward_propagate(inputs_4d, forward_propagation);
//}

/*
void ConvolutionalLayer::forward_propagate(const Tensor<type, 4>& inputs,
                                           Tensor<type, 1> potential_parameters,
                                           LayerForwardPropagation* forward_propagation)
{
    ConvolutionalLayerForwardPropagation* convolutional_layer_forward_propagation = static_cast<ConvolutionalLayerForwardPropagation*>(forward_propagation);

    const Tensor<Index, 1> outputs_dimensions = get_outputs_dimensions();

    convolutional_layer_forward_propagation->combinations.resize(outputs_dimensions(0),
                                                                    outputs_dimensions(1),
                                                                    outputs_dimensions(2),
                                                                    outputs_dimensions(3));

    convolutional_layer_forward_propagation->activations.resize(outputs_dimensions(0),
                                                                   outputs_dimensions(1),
                                                                   outputs_dimensions(2),
                                                                   outputs_dimensions(3));

    convolutional_layer_forward_propagation->activations_derivatives.resize(outputs_dimensions(0),
                                                                               outputs_dimensions(1),
                                                                               outputs_dimensions(2),
                                                                               outputs_dimensions(3));

    const Index kernels_number = synaptic_weights.dimension(0);

    const TensorMap<Tensor<type, 2>> potential_biases(potential_parameters.data(),
                                                      kernels_number,
                                                      1);

    //    TensorMap<Tensor<type, 4>> potential_synaptic_weights(potential_parameters.data() + kernels_number,
    //                                                          synaptic_weights.dimension(3),
    //                                                          synaptic_weights.dimension(2),
    //                                                          synaptic_weights.dimension(1),
    //                                                          kernels_number);


    const Index kernels_channels_number = get_kernels_channels_number();
    const Index kernels_rows_number = get_kernels_rows_number();
    const Index kernels_columns_number = get_kernels_columns_number();

    Tensor<type, 4> potential_synaptic_weights(kernels_number,
                                               kernels_channels_number,
                                               kernels_rows_number,
                                               kernels_columns_number);
    Index element_index = 0;

    //#pragma omp for
    for(Index i = 0; i < kernels_number; i++)
    {
        for(Index j = 0; j < kernels_channels_number; j++)
        {
            for(Index k = 0; k < kernels_rows_number; k++)
            {
                for(Index l = 0; l < kernels_columns_number; l++)
                {
                    potential_synaptic_weights(i ,j, k, l) = potential_parameters(kernels_number + element_index);
                    element_index ++;
                }
            }
        }
    }

    calculate_convolutions(inputs,
                           potential_biases,
                           potential_synaptic_weights,
                           convolutional_layer_forward_propagation->combinations);

    calculate_activations_derivatives(convolutional_layer_forward_propagation->combinations,
                                      convolutional_layer_forward_propagation->activations,
                                      convolutional_layer_forward_propagation->activations_derivatives);
}
*/
/*
void ConvolutionalLayer::forward_propagate(const Tensor<type, 2>& inputs,
                                           Tensor<type, 1> potential_parameters,
                                           LayerForwardPropagation* forward_propagation)
{
    const Eigen::array<Eigen::Index, 4> four_dims = {
        input_variables_dimensions(0),
        input_variables_dimensions(1),
        input_variables_dimensions(2),
        input_variables_dimensions(3)};

    const Eigen::array<Eigen::Index, 2> shuffle_dims_2D = {1, 0};
    const Eigen::array<Eigen::Index, 4> shuffle_dims_4D = {3, 2, 1, 0};

    const Tensor<type, 4> inputs_4d = inputs.shuffle(shuffle_dims_2D).reshape(four_dims).shuffle(shuffle_dims_4D);
//    const Tensor<type, 4> inputs_4d = inputs.reshape(four_dims);

    forward_propagate(inputs_4d, potential_parameters, forward_propagation);
}
*/
/*
void ConvolutionalLayer::calculate_hidden_delta(LayerForwardPropagation*,
                                                LayerBackPropagation*,
                                                LayerBackPropagation*) const
{

    ConvolutionalLayerForwardPropagation* convolutional_layer_forward_propagation = static_cast<ConvolutionalLayerForwardPropagation*>(forward_propagation);

    const Type next_layer_type = next_layer_pointer->get_type();

    if(next_layer_type == Type::Convolutional)
    {
        ConvolutionalLayer* convolutional_layer = dynamic_cast<ConvolutionalLayer*>(next_layer_pointer);

//        calculate_hidden_delta_convolutional(convolutional_layer,
//                                             activations,
//                                             activations_derivatives,
//                                             next_layer_delta,
//                                             hidden_delta);
    }
    else if(next_layer_type == Type::Pooling)
    {
        PoolingLayer* pooling_layer = dynamic_cast<PoolingLayer*>(next_layer_pointer);

        calculate_hidden_delta_pooling(pooling_layer,
                                       convolutional_layer_forward_propagation->activations,
                                       convolutional_layer_forward_propagation->activations_derivatives,
                                       next_layer_delta,
                                       hidden_delta);
    }
    else if(next_layer_type == Type::Perceptron)
    {
        PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(next_layer_pointer);

//        calculate_hidden_delta_perceptron(perceptron_layer,
//                                          convolutional_layer_forward_propagation->activations,
//                                          convolutional_layer_forward_propagation->activations_derivatives,
//                                          next_layer_delta,
//                                          hidden_delta);

    }
    else if(next_layer_type == Type::Probabilistic)
    {
        ProbabilisticLayer* probabilistic_layer = dynamic_cast<ProbabilisticLayer*>(next_layer_pointer);

        calculate_hidden_delta_probabilistic(probabilistic_layer,
                                             convolutional_layer_forward_propagation->activations,
                                             convolutional_layer_forward_propagation->activations_derivatives,
                                             next_layer_delta,
                                             hidden_delta);
    }

}


void ConvolutionalLayer::calculate_hidden_delta_convolutional(ConvolutionalLayer* next_layer_pointer,
                                                              const Tensor<type, 4>& activations,
                                                              const Tensor<type, 4>& activations_derivatives,
                                                              const Tensor<type, 4>& next_layer_delta,
                                                              Tensor<type, 2>& hidden_delta) const
{

    // Current layer's values

    const auto images_number = next_layer_delta.dimension(3);

    const Index kernels_number = get_kernels_number();
    const Index output_rows_number = get_outputs_rows_number();
    const Index output_columns_number = get_outputs_columns_number();

    // Next layer's values

    const Index next_layers_kernels_number = next_layer_pointer->get_kernels_number();
    const Index next_layers_kernel_rows = next_layer_pointer->get_kernels_rows_number();
    const Index next_layers_kernel_columns = next_layer_pointer->get_kernels_columns_number();

    const Index next_layers_output_rows = next_layer_pointer->get_outputs_rows_number();
    const Index next_layers_output_columns = next_layer_pointer->get_outputs_columns_number();

    const Index next_layers_row_stride = next_layer_pointer->get_row_stride();
    const Index next_layers_column_stride = next_layer_pointer->get_column_stride();

    const Tensor<type, 4> next_layers_weights = next_layer_pointer->get_synaptic_weights();

    // Hidden delta calculation

//    hidden_delta.resize(images_number, kernels_number, output_rows_number, output_columns_number);
    hidden_delta.resize(images_number, kernels_number * output_rows_number * output_columns_number);

    const Index size = hidden_delta.size();

    #pragma omp parallel for

    for(Index tensor_index = 0; tensor_index < size; tensor_index++)
    {
        const Index image_index = tensor_index / (kernels_number * output_rows_number * output_columns_number);
        const Index channel_index = (tensor_index / (output_rows_number * output_columns_number)) % kernels_number;
        const Index row_index = (tensor_index / output_columns_number) % output_rows_number;
        const Index column_index = tensor_index % output_columns_number;

        long double sum = 0.0;

        const Index lower_row_index = (row_index - next_layers_kernel_rows) / next_layers_row_stride + 1;
        const Index upper_row_index = min(row_index/next_layers_row_stride + 1, next_layers_output_rows);
        const Index lower_column_index = (column_index - next_layers_kernel_columns) / next_layers_column_stride + 1;
        const Index upper_column_index = min(column_index / next_layers_column_stride + 1, next_layers_output_columns);

        for(Index i = 0; i < next_layers_kernels_number; i++)
        {
            for(Index j = lower_row_index; j < upper_row_index; j++)
            {
                for(Index k = lower_column_index; k < upper_column_index; k++)
                {
                    const type delta_element = next_layer_delta(image_index, i, j, k);

                    const type weight = next_layers_weights(row_index - j * next_layers_row_stride,
                                                            column_index - k * next_layers_column_stride,
                                                            channel_index,
                                                            i);

                    sum += delta_element*weight;
                }
            }
        }
//        hidden_delta(row_index, column_index, channel_index, image_index) = sum;
        hidden_delta(row_index, column_index + channel_index + image_index) = sum;
    }

//    hidden_delta = hidden_delta*activations_derivatives;
}


void ConvolutionalLayer::calculate_hidden_delta_pooling(PoolingLayer* next_layer_pointer,
                                                        const Tensor<type, 4>& activations,
                                                        const Tensor<type, 4>& activations_derivatives,
                                                        const Tensor<type, 2>& next_layer_delta,
                                                        Tensor<type, 2>& hidden_delta) const
{

        switch(next_layer_pointer->get_pooling_method())
        {
            case opennn::PoolingLayer::PoolingMethod::NoPooling:
            {
//                return next_layer_delta;
                break;
            }
            case opennn::PoolingLayer::PoolingMethod::AveragePooling:
            {
            // Current layer's values

                const Index images_number = next_layer_delta.dimension(0);
                const Index kernels_number = get_kernels_number();
                const Index output_rows_number = get_outputs_rows_number();
                const Index output_columns_number = get_outputs_columns_number();

                // Next layer's values

                const Index next_layers_pool_rows = next_layer_pointer->get_pool_rows_number();
                const Index next_layers_pool_columns = next_layer_pointer->get_pool_columns_number();
                const Index next_layers_output_rows = next_layer_pointer->get_outputs_rows_number();
                const Index next_layers_output_columns = next_layer_pointer->get_outputs_columns_number();
                const Index next_layers_row_stride = next_layer_pointer->get_row_stride();
                const Index next_layers_column_stride = next_layer_pointer->get_column_stride();

                // Hidden delta calculation

                hidden_delta.resize(images_number, kernels_number * output_rows_number * output_columns_number);

                const Index size = hidden_delta.size();

                #pragma omp parallel for

                for(Index tensor_index = 0; tensor_index < size; tensor_index++)
                {
                    const Index image_index = tensor_index/(kernels_number*output_rows_number*output_columns_number);
                    const Index channel_index = (tensor_index/(output_rows_number*output_columns_number))%kernels_number;
                    const Index row_index = (tensor_index/output_columns_number)%output_rows_number;
                    const Index column_index = tensor_index%output_columns_number;

                    long double sum = 0.0;

                    const Index lower_row_index = (row_index - next_layers_pool_rows)/next_layers_row_stride + 1;
                    const Index upper_row_index = min(row_index/next_layers_row_stride + 1, next_layers_output_rows);
                    const Index lower_column_index = (column_index - next_layers_pool_columns)/next_layers_column_stride + 1;
                    const Index upper_column_index = min(column_index/next_layers_column_stride + 1, next_layers_output_columns);

                    for(Index i = lower_row_index; i < upper_row_index; i++)
                    {
                        for(Index j = lower_column_index; j < upper_column_index; j++)
                        {
//                            const type delta_element = next_layer_delta(image_index, channel_index, i, j);

//                            sum += delta_element;
                        }
                    }

//                    hidden_delta(image_index, channel_index, row_index, column_index) = sum;
                    hidden_delta(image_index, channel_index + row_index + column_index) = sum;
                }

//                return (hidden_delta*activations_derivatives)/(next_layers_pool_rows*next_layers_pool_columns);
                break;
            }
            case opennn::PoolingLayer::PoolingMethod::MaxPooling:
            {
                // Current layer's values

                const Index images_number = next_layer_delta.dimension(0);
                const Index kernels_number = get_kernels_number();
                const Index output_rows_number = get_outputs_rows_number();
                const Index output_columns_number = get_outputs_columns_number();

                // Next layer's values

                const Index next_layers_pool_rows = next_layer_pointer->get_pool_rows_number();
                const Index next_layers_pool_columns = next_layer_pointer->get_pool_columns_number();
                const Index next_layers_output_rows = next_layer_pointer->get_outputs_rows_number();
                const Index next_layers_output_columns = next_layer_pointer->get_outputs_columns_number();
                const Index next_layers_row_stride = next_layer_pointer->get_row_stride();
                const Index next_layers_column_stride = next_layer_pointer->get_column_stride();

                // Hidden delta calculation

                hidden_delta.resize(images_number, kernels_number * output_rows_number * output_columns_number);

                const Index size = hidden_delta.size();

                #pragma omp parallel for

                for(Index tensor_index = 0; tensor_index < size; tensor_index++)
                {
                    const Index image_index = tensor_index/(kernels_number*output_rows_number*output_columns_number);
                    const Index channel_index = (tensor_index/(output_rows_number*output_columns_number))%kernels_number;
                    const Index row_index = (tensor_index/output_columns_number)%output_rows_number;
                    const Index column_index = tensor_index%output_columns_number;

                    long double sum = 0.0;

                    const Index lower_row_index = (row_index - next_layers_pool_rows)/next_layers_row_stride + 1;
                    const Index upper_row_index = min(row_index/next_layers_row_stride + 1, next_layers_output_rows);
                    const Index lower_column_index = (column_index - next_layers_pool_columns)/next_layers_column_stride + 1;
                    const Index upper_column_index = min(column_index/next_layers_column_stride + 1, next_layers_output_columns);

                    for(Index i = lower_row_index; i < upper_row_index; i++)
                    {
                        for(Index j = lower_column_index; j < upper_column_index; j++)
                        {
//                            const type delta_element = next_layer_delta(image_index, channel_index, i, j);

                            Tensor<type, 2> activations_current_submatrix(next_layers_pool_rows, next_layers_pool_columns);

                            for(Index submatrix_row_index = 0; submatrix_row_index < next_layers_pool_rows; submatrix_row_index++)
                            {
                                for(Index submatrix_column_index = 0; submatrix_column_index < next_layers_pool_columns; submatrix_column_index++)
                                {
//                                    activations_current_submatrix(submatrix_row_index, submatrix_column_index) =
//                                            activations(image_index, channel_index, i*next_layers_row_stride + submatrix_row_index, j*next_layers_column_stride + submatrix_column_index);
                                }
                            }

                            Tensor<type, 2> multiply_not_multiply(next_layers_pool_rows, next_layers_pool_columns);

                            type max_value = activations_current_submatrix(0,0);

                            for(Index submatrix_row_index = 0; submatrix_row_index < next_layers_pool_rows; submatrix_row_index++)
                            {
                                for(Index submatrix_column_index = 0; submatrix_column_index < next_layers_pool_columns; submatrix_column_index++)
                                {
                                    if(activations_current_submatrix(submatrix_row_index, submatrix_column_index) > max_value)
                                    {
                                        max_value = activations_current_submatrix(submatrix_row_index, submatrix_column_index);

//                                        multiply_not_multiply.resize(next_layers_pool_rows, next_layers_pool_columns, 0.0);
                                        multiply_not_multiply(submatrix_row_index, submatrix_column_index) = type(1);
                                    }
                                }
                            }

                            const type max_derivative = multiply_not_multiply(row_index - i*next_layers_row_stride, column_index - j*next_layers_column_stride);

//                            sum += delta_element*max_derivative;
                        }
                    }

//                    hidden_delta(image_index, channel_index, row_index, column_index) = sum;
                    hidden_delta(image_index + channel_index + row_index + column_index) = sum;
                }

//                return hidden_delta*activations_derivatives;
                break;
            }
        }
}


/// @todo


void ConvolutionalLayer::calculate_hidden_delta_perceptron(const PerceptronLayer* next_layer_pointer,
                                                           const Tensor<type, 4>& ,
                                                           const Tensor<type, 2>& activations_derivatives,
                                                           const Tensor<type, 2>& next_layer_delta,
                                                           Tensor<type, 2>& hidden_delta) const
{
    // activations, activations_derivatives: images number, kernels number, output rows number, output columns number

    // Convolutional layer

//    const Index images_number = next_layer_delta.dimension(0);
//    const Index kernels_number = get_kernels_number();
//    const Index output_rows_number = get_outputs_rows_number();
//    const Index output_columns_number = get_outputs_columns_number();

    // Perceptron layer

//    const Tensor<type, 2>& next_layer_weights = next_layer_pointer->get_synaptic_weights();

//    hidden_delta = next_layer_delta.contract(next_layer_weights, A_BT);

//    hidden_delta.device(*thread_pool_device) = hidden_delta*activations_derivatives;

        // Next layer's values

        const Index next_layers_output_columns = next_layer_delta.dimension(1);

        const Tensor<type, 2>& next_layers_weights = next_layer_pointer->get_synaptic_weights();

        // Hidden delta calculation

//        hidden_delta.resize(images_number, kernels_number * output_rows_number * output_columns_number);

        const Index size = hidden_delta.size();

//        #pragma omp parallel for

        for(Index tensor_index = 0; tensor_index < size; tensor_index++)
        {
//            const Index image_index = tensor_index/(kernels_number*output_rows_number*output_columns_number);
//            const Index channel_index = (tensor_index/(output_rows_number*output_columns_number))%kernels_number;
//            const Index row_index = (tensor_index/output_columns_number)%output_rows_number;
//            const Index column_index = tensor_index%output_columns_number;

            long double sum = 0.0;

//            for(Index sum_index = 0; sum_index < next_layers_output_columns; sum_index++)
//            {
//                const type delta_element = next_layer_delta(image_index, sum_index);

//                const type weight = next_layers_weights(channel_index + row_index*kernels_number + column_index*kernels_number*output_rows_number, sum_index);

//                sum += delta_element*weight;
//            }
//            hidden_delta(row_index, column_index, channel_index, image_index) = sum;
//            hidden_delta(row_index, column_index + channel_index + image_index) = sum;
        }
}
*/

void ConvolutionalLayer::calculate_hidden_delta_probabilistic(ProbabilisticLayer* next_layer_pointer,
                                                              const Tensor<type, 4>& activations,
                                                              const Tensor<type, 4>& activations_derivatives,
                                                              const Tensor<type, 2>& next_layer_delta,
                                                              Tensor<type, 2>& hidden_delta) const
{
    // Current layer's values
    const Index images_number = next_layer_delta.dimension(0);
    const Index kernels_number = get_kernels_number();
    const Index output_rows_number = get_outputs_rows_number();
    const Index output_columns_number = get_outputs_columns_number();

    // Next layer's values

    const Index next_layers_output_columns = next_layer_delta.dimension(1);

    const Tensor<type, 2> next_layers_weights = next_layer_pointer->get_synaptic_weights();

    // Hidden delta calculation

    hidden_delta.resize(images_number, kernels_number * output_rows_number * output_columns_number);

    const Index size = hidden_delta.size(); // Number of total parameters

    #pragma omp parallel for

    for(Index tensor_index = 0; tensor_index < size; tensor_index++)
    {
//        const Index image_index = tensor_index / (kernels_number * output_rows_number * output_columns_number);

//        const Index channel_index = (tensor_index / (output_rows_number * output_columns_number)) % kernels_number;
//        const Index row_index = (tensor_index / output_columns_number) % output_rows_number;
//        const Index column_index = tensor_index % output_columns_number;

        const Index image_index = tensor_index / (kernels_number * output_rows_number * output_columns_number);

        const Index channel_index = (tensor_index / (output_rows_number * output_columns_number)) % kernels_number;
        const Index row_index = (tensor_index / output_columns_number) % output_rows_number;
        const Index column_index = tensor_index % output_columns_number;

        long double sum = 0.0;

        for(Index sum_index = 0; sum_index < next_layers_output_columns; sum_index++)
        {
//            const type delta_element = next_layer_delta(image_index, sum_index);

//            const type weight = next_layers_weights(sum_index,
//                                                    channel_index +
            //                                        (row_index * kernels_number) +
//                                                    (column_index * kernels_number * output_rows_number));

            const type delta_element = next_layer_delta(image_index, sum_index);

            const type weight = next_layers_weights(channel_index + row_index + column_index,
                                                    sum_index);

//            sum += delta_element * weight;
            sum += type(0);
        }

//        hidden_delta(image_indexº, channel_index, row_index, column_index) = sum;
        hidden_delta(image_index, channel_index + row_index + column_index) = sum;
    }

    hidden_delta.device(*thread_pool_device) = hidden_delta * activations_derivatives;
}


void ConvolutionalLayer::calculate_error_gradient(type* inputs_data,
                                                  LayerForwardPropagation* forward_propagation,
                                                  LayerBackPropagation* back_propagation) const
{   
/*
    Tensor<type, 4> layers_inputs;

    switch(convolution_type) {

        case opennn::ConvolutionalLayer::ConvolutionType::Valid:
        {
            layers_inputs = inputs;
        }
        break;

        case opennn::ConvolutionalLayer::ConvolutionType::Same:
        {
            layers_inputs = inputs;

//            layers_inputs.resize(inputs.dimension(0) + get_padding_height(),
//                                 inputs.dimension(1) + get_padding_width(),
//                                 inputs.dimension(2),
//                                 inputs.dimension(3));

//            for(Index image_number = 0; image_number < inputs.dimension(0); image_number++)
//            {
//                    layers_inputs.set_tensor(image_number, insert_padding(previous_layers_outputs.get_tensor(image_number)));
//            }
        }
        break;
    }

    const Index images_number = inputs.dimension(0);
    const Index kernels_number = get_kernels_number();

    const Index kernel_channels_number = get_kernels_channels_number();

    const Index kernels_rows_number = get_kernels_rows_number();
    const Index kernels_columns_number = get_kernels_columns_number();

    const Eigen::array<ptrdiff_t, 3> dims = {0, 1, 2};

    Index image_index = 0;
    Index kernel_index = 0;

    for(Index i = 0; i < images_number; i++)
    {
        image_index = i*kernels_number*(kernels_rows_number*kernels_columns_number);

        for(Index j = 0; j < kernels_number; j++)
        {
            // Extract needed hidden_delta
            // Convolve layer_inputs and hidden_delta

            kernel_index = j*kernels_rows_number*kernels_columns_number;

//            const TensorMap<Tensor<type, 3>> delta_map(back_propagation.delta.data() + image_index + kernel_index,
//            Eigen::array<Index,3>({kernel_channels_number, kernels_rows_number, kernels_columns_number}));
        }
    }
*/
}


void ConvolutionalLayer::insert_gradient(LayerBackPropagation* back_propagation, const Index& index, Tensor<type, 1>& gradient) const
{
    ConvolutionalLayerBackPropagation* convolutional_layer_back_propagation =
            static_cast<ConvolutionalLayerBackPropagation*>(back_propagation);

    const Index biases_number = biases.size();
    const Index synaptic_weights_number = synaptic_weights.size();

    memcpy(gradient.data() + index,
           convolutional_layer_back_propagation->biases_derivatives.data(),
           static_cast<size_t>(biases_number)*sizeof(type));

    memcpy(gradient.data() + index + biases_number,
           convolutional_layer_back_propagation->synaptic_weights_derivatives.data(),
           static_cast<size_t>(synaptic_weights_number)*sizeof(type));
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
    return (input_variables_dimensions(0) - get_kernels_rows_number() + 1);

    ///@todo padding

//    const Index kernels_rows_number = get_kernels_rows_number();

//    const Index padding_height = get_padding_height();

//    return ((input_variables_dimensions(2) - kernels_rows_number + 2 * padding_height)/row_stride) + 1;
}


/// Returns the number of columns the result of applying the layer's kernels to an image will have.

Index ConvolutionalLayer::get_outputs_columns_number() const
{
    return (input_variables_dimensions(1) - get_kernels_columns_number()+1);

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
        return column_stride*(input_variables_dimensions[2] - 1) - input_variables_dimensions[2] + get_kernels_columns_number();
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
        return row_stride*(input_variables_dimensions[1] - 1) - input_variables_dimensions[1] + get_kernels_rows_number();
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
//    Tensor<type, 1> parameters = synaptic_weights.reshape(Eigen::array<Index, 1>{get_synaptic_weights_number()});
    Tensor<type, 1> parameters(get_parameters_number());

    const Index kernels_number = get_kernels_number();
    const Index kernels_channels_number = get_kernels_channels_number();
    const Index kernels_rows_number = get_kernels_rows_number();
    const Index kernels_columns_number = get_kernels_columns_number();


    Index element_index = 0;
#pragma omp for
    for(Index i = 0; i < kernels_number; i++)
    {
        for(Index j = 0; j < kernels_channels_number; j++)
        {
            for(Index k = 0; k < kernels_rows_number; k++)
            {
                for(Index l = 0; l < kernels_columns_number; l++)
                {
                    parameters(element_index + kernels_number) = synaptic_weights(i ,j, k, l);
                    element_index ++;
                }
            }
        }
    }

    for(int i = 0; i < biases.size(); i++)
    {
        parameters(i) = biases(i);
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
    const Index kernels_channels_number = new_inputs_dimensions[2];
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

//        input_variables_dimensions.set(new_inputs_dimensions);

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

    synaptic_weights.resize(kernels_number, kernels_channels_number, kernels_rows_number, kernels_columns_number);
    biases.resize(kernels_number);

    memcpy(biases.data(),
           new_parameters.data(),
           static_cast<size_t>(kernels_number)*sizeof(type));

    Index element_index = kernels_number;

#pragma omp for
    for(Index i = 0; i < kernels_number; i++)
    {
        for(Index j = 0; j < kernels_channels_number; j++)
        {
            for(Index k = 0; k < kernels_rows_number; k++)
            {
                for(Index l = 0; l < kernels_columns_number; l++)
                {
                    synaptic_weights(i ,j, k, l) = new_parameters(element_index);
                    element_index ++;
                }
            }
        }
    }
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

/// Returns the dimensions of the input.

Tensor<Index, 1> ConvolutionalLayer::get_input_variables_dimenisons() const
{
    return input_variables_dimensions;
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

    // Convolutional type

    file_stream.OpenElement("ConvolutionType");

    file_stream.PushText(write_convolution_type().c_str());

    file_stream.CloseElement();

    // Input variables dimensions

    file_stream.OpenElement("InputDimensions");

    buffer.str("");
    buffer << get_input_variables_dimensions();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Column stride

    file_stream.OpenElement("ColumnStride");

    buffer.str("");
    buffer << get_column_stride();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Row stride

    file_stream.OpenElement("RowStride");

    buffer.str("");
    buffer << get_row_stride();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Biases

    file_stream.OpenElement("Biases");

    buffer.str("");
    buffer << get_biases();

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

    // Convolution type element

    const tinyxml2::XMLElement* convolution_type_element = convolutional_layer_element->FirstChildElement("ConvolutionType");

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

    const tinyxml2::XMLElement* input_variables_dimensions_element = convolutional_layer_element->FirstChildElement("InputDimensions");

    if(!input_variables_dimensions_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional input variables dimensions element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const string input_variables_dimensions_string = input_variables_dimensions_element->GetText();

//    set_input_variables_dimenisons(input_variables_dimensions_string);

    // Column stride

    const tinyxml2::XMLElement* column_stride_element = convolutional_layer_element->FirstChildElement("ColumnStride");

    if(!column_stride_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional column stride element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const string column_stride_string = column_stride_element->GetText();

    set_column_stride(static_cast<Index>(stoi(column_stride_string)));

    // Row stride

    const tinyxml2::XMLElement* row_stride_element = convolutional_layer_element->FirstChildElement("RowStride");

    if(!row_stride_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional row stride element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const string row_stride_string = row_stride_element->GetText();

    set_row_stride(static_cast<Index>(stoi(row_stride_string)));

    // Biases element

    const tinyxml2::XMLElement* biases_element = convolutional_layer_element->FirstChildElement("Biases");

    if(!biases_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Convolutional biases element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const string biases_string = biases_element->GetText();

    set_biases(to_type_vector(biases_string, ' '));

    // Activation function

    const tinyxml2::XMLElement* activation_function_element = convolutional_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
    {
        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "ActivationFunction element is nullptr.\n";

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

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
