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

namespace OpenNN
{

/// Default constructor.
/// It creates an empty ConvolutionalLayer object.

ConvolutionalLayer::ConvolutionalLayer() : Layer()
{
    layer_type = Layer::Convolutional;
}


/// Inputs' dimensions modifier constructor.
/// After setting new dimensions for the inputs, it creates and initializes a ConvolutionalLayer object
/// with a number of filters of a given size.
/// The initialization values are random values from a normal distribution.
/// @param new_inputs_dimensions A vector containing the new inputs' dimensions.
/// @param filters_dimensions A vector containing the number of filters, their rows and columns.

ConvolutionalLayer::ConvolutionalLayer(const Tensor<Index, 1>& new_inputs_dimensions,
                                       const Tensor<Index, 1>& new_filters_dimensions) : Layer()
{
    layer_type = Layer::Convolutional;

    set(new_inputs_dimensions, new_filters_dimensions);
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


void ConvolutionalLayer::insert_padding(const Tensor<type, 4>& inputs, Tensor<type, 4>& padded_output) // @todo Add stride
{
    switch(convolution_type)
    {
        case Valid: padded_output = inputs; return;

        case Same:
        {
                Eigen::array<pair<int, int>, 4> paddings;
                const int pad = int(0.5 *(get_filters_rows_number() - 1));
                paddings[0] = make_pair(0, 0);
                paddings[1] = make_pair(0, 0);
                paddings[2] = make_pair(pad, pad);
                paddings[3] = make_pair(pad, pad);
                padded_output = inputs.pad(paddings);
                return;
        }

        default: return;
    }
}


/// Calculate combinations
void ConvolutionalLayer::calculate_combinations(const Tensor<type, 4>& inputs, Tensor<type, 4> & combinations) const
{
    const Index number_of_kernels = synaptic_weights.dimension(0);
    const Index number_of_images = inputs.dimension(0);

    const Eigen::array<ptrdiff_t, 3> dims = {0, 1, 2};

    Tensor<type, 3> kernel;

    #pragma omp parallel for
    for(Index i = 0; i < number_of_images; i++)
    {
        for(Index j = 0; j < number_of_kernels; j++)
        {
            kernel = synaptic_weights.chip(j, 0);
            combinations.chip(i, 0).chip(j, 0) = inputs.chip(i, 0).convolve(kernel, dims) + biases(j);
        }
    }
}


void ConvolutionalLayer::calculate_combinations(const Tensor<type, 4>& inputs,
                                                const Tensor<type, 2>& potential_biases,
                                                const Tensor<type, 4>& potential_synaptic_weights,
                                                Tensor<type, 4>& combinations_4d) const
{
    const Index number_of_kernels = potential_synaptic_weights.dimension(0);
    const Index number_of_images = inputs.dimension(0);

    combinations_4d.resize(number_of_images, number_of_kernels, get_outputs_rows_number(), get_outputs_columns_number());

    const Eigen::array<ptrdiff_t, 3> dims = {0, 1, 2};

    Tensor<type, 3> kernel;

    #pragma omp parallel for
    for(Index i = 0; i < number_of_images; i++)
    {
        for(Index j = 0; j < number_of_kernels; j++)
        {
            kernel = potential_synaptic_weights.chip(j, 0);
            combinations_4d.chip(i, 0).chip(j, 0) = inputs.chip(i, 0).convolve(kernel, dims) + potential_biases(j, 0);
        }
    }
}



/// Calculates activations
void ConvolutionalLayer::calculate_activations(const Tensor<type, 4>& inputs, Tensor<type, 4>& activations) const
{
    switch(activation_function)
    {
        case Linear: linear(inputs, activations); return;

        case Logistic: logistic(inputs, activations); return;

        case HyperbolicTangent: hyperbolic_tangent(inputs, activations); return;

        case Threshold: threshold(inputs, activations); return;

        case SymmetricThreshold: symmetric_threshold(inputs, activations); return;

        case RectifiedLinear: rectified_linear(inputs, activations); return;

        case ScaledExponentialLinear: scaled_exponential_linear(inputs, activations); return;

        case SoftPlus: soft_plus(inputs, activations); return;

        case SoftSign: soft_sign(inputs, activations); return;

        case HardSigmoid: hard_sigmoid(inputs, activations); return;

        case ExponentialLinear: exponential_linear(inputs, activations); return;
    }
}


void ConvolutionalLayer::calculate_activations_derivatives(const Tensor<type, 4>& combinations_4d,
                                                           Tensor<type, 4>& activations,
                                                           Tensor<type, 4>& activations_derivatives) const
{
    switch(activation_function)
    {
        case Linear: linear_derivatives(combinations_4d, activations, activations_derivatives); return;

        case Logistic: logistic_derivatives(combinations_4d, activations, activations_derivatives); return;

        case HyperbolicTangent: hyperbolic_tangent_derivatives(combinations_4d, activations, activations_derivatives); return;

        case Threshold: threshold_derivatives(combinations_4d, activations, activations_derivatives); return;

        case SymmetricThreshold: symmetric_threshold_derivatives(combinations_4d, activations, activations_derivatives); return;

        case RectifiedLinear: rectified_linear_derivatives(combinations_4d, activations, activations_derivatives); return;

        case ScaledExponentialLinear: scaled_exponential_linear_derivatives(combinations_4d, activations, activations_derivatives); return;

        case SoftPlus: soft_plus_derivatives(combinations_4d, activations, activations_derivatives); return;

        case SoftSign: soft_sign_derivatives(combinations_4d, activations, activations_derivatives); return;

        case HardSigmoid: hard_sigmoid_derivatives(combinations_4d, activations, activations_derivatives); return;

        case ExponentialLinear: exponential_linear_derivatives(combinations_4d, activations, activations_derivatives); return;
    }
}


/// Returns the output of the convolutional layer applied to a batch of images.
/// @param inputs The batch of images.
/// @param outputs Tensor where the outputs will be stored.

void ConvolutionalLayer::calculate_outputs(const Tensor<type, 4>& inputs, Tensor<type, 4>& outputs)
{
    const Tensor<Index, 1> outputs_dimensions = get_outputs_dimensions();

    outputs.resize(outputs_dimensions(0), outputs_dimensions(1), outputs_dimensions(2), outputs_dimensions(3));

    Tensor<type, 4> combinations(outputs_dimensions(0), outputs_dimensions(1), outputs_dimensions(2), outputs_dimensions(3));

    calculate_combinations(inputs, combinations);
    calculate_activations(combinations, outputs);
}


//void ConvolutionalLayer::calculate_outputs(const Tensor<type, 2>& inputs, Tensor<type, 2>& outputs)
//{
//    const Tensor<Index, 1> outputs_dimensions = get_outputs_dimensions();

//    const Tensor<Index, 1> inputs_dimensions = get_input_variables_dimensions();

//    const Eigen::array<Eigen::Index, 2> shuffle_dims = {1, 0};
//    const Eigen::array<Eigen::Index, 4> inputs_dimensions_array = {inputs_dimensions(0),
//                                                                   inputs_dimensions(1),
//                                                                   inputs_dimensions(2),
//                                                                   inputs_dimensions(3)};

//    const Eigen::array<Eigen::Index, 4> outputs_dimensions_array = {outputs_dimensions(0),
//                                                                    outputs_dimensions(1),
//                                                                    outputs_dimensions(2),
//                                                                    outputs_dimensions(3)};

//    const Tensor<type, 4> inputs_temp = inputs.shuffle(shuffle_dims).reshape(inputs_dimensions_array);
//    Tensor<type, 4> outputs_temp = outputs.reshape(outputs_dimensions_array);

//    calculate_outputs(inputs_temp, outputs_temp);

//    const Eigen::array<Eigen::Index, 2> output_dims = {outputs_dimensions(0) * outputs_dimensions(1) * outputs_dimensions(2), outputs_dimensions(3)};

//    outputs = outputs_temp.reshape(output_dims).shuffle(shuffle_dims);
//}


void ConvolutionalLayer::calculate_outputs(const Tensor<type, 4>& inputs, Tensor<type, 2>& outputs)
{
    const Tensor<Index, 1> outputs_dimensions = get_outputs_dimensions();

    const Tensor<Index, 1> inputs_dimensions = get_input_variables_dimensions();

    const Eigen::array<Eigen::Index, 2> shuffle_dims = {1, 0};

    const Eigen::array<Eigen::Index, 4> outputs_dimensions_array = {outputs_dimensions(0),
                                                                    outputs_dimensions(1),
                                                                    outputs_dimensions(2),
                                                                    outputs_dimensions(3)};

    Tensor<type, 4> outputs_temp = outputs.reshape(outputs_dimensions_array);

    calculate_outputs(inputs, outputs_temp);

    const Eigen::array<Eigen::Index, 2> output_dims = {outputs_dimensions(0) * outputs_dimensions(1) * outputs_dimensions(2), outputs_dimensions(3)};

    outputs = outputs_temp.reshape(output_dims).shuffle(shuffle_dims);
}


void ConvolutionalLayer::forward_propagate(const Tensor<type, 4> &inputs, ForwardPropagation &forward_propagation) const
{
    const Tensor<Index, 1> outputs_dimensions = get_outputs_dimensions();

    forward_propagation.combinations_4d.resize(outputs_dimensions(0),
                                               outputs_dimensions(1),
                                               outputs_dimensions(2),
                                               outputs_dimensions(3));

    forward_propagation.activations_4d.resize(outputs_dimensions(0),
                                              outputs_dimensions(1),
                                              outputs_dimensions(2),
                                              outputs_dimensions(3));

    forward_propagation.activations_derivatives_4d.resize(outputs_dimensions(0),
                                                          outputs_dimensions(1),
                                                          outputs_dimensions(2),
                                                          outputs_dimensions(3));

    calculate_combinations(inputs,
                           forward_propagation.combinations_4d);

    calculate_activations_derivatives(forward_propagation.combinations_4d,
                                      forward_propagation.activations_4d,
                                      forward_propagation.activations_derivatives_4d);

    to_2d(forward_propagation.combinations_4d, forward_propagation.combinations_2d);
    to_2d(forward_propagation.activations_4d, forward_propagation.activations_2d);
    to_2d(forward_propagation.activations_derivatives_4d, forward_propagation.activations_derivatives_2d);

}


void ConvolutionalLayer::forward_propagate(const Tensor<type, 2> &inputs, ForwardPropagation &forward_propagation) const
{
    const Eigen::array<Eigen::Index, 4> four_dims = {input_variables_dimensions(3), // columns number
                                                     input_variables_dimensions(2), // rows number
                                                     input_variables_dimensions(1), // channels number
                                                     inputs.dimension(0)}; // images number
    const Eigen::array<Eigen::Index, 2> shuffle_dims_2D = {1, 0};
    const Eigen::array<Eigen::Index, 4> shuffle_dims_4D = {3, 2, 1, 0};

    const Tensor<type, 4> inputs_4d = inputs.shuffle(shuffle_dims_2D).reshape(four_dims).shuffle(shuffle_dims_4D);

    forward_propagate(inputs_4d, forward_propagation);
}


void ConvolutionalLayer::forward_propagate(const Tensor<type, 4>& inputs,
                                           Tensor<type, 1> potential_parameters,
                                           ForwardPropagation& forward_propagation) const
{
    const Tensor<Index, 1> output_dimensions = get_outputs_dimensions();

    forward_propagation.combinations_4d.resize(output_dimensions(0),
                                               output_dimensions(1),
                                               output_dimensions(2),
                                               output_dimensions(3));

    forward_propagation.activations_4d.resize(output_dimensions(0),
                                              output_dimensions(1),
                                              output_dimensions(2),
                                              output_dimensions(3));

    forward_propagation.activations_derivatives_4d.resize(output_dimensions(0),
                                                          output_dimensions(1),
                                                          output_dimensions(2),
                                                          output_dimensions(3));

    const Index filters_number = synaptic_weights.dimension(0);

    const TensorMap<Tensor<type, 2>> potential_biases(potential_parameters.data(),
                                                      filters_number,
                                                      1);

//    TensorMap<Tensor<type, 4>> potential_synaptic_weights(potential_parameters.data() + filters_number,
//                                                          synaptic_weights.dimension(3),
//                                                          synaptic_weights.dimension(2),
//                                                          synaptic_weights.dimension(1),
//                                                          filters_number);


    const Index filters_channels_number = get_filters_channels_number();
    const Index filters_rows_number = get_filters_rows_number();
    const Index filters_columns_number = get_filters_columns_number();

    Tensor<type, 4> potential_synaptic_weights(filters_number,
                                               filters_channels_number,
                                               filters_rows_number,
                                               filters_columns_number);
    Index element_index = 0;

//#pragma omp for
    for(Index i = 0; i < filters_number; i++)
    {
        for(Index j = 0; j < filters_channels_number; j++)
        {
            for(Index k = 0; k < filters_rows_number; k++)
            {
                for(Index l = 0; l < filters_columns_number; l++)
                {
                    potential_synaptic_weights(i ,j, k, l) = potential_parameters(filters_number + element_index);
                    element_index ++;
                }
            }
        }
    }

    calculate_combinations(inputs,
                           potential_biases,
                           potential_synaptic_weights,
                           forward_propagation.combinations_4d);

    calculate_activations_derivatives(forward_propagation.combinations_4d,
                                      forward_propagation.activations_4d,
                                      forward_propagation.activations_derivatives_4d);

    to_2d(forward_propagation.combinations_4d, forward_propagation.combinations_2d);
    to_2d(forward_propagation.activations_4d, forward_propagation.activations_2d);
    to_2d(forward_propagation.activations_derivatives_4d, forward_propagation.activations_derivatives_2d);
}


void ConvolutionalLayer::forward_propagate(const Tensor<type, 2>& inputs,
                                           Tensor<type, 1> potential_parameters,
                                           ForwardPropagation& forward_propagation) const
{
    const Eigen::array<Eigen::Index, 4> four_dims = {input_variables_dimensions(3), // columns number
                                                     input_variables_dimensions(2), // rows number
                                                     input_variables_dimensions(1), // channels number
                                                     inputs.dimension(0)}; // images number
    const Eigen::array<Eigen::Index, 2> shuffle_dims_2D = {1, 0};
    const Eigen::array<Eigen::Index, 4> shuffle_dims_4D = {3, 2, 1, 0};

    const Tensor<type, 4> inputs_4d = inputs.shuffle(shuffle_dims_2D).reshape(four_dims).shuffle(shuffle_dims_4D);

    forward_propagate(inputs_4d, potential_parameters, forward_propagation);
}


void ConvolutionalLayer::calculate_hidden_delta(Layer* next_layer_pointer,
                                                const Tensor<type, 2>&,
                                                ForwardPropagation& forward_propagation,
                                                const Tensor<type, 2>& next_layer_delta,
                                                Tensor<type, 2>& hidden_delta) const
{
    const Type next_layer_type = next_layer_pointer->get_type();

    if(next_layer_type == Convolutional)
    {
        ConvolutionalLayer* convolutional_layer = dynamic_cast<ConvolutionalLayer*>(next_layer_pointer);

//        calculate_hidden_delta_convolutional(convolutional_layer,
//                                             activations_2d,
//                                             activations_derivatives,
//                                             next_layer_delta,
//                                             hidden_delta);
    }
    else if(next_layer_type == Pooling)
    {
        PoolingLayer* pooling_layer = dynamic_cast<PoolingLayer*>(next_layer_pointer);

        calculate_hidden_delta_pooling(pooling_layer,
                                       forward_propagation.activations_4d,
                                       forward_propagation.activations_derivatives_4d,
                                       next_layer_delta,
                                       hidden_delta);
    }
    else if(next_layer_type == Perceptron)
    {
        PerceptronLayer* perceptron_layer = dynamic_cast<PerceptronLayer*>(next_layer_pointer);

        calculate_hidden_delta_perceptron(perceptron_layer,
                                          forward_propagation.activations_4d,
                                          forward_propagation.activations_derivatives_4d,
                                          next_layer_delta,
                                          hidden_delta);
    }
    else if(next_layer_type == Probabilistic)
    {
        ProbabilisticLayer* probabilistic_layer = dynamic_cast<ProbabilisticLayer*>(next_layer_pointer);

        calculate_hidden_delta_probabilistic(probabilistic_layer,
                                             forward_propagation.activations_4d,
                                             forward_propagation.activations_derivatives_4d,
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

    const Index filters_number = get_filters_number();
    const Index output_rows_number = get_outputs_rows_number();
    const Index output_columns_number = get_outputs_columns_number();

    // Next layer's values

    const Index next_layers_filters_number = next_layer_pointer->get_filters_number();
    const Index next_layers_filter_rows = next_layer_pointer->get_filters_rows_number();
    const Index next_layers_filter_columns = next_layer_pointer->get_filters_columns_number();

    const Index next_layers_output_rows = next_layer_pointer->get_outputs_rows_number();
    const Index next_layers_output_columns = next_layer_pointer->get_outputs_columns_number();

    const Index next_layers_row_stride = next_layer_pointer->get_row_stride();
    const Index next_layers_column_stride = next_layer_pointer->get_column_stride();

    const Tensor<type, 4> next_layers_weights = next_layer_pointer->get_synaptic_weights();

    // Hidden delta calculation

//    hidden_delta.resize(images_number, filters_number, output_rows_number, output_columns_number);
    hidden_delta.resize(images_number, filters_number * output_rows_number * output_columns_number);

    const Index size = hidden_delta.size();

    #pragma omp parallel for

    for(Index tensor_index = 0; tensor_index < size; tensor_index++)
    {
        const Index image_index = tensor_index / (filters_number * output_rows_number * output_columns_number);
        const Index channel_index = (tensor_index / (output_rows_number * output_columns_number)) % filters_number;
        const Index row_index = (tensor_index / output_columns_number) % output_rows_number;
        const Index column_index = tensor_index % output_columns_number;

        type sum = 0;

        const Index lower_row_index = (row_index - next_layers_filter_rows) / next_layers_row_stride + 1;
        const Index upper_row_index = min(row_index/next_layers_row_stride + 1, next_layers_output_rows);
        const Index lower_column_index = (column_index - next_layers_filter_columns) / next_layers_column_stride + 1;
        const Index upper_column_index = min(column_index / next_layers_column_stride + 1, next_layers_output_columns);

        for(Index i = 0; i < next_layers_filters_number; i++)
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
                                                        const Tensor<type, 4>& activations_2d,
                                                        const Tensor<type, 4>& activations_derivatives,
                                                        const Tensor<type, 2>& next_layer_delta,
                                                        Tensor<type, 2>& hidden_delta) const
{

        switch(next_layer_pointer->get_pooling_method())
        {
            case OpenNN::PoolingLayer::PoolingMethod::NoPooling:
            {
//                return next_layer_delta;
            }

            case OpenNN::PoolingLayer::PoolingMethod::AveragePooling:
            {
                // Current layer's values

                const Index images_number = next_layer_delta.dimension(0);
                const Index filters_number = get_filters_number();
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

                hidden_delta.resize(images_number, filters_number * output_rows_number * output_columns_number);

                const Index size = hidden_delta.size();

                #pragma omp parallel for

                for(Index tensor_index = 0; tensor_index < size; tensor_index++)
                {
                    const Index image_index = tensor_index/(filters_number*output_rows_number*output_columns_number);
                    const Index channel_index = (tensor_index/(output_rows_number*output_columns_number))%filters_number;
                    const Index row_index = (tensor_index/output_columns_number)%output_rows_number;
                    const Index column_index = tensor_index%output_columns_number;

                    type sum = 0;

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
            }

            case OpenNN::PoolingLayer::PoolingMethod::MaxPooling:
            {
                // Current layer's values

                const Index images_number = next_layer_delta.dimension(0);
                const Index filters_number = get_filters_number();
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

                hidden_delta.resize(images_number, filters_number * output_rows_number * output_columns_number);

                const Index size = hidden_delta.size();

                #pragma omp parallel for

                for(Index tensor_index = 0; tensor_index < size; tensor_index++)
                {
                    const Index image_index = tensor_index/(filters_number*output_rows_number*output_columns_number);
                    const Index channel_index = (tensor_index/(output_rows_number*output_columns_number))%filters_number;
                    const Index row_index = (tensor_index/output_columns_number)%output_rows_number;
                    const Index column_index = tensor_index%output_columns_number;

                    type sum = 0;

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
//                                            activations_2d(image_index, channel_index, i*next_layers_row_stride + submatrix_row_index, j*next_layers_column_stride + submatrix_column_index);
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
                                        multiply_not_multiply(submatrix_row_index, submatrix_column_index) = 1.0;
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
            }
        }
}


void ConvolutionalLayer::calculate_hidden_delta_perceptron(const PerceptronLayer* next_layer_pointer,
                                                           const Tensor<type, 4>&,
                                                           const Tensor<type, 4>& activations_derivatives,
                                                           const Tensor<type, 2>& next_layer_delta,
                                                           Tensor<type, 2>& hidden_delta) const
{

        // Current layer's values

        const Index images_number = next_layer_delta.dimension(0);
        const Index filters_number = get_filters_number();
        const Index output_rows_number = get_outputs_rows_number();
        const Index output_columns_number = get_outputs_columns_number();

        // Next layer's values

        const Index next_layers_output_columns = next_layer_delta.dimension(1);

        const Tensor<type, 2>& next_layers_weights = next_layer_pointer->get_synaptic_weights();

        // Hidden delta calculation

        hidden_delta.resize(images_number, filters_number * output_rows_number * output_columns_number);

        const Index size = hidden_delta.size();

        #pragma omp parallel for

        for(Index tensor_index = 0; tensor_index < size; tensor_index++)
        {
            const Index image_index = tensor_index/(filters_number*output_rows_number*output_columns_number);
            const Index channel_index = (tensor_index/(output_rows_number*output_columns_number))%filters_number;
            const Index row_index = (tensor_index/output_columns_number)%output_rows_number;
            const Index column_index = tensor_index%output_columns_number;

            type sum = 0;

            for(Index sum_index = 0; sum_index < next_layers_output_columns; sum_index++)
            {
                const type delta_element = next_layer_delta(image_index, sum_index);

                const type weight = next_layers_weights(channel_index + row_index*filters_number + column_index*filters_number*output_rows_number, sum_index);

                sum += delta_element*weight;
            }
//            hidden_delta(row_index, column_index, channel_index, image_index) = sum;
            hidden_delta(row_index, column_index + channel_index + image_index) = sum;
        }
//        return hidden_delta*activations_derivatives;
}


void ConvolutionalLayer::calculate_hidden_delta_probabilistic(ProbabilisticLayer* next_layer_pointer,
                                                              const Tensor<type, 4>& activations,
                                                              const Tensor<type, 4>& activations_derivatives,
                                                              const Tensor<type, 2>& next_layer_delta,
                                                              Tensor<type, 2>& hidden_delta) const
{

    // Current layer's values
    const Index images_number = next_layer_delta.dimension(0);
    const Index filters_number = get_filters_number();
    const Index output_rows_number = get_outputs_rows_number();
    const Index output_columns_number = get_outputs_columns_number();

    // Next layer's values

    const Index next_layers_output_columns = next_layer_delta.dimension(1);

    const Tensor<type, 2> next_layers_weights = next_layer_pointer->get_synaptic_weights();

    // Hidden delta calculation

    hidden_delta.resize(images_number, filters_number * output_rows_number * output_columns_number);

    const Index size = hidden_delta.size(); // Number of total parameters

    #pragma omp parallel for

    for(Index tensor_index = 0; tensor_index < size; tensor_index++)
    {
//        const Index image_index = tensor_index / (filters_number * output_rows_number * output_columns_number);

//        const Index channel_index = (tensor_index / (output_rows_number * output_columns_number)) % filters_number;
//        const Index row_index = (tensor_index / output_columns_number) % output_rows_number;
//        const Index column_index = tensor_index % output_columns_number;

        const Index image_index = tensor_index / (filters_number * output_rows_number * output_columns_number);

        const Index channel_index = (tensor_index / (output_rows_number * output_columns_number)) % filters_number;
        const Index row_index = (tensor_index / output_columns_number) % output_rows_number;
        const Index column_index = tensor_index % output_columns_number;

        type sum = 0;

        for(Index sum_index = 0; sum_index < next_layers_output_columns; sum_index++)
        {
//            const type delta_element = next_layer_delta(image_index, sum_index);

//            const type weight = next_layers_weights(sum_index,
//                                                    channel_index + (row_index * filters_number) + (column_index * filters_number * output_rows_number));

            const type delta_element = next_layer_delta(image_index, sum_index);


            const type weight = next_layers_weights(channel_index + row_index + column_index,
                                                    sum_index);

//            sum += delta_element * weight;
            sum += 0;
        }

//        hidden_delta(image_index, channel_index, row_index, column_index) = sum;
        hidden_delta(image_index, channel_index + row_index + column_index) = sum;
    }

    hidden_delta.device(*thread_pool_device) = hidden_delta * activations_derivatives;
}


void ConvolutionalLayer::calculate_error_gradient(const Tensor<type, 4>& previous_layers_outputs,
                                                  const Layer::ForwardPropagation& forward_propagation,
                                                  const Tensor<type, 4>& layer_deltas,
                                                  Tensor<type, 1>& layer_error_gradient)
{
        Tensor<type, 4> layers_inputs;

        switch(convolution_type) {

            case OpenNN::ConvolutionalLayer::ConvolutionType::Valid:
            {
                layers_inputs = previous_layers_outputs;
            }
            break;

            case OpenNN::ConvolutionalLayer::ConvolutionType::Same:
            {
                layers_inputs.resize(previous_layers_outputs.dimension(0) + get_padding_height(),
                                     previous_layers_outputs.dimension(1) + get_padding_width(),
                                     previous_layers_outputs.dimension(2),
                                     previous_layers_outputs.dimension(3));

                for(Index image_number = 0; image_number < previous_layers_outputs.dimension(0); image_number++)
                {
//                    layers_inputs.set_tensor(image_number, insert_padding(previous_layers_outputs.get_tensor(image_number)));
                }
            }
            break;
        }

        // Gradient declaration and values used

        const Index parameters_number = get_parameters_number();

        layer_error_gradient.resize(parameters_number);
        layer_error_gradient.setZero();

        const Index images_number = layer_deltas.dimension(0);
        const Index filters_number = get_filters_number();
        const Index filters_channels_number = get_filters_channels_number();
        const Index filters_rows_number = get_filters_rows_number();
        const Index filters_columns_number = get_filters_columns_number();
        const Index output_rows_number = get_outputs_rows_number();
        const Index output_columns_number = get_outputs_columns_number();

        // Synaptic weights

        const Index synaptic_weights_number = get_synaptic_weights_number();

        for(Index gradient_index = 0; gradient_index < synaptic_weights_number; gradient_index++)
        {

            Index filter_index = gradient_index % filters_number;
            Index channel_index = (gradient_index / filters_number) % filters_channels_number;
            Index row_index = (gradient_index / (filters_number * filters_channels_number)) % filters_rows_number;
            Index column_index = (gradient_index / (filters_number * filters_channels_number * filters_rows_number)) % filters_columns_number;

            type sum = 0;

            for(Index i = 0; i < output_rows_number; i++)
            {
                for(Index j = 0; j < output_columns_number; j++)
                {
                    for(Index k = 0; k < images_number; k++)
                    {
                        const type delta_element = layer_deltas(i, filter_index, j, k);

                        const type input_element = layers_inputs(i * row_stride + row_index, j * column_stride + column_index, channel_index, k);

                        sum += delta_element * input_element;
                    }
                }
            }

            layer_error_gradient(gradient_index) = sum;
        }

        // Biases

        for(Index gradient_index = synaptic_weights_number; gradient_index < parameters_number; gradient_index++) // Start after the synaptic weights
        {
            Index bias_index = gradient_index - synaptic_weights_number;  // Increment 1

            type sum = 0;

            for(Index i = 0; i < images_number; i++)
            {
                for(Index j = 0; j < output_rows_number; j++)
                {
                    for(Index k = 0; k < output_columns_number; k++)
                    {
                        sum += layer_deltas(i, bias_index, j, k);
                    }
                }
            }

            layer_error_gradient(gradient_index) = sum;
        }
}


void ConvolutionalLayer::calculate_error_gradient(const Tensor<type, 4>& inputs,
                                                  const Layer::ForwardPropagation& forward_propagation,
                                                  Layer::BackPropagation& back_propagation) const
{
    Tensor<type, 4> layers_inputs;

    switch(convolution_type) {

        case OpenNN::ConvolutionalLayer::ConvolutionType::Valid:
        {
            layers_inputs = inputs;
        }
        break;

        case OpenNN::ConvolutionalLayer::ConvolutionType::Same:
        {
            layers_inputs.resize(inputs.dimension(0) + get_padding_height(),
                                 inputs.dimension(1) + get_padding_width(),
                                 inputs.dimension(2),
                                 inputs.dimension(3));

            for(Index image_number = 0; image_number < inputs.dimension(0); image_number++)
            {
//                    layers_inputs.set_tensor(image_number, insert_padding(previous_layers_outputs.get_tensor(image_number)));
            }
        }
        break;
    }

    // Gradient declaration and values used

    const Index parameters_number = get_parameters_number();

    back_propagation.synaptic_weights_derivatives.resize(1, parameters_number);
    back_propagation.synaptic_weights_derivatives.setZero();

    const Index images_number = back_propagation.delta.dimension(0);
    const Index filters_number = get_filters_number();
    const Index filters_channels_number = get_filters_channels_number();
    const Index filters_rows_number = get_filters_rows_number();
    const Index filters_columns_number = get_filters_columns_number();
    const Index output_rows_number = get_outputs_rows_number();
    const Index output_columns_number = get_outputs_columns_number();

    // Synaptic weights

    const Index synaptic_weights_number = get_synaptic_weights_number();

    for(Index gradient_index = 0; gradient_index < synaptic_weights_number; gradient_index++)
    {

        Index filter_index = gradient_index % filters_number;
        Index channel_index = (gradient_index / filters_number) % filters_channels_number;
        Index row_index = (gradient_index / (filters_number * filters_channels_number)) % filters_rows_number;
        Index column_index = (gradient_index / (filters_number * filters_channels_number * filters_rows_number)) % filters_columns_number;

        type sum = 0;

        for(Index i = 0; i < output_rows_number; i++)
        {
            for(Index j = 0; j < output_columns_number; j++)
            {
                for(Index k = 0; k < images_number; k++)
                {
//                    const type delta_element = back_propagation.delta_4d(i, filter_index, j, k);
                    const type delta_element = 0;

//                    const type input_element = layers_inputs(i * row_stride + row_index, j * column_stride + column_index, channel_index, k);
                    const type input_element = 0;

                    sum += delta_element * input_element;
                }
            }
        }

        back_propagation.synaptic_weights_derivatives(gradient_index) = sum;
    }

    // Biases

    for(Index gradient_index = synaptic_weights_number; gradient_index < parameters_number; gradient_index++) // Start after the synaptic weights
    {
        Index bias_index = gradient_index - synaptic_weights_number;  // Increment 1

        type sum = 0;

        for(Index i = 0; i < images_number; i++)
        {
            for(Index j = 0; j < output_rows_number; j++)
            {
                for(Index k = 0; k < output_columns_number; k++)
                {
//                    sum += back_propagation.delta_4d(i, bias_index, j, k);
                    sum += 0;
                }
            }
        }

        back_propagation.synaptic_weights_derivatives(gradient_index) = sum;
    }
}

void ConvolutionalLayer::calculate_error_gradient(const Tensor<type, 2>& inputs,
                                                  const Layer::ForwardPropagation& forward_propagation,
                                                  Layer::BackPropagation& back_propagation) const
{
    const Eigen::array<Eigen::Index, 4> four_dims = {input_variables_dimensions(3), // columns number
                                                     input_variables_dimensions(2), // rows number
                                                     input_variables_dimensions(1), // channels number
                                                     inputs.dimension(0)}; // images number
    const Eigen::array<Eigen::Index, 2> shuffle_dims_2D = {1, 0};
    const Eigen::array<Eigen::Index, 4> shuffle_dims_4D = {3, 2, 1, 0};

    const Tensor<type, 4> inputs_4d = inputs.shuffle(shuffle_dims_2D).reshape(four_dims).shuffle(shuffle_dims_4D);

    calculate_error_gradient(inputs_4d, forward_propagation, back_propagation);
}


void ConvolutionalLayer::insert_gradient(const BackPropagation& back_propagation, const Index& index, Tensor<type, 1>& gradient) const
{
    const Index biases_number = biases.size();
    const Index synaptic_weights_number = synaptic_weights.size();

    memcpy(gradient.data() + index,
           back_propagation.biases_derivatives.data(),
           static_cast<size_t>(biases_number)*sizeof(type));

    memcpy(gradient.data() + index + biases_number,
           back_propagation.synaptic_weights_derivatives.data(),
           static_cast<size_t>(synaptic_weights_number)*sizeof(type));
}


/// Returns the convolutional layer's activation function.

ConvolutionalLayer::ActivationFunction ConvolutionalLayer::get_activation_function() const
{
    return activation_function;
}


/// Returns the number of rows the result of applying the layer's filters to an image will have.

Index ConvolutionalLayer::get_outputs_rows_number() const
{
    const Index filters_rows_number = get_filters_rows_number();

    const Index padding_height = get_padding_height();

    return ((input_variables_dimensions(2) - filters_rows_number + 2 * padding_height)/row_stride) + 1;
}


/// Returns the number of columns the result of applying the layer's filters to an image will have.

Index ConvolutionalLayer::get_outputs_columns_number() const
{
    const Index filters_columns_number = get_filters_columns_number();

    const Index padding_width = get_padding_width();

    return ((input_variables_dimensions(3) - filters_columns_number + 2 * padding_width)/column_stride) + 1;
}


/// Returns a vector containing the number of channels, rows and columns of the result of applying the layer's filters to an image.

Tensor<Index, 1> ConvolutionalLayer::get_outputs_dimensions() const
{
    Tensor<Index, 1> outputs_dimensions(4);

    outputs_dimensions(0) = input_variables_dimensions(0); // Number of images
    outputs_dimensions(1) = get_filters_number();
    outputs_dimensions(2) = get_outputs_rows_number();
    outputs_dimensions(3) = get_outputs_columns_number();

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


///Returns the number of filters of the layer.

Index ConvolutionalLayer::get_filters_number() const
{
    return synaptic_weights.dimension(0);
}


/// Returns the number of channels of the layer's filters.

Index ConvolutionalLayer::get_filters_channels_number() const
{
    return synaptic_weights.dimension(1);
}


/// Returns the number of rows of the layer's filters.

Index  ConvolutionalLayer::get_filters_rows_number() const
{
    return synaptic_weights.dimension(2);
}


/// Returns the number of columns of the layer's filters.

Index ConvolutionalLayer::get_filters_columns_number() const
{
    return synaptic_weights.dimension(3);
}


/// Returns the total number of columns of zeroes to be added to an image before applying a filter, which depends on the padding option set.

Index ConvolutionalLayer::get_padding_width() const
{
    switch(convolution_type)
    {
    case Valid:
    {
        return 0;
    }

    case Same:
    {
        return column_stride*(input_variables_dimensions[2] - 1) - input_variables_dimensions[2] + get_filters_columns_number();
    }
    }

    return 0;
}


/// Returns the total number of rows of zeroes to be added to an image before applying a filter, which depends on the padding option set.

Index ConvolutionalLayer::get_padding_height() const
{
    switch(convolution_type)
    {
    case Valid:
    {
        return 0;
    }

    case Same:
    {
        return row_stride*(input_variables_dimensions[1] - 1) - input_variables_dimensions[1] + get_filters_rows_number();
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
//    return get_filters_number() * get_outputs_rows_number() * get_outputs_columns_number();
    return get_filters_number();
}


/// Returns the layer's parameters in the form of a vector.

Tensor<type, 1> ConvolutionalLayer::get_parameters() const
{
//    Tensor<type, 1> parameters = synaptic_weights.reshape(Eigen::array<Index, 1>{get_synaptic_weights_number()});
    Tensor<type, 1> parameters(get_parameters_number());

    const Index filters_number = get_filters_number();
    const Index filters_channels_number = get_filters_channels_number();
    const Index filters_rows_number = get_filters_rows_number();
    const Index filters_columns_number = get_filters_columns_number();


    Index element_index = 0;
#pragma omp for
    for(Index i = 0; i < filters_number; i++)
    {
        for(Index j = 0; j < filters_channels_number; j++)
        {
            for(Index k = 0; k < filters_rows_number; k++)
            {
                for(Index l = 0; l < filters_columns_number; l++)
                {
                    parameters(element_index + filters_number) = synaptic_weights(i ,j, k, l);
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
/// @param new_inputs_dimensions A vector containing the desired inputs' dimensions (rows number, columns number, number of channels, number of images).
/// @param new_filters_dimensions A vector containing the desired filters' dimensions (rows number, columns number, number of channels, number of filters).

void ConvolutionalLayer::set(const Tensor<Index, 1>& new_inputs_dimensions, const Tensor<Index, 1>& new_filters_dimensions)
{
#ifdef __OPENNN_DEBUG__

    const Index inputs_dimensions_number = new_inputs_dimensions.size();

    if(inputs_dimensions_number != 4)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "ConvolutionalLayer(const Tensor<Index, 1>&) constructor.\n"
               << "Number of inputs dimensions (" << inputs_dimensions_number << ") must be 4 (number of images, channels, rows, columns).\n";

        throw logic_error(buffer.str());
    }

#endif

#ifdef __OPENNN_DEBUG__

    const Index filters_dimensions_number = new_filters_dimensions.size();

    if(filters_dimensions_number != 4)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void set(const Tensor<Index, 1>&) method.\n"
               << "Number of filters dimensions (" << filters_dimensions_number << ") must be 4 (number of images, filters, rows, columns).\n";

        throw logic_error(buffer.str());
    }

#endif

    const Index filters_number = new_filters_dimensions[0];
    const Index filters_channels_number = new_inputs_dimensions[1];
    const Index filters_rows_number = new_filters_dimensions[2];
    const Index filters_columns_number = new_filters_dimensions[3];

    biases.resize(filters_number);
    biases.setRandom();


    synaptic_weights.resize(filters_number, filters_channels_number, filters_rows_number, filters_columns_number);
    synaptic_weights.setRandom();

    input_variables_dimensions = new_inputs_dimensions;
}


/// Sets and initializes the layer's parameters in accordance with the dimensions taken as input.
/// The initialization values are random values from a normal distribution.
/// @param new_inputs Layer inputs.
/// @param new_filters Layer synaptic weights.
/// @param new_biases Layer biases.

void ConvolutionalLayer::set(const Tensor<type, 4>& new_inputs, const Tensor<type, 4>& new_kernels, const Tensor<type, 1>& new_biases)
{
#ifdef __OPENNN_DEBUG__

    if(new_kernels.dimension(3) != new_biases.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void set(const Tensor<type, 4>& , const Tensor<type, 4>& , const Tensor<type, 1>& ) method.\n"
               << "Biases size must be equal to number of filters.\n";

        throw logic_error(buffer.str());
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


// Sets the parameters to random numbers using Eigen's setRandom.

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


/// Sets the filters' row stride.
/// @param new_stride_row The desired row stride.

void ConvolutionalLayer::set_row_stride(const Index& new_stride_row)
{
    if(new_stride_row <= 0)
    {
        throw ("EXCEPTION: new_stride_row must be a positive number");
    }

    row_stride = new_stride_row;
}


/// Sets the filters' column stride.
/// @param new_stride_row The desired column stride.

void ConvolutionalLayer::set_column_stride(const Index& new_stride_column)
{
    if(new_stride_column <= 0)
    {
        throw ("EXCEPTION: new_stride_column must be a positive number");
    }

    column_stride = new_stride_column;
}


/// Sets the synaptic weights and biases to the given values.
/// @param new_parameters A vector containing the synaptic weights and biases, in this order.

void ConvolutionalLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& )
{
    const Index filters_number = get_filters_number();
    const Index filters_channels_number = get_filters_channels_number();
    const Index filters_rows_number = get_filters_rows_number();
    const Index filters_columns_number = get_filters_columns_number();

    synaptic_weights.resize(filters_number, filters_channels_number, filters_rows_number, filters_columns_number);
    biases.resize(filters_number);

    const Index synaptic_weights_number = synaptic_weights.size();

    Index element_index = 0;
#pragma omp for
    for(Index i = 0; i < filters_number; i++)
    {
        for(Index j = 0; j < filters_channels_number; j++)
        {
            for(Index k = 0; k < filters_rows_number; k++)
            {
                for(Index l = 0; l < filters_columns_number; l++)
                {
                    synaptic_weights(i ,j, k, l) = new_parameters(filters_number + element_index);
                    element_index ++;
                }
            }
        }
    }

    for(Index i = 0; i < filters_number; i ++)
    {
        biases(i) = new_parameters(i);
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


/// Returns the number of layer's synaptic weights

Index ConvolutionalLayer::get_synaptic_weights_number() const
{
    return synaptic_weights.size();
}


/// Returns the number of channels of the input.

Index ConvolutionalLayer::get_inputs_channels_number() const
{
    return input_variables_dimensions[1];
}


/// Returns the number of rows of the input.

Index ConvolutionalLayer::get_inputs_rows_number() const
{
    return input_variables_dimensions[2];
}


/// Returns the number of columns of the input.

Index ConvolutionalLayer::get_inputs_columns_number() const
{
    return input_variables_dimensions[3];
}


void ConvolutionalLayer::to_2d(const Tensor<type, 4>& inputs_4d, Tensor<type, 2> output_2d) const
{

    output_2d.resize(inputs_4d.dimension(0),
                     inputs_4d.dimension(1) * inputs_4d.dimension(2) * inputs_4d.dimension(3));

    Index element_index = 0;

#pragma omp for
    for(Index i = 0; i < inputs_4d.dimension(0); i++)
    {
        for(Index j = 0; j < inputs_4d.dimension(1); j++)
        {
            for(Index k = 0; k < inputs_4d.dimension(2); k++)
            {
                for(Index l = 0; l < inputs_4d.dimension(3); l++)
                {
                    output_2d(element_index) = inputs_4d(i ,j, k, l);
                    element_index ++;
                }
            }
        }
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
