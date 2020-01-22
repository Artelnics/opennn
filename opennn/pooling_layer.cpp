//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pooling_layer.h"

namespace OpenNN {

/// Default constructor.
/// It creates an empty PoolingLayer object.

PoolingLayer::PoolingLayer() : Layer()
{
    set_default();
}

/// Input size setter constructor.
/// After setting new dimensions for the input, it creates an empty PoolingLayer object.
/// @param new_input_variables_dimensions A vector containing the new number of channels, rows and columns for the input.

PoolingLayer::PoolingLayer(const VectorXi& new_input_variables_dimensions) : Layer()
{
    input_variables_dimensions = new_input_variables_dimensions;

    set_default();
}


/// Input size setter constructor.
/// After setting new dimensions for the input, it creates an empty PoolingLayer object.
/// @param new_input_variables_dimensions A vector containing the desired number of rows and columns for the input.
/// @param pool_dimensions A vector containing the desired number of rows and columns for the pool.

PoolingLayer::PoolingLayer(const VectorXi& new_input_variables_dimensions, const VectorXi& pool_dimensions) : Layer()
{
    input_variables_dimensions = new_input_variables_dimensions;

    pool_rows_number = pool_dimensions[0];

    pool_columns_number = pool_dimensions[1];

    set_default();
}


/// Destructor

PoolingLayer::~PoolingLayer()
{
}


/// Returns the output of the pooling layer applied to a batch of images.
/// @param inputs The batch of images.

Tensor<type, 2> PoolingLayer::calculate_outputs(const Tensor<type, 2>& inputs)
{
    #ifdef __OPENNN_DEBUG__

        const int input_variables_dimensions_number = inputs.rank();

        if(input_variables_dimensions_number != 4)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
                  << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) method.\n"
                  << "Number of inputs dimensions (" << input_variables_dimensions_number << ") must be 4 (batch, filters, rows, columns).\n";

            throw logic_error(buffer.str());
        }

    #endif

    switch(pooling_method)
    {
        case NoPooling:
        {
            return calculate_no_pooling_outputs(inputs);
        }
        case MaxPooling:
        {
            return calculate_max_pooling_outputs(inputs);
        }
        case AveragePooling:
        {
            return calculate_average_pooling_outputs(inputs);
        }
    }

    return Tensor<type, 2>();
}


/// Returns the result of applying average pooling to a batch of images.
/// @param inputs The batch of images.

Tensor<type, 2> PoolingLayer::calculate_average_pooling_outputs(const Tensor<type, 2>& inputs) const
{
/*
    const int images_number = inputs.dimension(0);

    const int channels_number = inputs.dimension(1);

    const int inputs_rows_number = inputs.dimension(2);

    const int inputs_columns_number = inputs.dimension(3);

    const int outputs_rows_number = (inputs_rows_number - pool_rows_number)/row_stride + 1;

    const int outputs_columns_number = (inputs_columns_number - pool_columns_number)/column_stride + 1;

    Tensor<type, 2> outputs(images_number, channels_number, outputs_rows_number, outputs_columns_number);

    for(int image_index = 0; image_index < images_number; image_index ++)
    {
        for(int channel_index = 0; channel_index < channels_number; channel_index ++)
        {
            for(int row_index = 0; row_index < outputs_rows_number; row_index ++)
            {
                for(int column_index = 0; column_index < outputs_columns_number; column_index ++)
                {
                    outputs(image_index, channel_index, row_index, column_index) = 0;

                    for(int window_row = 0; window_row < pool_rows_number; window_row ++)
                    {
                        const int row = row_index*row_stride + window_row;

                        for(int window_column = 0; window_column < pool_columns_number; window_column ++)
                        {
                            const int column = column_index*column_stride + window_column;

                            outputs(image_index, channel_index, row_index, column_index) += inputs(image_index, channel_index, row, column);
                        }
                    }

                    outputs(image_index, channel_index, row_index, column_index) /= pool_rows_number*pool_columns_number;
                }
            }
        }
    }

    return outputs;
*/
    return Tensor<type, 2>();
}


/// Returns the result of applying no pooling to a batch of images.
/// @param inputs The batch of images.

Tensor<type, 2> PoolingLayer::calculate_no_pooling_outputs(const Tensor<type, 2>& inputs) const
{
    return inputs;
}


/// Returns the result of applying max pooling to a batch of images.
/// @param inputs The batch of images.

Tensor<type, 2> PoolingLayer::calculate_max_pooling_outputs(const Tensor<type, 2>& inputs) const
{
/*
    const int images_number = inputs.dimension(0);

    const int channels_number = inputs.dimension(1);

    const int inputs_rows_number = inputs.dimension(2);

    const int inputs_columns_number = inputs.dimension(3);

    const int outputs_rows_number = (inputs_rows_number - pool_rows_number)/row_stride + 1;

    const int outputs_columns_number = (inputs_columns_number - pool_columns_number)/column_stride + 1;

    Tensor<type, 2> outputs(images_number, channels_number, outputs_rows_number, outputs_columns_number);

    for(int image_index = 0; image_index < images_number; image_index ++)
    {
        for(int channel_index = 0; channel_index < channels_number; channel_index ++)
        {
            for(int row_index = 0; row_index < outputs_rows_number; row_index ++)
            {
                for(int column_index = 0; column_index < outputs_columns_number; column_index ++)
                {
                    outputs(image_index, channel_index, row_index, column_index) =
                            inputs(image_index, channel_index, row_index*row_stride, column_index*column_stride);

                    for(int window_row = 0; window_row < pool_rows_number; window_row ++)
                    {
                        const int row = row_index*row_stride + window_row;

                        for(int window_column = 0; window_column < pool_columns_number; window_column ++)
                        {
                            const int column = column_index*column_stride + window_column;

                            if(inputs(image_index, channel_index, row, column) > outputs(image_index, channel_index, row_index, column_index))
                            {
                                outputs(image_index, channel_index, row_index, column_index) = inputs(image_index, channel_index, row, column);
                            }

                        }
                    }

                }
            }
        }
    }

    return outputs;
*/
    return Tensor<type, 2>();

}


Layer::ForwardPropagation PoolingLayer::calculate_forward_propagation(const Tensor<type, 2>& inputs)
{
    ForwardPropagation layers;

    layers.activations = calculate_outputs(inputs);

    layers.activations_derivatives = calculate_activations_derivatives(layers.activations);

    return layers;
}


/// Returns the result of applying the derivative of the previously set activation method to a batch of images.
/// @param inputs The batch of images.

Tensor<type, 2> PoolingLayer::calculate_activations_derivatives(const Tensor<type, 2>& inputs) const
{
    switch(pooling_method)
    {
        case NoPooling:
        {
            return calculate_no_pooling_activations_derivatives(inputs);
        }
        case AveragePooling:
        {
            return calculate_average_pooling_activations_derivatives(inputs);
        }
        case MaxPooling:
        {
            return calculate_max_pooling_activations_derivatives(inputs);
        }
    }

    return Tensor<type, 2>();
}


/// Returns the result of applying the no pooling activation method derivative to a batch of images.
/// @param inputs The batch of images.

Tensor<type, 2> PoolingLayer::calculate_no_pooling_activations_derivatives(const Tensor<type, 2>& inputs) const
{
/*
    return Tensor<type, 2>(inputs.dimensions(), 1.0);
*/
    return Tensor<type, 2>();
}


/// Returns the result of applying the no pooling activation method derivative to a batch of images.
/// @param inputs The batch of images.

Tensor<type, 2> PoolingLayer::calculate_average_pooling_activations_derivatives(const Tensor<type, 2>& inputs) const
{
/*
    return Tensor<type, 2>(inputs.dimensions(), 1.0);
*/
    return Tensor<type, 2>();
}


Tensor<type, 2> PoolingLayer::calculate_max_pooling_activations_derivatives(const Tensor<type, 2>& inputs) const
{
/*
    return Tensor<type, 2>(inputs.dimensions(), 1.0);
*/
    return Tensor<type, 2>();
}


Tensor<type, 2> PoolingLayer::calculate_output_delta(const Tensor<type, 2>&, const Tensor<type, 2>& output_gradient) const
{
    return output_gradient;
}


Tensor<type, 2> PoolingLayer::calculate_hidden_delta(Layer* next_layer_pointer,
                                                    const Tensor<type, 2>& activations,
                                                    const Tensor<type, 2>& activations_derivatives,
                                                    const Tensor<type, 2>& next_layer_delta) const
{
    if(pooling_method == NoPooling) return next_layer_delta;
    else
    {
        const Type layer_type = next_layer_pointer->get_type();

        if(layer_type == Convolutional)
        {
            ConvolutionalLayer* convolutional_layer = dynamic_cast<ConvolutionalLayer*>(next_layer_pointer);

            return calculate_hidden_delta_convolutional(convolutional_layer, activations, activations_derivatives, next_layer_delta);
        }
        else if(layer_type == Pooling)
        {
            PoolingLayer* pooling_layer = dynamic_cast<PoolingLayer*>(next_layer_pointer);

            return calculate_hidden_delta_pooling(pooling_layer, activations, activations_derivatives, next_layer_delta);
        }
        else if(layer_type == Perceptron)
        {
            PerceptronLayer* perceptron_layer = dynamic_cast<PerceptronLayer*>(next_layer_pointer);

            return calculate_hidden_delta_perceptron(perceptron_layer, activations, activations_derivatives, next_layer_delta);
        }
        else if(layer_type == Probabilistic)
        {
            ProbabilisticLayer* probabilistic_layer = dynamic_cast<ProbabilisticLayer*>(next_layer_pointer);

            return calculate_hidden_delta_probabilistic(probabilistic_layer, activations, activations_derivatives, next_layer_delta);
        }
    }

    return Tensor<type, 2>();
}


Tensor<type, 2> PoolingLayer::calculate_hidden_delta_convolutional(ConvolutionalLayer* next_layer_pointer,
                                                                  const Tensor<type, 2>&,
                                                                  const Tensor<type, 2>&,
                                                                  const Tensor<type, 2>& next_layer_delta) const
{
    // Current layer's values

    const int images_number = next_layer_delta.dimension(0);
    const int channels_number = get_inputs_channels_number();
    const int output_rows_number = get_outputs_rows_number();
    const int output_columns_number = get_outputs_columns_number();

    // Next layer's values

    const int next_layers_filters_number = next_layer_pointer->get_filters_number();
    const int next_layers_filter_rows = next_layer_pointer->get_filters_rows_number();
    const int next_layers_filter_columns = next_layer_pointer->get_filters_columns_number();
    const int next_layers_output_rows = next_layer_pointer->get_outputs_rows_number();
    const int next_layers_output_columns = next_layer_pointer->get_outputs_columns_number();
    const int next_layers_row_stride = next_layer_pointer->get_row_stride();
    const int next_layers_column_stride = next_layer_pointer->get_column_stride();

    const Tensor<type, 2> next_layers_weights = next_layer_pointer->get_synaptic_weights();

    // Hidden delta calculation
/*
    Tensor<type, 2> hidden_delta(VectorXi({images_number, channels_number, output_rows_number, output_columns_number}));

    const int size = hidden_delta.size();

    #pragma omp parallel for

    for(int tensor_index = 0; tensor_index < size; tensor_index++)
    {
        const int image_index = tensor_index/(channels_number*output_rows_number*output_columns_number);
        const int channel_index = (tensor_index/(output_rows_number*output_columns_number))%channels_number;
        const int row_index = (tensor_index/output_columns_number)%output_rows_number;
        const int column_index = tensor_index%output_columns_number;

        double sum = 0.0;

        const int lower_row_index = (row_index - next_layers_filter_rows)/next_layers_row_stride + 1;
        const int upper_row_index = min(row_index/next_layers_row_stride + 1, next_layers_output_rows);
        const int lower_column_index = (column_index - next_layers_filter_columns)/next_layers_column_stride + 1;
        const int upper_column_index = min(column_index/next_layers_column_stride + 1, next_layers_output_columns);

        for(int i = 0; i < next_layers_filters_number; i++)
        {
            for(int j = lower_row_index; j < upper_row_index; j++)
            {
                for(int k = lower_column_index; k < upper_column_index; k++)
                {
                    const double delta_element = next_layer_delta(image_index, i, j, k);

                    const double weight = next_layers_weights(i, channel_index, row_index - j*next_layers_row_stride, column_index - k*next_layers_column_stride);

                    sum += delta_element*weight;
                }
            }
        }

        hidden_delta(image_index, channel_index, row_index, column_index) = sum;
    }

    return hidden_delta;
*/
    return Tensor<type, 2>();

}


Tensor<type, 2> PoolingLayer::calculate_hidden_delta_pooling(PoolingLayer* next_layer_pointer,
                                                            const Tensor<type, 2>& activations,
                                                            const Tensor<type, 2>&,
                                                            const Tensor<type, 2>& next_layer_delta) const
{
/*
    switch(next_layer_pointer->get_pooling_method())
    {
        case OpenNN::PoolingLayer::PoolingMethod::NoPooling:
        {
            return next_layer_delta;
        }

        case OpenNN::PoolingLayer::PoolingMethod::AveragePooling:
        {
            // Current layer's values

            const int images_number = next_layer_delta.dimension(0);
            const int channels_number = get_inputs_channels_number();
            const int output_rows_number = get_outputs_rows_number();
            const int output_columns_number = get_outputs_columns_number();

            // Next layer's values

            const int next_layers_pool_rows = next_layer_pointer->get_pool_rows_number();
            const int next_layers_pool_columns = next_layer_pointer->get_pool_columns_number();
            const int next_layers_output_rows = next_layer_pointer->get_outputs_rows_number();
            const int next_layers_output_columns = next_layer_pointer->get_outputs_columns_number();
            const int next_layers_row_stride = next_layer_pointer->get_row_stride();
            const int next_layers_column_stride = next_layer_pointer->get_column_stride();

            // Hidden delta calculation

            Tensor<type, 2> hidden_delta(VectorXi({images_number, channels_number, output_rows_number, output_columns_number}));

            const int size = hidden_delta.size();

            #pragma omp parallel for

            for(int tensor_index = 0; tensor_index < size; tensor_index++)
            {
                const int image_index = tensor_index/(channels_number*output_rows_number*output_columns_number);
                const int channel_index = (tensor_index/(output_rows_number*output_columns_number))%channels_number;
                const int row_index = (tensor_index/output_columns_number)%output_rows_number;
                const int column_index = tensor_index%output_columns_number;

                double sum = 0.0;

                const int lower_row_index = (row_index - next_layers_pool_rows)/next_layers_row_stride + 1;
                const int upper_row_index = min(row_index/next_layers_row_stride + 1, next_layers_output_rows);
                const int lower_column_index = (column_index - next_layers_pool_columns)/next_layers_column_stride + 1;
                const int upper_column_index = min(column_index/next_layers_column_stride + 1, next_layers_output_columns);

                for(int i = lower_row_index; i < upper_row_index; i++)
                {
                    for(int j = lower_column_index; j < upper_column_index; j++)
                    {
                        const double delta_element = next_layer_delta(image_index, channel_index, i, j);

                        sum += delta_element;
                    }
                }

                hidden_delta(image_index, channel_index, row_index, column_index) = sum;
            }

            return hidden_delta/(next_layers_pool_rows*next_layers_pool_columns);
        }

        case OpenNN::PoolingLayer::PoolingMethod::MaxPooling:
        {
            // Current layer's values

            const int images_number = next_layer_delta.dimension(0);
            const int channels_number = get_inputs_channels_number();
            const int output_rows_number = get_outputs_rows_number();
            const int output_columns_number = get_outputs_columns_number();

            // Next layer's values

            const int next_layers_pool_rows = next_layer_pointer->get_pool_rows_number();
            const int next_layers_pool_columns = next_layer_pointer->get_pool_columns_number();
            const int next_layers_output_rows = next_layer_pointer->get_outputs_rows_number();
            const int next_layers_output_columns = next_layer_pointer->get_outputs_columns_number();
            const int next_layers_row_stride = next_layer_pointer->get_row_stride();
            const int next_layers_column_stride = next_layer_pointer->get_column_stride();

            // Hidden delta calculation

            Tensor<type, 2> hidden_delta(VectorXi({images_number, channels_number, output_rows_number, output_columns_number}));

            const int size = hidden_delta.size();

            #pragma omp parallel for

            for(int tensor_index = 0; tensor_index < size; tensor_index++)
            {
                const int image_index = tensor_index/(channels_number*output_rows_number*output_columns_number);
                const int channel_index = (tensor_index/(output_rows_number*output_columns_number))%channels_number;
                const int row_index = (tensor_index/output_columns_number)%output_rows_number;
                const int column_index = tensor_index%output_columns_number;

                double sum = 0.0;

                const int lower_row_index = (row_index - next_layers_pool_rows)/next_layers_row_stride + 1;
                const int upper_row_index = min(row_index/next_layers_row_stride + 1, next_layers_output_rows);
                const int lower_column_index = (column_index - next_layers_pool_columns)/next_layers_column_stride + 1;
                const int upper_column_index = min(column_index/next_layers_column_stride + 1, next_layers_output_columns);

                for(int i = lower_row_index; i < upper_row_index; i++)
                {
                    for(int j = lower_column_index; j < upper_column_index; j++)
                    {
                        Tensor<type, 2> activations_current_submatrix(next_layers_pool_rows, next_layers_pool_columns);

                        for(int submatrix_row_index = 0; submatrix_row_index < next_layers_pool_rows; submatrix_row_index++)
                        {
                            for(int submatrix_column_index = 0; submatrix_column_index < next_layers_pool_columns; submatrix_column_index++)
                            {
                                activations_current_submatrix(submatrix_row_index, submatrix_column_index) =
                                        activations(image_index, channel_index, i*next_layers_row_stride + submatrix_row_index, j*next_layers_column_stride + submatrix_column_index);
                            }
                        }

                        Tensor<type, 2> multiply_not_multiply(next_layers_pool_rows, next_layers_pool_columns);

                        double max_value = activations_current_submatrix(0,0);

                        for(int submatrix_row_index = 0; submatrix_row_index < next_layers_pool_rows; submatrix_row_index++)
                        {
                            for(int submatrix_column_index = 0; submatrix_column_index < next_layers_pool_columns; submatrix_column_index++)
                            {
                                if(activations_current_submatrix(submatrix_row_index, submatrix_column_index) > max_value)
                                {
                                    max_value = activations_current_submatrix(submatrix_row_index, submatrix_column_index);

                                    multiply_not_multiply = Tensor<type, 2>(next_layers_pool_rows, next_layers_pool_columns, 0.0);
                                    multiply_not_multiply(submatrix_row_index, submatrix_column_index) = 1.0;
                                }
                            }
                        }

                        const double delta_element = next_layer_delta(image_index, channel_index, i, j);

                        const double max_derivative = multiply_not_multiply(row_index - i*next_layers_row_stride, column_index - j*next_layers_column_stride);

                        sum += delta_element*max_derivative;
                    }
                }

                hidden_delta(image_index, channel_index, row_index, column_index) = sum;
            }

            return hidden_delta;
        }
    }
*/
    return Tensor<type, 2>();
}


Tensor<type, 2> PoolingLayer::calculate_hidden_delta_perceptron(PerceptronLayer* next_layer_pointer,
                                                               const Tensor<type, 2>&,
                                                               const Tensor<type, 2>&,
                                                               const Tensor<type, 2>& next_layer_delta) const
{
/*
    // Current layer's values

    const int images_number = next_layer_delta.dimension(0);
    const int channels_number = get_inputs_channels_number();
    const int output_rows_number = get_outputs_rows_number();
    const int output_columns_number = get_outputs_columns_number();

    // Next layer's values

    const int next_layers_output_columns = next_layer_delta.dimension(1);

    const MatrixXd next_layers_weights = next_layer_pointer->get_synaptic_weights();

    // Hidden delta calculation

    Tensor<type, 2> hidden_delta(VectorXi({images_number, channels_number, output_rows_number, output_columns_number}));

    const int size = hidden_delta.size();

    #pragma omp parallel for

    for(int tensor_index = 0; tensor_index < size; tensor_index++)
    {
        int image_index = tensor_index/(channels_number*output_rows_number*output_columns_number);
        int channel_index = (tensor_index/(output_rows_number*output_columns_number))%channels_number;
        int row_index = (tensor_index/output_columns_number)%output_rows_number;
        int column_index = tensor_index%output_columns_number;

        double sum = 0.0;

        for(int sum_index = 0; sum_index < next_layers_output_columns; sum_index++)
        {
            const double delta_element = next_layer_delta(image_index, sum_index);

            const double weight = next_layers_weights(channel_index + row_index*channels_number + column_index*channels_number*output_rows_number, sum_index);

            sum += delta_element*weight;
        }

        hidden_delta(image_index, channel_index, row_index, column_index) = sum;
    }

    return hidden_delta;
 */
    return Tensor<type, 2>();

}


Tensor<type, 2> PoolingLayer::calculate_hidden_delta_probabilistic(ProbabilisticLayer* next_layer_pointer,
                                                                  const Tensor<type, 2>&,
                                                                  const Tensor<type, 2>&,
                                                                  const Tensor<type, 2>& next_layer_delta) const
{
/*
    // Current layer's values

    const int images_number = next_layer_delta.dimension(0);
    const int channels_number = get_inputs_channels_number();
    const int output_rows_number = get_outputs_rows_number();
    const int output_columns_number = get_outputs_columns_number();

    // Next layer's values

    const int next_layers_output_columns = next_layer_delta.dimension(1);

    const MatrixXd next_layers_weights = next_layer_pointer->get_synaptic_weights();

    // Hidden delta calculation

    Tensor<type, 2> hidden_delta(VectorXi({images_number, channels_number, output_rows_number, output_columns_number}));

    const int size = hidden_delta.size();

    #pragma omp parallel for

    for(int tensor_index = 0; tensor_index < size; tensor_index++)
    {
        const int image_index = tensor_index/(channels_number*output_rows_number*output_columns_number);
        const int channel_index = (tensor_index/(output_rows_number*output_columns_number))%channels_number;
        const int row_index = (tensor_index/output_columns_number)%output_rows_number;
        const int column_index = tensor_index%output_columns_number;

        double sum = 0.0;

        for(int sum_index = 0; sum_index < next_layers_output_columns; sum_index++)
        {
            const double delta_element = next_layer_delta(image_index, sum_index);

            const double weight = next_layers_weights(channel_index + row_index*channels_number + column_index*channels_number*output_rows_number, sum_index);

            sum += delta_element*weight;
        }

        hidden_delta(image_index, channel_index, row_index, column_index) = sum;
    }

    return hidden_delta;
*/
    return Tensor<type, 2>();

}


Tensor<type, 1> PoolingLayer::calculate_error_gradient(const Tensor<type, 2>& ,
                                                      const Layer::ForwardPropagation&,
                                                      const Tensor<type, 2>&)
{
    return Tensor<type, 1>();
}


/// Returns the number of neurons the layer applies to an image.

int PoolingLayer::get_neurons_number() const
{
    return get_outputs_rows_number() * get_outputs_columns_number();
}


/// Returns a vector containing the positions on the input image covered by a given neuron.
/// @param neuron The neuron's number.

VectorXi PoolingLayer::get_inputs_indices(const int& neuron) const
{
    if(neuron > get_neurons_number() - 1)
    {
        return VectorXi();
    }

    const int row_index = neuron / get_outputs_columns_number();
    const int column_index = neuron % get_outputs_columns_number();

    VectorXi indices;

    // With stride = 1
/*
    for(int i = row_index; i < row_index + pool_rows_number; i++)
    {
        for(int j = column_index; j < column_index + pool_columns_number; j++)
        {
            indices.push_back(i*input_variables_dimensions[2] + j );
        }
    }
*/
    return indices;
}


/// Returns the layer's input's dimensions.

VectorXi PoolingLayer::get_input_variables_dimensions() const
{
/*
    const Index batch_instances_number = 0;
    const int filters_number = 0;
    const int outputs_rows_number = get_outputs_rows_number();
    const int outputs_columns_number = get_outputs_columns_number();

    return VectorXi({batch_instances_number, filters_number, outputs_rows_number, outputs_columns_number});
*/

    return VectorXi();
}


/// Returns the layer's outputs dimensions.

VectorXi PoolingLayer::get_outputs_dimensions() const
{
    VectorXi outputs_dimensions(3);

    outputs_dimensions[0] = input_variables_dimensions[0];
    outputs_dimensions[1] = get_outputs_rows_number();
    outputs_dimensions[2] = get_outputs_columns_number();

    return outputs_dimensions;
}

/// Returns the number of inputs of the layer.

int PoolingLayer::get_inputs_number() const
{
/*
    return input_variables_dimensions.calculate_product();
*/
    return 0;
}


/// Returns the number of channels of the layers' input.

int PoolingLayer::get_inputs_channels_number() const
{
    return input_variables_dimensions[0];
}


/// Returns the number of rows of the layer's input.

int PoolingLayer::get_inputs_rows_number() const
{
    return input_variables_dimensions[1];
}


/// Returns the number of columns of the layer's input.

int PoolingLayer::get_inputs_columns_number() const
{
    return input_variables_dimensions[2];
}


/// Returns the number of rows of the layer's output.

int PoolingLayer::get_outputs_rows_number() const
{
    return (input_variables_dimensions[1] - pool_rows_number)/row_stride + 1;
}


/// Returns the number of columns of the layer's output.

int PoolingLayer::get_outputs_columns_number() const
{
    return (input_variables_dimensions[2] - pool_columns_number)/column_stride + 1;
}


/// Returns the padding width.

int PoolingLayer::get_padding_width() const
{
    return padding_width;
}


/// Returns the pooling filter's row stride.

int PoolingLayer::get_row_stride() const
{
    return row_stride;
}


/// Returns the pooling filter's column stride.

int PoolingLayer::get_column_stride() const
{
    return column_stride;
}


/// Returns the number of rows of the pooling filter.

int PoolingLayer::get_pool_rows_number() const
{
    return pool_rows_number;
}


/// Returns the number of columns of the pooling filter.

int PoolingLayer::get_pool_columns_number() const
{
    return pool_columns_number;
}


/// Returns the number of parameters of the layer.

int PoolingLayer::get_parameters_number() const
{
    return 0;
}


/// Returns the layer's parameters.

Tensor<type, 1> PoolingLayer::get_parameters() const
{
    return Tensor<type, 1>();
}


/// Returns the pooling method.

PoolingLayer::PoolingMethod PoolingLayer::get_pooling_method() const
{
    return pooling_method;
}


/// Sets the number of rows of the layer's input.
/// @param new_input_rows_number The desired rows number.

void PoolingLayer::set_input_variables_dimensions(const VectorXi& new_input_variables_dimensions)
{
    input_variables_dimensions = new_input_variables_dimensions;
}


/// Sets the padding width.
/// @param new_padding_width The desired width.

void PoolingLayer::set_padding_width(const int& new_padding_width)
{
    padding_width = new_padding_width;
}


/// Sets the pooling filter's row stride.
/// @param new_row_stride The desired row stride.

void PoolingLayer::set_row_stride(const int& new_row_stride)
{
    row_stride = new_row_stride;
}


/// Sets the pooling filter's column stride.
/// @param new_column_stride The desired column stride.

void PoolingLayer::set_column_stride(const int& new_column_stride)
{
    column_stride = new_column_stride;
}


/// Sets the pooling filter's dimensions.
/// @param new_pool_rows_number The desired number of rows.
/// @param new_pool_columns_number The desired number of columns.

void PoolingLayer::set_pool_size(const int& new_pool_rows_number,
                                 const int& new_pool_columns_number)
{
    pool_rows_number = new_pool_rows_number;

    pool_columns_number = new_pool_columns_number;
}


/// Sets the layer's pooling method.
/// @param new_pooling_method The desired method.

void PoolingLayer::set_pooling_method(const PoolingMethod& new_pooling_method)
{
    pooling_method = new_pooling_method;
}


/// Sets the layer type to Layer::Pooling.

void PoolingLayer::set_default()
{
    layer_type = Layer::Pooling;
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
