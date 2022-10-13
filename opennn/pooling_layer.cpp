//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pooling_layer.h"

namespace opennn
{

/// Default constructor.
/// It creates an empty PoolingLayer object.

PoolingLayer::PoolingLayer() : Layer()
{
    set_default();
}

/// Input size setter constructor.
/// After setting new dimensions for the input, it creates an empty PoolingLayer object.
/// @param new_input_variables_dimensions A vector containing the new number of channels, rows and columns for the input.

PoolingLayer::PoolingLayer(const Tensor<Index, 1>& new_input_variables_dimensions) : Layer()
{
    set_default();
}


/// Input size setter constructor.
/// After setting new dimensions for the input, it creates an empty PoolingLayer object.
/// @param new_input_variables_dimensions A vector containing the desired number of rows and columns for the input.
/// @param pool_dimensions A vector containing the desired number of rows and columns for the pool.

PoolingLayer::PoolingLayer(const Tensor<Index, 1>& new_input_variables_dimensions, const Tensor<Index, 1>& pool_dimensions) : Layer()
{
    pool_rows_number = pool_dimensions[0];

    pool_columns_number = pool_dimensions[1];

    set_default();
}


/// Returns the output of the pooling layer applied to a batch of images.
/// @param inputs The batch of images.

void PoolingLayer::calculate_outputs(type* inputs_data, const Tensor<Index, 1>& inputs_dimensions,
                                     type* outputs_data, const Tensor<Index, 1>& outputs_dimensions)
{
    if(inputs_dimensions.size() != 4)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) method.\n"
               << "Number of inputs dimensions (" << inputs_dimensions.size() << ") must be 4: (batch, filters, rows, columns).\n";

        throw invalid_argument(buffer.str());
    }
    /// @todo Change everything

    const TensorMap<Tensor<type, 4>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1), inputs_dimensions(2), inputs_dimensions(3));
    TensorMap<Tensor<type, 4>> outputs(outputs_data, outputs_dimensions(0), outputs_dimensions(1), outputs_dimensions(2), outputs_dimensions(3));

    switch(pooling_method)
    {
    case PoolingMethod::NoPooling:
        outputs = calculate_no_pooling_outputs(inputs);

    case PoolingMethod::MaxPooling:
        outputs = calculate_max_pooling_outputs(inputs);

    case PoolingMethod::AveragePooling:
        outputs = calculate_average_pooling_outputs(inputs);
    }

}


/// Returns the result of applying average pooling to a batch of images.
/// @param inputs The batch of images.

Tensor<type, 4> PoolingLayer::calculate_average_pooling_outputs(const Tensor<type, 4>& inputs) const
{
    const Index images_number = inputs.dimension(0);

    const Index channels_number = inputs.dimension(1);

    const Index inputs_rows_number = inputs.dimension(2);

    const Index inputs_columns_number = inputs.dimension(3);

    const Index outputs_rows_number = (inputs_rows_number - pool_rows_number)/row_stride + 1;

    const Index outputs_columns_number = (inputs_columns_number - pool_columns_number)/column_stride + 1;

    Tensor<type, 4> outputs(images_number, channels_number, outputs_rows_number, outputs_columns_number);

    for(Index image_index = 0; image_index < images_number; image_index ++)
    {
        for(Index channel_index = 0; channel_index < channels_number; channel_index ++)
        {
            for(Index row_index = 0; row_index < outputs_rows_number; row_index ++)
            {
                for(Index column_index = 0; column_index < outputs_columns_number; column_index ++)
                {
                    outputs(image_index, channel_index, row_index, column_index) = type(0);

                    for(Index window_row = 0; window_row < pool_rows_number; window_row ++)
                    {
                        const Index row = row_index*row_stride + window_row;

                        for(Index window_column = 0; window_column < pool_columns_number; window_column ++)
                        {
                            const Index column = column_index*column_stride + window_column;

                            outputs(image_index, channel_index, row_index, column_index) += inputs(image_index, channel_index, row, column);
                        }
                    }

                    outputs(image_index, channel_index, row_index, column_index) /= type(pool_rows_number*pool_columns_number);
                }
            }
        }
    }

    return outputs;
}


/// Returns the result of applying no pooling to a batch of images.
/// @param inputs The batch of images.

Tensor<type, 4> PoolingLayer::calculate_no_pooling_outputs(const Tensor<type, 4>& inputs) const
{
    return inputs;
}


/// Returns the result of applying max pooling to a batch of images.
/// @param inputs The batch of images.

Tensor<type, 4> PoolingLayer::calculate_max_pooling_outputs(const Tensor<type, 4>& inputs) const
{
    const Index images_number = inputs.dimension(0);

    const Index channels_number = inputs.dimension(1);

    const Index inputs_rows_number = inputs.dimension(2);

    const Index inputs_columns_number = inputs.dimension(3);

    const Index outputs_rows_number = (inputs_rows_number - pool_rows_number)/row_stride + 1;

    const Index outputs_columns_number = (inputs_columns_number - pool_columns_number)/column_stride + 1;

    Tensor<type, 4> outputs(images_number, channels_number, outputs_rows_number, outputs_columns_number);

    for(Index image_index = 0; image_index < images_number; image_index ++)
    {
        for(Index channel_index = 0; channel_index < channels_number; channel_index ++)
        {
            for(Index row_index = 0; row_index < outputs_rows_number; row_index ++)
            {
                for(Index column_index = 0; column_index < outputs_columns_number; column_index ++)
                {
                    outputs(image_index, channel_index, row_index, column_index) =
                            inputs(image_index, channel_index, row_index*row_stride, column_index*column_stride);

                    for(Index window_row = 0; window_row < pool_rows_number; window_row ++)
                    {
                        const Index row = row_index*row_stride + window_row;

                        for(Index window_column = 0; window_column < pool_columns_number; window_column ++)
                        {
                            const Index column = column_index*column_stride + window_column;

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
}


void PoolingLayer::calculate_hidden_delta(LayerForwardPropagation*,
                                                     LayerBackPropagation*,
                                                     LayerBackPropagation*) const
{/*
    if(pooling_method == PoolingMethod::NoPooling) return next_layer_delta;

    else
    {
        const Type layer_type = next_layer_pointer->get_type();

        if(layer_type == Type::Convolutional)
        {
            ConvolutionalLayer* convolutional_layer = dynamic_cast<ConvolutionalLayer*>(next_layer_pointer);

            return calculate_hidden_delta_convolutional(convolutional_layer, activations, activations_derivatives, next_layer_delta);
        }
        else if(layer_type == Type::Pooling)
        {
            PoolingLayer* pooling_layer = dynamic_cast<PoolingLayer*>(next_layer_pointer);

            return calculate_hidden_delta_pooling(pooling_layer, activations, activations_derivatives, next_layer_delta);
        }
        else if(layer_type == Type::Perceptron)
        {
            PerceptronLayer* perceptron_layer = static_cast<PerceptronLayer*>(next_layer_pointer);

            return calculate_hidden_delta_perceptron(perceptron_layer, activations, activations_derivatives, next_layer_delta);
        }
        else if(layer_type == Type::Probabilistic)
        {
            ProbabilisticLayer* probabilistic_layer = dynamic_cast<ProbabilisticLayer*>(next_layer_pointer);

            return calculate_hidden_delta_probabilistic(probabilistic_layer, activations, activations_derivatives, next_layer_delta);
        }
    }

    return Tensor<type, 4>();
    */
}

/*
Tensor<type, 4> PoolingLayer::calculate_hidden_delta_convolutional(ConvolutionalLayer* next_layer_pointer,
        const Tensor<type, 4>&,
        const Tensor<type, 4>&,
        const Tensor<type, 4>& next_layer_delta) const
{
    // Current layer's values

    const Index images_number = next_layer_delta.dimension(0);
    const Index channels_number = get_inputs_channels_number();
    const Index output_rows_number = get_outputs_rows_number();
    const Index output_columns_number = get_outputs_columns_number();

    // Next layer's values

    const Index next_layers_filters_number = next_layer_pointer->get_kernels_number();
    const Index next_layers_filter_rows = next_layer_pointer->get_kernels_rows_number();
    const Index next_layers_filter_columns = next_layer_pointer->get_kernels_columns_number();
    const Index next_layers_output_rows = next_layer_pointer->get_outputs_rows_number();
    const Index next_layers_output_columns = next_layer_pointer->get_outputs_columns_number();
    const Index next_layers_row_stride = next_layer_pointer->get_row_stride();
    const Index next_layers_column_stride = next_layer_pointer->get_column_stride();

    const Tensor<type, 4> next_layers_weights = next_layer_pointer->get_synaptic_weights();

    // Hidden delta calculation

        Tensor<type, 4> hidden_delta(images_number, channels_number, output_rows_number, output_columns_number);

        const Index size = hidden_delta.size();

        #pragma omp parallel for

        for(Index tensor_index = 0; tensor_index < size; tensor_index++)
        {
            const Index image_index = tensor_index/(channels_number*output_rows_number*output_columns_number);
            const Index channel_index = (tensor_index/(output_rows_number*output_columns_number))%channels_number;
            const Index row_index = (tensor_index/output_columns_number)%output_rows_number;
            const Index column_index = tensor_index%output_columns_number;

            long double sum = 0.0;

            const Index lower_row_index = (row_index - next_layers_filter_rows)/next_layers_row_stride + 1;
            const Index upper_row_index = min(row_index/next_layers_row_stride + 1, next_layers_output_rows);
            const Index lower_column_index = (column_index - next_layers_filter_columns)/next_layers_column_stride + 1;
            const Index upper_column_index = min(column_index/next_layers_column_stride + 1, next_layers_output_columns);

            for(Index i = 0; i < next_layers_filters_number; i++)
            {
                for(Index j = lower_row_index; j < upper_row_index; j++)
                {
                    for(Index k = lower_column_index; k < upper_column_index; k++)
                    {
                        const type delta_element = next_layer_delta(image_index, i, j, k);

                        const type weight = next_layers_weights(i, channel_index, row_index - j*next_layers_row_stride, column_index - k*next_layers_column_stride);

                        sum += delta_element*weight;
                    }
                }
            }

            hidden_delta(image_index, channel_index, row_index, column_index) = sum;
        }

        return hidden_delta;
}
*/

Tensor<type, 4> PoolingLayer::calculate_hidden_delta_pooling(PoolingLayer* next_layer_pointer,
        const Tensor<type, 4>& activations,
        const Tensor<type, 4>&,
        const Tensor<type, 4>& next_layer_delta) const
{
        switch(next_layer_pointer->get_pooling_method())
        {
            case opennn::PoolingLayer::PoolingMethod::NoPooling:
            {
                return next_layer_delta;
            }

            case opennn::PoolingLayer::PoolingMethod::AveragePooling:
            {
                // Current layer's values

                const Index images_number = next_layer_delta.dimension(0);
                const Index channels_number = get_inputs_channels_number();
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

                Tensor<type, 4> hidden_delta(images_number, channels_number, output_rows_number, output_columns_number);

                const Index size = hidden_delta.size();

                #pragma omp parallel for

                for(Index tensor_index = 0; tensor_index < size; tensor_index++)
                {
                    const Index image_index = tensor_index/(channels_number*output_rows_number*output_columns_number);
                    const Index channel_index = (tensor_index/(output_rows_number*output_columns_number))%channels_number;
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
                            const type delta_element = next_layer_delta(image_index, channel_index, i, j);

                            sum += delta_element;
                        }
                    }

                    hidden_delta(image_index, channel_index, row_index, column_index) = sum;
                }

//                return hidden_delta/(next_layers_pool_rows*next_layers_pool_columns);
            }

            case opennn::PoolingLayer::PoolingMethod::MaxPooling:
            {
                // Current layer's values

                const Index images_number = next_layer_delta.dimension(0);
                const Index channels_number = get_inputs_channels_number();
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

                Tensor<type, 4> hidden_delta(images_number, channels_number, output_rows_number, output_columns_number);

                const Index size = hidden_delta.size();

                #pragma omp parallel for

                for(Index tensor_index = 0; tensor_index < size; tensor_index++)
                {
                    const Index image_index = tensor_index/(channels_number*output_rows_number*output_columns_number);
                    const Index channel_index = (tensor_index/(output_rows_number*output_columns_number))%channels_number;
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
                            Tensor<type, 2> activations_current_submatrix(next_layers_pool_rows, next_layers_pool_columns);

                            for(Index submatrix_row_index = 0; submatrix_row_index < next_layers_pool_rows; submatrix_row_index++)
                            {
                                for(Index submatrix_column_index = 0; submatrix_column_index < next_layers_pool_columns; submatrix_column_index++)
                                {
                                    activations_current_submatrix(submatrix_row_index, submatrix_column_index) =
                                            activations(image_index, channel_index, i*next_layers_row_stride + submatrix_row_index, j*next_layers_column_stride + submatrix_column_index);
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

                                        //multiply_not_multiply.resize(next_layers_pool_rows, next_layers_pool_columns, 0.0);
                                        //multiply_not_multiply(submatrix_row_index, submatrix_column_index) = type(1);
                                    }
                                }
                            }

                            const type delta_element = next_layer_delta(image_index, channel_index, i, j);

                            const type max_derivative = multiply_not_multiply(row_index - i*next_layers_row_stride, column_index - j*next_layers_column_stride);

                            sum += delta_element*max_derivative;
                        }
                    }

                    hidden_delta(image_index, channel_index, row_index, column_index) = sum;
                }

                return hidden_delta;
                break;
            }
        }

    return Tensor<type, 4>();
}


Tensor<type, 4> PoolingLayer::calculate_hidden_delta_perceptron(PerceptronLayer* next_layer_pointer,
        const Tensor<type, 4>&,
        const Tensor<type, 4>&,
        const Tensor<type, 4>& next_layer_delta) const
{
    // Current layer's values

    const Index images_number = next_layer_delta.dimension(0);
    const Index channels_number = get_inputs_channels_number();
    const Index output_rows_number = get_outputs_rows_number();
    const Index output_columns_number = get_outputs_columns_number();

    // Next layer's values

    const Index next_layers_output_columns = next_layer_delta.dimension(1);

    const Tensor<type, 2> next_layers_weights = next_layer_pointer->get_synaptic_weights();

    // Hidden delta calculation

    Tensor<type, 4> hidden_delta(images_number, channels_number, output_rows_number, output_columns_number);

    const Index size = hidden_delta.size();

    #pragma omp parallel for

    for(Index tensor_index = 0; tensor_index < size; tensor_index++)
    {
        //Index image_index = tensor_index/(channels_number*output_rows_number*output_columns_number);
        //Index channel_index = (tensor_index/(output_rows_number*output_columns_number))%channels_number;
        //Index row_index = (tensor_index/output_columns_number)%output_rows_number;
        //Index column_index = tensor_index%output_columns_number;

        //long double sum = 0.0;

        for(Index sum_index = 0; sum_index < next_layers_output_columns; sum_index++)
        {
            //const type delta_element = next_layer_delta(image_index, sum_index);

            //const type weight = next_layers_weights(channel_index + row_index*channels_number + column_index*channels_number*output_rows_number, sum_index);

            //sum += delta_element*weight;
        }

        //hidden_delta(image_index, channel_index, row_index, column_index) = sum;
    }

    return hidden_delta;
}


Tensor<type, 4> PoolingLayer::calculate_hidden_delta_probabilistic(ProbabilisticLayer* next_layer_pointer,
        const Tensor<type, 4>&,
        const Tensor<type, 4>&,
        const Tensor<type, 4>& next_layer_delta) const
{
    // Current layer's values

    const Index images_number = next_layer_delta.dimension(0);
    const Index channels_number = get_inputs_channels_number();
    const Index output_rows_number = get_outputs_rows_number();
    const Index output_columns_number = get_outputs_columns_number();

    // Next layer's values

    const Index next_layers_output_columns = next_layer_delta.dimension(1);

    const Tensor<type, 2> next_layers_weights = next_layer_pointer->get_synaptic_weights();

    // Hidden delta calculation

    Tensor<type, 4> hidden_delta(images_number, channels_number, output_rows_number, output_columns_number);

    const Index size = hidden_delta.size();

    #pragma omp parallel for

    for(Index tensor_index = 0; tensor_index < size; tensor_index++)
    {
        //const Index image_index = tensor_index/(channels_number*output_rows_number*output_columns_number);
        //const Index channel_index = (tensor_index/(output_rows_number*output_columns_number))%channels_number;
        //const Index row_index = (tensor_index/output_columns_number)%output_rows_number;
        //const Index column_index = tensor_index%output_columns_number;

        //long double sum = 0.0;

        for(Index sum_index = 0; sum_index < next_layers_output_columns; sum_index++)
        {
            // const type delta_element = next_layer_delta(image_index, sum_index);

            // const type weight = next_layers_weights(channel_index + row_index*channels_number + column_index*channels_number*output_rows_number, sum_index);

            // sum += delta_element*weight;
        }

        //hidden_delta(image_index, channel_index, row_index, column_index) = sum;
    }

    return hidden_delta;
}



/// Returns the number of neurons the layer applies to an image.

Index PoolingLayer::get_neurons_number() const
{
    return get_outputs_rows_number() * get_outputs_columns_number();
}


/// Returns the layer's outputs dimensions.

Tensor<Index, 1> PoolingLayer::get_outputs_dimensions() const
{
    Tensor<Index, 1> outputs_dimensions(3);

//    outputs_dimensions[0] = input_variables_dimensions[0];
//    outputs_dimensions[1] = get_outputs_rows_number();
//    outputs_dimensions[2] = get_outputs_columns_number();

    return outputs_dimensions;
}


/// Returns the number of inputs of the layer.

Index PoolingLayer::get_inputs_number() const
{
    return input_variables_dimensions.size();
}


/// Returns the number of channels of the layers' input.

Index PoolingLayer::get_inputs_channels_number() const
{
//    return input_variables_dimensions[0];

    return 0;
}


/// Returns the number of rows of the layer's input.

Index PoolingLayer::get_inputs_rows_number() const
{
//    return input_variables_dimensions[1];

    return 0;
}


/// Returns the number of columns of the layer's input.

Index PoolingLayer::get_inputs_columns_number() const
{
//    return input_variables_dimensions[2];

    return 0;
}


/// Returns the number of rows of the layer's output.

Index PoolingLayer::get_outputs_rows_number() const
{
//    return (input_variables_dimensions[1] - pool_rows_number)/row_stride + 1;

    return 0;
}


/// Returns the number of columns of the layer's output.

Index PoolingLayer::get_outputs_columns_number() const
{
//    return (input_variables_dimensions[2] - pool_columns_number)/column_stride + 1;

    return 0;
}


/// Returns the padding width.

Index PoolingLayer::get_padding_width() const
{
    return padding_width;
}


/// Returns the pooling filter's row stride.

Index PoolingLayer::get_row_stride() const
{
    return row_stride;
}


/// Returns the pooling filter's column stride.

Index PoolingLayer::get_column_stride() const
{
    return column_stride;
}


/// Returns the number of rows of the pooling filter.

Index PoolingLayer::get_pool_rows_number() const
{
    return pool_rows_number;
}


/// Returns the number of columns of the pooling filter.

Index PoolingLayer::get_pool_columns_number() const
{
    return pool_columns_number;
}


/// Returns the number of parameters of the layer.

Index PoolingLayer::get_parameters_number() const
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


/// Returns the input_variables_dimensions.

Tensor<Index, 1> PoolingLayer::get_input_variables_dimensions() const
{
    return input_variables_dimensions;
}


/// Returns a string with the name of the pooling layer method.
/// This can be NoPooling, MaxPooling and AveragePooling.

string PoolingLayer::write_pooling_method() const
{
    switch(pooling_method)
    {
    case PoolingMethod::NoPooling:
        return "NoPooling";

    case PoolingMethod::MaxPooling:
        return "MaxPooling";

    case PoolingMethod::AveragePooling:
        return "AveragePooling";
    }

    return string();
}


/// Sets the number of rows of the layer's input.
/// @param new_input_rows_number The desired rows number.

void PoolingLayer::set_input_variables_dimensions(const Tensor<Index, 1>& new_input_variables_dimensions)
{
    input_variables_dimensions = new_input_variables_dimensions;
}


/// Sets the padding width.
/// @param new_padding_width The desired width.

void PoolingLayer::set_padding_width(const Index& new_padding_width)
{
    padding_width = new_padding_width;
}


/// Sets the pooling filter's row stride.
/// @param new_row_stride The desired row stride.

void PoolingLayer::set_row_stride(const Index& new_row_stride)
{
    row_stride = new_row_stride;
}


/// Sets the pooling filter's column stride.
/// @param new_column_stride The desired column stride.

void PoolingLayer::set_column_stride(const Index& new_column_stride)
{
    column_stride = new_column_stride;
}


/// Sets the pooling filter's dimensions.
/// @param new_pool_rows_number The desired number of rows.
/// @param new_pool_columns_number The desired number of columns.

void PoolingLayer::set_pool_size(const Index& new_pool_rows_number,
                                 const Index& new_pool_columns_number)
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


void PoolingLayer::set_pooling_method(const string& new_pooling_method)
{
    if(new_pooling_method == "NoPooling")
    {
        pooling_method = PoolingMethod::NoPooling;
    }
    else if(new_pooling_method == "MaxPooling")
    {
        pooling_method = PoolingMethod::MaxPooling;
    }
    else if(new_pooling_method == "AveragePooling")
    {
        pooling_method = PoolingMethod::AveragePooling;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void set_pooling_type(const string&) method.\n"
               << "Unknown pooling type: " << new_pooling_method << ".\n";

        throw invalid_argument(buffer.str());
    }
}


/// Sets the layer type to Layer::Pooling.

void PoolingLayer::set_default()
{
    layer_type = Layer::Type::Pooling;
}

/// Serializes the convolutional layer object into an XML document of the TinyXML.
/// See the OpenNN manual for more information about the format of this document.

void PoolingLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Pooling layer

    file_stream.OpenElement("PoolingLayer");

    // Pooling method

    file_stream.OpenElement("PoolingMethod");

    file_stream.PushText(write_pooling_method().c_str());

    file_stream.CloseElement();

    // Inputs variables dimensions

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

    //Row stride

    file_stream.OpenElement("RowStride");

    buffer.str("");
    buffer << get_row_stride();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Pool columns number

    file_stream.OpenElement("PoolColumnsNumber");

    buffer.str("");
    buffer << get_pool_columns_number();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Pool rows number

    file_stream.OpenElement("PoolRowsNumber");

    buffer.str("");
    buffer << get_pool_rows_number();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Padding width

    file_stream.OpenElement("PaddingWidth");

    buffer.str("");
    buffer << get_padding_width();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this convolutional layer object.
/// @param document TinyXML document containing the member data.

void PoolingLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Pooling layer

    const tinyxml2::XMLElement* pooling_layer_element = document.FirstChildElement("PoolingLayer");

    if(!pooling_layer_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling layer element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Pooling method element

    const tinyxml2::XMLElement* pooling_method_element = pooling_layer_element->FirstChildElement("PoolingMethod");

    if(!pooling_method_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling method element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const string pooling_method_string = pooling_method_element->GetText();

    set_pooling_method(pooling_method_string);

    // Input variables dimensions element

    const tinyxml2::XMLElement* input_variables_dimensions_element = pooling_layer_element->FirstChildElement("InputDimensions");

    if(!input_variables_dimensions_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling input variables dimensions element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const string input_variables_dimensions_string = input_variables_dimensions_element->GetText();

//    set_input_variables_dimenisons(input_variables_dimensions_string);

    // Column stride

    const tinyxml2::XMLElement* column_stride_element = pooling_layer_element->FirstChildElement("ColumnStride");

    if(!column_stride_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling column stride element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const string column_stride_string = column_stride_element->GetText();

    set_column_stride(static_cast<Index>(stoi(column_stride_string)));

    // Row stride

    const tinyxml2::XMLElement* row_stride_element = pooling_layer_element->FirstChildElement("RowStride");

    if(!row_stride_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling row stride element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const string row_stride_string = row_stride_element->GetText();

    set_row_stride(static_cast<Index>(stoi(row_stride_string)));

    // Pool columns number

    const tinyxml2::XMLElement* pool_columns_number_element = pooling_layer_element->FirstChildElement("PoolColumnsNumber");

    if(!pool_columns_number_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling columns number element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const string pool_columns_number_string = pool_columns_number_element->GetText();

    // Pool rows number

    const tinyxml2::XMLElement* pool_rows_number_element = pooling_layer_element->FirstChildElement("PoolRowsNumber");

    if(!pool_rows_number_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling rows number element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const string pool_rows_number_string = pool_rows_number_element->GetText();

    set_pool_size(static_cast<Index>(stoi(pool_rows_number_string)), static_cast<Index>(stoi(pool_columns_number_string)));

    // Padding Width

    const tinyxml2::XMLElement* padding_width_element = pooling_layer_element->FirstChildElement("PaddingWidth");

    if(!padding_width_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Padding width element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(padding_width_element->GetText())
    {
        const string padding_width_string = padding_width_element->GetText();

        set_padding_width(static_cast<Index>(stoi(padding_width_string)));
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
