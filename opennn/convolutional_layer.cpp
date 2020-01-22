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

namespace OpenNN
{

/// Default constructor.
/// It creates an empty ConvolutionalLayer object.

ConvolutionalLayer::ConvolutionalLayer() : Layer()
{
    layer_type = Layer::Convolutional;
}


/// Inputs' dimensions modifier constructor.
/// After setting new dimensions for the inputs, it creates and initializes a ConvolutionalLayer object with a number of filters of a given size.
/// The initialization values are random values from a normal distribution.
/// @param new_inputs_dimensions A vector containing the new inputs' dimensions.
/// @param filters_dimensions A vector containing the number of filters, their rows and columns.

ConvolutionalLayer::ConvolutionalLayer(const Tensor<int, 1>& new_inputs_dimensions, const Tensor<int, 1>& new_filters_dimensions) : Layer()
{
    layer_type = Layer::Convolutional;

    set(new_inputs_dimensions, new_filters_dimensions);
}


/// Returns a boolean, true if convolutional layer is empty and false otherwise.

bool ConvolutionalLayer::is_empty() const
{
/*
    if(biases.empty() && synaptic_weights.empty())
    {
        return true;
    }
*/
    return false;
}


/// Returns the result of applying the previously set activation function to a batch of images.
/// @param convolutions The batch of images.

Tensor<type, 2> ConvolutionalLayer::calculate_activations(const Tensor<type, 2>& convolutions) const
{

   switch(activation_function)
   {
       case ConvolutionalLayer::Linear:
       {
            return linear(convolutions);
       }

       case ConvolutionalLayer::Logistic:
       {
            return logistic(convolutions);
       }

       case ConvolutionalLayer::HyperbolicTangent:
       {
            return hyperbolic_tangent(convolutions);
       }

       case ConvolutionalLayer::Threshold:
       {
            return threshold(convolutions);
       }

       case ConvolutionalLayer::SymmetricThreshold:
       {
            return symmetric_threshold(convolutions);
       }

       case ConvolutionalLayer::RectifiedLinear:
       {
            return rectified_linear(convolutions);
       }

       case ConvolutionalLayer::ScaledExponentialLinear:
       {
            return scaled_exponential_linear(convolutions);
       }

       case ConvolutionalLayer::SoftPlus:
       {
            return soft_plus(convolutions);
       }

       case ConvolutionalLayer::SoftSign:
       {
            return soft_sign(convolutions);
       }

       case ConvolutionalLayer::HardSigmoid:
       {
            return hard_sigmoid(convolutions);
       }

       case ConvolutionalLayer::ExponentialLinear:
       {
            return exponential_linear(convolutions);
       }
   }

    return Tensor<type, 2>();
}


/// Returns the result of applying the convolutional layer's filters and biases to a batch of images.
/// @param inputs The batch of images.

Tensor<type, 2> ConvolutionalLayer::calculate_combinations(const Tensor<type, 2>& inputs) const
{
    #ifdef __OPENNN_DEBUG__

    const int inputs_dimensions_number = inputs.rank();

    if(inputs_dimensions_number != 4)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
              << "Tensor<type, 2> calculate_convolutions(const Tensor<type, 2>&) method.\n"
              << "Number of inputs dimensions (" << inputs_dimensions_number << ") must be 4 (batch, channels, rows, columns).\n";

       throw logic_error(buffer.str());
    }

    const int inputs_channels_number = inputs.dimension(1);
    const int filters_channels_number  = get_filters_channels_number();

    if(filters_channels_number != inputs_channels_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
              << "Tensor<type, 2> calculate_convolutions(const Tensor<type, 2>&) method.\n"
              << "Number of input channels (" << inputs_channels_number << ") must be equal to number of filters channels (" << filters_channels_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    // Inputs

    const int images_number = inputs.dimension(0);

    // Filters

    const int filters_number = get_filters_number();

    // Outputs

    const int outputs_rows_number = get_outputs_rows_number();
    const int outputs_columns_number = get_outputs_columns_number();
/*
    Tensor<type, 2> convolutions({images_number, filters_number, outputs_rows_number, outputs_columns_number}, 0.0);

    Tensor<type, 2> image(Tensor<int, 1>({filters_number, outputs_rows_number, outputs_columns_number}));

    Tensor<type, 2> convolution(outputs_rows_number, outputs_columns_number);

    #pragma omp parallel for private(image, convolution)

    for(int image_index = 0; image_index < images_number; image_index++)
    {
        image = inputs.get_tensor(image_index);

        if(padding_option == Same) image = insert_padding(image);

        for(int filter_index = 0; filter_index < filters_number; filter_index++)
        {
            convolution = calculate_image_convolution(image, synaptic_weights.get_tensor(filter_index)) + biases[filter_index];

            convolutions.set_matrix(image_index, filter_index, convolution);
        }
    }

    return convolutions;
*/
    return Tensor<type, 2>();
}


/// Returns the result of applying the parameters passed as argument to a batch of images.
/// @param inputs The batch of images.
/// @param parameters The parameters.

Tensor<type, 2> ConvolutionalLayer::calculate_combinations(const Tensor<type, 2>& inputs, const Tensor<type, 1>& parameters) const
{
/*
    #ifdef __OPENNN_DEBUG__

    const int inputs_dimensions_number = inputs.rank();

    if(inputs_dimensions_number != 4)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
              << "Tensor<type, 2> calculate_convolutions(const Tensor<type, 2>&) method.\n"
              << "Number of inputs dimensions (" << inputs_dimensions_number << ") must be 4 (batch, channels, rows, columns).\n";

       throw logic_error(buffer.str());
    }

    const int inputs_channels_number = inputs.dimension(1);
    const int filters_channels_number  = get_filters_channels_number();

    if(filters_channels_number != inputs_channels_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
              << "Tensor<type, 2> calculate_convolutions(const Tensor<type, 2>&) method.\n"
              << "Number of input channels (" << inputs_channels_number << ") must be equal to number of filters channels (" << filters_channels_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    // Weights and biases

    Tensor<type, 2> new_synaptic_weights = extract_synaptic_weights(parameters);

    Tensor<type, 1> new_biases = extract_biases(parameters);

    // Inputs

    const int images_number = inputs.dimension(0);

    // Filters

    const int filters_number = get_filters_number();

    // Outputs

    const int outputs_rows_number = get_outputs_rows_number();
    const int outputs_columns_number = get_outputs_columns_number();

    Tensor<type, 2> convolutions({images_number, filters_number, outputs_rows_number, outputs_columns_number}, 0.0);

    Tensor<type, 2> image(Tensor<int, 1>({filters_number, outputs_rows_number, outputs_columns_number}));

    Tensor<type, 2> convolution(outputs_rows_number, outputs_columns_number);

    #pragma omp parallel for private(image, convolution)

    for(int image_index = 0; image_index < images_number; image_index++)
    {
        image = inputs.get_tensor(image_index);

        if(padding_option == Same) image = insert_padding(image);

        for(int filter_index = 0; filter_index < filters_number; filter_index++)
        {
            convolution = calculate_image_convolution(image, new_synaptic_weights.get_tensor(filter_index)) + new_biases[filter_index];

            convolutions.set_matrix(image_index, filter_index, convolution);
        }
    }

    return convolutions;
*/
    return Tensor<type, 2>();
}


/// Returns the result of applying the derivative of the previously set activation function to a batch of images.
/// @param convolutions The batch of images.

Tensor<type, 2> ConvolutionalLayer::calculate_activations_derivatives(const Tensor<type, 2>& combinations) const
{
    #ifdef __OPENNN_DEBUG__

        const int combinations_dimensions_number = combinations.rank();

        if(combinations_dimensions_number != 4)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
                  << "Tensor<type, 2> calculate_activations_derivatives(const Tensor<type, 2>&) method.\n"
                  << "Number of combinations dimensions (" << combinations_dimensions_number << ") must be 4 (batch, filters, rows, columns).\n";

           throw logic_error(buffer.str());
        }

    #endif
/*
    switch(activation_function)
    {
        case Linear:
        {
            return linear_derivatives(combinations);
        }

        case Logistic:
        {
            return logistic_derivatives(combinations);
        }

        case HyperbolicTangent:
        {
            return hyperbolic_tangent_derivatives(combinations);
        }

        case Threshold:
        {
            return threshold_derivatives(combinations);
        }

        case SymmetricThreshold:
        {
            return symmetric_threshold_derivatives(combinations);
        }

        case RectifiedLinear:
        {
            return rectified_linear_derivatives(combinations);
        }

        case ScaledExponentialLinear:
        {
            return scaled_exponential_linear_derivatives(combinations);
        }

        case SoftPlus:
        {
            return soft_plus_derivatives(combinations);
        }

        case SoftSign:
        {
            return soft_sign_derivatives(combinations);
        }

        case HardSigmoid:
        {
            return hard_sigmoid_derivatives(combinations);
        }

        case ExponentialLinear:
        {
            return exponential_linear_derivatives(combinations);
        }
    }
*/
    return Tensor<type, 2>();
}


/// Returns the output of the convolutional layer applied to a batch of images.
/// @param inputs The batch of images.

Tensor<type, 2> ConvolutionalLayer::calculate_outputs(const Tensor<type, 2>& inputs)
{
    return calculate_activations(calculate_combinations(inputs));
}


/// Returns the output of the convolutional layer with given parameters applied to a batch of images.
/// @param inputs The batch of images.
/// @param parameters The parameters.

Tensor<type, 2> ConvolutionalLayer::calculate_outputs(const Tensor<type, 2>& inputs, const Tensor<type, 1>& parameters)
{
    return calculate_activations(calculate_combinations(inputs, parameters));
}


Layer::ForwardPropagation ConvolutionalLayer::calculate_forward_propagation(const Tensor<type, 2>& inputs)
{
    ForwardPropagation layers;

    const Tensor<type, 2> combinations = calculate_combinations(inputs);

    layers.activations = calculate_activations(combinations);

    layers.activations_derivatives = calculate_activations_derivatives(combinations);

    return layers;
}


Tensor<type, 2> ConvolutionalLayer::calculate_output_delta(const Tensor<type, 2>& activations_derivatives, const Tensor<type, 2>& output_gradient) const
{
    return activations_derivatives*output_gradient;
}


Tensor<type, 2> ConvolutionalLayer::calculate_hidden_delta(Layer* next_layer_pointer,
                                                          const Tensor<type, 2>& activations,
                                                          const Tensor<type, 2>& activations_derivatives,
                                                          const Tensor<type, 2>& next_layer_delta) const
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

    return Tensor<type, 2>();
}


Tensor<type, 2> ConvolutionalLayer::calculate_hidden_delta_convolutional(ConvolutionalLayer* next_layer_pointer,
                                                                        const Tensor<type, 2>&,
                                                                        const Tensor<type, 2>& activations_derivatives,
                                                                        const Tensor<type, 2>& next_layer_delta) const
{
/*
    // Current layer's values

    const int images_number = next_layer_delta.dimension(0);
    const int filters_number = get_filters_number();
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

    Tensor<type, 2> hidden_delta(Tensor<int, 1>({images_number, filters_number, output_rows_number, output_columns_number}));

    const int size = hidden_delta.size();

    #pragma omp parallel for

    for(int tensor_index = 0; tensor_index < size; tensor_index++)
    {
        const int image_index = tensor_index/(filters_number*output_rows_number*output_columns_number);
        const int channel_index = (tensor_index/(output_rows_number*output_columns_number))%filters_number;
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

    return hidden_delta*activations_derivatives;
*/
    return Tensor<type, 2>();

}


Tensor<type, 2> ConvolutionalLayer::calculate_hidden_delta_pooling(PoolingLayer* next_layer_pointer,
                                                                  const Tensor<type, 2>& activations,
                                                                  const Tensor<type, 2>& activations_derivatives,
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
            const int filters_number = get_filters_number();
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

            Tensor<type, 2> hidden_delta(Tensor<int, 1>({images_number, filters_number, output_rows_number, output_columns_number}));

            const int size = hidden_delta.size();

            #pragma omp parallel for

            for(int tensor_index = 0; tensor_index < size; tensor_index++)
            {
                const int image_index = tensor_index/(filters_number*output_rows_number*output_columns_number);
                const int channel_index = (tensor_index/(output_rows_number*output_columns_number))%filters_number;
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

            return (hidden_delta*activations_derivatives)/(next_layers_pool_rows*next_layers_pool_columns);
        }

        case OpenNN::PoolingLayer::PoolingMethod::MaxPooling:
        {
            // Current layer's values

            const int images_number = next_layer_delta.dimension(0);
            const int filters_number = get_filters_number();
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

            Tensor<type, 2> hidden_delta(Tensor<int, 1>({images_number, filters_number, output_rows_number, output_columns_number}));

            const int size = hidden_delta.size();

            #pragma omp parallel for

            for(int tensor_index = 0; tensor_index < size; tensor_index++)
            {
                const int image_index = tensor_index/(filters_number*output_rows_number*output_columns_number);
                const int channel_index = (tensor_index/(output_rows_number*output_columns_number))%filters_number;
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

                        const double max_derivative = multiply_not_multiply(row_index - i*next_layers_row_stride, column_index - j*next_layers_column_stride);

                        sum += delta_element*max_derivative;
                    }
                }

                hidden_delta(image_index, channel_index, row_index, column_index) = sum;
            }

            return hidden_delta*activations_derivatives;
        }
    }
*/
    return Tensor<type, 2>();
}


Tensor<type, 2> ConvolutionalLayer::calculate_hidden_delta_perceptron(PerceptronLayer* next_layer_pointer,
                                                                     const Tensor<type, 2>&,
                                                                     const Tensor<type, 2>& activations_derivatives,
                                                                     const Tensor<type, 2>& next_layer_delta) const
{
/*
    // Current layer's values

    const int images_number = next_layer_delta.dimension(0);
    const int filters_number = get_filters_number();
    const int output_rows_number = get_outputs_rows_number();
    const int output_columns_number = get_outputs_columns_number();

    // Next layer's values

    const int next_layers_output_columns = next_layer_delta.dimension(1);

    const MatrixXd& next_layers_weights = next_layer_pointer->get_synaptic_weights();

    // Hidden delta calculation

    Tensor<type, 2> hidden_delta(Tensor<int, 1>({images_number, filters_number, output_rows_number, output_columns_number}));

    const int size = hidden_delta.size();

    #pragma omp parallel for

    for(int tensor_index = 0; tensor_index < size; tensor_index++)
    {
        const int image_index = tensor_index/(filters_number*output_rows_number*output_columns_number);
        const int channel_index = (tensor_index/(output_rows_number*output_columns_number))%filters_number;
        const int row_index = (tensor_index/output_columns_number)%output_rows_number;
        const int column_index = tensor_index%output_columns_number;

        double sum = 0.0;

        for(int sum_index = 0; sum_index < next_layers_output_columns; sum_index++)
        {
            const double delta_element = next_layer_delta(image_index, sum_index);

            const double weight = next_layers_weights(channel_index + row_index*filters_number + column_index*filters_number*output_rows_number, sum_index);

            sum += delta_element*weight;
        }

        hidden_delta(image_index, channel_index, row_index, column_index) = sum;
    }

    return hidden_delta*activations_derivatives;
*/
    return Tensor<type, 2>();

}


Tensor<type, 2> ConvolutionalLayer::calculate_hidden_delta_probabilistic(ProbabilisticLayer* next_layer_pointer,
                                                                     const Tensor<type, 2>&,
                                                                     const Tensor<type, 2>& activations_derivatives,
                                                                     const Tensor<type, 2>& next_layer_delta) const
{
/*
    // Current layer's values

    const int images_number = next_layer_delta.dimension(0);
    const int filters_number = get_filters_number();
    const int output_rows_number = get_outputs_rows_number();
    const int output_columns_number = get_outputs_columns_number();

    // Next layer's values

    const int next_layers_output_columns = next_layer_delta.dimension(1);

    const MatrixXd next_layers_weights = next_layer_pointer->get_synaptic_weights();

    // Hidden delta calculation

    Tensor<type, 2> hidden_delta(Tensor<int, 1>({images_number, filters_number, output_rows_number, output_columns_number}));

    const int size = hidden_delta.size();

    #pragma omp parallel for

    for(int tensor_index = 0; tensor_index < size; tensor_index++)
    {
        const int image_index = tensor_index/(filters_number*output_rows_number*output_columns_number);
        const int channel_index = (tensor_index/(output_rows_number*output_columns_number))%filters_number;
        const int row_index = (tensor_index/output_columns_number)%output_rows_number;
        const int column_index = tensor_index%output_columns_number;

        double sum = 0.0;

        for(int sum_index = 0; sum_index < next_layers_output_columns; sum_index++)
        {
            const double delta_element = next_layer_delta(image_index, sum_index);

            const double weight = next_layers_weights(channel_index + row_index*filters_number + column_index*filters_number*output_rows_number, sum_index);

            sum += delta_element*weight;
        }

        hidden_delta(image_index, channel_index, row_index, column_index) = sum;
    }

    return hidden_delta*activations_derivatives;
*/
    return Tensor<type, 2>();

}


Tensor<type, 1> ConvolutionalLayer::calculate_error_gradient(const Tensor<type, 2>& previous_layers_outputs,
                                                         const Layer::ForwardPropagation& ,
                                                         const Tensor<type, 2>& layer_deltas)
{
/*
    Tensor<type, 2> layers_inputs;

    switch (get_padding_option()) {

        case OpenNN::ConvolutionalLayer::PaddingOption::NoPadding:
        {
            layers_inputs = previous_layers_outputs;
        }
        break;

        case OpenNN::ConvolutionalLayer::PaddingOption::Same:
        {
            layers_inputs.set(Tensor<int, 1>({previous_layers_outputs.dimension(0), previous_layers_outputs.dimension(1),
                                              previous_layers_outputs.dimension(2) + get_padding_height(), previous_layers_outputs.dimension(3) + get_padding_width()}));

            for(int image_number = 0; image_number < previous_layers_outputs.dimension(0); image_number++)
            {
                layers_inputs.set_tensor(image_number, insert_padding(previous_layers_outputs.get_tensor(image_number)));
            }
        }
        break;
    }

    // Gradient declaration and values used

    const int parameters_number = get_parameters_number();

    Tensor<type, 1> layer_error_gradient(parameters_number, 0.0);

    const int images_number = layer_deltas.dimension(0);
    const int filters_number = get_filters_number();
    const int filters_channels_number = get_filters_channels_number();
    const int filters_rows_number = get_filters_rows_number();
    const int filters_columns_number = get_filters_columns_number();
    const int output_rows_number = get_outputs_rows_number();
    const int output_columns_number = get_outputs_columns_number();

    // Synaptic weights

    const int synaptic_weights_number = get_synaptic_weights().size();

    for(int gradient_index = 0; gradient_index < synaptic_weights_number; gradient_index++)
    {
        int filter_index = gradient_index%filters_number;
        int channel_index = (gradient_index/filters_number)%filters_channels_number;
        int row_index = (gradient_index/(filters_number*filters_channels_number))%filters_rows_number;
        int column_index = (gradient_index/(filters_number*filters_channels_number*filters_rows_number))%filters_columns_number;

        double sum = 0.0;

        for(int i = 0; i < images_number; i++)
        {
            for(int j = 0; j < output_rows_number; j++)
            {
                for(int k = 0; k < output_columns_number; k++)
                {
                    const double delta_element = layer_deltas(i, filter_index, j, k);

                    const double input_element = layers_inputs(i, channel_index, j*row_stride + row_index, k*column_stride + column_index);

                    sum += delta_element*input_element;
                }
            }
        }

        layer_error_gradient[gradient_index] += sum;
    }

    // Biases

    for(int gradient_index = synaptic_weights_number; gradient_index < parameters_number; gradient_index++)
    {
        int bias_index = gradient_index - synaptic_weights_number;

        double sum = 0.0;

        for(int i = 0; i < images_number; i++)
        {
            for(int j = 0; j < output_rows_number; j++)
            {
                for(int k = 0; k < output_columns_number; k++)
                {
                    const double delta_element = layer_deltas(i, bias_index, j, k);

                    sum += delta_element;
                }
            }
        }

        layer_error_gradient[gradient_index] += sum;
    }

    return layer_error_gradient;
*/
    return Tensor<type, 1>();

}


/// Returns the result of applying a single filter to a single image.
/// @param image The image.
/// @param filter The filter.

Tensor<type, 2> ConvolutionalLayer::calculate_image_convolution(const Tensor<type, 2>& image,
                                                               const Tensor<type, 2>& filter) const
{
    #ifdef __OPENNN_DEBUG__

    if(image.rank() != 3)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
              << "Tensor<type, 2> calculate_image_convolution(const Tensor<type, 2>&, const Tensor<type, 2>&) method.\n"
              << "Number of dimensions in image (" << image.rank() << ") must be 3 (channels, rows, columns).\n";

       throw logic_error(buffer.str());
    }

    if(filter.rank() != 3)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
              << "Tensor<type, 2> calculate_image_convolution(const Tensor<type, 2>&, const Tensor<type, 2>&) method.\n"
              << "Number of dimensions in filter (" << filter.rank() << ") must be 3 (channels, rows, columns).\n";

       throw logic_error(buffer.str());
    }

    #endif

    const int image_channels_number = image.dimension(0);
    const int filter_channels_number = filter.dimension(0);

    #ifdef __OPENNN_DEBUG__

        if(image_channels_number != filter_channels_number)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
                  << "void set(const Tensor<int, 1>&) method.\n"
                  << "Number of channels in image (" << image_channels_number << ") must be equal to number of channels in filter (" << filter_channels_number << ").\n";

           throw logic_error(buffer.str());
        }

    #endif

    const int image_rows_number = image.dimension(1);
    const int image_columns_number = image.dimension(2);

    const int filter_rows_number = filter.dimension(1);
    const int filter_columns_number = filter.dimension(2);

    const int outputs_rows_number = (image_rows_number - filter_rows_number)/(row_stride) + 1;
    const int outputs_columns_number = (image_columns_number - filter_columns_number)/(column_stride) + 1;

    Tensor<type, 2> convolutions(outputs_rows_number, outputs_columns_number);
/*
    for(int channel_index = 0; channel_index < filter_channels_number; channel_index++)
    {
        for(int row_index = 0; row_index < outputs_rows_number; row_index++)
        {
            for(int column_index = 0; column_index < outputs_columns_number; column_index++)
            {
                for(int window_row = 0; window_row < filter_rows_number; window_row++)
                {
                    const int row = row_index * row_stride + window_row;

                    for(int window_column = 0; window_column < filter_columns_number;  window_column++)
                    {
                        const int column = column_index * column_stride + window_column;

                        convolutions(row_index, column_index)
                                += image(channel_index, row, column)
                                * filter(channel_index, window_row, window_column);
                    }
                }
            }
        }
    }
*/
    return convolutions;
}


/// Returns the convolutional layer's activation function.

ConvolutionalLayer::ActivationFunction ConvolutionalLayer::get_activation_function() const
{
    return activation_function;
}


/// Returns the number of rows the result of applying the layer's filters to an image will have.

int ConvolutionalLayer::get_outputs_rows_number() const
{
    const int filters_rows_number = get_filters_rows_number();

    const int padding_height = get_padding_height();

    return (input_variables_dimensions[1] - filters_rows_number + padding_height)/row_stride + 1;
}


/// Returns the number of columns the result of applying the layer's filters to an image will have.

int ConvolutionalLayer::get_outputs_columns_number() const
{
    const int filters_columns_number = get_filters_columns_number();

    const int padding_width = get_padding_width();

    return (input_variables_dimensions[2] - filters_columns_number + padding_width)/column_stride + 1;
}


/// Returns a vector containing the number of channels, rows and columns of the result of applying the layer's filters to an image.

Tensor<int, 1> ConvolutionalLayer::get_outputs_dimensions() const
{
    Tensor<int, 1> outputs_dimensions(3);

    outputs_dimensions[0] = get_filters_number();
    outputs_dimensions[1] = get_outputs_rows_number();
    outputs_dimensions[2] = get_outputs_columns_number();

    return outputs_dimensions;
}


/// Returns the padding option.

ConvolutionalLayer::PaddingOption ConvolutionalLayer::get_padding_option() const
{
    return padding_option;
}


/// Returns the column stride.

int ConvolutionalLayer::get_column_stride() const
{
    return column_stride;
}


/// Returns the row stride.

int ConvolutionalLayer::get_row_stride() const
{
    return row_stride;
}


///Returns the number of filters of the layer.

int ConvolutionalLayer::get_filters_number() const
{
    return synaptic_weights.dimension(0);
}


/// Returns the number of channels of the layer's filters.

int ConvolutionalLayer::get_filters_channels_number() const
{
    return synaptic_weights.dimension(1);
}


/// Returns the number of rows of the layer's filters.

int  ConvolutionalLayer::get_filters_rows_number() const
{
    return synaptic_weights.dimension(2);
}


/// Returns the number of columns of the layer's filters.

int ConvolutionalLayer::get_filters_columns_number() const
{
    return synaptic_weights.dimension(3);
}


/// Returns the total number of columns of zeroes to be added to an image before applying a filter, which depends on the padding option set.

int ConvolutionalLayer::get_padding_width() const
{
    switch(padding_option)
    {
        case NoPadding:
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

int ConvolutionalLayer::get_padding_height() const
{
    switch(padding_option)
    {
        case NoPadding:
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


int ConvolutionalLayer::get_inputs_number() const
{
    return get_inputs_channels_number()*get_inputs_rows_number()*get_inputs_columns_number();
}


int ConvolutionalLayer::get_neurons_number() const
{
    return get_filters_number()*get_outputs_rows_number()*get_outputs_columns_number();
}


/// Returns the layer's parameters in the form of a vector.

Tensor<type, 1> ConvolutionalLayer::get_parameters() const
{
/*
    return synaptic_weights.to_vector().assemble(biases);
  */
    return Tensor<type, 1>();

}


/// Returns the number of parameters of the layer.

int ConvolutionalLayer::get_parameters_number() const
{
    const int biases_number = biases.size();

    const int synaptic_weights_number = synaptic_weights.size();

    return synaptic_weights_number + biases_number;
}


/// Sets and initializes the layer's parameters in accordance with the dimensions taken as input.
/// The initialization values are random values from a normal distribution.
/// @param filters_dimensions A vector containing the desired filters' dimensions (number of filters, number of channels, rows and columns).

void ConvolutionalLayer::set(const Tensor<int, 1>& new_inputs_dimensions, const Tensor<int, 1>& new_filters_dimensions)
{
    #ifdef __OPENNN_DEBUG__

    const int inputs_dimensions_number = new_inputs_dimensions.size();

    if(inputs_dimensions_number != 3)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "ConvolutionalLayer(const Tensor<int, 1>&) constructor.\n"
               << "Number of inputs dimensions (" << inputs_dimensions_number << ") must be 3 (channels, rows, columns).\n";

        throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const int filters_dimensions_number = new_filters_dimensions.size();

    if(filters_dimensions_number != 3)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void set(const Tensor<int, 1>&) method.\n"
               << "Number of filters dimensions (" << filters_dimensions_number << ") must be 3 (filters, rows, columns).\n";

        throw logic_error(buffer.str());
    }

    #endif
/*
    input_variables_dimensions.set(new_inputs_dimensions);

    const int filters_number = new_filters_dimensions[0];
    const int filters_channels_number = new_inputs_dimensions[0];
    const int filters_rows_number = new_filters_dimensions[1];
    const int filters_columns_number = new_filters_dimensions[2];

    biases.set(filters_number);
    biases.setRandom();

    synaptic_weights.set(Tensor<int, 1>({filters_number, filters_channels_number, filters_rows_number, filters_columns_number}));
    synaptic_weights.setRandom();
*/
}


/// Initializes the layer's biases to a given value.
/// @param value The desired value.

void ConvolutionalLayer::initialize_biases(const double& value)
{
/*
    biases.setConstant(value);
*/
}


/// Initializes the layer's synaptic weights to a given value.
/// @param value The desired value.

void ConvolutionalLayer::initialize_synaptic_weights(const double& value)
{
/*
    synaptic_weights.setConstant(value);
*/
}


/// Initializes the layer's parameters to a given value.
/// @param value The desired value.

void ConvolutionalLayer::initialize_parameters(const double& value)
{
    initialize_biases(value);

    initialize_synaptic_weights(value);
}


/// Returns the input image with rows and columns of zeroes added in accordance with the padding option set.
/// @param image The image to be padded.

Tensor<type, 2> ConvolutionalLayer::insert_padding(const Tensor<type, 2>& image) const
{
/*
    const int dimensions_number = image.rank(); // 3

    const int padding_width = get_padding_width();
    const int padding_height = get_padding_height();

    const int padding_left = padding_width/2;
    const int padding_right = padding_width/2 + padding_width%2;
    const int padding_top = padding_height/2;
    const int padding_bottom = padding_width/2 + padding_height%2;

    const int channels_number = image.dimension(0);

    const int new_rows_number = image.dimension(1) + padding_height;
    const int new_columns_number = image.dimension(2) + padding_width;

    Tensor<type, 2> new_image({channels_number, new_rows_number, new_columns_number}, 0.0);

    for(int channel = 0; channel < channels_number; channel++)
    {
        for(int row = padding_top; row < new_rows_number - padding_bottom; row++)
        {
            for(int column = padding_left; column < new_columns_number - padding_right; column++)
            {
                new_image(channel, row, column) = image(channel, row - padding_top, column - padding_left);
            }
        }
    }

    return new_image;
*/
    return Tensor<type, 2>();

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

void ConvolutionalLayer::set_synaptic_weights(const Tensor<type, 2>& new_synaptic_weights)
{
    synaptic_weights = new_synaptic_weights;
}


/// Sets the padding option.
/// @param new_padding_option The desired padding option.

void ConvolutionalLayer::set_padding_option(const ConvolutionalLayer::PaddingOption& new_padding_option)
{
    padding_option = new_padding_option;
}


/// Sets the filters' row stride.
/// @param new_stride_row The desired row stride.

void ConvolutionalLayer::set_row_stride(const int& new_stride_row)
{
    if(new_stride_row == 0)
    {
        throw ("EXCEPTION: new_stride_row must be a positive number");
    }

    row_stride = new_stride_row;
}


/// Sets the filters' column stride.
/// @param new_stride_row The desired column stride.

void ConvolutionalLayer::set_column_stride(const int& new_stride_column)
{
    if(new_stride_column == 0)
    {
        throw ("EXCEPTION: new_stride_column must be a positive number");
    }

    column_stride = new_stride_column;
}


/// Sets the synaptic weights and biases to the given values.
/// @param new_parameters A vector containing the synaptic weights and biases, in this order.

void ConvolutionalLayer::set_parameters(const Tensor<type, 1>& new_parameters)
{
    const int synaptic_weights_number = synaptic_weights.size();

    const int parameters_number = get_parameters_number();

    const int filters_number = get_filters_number();
    const int filters_channels_number = get_filters_channels_number();
    const int filters_rows_number = get_filters_rows_number();
    const int filters_columns_number = get_filters_columns_number();
/*
    synaptic_weights = new_parameters.get_subvector(0, synaptic_weights_number-1).to_tensor({filters_number, filters_channels_number, filters_rows_number, filters_columns_number});

    biases = new_parameters.get_subvector(synaptic_weights_number, parameters_number-1);
*/
}


/// Returns the layer's biases.

Tensor<type, 1> ConvolutionalLayer::get_biases() const
{
    return biases;
}


Tensor<type, 1> ConvolutionalLayer::extract_biases(const Tensor<type, 1>& parameters) const
{
/*
    return parameters.get_last(get_filters_number());
*/
    return Tensor<type, 1>();

}


/// Returns the layer's synaptic weights.

Tensor<type, 2> ConvolutionalLayer::get_synaptic_weights() const
{
   return synaptic_weights;
}


Tensor<type, 2> ConvolutionalLayer::extract_synaptic_weights(const Tensor<type, 1>& parameters) const
{
/*
    return parameters.get_first(synaptic_weights.size()).to_tensor({get_filters_number(), get_filters_channels_number(), get_filters_rows_number(), get_filters_columns_number()});
*/
    return Tensor<type, 2>();
}

/// Returns the number of channels of the input.

int ConvolutionalLayer::get_inputs_channels_number() const
{
    return input_variables_dimensions[0];
}


/// Returns the number of rows of the input.

int ConvolutionalLayer::get_inputs_rows_number() const
{
    return input_variables_dimensions[1];
}


/// Returns the number of columns of the input.

int ConvolutionalLayer::get_inputs_columns_number() const
{
    return input_variables_dimensions[2];
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
