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

ConvolutionalLayer::ConvolutionalLayer(const Vector<size_t>& new_inputs_dimensions, const Vector<size_t>& new_filters_dimensions) : Layer()
{
    layer_type = Layer::Convolutional;

    set(new_inputs_dimensions, new_filters_dimensions);
}


/// Returns a boolean, true if convolutional layer is empty and false otherwise.

bool ConvolutionalLayer::is_empty() const
{
    if(biases.empty() && synaptic_weights.empty())
    {
        return true;
    }

    return false;
}


/// Returns the result of applying the previously set activation function to a batch of images.
/// @param convolutions The batch of images.

Tensor<double> ConvolutionalLayer::calculate_activations(const Tensor<double>& convolutions) const
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

    return Tensor<double>();
}


/// Returns the result of applying the convolutional layer's filters and biases to a batch of images.
/// @param inputs The batch of images.

Tensor<double> ConvolutionalLayer::calculate_convolutions(const Tensor<double>& inputs) const
{
    #ifdef __OPENNN_DEBUG__

    const size_t inputs_dimensions_number = inputs.get_dimensions_number();

    if(inputs_dimensions_number != 4)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
              << "Tensor<double> calculate_convolutions(const Tensor<double>&) method.\n"
              << "Number of inputs dimensions (" << inputs_dimensions_number << ") must be 4 (batch, channels, rows, columns).\n";

       throw logic_error(buffer.str());
    }

    const size_t inputs_channels_number = inputs.get_dimension(1);
    const size_t filters_channels_number  = get_filters_channels_number();

    if(filters_channels_number != inputs_channels_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
              << "Tensor<double> calculate_convolutions(const Tensor<double>&) method.\n"
              << "Number of input channels (" << inputs_channels_number << ") must be equal to number of filters channels (" << filters_channels_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    // Inputs

    const size_t images_number = inputs.get_dimension(0);

    // Filters

    const size_t filters_number = get_filters_number();

    // Outputs

    const size_t outputs_rows_number = get_outputs_rows_number();
    const size_t outputs_columns_number = get_outputs_columns_number();

    Tensor<double> convolutions({images_number, filters_number, outputs_rows_number, outputs_columns_number}, 0.0);

    Tensor<double> image(Vector<size_t>({filters_number, outputs_rows_number, outputs_columns_number}));

    Matrix<double> convolution(outputs_rows_number, outputs_columns_number);

    #pragma omp parallel for private(image, convolution)

    for(size_t image_index = 0; image_index < images_number; image_index++)
    {
        image = inputs.get_tensor(image_index);

        if(padding_option == Same) image = insert_padding(image);

        for(size_t filter_index = 0; filter_index < filters_number; filter_index++)
        {
            convolution = calculate_image_convolution(image, synaptic_weights.get_tensor(filter_index)) + biases[filter_index];

            convolutions.set_matrix(image_index, filter_index, convolution);
        }
    }

    return convolutions;
}


/// Returns the result of applying the parameters passed as argument to a batch of images.
/// @param inputs The batch of images.
/// @param parameters The parameters.

Tensor<double> ConvolutionalLayer::calculate_convolutions(const Tensor<double>& inputs, const Vector<double>& parameters) const
{
    #ifdef __OPENNN_DEBUG__

    const size_t inputs_dimensions_number = inputs.get_dimensions_number();

    if(inputs_dimensions_number != 4)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
              << "Tensor<double> calculate_convolutions(const Tensor<double>&) method.\n"
              << "Number of inputs dimensions (" << inputs_dimensions_number << ") must be 4 (batch, channels, rows, columns).\n";

       throw logic_error(buffer.str());
    }

    const size_t inputs_channels_number = inputs.get_dimension(1);
    const size_t filters_channels_number  = get_filters_channels_number();

    if(filters_channels_number != inputs_channels_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
              << "Tensor<double> calculate_convolutions(const Tensor<double>&) method.\n"
              << "Number of input channels (" << inputs_channels_number << ") must be equal to number of filters channels (" << filters_channels_number << ").\n";

       throw logic_error(buffer.str());
    }

//    const size_t inputs_rows_number = inputs.get_dimension(2);
//    const size_t inputs_columns_number = inputs.get_dimension(3);

//    const size_t filters_rows_number  = get_filters_rows_number();
//    const size_t filters_columns_number = get_filters_columns_number();

//    if(static_cast<int>(inputs_rows_number - filters_rows_number + padding_height) < 0)
//    {
//       ostringstream buffer;

//       buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
//              << "Tensor<double> calculate_combinations(const Tensor<double>&) method.\n"
//              << "Resulting outputs rows number is zero or negative.\n";

//       throw logic_error(buffer.str());
//    }

//    if(static_cast<int>(inputs_columns_number - filters_columns_number + padding_width) < 0)
//    {
//       ostringstream buffer;

//       buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
//              << "Tensor<double> calculate_combinations(const Tensor<double>&) method.\n"
//              << "Resulting outputs columns number is zero or negative.\n";

//       throw logic_error(buffer.str());
//    }

    #endif

    // Weights and biases

    Tensor<double> new_synaptic_weights = extract_synaptic_weights(parameters);

    Vector<double> new_biases = extract_biases(parameters);

    // Inputs

    const size_t images_number = inputs.get_dimension(0);

    // Filters

    const size_t filters_number = get_filters_number();

    // Outputs

    const size_t outputs_rows_number = get_outputs_rows_number();
    const size_t outputs_columns_number = get_outputs_columns_number();

    Tensor<double> convolutions({images_number, filters_number, outputs_rows_number, outputs_columns_number}, 0.0);

    for(size_t image_index = 0; image_index < images_number; image_index++)
    {
        Tensor<double> image = inputs.get_tensor(image_index);

        if(padding_option == Same) image = insert_padding(image);

        for(size_t filter_index = 0; filter_index < filters_number; filter_index++)
        {
            const Matrix<double> current_convolutions = calculate_image_convolution(image, new_synaptic_weights.get_tensor(filter_index)) + new_biases[filter_index];

            convolutions.set_matrix(image_index, filter_index, current_convolutions);
        }
    }

    return convolutions;
}


/// Returns the result of applying the derivative of the previously set activation function to a batch of images.
/// @param convolutions The batch of images.

Tensor<double> ConvolutionalLayer::calculate_activations_derivatives(const Tensor<double>& combinations) const
{
    #ifdef __OPENNN_DEBUG__

        const size_t combinations_dimensions_number = combinations.get_dimensions_number();

        if(combinations_dimensions_number != 4)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
                  << "Tensor<double> calculate_activations_derivatives(const Tensor<double>&) method.\n"
                  << "Number of combinations dimensions (" << combinations_dimensions_number << ") must be 4 (batch, filters, rows, columns).\n";

           throw logic_error(buffer.str());
        }

    #endif

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

    return Tensor<double>();
}


/// Returns the output of the convolutional layer applied to a batch of images.
/// @param inputs The batch of images.

Tensor<double> ConvolutionalLayer::calculate_outputs(const Tensor<double>& inputs)
{
    return calculate_activations(calculate_convolutions(inputs));
}


/// Returns the output of the convolutional layer with given parameters applied to a batch of images.
/// @param inputs The batch of images.
/// @param parameters The parameters.

Tensor<double> ConvolutionalLayer::calculate_outputs(const Tensor<double>& inputs, const Vector<double>& parameters)
{
    return calculate_activations(calculate_convolutions(inputs, parameters));
}


Layer::FirstOrderActivations ConvolutionalLayer::calculate_first_order_activations(const Tensor<double>& inputs)
{
    FirstOrderActivations first_order_activations;

    const Tensor<double> combinations = calculate_convolutions(inputs);

    first_order_activations.activations = calculate_activations(combinations);

    first_order_activations.activations_derivatives = calculate_activations_derivatives(combinations);

    return first_order_activations;
}


Tensor<double> ConvolutionalLayer::calculate_output_delta(const Tensor<double>& activations_derivatives, const Tensor<double>& output_gradient) const
{
    return activations_derivatives*output_gradient;
}


Tensor<double> ConvolutionalLayer::calculate_hidden_delta(Layer* next_layer_pointer,
                                                          const Tensor<double>& activations,
                                                          const Tensor<double>& activations_derivatives,
                                                          const Tensor<double>& next_layer_delta) const
{
    const Layer::LayerType layer_type = next_layer_pointer->get_type();

    if(layer_type == LayerType::Convolutional)
    {   
        ConvolutionalLayer* convolutional_layer = dynamic_cast<ConvolutionalLayer*>(next_layer_pointer);

        return calculate_hidden_delta_convolutional(convolutional_layer, activations, activations_derivatives, next_layer_delta);
    }
    else if(layer_type == LayerType::Pooling)
    {
        PoolingLayer* pooling_layer = dynamic_cast<PoolingLayer*>(next_layer_pointer);

        return calculate_hidden_delta_pooling(pooling_layer, activations, activations_derivatives, next_layer_delta);
    }
    else if(layer_type == LayerType::Perceptron)
    {
        PerceptronLayer* perceptron_layer = dynamic_cast<PerceptronLayer*>(next_layer_pointer);

        return calculate_hidden_delta_perceptron(perceptron_layer, activations, activations_derivatives, next_layer_delta);
    }
    else if(layer_type == LayerType::Probabilistic)
    {
        ProbabilisticLayer* probabilistic_layer = dynamic_cast<ProbabilisticLayer*>(next_layer_pointer);

        return calculate_hidden_delta_probabilistic(probabilistic_layer, activations, activations_derivatives, next_layer_delta);
    }

    return Tensor<double>();
}


Tensor<double> ConvolutionalLayer::calculate_hidden_delta_convolutional(ConvolutionalLayer* next_layer_pointer,
                                                                        const Tensor<double>&,
                                                                        const Tensor<double>& activations_derivatives,
                                                                        const Tensor<double>& next_layer_delta) const
{
    // Current layer's values

    const size_t filters_number = get_filters_number();
    const size_t output_rows_number = get_outputs_rows_number();
    const size_t output_columns_number = get_outputs_columns_number();

    // Next layer's values

    const size_t next_layers_filters_number = next_layer_pointer->get_filters_number();
    const size_t next_layers_filter_rows = next_layer_pointer->get_filters_rows_number();
    const size_t next_layers_filter_columns = next_layer_pointer->get_filters_columns_number();
    const size_t next_layers_output_rows = next_layer_pointer->get_outputs_rows_number();
    const size_t next_layers_output_columns = next_layer_pointer->get_outputs_columns_number();
    const size_t next_layers_row_stride = next_layer_pointer->get_row_stride();
    const size_t next_layers_column_stride = next_layer_pointer->get_column_stride();
    const Tensor<double> next_layers_weights = next_layer_pointer->get_synaptic_weights();

    Tensor<double> hidden_delta({next_layer_delta.get_dimension(0), filters_number, output_rows_number, output_columns_number}, 0.0);

    const size_t size = hidden_delta.size();

    #pragma omp parallel for

    for(size_t tensor_index = 0; tensor_index < size; tensor_index++)
    {
        const size_t image_index = tensor_index/(filters_number*output_rows_number*output_columns_number);
        const size_t channel_index = (tensor_index/(output_rows_number * output_columns_number))%filters_number;
        const size_t row_index = (tensor_index/output_columns_number)%output_rows_number;
        const size_t column_index = tensor_index%output_columns_number;

        double sum = 0.0;

        for(size_t i = 0; i < next_layers_filters_number; i++)
        {
            for(size_t j = 0; j < next_layers_output_rows; j++)
            {
                if(static_cast<int>(row_index) - static_cast<int>(j*next_layers_row_stride) >= 0
                && row_index - j*next_layers_row_stride < next_layers_filter_rows)
                {
                    for(size_t k = 0; k < next_layers_output_columns; k++)
                    {
                        const double delta_element = next_layer_delta(image_index, i, j, k);

                        const double weight = next_layers_weights(i, channel_index, row_index - j*next_layers_row_stride, column_index - k*next_layers_column_stride);

                        if(static_cast<int>(column_index) - static_cast<int>(k*next_layers_column_stride) >= 0
                        && column_index - k*next_layers_column_stride < next_layers_filter_columns)
                        {
                            sum += delta_element * weight;
                        }
                    }
                }
            }
        }

        hidden_delta(image_index, channel_index, row_index, column_index) += sum;
    }

    return hidden_delta * activations_derivatives;
}


Tensor<double> ConvolutionalLayer::calculate_hidden_delta_pooling(PoolingLayer* next_layer_pointer,
                                                                  const Tensor<double>&,
                                                                  const Tensor<double>& activations_derivatives,
                                                                  const Tensor<double>& next_layer_delta) const
{
    switch(next_layer_pointer->get_pooling_method())
    {
        case OpenNN::PoolingLayer::PoolingMethod::NoPooling:
        {
            return next_layer_delta;
        }

        case OpenNN::PoolingLayer::PoolingMethod::AveragePooling:
        {
            Tensor<double> hidden_delta({next_layer_delta.get_dimension(0), get_filters_number(), get_outputs_rows_number(), get_outputs_columns_number()}, 0.0);

            const size_t size = hidden_delta.size();

            for(size_t s = 0; s < size; s++)
            {
                int image_index = static_cast<int>(s/(get_filters_number() * get_outputs_rows_number() * get_outputs_columns_number()));
                int channel_index = static_cast<int>((s/(get_outputs_rows_number() * get_outputs_columns_number()))%(get_filters_number()));
                int row_index = static_cast<int>((s/(get_outputs_columns_number()))%(get_outputs_rows_number()));
                int column_index = static_cast<int>(s%(get_outputs_columns_number()));

                double sum = 0.0;

                for(int i = 0; i < static_cast<int>(next_layer_pointer->get_outputs_rows_number()); i++)
                {
                    if((row_index - (i * static_cast<int>(next_layer_pointer->get_row_stride()))) >= 0 &&
                            (row_index - (i * static_cast<int>(next_layer_pointer->get_row_stride()))) < static_cast<int>(next_layer_pointer->get_pool_rows_number()))
                    {
                        for(int j = 0; j < static_cast<int>(next_layer_pointer->get_outputs_columns_number()); j++)
                        {
                            if((column_index - (j * static_cast<int>(next_layer_pointer->get_row_stride()))) >= 0 &&
                                    (column_index - (j * static_cast<int>(next_layer_pointer->get_row_stride()))) < static_cast<int>(next_layer_pointer->get_pool_columns_number()))
                            {
                                sum += next_layer_delta(static_cast<size_t>(image_index), static_cast<size_t>(channel_index), static_cast<size_t>(i), static_cast<size_t>(j));
                            }
                        }
                    }
                }

                hidden_delta(static_cast<size_t>(image_index), static_cast<size_t>(channel_index), static_cast<size_t>(row_index), static_cast<size_t>(column_index)) +=
                        sum/(next_layer_pointer->get_pool_rows_number() * next_layer_pointer->get_pool_columns_number());
            }

            return hidden_delta * activations_derivatives;
        }

        case OpenNN::PoolingLayer::PoolingMethod::MaxPooling:
        {
            return Tensor<double>({next_layer_delta.get_dimension(0), get_filters_number(), get_outputs_rows_number(), get_outputs_columns_number()}, 0.0);
        }
    }
    return Tensor<double>();
}


Tensor<double> ConvolutionalLayer::calculate_hidden_delta_perceptron(PerceptronLayer* next_layer_pointer,
                                                                     const Tensor<double>&,
                                                                     const Tensor<double>& activations_derivatives,
                                                                     const Tensor<double>& next_layer_delta) const
{
    Tensor<double> hidden_delta(Vector<size_t>({next_layer_delta.get_dimension(0), activations_derivatives.get_dimension(1), get_outputs_rows_number(), get_outputs_columns_number()}));

    const size_t size = hidden_delta.size();

    for(size_t index = 0; index < size; index++)
    {
        size_t image_index = index/(activations_derivatives.get_dimension(1) * get_outputs_rows_number() * get_outputs_columns_number());
        size_t channel_index = (index/(get_outputs_rows_number() * get_outputs_columns_number()))%(activations_derivatives.get_dimension(1));
        size_t row_index = (index/(get_outputs_columns_number()))%(get_outputs_rows_number());
        size_t column_index = index%(get_outputs_columns_number());

        double sum = 0.0;

        for(size_t s = 0; s < next_layer_delta.get_dimension(1); s++)
        {
            sum += next_layer_delta(image_index, s) * next_layer_pointer->get_synaptic_weights()(
                        channel_index + row_index * activations_derivatives.get_dimension(1) + column_index * activations_derivatives.get_dimension(1) * get_outputs_rows_number(), s);
        }

        hidden_delta(static_cast<size_t>(image_index), static_cast<size_t>(channel_index), static_cast<size_t>(row_index), static_cast<size_t>(column_index)) += sum;
    }

    return hidden_delta * activations_derivatives;
}


/// @todo

Tensor<double> ConvolutionalLayer::calculate_hidden_delta_probabilistic(ProbabilisticLayer* next_layer_pointer,
                                                                     const Tensor<double>&,
                                                                     const Tensor<double>& activations_derivatives,
                                                                     const Tensor<double>& next_layer_delta) const
{
    return Tensor<double>();
}


Vector<double> ConvolutionalLayer::calculate_error_gradient(const Tensor<double>& previous_layers_outputs,
                                                         const Layer::FirstOrderActivations& ,
                                                         const Tensor<double>& layer_deltas)
{
    Tensor<double> layers_inputs;

    switch (get_padding_option()) {

        case OpenNN::ConvolutionalLayer::PaddingOption::NoPadding:
        {
            layers_inputs = previous_layers_outputs;
        }
        break;

        case OpenNN::ConvolutionalLayer::PaddingOption::Same:
        {
            layers_inputs.set(Vector<size_t>({previous_layers_outputs.get_dimension(0), previous_layers_outputs.get_dimension(1),
                                              previous_layers_outputs.get_dimension(2) + get_padding_height(), previous_layers_outputs.get_dimension(3) + get_padding_width()}));

            for(size_t image_number = 0; image_number < previous_layers_outputs.get_dimension(0); image_number++)
            {
                layers_inputs.set_tensor(image_number, insert_padding(previous_layers_outputs.get_tensor(image_number)));
            }
        }
        break;
    }

    const size_t parameters_number = get_parameters_number();

    const size_t synaptic_weights_number = get_synaptic_weights().size();

    Vector<double> layer_error_gradient(parameters_number, 0.0);

    // Synaptic weights

    for(size_t index = 0; index < synaptic_weights_number; index++)
    {
        size_t filter_index = index%get_filters_number();
        size_t channel_index = (index/get_filters_number())%(get_filters_channels_number());
        size_t row_index = (index/(get_filters_number() * get_filters_channels_number()))%(get_filters_rows_number());
        size_t column_index = (index/(get_filters_number() * get_filters_channels_number() * get_filters_rows_number()))%(get_filters_columns_number());

        double sum = 0.0;

        for(size_t i = 0; i < layer_deltas.get_dimension(0); i++)
        {
            for(size_t j = 0; j < get_outputs_rows_number(); j++)
            {
                for(size_t k = 0; k < get_outputs_columns_number(); k++)
                {
                    sum += layer_deltas(i, filter_index, j, k) * layers_inputs(i, channel_index, j * row_stride + row_index, k * column_stride + column_index);
                }
            }
        }

        layer_error_gradient[index] += sum;
    }

    // Biases

    for(size_t index = synaptic_weights_number; index < parameters_number; index++)
    {
        size_t bias_index = index - synaptic_weights_number;

        double sum = 0.0;

        for(size_t i = 0; i < layer_deltas.get_dimension(0); i++)
        {
            for(size_t j = 0; j < get_outputs_rows_number(); j++)
            {
                for(size_t k = 0; k < get_outputs_columns_number(); k++)
                {
                    sum += layer_deltas(i, bias_index, j, k);
                }
            }
        }

        layer_error_gradient[index] += sum;
    }

    return layer_error_gradient;
}


/// Returns the result of applying a single filter to a single image.
/// @param image The image.
/// @param filter The filter.

Matrix<double> ConvolutionalLayer::calculate_image_convolution(const Tensor<double>& image,
                                                               const Tensor<double>& filter) const
{
    #ifdef __OPENNN_DEBUG__

    if(image.get_dimensions_number() != 3)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
              << "Matrix<double> calculate_image_convolution(const Tensor<double>&, const Tensor<double>&) method.\n"
              << "Number of dimensions in image (" << image.get_dimensions_number() << ") must be 3 (channels, rows, columns).\n";

       throw logic_error(buffer.str());
    }

    if(filter.get_dimensions_number() != 3)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
              << "Matrix<double> calculate_image_convolution(const Tensor<double>&, const Tensor<double>&) method.\n"
              << "Number of dimensions in filter (" << filter.get_dimensions_number() << ") must be 3 (channels, rows, columns).\n";

       throw logic_error(buffer.str());
    }

    #endif

    const size_t image_channels_number = image.get_dimension(0);
    const size_t filter_channels_number = filter.get_dimension(0);

    #ifdef __OPENNN_DEBUG__

        if(image_channels_number != filter_channels_number)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
                  << "void set(const Vector<size_t>&) method.\n"
                  << "Number of channels in image (" << image_channels_number << ") must be equal to number of channels in filter (" << filter_channels_number << ").\n";

           throw logic_error(buffer.str());
        }

    #endif

    const size_t image_rows_number = image.get_dimension(1);
    const size_t image_columns_number = image.get_dimension(2);

    const size_t filter_rows_number = filter.get_dimension(1);
    const size_t filter_columns_number = filter.get_dimension(2);

    const size_t outputs_rows_number = (image_rows_number - filter_rows_number)/(row_stride) + 1;
    const size_t outputs_columns_number = (image_columns_number - filter_columns_number)/(column_stride) + 1;

    Matrix<double> convolutions(outputs_rows_number, outputs_columns_number, 0.0);

    for(size_t channel_index = 0; channel_index < filter_channels_number; channel_index++)
    {
        for(size_t row_index = 0; row_index < outputs_rows_number; row_index++)
        {
            for(size_t column_index = 0; column_index < outputs_columns_number; column_index++)
            {
                for(size_t window_row = 0; window_row < filter_rows_number; window_row++)
                {
                    const size_t row = row_index * row_stride + window_row;

                    for(size_t window_column = 0; window_column < filter_columns_number;  window_column++)
                    {
                        const size_t column = column_index * column_stride + window_column;

                        convolutions(row_index, column_index)
                                += image(channel_index, row, column)
                                * filter(channel_index, window_row, window_column);
                    }
                }
            }
        }
    }

    return convolutions;
}


/// Returns the convolutional layer's activation function.

ConvolutionalLayer::ActivationFunction ConvolutionalLayer::get_activation_function() const
{
    return activation_function;
}


/// Returns the number of rows the result of applying the layer's filters to an image will have.

size_t ConvolutionalLayer::get_outputs_rows_number() const
{
    const size_t filters_rows_number = get_filters_rows_number();

    const size_t padding_height = get_padding_height();

    return (inputs_dimensions[1] - filters_rows_number + padding_height)/row_stride + 1;
}


/// Returns the number of columns the result of applying the layer's filters to an image will have.

size_t ConvolutionalLayer::get_outputs_columns_number() const
{
    const size_t filters_columns_number = get_filters_columns_number();

    const size_t padding_width = get_padding_width();

    return (inputs_dimensions[2] - filters_columns_number + padding_width)/column_stride + 1;
}


/// Returns a vector containing the number of channels, rows and columns of the result of applying the layer's filters to an image.

Vector<size_t> ConvolutionalLayer::get_outputs_dimensions() const
{
    Vector<size_t> outputs_dimensions(3);

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

size_t ConvolutionalLayer::get_column_stride() const
{
    return column_stride;
}


/// Returns the row stride.

size_t ConvolutionalLayer::get_row_stride() const
{
    return row_stride;
}


///Returns the number of filters of the layer.

size_t ConvolutionalLayer::get_filters_number() const
{
    return synaptic_weights.get_dimension(0);
}


/// Returns the number of channels of the layer's filters.

size_t ConvolutionalLayer::get_filters_channels_number() const
{
    return synaptic_weights.get_dimension(1);
}


/// Returns the number of rows of the layer's filters.

size_t  ConvolutionalLayer::get_filters_rows_number() const
{
    return synaptic_weights.get_dimension(2);
}


/// Returns the number of columns of the layer's filters.

size_t ConvolutionalLayer::get_filters_columns_number() const
{
    return synaptic_weights.get_dimension(3);
}


/// Returns the total number of columns of zeroes to be added to an image before applying a filter, which depends on the padding option set.

size_t ConvolutionalLayer::get_padding_width() const
{
    switch(padding_option)
    {
        case NoPadding:
        {
            return 0;
        }

        case Same:
        {
            return column_stride*(inputs_dimensions[2] - 1) - inputs_dimensions[2] + get_filters_columns_number();
        }
    }

    return 0;
}


/// Returns the total number of rows of zeroes to be added to an image before applying a filter, which depends on the padding option set.

size_t ConvolutionalLayer::get_padding_height() const
{
    switch(padding_option)
    {
        case NoPadding:
        {
            return 0;
        }

        case Same:
        {
            return row_stride*(inputs_dimensions[1] - 1) - inputs_dimensions[1] + get_filters_rows_number();
        }
    }

    return 0;
}


size_t ConvolutionalLayer::get_inputs_number() const
{
    return get_inputs_channels_number()*get_inputs_rows_number()*get_inputs_columns_number();
}


size_t ConvolutionalLayer::get_neurons_number() const
{
    return get_filters_number()*get_outputs_rows_number()*get_outputs_columns_number();
}


/// Returns the layer's parameters in the form of a vector.

Vector<double> ConvolutionalLayer::get_parameters() const
{
    return synaptic_weights.to_vector().assemble(biases);
}


/// Returns the number of parameters of the layer.

size_t ConvolutionalLayer::get_parameters_number() const
{
    const size_t biases_number = biases.size();

    const size_t synaptic_weights_number = synaptic_weights.size();

    return synaptic_weights_number + biases_number;
}


/// Sets and initializes the layer's parameters in accordance with the dimensions taken as input.
/// The initialization values are random values from a normal distribution.
/// @param filters_dimensions A vector containing the desired filters' dimensions (number of filters, number of channels, rows and columns).

void ConvolutionalLayer::set(const Vector<size_t>& new_inputs_dimensions, const Vector<size_t>& new_filters_dimensions)
{
    #ifdef __OPENNN_DEBUG__

    const size_t inputs_dimensions_number = new_inputs_dimensions.size();

    if(inputs_dimensions_number != 3)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "ConvolutionalLayer(const Vector<size_t>&) constructor.\n"
               << "Number of inputs dimensions (" << inputs_dimensions_number << ") must be 3 (channels, rows, columns).\n";

        throw logic_error(buffer.str());
    }

    #endif

    #ifdef __OPENNN_DEBUG__

    const size_t filters_dimensions_number = new_filters_dimensions.size();

    if(filters_dimensions_number != 3)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ConvolutionalLayer class.\n"
               << "void set(const Vector<size_t>&) method.\n"
               << "Number of filters dimensions (" << filters_dimensions_number << ") must be 3 (filters, rows, columns).\n";

        throw logic_error(buffer.str());
    }

    #endif

    inputs_dimensions.set(new_inputs_dimensions);

    const size_t filters_number = new_filters_dimensions[0];
    const size_t filters_channels_number = new_inputs_dimensions[0];
    const size_t filters_rows_number = new_filters_dimensions[1];
    const size_t filters_columns_number = new_filters_dimensions[2];

    biases.set(filters_number);
    biases.randomize_normal();

    synaptic_weights.set(Vector<size_t>({filters_number, filters_channels_number, filters_rows_number, filters_columns_number}));
    synaptic_weights.randomize_normal();
}


/// Initializes the layer's biases to a given value.
/// @param value The desired value.

void ConvolutionalLayer::initialize_biases(const double& value)
{
    biases.initialize(value);
}


/// Initializes the layer's synaptic weights to a given value.
/// @param value The desired value.

void ConvolutionalLayer::initialize_synaptic_weights(const double& value)
{
    synaptic_weights.initialize(value);
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

Tensor<double> ConvolutionalLayer::insert_padding(const Tensor<double>& image) const
{
    const size_t dimensions_number = image.get_dimensions_number(); // 3

    const size_t padding_width = get_padding_width();
    const size_t padding_height = get_padding_height();

    const size_t padding_left = padding_width/2;
    const size_t padding_right = padding_width/2 + padding_width%2;
    const size_t padding_top = padding_height/2;
    const size_t padding_bottom = padding_width/2 + padding_height%2;

    const size_t channels_number = image.get_dimension(0);

    const size_t new_rows_number = image.get_dimension(1) + padding_height;
    const size_t new_columns_number = image.get_dimension(2) + padding_width;

    Tensor<double> new_image({channels_number, new_rows_number, new_columns_number}, 0.0);

    for(size_t channel = 0; channel < channels_number; channel++)
    {
        for(size_t row = padding_top; row < new_rows_number - padding_bottom; row++)
        {
            for(size_t column = padding_left; column < new_columns_number - padding_right; column++)
            {
                new_image(channel, row, column) = image(channel, row - padding_top, column - padding_left);
            }
        }
    }

    return new_image;
}


/// Sets the layer's activation function.
/// @param new_activation_function The desired activation function.

void ConvolutionalLayer::set_activation_function(const ConvolutionalLayer::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


/// Sets the layer's biases.
/// @param new_biases The desired biases.

void ConvolutionalLayer::set_biases(const Vector<double>& new_biases)
{
    biases = new_biases;
}


/// Sets the layer's synaptic weights.
/// @param new_synaptic_weights The desired synaptic weights.

void ConvolutionalLayer::set_synaptic_weights(const Tensor<double>& new_synaptic_weights)
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

void ConvolutionalLayer::set_row_stride(const size_t& new_stride_row)
{
    if(new_stride_row == 0)
    {
        throw ("EXCEPTION: new_stride_row must be a positive number");
    }

    row_stride = new_stride_row;
}


/// Sets the filters' column stride.
/// @param new_stride_row The desired column stride.

void ConvolutionalLayer::set_column_stride(const size_t& new_stride_column)
{
    if(new_stride_column == 0)
    {
        throw ("EXCEPTION: new_stride_column must be a positive number");
    }

    column_stride = new_stride_column;
}


/// Sets the synaptic weights and biases to the given values.
/// @param new_parameters A vector containing the synaptic weights and biases, in this order.

void ConvolutionalLayer::set_parameters(const Vector<double>& new_parameters)
{
    const size_t synaptic_weights_number = synaptic_weights.size();

    const size_t parameters_number = get_parameters_number();

    const size_t filters_number = get_filters_number();
    const size_t filters_channels_number = get_filters_channels_number();
    const size_t filters_rows_number = get_filters_rows_number();
    const size_t filters_columns_number = get_filters_columns_number();

    synaptic_weights = new_parameters.get_subvector(0, synaptic_weights_number-1).to_tensor({filters_number, filters_channels_number, filters_rows_number, filters_columns_number});

    biases = new_parameters.get_subvector(synaptic_weights_number, parameters_number-1);
}


/// Returns the layer's biases.

Vector<double> ConvolutionalLayer::get_biases() const
{
    return biases;
}


Vector<double> ConvolutionalLayer::extract_biases(const Vector<double>& parameters) const
{
    return parameters.get_last(get_filters_number());
}


/// Returns the layer's synaptic weights.

Tensor<double> ConvolutionalLayer::get_synaptic_weights() const
{
   return synaptic_weights;
}


Tensor<double> ConvolutionalLayer::extract_synaptic_weights(const Vector<double>& parameters) const
{
    return parameters.get_first(synaptic_weights.size()).to_tensor({get_filters_number(), get_filters_channels_number(), get_filters_rows_number(), get_filters_columns_number()});
}

/// Returns the number of channels of the input.

size_t ConvolutionalLayer::get_inputs_channels_number() const
{
    return inputs_dimensions[0];
}


/// Returns the number of rows of the input.

size_t ConvolutionalLayer::get_inputs_rows_number() const
{
    return inputs_dimensions[1];
}


/// Returns the number of columns of the input.

size_t ConvolutionalLayer::get_inputs_columns_number() const
{
    return inputs_dimensions[2];
}


}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
