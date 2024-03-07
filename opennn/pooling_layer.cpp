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
/// @param new_input_variables_dimensions A vector containing the new number of channels, rows and raw_variables for the input.

PoolingLayer::PoolingLayer(const Tensor<Index, 1>& new_input_variables_dimensions) : Layer()
{
    set_default();
}


/// Input size setter constructor.
/// After setting new dimensions for the input, it creates an empty PoolingLayer object.
/// @param new_input_variables_dimensions A vector containing the desired number of rows and raw_variables for the input.
/// @param pool_dimensions A vector containing the desired number of rows and raw_variables for the pool.

PoolingLayer::PoolingLayer(const Tensor<Index, 1>& new_input_variables_dimensions, const Tensor<Index, 1>& pool_dimensions) : Layer()
{ 
    set(new_input_variables_dimensions, pool_dimensions);

    inputs_dimensions = new_input_variables_dimensions;

    pool_rows_number = pool_dimensions[0];
    pool_raw_variables_number = pool_dimensions[1];

    set_default();
}
/// Returns the number of neurons the layer applies to an image.

Index PoolingLayer::get_neurons_number() const
{
    return get_outputs_rows_number() * get_outputs_raw_variables_number();
}


/// Returns the layer's outputs dimensions.

Tensor<Index, 1> PoolingLayer::get_outputs_dimensions() const
{
    Tensor<Index, 1> outputs_dimensions(3);

    outputs_dimensions[0] = get_outputs_rows_number();
    outputs_dimensions[1] = get_outputs_raw_variables_number();
    outputs_dimensions[2] = inputs_dimensions[2];

    return outputs_dimensions;
}


/// Returns the number of inputs of the layer.

Index PoolingLayer::get_inputs_number() const
{
    return inputs_dimensions.size();
}



/// Returns the number of rows of the layer's input.

Index PoolingLayer::get_inputs_rows_number() const
{
    return inputs_dimensions[0];
}


/// Returns the number of raw_variables of the layer's input.

Index PoolingLayer::get_inputs_raw_variables_number() const
{
    return inputs_dimensions[1];
}


/// Returns the number of channels of the layers' input.

Index PoolingLayer::get_channels_number() const
{
    return inputs_dimensions[2];
}


/// Returns the number of rows of the layer's output.

Index PoolingLayer::get_outputs_rows_number() const
{
    type padding = type(0);

    const Index inputs_rows_number = get_inputs_rows_number();

    return (inputs_rows_number - pool_rows_number + 2*padding)/row_stride + 1;
}


/// Returns the number of raw_variables of the layer's output.

Index PoolingLayer::get_outputs_raw_variables_number() const
{
    type padding = type(0);

    const Index inputs_raw_variables_number = get_inputs_raw_variables_number();

    return (inputs_raw_variables_number - pool_raw_variables_number + 2*padding)/column_stride + 1;
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

Index PoolingLayer::get_raw_variable_stride() const
{
    return column_stride;
}


/// Returns the number of rows of the pooling filter.

Index PoolingLayer::get_pool_rows_number() const
{
    return pool_rows_number;
}


/// Returns the number of raw_variables of the pooling filter.

Index PoolingLayer::get_pool_raw_variables_number() const
{
    return pool_raw_variables_number;
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

Tensor<Index, 1> PoolingLayer::get_inputs_dimensions() const
{
    return inputs_dimensions;
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


void PoolingLayer::set(const Tensor<Index, 1>& new_input_variables_dimensions, const Tensor<Index, 1>& new_pool_dimensions)
{
    inputs_dimensions = new_input_variables_dimensions;

    pool_rows_number = new_pool_dimensions[0];
    pool_raw_variables_number = new_pool_dimensions[1];

    set_default();
}

void PoolingLayer::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}

/// Sets the number of rows of the layer's input.
/// @param new_input_rows_number The desired rows number.

void PoolingLayer::set_inputs_dimensions(const Tensor<Index, 1>& new_inputs_dimensions)
{
    inputs_dimensions = new_inputs_dimensions;
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
/// @param new_raw_variable_stride The desired column stride.

void PoolingLayer::set_raw_variable_stride(const Index& new_raw_variable_stride)
{
    column_stride = new_raw_variable_stride;
}


/// Sets the pooling filter's dimensions.
/// @param new_pool_rows_number The desired number of rows.
/// @param new_pool_raw_variables_number The desired number of raw_variables.

void PoolingLayer::set_pool_size(const Index& new_pool_rows_number,
                                 const Index& new_pool_raw_variables_number)
{
    pool_rows_number = new_pool_rows_number;

    pool_raw_variables_number = new_pool_raw_variables_number;
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

        throw runtime_error(buffer.str());
    }
}


/// Sets the layer type to Layer::Pooling.

void PoolingLayer::set_default()
{
    layer_type = Layer::Type::Pooling;
}


void PoolingLayer::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                     LayerForwardPropagation* layer_forward_propagation,
                                     const bool& is_training)
{
    const TensorMap<Tensor<type, 4>> inputs(inputs_pair(0).first,
                                            inputs_pair(0).second[0],
                                            inputs_pair(0).second[1],
                                            inputs_pair(0).second[2],
                                            inputs_pair(0).second[3]);

    switch(pooling_method)
    {
        case PoolingMethod::MaxPooling:
            forward_propagate_max_pooling(inputs,
                                          layer_forward_propagation,
                                          is_training);
            break;

        case PoolingMethod::AveragePooling:
            forward_propagate_average_pooling(inputs,
                                              layer_forward_propagation,
                                              is_training);
            break;

        case PoolingMethod::NoPooling:
            forward_propagate_no_pooling(inputs,
                                         layer_forward_propagation,
                                         is_training);
            break;
    }
}


/// Returns the result of applying average pooling to a batch of images.
/// @param inputs The batch of images.

void PoolingLayer::forward_propagate_average_pooling(const Tensor<type, 4>& inputs,
                       LayerForwardPropagation* layer_forward_propagation,
                       const bool& is_training) const
{
    const type kernel_size = type(pool_rows_number * pool_raw_variables_number);

    PoolingLayerForwardPropagation* pooling_layer_forward_propagation
            = static_cast<PoolingLayerForwardPropagation*>(layer_forward_propagation);

    Tensor<type, 4>& outputs = pooling_layer_forward_propagation->outputs;

    /// @todo do not create tensor

    Tensor<type, 4> kernel(1, pool_rows_number, pool_raw_variables_number, 1);

    kernel.setConstant(type(1.0/kernel_size));

    outputs = inputs.convolve(kernel, convolution_dimensions);

}


/// Returns the result of applying no pooling to a batch of images.
/// @param inputs The batch of images.

void PoolingLayer::forward_propagate_no_pooling(const Tensor<type, 4>& inputs,
                                                LayerForwardPropagation* layer_forward_propagation,
                                                const bool& is_training)
{
    PoolingLayerForwardPropagation* pooling_layer_forward_propagation
            = static_cast<PoolingLayerForwardPropagation*>(layer_forward_propagation);

    Tensor<type, 4>& outputs = pooling_layer_forward_propagation->outputs;

    outputs = inputs;
}


/// Returns the result of applying max pooling to a batch of images.

void PoolingLayer::forward_propagate_max_pooling(const Tensor<type, 4>& inputs,
                                                 LayerForwardPropagation* layer_forward_propagation,
                                                 const bool& is_training) const
{
    const Index oututs_raw_variables_number = get_outputs_raw_variables_number();
    const Index oututs_rows_number = get_outputs_rows_number();
    const Index outputs_channels_number = get_channels_number();

    PoolingLayerForwardPropagation* pooling_layer_forward_propagation
            = static_cast<PoolingLayerForwardPropagation*>(layer_forward_propagation);

    const Index batch_samples_number = pooling_layer_forward_propagation->batch_samples_number;

    const Index in_rows_stride = 1;
    const Index in_raw_variables_stride = 1;

    Tensor<type, 5>& image_patches = pooling_layer_forward_propagation->image_patches;
    Tensor<type, 4>& outputs = pooling_layer_forward_propagation->outputs;

    const Eigen::array<ptrdiff_t, 4> outputs_dimensions_array({batch_samples_number,
                                                               oututs_rows_number,
                                                               oututs_raw_variables_number,
                                                               outputs_channels_number});

    image_patches.device(*thread_pool_device)
            = inputs.extract_image_patches(pool_rows_number,
                                           pool_raw_variables_number,
                                           row_stride,
                                           column_stride,
                                           in_rows_stride,
                                           in_raw_variables_stride,
                                           PADDING_VALID,
                                           type(padding_width));

    outputs.device(*thread_pool_device)
            = image_patches.maximum(max_pooling_dimensions).reshape(outputs_dimensions_array);

}


void PoolingLayer::calculate_hidden_delta(LayerForwardPropagation* next_forward_propagation,
                                          LayerBackPropagation* next_back_propagation,
                                          LayerBackPropagation* this_layeer_back_propagation) const
{
    /// @todo put switch

    if(pooling_method == PoolingMethod::NoPooling)
    {
        //return next_layer_delta;
    }
    else
    {
        const Type layer_type = next_forward_propagation->layer->get_type();

        if(layer_type == Type::Convolutional)
        {
            // ConvolutionalLayer* convolutional_layer = dynamic_cast<ConvolutionalLayer*>(next_layer);

            // return calculate_hidden_delta_convolutional(convolutional_layer, activations, activations_derivatives, next_layer_delta);
        }
        else if(layer_type == Type::Pooling)
        {
            // PoolingLayer* pooling_layer = dynamic_cast<PoolingLayer*>(next_layer);

            // return calculate_hidden_delta_pooling(pooling_layer, activations, activations_derivatives, next_layer_delta);
        }
    }
}


void PoolingLayer::calculate_hidden_delta(ConvolutionalLayerForwardPropagation* next_forward_propagation,
                                          ConvolutionalLayerBackPropagation* next_back_propagation,
                                          ConvolutionalLayerBackPropagation* this_back_propagation) const
{
    /*
    // Current layer's values

//    const Index batch_samples_number = this_back_propagation->batch_samples_number;
    const Index channels_number = get_channels_number();
    const Index output_rows_number = get_outputs_rows_number();
    const Index output_raw_variables_number = get_outputs_raw_variables_number();

    // Next layer's values

    const ConvolutionalLayer* convolutional_layer
            = static_cast<ConvolutionalLayer*>(next_forward_propagation->layer);

    const Index next_layers_filters_number = convolutional_layer->get_kernels_number();
    const Index next_layers_filter_rows = convolutional_layer->get_kernels_rows_number();
    const Index next_layers_filter_raw_variables = convolutional_layer->get_kernels_raw_variables_number();
    const Index next_layers_output_rows = convolutional_layer->get_outputs_rows_number();
    const Index next_layers_output_raw_variables = convolutional_layer->get_outputs_raw_variables_number();
    const Index next_layers_row_stride = convolutional_layer->get_row_stride();
    const Index next_layers_raw_variable_stride = convolutional_layer->get_raw_variable_stride();

    const Tensor<type, 4> next_layers_weights = next_layer->get_synaptic_weights();

    // Hidden delta calculation

        Tensor<type, 4> hidden_delta(images_number, channels_number, output_rows_number, output_raw_variables_number);

        const Index size = hidden_delta.size();

        #pragma omp parallel for

        for(Index tensor_index = 0; tensor_index < size; tensor_index++)
        {
            const Index image_index = tensor_index/(channels_number*output_rows_number*output_raw_variables_number);
            const Index channel_index = (tensor_index/(output_rows_number*output_raw_variables_number))%channels_number;
            const Index row_index = (tensor_index/output_raw_variables_number)%output_rows_number;
            const Index raw_variable_index = tensor_index%output_raw_variables_number;

            long double sum = 0.0;

            const Index lower_row_index = (row_index - next_layers_filter_rows)/next_layers_row_stride + 1;
            const Index upper_row_index = min(row_index/next_layers_row_stride + 1, next_layers_output_rows);
            const Index lower_raw_variable_index = (raw_variable_index - next_layers_filter_raw_variables)/next_layers_raw_variable_stride + 1;
            const Index upper_raw_variable_index = min(raw_variable_index/next_layers_raw_variable_stride + 1, next_layers_output_raw_variables);

            for(Index i = 0; i < next_layers_filters_number; i++)
            {
                for(Index j = lower_row_index; j < upper_row_index; j++)
                {
                    for(Index k = lower_raw_variable_index; k < upper_raw_variable_index; k++)
                    {
                        const type delta_element = next_layer_delta(image_index, i, j, k);

                        const type weight = next_layers_weights(i, channel_index, row_index - j*next_layers_row_stride, raw_variable_index - k*next_layers_raw_variable_stride);

                        sum += delta_element*weight;
                    }
                }
            }

            hidden_delta(image_index, channel_index, row_index, raw_variable_index) = sum;
        }
*/
}


void PoolingLayer::calculate_hidden_delta(PoolingLayerForwardPropagation* next_forward_propagation,
                                          PoolingLayerBackPropagation* next_back_propagation,
                                          ConvolutionalLayerBackPropagation* this_back_propagation) const
{
/*
        switch(next_layer->get_pooling_method())
        {
            case opennn::PoolingLayer::PoolingMethod::NoPooling:
            {
                return next_layer_delta;
            }

            case opennn::PoolingLayer::PoolingMethod::AveragePooling:
            {
                // Current layer's values

                const Index batch_samples_number = next_layer_delta.dimension(0);
                const Index channels_number = get_channels_number();
                const Index output_rows_number = get_outputs_rows_number();
                const Index output_raw_variables_number = get_outputs_raw_variables_number();

                // Next layer's values

                const Index next_layers_pool_rows = next_layer->get_pool_rows_number();
                const Index next_layers_pool_raw_variables = next_layer->get_pool_raw_variables_number();
                const Index next_layers_output_rows = next_layer->get_outputs_rows_number();
                const Index next_layers_output_raw_variables = next_layer->get_outputs_raw_variables_number();
                const Index next_layers_row_stride = next_layer->get_row_stride();
                const Index next_layers_raw_variable_stride = next_layer->get_raw_variable_stride();

                // Hidden delta calculation

                Tensor<type, 4> hidden_delta(batch_samples_number, channels_number, output_rows_number, output_raw_variables_number);

                const Index size = hidden_delta.size();

                #pragma omp parallel for

                for(Index tensor_index = 0; tensor_index < size; tensor_index++)
                {
                    const Index image_index = tensor_index/(channels_number*output_rows_number*output_raw_variables_number);
                    const Index channel_index = (tensor_index/(output_rows_number*output_raw_variables_number))%channels_number;
                    const Index row_index = (tensor_index/output_raw_variables_number)%output_rows_number;
                    const Index raw_variable_index = tensor_index%output_raw_variables_number;

                    long double sum = 0.0;

                    const Index lower_row_index = (row_index - next_layers_pool_rows)/next_layers_row_stride + 1;
                    const Index upper_row_index = min(row_index/next_layers_row_stride + 1, next_layers_output_rows);
                    const Index lower_raw_variable_index = (raw_variable_index - next_layers_pool_raw_variables)/next_layers_raw_variable_stride + 1;
                    const Index upper_raw_variable_index = min(raw_variable_index/next_layers_raw_variable_stride + 1, next_layers_output_raw_variables);

                    for(Index i = lower_row_index; i < upper_row_index; i++)
                    {
                        for(Index j = lower_raw_variable_index; j < upper_raw_variable_index; j++)
                        {
                            const type delta_element = next_layer_delta(image_index, channel_index, i, j);

                            sum += delta_element;
                        }
                    }

                    hidden_delta(image_index, channel_index, row_index, raw_variable_index) = sum;
                }

//                return hidden_delta/(next_layers_pool_rows*next_layers_pool_raw_variables);
            }

            case opennn::PoolingLayer::PoolingMethod::MaxPooling:
            {
                // Current layer's values

                const Index images_number = next_layer_delta.dimension(0);
                const Index channels_number = get_channels_number();
                const Index output_rows_number = get_outputs_rows_number();
                const Index output_raw_variables_number = get_outputs_raw_variables_number();

                // Next layer's values

                const Index next_layers_pool_rows = next_layer->get_pool_rows_number();
                const Index next_layers_pool_raw_variables = next_layer->get_pool_raw_variables_number();
                const Index next_layers_output_rows = next_layer->get_outputs_rows_number();
                const Index next_layers_output_raw_variables = next_layer->get_outputs_raw_variables_number();
                const Index next_layers_row_stride = next_layer->get_row_stride();
                const Index next_layers_raw_variable_stride = next_layer->get_raw_variable_stride();

                // Hidden delta calculation

                Tensor<type, 4> hidden_delta(images_number, channels_number, output_rows_number, output_raw_variables_number);

                const Index size = hidden_delta.size();

                #pragma omp parallel for

                for(Index tensor_index = 0; tensor_index < size; tensor_index++)
                {
                    const Index image_index = tensor_index/(channels_number*output_rows_number*output_raw_variables_number);
                    const Index channel_index = (tensor_index/(output_rows_number*output_raw_variables_number))%channels_number;
                    const Index row_index = (tensor_index/output_raw_variables_number)%output_rows_number;
                    const Index raw_variable_index = tensor_index%output_raw_variables_number;

                    long double sum = 0.0;

                    const Index lower_row_index = (row_index - next_layers_pool_rows)/next_layers_row_stride + 1;
                    const Index upper_row_index = min(row_index/next_layers_row_stride + 1, next_layers_output_rows);
                    const Index lower_raw_variable_index = (raw_variable_index - next_layers_pool_raw_variables)/next_layers_raw_variable_stride + 1;
                    const Index upper_raw_variable_index = min(raw_variable_index/next_layers_raw_variable_stride + 1, next_layers_output_raw_variables);

                    for(Index i = lower_row_index; i < upper_row_index; i++)
                    {
                        for(Index j = lower_raw_variable_index; j < upper_raw_variable_index; j++)
                        {
                            Tensor<type, 2> activations_current_submatrix(next_layers_pool_rows, next_layers_pool_raw_variables);

                            for(Index submatrix_row_index = 0; submatrix_row_index < next_layers_pool_rows; submatrix_row_index++)
                            {
                                for(Index submatrix_raw_variable_index = 0; submatrix_raw_variable_index < next_layers_pool_raw_variables; submatrix_raw_variable_index++)
                                {
                                    activations_current_submatrix(submatrix_row_index, submatrix_raw_variable_index) =
                                            activations(image_index, channel_index, i*next_layers_row_stride + submatrix_row_index, j*next_layers_raw_variable_stride + submatrix_raw_variable_index);
                                }
                            }

                            Tensor<type, 2> multiply_not_multiply(next_layers_pool_rows, next_layers_pool_raw_variables);

                            type max_value = activations_current_submatrix(0,0);

                            for(Index submatrix_row_index = 0; submatrix_row_index < next_layers_pool_rows; submatrix_row_index++)
                            {
                                for(Index submatrix_raw_variable_index = 0; submatrix_raw_variable_index < next_layers_pool_raw_variables; submatrix_raw_variable_index++)
                                {
                                    if(activations_current_submatrix(submatrix_row_index, submatrix_raw_variable_index) > max_value)
                                    {
                                        max_value = activations_current_submatrix(submatrix_row_index, submatrix_raw_variable_index);

                                        //multiply_not_multiply.resize(next_layers_pool_rows, next_layers_pool_raw_variables, 0.0);
                                        //multiply_not_multiply(submatrix_row_index, submatrix_raw_variable_index) = type(1);
                                    }
                                }
                            }

                            const type delta_element = next_layer_delta(image_index, channel_index, i, j);

                            const type max_derivative = multiply_not_multiply(row_index - i*next_layers_row_stride, raw_variable_index - j*next_layers_raw_variable_stride);

                            sum += delta_element*max_derivative;
                        }
                    }

                    hidden_delta(image_index, channel_index, row_index, raw_variable_index) = sum;
                }

                return hidden_delta;
                break;
            }
        }
*/
}


/// Serializes the convolutional layer object into an XML document of the TinyXML.
/// See the OpenNN manual for more information about the format of this document.

void PoolingLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Pooling layer

    file_stream.OpenElement("PoolingLayer");

    // Layer name

    file_stream.OpenElement("LayerName");

    buffer.str("");
    buffer << layer_name;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Image size

    file_stream.OpenElement("InputsVariablesDimensions");

    buffer.str("");
    buffer << "1 1 1";

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    //Filters number

    file_stream.OpenElement("FiltersNumber");

    buffer.str("");
    buffer << 9;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    // Filters size

    file_stream.OpenElement("FiltersSize");

    buffer.str("");
    buffer << 9;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Activation function

    file_stream.OpenElement("ActivationFunction");

    file_stream.PushText(write_pooling_method().c_str());

    file_stream.CloseElement();


    //_______________________________________________________

    // Pooling method

    file_stream.OpenElement("PoolingMethod");

    file_stream.PushText(write_pooling_method().c_str());

    file_stream.CloseElement();

    // Inputs variables dimensions

    file_stream.OpenElement("InputDimensions");

    buffer.str("");
    buffer << get_inputs_dimensions();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Column stride

    file_stream.OpenElement("raw_variablestride");

    buffer.str("");
    buffer << get_raw_variable_stride();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    //Row stride

    file_stream.OpenElement("RowStride");

    buffer.str("");
    buffer << get_row_stride();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Pool raw_variables number

    file_stream.OpenElement("Poolraw_variablesNumber");

    buffer.str("");
    buffer << get_pool_raw_variables_number();

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

        throw runtime_error(buffer.str());
    }

    // Pooling method element

    const tinyxml2::XMLElement* pooling_method_element = pooling_layer_element->FirstChildElement("PoolingMethod");

    if(!pooling_method_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling method element is nullptr.\n";

        throw runtime_error(buffer.str());
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

        throw runtime_error(buffer.str());
    }

    const string input_variables_dimensions_string = input_variables_dimensions_element->GetText();

//    set_input_variables_dimenisons(input_variables_dimensions_string);

    // Column stride

    const tinyxml2::XMLElement* column_stride_element = pooling_layer_element->FirstChildElement("raw_variablestride");

    if(!column_stride_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling column stride element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string column_stride_string = column_stride_element->GetText();

    set_raw_variable_stride(Index(stoi(column_stride_string)));

    // Row stride

    const tinyxml2::XMLElement* row_stride_element = pooling_layer_element->FirstChildElement("RowStride");

    if(!row_stride_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling row stride element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string row_stride_string = row_stride_element->GetText();

    set_row_stride(Index(stoi(row_stride_string)));

    // Pool raw_variables number

    const tinyxml2::XMLElement* pool_raw_variables_number_element = pooling_layer_element->FirstChildElement("Poolraw_variablesNumber");

    if(!pool_raw_variables_number_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling raw_variables number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string pool_raw_variables_number_string = pool_raw_variables_number_element->GetText();

    // Pool rows number

    const tinyxml2::XMLElement* pool_rows_number_element = pooling_layer_element->FirstChildElement("PoolRowsNumber");

    if(!pool_rows_number_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling rows number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string pool_rows_number_string = pool_rows_number_element->GetText();

    set_pool_size(Index(stoi(pool_rows_number_string)), Index(stoi(pool_raw_variables_number_string)));

    // Padding Width

    const tinyxml2::XMLElement* padding_width_element = pooling_layer_element->FirstChildElement("PaddingWidth");

    if(!padding_width_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Padding width element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(padding_width_element->GetText())
    {
        const string padding_width_string = padding_width_element->GetText();

        set_padding_width(Index(stoi(padding_width_string)));
    }
}

pair<type*, dimensions> PoolingLayerForwardPropagation::get_outputs_pair() const
{
    const Index neurons_number = layer->get_neurons_number();

    return pair<type*, dimensions>(outputs_data, { { batch_samples_number, neurons_number, 1, 1 } });
}

void PoolingLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_samples_number = new_batch_samples_number;

    layer = new_layer;

    const PoolingLayer* pooling_layer = static_cast<PoolingLayer*>(layer);

    const Index pool_rows_number = pooling_layer->get_pool_rows_number();

    const Index pool_raw_variables_number = pooling_layer->get_pool_raw_variables_number();

    const Index outputs_rows_number = pooling_layer->get_outputs_rows_number();

    const Index outputs_raw_variables_number = pooling_layer->get_outputs_raw_variables_number();

    const Index channels_number = pooling_layer->get_channels_number();

    outputs.resize(batch_samples_number,
        outputs_rows_number,
        outputs_raw_variables_number,
        channels_number);

    outputs_data = outputs.data();

    image_patches.resize(batch_samples_number,
        pool_rows_number,
        pool_raw_variables_number,
        outputs_rows_number * outputs_raw_variables_number,
        channels_number);
}

pair<type*, dimensions> PoolingLayerBackPropagation::get_deltas_pair() const
{
    const PoolingLayer* pooling_layer = static_cast<PoolingLayer*>(layer);

    const Index kernels_number = 0;// = pooling_layer->get_kernels_number();

    const Index outputs_raw_variables_number = pooling_layer->get_outputs_raw_variables_number();

    const Index outputs_rows_number = pooling_layer->get_outputs_rows_number();

    return pair<type*, dimensions>(deltas_data, { { batch_samples_number,
        kernels_number,
        outputs_rows_number,
        outputs_raw_variables_number } });
}

void PoolingLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_samples_number = new_batch_samples_number;

    layer = new_layer;

    const PoolingLayer* pooling_layer = static_cast<PoolingLayer*>(layer);

    const Index outputs_rows_number = pooling_layer->get_outputs_rows_number();
    const Index outputs_raw_variables_number = pooling_layer->get_outputs_raw_variables_number();

    const Index kernels_number = 0;

    deltas.resize(batch_samples_number,
        kernels_number,
        outputs_rows_number,
        outputs_raw_variables_number);

    deltas_data = deltas.data();
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
