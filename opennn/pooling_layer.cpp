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
/// @param new_input_dimensions A vector containing the desired number of rows and columns for the input.
/// @param pool_dimensions A vector containing the desired number of rows and columns for the pool.


PoolingLayer::PoolingLayer(const dimensions& new_input_dimensions, const dimensions& new_pool_dimensions) : Layer()
{
    set(new_input_dimensions, new_pool_dimensions);

    input_dimensions = new_input_dimensions;

    pool_height = new_pool_dimensions[0];
    pool_width = new_pool_dimensions[1];

    set_default();
}


/// Returns the number of neurons the layer applies to an image.

Index PoolingLayer::get_neurons_number() const
{
    return get_output_height() * get_output_width() * get_channels_number();
}


/// Returns the layer's outputs dimensions.

dimensions PoolingLayer::get_output_dimensions() const
{
    const Index rows_number = get_output_height();
    const Index columns_number = get_output_width();
    const Index channels_number = input_dimensions[2];

    return { rows_number, columns_number, channels_number };
}


/// Returns the number of inputs of the layer.

Index PoolingLayer::get_inputs_number() const
{
    return input_dimensions.size();
}



/// Returns the number of rows of the layer's input.

Index PoolingLayer::get_input_height() const
{
    return input_dimensions[0];
}


/// Returns the number of columns of the layer's input.

Index PoolingLayer::get_input_width() const
{
    return input_dimensions[1];
}


/// Returns the number of channels of the layers' input.

Index PoolingLayer::get_channels_number() const
{
    return input_dimensions[2];
}


/// Returns the number of rows of the layer's output.

Index PoolingLayer::get_output_height() const
{
    type padding = type(0);

    const Index input_height = get_input_height();

    return (input_height - pool_height + 2*padding)/row_stride + 1;
}


/// Returns the number of columns of the layer's output.

Index PoolingLayer::get_output_width() const
{
    type padding = type(0);

    const Index input_width = get_input_width();

    return (input_width - pool_width + 2*padding)/column_stride + 1;
}


/// Returns the padding heigth.

Index PoolingLayer::get_padding_heigth() const
{
    return padding_heigth;
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


/// Returns the pooling filter's raw_variable stride.

Index PoolingLayer::get_column_stride() const
{
    return column_stride;
}


/// Returns the number of rows of the pooling filter.

Index PoolingLayer::get_pool_height() const
{
    return pool_height;
}


/// Returns the number of columns of the pooling filter.

Index PoolingLayer::get_pool_width() const
{
    return pool_width;
}


/// Returns the pooling method.

PoolingLayer::PoolingMethod PoolingLayer::get_pooling_method() const
{
    return pooling_method;
}


/// Returns the input_dimensions.

dimensions PoolingLayer::get_inputs_dimensions() const
{
    return input_dimensions;
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


void PoolingLayer::set(const dimensions& new_input_dimensions, const dimensions& new_pool_dimensions)
{
    if(new_input_dimensions.size() != 3)
        throw runtime_error("Input dimensions must be 3");

    if(new_pool_dimensions.size() != 2)
        throw runtime_error("Pool dimensions must be 2");

    input_dimensions = new_input_dimensions;

    pool_height = new_pool_dimensions[0];
    pool_width = new_pool_dimensions[1];

    set_default();
}


void PoolingLayer::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}


/// Sets the number of rows of the layer's input.
/// @param new_input_rows_number The desired rows number.

void PoolingLayer::set_inputs_dimensions(const dimensions& new_input_dimensions)
{
    input_dimensions = new_input_dimensions;
}


/// Sets the padding heigth.
/// @param new_padding_heigth The desired heigth.

void PoolingLayer::set_padding_heigth(const Index& new_padding_heigth)
{
    padding_heigth = new_padding_heigth;
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


/// Sets the pooling filter's raw_variable stride.
/// @param new_raw_variable_stride The desired raw_variable stride.

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
    pool_height = new_pool_rows_number;

    pool_width = new_pool_columns_number;
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
    const type pool_size = type(pool_height * pool_width);

    PoolingLayerForwardPropagation* pooling_layer_forward_propagation
            = static_cast<PoolingLayerForwardPropagation*>(layer_forward_propagation);

    Tensor<type, 4>& outputs = pooling_layer_forward_propagation->outputs;

    /// @todo do not create tensor

    Tensor<type, 4> pool(1, pool_height, pool_width, 1);

    pool.setConstant(type(1.0/pool_size));

    outputs.device(*thread_pool_device) = inputs.convolve(pool, average_pooling_dimensions);
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

    outputs.device(*thread_pool_device) = inputs;
}


/// Returns the result of applying max pooling to a batch of images.

void PoolingLayer::forward_propagate_max_pooling(const Tensor<type, 4>& inputs,
                                                 LayerForwardPropagation* layer_forward_propagation,
                                                 const bool& is_training) const
{

    const Index output_width = get_output_width();
    const Index output_height = get_output_height();
    const Index output_channels = get_channels_number();

    PoolingLayerForwardPropagation* pooling_layer_forward_propagation
            = static_cast<PoolingLayerForwardPropagation*>(layer_forward_propagation);

    const Index batch_samples_number = pooling_layer_forward_propagation->batch_samples_number;

    const Index in_rows_stride = 1;
    const Index in_columns_stride = 1;

    Tensor<type, 5>& image_patches = pooling_layer_forward_propagation->image_patches;
    Tensor<type, 4>& outputs = pooling_layer_forward_propagation->outputs;

    const Eigen::array<ptrdiff_t, 4> outputs_dimensions_array({batch_samples_number,
                                                               output_height,
                                                               output_width,
                                                               output_channels});

    image_patches.device(*thread_pool_device)
            = inputs.extract_image_patches(pool_height,
                                           pool_width,
                                           row_stride,
                                           column_stride,
                                           in_rows_stride,
                                           in_columns_stride,
                                           PADDING_VALID,
                                           type(padding_width));

    outputs.device(*thread_pool_device)
            = image_patches.maximum(max_pooling_dimensions).reshape(outputs_dimensions_array);

    // Extract maximum indices

    pooling_layer_forward_propagation->inputs_max_indices.resize(inputs.dimension(0),
                                                                 inputs.dimension(1),
                                                                 inputs.dimension(2),
                                                                 inputs.dimension(3));

    pooling_layer_forward_propagation->inputs_max_indices.setZero();

    Index outputs_index = 0;

    for(Index i = 0; i < pooling_layer_forward_propagation->inputs_max_indices.size(); i++)
    {

        if(abs(inputs(i) - outputs(outputs_index)) < 1e-3)
        {
            pooling_layer_forward_propagation->inputs_max_indices(i) = 1;
            outputs_index++;
        }
    }

}


void PoolingLayer::back_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                  const Tensor<pair<type*, dimensions>, 1>& deltas_pair,
                                  LayerForwardPropagation* forward_propagation,
                                  LayerBackPropagation* back_propagation) const
{

    const TensorMap<Tensor<type, 2>> deltas(deltas_pair(0).first,
                                            deltas_pair(0).second[0],
                                            deltas_pair(0).second[1]);

    // Forward propagation

//    const PoolingLayerForwardPropagation* pooling_layer_forward_propagation =
//        static_cast<PoolingLayerForwardPropagation*>(forward_propagation);

    // Back propagation

    PoolingLayerBackPropagation* pooling_layer_back_propagation =
        static_cast<PoolingLayerBackPropagation*>(back_propagation);

    Tensor<type, 4>& input_derivatives = pooling_layer_back_propagation->input_derivatives;

    /// @todo calculate input derivatives (= deltas for previous layer)

    //input_derivatives.device(*thread_pool_device) = 0;

    /* Do this at the end of back_propagate(), with input_derivatives instead of deltas
void PoolingLayer::calculate_hidden_delta(PoolingLayerForwardPropagation* next_pooling_layer_forward_propagation,
                                          PoolingLayerBackPropagation* next_pooling_layer_back_propagation,
                                          PoolingLayerForwardPropagation* this_layer_forward_propagation,
                                          PoolingLayerBackPropagation* this_pooling_layer_back_propagation) const
{

//    const Index inputs_size = next_pooling_layer_forward_propagation->inputs_max_indices.size();

//    Index delta_index = 0;

//    this_convolutional_layer_back_propagation->deltas.setZero();

//    for(Index i = 0; i < inputs_size; i++)
//    {
//        if(next_pooling_layer_forward_propagation->inputs_max_indices(delta_index) == 1)
//        {
//            this_convolutional_layer_back_propagation->deltas(i) = next_pooling_layer_back_propagation->deltas(i);
//            delta_index++;
//        }
//    }

    return;
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

    // Pooling method

    file_stream.OpenElement("PoolingMethod");

    file_stream.PushText(write_pooling_method().c_str());

    file_stream.CloseElement();

    // Inputs variables dimensions

    file_stream.OpenElement("InputDimensions");

    buffer.str("");
    buffer << get_inputs_dimensions()[0] << get_inputs_dimensions()[1] << get_inputs_dimensions()[2];

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // raw_variable stride

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
    buffer << get_pool_width();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Pool rows number

    file_stream.OpenElement("PoolRowsNumber");

    buffer.str("");
    buffer << get_pool_height();

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

    // raw_variable stride

    const tinyxml2::XMLElement* column_stride_element = pooling_layer_element->FirstChildElement("ColumnStride");

    if(!column_stride_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling raw_variable stride element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string column_stride_string = column_stride_element->GetText();

    set_column_stride(Index(stoi(column_stride_string)));

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

    const tinyxml2::XMLElement* pool_columns_number_element = pooling_layer_element->FirstChildElement("PoolColumnsNumber");

    if(!pool_columns_number_element)
    {
        buffer << "OpenNN Exception: PoolingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pooling columns number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    const string pool_columns_number_string = pool_columns_number_element->GetText();

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

    set_pool_size(Index(stoi(pool_rows_number_string)), Index(stoi(pool_columns_number_string)));

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


PoolingLayerForwardPropagation::PoolingLayerForwardPropagation()
    : LayerForwardPropagation()
{
}


PoolingLayerForwardPropagation::PoolingLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


pair<type*, dimensions> PoolingLayerForwardPropagation::get_outputs_pair() const
{
    const PoolingLayer* pooling_layer = static_cast<PoolingLayer*>(layer);

    const Index output_height = pooling_layer->get_output_height();
    const Index output_width = pooling_layer->get_output_width();
    const Index channels_number = pooling_layer->get_channels_number();

    const dimensions output_dimensions = { batch_samples_number, output_height, output_width, channels_number};

    return pair<type*, dimensions>(outputs_data, output_dimensions);
}


void PoolingLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_samples_number = new_batch_samples_number;

    layer = new_layer;

    const PoolingLayer* pooling_layer = static_cast<PoolingLayer*>(layer);

    const Index pool_height = pooling_layer->get_pool_height();

    const Index pool_width = pooling_layer->get_pool_width();

    const Index output_height = pooling_layer->get_output_height();

    const Index output_width = pooling_layer->get_output_width();

    const Index channels_number = pooling_layer->get_channels_number();

    outputs.resize(batch_samples_number,
                   output_height,
                   output_width,
                   channels_number);

    outputs_data = outputs.data();

    image_patches.resize(batch_samples_number,
                         pool_height,
                         pool_width,
                         output_height * output_width,
                         channels_number);
}


void PoolingLayerForwardPropagation::print() const
{
    cout << "Pooling layer forward propagation" << endl;

    cout << "Outputs:" << endl;

    cout << outputs(0) << endl;

    cout << "Image patches" << endl;
    cout << image_patches << endl;
}


PoolingLayerBackPropagation::PoolingLayerBackPropagation() : LayerBackPropagation()
{
}


PoolingLayerBackPropagation::PoolingLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer);
}


PoolingLayerBackPropagation::~PoolingLayerBackPropagation()
{
}


void PoolingLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_samples_number = new_batch_samples_number;

    layer = new_layer;

    const PoolingLayer* pooling_layer = static_cast<PoolingLayer*>(layer);

    const dimensions& input_dimensions = pooling_layer->get_inputs_dimensions();

    input_derivatives.resize(batch_samples_number, input_dimensions[0], input_dimensions[1], input_dimensions[2]);

    inputs_derivatives.resize(1);
    inputs_derivatives(0).first = input_derivatives.data();
    inputs_derivatives(0).second = { batch_samples_number, input_dimensions[0], input_dimensions[1], input_dimensions[2] };
}


void PoolingLayerBackPropagation::print() const
{
    cout << "Pooling layer back propagation" << endl;
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
