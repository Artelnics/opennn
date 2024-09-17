//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>

#include "tensors.h"
#include "pooling_layer.h"

namespace opennn
{

PoolingLayer::PoolingLayer() : Layer()
{
    set_default();
}


PoolingLayer::PoolingLayer(const dimensions& new_input_dimensions, const dimensions& new_pool_dimensions) : Layer()
{
    set(new_input_dimensions, new_pool_dimensions);

    input_dimensions = new_input_dimensions;

    pool_height = new_pool_dimensions[0];
    pool_width = new_pool_dimensions[1];

    set_default();
}


Index PoolingLayer::get_neurons_number() const
{
    return get_output_height() * get_output_width() * get_channels_number();
}


dimensions PoolingLayer::get_output_dimensions() const
{
    const Index rows_number = get_output_height();
    const Index columns_number = get_output_width();
    const Index channels = input_dimensions[2];

    return { rows_number, columns_number, channels };
}


Index PoolingLayer::get_inputs_number() const
{
    return input_dimensions.size();
}


Index PoolingLayer::get_input_height() const
{
    return input_dimensions[0];
}


Index PoolingLayer::get_input_width() const
{
    return input_dimensions[1];
}


Index PoolingLayer::get_channels_number() const
{
    return input_dimensions[2];
}


Index PoolingLayer::get_output_height() const
{
    type padding = type(0);

    const Index input_height = get_input_height();

    return (input_height - pool_height + 2*padding)/row_stride + 1;
}


Index PoolingLayer::get_output_width() const
{
    type padding = type(0);

    const Index input_width = get_input_width();

    return (input_width - pool_width + 2*padding)/column_stride + 1;
}


Index PoolingLayer::get_padding_heigth() const
{
    return padding_heigth;
}


Index PoolingLayer::get_padding_width() const
{
    return padding_width;
}


Index PoolingLayer::get_row_stride() const
{
    return row_stride;
}


Index PoolingLayer::get_column_stride() const
{
    return column_stride;
}


Index PoolingLayer::get_pool_height() const
{
    return pool_height;
}


Index PoolingLayer::get_pool_width() const
{
    return pool_width;
}


PoolingLayer::PoolingMethod PoolingLayer::get_pooling_method() const
{
    return pooling_method;
}


dimensions PoolingLayer::get_input_dimensions() const
{
    return input_dimensions;
}


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

    if (new_pool_dimensions[0] > new_input_dimensions[0] || new_pool_dimensions[1] > new_input_dimensions[1])
        throw runtime_error("Pool dimensions cannot be bigger than input dimensions");

    input_dimensions = new_input_dimensions;

    pool_height = new_pool_dimensions[0];
    pool_width = new_pool_dimensions[1];

    set_default();
}


void PoolingLayer::set_name(const string& new_layer_name)
{
    name = new_layer_name;
}


void PoolingLayer::set_inputs_dimensions(const dimensions& new_input_dimensions)
{
    input_dimensions = new_input_dimensions;
}


void PoolingLayer::set_padding_heigth(const Index& new_padding_heigth)
{
    padding_heigth = new_padding_heigth;
}


void PoolingLayer::set_padding_width(const Index& new_padding_width)
{
    padding_width = new_padding_width;
}


void PoolingLayer::set_row_stride(const Index& new_row_stride)
{
    row_stride = new_row_stride;
}


void PoolingLayer::set_column_stride(const Index& new_column_stride)
{
    column_stride = new_column_stride;
}


void PoolingLayer::set_pool_size(const Index& new_pool_rows_number,
                                 const Index& new_pool_columns_number)
{
    pool_height = new_pool_rows_number;

    pool_width = new_pool_columns_number;
}


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
        throw runtime_error("Unknown pooling type: " + new_pooling_method + ".\batch_index");
    }
}


void PoolingLayer::set_default()
{
    layer_type = Layer::Type::Pooling;

    name = "pooling_layer";
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


void PoolingLayer::forward_propagate_average_pooling(const Tensor<type, 4>& inputs,
                                                     LayerForwardPropagation* layer_forward_propagation,
                                                     const bool& is_training) const
{

    PoolingLayerForwardPropagation* pooling_layer_forward_propagation
            = static_cast<PoolingLayerForwardPropagation*>(layer_forward_propagation);

    const Tensor<type, 4>& pool = pooling_layer_forward_propagation->pool;

    Tensor<type, 4>& outputs = pooling_layer_forward_propagation->outputs;

    outputs.device(*thread_pool_device) = inputs.convolve(pool, average_pooling_dimensions);
}


void PoolingLayer::forward_propagate_no_pooling(const Tensor<type, 4>& inputs,
                                                LayerForwardPropagation* layer_forward_propagation,
                                                const bool& is_training)
{
    PoolingLayerForwardPropagation* pooling_layer_forward_propagation
            = static_cast<PoolingLayerForwardPropagation*>(layer_forward_propagation);

    Tensor<type, 4>& outputs = pooling_layer_forward_propagation->outputs;

    outputs.device(*thread_pool_device) = inputs;
}


void PoolingLayer::forward_propagate_max_pooling(const Tensor<type, 4>& inputs,
                                                 LayerForwardPropagation* layer_forward_propagation,
                                                 const bool& is_training) const
{

    const Index input_height = inputs.dimension(1);
    const Index input_width = inputs.dimension(2);

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
    if (is_training)
    {


    }
}


void PoolingLayer::back_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                  const Tensor<pair<type*, dimensions>, 1>& deltas_pair,
                                  LayerForwardPropagation* forward_propagation,
                                  LayerBackPropagation* back_propagation) const
{

    // Inputs

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

    // Forward propagation

    const PoolingLayerForwardPropagation* pooling_layer_forward_propagation =
          static_cast<PoolingLayerForwardPropagation*>(forward_propagation);

    const Tensor<type, 4> outputs = pooling_layer_forward_propagation->outputs;
    
    switch(pooling_method)
    {
    case PoolingMethod::MaxPooling:
        back_propagate_max_pooling(inputs,
                                   deltas,
                                   back_propagation);
        break;

    case PoolingMethod::AveragePooling:
        back_propagate_average_pooling(inputs,
                                       deltas,
                                       back_propagation);
        break;

    case PoolingMethod::NoPooling:

        PoolingLayerBackPropagation* pooling_layer_back_propagation =
            static_cast<PoolingLayerBackPropagation*>(back_propagation);

        Tensor<type, 4>& input_derivatives = pooling_layer_back_propagation->input_derivatives;

        input_derivatives = deltas;
        break;
    }
}


void PoolingLayer::back_propagate_max_pooling(const Tensor<type, 4>& inputs,
                                              const Tensor<type, 4>& deltas,
                                              LayerBackPropagation* back_propagation) const
{
    const Index batch_samples_number = inputs.dimension(0);

    const Index input_height = inputs.dimension(1);
    const Index input_width = inputs.dimension(2);
    const Index input_channels = inputs.dimension(3);

    const Index output_height = deltas.dimension(1);
    const Index output_width = deltas.dimension(2);

    // Back propagation

    PoolingLayerBackPropagation* pooling_layer_back_propagation =
        static_cast<PoolingLayerBackPropagation*>(back_propagation);

    Tensor<type, 4>& input_derivatives = pooling_layer_back_propagation->input_derivatives;

    input_derivatives.setZero();

    for (Index batch_index = 0; batch_index < batch_samples_number; ++batch_index)
    {
        for (Index channel_index = 0; channel_index < input_channels; ++channel_index)
        {
            for (Index output_height_index = 0; output_height_index < output_height; ++output_height_index)
            {
                const Index height_start = output_height_index * row_stride;
                const Index height_end = min(height_start + pool_height, input_height);

                for (Index output_width_index = 0; output_width_index < output_width; ++output_width_index)
                {
                    
                    const Index width_start = output_width_index * column_stride;
                    const Index width_end = min(width_start + pool_width, input_width);

                    // @todo calculate input maximal indices in forward propagate 

                    type maximum = -numeric_limits<type>::infinity();
                    Index maximal_height = height_start;
                    Index maximal_width = width_start;

                    for (Index height_index = height_start; height_index < height_end; ++height_index)
                    {
                        for (Index width_index = width_start; width_index < width_end; ++width_index)
                        {
                            if (inputs(batch_index, height_index, width_index, channel_index) > maximum)
                            {
                                maximum = inputs(batch_index, height_index, width_index, channel_index);
                                maximal_height = height_index;
                                maximal_width = width_index;
                            }
                        }
                    }

                    input_derivatives(batch_index, maximal_height, maximal_width, channel_index)
                        += deltas(batch_index, output_height_index, output_width_index, channel_index);
                }
            }
        }
    }
}


void PoolingLayer::back_propagate_average_pooling(const Tensor<type, 4>& inputs,
                                                  const Tensor<type, 4>& deltas,
                                                  LayerBackPropagation* back_propagation) const
{
    const Index batch_samples_number = inputs.dimension(0);

    const Index input_height = inputs.dimension(1);
    const Index input_width = inputs.dimension(2);
    const Index input_channels = inputs.dimension(3);

    const Index output_height = deltas.dimension(1);
    const Index output_width = deltas.dimension(2);

    const Index pool_size = pool_height*pool_width;

    const Eigen::array<Index, 4> extents = { 1, pool_height, pool_width, 1 };
    Tensor<type, 4> gradient_tensor(extents);

    // Back propagation

    PoolingLayerBackPropagation* pooling_layer_back_propagation =
        static_cast<PoolingLayerBackPropagation*>(back_propagation);

    Tensor<type, 4>& input_derivatives = pooling_layer_back_propagation->input_derivatives;

    input_derivatives.setZero();

    //#pragma omp paralell for

    for (Index batch_index = 0; batch_index < batch_samples_number; batch_index++)
    {
        for (Index channel_index = 0; channel_index < input_channels; channel_index++)
        {
            for (Index output_height_index = 0; output_height_index < output_height; output_height_index++)
            {
                const Index height_start = output_height_index * row_stride;
                const Index height_end = min(height_start + pool_height, input_height);

                for (Index output_width_index = 0; output_width_index < output_width; output_width_index++)
                {
                    const Index width_start = output_width_index * column_stride;
                    const Index width_end = min(width_start + pool_width, input_width);

                    const type gradient = deltas(batch_index, output_height_index, output_width_index, channel_index) / type(pool_size);

                    const Eigen::array<Index, 4> offsets = { batch_index, height_start, width_start, channel_index };

                    gradient_tensor.setConstant(gradient);

                    input_derivatives.slice(offsets, extents) += gradient_tensor;
                }
            }
        }
    }
}


void PoolingLayer::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // PoolingLayer layer

    file_stream.OpenElement("PoolingLayer");

    // Layer name

    file_stream.OpenElement("Name");
    file_stream.PushText(name.c_str());
    file_stream.CloseElement();

    // Image size

    file_stream.OpenElement("InputDimensions");
    file_stream.PushText(string("1 1 1").c_str());
    file_stream.CloseElement();

    //Filters number

    file_stream.OpenElement("FiltersNumber");
    file_stream.PushText(string("9").c_str());
    file_stream.CloseElement();

    // Filters size

    file_stream.OpenElement("FiltersSize");
    file_stream.PushText(string("9").c_str());
    file_stream.CloseElement();

    // PoolingLayer method

    file_stream.OpenElement("PoolingMethod");
    file_stream.PushText(write_pooling_method().c_str());
    file_stream.CloseElement();

    // Inputs variables dimensions

    file_stream.OpenElement("InputDimensions");
    file_stream.PushText(dimensions_to_string(get_input_dimensions()).c_str());
    file_stream.CloseElement();

    // raw_variable stride

    file_stream.OpenElement("ColumnStride");
    file_stream.PushText(to_string(get_column_stride()).c_str());
    file_stream.CloseElement();

    //Row stride

    file_stream.OpenElement("RowStride");
    file_stream.PushText(to_string(get_row_stride()).c_str());
    file_stream.CloseElement();

    // Pool columns number

    file_stream.OpenElement("PoolColumnsNumber");
    file_stream.PushText(to_string(get_pool_width()).c_str());
    file_stream.CloseElement();

    // Pool rows number

    file_stream.OpenElement("PoolRowsNumber");
    file_stream.PushText(to_string(get_pool_height()).c_str());
    file_stream.CloseElement();

    // Padding width

    file_stream.OpenElement("PaddingWidth");
    file_stream.PushText(to_string(get_padding_width()).c_str());
    file_stream.CloseElement();

    file_stream.CloseElement();
}


void PoolingLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // PoolingLayer layer

    const tinyxml2::XMLElement* pooling_layer_element = document.FirstChildElement("PoolingLayer");

    if(!pooling_layer_element)
        throw runtime_error("PoolingLayer layer element is nullptr.\batch_index");

    // PoolingLayer method element

    const tinyxml2::XMLElement* pooling_method_element = pooling_layer_element->FirstChildElement("PoolingMethod");

    if(!pooling_method_element)
        throw runtime_error("PoolingLayer method element is nullptr.\batch_index");

    set_pooling_method(pooling_method_element->GetText());

    // Input variables dimensions element

    const tinyxml2::XMLElement* input_dimensions_element = pooling_layer_element->FirstChildElement("InputDimensions");

    if(!input_dimensions_element)
        throw runtime_error("PoolingLayer input variables dimensions element is nullptr.\batch_index");

//    set_input_dimensions(input_dimensions_element->GetText());

    // raw_variable stride

    const tinyxml2::XMLElement* column_stride_element = pooling_layer_element->FirstChildElement("ColumnStride");

    if(!column_stride_element)
        throw runtime_error("PoolingLayer column stride element is nullptr.\batch_index");

    set_column_stride(Index(stoi(column_stride_element->GetText())));

    // Row stride

    const tinyxml2::XMLElement* row_stride_element = pooling_layer_element->FirstChildElement("RowStride");

    if(!row_stride_element)
        throw runtime_error("PoolingLayer row stride element is nullptr.\batch_index");

    set_row_stride(Index(stoi(row_stride_element->GetText())));

    // Pool raw_variables number

    const tinyxml2::XMLElement* pool_columns_number_element = pooling_layer_element->FirstChildElement("PoolColumnsNumber");

    if(!pool_columns_number_element)
        throw runtime_error("PoolingLayer columns number element is nullptr.\batch_index");

    const string pool_columns_number_string = pool_columns_number_element->GetText();

    // Pool rows number

    const tinyxml2::XMLElement* pool_rows_number_element = pooling_layer_element->FirstChildElement("PoolRowsNumber");

    if(!pool_rows_number_element)
        throw runtime_error("PoolingLayer rows number element is nullptr.\batch_index");

    const string pool_rows_number_string = pool_rows_number_element->GetText();

    set_pool_size(Index(stoi(pool_rows_number_string)), Index(stoi(pool_columns_number_string)));

    // Padding Width

    const tinyxml2::XMLElement* padding_width_element = pooling_layer_element->FirstChildElement("PaddingWidth");

    if(!padding_width_element)
        throw runtime_error("Padding width element is nullptr.\batch_index");

    if(padding_width_element->GetText())
        set_padding_width(Index(stoi(padding_width_element->GetText())));
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
    const Index channels = pooling_layer->get_channels_number();

    const dimensions output_dimensions = { batch_samples_number, output_height, output_width, channels};

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

    const Index channels = pooling_layer->get_channels_number();

    const type pool_size = type(pool_height * pool_width);

    const dimensions input_dimensions = pooling_layer->get_input_dimensions();

    const Index input_height = input_dimensions[1];

    const Index input_width = input_dimensions[2];

    outputs.resize(batch_samples_number,
                   output_height,
                   output_width,
                   channels);

    outputs_data = outputs.data();

    image_patches.resize(batch_samples_number,
                         pool_height,
                         pool_width,
                         output_height * output_width,
                         channels);

    pool.resize(1, pool_height, pool_width, 1);

    pool.setConstant(type(1.0 / pool_size));

    inputs_maximal_indices.resize(batch_samples_number,
                                  input_height,
                                  input_width,
                                  channels);
}


void PoolingLayerForwardPropagation::print() const
{
    cout << "PoolingLayer layer forward propagation" << endl;

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

    const dimensions& input_dimensions = pooling_layer->get_input_dimensions();

    input_derivatives.resize(batch_samples_number, input_dimensions[0], input_dimensions[1], input_dimensions[2]);

    inputs_derivatives.resize(1);
    inputs_derivatives(0).first = input_derivatives.data();
    inputs_derivatives(0).second = { batch_samples_number, input_dimensions[0], input_dimensions[1], input_dimensions[2] };
}


void PoolingLayerBackPropagation::print() const
{
    cout << "PoolingLayer layer back propagation" << endl;

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
