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

PoolingLayer::PoolingLayer(const dimensions& new_input_dimensions, 
                           const dimensions& new_pool_dimensions,
                           const dimensions& new_stride_dimensions,
                           const dimensions& new_padding_dimensions,
                           const PoolingMethod& new_pooling_method,
                           const string new_name) : Layer()
{
    layer_type = Layer::Type::Pooling;

    set(new_input_dimensions,
        new_pool_dimensions,
        new_stride_dimensions,
        new_padding_dimensions,
        new_pooling_method,
        new_name);
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
    const type padding = type(0);

    const Index input_height = get_input_height();

    return (input_height - pool_height + 2*padding)/row_stride + 1;
}


Index PoolingLayer::get_output_width() const
{
    const type padding = type(0);

    const Index input_width = get_input_width();

    return (input_width - pool_width + 2*padding)/column_stride + 1;
}


Index PoolingLayer::get_padding_height() const
{
    return padding_height;
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


void PoolingLayer::print() const
{
    cout << "Pooling layer" << endl;
    cout << "Input dimensions: " << endl;
    print_dimensions(input_dimensions);
    cout << "Output dimensions: " << endl;
    print_dimensions(get_output_dimensions());
}


string PoolingLayer::write_pooling_method() const
{
    switch(pooling_method)
    {
    case PoolingMethod::MaxPooling:
        return "MaxPooling";

    case PoolingMethod::AveragePooling:
        return "AveragePooling";
    }

    return string();
}


void PoolingLayer::set(const dimensions& new_input_dimensions, 
                       const dimensions& new_pool_dimensions,
                       const dimensions& new_stride_dimensions,
                       const dimensions& new_padding_dimensions,
                       const PoolingMethod& new_pooling_method,
                       const string new_name)
{
    if(new_pool_dimensions.size() != 2)
        throw runtime_error("Pool dimensions must be 2");

    if (new_stride_dimensions.size() != 2)
        throw runtime_error("Stride dimensions must be 2");

    if (new_padding_dimensions.size() != 2)
        throw runtime_error("Padding dimensions must be 2");

    if (new_pool_dimensions[0] > new_input_dimensions[0] || new_pool_dimensions[1] > new_input_dimensions[1])
        throw runtime_error("Pool dimensions cannot be bigger than input dimensions");

    if (new_stride_dimensions[0] <= 0 || new_stride_dimensions[1] <= 0)
        throw runtime_error("Stride dimensions cannot be 0 or lower");

    if (new_stride_dimensions[0] > new_input_dimensions[0] || new_stride_dimensions[1] > new_input_dimensions[0])
        throw runtime_error("Stride dimensions cannot be bigger than input dimensions");

    if (new_padding_dimensions[0] < 0 || new_padding_dimensions[1] < 0)
        throw runtime_error("Padding dimensions cannot be lower than 0");

    set_input_dimensions(new_input_dimensions);

    set_pool_size(new_pool_dimensions[0], new_pool_dimensions[1]);
    
    set_row_stride(new_stride_dimensions[0]);
    set_column_stride(new_stride_dimensions[1]);

    set_padding_height(new_padding_dimensions[0]);
    set_padding_width(new_padding_dimensions[1]);

    set_pooling_method(new_pooling_method);

    set_name(new_name);

    layer_type = Layer::Type::Pooling;

    name = "pooling_layer";
}


void PoolingLayer::set_input_dimensions(const dimensions& new_input_dimensions)
{
    if (new_input_dimensions.size() != 3)
        throw runtime_error("Input dimensions must be 3");

    input_dimensions = new_input_dimensions;
}


void PoolingLayer::set_padding_height(const Index& new_padding_height)
{
    padding_height = new_padding_height;
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
    if(new_pooling_method == "MaxPooling")
        pooling_method = PoolingMethod::MaxPooling;
    else if(new_pooling_method == "AveragePooling")
        pooling_method = PoolingMethod::AveragePooling;
    else
        throw runtime_error("Unknown pooling type: " + new_pooling_method + ".\batch_index");
}



void PoolingLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                     unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                     const bool& is_training)
{
    const TensorMap<Tensor<type, 4>> inputs = tensor_map_4(input_pairs[0]);

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
    }
}


void PoolingLayer::forward_propagate_average_pooling(const Tensor<type, 4>& inputs,
                                                     unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                                     const bool& is_training) const
{ 
    PoolingLayerForwardPropagation* pooling_layer_forward_propagation =
        static_cast<PoolingLayerForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 5>& image_patches = pooling_layer_forward_propagation->image_patches;

    const Eigen::array<int, 2>& reduction_dimensions = pooling_layer_forward_propagation->reduction_dimensions;

    Tensor<type, 4>& outputs = pooling_layer_forward_propagation->outputs;

    image_patches.device(*thread_pool_device) = inputs.extract_image_patches(
        pool_height,
        pool_width,
        row_stride,
        column_stride,
        1, 1,
        PADDING_VALID,
        type(padding_width));

    outputs.device(*thread_pool_device) = image_patches.mean(reduction_dimensions);
}


void PoolingLayer::forward_propagate_max_pooling(const Tensor<type, 4>& inputs,
                                                 unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                                 const bool& is_training) const
{
    const Index batch_samples_number = inputs.dimension(0);

    const Index output_width = get_output_width();
    const Index output_height = get_output_height();
    const Index channels = get_channels_number();

    PoolingLayerForwardPropagation* pooling_layer_forward_propagation =
        static_cast<PoolingLayerForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 5>& image_patches = pooling_layer_forward_propagation->image_patches;
    Tensor<type, 4>& outputs = pooling_layer_forward_propagation->outputs;

    const Eigen::array<ptrdiff_t, 4> outputs_dimensions_array({batch_samples_number,
                                                               output_height,
                                                               output_width,
                                                               channels});

    image_patches.device(*thread_pool_device) = inputs.extract_image_patches(
        pool_height,
        pool_width,
        row_stride,
        column_stride,
        1,1,
        PADDING_VALID,
        type(padding_width));

    outputs.device(*thread_pool_device)
        = image_patches.maximum(max_pooling_dimensions).reshape(outputs_dimensions_array);

    if (!is_training) return;

    const Index pool_size = pool_height * pool_width;
    const Index output_size = output_height * output_width * channels;
    const Eigen::array<ptrdiff_t, 3> output_dimensions({ output_height,output_width,channels });

    Tensor<Index, 4>& maximal_indices = pooling_layer_forward_propagation->maximal_indices;

    const Eigen::array<Index, 2> reshape_dimensions = { pool_size, output_size };

    #pragma omp parallel for

    for (Index batch_index = 0; batch_index < batch_samples_number; batch_index++)
    {
        const Tensor<type, 2> patches_flat = image_patches.chip(batch_index, 0)
                                                          .shuffle(shuffle_dimensions)
                                                          .reshape(reshape_dimensions);

        maximal_indices.chip(batch_index, 0) = patches_flat.argmax(0).reshape(output_dimensions);
    }
}


void PoolingLayer::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                  const vector<pair<type*, dimensions>>& delta_pairs,
                                  unique_ptr<LayerForwardPropagation>& forward_propagation,
                                  unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 4>> inputs = tensor_map_4(input_pairs[0]);
    const TensorMap<Tensor<type, 4>> deltas = tensor_map_4(delta_pairs[0]);

    switch(pooling_method)
    {
    case PoolingMethod::MaxPooling:
        back_propagate_max_pooling(inputs,
                                   deltas,
                                   forward_propagation,
                                   back_propagation);
        break;

    case PoolingMethod::AveragePooling:
        back_propagate_average_pooling(inputs,
                                       deltas,
                                       back_propagation);
        break;
    }

}


void PoolingLayer::back_propagate_max_pooling(const Tensor<type, 4>& inputs,
                                              const Tensor<type, 4>& deltas,
                                              unique_ptr<LayerForwardPropagation>& forward_propagation,
                                              unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_samples_number = inputs.dimension(0);

    const Index channels = inputs.dimension(3);

    const Index output_height = deltas.dimension(1);
    const Index output_width = deltas.dimension(2);

    // Forward propagation

    PoolingLayerForwardPropagation* pooling_layer_forward_propagation =
        static_cast<PoolingLayerForwardPropagation*>(forward_propagation.get());

    Tensor<Index, 4>& maximal_indices = pooling_layer_forward_propagation->maximal_indices;

    // Back propagation

    PoolingLayerBackPropagation* pooling_layer_back_propagation =
        static_cast<PoolingLayerBackPropagation*>(back_propagation.get());

    Tensor<type, 4>& input_derivatives = pooling_layer_back_propagation->input_derivatives;

    // Input derivatives
    
    input_derivatives.setZero();

    #pragma omp parallel
    for (Index batch_index = 0; batch_index < batch_samples_number; batch_index++)
        for (Index channel_index = 0; channel_index < channels; channel_index++)
            for (Index output_height_index = 0; output_height_index < output_height; output_height_index++)
                for (Index output_width_index = 0; output_width_index < output_width; output_width_index++)
                {
                    const Index maximal_index = maximal_indices(batch_index, output_height_index, output_width_index, channel_index);

                    const Index input_row = output_height_index * row_stride + maximal_index % pool_height;
                    const Index input_column = output_width_index * column_stride + maximal_index / pool_width;

                    input_derivatives(batch_index, input_row, input_column, channel_index) 
                        += deltas(batch_index, output_height_index, output_width_index, channel_index);
                }
}


void PoolingLayer::back_propagate_average_pooling(const Tensor<type, 4>& inputs,
                                                  const Tensor<type, 4>& deltas,
                                                  unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_samples_number = inputs.dimension(0);

    const Index input_height = inputs.dimension(1);
    const Index input_width = inputs.dimension(2);
    const Index channels = inputs.dimension(3);

    const Index output_height = deltas.dimension(1);
    const Index output_width = deltas.dimension(2);

    const Index pool_size = pool_height * pool_width;

    const Eigen::array<Index, 4> grad_extents = { batch_samples_number, 1, 1, 1 };

    // Back propagation

    PoolingLayerBackPropagation* pooling_layer_back_propagation =
        static_cast<PoolingLayerBackPropagation*>(back_propagation.get());

    Tensor<type, 4>& input_derivatives = pooling_layer_back_propagation->input_derivatives;

    Tensor<type, 4>& deltas_by_pool_size = pooling_layer_back_propagation->deltas_by_pool_size;

    deltas_by_pool_size.device(*thread_pool_device) = deltas / type(pool_size);

    // Input derivatives

    #pragma omp parallel for
    for (Index channel_index = 0; channel_index < channels; channel_index++)
        for (Index output_height_index = 0; output_height_index < output_height; output_height_index++)
        {
            const Index height_start = output_height_index * row_stride;
            const Index height_end = min(height_start + pool_height, input_height);

            for (Index output_width_index = 0; output_width_index < output_width; output_width_index++)
            {
                const Index width_start = output_width_index * column_stride;
                const Index width_end = min(width_start + pool_width, input_width);

                const Eigen::array<Index, 4> grad_offsets = { 0, output_height_index, output_width_index, channel_index };
                const Eigen::array<Index, 4> broadcast_dims = { 1, height_end - height_start, width_end - width_start, 1 };

                const Eigen::array<Index, 4> offsets = { 0, height_start, width_start, channel_index };
                const Eigen::array<Index, 4> extents = { batch_samples_number, height_end - height_start, width_end - width_start, 1 };

                input_derivatives.slice(offsets, extents) +=
                    deltas_by_pool_size.slice(grad_offsets, grad_extents).broadcast(broadcast_dims);
            }
        }
}


void PoolingLayer::to_XML(tinyxml2::XMLPrinter& printer) const
{
    printer.OpenElement("Pooling");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "InputDimensions", dimensions_to_string(get_input_dimensions()));
    add_xml_element(printer, "PoolHeight", to_string(get_pool_height()));
    add_xml_element(printer, "PoolWidth", to_string(get_pool_width()));
    add_xml_element(printer, "PoolingMethod", write_pooling_method());
    add_xml_element(printer, "ColumnStride", to_string(get_column_stride()));
    add_xml_element(printer, "RowStride", to_string(get_row_stride()));
    add_xml_element(printer, "PaddingHeight", to_string(get_padding_height()));
    add_xml_element(printer, "PaddingWidth", to_string(get_padding_width()));

    printer.CloseElement();
}


void PoolingLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* pooling_layer_element = document.FirstChildElement("Pooling");

    if(!pooling_layer_element)
        throw runtime_error("PoolingLayer layer element is nullptr.\batch_index");

    set_name(read_xml_string(pooling_layer_element, "Name"));

    set_input_dimensions(string_to_dimensions(read_xml_string(pooling_layer_element, "InputDimensions")));

    set_pool_size(read_xml_index(pooling_layer_element, "PoolHeight"), read_xml_index(pooling_layer_element, "PoolWidth"));

    set_pooling_method(read_xml_string(pooling_layer_element, "PoolingMethod"));

    set_column_stride(read_xml_index(pooling_layer_element, "ColumnStride"));
    set_row_stride(read_xml_index(pooling_layer_element, "RowStride"));

    set_padding_height(read_xml_index(pooling_layer_element, "PaddingHeight"));
    set_padding_width(read_xml_index(pooling_layer_element, "PaddingWidth"));

}


PoolingLayerForwardPropagation::PoolingLayerForwardPropagation(const Index& new_batch_samples_number, 
                                                               Layer* new_layer)
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

    return {(type*)outputs.data(), {batch_samples_number, output_height, output_width, channels}};
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
    
    outputs.resize(batch_samples_number,
                   output_height,
                   output_width,
                   channels);

    image_patches.resize(batch_samples_number,
                         pool_height,
                         pool_width,
                         output_height * output_width,
                         channels);
    
    maximal_indices.resize(batch_samples_number,
                           output_height,
                           output_width,
                           channels);
}


void PoolingLayerForwardPropagation::print() const
{
    cout << "PoolingLayer layer forward propagation" << endl
         << "Outputs:" << endl
         << outputs(0) << endl
         << "Image patches" << endl
         << image_patches << endl;
}


PoolingLayerBackPropagation::PoolingLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer);
}


void PoolingLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    batch_samples_number = new_batch_samples_number;

    layer = new_layer;

    const PoolingLayer* pooling_layer = static_cast<PoolingLayer*>(layer);

    const dimensions& input_dimensions = pooling_layer->get_input_dimensions();
    const dimensions& output_dimensions = pooling_layer->get_output_dimensions();   

    deltas_by_pool_size.resize(batch_samples_number, output_dimensions[0], output_dimensions[1], output_dimensions[2]);

    input_derivatives.resize(batch_samples_number, input_dimensions[0], input_dimensions[1], input_dimensions[2]);
}


vector<pair<type*, dimensions>> PoolingLayerBackPropagation::get_input_derivative_pairs() const
{
    const PoolingLayer* pooling_layer = static_cast<PoolingLayer*>(layer);

    const dimensions& input_dimensions = pooling_layer->get_input_dimensions();

    return {{(type*)(input_derivatives.data()),
            {batch_samples_number, input_dimensions[0], input_dimensions[1], input_dimensions[2]}} };
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
