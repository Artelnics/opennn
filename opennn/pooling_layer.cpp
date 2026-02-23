//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "pooling_layer.h"

namespace opennn
{

Pooling::Pooling(const Shape& new_input_shape,
                 const Shape& new_pool_dimensions,
                 const Shape& new_stride_shape,
                 const Shape& new_padding_dimensions,
                 const string& new_pooling_method,
                 const string& new_name) : Layer()
{
    name = "Pooling";

    set(new_input_shape,
        new_pool_dimensions,
        new_stride_shape,
        new_padding_dimensions,
        new_pooling_method,
        new_name);
}


Shape Pooling::get_output_shape() const
{
    const Index rows_number = get_output_height();
    const Index columns_number = get_output_width();
    const Index channels = input_shape[2];

    return { rows_number, columns_number, channels };
}


Index Pooling::get_input_height() const
{
    return input_shape[0];
}


Index Pooling::get_input_width() const
{
    return input_shape[1];
}


Index Pooling::get_channels_number() const
{
    return input_shape[2];
}


Index Pooling::get_output_height() const
{
    return (get_input_height() - pool_height + 2 * padding_height) / row_stride + 1;
}


Index Pooling::get_output_width() const
{
    return (get_input_width() - pool_width + 2 * padding_width) / column_stride + 1;
}


Index Pooling::get_padding_height() const
{
    return padding_height;
}


Index Pooling::get_padding_width() const
{
    return padding_width;
}


Index Pooling::get_row_stride() const
{
    return row_stride;
}


Index Pooling::get_column_stride() const
{
    return column_stride;
}


Index Pooling::get_pool_height() const
{
    return pool_height;
}


Index Pooling::get_pool_width() const
{
    return pool_width;
}


string Pooling::get_pooling_method() const
{
    return pooling_method;
}


Shape Pooling::get_input_shape() const
{
    return input_shape;
}


void Pooling::print() const
{
    cout << "Pooling layer" << endl
         << "Input shape: " << input_shape << endl
         << "Output shape: " << get_output_shape() << endl;
}


void Pooling::set(const Shape& new_input_shape,
                  const Shape& new_pool_dimensions,
                  const Shape& new_stride_shape,
                  const Shape& new_padding_dimensions,
                  const string& new_pooling_method,
                  const string& new_label)
{
    if(new_pool_dimensions.size() != 2)
        throw runtime_error("Pool shape must be 2");

    if (new_stride_shape.size() != 2)
        throw runtime_error("Stride shape must be 2");

    if (new_padding_dimensions.size() != 2)
        throw runtime_error("Padding shape must be 2");

    if (new_pool_dimensions[0] > new_input_shape[0] || new_pool_dimensions[1] > new_input_shape[1])
        throw runtime_error("Pool shape cannot be bigger than input shape");

    if (new_stride_shape[0] <= 0 || new_stride_shape[1] <= 0)
        throw runtime_error("Stride shape cannot be 0 or lower");

    if (new_stride_shape[0] > new_input_shape[0] || new_stride_shape[1] > new_input_shape[0])
        throw runtime_error("Stride shape cannot be bigger than input shape");

    if (new_padding_dimensions[0] < 0 || new_padding_dimensions[1] < 0)
        throw runtime_error("Padding shape cannot be lower than 0");

    input_shape = new_input_shape;

    set_pool_size(new_pool_dimensions[0], new_pool_dimensions[1]);

    set_row_stride(new_stride_shape[0]);
    set_column_stride(new_stride_shape[1]);

    set_padding_height(new_padding_dimensions[0]);
    set_padding_width(new_padding_dimensions[1]);

    set_pooling_method(new_pooling_method);

    set_label(new_label);

    label = "pooling_layer";

#ifdef OPENNN_CUDA

    // Pooling descriptor

    cudnnCreatePoolingDescriptor(&pooling_descriptor);

    cudnnSetPooling2dDescriptor(pooling_descriptor,
                                pooling_mode,
                                CUDNN_PROPAGATE_NAN,
                                pool_height, pool_width,
                                padding_height, padding_width,
                                row_stride, column_stride);

#endif

}


void Pooling::set_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.size() != 3)
        throw runtime_error("Input shape must be 3");

    input_shape = new_input_shape;
}


void Pooling::set_padding_height(const Index new_padding_height)
{
    padding_height = new_padding_height;
}


void Pooling::set_padding_width(const Index new_padding_width)
{
    padding_width = new_padding_width;
}


void Pooling::set_row_stride(const Index new_row_stride)
{
    row_stride = new_row_stride;
}


void Pooling::set_column_stride(const Index new_column_stride)
{
    column_stride = new_column_stride;
}


void Pooling::set_pool_size(const Index new_pool_rows_number,
                            Index new_pool_columns_number)
{
    pool_height = new_pool_rows_number;
    pool_width = new_pool_columns_number;
}


void Pooling::set_pooling_method(const string& new_pooling_method)
{
    if(new_pooling_method != "MaxPooling" && new_pooling_method != "AveragePooling")
        throw runtime_error("Unknown pooling type: " + new_pooling_method);

    pooling_method = new_pooling_method;
}


void Pooling::forward_propagate(const vector<TensorView>& input_views,
                                unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                bool is_training)
{
    const TensorMap4 inputs = tensor_map<4>(input_views[0]);

    if(pooling_method == "MaxPooling")
        forward_propagate_max_pooling(inputs,
                                      layer_forward_propagation,
                                      is_training);
    else if(pooling_method == "AveragePooling")
        forward_propagate_average_pooling(inputs,
                                          layer_forward_propagation,
                                          is_training);
}


void Pooling::forward_propagate_average_pooling(const Tensor4& inputs,
                                                unique_ptr<LayerForwardPropagation>& forward_propagation,
                                                bool) const
{
    TensorMap4 outputs = tensor_map<4>(forward_propagation->outputs);
    const Index batch_size = inputs.dimension(0);
    const Index input_height = inputs.dimension(1);
    const Index input_width = inputs.dimension(2);
    const Index channels = inputs.dimension(3);
    const Index output_height = outputs.dimension(1);
    const Index output_width = outputs.dimension(2);
    const type inv_pool_size = type(1) / (pool_height * pool_width);

    #pragma omp parallel for collapse(2)
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for(Index channel_index = 0; channel_index < channels; ++channel_index)
            for(Index output_row = 0; output_row < output_height; ++output_row)
                for(Index output_column = 0; output_column < output_width; ++output_column)
                {
                    const Index input_row_start = output_row * row_stride - padding_height;
                    const Index input_column_start = output_column * column_stride - padding_width;

                    type sum = 0;

                    for(Index pool_row = 0; pool_row < pool_height; ++pool_row)
                        for(Index pool_column = 0; pool_column < pool_width; ++pool_column)
                        {
                            const Index input_row = input_row_start + pool_row;
                            const Index input_column = input_column_start + pool_column;

                            if(input_row >= 0 && input_row < input_height && input_column >= 0 && input_column < input_width)
                                sum += inputs(batch_index, input_row, input_column, channel_index);
                        }

                    outputs(batch_index, output_row, output_column, channel_index) = sum * inv_pool_size;
                }
}


void Pooling::forward_propagate_max_pooling(const Tensor4& inputs,
                                            unique_ptr<LayerForwardPropagation>& forward_propagation,
                                            bool is_training) const
{
    TensorMap4 outputs = tensor_map<4>(forward_propagation->outputs);
    PoolingForwardPropagation* pooling_forward_propagation = static_cast<PoolingForwardPropagation*>(forward_propagation.get());

    const Index batch_size = inputs.dimension(0);
    const Index input_height = inputs.dimension(1);
    const Index input_width = inputs.dimension(2);
    const Index channels = inputs.dimension(3);
    const Index output_height = outputs.dimension(1);
    const Index output_width = outputs.dimension(2);

    #pragma omp parallel for collapse(2)
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for(Index channel_index = 0; channel_index < channels; ++channel_index)
            for(Index output_row = 0; output_row < output_height; ++output_row)
                for(Index output_column = 0; output_column < output_width; ++output_column)
                {
                    const Index input_row_start = output_row * row_stride - padding_height;
                    const Index input_column_start = output_column * column_stride - padding_width;

                    type maximum_value = -numeric_limits<type>::infinity();
                    Index maximum_index = 0;

                    for(Index pool_row = 0; pool_row < pool_height; ++pool_row)
                        for(Index pool_column = 0; pool_column < pool_width; ++pool_column)
                        {
                            const Index input_row = input_row_start + pool_row;
                            const Index input_column = input_column_start + pool_column;

                            if(input_row >= 0 && input_row < input_height && input_column >= 0 && input_column < input_width)
                            {
                                const type current_value = inputs(batch_index, input_row, input_column, channel_index);

                                if(current_value > maximum_value)
                                {
                                    maximum_value = current_value;
                                    maximum_index = pool_row * pool_width + pool_column;
                                }
                            }
                        }

                    outputs(batch_index, output_row, output_column, channel_index) =
                        (maximum_value == -numeric_limits<type>::infinity()) ? type(0) : maximum_value;

                    if(is_training)
                        pooling_forward_propagation->maximal_indices(batch_index, output_row, output_column, channel_index) = maximum_index;
                }
}


void Pooling::back_propagate(const vector<TensorView>& input_views,
                             const vector<TensorView>& output_gradient_views,
                             unique_ptr<LayerForwardPropagation>& forward_propagation,
                             unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap4 inputs = tensor_map<4>(input_views[0]);
    const TensorMap4 output_gradients = tensor_map<4>(output_gradient_views[0]);

    if(pooling_method == "MaxPooling")
        back_propagate_max_pooling(inputs,
                                   output_gradients,
                                   forward_propagation,
                                   back_propagation);
    else if(pooling_method == "AveragePooling")
        back_propagate_average_pooling(inputs,
                                       output_gradients,
                                       back_propagation);
}


void Pooling::back_propagate_max_pooling(const Tensor4& inputs,
                                         const Tensor4& output_gradients,
                                         unique_ptr<LayerForwardPropagation>& forward_propagation,
                                         unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_size = inputs.dimension(0);
    const Index input_height = inputs.dimension(1);
    const Index input_width = inputs.dimension(2);
    const Index channels = inputs.dimension(3);

    const Index output_height = output_gradients.dimension(1);
    const Index output_width = output_gradients.dimension(2);

    PoolingForwardPropagation* pooling_forward_propagation = static_cast<PoolingForwardPropagation*>(forward_propagation.get());
    TensorMap4 input_gradients = tensor_map<4>(back_propagation->input_gradients[0]);

    input_gradients.setZero();

    #pragma omp parallel for collapse(2)
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for(Index channel_index = 0; channel_index < channels; ++channel_index)
            for(Index output_row = 0; output_row < output_height; ++output_row)
                for(Index output_column = 0; output_column < output_width; ++output_column)
                {
                    const Index max_index_flat = pooling_forward_propagation->maximal_indices(batch_index, output_row, output_column, channel_index);

                    const Index pool_row = max_index_flat / pool_width;
                    const Index pool_column = max_index_flat % pool_width;

                    const Index input_row = output_row * row_stride + pool_row - padding_height;
                    const Index input_column = output_column * column_stride + pool_column - padding_width;

                    if(input_row >= 0 && input_row < input_height && input_column >= 0 && input_column < input_width)
                        input_gradients(batch_index, input_row, input_column, channel_index) +=
                            output_gradients(batch_index, output_row, output_column, channel_index);
                }
}


void Pooling::back_propagate_average_pooling(const Tensor4& inputs,
                                             const Tensor4& output_gradients,
                                             unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_size = inputs.dimension(0);
    const Index input_height = inputs.dimension(1);
    const Index input_width = inputs.dimension(2);
    const Index channels = inputs.dimension(3);

    const Index output_height = output_gradients.dimension(1);
    const Index output_width = output_gradients.dimension(2);

    const type inv_pool_size = type(1) / (pool_height * pool_width);

    TensorMap4 input_gradients = tensor_map<4>(back_propagation->input_gradients[0]);
    input_gradients.setZero();

    #pragma omp parallel for collapse(2)
    for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for(Index channel_index = 0; channel_index < channels; ++channel_index)
            for(Index output_row = 0; output_row < output_height; ++output_row)
                for(Index output_column = 0; output_column < output_width; ++output_column)
                {
                    const type average_gradient = output_gradients(batch_index, output_row, output_column, channel_index) * inv_pool_size;

                    const Index input_row_start = output_row * row_stride - padding_height;
                    const Index input_column_start = output_column * column_stride - padding_width;

                    for(Index pool_row = 0; pool_row < pool_height; ++pool_row)
                        for(Index pool_column = 0; pool_column < pool_width; ++pool_column)
                        {
                            const Index input_row = input_row_start + pool_row;
                            const Index input_column = input_column_start + pool_column;

                            if(input_row >= 0 && input_row < input_height && input_column >= 0 && input_column < input_width)
                                input_gradients(batch_index, input_row, input_column, channel_index) += average_gradient;
                        }
                }
}


#ifdef OPENNN_CUDA

void Pooling::forward_propagate(const vector<TensorViewCuda>& inputs,
                                     unique_ptr<LayerForwardPropagationCuda>& forward_propagation,
                                     bool is_training)
{
    TensorViewCuda outputs = forward_propagation->outputs;

    // Forward propagation

    PoolingForwardPropagationCuda* pooling_forward_propagation
        = static_cast<PoolingForwardPropagationCuda*>(forward_propagation.get());

    const cudnnTensorDescriptor_t input_tensor_descriptor = pooling_forward_propagation->input_tensor_descriptor;

    // Pooling

    CHECK_CUDNN(cudnnPoolingForward(get_cudnn_handle(),
        pooling_descriptor,
        &alpha,
        input_tensor_descriptor,
        inputs[0].data,
        &beta,
        outputs.get_descriptor(),
        outputs.data));
}


void Pooling::back_propagate(const vector<TensorViewCuda>& inputs,
                                  const vector<TensorViewCuda>& output_gradients,
                                  unique_ptr<LayerForwardPropagationCuda>& forward_propagation,
                                  unique_ptr<LayerBackPropagationCuda>& back_propagation) const
{
    // Forward propagation
    
    const TensorViewCuda& outputs = forward_propagation->outputs;

    const PoolingForwardPropagationCuda* pooling_forward_propagation
        = static_cast<PoolingForwardPropagationCuda*>(forward_propagation.get());

    const cudnnTensorDescriptor_t input_tensor_descriptor = pooling_forward_propagation->input_tensor_descriptor;

    // Back propagation

    type* input_gradients = back_propagation->input_gradients[0].data;

    // Pooling

    CHECK_CUDNN(cudnnPoolingBackward(get_cudnn_handle(),
        pooling_descriptor,
        &alpha,
        outputs.get_descriptor(),
        outputs.data,
        outputs.get_descriptor(),
        output_gradients[0].data,
        input_tensor_descriptor,
        inputs[0].data,
        &beta,
        input_tensor_descriptor,
        input_gradients));
}

#endif


void Pooling::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Pooling");

    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "InputDimensions", shape_to_string(get_input_shape()));
    add_xml_element(printer, "PoolHeight", to_string(get_pool_height()));
    add_xml_element(printer, "PoolWidth", to_string(get_pool_width()));
    add_xml_element(printer, "PoolingMethod", pooling_method);
    add_xml_element(printer, "ColumnStride", to_string(get_column_stride()));
    add_xml_element(printer, "RowStride", to_string(get_row_stride()));
    add_xml_element(printer, "PaddingHeight", to_string(get_padding_height()));
    add_xml_element(printer, "PaddingWidth", to_string(get_padding_width()));

    printer.CloseElement();
}


void Pooling::from_XML(const XMLDocument& document)
{
    const XMLElement* pooling_layer_element = document.FirstChildElement("Pooling");

    if(!pooling_layer_element)
        throw runtime_error("Pooling layer element is nullptr.\batch_index");

    set_label(read_xml_string(pooling_layer_element, "Label"));
    set_input_shape(string_to_shape(read_xml_string(pooling_layer_element, "InputDimensions")));
    set_pool_size(read_xml_index(pooling_layer_element, "PoolHeight"), read_xml_index(pooling_layer_element, "PoolWidth"));
    set_pooling_method(read_xml_string(pooling_layer_element, "PoolingMethod"));
    set_column_stride(read_xml_index(pooling_layer_element, "ColumnStride"));
    set_row_stride(read_xml_index(pooling_layer_element, "RowStride"));
    set_padding_height(read_xml_index(pooling_layer_element, "PaddingHeight"));
    set_padding_width(read_xml_index(pooling_layer_element, "PaddingWidth"));
}


PoolingForwardPropagation::PoolingForwardPropagation(const Index new_batch_size,
                                                     Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


void PoolingForwardPropagation::initialize()
{
    const Pooling* pooling_layer = static_cast<Pooling*>(layer);

    const Index output_height = pooling_layer->get_output_height();
    const Index output_width = pooling_layer->get_output_width();
    const Index channels = pooling_layer->get_channels_number();

    outputs.shape = {batch_size, output_height, output_width, channels};

    if (pooling_layer->get_pooling_method() == "MaxPooling")
        maximal_indices.resize(batch_size,
                               output_height,
                               output_width,
                               channels);
}


vector<TensorView*> PoolingForwardPropagation::get_workspace_views()
{
    return {&outputs};
}


void PoolingForwardPropagation::print() const
{
    cout << "Pooling layer forward propagation" << endl
         << "Outputs:" << endl
         << outputs.shape << endl;
}


PoolingBackPropagation::PoolingBackPropagation(const Index new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


void PoolingBackPropagation::initialize()
{
    const Pooling* pooling_layer = static_cast<Pooling*>(layer);

    const Shape& input_shape = pooling_layer->get_input_shape();

    Shape full_input_shape = { batch_size };
    full_input_shape.insert(full_input_shape.end(), input_shape.begin(), input_shape.end());

    input_gradients_memory.resize(1);
    input_gradients_memory[0].resize(full_input_shape.count());
    input_gradients.resize(1);
    input_gradients[0].data = input_gradients_memory[0].data();
    input_gradients[0].shape = full_input_shape;
}


void PoolingBackPropagation::print() const
{
    cout << "Pooling layer back propagation" << endl;
    cout << "Input output_gradients:" << endl
         << input_gradients[0].shape << endl;
}


#ifdef OPENNN_CUDA

PoolingForwardPropagationCuda::PoolingForwardPropagationCuda(const Index new_batch_size, Layer* new_layer)
    : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void PoolingForwardPropagationCuda::initialize()
{
    const Pooling* pooling_layer = static_cast<Pooling*>(layer);

    const Index input_height = pooling_layer->get_input_height();
    const Index input_width = pooling_layer->get_input_width();
    const Index channels = pooling_layer->get_channels_number();

    const Index output_height = pooling_layer->get_output_height();
    const Index output_width = pooling_layer->get_output_width();

    // Inputs
    
    cudnnCreateTensorDescriptor(&input_tensor_descriptor);

    cudnnSetTensor4dDescriptor(input_tensor_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               batch_size,
                               channels,
                               input_height,
                               input_width);

    // Outputs

    outputs.set_descriptor({ batch_size, channels, output_height, output_width });
}


vector<TensorViewCuda*> PoolingForwardPropagationCuda::get_workspace_views()
{
    return { &outputs };
}


void PoolingForwardPropagationCuda::print() const
{
    const Pooling* pooling_layer = static_cast<const Pooling*>(layer);

    const Index output_height = pooling_layer->get_output_height();
    const Index output_width = pooling_layer->get_output_width();
    const Index channels = pooling_layer->get_channels_number();

    cout << "Pooling layer forward propagation CUDA" << endl
         << "Outputs:" << endl
         << matrix_4d_from_device(outputs.data, batch_size, output_height, output_width, channels) << endl;
}


void PoolingForwardPropagationCuda::free()
{
    cudnnDestroyTensorDescriptor(input_tensor_descriptor);
    input_tensor_descriptor = nullptr;
}


PoolingBackPropagationCuda::PoolingBackPropagationCuda(const Index new_batch_size, Layer* new_layer)
    : LayerBackPropagationCuda()
{
    set(new_batch_size, new_layer);
}


void PoolingBackPropagationCuda::initialize()
{
    const Pooling* pooling_layer = static_cast<Pooling*>(layer);

    const Index input_height = pooling_layer->get_input_height();
    const Index input_width = pooling_layer->get_input_width();
    const Index channels = pooling_layer->get_channels_number();

    // Input derivatives

    input_gradients.resize(1);
    input_gradients[0].resize({ batch_size, input_height, input_width, channels});
}


void PoolingBackPropagationCuda::print() const
{
    const Pooling* pooling_layer = static_cast<const Pooling*>(layer);

    const Index input_height = pooling_layer->get_input_height();
    const Index input_width = pooling_layer->get_input_width();
    const Index channels = pooling_layer->get_channels_number();

    cout << "Pooling layer back propagation CUDA" << endl
         << "Input output_gradients:" << endl
         << matrix_4d_from_device(input_gradients[0].data, batch_size, input_height, input_width, channels) << endl;
}


REGISTER(LayerForwardPropagationCuda, PoolingForwardPropagationCuda, "Pooling")
REGISTER(LayerBackPropagationCuda, PoolingBackPropagationCuda, "Pooling")

#endif

REGISTER(Layer, Pooling, "Pooling")
REGISTER(LayerForwardPropagation, PoolingForwardPropagation, "Pooling")
REGISTER(LayerBackPropagation, PoolingBackPropagation, "Pooling")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
