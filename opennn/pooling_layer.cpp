//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "pooling_layer.h"
#include "loss.h"

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


Index Pooling::get_output_height() const
{
    return (get_input_height() - pool_height + 2 * padding_height) / row_stride + 1;
}


Index Pooling::get_output_width() const
{
    return (get_input_width() - pool_width + 2 * padding_width) / column_stride + 1;
}


void Pooling::set(const Shape& new_input_shape,
                  const Shape& new_pool_dimensions,
                  const Shape& new_stride_shape,
                  const Shape& new_padding_dimensions,
                  const string& new_pooling_method,
                  const string& new_label)
{
    if(new_pool_dimensions.rank != 2)
        throw runtime_error("Pool shape must be 2");

    if (new_stride_shape.rank != 2)
        throw runtime_error("Stride shape must be 2");

    if (new_padding_dimensions.rank != 2)
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

#ifdef CUDA

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
    if (new_input_shape.rank != 3)
        throw runtime_error("Input shape must be 3");

    input_shape = new_input_shape;
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

#ifdef CUDA
    if (pooling_method == "MaxPooling")
        pooling_mode = CUDNN_POOLING_MAX;
    else if (pooling_method == "AveragePooling")
        pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
#endif
}


void Pooling::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training)
{
    const TensorView& input = forward_propagation.views[layer][Inputs][0];
    TensorView& output = forward_propagation.views[layer][Outputs][0];

    PoolingArguments args;
    args.pool_dimensions = {pool_height, pool_width};
    args.stride_shape = {row_stride, column_stride};
    args.padding_shape = {padding_height, padding_width};

    if(pooling_method == "MaxPooling")
    {
        TensorView& maximal_indices = forward_propagation.views[layer][MaximalIndices][0];
        max_pooling(input, output, maximal_indices, args, is_training);
    }
    else if(pooling_method == "AveragePooling")
        average_pooling(input, output, args);
}


void Pooling::back_propagate(ForwardPropagation& forward_propagation,
                             BackPropagation& back_propagation,
                             size_t layer) const
{
    const TensorMap4 output_gradients = tensor_map<4>(back_propagation.backward_views[layer][OutputGradients][0]);
    TensorMap4 input_gradients = tensor_map<4>(back_propagation.backward_views[layer][InputGradients][0]);

    if(pooling_method == "MaxPooling")
    {
        const TensorMap4 maximal_indices = tensor_map<4>(forward_propagation.views[layer][MaximalIndices][0]);

        const Index batch_size = input_gradients.dimension(0);
        const Index input_height = input_gradients.dimension(1);
        const Index input_width = input_gradients.dimension(2);
        const Index channels = input_gradients.dimension(3);

        const Index output_height = output_gradients.dimension(1);
        const Index output_width = output_gradients.dimension(2);

        input_gradients.setZero();

        #pragma omp parallel for collapse(2)
        for(Index batch_index = 0; batch_index < batch_size; ++batch_index)
            for(Index channel_index = 0; channel_index < channels; ++channel_index)
                for(Index output_row = 0; output_row < output_height; ++output_row)
                    for(Index output_column = 0; output_column < output_width; ++output_column)
                    {
                        const Index max_index_flat = static_cast<Index>(
                            maximal_indices(batch_index, output_row, output_column, channel_index));

                        const Index pool_row = max_index_flat / pool_width;
                        const Index pool_column = max_index_flat % pool_width;

                        const Index input_row = output_row * row_stride + pool_row - padding_height;
                        const Index input_column = output_column * column_stride + pool_column - padding_width;

                        if(input_row >= 0 && input_row < input_height && input_column >= 0 && input_column < input_width)
                            input_gradients(batch_index, input_row, input_column, channel_index) +=
                                output_gradients(batch_index, output_row, output_column, channel_index);
                    }

    }
    else if(pooling_method == "AveragePooling")
    {
        const Index batch_size = input_gradients.dimension(0);
        const Index input_height = input_gradients.dimension(1);
        const Index input_width = input_gradients.dimension(2);
        const Index channels = input_gradients.dimension(3);

        const Index output_height = output_gradients.dimension(1);
        const Index output_width = output_gradients.dimension(2);

        const type inv_pool_size = type(1) / (pool_height * pool_width);

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
}


void Pooling::back_propagate_max_pooling(const Tensor4& output_gradients,
                                         Tensor4& input_gradients) const
{
/*
    const Index batch_size = input_gradients.dimension(0);
    const Index input_height = input_gradients.dimension(1);
    const Index input_width = input_gradients.dimension(2);
    const Index channels = input_gradients.dimension(3);

    const Index output_height = output_gradients.dimension(1);
    const Index output_width = output_gradients.dimension(2);

    input_gradients.setZero();
/*
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
*/
}


void Pooling::back_propagate_average_pooling(const Tensor4& output_gradients, Tensor4& input_gradients) const
{
    const Index batch_size = input_gradients.dimension(0);
    const Index input_height = input_gradients.dimension(1);
    const Index input_width = input_gradients.dimension(2);
    const Index channels = input_gradients.dimension(3);

    const Index output_height = output_gradients.dimension(1);
    const Index output_width = output_gradients.dimension(2);

    const type inv_pool_size = type(1) / (pool_height * pool_width);

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


void Pooling::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Pooling");

    write_xml_properties(printer, {
        {"Label", label},
        {"InputDimensions", shape_to_string(get_input_shape())},
        {"PoolHeight", to_string(get_pool_height())},
        {"PoolWidth", to_string(get_pool_width())},
        {"PoolingMethod", pooling_method},
        {"ColumnStride", to_string(get_column_stride())},
        {"RowStride", to_string(get_row_stride())},
        {"PaddingHeight", to_string(get_padding_height())},
        {"PaddingWidth", to_string(get_padding_width())}
    });

    printer.CloseElement();
}


void Pooling::from_XML(const XMLDocument& document)
{
    const XMLElement* pooling_layer_element = get_xml_root(document, "Pooling");

    set_label(read_xml_string(pooling_layer_element, "Label"));
    set_input_shape(string_to_shape(read_xml_string(pooling_layer_element, "InputDimensions")));
    set_pool_size(read_xml_index(pooling_layer_element, "PoolHeight"), read_xml_index(pooling_layer_element, "PoolWidth"));
    set_pooling_method(read_xml_string(pooling_layer_element, "PoolingMethod"));
    set_column_stride(read_xml_index(pooling_layer_element, "ColumnStride"));
    set_row_stride(read_xml_index(pooling_layer_element, "RowStride"));
    set_padding_height(read_xml_index(pooling_layer_element, "PaddingHeight"));
    set_padding_width(read_xml_index(pooling_layer_element, "PaddingWidth"));
}


#ifdef CUDA

void PoolingForwardPropagationCuda::initialize()
{
    const Pooling* pooling_layer = static_cast<Pooling*>(layer);

    const Index input_height = pooling_layer->get_input_height();
    const Index input_width = pooling_layer->get_input_width();
    const Index channels = pooling_layer->get_channels_number();

    const Index output_height = pooling_layer->get_output_height();
    const Index output_width = pooling_layer->get_output_width();

    // Inputs
    
    cudnnSetTensor4dDescriptor(input_tensor_descriptor,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_FLOAT,
                               batch_size,
                               channels,
                               input_height,
                               input_width);

    // Outputs

    outputs.set_descriptor({ batch_size, output_height, output_width, channels });
}


vector<TensorView*> PoolingForwardPropagationCuda::get_workspace_views()
{
    return { &outputs };
}


void PoolingForwardPropagationCuda::free()
{
    cudnnDestroyTensorDescriptor(input_tensor_descriptor);
    input_tensor_descriptor = nullptr;
}


void PoolingBackPropagationCuda::initialize()
{
    const Pooling* pooling_layer = static_cast<Pooling*>(layer);

    const Index input_height = pooling_layer->get_input_height();
    const Index input_width = pooling_layer->get_input_width();
    const Index channels = pooling_layer->get_channels_number();

    // Input derivatives

    input_gradients = {TensorView({batch_size, input_height, input_width, channels})};
}

#endif

REGISTER(Layer, Pooling, "Pooling")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
