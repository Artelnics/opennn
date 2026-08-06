//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pooling_layer.h"
#include "enum_map.h"

namespace opennn
{

namespace
{

const EnumMap<PoolingMethod>& pooling_method_map()
{
    static const vector<EnumMap<PoolingMethod>::Entry> entries = {
        {PoolingMethod::MaxPooling,     "MaxPooling"},
        {PoolingMethod::AveragePooling, "AveragePooling"},
        {PoolingMethod::FirstToken,     "FirstToken"}
    };
    static const EnumMap<PoolingMethod> instance{entries};
    return instance;
}

void validate_pooling_configuration(const Shape& input_shape,
                                    const Shape& pool_shape,
                                    const Shape& stride_shape,
                                    const Shape& padding_shape,
                                    const string& label)
{
    throw_if(input_shape.rank != 3,
             "Pooling layer '{}': input shape must have 3 dimensions, read {}.",
             label, input_shape.rank);
    throw_if(pool_shape.rank != 2,
             "Pooling layer '{}': pool shape must have 2 dimensions, read {}.",
             label, pool_shape.rank);
    throw_if(stride_shape.rank != 2,
             "Pooling layer '{}': stride must have 2 dimensions, read {}.",
             label, stride_shape.rank);
    throw_if(padding_shape.rank != 2,
             "Pooling layer '{}': padding must have 2 dimensions, read {}.",
             label, padding_shape.rank);

    throw_if(pool_shape[0] <= 0 || pool_shape[1] <= 0,
             "Pooling layer '{}': pool size must be positive, read {}.",
             label, shape_to_string(pool_shape));
    throw_if(stride_shape[0] <= 0 || stride_shape[1] <= 0,
             "Pooling layer '{}': stride must be positive, read {}.",
             label, shape_to_string(stride_shape));
    throw_if(padding_shape[0] < 0 || padding_shape[1] < 0,
             "Pooling layer '{}': padding cannot be negative, read {}.",
             label, shape_to_string(padding_shape));

    const Index padded_height = input_shape[0] + 2 * padding_shape[0];
    const Index padded_width  = input_shape[1] + 2 * padding_shape[1];
    const Shape padded_input_shape{padded_height, padded_width, input_shape[2]};
    throw_if(pool_shape[0] > padded_height || pool_shape[1] > padded_width,
             "Pooling layer '{}': pool shape {} cannot be bigger than padded input shape {}.",
             label, shape_to_string(pool_shape), shape_to_string(padded_input_shape));
    throw_if(stride_shape[0] > padded_height || stride_shape[1] > padded_width,
             "Pooling layer '{}': stride {} cannot be bigger than padded input shape {}.",
             label, shape_to_string(stride_shape), shape_to_string(padded_input_shape));
}

}

const string& pooling_method_to_string(PoolingMethod method)
{
    return pooling_method_map().to_string(method);
}

PoolingMethod string_to_pooling_method(const string& name)
{
    return pooling_method_map().from_string(name);
}

Pooling::Pooling(const Shape& new_input_shape,
                 const Shape& new_pool_dimensions,
                 const Shape& new_stride_shape,
                 const Shape& new_padding_dimensions,
                 const string& new_pooling_method,
                 const string& new_name)
    : Layer(LayerType::Pooling)
{
    operators = {&pool};
    set(new_input_shape,
        new_pool_dimensions,
        new_stride_shape,
        new_padding_dimensions,
        new_pooling_method,
        new_name);
}

Shape Pooling::get_output_shape() const
{
    return { get_output_height(), get_output_width(), input_channels };
}

Index Pooling::get_output_height() const
{
    return (input_height - pool_height + 2 * padding_height) / row_stride + 1;
}

Index Pooling::get_output_width() const
{
    return (input_width - pool_width + 2 * padding_width) / column_stride + 1;
}

bool Pooling::is_passthrough() const noexcept
{
    return pool_height == 1
        && pool_width == 1
        && row_stride == 1
        && column_stride == 1
        && padding_height == 0
        && padding_width == 0;
}

vector<TensorSpec> Pooling::get_forward_specs(Index batch_size) const
{
    if (is_passthrough())
        return {};

    const Shape out_shape = get_output_shape();

    const Shape indices_shape = (pooling_method == PoolingMethod::MaxPooling
                                 && compute_device != Device::CUDA)
        ? Shape{batch_size}.append(out_shape)
        : Shape{};

    return {
        {indices_shape,                           Type::FP32},
        {Shape{batch_size}.append(out_shape), compute_dtype},
    };
}

vector<TensorSpec> Pooling::get_backward_specs(Index batch_size) const
{
    if (is_passthrough())
        return {};

    return Layer::get_backward_specs(batch_size);
}

void Pooling::forward_propagate(ForwardPropagation& forward_propagation,
                                size_t layer,
                                bool is_training)
{
    if (is_passthrough())
        return;

    Layer::forward_propagate(forward_propagation, layer, is_training);
}

void Pooling::back_propagate(ForwardPropagation& forward_propagation,
                             BackPropagation& back_propagation,
                             size_t layer) const
{
    if (is_passthrough())
        return;

    Layer::back_propagate(forward_propagation, back_propagation, layer);
}

void Pooling::update_pool_operator()
{
    pool.set(input_height, input_width, input_channels,
             pool_height, pool_width,
             row_stride, column_stride,
             padding_height, padding_width,
             pooling_method == PoolingMethod::MaxPooling ? PoolOperator::Max : PoolOperator::Average);

    pool.output_slots = {Output, MaximalIndices};
}

void Pooling::set(const Shape& new_input_shape,
                  const Shape& new_pool_dimensions,
                  const Shape& new_stride_shape,
                  const Shape& new_padding_dimensions,
                  const string& new_pooling_method,
                  const string& new_label)
{
    validate_pooling_configuration(new_input_shape,
                                   new_pool_dimensions,
                                   new_stride_shape,
                                   new_padding_dimensions,
                                   new_label);

    input_height    = new_input_shape[0];
    input_width     = new_input_shape[1];
    input_channels  = new_input_shape[2];

    pool_height     = new_pool_dimensions[0];
    pool_width      = new_pool_dimensions[1];

    row_stride      = new_stride_shape[0];
    column_stride   = new_stride_shape[1];

    padding_height  = new_padding_dimensions[0];
    padding_width   = new_padding_dimensions[1];

    pooling_method  = string_to_pooling_method(new_pooling_method);

    set_label(new_label);

    update_pool_operator();
}

void Pooling::set_input_shape(const Shape& new_input_shape)
{
    throw_if(new_input_shape.rank != 3, "Input shape must be 3");

    input_height = new_input_shape[0];
    input_width = new_input_shape[1];
    input_channels = new_input_shape[2];

    update_pool_operator();
}

void Pooling::set_pooling_method(const string& new_pooling_method)
{
    pooling_method = string_to_pooling_method(new_pooling_method);

    update_pool_operator();
}

void Pooling::read_JSON_body(const Json* pooling_layer_element)
{
    const Shape pool_shape{
        read_json_index(pooling_layer_element, "PoolHeight"),
        read_json_index(pooling_layer_element, "PoolWidth")
    };
    const Shape stride_shape{
        read_json_index(pooling_layer_element, "RowStride"),
        read_json_index(pooling_layer_element, "ColumnStride")
    };
    const Shape padding_shape{
        read_json_index(pooling_layer_element, "PaddingHeight"),
        read_json_index(pooling_layer_element, "PaddingWidth")
    };

    validate_pooling_configuration(get_input_shape(),
                                   pool_shape,
                                   stride_shape,
                                   padding_shape,
                                   label);

    const PoolingMethod new_pooling_method =
        string_to_pooling_method(read_json_string(pooling_layer_element, "PoolingMethod"));

    pool_height     = pool_shape[0];
    pool_width      = pool_shape[1];
    row_stride      = stride_shape[0];
    column_stride   = stride_shape[1];
    padding_height  = padding_shape[0];
    padding_width   = padding_shape[1];
    pooling_method  = new_pooling_method;

    update_pool_operator();
}

void Pooling::write_JSON_body(JsonWriter& printer) const
{
    write_json(printer, {
        {"PoolHeight", get_pool_height()},
        {"PoolWidth", get_pool_width()},
        {"PoolingMethod", pooling_method_to_string(pooling_method)},
        {"ColumnStride", get_column_stride()},
        {"RowStride", get_row_stride()},
        {"PaddingHeight", get_padding_height()},
        {"PaddingWidth", get_padding_width()}
    });
}


}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
