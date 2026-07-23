//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
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

vector<TensorSpec> Pooling::get_forward_specs(Index batch_size) const
{
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
    throw_if(new_pool_dimensions.rank != 2, "Pool shape must be 2");

    throw_if(new_stride_shape.rank != 2, "Stride shape must be 2");

    throw_if(new_padding_dimensions.rank != 2, "Padding shape must be 2");

    throw_if(new_stride_shape[0] <= 0 || new_stride_shape[1] <= 0, "Stride must be positive.");

    throw_if(new_padding_dimensions[0] < 0 || new_padding_dimensions[1] < 0, "Padding shape cannot be negative");

    throw_if(new_pool_dimensions[0] > new_input_shape[0] + 2 * new_padding_dimensions[0]
             || new_pool_dimensions[1] > new_input_shape[1] + 2 * new_padding_dimensions[1],
             "Pool shape cannot be bigger than padded input shape");

    throw_if(new_stride_shape[0] > new_input_shape[0] + 2 * new_padding_dimensions[0]
             || new_stride_shape[1] > new_input_shape[1] + 2 * new_padding_dimensions[1],
             "Stride shape cannot be bigger than padded input shape");

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

void Pooling::set_pool_size(Index new_pool_rows_number, Index new_pool_columns_number)
{
    pool_height = new_pool_rows_number;
    pool_width = new_pool_columns_number;

    update_pool_operator();
}

void Pooling::set_row_stride(Index new_row_stride)
{
    throw_if(new_row_stride <= 0, "Row stride must be positive.");

    row_stride = new_row_stride;

    update_pool_operator();
}

void Pooling::set_column_stride(Index new_column_stride)
{
    throw_if(new_column_stride <= 0, "Column stride must be positive.");

    column_stride = new_column_stride;

    update_pool_operator();
}

void Pooling::set_padding_height(Index new_padding_height)
{
    throw_if(new_padding_height < 0, "Padding height cannot be negative.");

    padding_height = new_padding_height;

    update_pool_operator();
}

void Pooling::set_padding_width(Index new_padding_width)
{
    throw_if(new_padding_width < 0, "Padding width cannot be negative.");

    padding_width = new_padding_width;

    update_pool_operator();
}

void Pooling::set_pooling_method(const string& new_pooling_method)
{
    pooling_method = string_to_pooling_method(new_pooling_method);

    update_pool_operator();
}

void Pooling::read_JSON_body(const Json* pooling_layer_element)
{
    pool_height     = read_json_index(pooling_layer_element, "PoolHeight");
    pool_width      = read_json_index(pooling_layer_element, "PoolWidth");

    row_stride      = read_json_index(pooling_layer_element, "RowStride");
    column_stride   = read_json_index(pooling_layer_element, "ColumnStride");

    padding_height  = read_json_index(pooling_layer_element, "PaddingHeight");
    padding_width   = read_json_index(pooling_layer_element, "PaddingWidth");

    pooling_method  = string_to_pooling_method(read_json_string(pooling_layer_element, "PoolingMethod"));

    update_pool_operator();
}

void Pooling::write_JSON_body(JsonWriter& printer) const
{
    write_json(printer, {
        {"PoolHeight", to_string(get_pool_height())},
        {"PoolWidth", to_string(get_pool_width())},
        {"PoolingMethod", pooling_method_to_string(pooling_method)},
        {"ColumnStride", to_string(get_column_stride())},
        {"RowStride", to_string(get_row_stride())},
        {"PaddingHeight", to_string(get_padding_height())},
        {"PaddingWidth", to_string(get_padding_width())}
    });
}

REGISTER(Layer, Pooling, "Pooling")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
