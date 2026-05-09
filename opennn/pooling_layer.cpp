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
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

Pooling::Pooling(const Shape& new_input_shape,
                 const Shape& new_pool_dimensions,
                 const Shape& new_stride_shape,
                 const Shape& new_padding_dimensions,
                 const string& new_pooling_method,
                 const string& new_name)
    : Layer("Pooling", LayerType::Pooling)
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

vector<pair<Shape, Type>> Pooling::get_forward_specs(Index batch_size) const
{
    const Shape out_shape = get_output_shape();

    vector<pair<Shape, Type>> specs;

    // MaximalIndices stores argmax positions (used by CPU backward and the
    // 3D max-pool kernels). They are read as float*, never as compute_dtype.
    if (pooling_method == PoolingMethod::MaxPooling)
        specs.push_back({Shape{batch_size}.append(out_shape), Type::FP32}); // MaximalIndices

    specs.push_back({Shape{batch_size}.append(out_shape), compute_dtype}); // Output (must be last)

    return specs;
}

void Pooling::update_pool_operator()
{
    pool.set(input_height, input_width, input_channels,
             pool_height, pool_width,
             row_stride, column_stride,
             padding_height, padding_width,
             pooling_method == PoolingMethod::MaxPooling ? 0 : 1);

    pool.input_slots = {Input};
    pool.output_slots = (pooling_method == PoolingMethod::MaxPooling)
        ? vector<size_t>{Output, MaximalIndices}
        : vector<size_t>{1};                       // {Output}; only 2 slots → Output is index 1

    pool.output_delta_slots = {OutputDelta};
    pool.input_delta_slots  = {InputDelta};
}
void Pooling::set(const Shape& new_input_shape,
                  const Shape& new_pool_dimensions,
                  const Shape& new_stride_shape,
                  const Shape& new_padding_dimensions,
                  const string& new_pooling_method,
                  const string& new_label)
{
    if (new_pool_dimensions.rank != 2)
        throw runtime_error("Pool shape must be 2");

    if (new_stride_shape.rank != 2)
        throw runtime_error("Stride shape must be 2");

    if (new_padding_dimensions.rank != 2)
        throw runtime_error("Padding shape must be 2");

    if (new_pool_dimensions[0] > new_input_shape[0] || new_pool_dimensions[1] > new_input_shape[1])
        throw runtime_error("Pool shape cannot be bigger than input shape");

    if (new_stride_shape[0] <= 0 || new_stride_shape[1] <= 0)
        throw runtime_error("Stride must be positive.");

    if (new_stride_shape[0] > new_input_shape[0] || new_stride_shape[1] > new_input_shape[1])
        throw runtime_error("Stride shape cannot be bigger than input shape");

    if (new_padding_dimensions[0] < 0 || new_padding_dimensions[1] < 0)
        throw runtime_error("Padding shape cannot be negative");

    // Direct assignment of all geometry; setters with side-effects are deferred
    // so we hit update_pool_operator() exactly once at the end.
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
    if (new_input_shape.rank != 3)
        throw runtime_error("Input shape must be 3");

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
    if (new_row_stride <= 0)
        throw runtime_error("Row stride must be positive.");

    row_stride = new_row_stride;

    update_pool_operator();
}

void Pooling::set_column_stride(Index new_column_stride)
{
    if (new_column_stride <= 0)
        throw runtime_error("Column stride must be positive.");

    column_stride = new_column_stride;

    update_pool_operator();
}

void Pooling::set_padding_height(Index new_padding_height)
{
    if (new_padding_height < 0)
        throw runtime_error("Padding height cannot be negative.");

    padding_height = new_padding_height;

    update_pool_operator();
}

void Pooling::set_padding_width(Index new_padding_width)
{
    if (new_padding_width < 0)
        throw runtime_error("Padding width cannot be negative.");

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
