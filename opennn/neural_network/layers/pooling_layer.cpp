//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/layers/pooling_layer.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/core/cuda/kernel_pooling.cuh"
#endif
#include "opennn/registry.h"
#include "opennn/core/enum_map.h"

#include "opennn/core/tensor_operations.h"
#include "opennn/core/device_backend.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"

namespace opennn
{

namespace {

struct PoolWindow
{
    Index batch, channel, out_row, out_col;
    Index in_row_start, pr_start, pr_end;
    Index in_col_start, pc_start, pc_end;
};

template<typename Visit>
void for_each_pool_window(Index batch_size, Index input_channels,
                          Index input_height, Index input_width,
                          Index output_height, Index output_width,
                          Index pool_height, Index pool_width,
                          Index row_stride, Index column_stride,
                          Index padding_height, Index padding_width,
                          Visit&& visit)
{
    const Index slices_count = batch_size * input_channels;

    #pragma omp parallel for schedule(static)
    for (Index slice = 0; slice < slices_count; ++slice)
    {
        const Index b = slice / input_channels;
        const Index c = slice % input_channels;
        for (Index out_row = 0; out_row < output_height; ++out_row)
        {
            const Index in_row_start = out_row * row_stride - padding_height;
            const Index pr_start = max(Index(0), -in_row_start);
            const Index pr_end = min(pool_height, input_height - in_row_start);

            for (Index out_col = 0; out_col < output_width; ++out_col)
            {
                const Index in_col_start = out_col * column_stride - padding_width;
                const Index pc_start = max(Index(0), -in_col_start);
                const Index pc_end = min(pool_width, input_width - in_col_start);

                visit(PoolWindow{b, c, out_row, out_col,
                                 in_row_start, pr_start, pr_end,
                                 in_col_start, pc_start, pc_end});
            }
        }
    }
}

template<typename Visit>
void for_each_pool_element(const PoolWindow& window, Visit&& visit)
{
    for (Index pool_row = window.pr_start; pool_row < window.pr_end; ++pool_row)
        for (Index pool_column = window.pc_start; pool_column < window.pc_end; ++pool_column)
            visit(pool_row, pool_column);
}

}

void pooling_2d_forward(const TensorView& input, TensorView& output, TensorView& maximal_indices,
                        Index input_height, Index input_width, Index input_channels,
                        Index pool_height, Index pool_width,
                        Index row_stride, Index column_stride,
                        Index padding_height, Index padding_width,
                        bool max_pooling)
{
    const TensorMap4 inputs = input.as_tensor<4>();
    TensorMap4 outputs      = output.as_tensor<4>();

    const Index batch_size    = inputs.dimension(0);
    const Index output_height = outputs.dimension(1);
    const Index output_width  = outputs.dimension(2);

    const bool write_indices = max_pooling && !maximal_indices.empty();
    TensorMap4 indices_map = write_indices
                           ? maximal_indices.as_tensor<4>()
                           : TensorMap4(nullptr, 0, 0, 0, 0);

    const auto max_pool_window = [&](const PoolWindow& window) {
        float best = NEG_INFINITY;
        Index argmax = 0;

        for_each_pool_element(window, [&](Index pool_row, Index pool_column) {
            const float value = inputs(window.batch, window.in_row_start + pool_row,
                                       window.in_col_start + pool_column, window.channel);
            if (value > best)
            {
                best = value;
                argmax = pool_row * pool_width + pool_column;
            }
        });

        outputs(window.batch, window.out_row, window.out_col, window.channel) = best;
        if (write_indices)
            indices_map(window.batch, window.out_row, window.out_col, window.channel) = argmax;
    };

    const float inv_pool_size = 1.0f / (pool_height * pool_width);
    const auto average_pool_window = [&](const PoolWindow& window) {
        float sum = 0;
        for_each_pool_element(window, [&](Index pool_row, Index pool_column) {
            sum += inputs(window.batch, window.in_row_start + pool_row,
                          window.in_col_start + pool_column, window.channel);
        });
        outputs(window.batch, window.out_row, window.out_col, window.channel) = sum * inv_pool_size;
    };

    if (max_pooling)
        for_each_pool_window(batch_size, input_channels, input_height, input_width,
                             output_height, output_width, pool_height, pool_width,
                             row_stride, column_stride, padding_height, padding_width,
                             max_pool_window);
    else
        for_each_pool_window(batch_size, input_channels, input_height, input_width,
                             output_height, output_width, pool_height, pool_width,
                             row_stride, column_stride, padding_height, padding_width,
                             average_pool_window);
}

void pooling_2d_backward(const TensorView& output_delta, const TensorView& maximal_indices,
                         TensorView& input_delta,
                         Index input_height, Index input_width, Index input_channels,
                         Index pool_height, Index pool_width,
                         Index row_stride, Index column_stride,
                         Index padding_height, Index padding_width,
                         bool max_pooling)
{
    const TensorMap4 output_deltas = output_delta.as_tensor<4>();
    TensorMap4       input_deltas  = input_delta.as_tensor<4>().setZero();

    const Index batch_size    = output_deltas.dimension(0);
    const Index output_height = output_deltas.dimension(1);
    const Index output_width  = output_deltas.dimension(2);

    if (max_pooling)
    {
        const TensorMap4 max_indices = maximal_indices.as_tensor<4>();
        for_each_pool_window(batch_size, input_channels, input_height, input_width,
                             output_height, output_width, pool_height, pool_width,
                             row_stride, column_stride, padding_height, padding_width,
            [&](const PoolWindow& window) {
                const Index argmax = static_cast<Index>(max_indices(
                    window.batch, window.out_row, window.out_col, window.channel));
                const Index in_row = window.in_row_start + argmax / pool_width;
                const Index in_col = window.in_col_start + argmax % pool_width;

                if (in_row < 0 || in_row >= input_height || in_col < 0 || in_col >= input_width)
                    return;

                input_deltas(window.batch, in_row, in_col, window.channel)
                    += output_deltas(window.batch, window.out_row, window.out_col, window.channel);
            });
        return;
    }

    const float inv_pool_size = 1.0f / (pool_height * pool_width);
    for_each_pool_window(batch_size, input_channels, input_height, input_width,
                         output_height, output_width, pool_height, pool_width,
                         row_stride, column_stride, padding_height, padding_width,
        [&](const PoolWindow& window) {
            const float avg_delta = output_deltas(window.batch, window.out_row, window.out_col, window.channel) * inv_pool_size;
            for_each_pool_element(window, [&](Index pool_row, Index pool_column) {
                input_deltas(window.batch, window.in_row_start + pool_row,
                             window.in_col_start + pool_column, window.channel) += avg_delta;
            });
        });
}


void PoolOperator::set(Index input_h, Index input_w, Index input_c,
               Index pool_h, Index pool_w,
               Index new_row_stride, Index new_column_stride,
               Index padding_h, Index padding_w,
               Method new_method)
{
    input_height    = input_h;
    input_width     = input_w;
    input_channels  = input_c;
    pool_height     = pool_h;
    pool_width      = pool_w;
    row_stride      = new_row_stride;
    column_stride   = new_column_stride;
    padding_height  = padding_h;
    padding_width   = padding_w;
    method          = new_method;

#ifdef OPENNN_HAS_CUDA
    CudnnDescriptor<cudnnPoolingDescriptor_t> descriptor;
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&descriptor.handle));
    descriptor.deleter = &cudnnDestroyPoolingDescriptor;
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(
        descriptor,
        method == Max ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        CUDNN_PROPAGATE_NAN,
        to_int(pool_height), to_int(pool_width),
        to_int(padding_height), to_int(padding_width),
        to_int(row_stride), to_int(column_stride)));
    pooling_descriptor = std::move(descriptor);
#endif
}

Index PoolOperator::get_output_height() const noexcept
{
    return (input_height - pool_height + 2 * padding_height) / row_stride + 1;
}

Index PoolOperator::get_output_width() const noexcept
{
    return (input_width - pool_width + 2 * padding_width) / column_stride + 1;
}

#ifdef OPENNN_HAS_CUDA

cudnnPoolingDescriptor_t PoolOperator::get_pooling_descriptor() const
{
    throw_if(!pooling_descriptor,
             "PoolOperator: pooling descriptor requested before set().");
    return pooling_descriptor;
}

#endif

#ifdef OPENNN_HAS_CUDA

MaxPoolGeometry PoolOperator::max_pool_geometry(const TensorView& input) const noexcept
{
    return {input.size() / (input_height * input_width * input_channels),
            to_int(input_height), to_int(input_width), to_int(input_channels),
            to_int(get_output_height()), to_int(get_output_width()),
            to_int(pool_height), to_int(pool_width),
            to_int(row_stride), to_int(column_stride),
            to_int(padding_height), to_int(padding_width)};
}

bool PoolOperator::own_max_pooling(const TensorView& input, const TensorView& mask) const noexcept
{
    if (method != Max || pool_height * pool_width > 255) return false;
    if (!input.is_fp32() && !input.is_bf16()) return false;
    switch (device::rung<device::MaxPoolingRung>())
    {
    case device::MaxPoolingRung::Cudnn:     return false;
    case device::MaxPoolingRung::OwnKernel: return true;
    case device::MaxPoolingRung::Auto:      break;
    }
    return !mask.empty();
}

#endif

void PoolOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    auto& forward_slots = forward_propagation.slots[layer];
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& output      = get_output(forward_propagation, layer);

    TensorView empty_indices;
    TensorView& indices = slot_or(forward_slots, output_slots, 1, empty_indices);

#ifdef OPENNN_HAS_CUDA
    if (input.is_cuda())
    {
        if (own_max_pooling(input, indices))
        {
            input.dispatch([&]<typename T>()
            {
                max_pooling_forward_cuda<T>(input.as<T>(), output.as<T>(),
                                            indices.empty() ? nullptr : indices.as<uint8_t>(),
                                            max_pool_geometry(input));
            });
            return;
        }

        CHECK_CUDNN(cudnnPoolingForward(device::get_cudnn_handle(),
            get_pooling_descriptor(),
            &one,  input.get_descriptor(),  input.get_data(),
            &zero, output.get_descriptor(), output.get_data()));
        return;
    }
#endif

    pooling_2d_forward(input, output, indices,
                       input_height, input_width, input_channels,
                       pool_height, pool_width,
                       row_stride, column_stride,
                       padding_height, padding_width,
                       method == Max);
}

void PoolOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    auto& forward_slots = forward_propagation.slots[layer];

    const TensorView& output_delta = get_output_delta(back_propagation, layer);
    TensorView& input_delta        = get_input_delta(back_propagation, layer);
    if (input_delta.empty()) return;

    TensorView empty_indices;
    const TensorView& indices = slot_or(forward_slots, output_slots, 1, empty_indices);

#ifdef OPENNN_HAS_CUDA
    if (output_delta.is_cuda())
    {
        const TensorView& input  = get_input(forward_propagation, layer);
        const TensorView& output = get_output(forward_propagation, layer);

        // The forward left the argmax mask exactly when it ran the library kernel.
        if (own_max_pooling(input, indices) && !indices.empty())
        {
            output_delta.dispatch([&]<typename T>()
            {
                max_pooling_backward_cuda<T>(output_delta.as<T>(), indices.as<uint8_t>(), input_delta.as<T>(),
                                             max_pool_geometry(input));
            });
            return;
        }

        CHECK_CUDNN(cudnnPoolingBackward(device::get_cudnn_handle(),
            get_pooling_descriptor(),
            &one,  output.get_descriptor(),       output.get_data(),
                   output_delta.get_descriptor(), output_delta.get_data(),
                   input.get_descriptor(),        input.get_data(),
            &zero, input_delta.get_descriptor(),  input_delta.get_data()));
        return;
    }
#endif

    pooling_2d_backward(output_delta, indices, input_delta,
                        input_height, input_width, input_channels,
                        pool_height, pool_width,
                        row_stride, column_stride,
                        padding_height, padding_width,
                        method == Max);
}

namespace
{

const EnumMap<PoolingMethod>& pooling_method_map()
{
    static const EnumMap<PoolingMethod> map{
        {PoolingMethod::MaxPooling,     "MaxPooling"},
        {PoolingMethod::AveragePooling, "AveragePooling"},
        {PoolingMethod::FirstToken,     "FirstToken"}
    };
    return map;
}

void validate_pooling_configuration(const Shape& input_shape,
                                    const Shape& pool_shape,
                                    const Shape& stride_shape,
                                    const Shape& padding_shape,
                                    const string& label)
{
    throw_if(input_shape.get_rank() != 3,
             "Pooling layer '{}': input shape must have 3 dimensions, read {}.",
             label, input_shape.get_rank());
    throw_if(pool_shape.get_rank() != 2,
             "Pooling layer '{}': pool shape must have 2 dimensions, read {}.",
             label, pool_shape.get_rank());
    throw_if(stride_shape.get_rank() != 2,
             "Pooling layer '{}': stride must have 2 dimensions, read {}.",
             label, stride_shape.get_rank());
    throw_if(padding_shape.get_rank() != 2,
             "Pooling layer '{}': padding must have 2 dimensions, read {}.",
             label, padding_shape.get_rank());

    throw_if(pool_shape[0] <= 0 || pool_shape[1] <= 0,
             "Pooling layer '{}': pool size must be positive, read {}.",
             label, shape_to_string(pool_shape));
    throw_if(stride_shape[0] <= 0 || stride_shape[1] <= 0,
             "Pooling layer '{}': stride must be positive, read {}.",
             label, shape_to_string(stride_shape));
    throw_if(padding_shape[0] < 0 || padding_shape[1] < 0,
             "Pooling layer '{}': padding cannot be negative, read {}.",
             label, shape_to_string(padding_shape));
    // Padding >= pool size would let a max-pooling window sit entirely inside the
    // padding, producing -infinity outputs.
    throw_if(padding_shape[0] >= pool_shape[0] || padding_shape[1] >= pool_shape[1],
             "Pooling layer '{}': padding {} must be smaller than the pool size {}.",
             label, shape_to_string(padding_shape), shape_to_string(pool_shape));

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

    // The argmax of each output window, for the backward: on the CPU as a
    // float per output; on CUDA a byte per output (the window position, see
    // max_pooling_forward_cuda), so a window has to fit a byte.
    const bool max_pooling = pooling_method == PoolingMethod::MaxPooling;
    const bool cuda = compute_device == Device::CUDA;
    const bool argmax_saved = max_pooling && (!cuda || pool_height * pool_width <= 255);

    return {
        {argmax_saved ? Shape{batch_size}.append(out_shape) : Shape{}, cuda ? Type::INT8 : Type::FP32},
        {Shape{batch_size}.append(out_shape),                          compute_dtype},
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

void Pooling::apply_input_shape(const Shape& new_input_shape)
{
    throw_if(new_input_shape.get_rank() != 3, "Input shape must be 3");

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
