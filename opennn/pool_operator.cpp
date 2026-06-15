//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pool_operator.h"
#include "device_backend.h"
#include "json.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

namespace
{

void configure_pooling_descriptor(PoolOp& op)
{
    if (op.pool_height <= 0 || op.pool_width <= 0) return;

    if (!op.pooling_descriptor)
    {
        CHECK_CUDNN(cudnnCreatePoolingDescriptor(&op.pooling_descriptor.handle));
        op.pooling_descriptor.deleter = &cudnnDestroyPoolingDescriptor;
    }

    const cudnnPoolingMode_t mode = (op.method == PoolOp::Max)
        ? CUDNN_POOLING_MAX
        : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

    CHECK_CUDNN(cudnnSetPooling2dDescriptor(op.pooling_descriptor,
                                            mode,
                                            CUDNN_PROPAGATE_NAN,
                                            to_int(op.pool_height), to_int(op.pool_width),
                                            to_int(op.padding_height), to_int(op.padding_width),
                                            to_int(op.row_stride), to_int(op.column_stride)));
}

}

#endif


void PoolOp::set(Index input_h, Index input_w, Index input_c,
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
    configure_pooling_descriptor(*this);
#endif
}

void PoolOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training)
{
    auto& forward_slots = fp.forward_slots[layer];
    const TensorView& input = get_input(fp, layer);
    TensorView& output      = get_output(fp, layer);

    TensorView empty;
    TensorView& indices = view_at_slot_or(forward_slots, output_slots, 1, empty);

    if (input.is_cuda())
    {
        apply_gpu(input, output);
        return;
    }
    apply_cpu(input, output, indices, is_training);
}

void PoolOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const
{
    auto& forward_slots = fp.forward_slots[layer];

    const TensorView& output_delta = get_output_delta(bp, layer);
    TensorView& input_delta        = get_input_delta(bp, layer);
    if (input_delta.empty()) return;

    TensorView empty;
    const TensorView& indices = view_at_slot_or(forward_slots, output_slots, 1, empty);

    if (output_delta.is_cuda())
    {
        const TensorView& input = get_input(fp, layer);
        const TensorView& output = get_output(fp, layer);
        apply_delta_gpu(input, output, output_delta, input_delta);
        return;
    }

    apply_delta_cpu(output_delta, indices, input_delta);
}

namespace {

struct PoolWindow
{
    Index batch, channel, out_row, out_col;
    Index in_row_start, pr_start, pr_end;
    Index in_col_start, pc_start, pc_end;
};

template<typename Visit>
void for_each_pool_window(const PoolOp& p,
                           Index batch_size, Index output_height, Index output_width,
                           Visit&& visit)
{
    #pragma omp parallel for collapse(2)
    for (Index b = 0; b < batch_size; ++b)
        for (Index c = 0; c < p.input_channels; ++c)
            for (Index out_row = 0; out_row < output_height; ++out_row)
            {
                const Index in_row_start = out_row * p.row_stride - p.padding_height;
                const Index pr_start = max(Index(0), -in_row_start);
                const Index pr_end   = min(p.pool_height, p.input_height - in_row_start);

                for (Index out_col = 0; out_col < output_width; ++out_col)
                {
                    const Index in_col_start = out_col * p.column_stride - p.padding_width;
                    const Index pc_start = max(Index(0), -in_col_start);
                    const Index pc_end   = min(p.pool_width, p.input_width - in_col_start);

                    visit(PoolWindow{b, c, out_row, out_col,
                                     in_row_start, pr_start, pr_end,
                                     in_col_start, pc_start, pc_end});
                }
            }
}

}

void PoolOp::apply_cpu(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training)
{
    const TensorMap4 inputs = input.as_tensor<4>();
    TensorMap4 outputs      = output.as_tensor<4>();

    const Index batch_size    = inputs.dimension(0);
    const Index output_height = outputs.dimension(1);
    const Index output_width  = outputs.dimension(2);

    if (method == Max)
    {
        const bool write_indices = !maximal_indices.empty();
        TensorMap4 indices_map = write_indices ? maximal_indices.as_tensor<4>() : TensorMap4(nullptr, 0, 0, 0, 0);
        for_each_pool_window(*this, batch_size, output_height, output_width,
            [&](const PoolWindow& window) {
                float best = NEG_INFINITY;
                Index argmax = 0;
                for (Index pr = window.pr_start; pr < window.pr_end; ++pr)
                    for (Index pc = window.pc_start; pc < window.pc_end; ++pc)
                    {
                        const float value = inputs(window.batch, window.in_row_start + pr,
                                                window.in_col_start + pc, window.channel);
                        if (value > best) { best = value; argmax = pr * pool_width + pc; }
                    }
                outputs(window.batch, window.out_row, window.out_col, window.channel) = best;
                if (write_indices)
                    indices_map(window.batch, window.out_row, window.out_col, window.channel) = argmax;
            });
        return;
    }

    const float inv_pool_size = 1.0f / (pool_height * pool_width);
    for_each_pool_window(*this, batch_size, output_height, output_width,
        [&](const PoolWindow& window) {
            float sum = 0;
            for (Index pr = window.pr_start; pr < window.pr_end; ++pr)
                for (Index pc = window.pc_start; pc < window.pc_end; ++pc)
                    sum += inputs(window.batch, window.in_row_start + pr,
                                  window.in_col_start + pc, window.channel);
            outputs(window.batch, window.out_row, window.out_col, window.channel) = sum * inv_pool_size;
        });
}

void PoolOp::apply_delta_cpu(const TensorView& output_delta,
                           const TensorView& maximal_indices,
                           TensorView& input_delta) const
{
    const TensorMap4 output_deltas = output_delta.as_tensor<4>();
    TensorMap4       input_deltas  = input_delta.as_tensor<4>().setZero();

    const Index batch_size    = output_deltas.dimension(0);
    const Index output_height = output_deltas.dimension(1);
    const Index output_width  = output_deltas.dimension(2);

    if (method == Max)
    {
        const TensorMap4 max_indices = maximal_indices.as_tensor<4>();

        #pragma omp parallel for collapse(2)
        for (Index b = 0; b < batch_size; ++b)
            for (Index c = 0; c < input_channels; ++c)
                for (Index out_row = 0; out_row < output_height; ++out_row)
                {
                    const Index in_row_start = out_row * row_stride - padding_height;
                    for (Index out_col = 0; out_col < output_width; ++out_col)
                    {
                        const Index in_col_start = out_col * column_stride - padding_width;
                        const Index argmax = static_cast<Index>(max_indices(b, out_row, out_col, c));
                        const Index pr = argmax / pool_width;
                        const Index pc = argmax % pool_width;
                        input_deltas(b, in_row_start + pr, in_col_start + pc, c)
                            += output_deltas(b, out_row, out_col, c);
                    }
                }
        return;
    }

    const float inv_pool_size = 1.0f / (pool_height * pool_width);
    for_each_pool_window(*this, batch_size, output_height, output_width,
        [&](const PoolWindow& window) {
            const float avg_delta = output_deltas(window.batch, window.out_row, window.out_col, window.channel) * inv_pool_size;
            for (Index pr = window.pr_start; pr < window.pr_end; ++pr)
                for (Index pc = window.pc_start; pc < window.pc_end; ++pc)
                    input_deltas(window.batch, window.in_row_start + pr,
                                 window.in_col_start + pc, window.channel) += avg_delta;
        });
}

#ifdef OPENNN_HAS_CUDA

void PoolOp::apply_gpu(const TensorView& input, TensorView& output)
{
    CHECK_CUDNN(cudnnPoolingForward(Backend::get_cudnn_handle(),
        pooling_descriptor,
        &one,
        input.get_descriptor(), input.data,
        &zero,
        output.get_descriptor(), output.data));
}

void PoolOp::apply_delta_gpu(const TensorView& input,
                           const TensorView& output,
                           const TensorView& output_delta,
                           TensorView& input_delta) const
{
    CHECK_CUDNN(cudnnPoolingBackward(Backend::get_cudnn_handle(),
        pooling_descriptor,
        &one,
        output.get_descriptor(),       output.data,
        output_delta.get_descriptor(), output_delta.data,
        input.get_descriptor(),        input.data,
        &zero,
        input_delta.get_descriptor(),  input_delta.data));
}

#else

void PoolOp::apply_gpu(const TensorView&, TensorView&)                                                { throw runtime_error("Pool::apply_gpu: CUDA support not compiled in."); }
void PoolOp::apply_delta_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&) const { throw runtime_error("Pool::apply_delta_gpu: CUDA support not compiled in."); }

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
