//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L 3 D   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/operators/pool3d_operator.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/core/cuda/kernel.cuh"
#endif

namespace opennn
{

// Defined below: against the CUDA kernels, or as throwing stubs.
static void max_pooling_3d_forward_gpu(const TensorView&, TensorView&, TensorView&, bool);
static void average_pooling_3d_forward_gpu(const TensorView&, TensorView&);
static void max_pooling_3d_backward_gpu(const TensorView&, const TensorView&, TensorView&);
static void average_pooling_3d_backward_gpu(const TensorView&, const TensorView&, TensorView&);
static void first_token_3d_forward_gpu(const TensorView&, TensorView&);
static void first_token_3d_backward_gpu(const TensorView&, TensorView&);

static void max_pooling_3d_forward_cpu(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training)
{
    const TensorMap3 inputs = input.as_tensor<3>();
    MatrixMap outputs = output.as_matrix();

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    MatrixMap max_indices = maximal_indices.as_matrix();

    #pragma omp parallel for schedule(static)
    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        outputs.row(batch_index).setConstant(NEG_INFINITY);

        for (Index step = 0; step < sequence_length; ++step)
        {
            const Map<const Array<float, 1, Dynamic>> step_features(&inputs(batch_index, step, 0), 1, features);
            const auto greater = (step_features > outputs.row(batch_index).array()).eval();
            if (is_training)
                max_indices.row(batch_index).array() = greater.select(to_type(step), max_indices.row(batch_index).array());
            outputs.row(batch_index).array() = greater.select(step_features, outputs.row(batch_index).array());
        }
    }
}

void max_pooling_3d_forward(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training)
{
    if (input.is_cuda()) { max_pooling_3d_forward_gpu(input, output, maximal_indices, is_training); return; }
    max_pooling_3d_forward_cpu(input, output, maximal_indices, is_training);
}

static void average_pooling_3d_forward_cpu(const TensorView& input, TensorView& output)
{
    const TensorMap3 inputs = input.as_tensor<3>();
    MatrixMap outputs = output.as_matrix();

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    #pragma omp parallel for schedule(static)
    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        const Map<const MatrixR> seq_matrix(&inputs(batch_index, 0, 0), sequence_length, features);

        const Index valid_count = ((seq_matrix.array() != 0.0f).rowwise().any()).count();

        if (valid_count == 0) { outputs.row(batch_index).setZero(); continue; }
        outputs.row(batch_index) = seq_matrix.colwise().sum() / to_type(valid_count);
    }
}

void average_pooling_3d_forward(const TensorView& input, TensorView& output)
{
    if (input.is_cuda()) { average_pooling_3d_forward_gpu(input, output); return; }
    average_pooling_3d_forward_cpu(input, output);
}

static void max_pooling_3d_backward_cpu(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta)
{
    const MatrixMap max_indices = maximal_indices.as_matrix();
    const MatrixMap output_delta_matrix = output_delta.as_matrix();
    TensorMap3 input_delta_map = input_delta.as_tensor<3>().setZero();

    const Index batch_size = output_delta_matrix.rows();
    const Index features = output_delta_matrix.cols();

    #pragma omp parallel for schedule(static)
    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for (Index feature_index = 0; feature_index < features; ++feature_index)
        {
            const Index step = static_cast<Index>(max_indices(batch_index, feature_index));
            input_delta_map(batch_index, step, feature_index) = output_delta_matrix(batch_index, feature_index);
        }
}

void max_pooling_3d_backward(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta)
{
    if (output_delta.is_cuda()) { max_pooling_3d_backward_gpu(maximal_indices, output_delta, input_delta); return; }
    max_pooling_3d_backward_cpu(maximal_indices, output_delta, input_delta);
}

static void average_pooling_3d_backward_cpu(const TensorView& input,
                                     const TensorView& output_delta,
                                     TensorView& input_delta)
{
    const TensorMap3 inputs = input.as_tensor<3>();
    const MatrixMap output_delta_matrix = output_delta.as_matrix();
    TensorMap3 input_delta_map = input_delta.as_tensor<3>().setZero();

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    #pragma omp parallel for schedule(static)
    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        const Map<const MatrixR> seq_matrix(&inputs(batch_index, 0, 0), sequence_length, features);
        const auto non_padding = (seq_matrix.array() != 0.0f).rowwise().any().eval();
        const Index valid_count = non_padding.count();

        if (valid_count == 0) continue;

        const float inverse_valid_count = 1.0f / to_type(valid_count);
        Map<MatrixR> gradient_matrix(&input_delta_map(batch_index, 0, 0), sequence_length, features);
        const auto output_row = output_delta_matrix.row(batch_index);

        for (Index step = 0; step < sequence_length; ++step)
            if (non_padding(step))
                gradient_matrix.row(step) = output_row * inverse_valid_count;
    }
}

void average_pooling_3d_backward(const TensorView& input,
                                 const TensorView& output_delta,
                                 TensorView& input_delta)
{
    if (output_delta.is_cuda()) { average_pooling_3d_backward_gpu(input, output_delta, input_delta); return; }
    average_pooling_3d_backward_cpu(input, output_delta, input_delta);
}

static void first_token_3d_forward_cpu(const TensorView& input, TensorView& output)
{
    const TensorMap3 inputs = input.as_tensor<3>();
    MatrixMap outputs = output.as_matrix();

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    const bool parallel = batch_size * features >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        const Map<const MatrixR> seq_matrix(&inputs(batch_index, 0, 0), sequence_length, features);
        outputs.row(batch_index) = seq_matrix.row(0);
    }
}

void first_token_3d_forward(const TensorView& input, TensorView& output)
{
    if (input.is_cuda()) { first_token_3d_forward_gpu(input, output); return; }
    first_token_3d_forward_cpu(input, output);
}

static void first_token_3d_backward_cpu(const TensorView& output_delta, TensorView& input_delta)
{
    const MatrixMap output_delta_matrix = output_delta.as_matrix();
    TensorMap3 input_delta_map = input_delta.as_tensor<3>().setZero();

    const Index batch_size = output_delta_matrix.rows();
    const Index sequence_length = input_delta_map.dimension(1);
    const Index features = output_delta_matrix.cols();

    const bool parallel = batch_size * features >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        Map<MatrixR> gradient_matrix(&input_delta_map(batch_index, 0, 0), sequence_length, features);
        gradient_matrix.row(0) = output_delta_matrix.row(batch_index);
    }
}

void first_token_3d_backward(const TensorView& output_delta, TensorView& input_delta)
{
    if (output_delta.is_cuda()) { first_token_3d_backward_gpu(output_delta, input_delta); return; }
    first_token_3d_backward_cpu(output_delta, input_delta);
}

#ifdef OPENNN_HAS_CUDA

static void max_pooling_3d_forward_gpu(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool  )
{
    output.dispatch([&]<typename T>() {
        max_pooling_3d_forward_cuda<T>(to_int(input.shape[0]) * to_int(input.shape[2]),
                                       input.as<T>(), output.as<T>(),
                                       maximal_indices.as<float>(),
                                       to_int(input.shape[1]),
                                       to_int(input.shape[2]));
    });
}

static void average_pooling_3d_forward_gpu(const TensorView& input, TensorView& output)
{
    output.dispatch([&]<typename T>() {
        average_pooling_3d_forward_cuda<T>(to_int(input.shape[0]) * to_int(input.shape[2]),
                                           input.as<T>(), output.as<T>(),
                                           to_int(input.shape[1]),
                                           to_int(input.shape[2]));
    });
}

static void max_pooling_3d_backward_gpu(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta)
{
    input_delta.dispatch([&]<typename T>() {
        input_delta.set_zero_async();
        max_pooling_3d_backward_cuda<T>(to_int(output_delta.shape[0]) * to_int(output_delta.shape[1]),
                                        output_delta.as<T>(), input_delta.as<T>(),
                                        maximal_indices.as<float>(),
                                        to_int(input_delta.shape[1]),
                                        to_int(output_delta.shape[1]));
    });
}

static void average_pooling_3d_backward_gpu(const TensorView& input,
                                     const TensorView& output_delta,
                                     TensorView& input_delta)
{
    input_delta.dispatch([&]<typename T>() {
        // No pre-zeroing: the kernel writes every element of input_delta.
        average_pooling_3d_backward_cuda<T>(to_int(input.shape[0]) * to_int(input.shape[2]),
                                            input.as<T>(), output_delta.as<T>(),
                                            input_delta.as<T>(),
                                            to_int(input.shape[1]),
                                            to_int(input.shape[2]));
    });
}

static void first_token_3d_forward_gpu(const TensorView& input, TensorView& output)
{
    output.dispatch([&]<typename T>() {
        first_token_3d_forward_cuda<T>(to_int(input.shape[0]), to_int(input.shape[1]), to_int(input.shape[2]),
                                       input.as<T>(), output.as<T>());
    });
}

static void first_token_3d_backward_gpu(const TensorView& output_delta, TensorView& input_delta)
{
    input_delta.dispatch([&]<typename T>() {
        input_delta.set_zero_async();
        first_token_3d_backward_cuda<T>(to_int(input_delta.shape[0]), to_int(input_delta.shape[1]), to_int(input_delta.shape[2]),
                                        output_delta.as<T>(), input_delta.as<T>());
    });
}

#else

static void max_pooling_3d_forward_gpu(const TensorView&, TensorView&, TensorView&, bool) { throw runtime_error("max_pooling_3d_forward_gpu: CUDA support not compiled in."); }
static void average_pooling_3d_forward_gpu(const TensorView&, TensorView&) { throw runtime_error("average_pooling_3d_forward_gpu: CUDA support not compiled in."); }
static void max_pooling_3d_backward_gpu(const TensorView&, const TensorView&, TensorView&) { throw runtime_error("max_pooling_3d_backward_gpu: CUDA support not compiled in."); }
static void average_pooling_3d_backward_gpu(const TensorView&, const TensorView&, TensorView&) { throw runtime_error("average_pooling_3d_backward_gpu: CUDA support not compiled in."); }
static void first_token_3d_forward_gpu(const TensorView&, TensorView&) { throw runtime_error("first_token_3d_forward_gpu: CUDA support not compiled in."); }
static void first_token_3d_backward_gpu(const TensorView&, TensorView&) { throw runtime_error("first_token_3d_backward_gpu: CUDA support not compiled in."); }

#endif


void Pool3dOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training)
{
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& output      = get_output(forward_propagation, layer);
    TensorView& indices     = get_output(forward_propagation, layer, 1);

    if (method == Max)
        max_pooling_3d_forward(input, output, indices, is_training);
    else if (method == First)
        first_token_3d_forward(input, output);
    else
        average_pooling_3d_forward(input, output);
}

void Pool3dOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    const TensorView& output_delta = get_output_delta(back_propagation, layer);
    TensorView& input_delta        = get_input_delta(back_propagation, layer);
    if (input_delta.empty()) return;

    if (method == Max)
        max_pooling_3d_backward(get_output(forward_propagation, layer, 1), output_delta, input_delta);
    else if (method == First)
        first_token_3d_backward(output_delta, input_delta);
    else
        average_pooling_3d_backward(get_input(forward_propagation, layer), output_delta, input_delta);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
