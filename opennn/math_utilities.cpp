//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M A T H   U T I L I T I E S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "math_utilities.h"
#include "random_utilities.h"
#include "cuda_dispatch.h"
#include "cuda_gemm.h"
#include "profiler.h"

namespace opennn
{

void pad(const TensorView& input, TensorView& output)
{
    if (is_gpu())
        throw runtime_error("pad: GPU implementation not available.");

    const TensorMap4 input_map = input.as_tensor<4>();
    TensorMap4 output_map = output.as_tensor<4>();

    const Index padding_height = (output.shape[1] - input.shape[1]) / 2;
    const Index padding_width = (output.shape[2] - input.shape[2]) / 2;

    const array<pair<Index,Index>, 4> paddings = {
        make_pair(Index(0), Index(0)),
        make_pair(padding_height, padding_height),
        make_pair(padding_width, padding_width),
        make_pair(Index(0), Index(0))
    };

    output_map.device(get_device()) = input_map.pad(paddings);
}

void bound_cpu(const TensorView& input,
               const TensorView& lower_bounds,
               const TensorView& upper_bounds,
               TensorView& output)
{
    const Index features = lower_bounds.size();

    const MatrixMap input_matrix = input.as_matrix();
    const VectorMap lower_bounds_vector = lower_bounds.as_vector();
    const VectorMap upper_bounds_vector = upper_bounds.as_vector();

    MatrixMap output_matrix = output.as_matrix();

    for (Index feature_index = 0; feature_index < features; ++feature_index)
        output_matrix.col(feature_index) = input_matrix.col(feature_index)
                                                        .cwiseMax(lower_bounds_vector(feature_index))
                                                        .cwiseMin(upper_bounds_vector(feature_index));
}

void bound(const TensorView& input,
           const TensorView& lower_bounds,
           const TensorView& upper_bounds,
           TensorView& output)
{
    IF_GPU({ bound_gpu(input, lower_bounds, upper_bounds, output); return; });
    bound_cpu(input, lower_bounds, upper_bounds, output);
}

void scale_cpu(const TensorView& input,
               const TensorView& minimums, const TensorView& maximums,
               const TensorView& means, const TensorView& standard_deviations,
               const TensorView& scalers,
               float min_range, float max_range,
               TensorView& output)
{
    const Index features = scalers.size();

    const MatrixMap input_matrix = input.as_matrix();
    const VectorMap minimums_vector = minimums.as_vector();
    const VectorMap maximums_vector = maximums.as_vector();
    const VectorMap means_vector  = means.as_vector();
    const VectorMap standard_deviations_vector  = standard_deviations.as_vector();
    const VectorMap scalers_vector   = scalers.as_vector();

    MatrixMap output_matrix = output.as_matrix();

    output_matrix.noalias() = input_matrix;

    for (Index feature_index = 0; feature_index < features; ++feature_index)
    {
        const int code = static_cast<int>(scalers_vector(feature_index));
        auto column = output_matrix.col(feature_index).array();

        switch (code)
        {
        case 1: // MinimumMaximum
            column = (column - minimums_vector(feature_index)) / ((maximums_vector(feature_index) - minimums_vector(feature_index)) + EPSILON)
                   * (max_range - min_range) + min_range;
            break;
        case 2: // MeanStandardDeviation
            column = (column - means_vector(feature_index)) / (standard_deviations_vector(feature_index) + EPSILON);
            break;
        case 3: // StandardDeviation
            column /= (standard_deviations_vector(feature_index) + EPSILON);
            break;
        case 4: // Logarithm
            column = column.log();
            break;
        case 5: // ImageMinMax
            column /= 255.0f;
            break;
        default: // None
            break;
        }
    }
}

void scale(const TensorView& input,
           const TensorView& minimums, const TensorView& maximums,
           const TensorView& means, const TensorView& standard_deviations,
           const TensorView& scalers,
           float min_range, float max_range,
           TensorView& output)
{
    IF_GPU({
        scale_gpu(input, minimums, maximums, means, standard_deviations, scalers,
                  min_range, max_range, output);
        return;
    });
    scale_cpu(input, minimums, maximums, means, standard_deviations, scalers,
              min_range, max_range, output);
}

void unscale_cpu(const TensorView& input,
                 const TensorView& minimums, const TensorView& maximums,
                 const TensorView& means, const TensorView& standard_deviations,
                 const TensorView& scalers,
                 float min_range, float max_range,
                 TensorView& output)
{
    const Index features = scalers.size();

    const MatrixMap input_matrix = input.as_matrix();
    const VectorMap minimums_vector = minimums.as_vector();
    const VectorMap maximums_vector = maximums.as_vector();
    const VectorMap means_vector  = means.as_vector();
    const VectorMap standard_deviations_vector  = standard_deviations.as_vector();
    const VectorMap scalers_vector   = scalers.as_vector();

    MatrixMap output_matrix = output.as_matrix();

    output_matrix.noalias() = input_matrix;

    for (Index feature_index = 0; feature_index < features; ++feature_index)
    {
        const int code = static_cast<int>(scalers_vector(feature_index));
        auto column = output_matrix.col(feature_index).array();

        switch (code)
        {
        case 1: // MinimumMaximum
            column = (column - min_range) / (max_range - min_range)
                   * (maximums_vector(feature_index) - minimums_vector(feature_index)) + minimums_vector(feature_index);
            break;
        case 2: // MeanStandardDeviation
            column = means_vector(feature_index) + column * standard_deviations_vector(feature_index);
            break;
        case 3: // StandardDeviation
            column *= standard_deviations_vector(feature_index);
            break;
        case 4: // Logarithm
            column = column.exp();
            break;
        case 5: // ImageMinMax
            column *= 255.0f;
            break;
        default: // None
            break;
        }
    }
}

void unscale(const TensorView& input,
             const TensorView& minimums, const TensorView& maximums,
             const TensorView& means, const TensorView& standard_deviations,
             const TensorView& scalers,
             float min_range, float max_range,
             TensorView& output)
{
    IF_GPU({
        unscale_gpu(input, minimums, maximums, means, standard_deviations, scalers,
                    min_range, max_range, output);
        return;
    });
    unscale_cpu(input, minimums, maximums, means, standard_deviations, scalers,
                min_range, max_range, output);
}

void copy_cpu(const TensorView& source, TensorView& destination)
{
    memcpy(destination.data, source.data, source.size() * sizeof(float));
}

void copy(const TensorView& source, TensorView& destination)
{
    if (source.size() != destination.size())
        throw runtime_error("Tensor sizes mismatch in copy operation.");

    IF_GPU({ copy_gpu(source, destination); return; });
    copy_cpu(source, destination);
}

void add_cpu(const TensorView& input_1,
             const TensorView& input_2,
             TensorView& output)
{
    output.as_vector().noalias() = input_1.as_vector() + input_2.as_vector();
}

void add(const TensorView& input_1,
         const TensorView& input_2,
         TensorView& output)
{
    if (input_1.size() != input_2.size() || input_1.size() != output.size())
        throw runtime_error("Tensor dimensions do not match.");

    IF_GPU({ add_gpu(input_1, input_2, output); return; });
    add_cpu(input_1, input_2, output);
}

void multiply_cpu(const TensorView& input_a, bool transpose_a,
                  const TensorView& input_b, bool transpose_b,
                  TensorView& output,
                  float alpha, float beta)
{
    const size_t rank = input_a.get_rank();
    const Index batch_count = input_a.size() / (input_a.shape[rank - 2] * input_a.shape[rank - 1]);

    #pragma omp parallel for
    for (Index batch_index = 0; batch_index < batch_count; ++batch_index)
    {
        const MatrixMap matrix_a = input_a.as_matrix(batch_index);
        const MatrixMap matrix_b = input_b.as_matrix(batch_index);
        MatrixMap matrix_output = output.as_matrix(batch_index);

        auto gemm_like = [&](auto A, auto B)
        {
            if (beta == 0.0f)
                matrix_output.noalias() = alpha * (A * B);
            else
                matrix_output.noalias() = alpha * (A * B) + beta * matrix_output;
        };

        if (!transpose_a && !transpose_b)       gemm_like(matrix_a,             matrix_b);
        else if (transpose_a && !transpose_b)   gemm_like(matrix_a.transpose(), matrix_b);
        else if (!transpose_a && transpose_b)   gemm_like(matrix_a,             matrix_b.transpose());
        else                                    gemm_like(matrix_a.transpose(), matrix_b.transpose());
    }
}

void multiply(const TensorView& input_a, bool transpose_a,
              const TensorView& input_b, bool transpose_b,
              TensorView& output,
              float alpha, float beta)
{
    IF_GPU({ multiply_gpu(input_a, transpose_a, input_b, transpose_b, output, alpha, beta); return; });
    multiply_cpu(input_a, transpose_a, input_b, transpose_b, output, alpha, beta);
}

void softmax_cpu(TensorView& output)
{
    MatrixMap output_matrix = output.as_flat_matrix();
    const Index rows = output_matrix.rows();

    #pragma omp parallel for
    for (Index i = 0; i < rows; ++i)
    {
        const float max_val = output_matrix.row(i).maxCoeff();
        output_matrix.row(i).array() = (output_matrix.row(i).array() - max_val).exp();
        output_matrix.row(i) /= output_matrix.row(i).sum();
    }
}

void softmax(TensorView& output)
{
    if (output.empty()) return;

    IF_GPU({ softmax_gpu(output); return; });
    softmax_cpu(output);
}

void max_pooling_3d_forward_cpu(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training)
{
    const TensorMap3 inputs = input.as_tensor<3>();
    MatrixMap outputs = output.as_matrix();

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    MatrixMap max_indices = maximal_indices.as_matrix();

    #pragma omp parallel for
    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        outputs.row(batch_index).setConstant(NEG_INFINITY);

        for (Index step = 0; step < sequence_length; ++step)
            for (Index feature_index = 0; feature_index < features; ++feature_index)
            {
                const float value = inputs(batch_index, step, feature_index);
                if (value > outputs(batch_index, feature_index))
                {
                    outputs(batch_index, feature_index) = value;
                    if (is_training) max_indices(batch_index, feature_index) = to_type(step);
                }
            }
    }
}

void max_pooling_3d_forward(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training)
{
    IF_GPU({ max_pooling_3d_forward_gpu(input, output, maximal_indices, is_training); return; });
    max_pooling_3d_forward_cpu(input, output, maximal_indices, is_training);
}

void average_pooling_3d_forward_cpu(const TensorView& input, TensorView& output)
{
    const TensorMap3 inputs = input.as_tensor<3>();
    MatrixMap outputs = output.as_matrix();

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    #pragma omp parallel for
    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        const Map<const MatrixR> seq_matrix(&inputs(batch_index, 0, 0), sequence_length, features);

        const Index valid_count = ((seq_matrix.array() != 0.0f).rowwise().any()).count();

        if (valid_count > 0)
            outputs.row(batch_index) = seq_matrix.colwise().sum() / to_type(valid_count);
        else
            outputs.row(batch_index).setZero();
    }
}

void average_pooling_3d_forward(const TensorView& input, TensorView& output)
{
    IF_GPU({ average_pooling_3d_forward_gpu(input, output); return; });
    average_pooling_3d_forward_cpu(input, output);
}

void max_pooling_3d_backward_cpu(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta)
{
    const MatrixMap max_indices = maximal_indices.as_matrix();
    const MatrixMap output_delta_matrix = output_delta.as_matrix();
    TensorMap3 input_delta_map = input_delta.as_tensor<3>().setZero();

    const Index batch_size = output_delta_matrix.rows();
    const Index features = output_delta_matrix.cols();

    #pragma omp parallel for
    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for (Index feature_index = 0; feature_index < features; ++feature_index)
        {
            const Index step = static_cast<Index>(max_indices(batch_index, feature_index));
            input_delta_map(batch_index, step, feature_index) = output_delta_matrix(batch_index, feature_index);
        }
}

void max_pooling_3d_backward(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta)
{
    IF_GPU({ max_pooling_3d_backward_gpu(maximal_indices, output_delta, input_delta); return; });
    max_pooling_3d_backward_cpu(maximal_indices, output_delta, input_delta);
}

void average_pooling_3d_backward_cpu(const TensorView& input,
                                     const TensorView& output_delta,
                                     TensorView& input_delta)
{
    const TensorMap3 inputs = input.as_tensor<3>();
    const MatrixMap output_delta_matrix = output_delta.as_matrix();
    TensorMap3 input_delta_map = input_delta.as_tensor<3>().setZero();

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index features = inputs.dimension(2);

    #pragma omp parallel for
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
    IF_GPU({ average_pooling_3d_backward_gpu(input, output_delta, input_delta); return; });
    average_pooling_3d_backward_cpu(input, output_delta, input_delta);
}

static void transpose_middle_axes(const float* src, float* dst,
                                  Index batch_size, Index src_m1, Index src_m2, Index D)
{
    #pragma omp parallel for collapse(3)
    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for (Index i = 0; i < src_m2; ++i)
            for (Index j = 0; j < src_m1; ++j)
                memcpy(dst + ((batch_index * src_m2 + i) * src_m1 + j) * D,
                       src + ((batch_index * src_m1 + j) * src_m2 + i) * D,
                       D * sizeof(float));
}

void split_heads_cpu(const TensorView& source, TensorView& destination)
{
    const Index batch_size = source.shape[0];
    const Index sequence_length = source.shape[1];
    const Index heads_number = source.shape[2];
    const Index head_dimension = source.shape[3];

    transpose_middle_axes(source.as<float>(), destination.as<float>(),
                          batch_size, sequence_length, heads_number, head_dimension);
}

void split_heads(const TensorView& source, TensorView& destination)
{
    IF_GPU({ split_heads_gpu(source, destination); return; });
    split_heads_cpu(source, destination);
}

void merge_heads_cpu(const TensorView& source, TensorView& destination)
{
    const Index batch_size = source.shape[0];
    const Index heads_number = source.shape[1];
    const Index sequence_length = source.shape[2];
    const Index head_dimension = source.shape[3];

    transpose_middle_axes(source.as<float>(), destination.as<float>(),
                          batch_size, heads_number, sequence_length, head_dimension);
}

void merge_heads(const TensorView& source, TensorView& destination)
{
    IF_GPU({ merge_heads_gpu(source, destination); return; });
    merge_heads_cpu(source, destination);
}

#ifdef OPENNN_HAS_CUDA

void bound_gpu(const TensorView& input,
               const TensorView& lower_bounds,
               const TensorView& upper_bounds,
               TensorView& output)
{
    const Index features = lower_bounds.size();

    visit_type_pair<Type::FP32, Type::BF16>(input.type, output.type, [&](auto in, auto out) {
        using TIn  = typename decltype(in)::type;
        using TOut = typename decltype(out)::type;
        bounding_cuda<TIn, TOut>(output.size(), to_int(features),
                                 input.as<TIn>(),
                                 lower_bounds.as_float(),
                                 upper_bounds.as_float(),
                                 output.as<TOut>());
    });
}

void scale_gpu(const TensorView& input,
               const TensorView& minimums, const TensorView& maximums,
               const TensorView& means, const TensorView& standard_deviations,
               const TensorView& scalers,
               float min_range, float max_range,
               TensorView& output)
{
    const Index features = scalers.size();

    visit_type_pair<Type::FP32, Type::BF16>(input.type, output.type, [&](auto in, auto out) {
        using TIn  = typename decltype(in)::type;
        using TOut = typename decltype(out)::type;
        scale_cuda<TIn, TOut>(output.size(), to_int(features),
                              input.as<TIn>(),
                              minimums.as_float(),
                              maximums.as_float(),
                              means.as_float(),
                              standard_deviations.as_float(),
                              scalers.as_float(),
                              min_range, max_range,
                              output.as<TOut>());
    });
}

void unscale_gpu(const TensorView& input,
                 const TensorView& minimums, const TensorView& maximums,
                 const TensorView& means, const TensorView& standard_deviations,
                 const TensorView& scalers,
                 float min_range, float max_range,
                 TensorView& output)
{
    const Index features = scalers.size();

    visit_type_pair<Type::FP32, Type::BF16>(input.type, output.type, [&](auto in, auto out) {
        using TIn  = typename decltype(in)::type;
        using TOut = typename decltype(out)::type;
        unscale_cuda<TIn, TOut>(output.size(), to_int(features),
                                input.as<TIn>(),
                                minimums.as_float(),
                                maximums.as_float(),
                                means.as_float(),
                                standard_deviations.as_float(),
                                scalers.as_float(),
                                min_range, max_range,
                                output.as<TOut>());
    });
}

void copy_gpu(const TensorView& source, TensorView& destination)
{
    CHECK_CUDA(cudaMemcpyAsync(destination.data, source.data, source.byte_size(),
                               cudaMemcpyDeviceToDevice, Backend::get_compute_stream()));
}

void add_gpu(const TensorView& input_1,
             const TensorView& input_2,
             TensorView& output)
{
    CHECK_CUDNN(cudnnOpTensor(Backend::get_cudnn_handle(),
                              Backend::get_operator_sum_descriptor(),
                              &one, input_1.get_descriptor(), input_1.data,
                              &one, input_2.get_descriptor(), input_2.data,
                              &zero, output.get_descriptor(), output.data));
}

void multiply_gpu(const TensorView& input_a, bool transpose_a,
                  const TensorView& input_b, bool transpose_b,
                  TensorView& output,
                  float alpha, float beta)
{
    const size_t rank = input_a.get_rank();

    const int rows_a = to_int(input_a.shape[rank - 2]);
    const int cols_a = to_int(input_a.shape[rank - 1]);
    const int rows_b = to_int(input_b.shape[rank - 2]);
    const int cols_b = to_int(input_b.shape[rank - 1]);

    const int output_columns = transpose_b ? rows_b : cols_b;
    const int output_rows = transpose_a ? cols_a : rows_a;
    const int inner_dimension = transpose_a ? rows_a : cols_a;

    const cublasOperation_t operation_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t operation_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    const int batch_count = to_int(input_a.size() / (rows_a * cols_a));
    const long long stride_a = rows_a * cols_a;
    const long long stride_b = rows_b * cols_b;
    const long long stride_output = output.shape[rank - 2] * output.shape[rank - 1];

    if (batch_count == 1)
        gemm_cuda(operation_b, operation_a,
                  output_columns, output_rows, inner_dimension,
                  input_b.data, input_b.cuda_dtype(), cols_b,
                  input_a.data, input_a.cuda_dtype(), cols_a,
                  output.data,  output.cuda_dtype(),  output_columns,
                  alpha, beta);
    else
        gemm_strided_batched_cuda(operation_b, operation_a,
                                  output_columns, output_rows, inner_dimension,
                                  input_b.data, cols_b, stride_b,
                                  input_a.data, cols_a, stride_a,
                                  output.data,  output_columns, stride_output,
                                  batch_count,
                                  output.cuda_dtype(),
                                  alpha, beta);
}

void softmax_gpu(TensorView& output)
{
    CHECK_CUDNN(cudnnSoftmaxForward(Backend::get_cudnn_handle(),
                                    CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &one,
                                    output.get_descriptor(), output.data,
                                    &zero,
                                    output.get_descriptor(), output.data));
}

void max_pooling_3d_forward_gpu(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool /* is_training */)
{
    output.dispatch([&](auto tag) {
        using T = decltype(tag);
        max_pooling_3d_forward_cuda<T>(to_int(input.shape[0]) * to_int(input.shape[2]),
                                       input.as<T>(), output.as<T>(),
                                       maximal_indices.as<float>(),
                                       to_int(input.shape[1]),
                                       to_int(input.shape[2]));
    });
}

void average_pooling_3d_forward_gpu(const TensorView& input, TensorView& output)
{
    output.dispatch([&](auto tag) {
        using T = decltype(tag);
        average_pooling_3d_forward_cuda<T>(to_int(input.shape[0]) * to_int(input.shape[2]),
                                           input.as<T>(), output.as<T>(),
                                           to_int(input.shape[1]),
                                           to_int(input.shape[2]));
    });
}

void max_pooling_3d_backward_gpu(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta)
{
    input_delta.dispatch([&](auto tag) {
        using T = decltype(tag);
        input_delta.set_zero_async();
        max_pooling_3d_backward_cuda<T>(to_int(output_delta.shape[0]) * to_int(output_delta.shape[1]),
                                        output_delta.as<T>(), input_delta.as<T>(),
                                        maximal_indices.as<float>(),
                                        to_int(input_delta.shape[1]),
                                        to_int(output_delta.shape[1]));
    });
}

void average_pooling_3d_backward_gpu(const TensorView& input,
                                     const TensorView& output_delta,
                                     TensorView& input_delta)
{
    input_delta.dispatch([&](auto tag) {
        using T = decltype(tag);
        input_delta.set_zero_async();
        average_pooling_3d_backward_cuda<T>(to_int(input.shape[0]) * to_int(input.shape[2]),
                                            input.as<T>(), output_delta.as<T>(),
                                            input_delta.as<T>(),
                                            to_int(input.shape[1]),
                                            to_int(input.shape[2]));
    });
}

void split_heads_gpu(const TensorView& source, TensorView& destination)
{
    const Index sequence_length = source.shape[1];
    const Index heads_number = source.shape[2];
    const Index head_dimension = source.shape[3];

    destination.dispatch([&](auto tag) {
        using T = decltype(tag);
        split_heads_cuda<T>(source.size(), source.as<T>(), destination.as<T>(),
                            to_int(sequence_length),
                            to_int(heads_number),
                            to_int(head_dimension));
    });
}

void merge_heads_gpu(const TensorView& source, TensorView& destination)
{
    const Index heads_number = source.shape[1];
    const Index sequence_length = source.shape[2];
    const Index head_dimension = source.shape[3];

    destination.dispatch([&](auto tag) {
        using T = decltype(tag);
        merge_heads_cuda<T>(source.size(), source.as<T>(), destination.as<T>(),
                            to_int(sequence_length),
                            to_int(heads_number),
                            to_int(head_dimension));
    });
}

#endif // OPENNN_HAS_CUDA

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
