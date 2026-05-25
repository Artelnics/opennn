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

#ifdef OPENNN_HAS_CUDA
static void bound_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&);
static void scale_gpu(const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, float, float, TensorView&);
static void unscale_gpu(const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, float, float, TensorView&);
static void copy_gpu(const TensorView&, TensorView&);
static void add_gpu(const TensorView&, const TensorView&, TensorView&);
static void multiply_gpu(const TensorView&, bool, const TensorView&, bool, TensorView&, float, float);
static void softmax_gpu(TensorView&);
static void activation_forward_gpu(TensorView&, ActivationFunction);
static void activation_backward_gpu(const TensorView&, TensorView&, ActivationFunction);
static void dropout_forward_gpu(TensorView&, Buffer&, float);
static void dropout_backward_gpu(TensorView&, const Buffer&, float);
static void linear_forward_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&, cublasLtEpilogue_t);
static void linear_backward_gpu(const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, TensorView&, bool);
static void layer_norm_forward_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&, TensorView&, TensorView&);
static void layer_norm_backward_gpu(const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, TensorView&);
static void embedding_lookup_forward_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&, Index, Index, Index, bool, bool);
static void embedding_lookup_backward_gpu(const TensorView&, const TensorView&, const TensorView&, Index, Index, bool);
static void max_pooling_3d_forward_gpu(const TensorView&, TensorView&, TensorView&, bool);
static void average_pooling_3d_forward_gpu(const TensorView&, TensorView&);
static void max_pooling_3d_backward_gpu(const TensorView&, const TensorView&, TensorView&);
static void average_pooling_3d_backward_gpu(const TensorView&, const TensorView&, TensorView&);
static void split_heads_gpu(const TensorView&, TensorView&);
static void merge_heads_gpu(const TensorView&, TensorView&);
#endif

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

static void bound_cpu(const TensorView& input,
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

static void scale_cpu(const TensorView& input,
               const TensorView& minimums, const TensorView& maximums,
               const TensorView& means, const TensorView& standard_deviations,
               const TensorView& scalers,
               float min_range, float max_range,
               TensorView& output)
{
    const Index features = scalers.size();
    if (features == 0) { output.as_matrix().noalias() = input.as_matrix(); return; }

    const MatrixMap input_matrix = input.as_flat_matrix();
    const VectorMap minimums_vector = minimums.as_vector();
    const VectorMap maximums_vector = maximums.as_vector();
    const VectorMap means_vector  = means.as_vector();
    const VectorMap standard_deviations_vector  = standard_deviations.as_vector();
    const VectorMap scalers_vector   = scalers.as_vector();

    MatrixMap output_matrix = output.as_flat_matrix();

    output_matrix.noalias() = input_matrix;

    const Index cols = output_matrix.cols();
    for (Index col = 0; col < cols; ++col)
    {
        const Index feature_index = col % features;
        const int code = static_cast<int>(scalers_vector(feature_index));
        auto column = output_matrix.col(col).array();

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

static void unscale_cpu(const TensorView& input,
                 const TensorView& minimums, const TensorView& maximums,
                 const TensorView& means, const TensorView& standard_deviations,
                 const TensorView& scalers,
                 float min_range, float max_range,
                 TensorView& output)
{
    const Index features = scalers.size();
    if (features == 0) { output.as_matrix().noalias() = input.as_matrix(); return; }

    const MatrixMap input_matrix = input.as_flat_matrix();
    const VectorMap minimums_vector = minimums.as_vector();
    const VectorMap maximums_vector = maximums.as_vector();
    const VectorMap means_vector  = means.as_vector();
    const VectorMap standard_deviations_vector  = standard_deviations.as_vector();
    const VectorMap scalers_vector   = scalers.as_vector();

    MatrixMap output_matrix = output.as_flat_matrix();

    output_matrix.noalias() = input_matrix;

    const Index cols = output_matrix.cols();
    for (Index col = 0; col < cols; ++col)
    {
        const Index feature_index = col % features;
        const int code = static_cast<int>(scalers_vector(feature_index));
        auto column = output_matrix.col(col).array();

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

static void copy_cpu(const TensorView& source, TensorView& destination)
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

static void add_cpu(const TensorView& input_1,
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

static void multiply_cpu(const TensorView& input_a, bool transpose_a,
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

static void softmax_cpu(TensorView& output)
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

static void activation_forward_cpu(TensorView& output, ActivationFunction function)
{
    auto a = output.as_vector().array();

    using enum ActivationFunction;
    switch (function)
    {
    case Identity:
    case Softmax:
        return;
    case Sigmoid:
        a = (1.0f + (-a).exp()).inverse();
        return;
    case Tanh:
        a = a.tanh();
        return;
    case ReLU:
        a = a.cwiseMax(0.0f);
        return;
    }
}

static void activation_backward_cpu(const TensorView& outputs, TensorView& delta, ActivationFunction function)
{
    const auto y = outputs.as_vector().array();
    auto       d = delta.as_vector().array();

    using enum ActivationFunction;
    switch (function)
    {
    case Identity:
    case Softmax:
        return;
    case Sigmoid:
        d *= y * (1.0f - y);
        return;
    case Tanh:
        d *= (1.0f - y.square());
        return;
    case ReLU:
        d = (y > 0.0f).select(d, 0.0f);
        return;
    }
}

void activation_forward(TensorView& output, ActivationFunction function)
{
    if (function == ActivationFunction::Identity || output.empty()) return;
    if (function == ActivationFunction::Softmax) { softmax(output); return; }

    IF_GPU({ activation_forward_gpu(output, function); return; });
    activation_forward_cpu(output, function);
}

void activation_backward(const TensorView& outputs, TensorView& delta, ActivationFunction function)
{
    if (function == ActivationFunction::Identity
        || function == ActivationFunction::Softmax
        || outputs.empty()) return;

    IF_GPU({ activation_backward_gpu(outputs, delta, function); return; });
    activation_backward_cpu(outputs, delta, function);
}

static void dropout_forward_cpu(TensorView& output, Buffer& mask, float rate)
{
    const Index n = output.size();
    mask.resize_bytes(n * Index(sizeof(float)), Device::CPU);

    const float scale = 1.0f / (1.0f - rate);
    float* data = output.as<float>();
    float* mask_data = mask.as<float>();

    #pragma omp parallel for
    for (Index i = 0; i < n; ++i)
    {
        const float mask_value = random_uniform(0.0f, 1.0f) < rate ? 0.0f : scale;
        mask_data[i] = mask_value;
        data[i] *= mask_value;
    }
}

static void dropout_backward_cpu(TensorView& delta, const Buffer& mask)
{
    const Index n = delta.size();
    Map<const VectorR, AlignedMax> mask_view(mask.as<float>(), n);
    delta.as_vector().array() *= mask_view.array();
}

void dropout_forward(TensorView& output, Buffer& mask, float rate)
{
    if (rate <= 0.0f) return;
    IF_GPU({ dropout_forward_gpu(output, mask, rate); return; });
    dropout_forward_cpu(output, mask, rate);
}

void dropout_backward(TensorView& delta, const Buffer& mask, float rate)
{
    if (rate <= 0.0f) return;
    IF_GPU({ dropout_backward_gpu(delta, mask, rate); return; });
    dropout_backward_cpu(delta, mask);
}

static void linear_forward_cpu(const TensorView& input, const TensorView& weights, const TensorView& bias,
                        TensorView& output, cublasLtEpilogue_t epilogue)
{
    output.as_flat_matrix().noalias() = (input.as_flat_matrix() * weights.as_matrix()).rowwise()
                                      + bias.as_vector().transpose();

    if (epilogue == CUBLASLT_EPILOGUE_RELU_BIAS)
        output.as_vector().array() = output.as_vector().array().cwiseMax(0.0f);
}

static void linear_backward_cpu(const TensorView& output_delta, const TensorView& input, const TensorView& weights,
                         const TensorView& weight_gradient, const TensorView& bias_gradient,
                         TensorView& input_delta, bool accumulate)
{
    weight_gradient.as_matrix().noalias() = input.as_flat_matrix().transpose() * output_delta.as_flat_matrix();
    bias_gradient.as_vector().noalias()   = output_delta.as_flat_matrix().colwise().sum();

    if (!input_delta.data || input_delta.size() == 0) return;

    auto input_delta_mat = input_delta.as_flat_matrix();
    const auto product   = output_delta.as_flat_matrix() * weights.as_matrix().transpose();

    if (accumulate) input_delta_mat.noalias() += product;
    else            input_delta_mat.noalias()  = product;
}

void linear_forward(const TensorView& input, const TensorView& weights, const TensorView& bias,
                    TensorView& output, cublasLtEpilogue_t epilogue)
{
    IF_GPU({ linear_forward_gpu(input, weights, bias, output, epilogue); return; });
    linear_forward_cpu(input, weights, bias, output, epilogue);
}

void linear_backward(const TensorView& output_delta, const TensorView& input, const TensorView& weights,
                     const TensorView& weight_gradient, const TensorView& bias_gradient,
                     TensorView& input_delta, bool accumulate_input_delta)
{
    IF_GPU({
        linear_backward_gpu(output_delta, input, weights, weight_gradient, bias_gradient,
                            input_delta, accumulate_input_delta);
        return;
    });
    linear_backward_cpu(output_delta, input, weights, weight_gradient, bias_gradient,
                        input_delta, accumulate_input_delta);
}

static void layer_norm_forward_cpu(const TensorView& input, const TensorView& gamma, const TensorView& beta,
                            TensorView& means, TensorView& standard_deviations,
                            TensorView& normalized, TensorView& output)
{
    const Index embedding_dimension = input.shape.back();
    const Index total_rows = input.size() / embedding_dimension;
    const float inv_D = 1.0f / to_type(embedding_dimension);

    const float* input_data = input.as<float>();
    float* means_data       = means.as<float>();
    float* stds_data        = standard_deviations.as<float>();
    float* normalized_data  = normalized.as<float>();
    float* output_data      = output.as<float>();
    const float* gamma_data = gamma.as<float>();
    const float* beta_data  = beta.as<float>();

    #pragma omp parallel for
    for (Index row = 0; row < total_rows; ++row)
    {
        const float* input_row = input_data + row * embedding_dimension;
        float* norm_row        = normalized_data + row * embedding_dimension;
        float* out_row         = output_data + row * embedding_dimension;

        float sum = 0;
        float sum_sq = 0;
        for (Index dim = 0; dim < embedding_dimension; ++dim)
        {
            const float value = input_row[dim];
            sum    += value;
            sum_sq += value * value;
        }

        const float mean    = sum * inv_D;
        const float std_val = sqrt(sum_sq * inv_D - mean * mean + EPSILON);
        const float inv_std = 1.0f / std_val;

        means_data[row] = mean;
        stds_data[row]  = std_val;

        for (Index dim = 0; dim < embedding_dimension; ++dim)
        {
            const float x_hat = (input_row[dim] - mean) * inv_std;
            norm_row[dim] = x_hat;
            out_row[dim]  = gamma_data[dim] * x_hat + beta_data[dim];
        }
    }
}

static void layer_norm_backward_cpu(const TensorView& output_delta,
                             const TensorView& standard_deviations,
                             const TensorView& normalized,
                             const TensorView& gamma,
                             const TensorView& gamma_gradient,
                             const TensorView& beta_gradient,
                             TensorView& input_delta)
{
    const Index embedding_dimension = output_delta.shape.back();
    const Index total_rows = output_delta.size() / embedding_dimension;
    const float inv_D = 1.0f / to_type(embedding_dimension);

    const MatrixMap output_delta_flat = output_delta.as_flat_matrix();
    const MatrixMap norm_flat         = normalized.as_flat_matrix();

    beta_gradient.as_vector().noalias()  = output_delta_flat.colwise().sum();
    gamma_gradient.as_vector().noalias() = (output_delta_flat.array() * norm_flat.array()).matrix().colwise().sum();

    const float* output_delta_data = output_delta.as<float>();
    const float* norm_data         = normalized.as<float>();
    const float* std_data          = standard_deviations.as<float>();
    const float* gamma_data        = gamma.as<float>();
    float* input_delta_data        = input_delta.as<float>();

    #pragma omp parallel for
    for (Index row = 0; row < total_rows; ++row)
    {
        const float* output_delta_row = output_delta_data + row * embedding_dimension;
        const float* norm_row         = norm_data + row * embedding_dimension;
        float* input_delta_row        = input_delta_data + row * embedding_dimension;
        const float inv_std = 1.0f / std_data[row];

        float sum_scaled_gradient      = 0;
        float sum_scaled_gradient_norm = 0;
        for (Index dim = 0; dim < embedding_dimension; ++dim)
        {
            const float scaled_gradient = gamma_data[dim] * output_delta_row[dim];
            sum_scaled_gradient      += scaled_gradient;
            sum_scaled_gradient_norm += scaled_gradient * norm_row[dim];
        }
        sum_scaled_gradient      *= inv_D;
        sum_scaled_gradient_norm *= inv_D;

        for (Index dim = 0; dim < embedding_dimension; ++dim)
        {
            const float scaled_gradient = gamma_data[dim] * output_delta_row[dim];
            input_delta_row[dim] = (scaled_gradient - sum_scaled_gradient
                                  - norm_row[dim] * sum_scaled_gradient_norm) * inv_std;
        }
    }
}

void layer_norm_forward(const TensorView& input, const TensorView& gamma, const TensorView& beta,
                        TensorView& means, TensorView& standard_deviations,
                        TensorView& normalized, TensorView& output)
{
    IF_GPU({ layer_norm_forward_gpu(input, gamma, beta, means, standard_deviations, output); return; });
    layer_norm_forward_cpu(input, gamma, beta, means, standard_deviations, normalized, output);
}

void layer_norm_backward(const TensorView& input, const TensorView& output_delta,
                         const TensorView& means, const TensorView& standard_deviations,
                         const TensorView& normalized, const TensorView& gamma,
                         const TensorView& gamma_gradient, const TensorView& beta_gradient,
                         TensorView& input_delta)
{
    IF_GPU({
        layer_norm_backward_gpu(input, output_delta, means, standard_deviations, gamma,
                                gamma_gradient, beta_gradient, input_delta);
        return;
    });
    layer_norm_backward_cpu(output_delta, standard_deviations, normalized, gamma,
                            gamma_gradient, beta_gradient, input_delta);
}

static void embedding_lookup_forward_cpu(const TensorView& indices, const TensorView& weights,
                                  const TensorView& positional_encoding, TensorView& output,
                                  Index sequence_length, Index embedding_dimension, Index vocabulary_size,
                                  bool scale_embedding, bool add_positional_encoding)
{
    const Index total_tokens = indices.size();

    MatrixMap output_mat        = output.as_flat_matrix();
    const MatrixMap weights_mat = weights.as_matrix();
    const float* input_indices  = indices.as<float>();

    static atomic<bool> out_of_range_warned{false};

    #pragma omp parallel for
    for (Index i = 0; i < total_tokens; ++i)
    {
        const Index token_id = static_cast<Index>(input_indices[i]);

        if (token_id == 0)
        {
            output_mat.row(i).setZero();
            continue;
        }

        if (token_id < 0 || token_id >= vocabulary_size)
        {
            if (!out_of_range_warned.exchange(true))
                cerr << "EmbeddingLookup warning: token id " << token_id
                     << " out of range [0, " << vocabulary_size
                     << "); zeroing row. Further warnings suppressed.\n";
            output_mat.row(i).setZero();
            continue;
        }

        output_mat.row(i).noalias() = weights_mat.row(token_id);

        if (scale_embedding)
            output_mat.row(i) *= sqrt(to_type(embedding_dimension));

        if (add_positional_encoding && token_id > 0)
            output_mat.row(i) += positional_encoding.as_matrix().row(i % sequence_length);
    }
}

static void embedding_lookup_backward_cpu(const TensorView& indices, const TensorView& output_delta,
                                   const TensorView& weight_gradient,
                                   Index embedding_dimension, Index vocabulary_size,
                                   bool scale_embedding)
{
    const Index total_elements = indices.size();

    MatrixMap output_delta_map = output_delta.as_flat_matrix();
    MatrixMap weight_gradients = weight_gradient.as_matrix().setZero();
    const float scale = scale_embedding ? sqrt(to_type(embedding_dimension)) : 1.0f;

    for (Index token_index = 0; token_index < total_elements; ++token_index)
    {
        const Index vocabulary_index = static_cast<Index>(indices.as<float>()[token_index]);

        if (vocabulary_index <= 0 || vocabulary_index >= vocabulary_size)
            continue;

        weight_gradients.row(vocabulary_index).noalias() += scale * output_delta_map.row(token_index);
    }
}

void embedding_lookup_forward(const TensorView& indices, const TensorView& weights,
                              const TensorView& positional_encoding, TensorView& output,
                              Index sequence_length, Index embedding_dimension, Index vocabulary_size,
                              bool scale_embedding, bool add_positional_encoding)
{
    IF_GPU({
        embedding_lookup_forward_gpu(indices, weights, positional_encoding, output,
                                     sequence_length, embedding_dimension, vocabulary_size,
                                     scale_embedding, add_positional_encoding);
        return;
    });
    embedding_lookup_forward_cpu(indices, weights, positional_encoding, output,
                                 sequence_length, embedding_dimension, vocabulary_size,
                                 scale_embedding, add_positional_encoding);
}

void embedding_lookup_backward(const TensorView& indices, const TensorView& output_delta,
                               const TensorView& weight_gradient,
                               Index embedding_dimension, Index vocabulary_size,
                               bool scale_embedding)
{
    IF_GPU({
        embedding_lookup_backward_gpu(indices, output_delta, weight_gradient,
                                      embedding_dimension, vocabulary_size, scale_embedding);
        return;
    });
    embedding_lookup_backward_cpu(indices, output_delta, weight_gradient,
                                  embedding_dimension, vocabulary_size, scale_embedding);
}

static void max_pooling_3d_forward_cpu(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training)
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
                if (value <= outputs(batch_index, feature_index)) continue;
                outputs(batch_index, feature_index) = value;
                if (is_training) max_indices(batch_index, feature_index) = to_type(step);
            }
    }
}

void max_pooling_3d_forward(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training)
{
    IF_GPU({ max_pooling_3d_forward_gpu(input, output, maximal_indices, is_training); return; });
    max_pooling_3d_forward_cpu(input, output, maximal_indices, is_training);
}

static void average_pooling_3d_forward_cpu(const TensorView& input, TensorView& output)
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

        if (valid_count == 0) { outputs.row(batch_index).setZero(); continue; }
        outputs.row(batch_index) = seq_matrix.colwise().sum() / to_type(valid_count);
    }
}

void average_pooling_3d_forward(const TensorView& input, TensorView& output)
{
    IF_GPU({ average_pooling_3d_forward_gpu(input, output); return; });
    average_pooling_3d_forward_cpu(input, output);
}

static void max_pooling_3d_backward_cpu(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta)
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

static void split_heads_cpu(const TensorView& source, TensorView& destination)
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

static void merge_heads_cpu(const TensorView& source, TensorView& destination)
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

static void bound_gpu(const TensorView& input,
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

static void scale_gpu(const TensorView& input,
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

static void unscale_gpu(const TensorView& input,
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

static void copy_gpu(const TensorView& source, TensorView& destination)
{
    CHECK_CUDA(cudaMemcpyAsync(destination.data, source.data, source.byte_size(),
                               cudaMemcpyDeviceToDevice, Backend::get_compute_stream()));
}

static void add_gpu(const TensorView& input_1,
             const TensorView& input_2,
             TensorView& output)
{
    CHECK_CUDNN(cudnnOpTensor(Backend::get_cudnn_handle(),
                              Backend::get_operator_sum_descriptor(),
                              &one, input_1.get_descriptor(), input_1.data,
                              &one, input_2.get_descriptor(), input_2.data,
                              &zero, output.get_descriptor(), output.data));
}

static void multiply_gpu(const TensorView& input_a, bool transpose_a,
                  const TensorView& input_b, bool transpose_b,
                  TensorView& output,
                  float alpha, float beta)
{
    const size_t rank_a = input_a.get_rank();
    const size_t rank_b = input_b.get_rank();

    int rows_a = to_int(input_a.shape[rank_a - 2]);
    int cols_a = to_int(input_a.shape[rank_a - 1]);
    const int rows_b = to_int(input_b.shape[rank_b - 2]);
    const int cols_b = to_int(input_b.shape[rank_b - 1]);

    // Rank-mismatched broadcast (e.g. {B,Q,E} @ {E,E}^T from CombinationOp::apply_delta_gpu):
    // flatten input_a to 2D and do a single GEMM with no batching. Mirrors multiply_cpu's
    // use of as_flat_matrix() for these calls.
    if (rank_b == 2 && rank_a > 2)
    {
        rows_a = to_int(input_a.size() / cols_a);
    }

    const int output_columns = transpose_b ? rows_b : cols_b;
    const int output_rows = transpose_a ? cols_a : rows_a;
    const int inner_dimension = transpose_a ? rows_a : cols_a;

    const cublasOperation_t operation_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t operation_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    const int batch_count = to_int(input_a.size() / (rows_a * cols_a));
    const long long stride_a = rows_a * cols_a;
    const long long stride_b = rows_b * cols_b;
    const long long stride_output = output.shape[output.get_rank() - 2] * output.shape[output.get_rank() - 1];

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

static void softmax_gpu(TensorView& output)
{
    CHECK_CUDNN(cudnnSoftmaxForward(Backend::get_cudnn_handle(),
                                    CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &one,
                                    output.get_descriptor(), output.data,
                                    &zero,
                                    output.get_descriptor(), output.data));
}

static void activation_forward_gpu(TensorView& output, ActivationFunction function)
{
    output.dispatch([&](auto tag)
    {
        using T = decltype(tag);
        activation_forward_cuda<T>(output.size(), output.as<T>(), static_cast<int>(function));
    });
    CHECK_CUDA(cudaPeekAtLastError());
}

static void activation_backward_gpu(const TensorView& outputs, TensorView& delta, ActivationFunction function)
{
    delta.dispatch([&](auto tag)
    {
        using T = decltype(tag);
        activation_backward_cuda<T>(delta.size(), outputs.as<T>(), delta.as<T>(), static_cast<int>(function));
    });
    CHECK_CUDA(cudaPeekAtLastError());
}

static void dropout_forward_gpu(TensorView& output, Buffer& mask, float rate)
{
    const Index n = output.size();
    if (mask.device_type != Device::CUDA || mask.bytes < n)
        mask.resize_bytes(n, Device::CUDA);

    const unsigned long long seed = static_cast<unsigned long long>(random_integer(0, 1 << 30));

    output.dispatch([&](auto tag)
    {
        using T = decltype(tag);
        dropout_forward_cuda<T>(n, output.as<T>(), mask.as<uint8_t>(), rate, seed);
    });
}

static void dropout_backward_gpu(TensorView& delta, const Buffer& mask, float rate)
{
    const Index n = delta.size();

    delta.dispatch([&](auto tag)
    {
        using T = decltype(tag);
        dropout_backward_cuda<T>(n, delta.as<T>(), delta.as<T>(), mask.as<uint8_t>(), rate);
    });
}

static void linear_forward_gpu(const TensorView& input, const TensorView& weights, const TensorView& bias,
                        TensorView& output, cublasLtEpilogue_t epilogue)
{
    const int input_columns  = to_int(input.shape.back());
    const int output_columns = to_int(weights.shape.back());
    const int total_rows     = to_int(input.size() / input.shape.back());

    const void* input_for_gemm = data_for_gemm_dtype(input, weights.type);
    const cudaDataType_t io_type = output.cuda_dtype();

    const LtMatmulPlan& plan = get_lt_gemm_plan(
        output_columns, total_rows, input_columns,
        CUBLAS_OP_N, CUBLAS_OP_N,
        epilogue, io_type, io_type);

    run_lt_matmul(plan, weights.data, input_for_gemm, output.data, bias.data);
}

static void linear_backward_gpu(const TensorView& output_delta, const TensorView& input, const TensorView& weights,
                         const TensorView& weight_gradient, const TensorView& bias_gradient,
                         TensorView& input_delta, bool accumulate_input_delta)
{
    const int input_columns  = to_int(input.shape.back());
    const int output_columns = to_int(output_delta.shape.back());
    const int total_rows     = to_int(input.size() / input.shape.back());

    const void* input_for_gemm = data_for_gemm_dtype(input, weights.type);

    const LtMatmulPlan& plan = get_lt_gemm_plan(
        output_columns, input_columns, total_rows,
        CUBLAS_OP_N, CUBLAS_OP_T,
        CUBLASLT_EPILOGUE_BGRADA,
        output_delta.cuda_dtype(),
        CUDA_R_32F);

    run_lt_matmul(plan, output_delta.data, input_for_gemm, weight_gradient.data, bias_gradient.as<float>());

    if (!input_delta.data || input_delta.size() == 0) return;

    multiply(output_delta, false, weights, true, input_delta, 1.0f,
             accumulate_input_delta ? 1.0f : 0.0f);
}

static void layer_norm_forward_gpu(const TensorView& input, const TensorView& gamma, const TensorView& beta,
                            TensorView& means, TensorView& standard_deviations, TensorView& output)
{
    const int rows = to_int(input.size() / input.shape.back());
    const int cols = to_int(input.shape.back());

    output.dispatch([&](auto tag) {
        using T = decltype(tag);
        layernorm_forward_cuda<T>(rows, cols,
                                  input.as<T>(), output.as<T>(),
                                  means.as<float>(), standard_deviations.as<float>(),
                                  gamma.as<float>(), beta.as<float>(), EPSILON);
    });
}

static void layer_norm_backward_gpu(const TensorView& input, const TensorView& output_delta,
                             const TensorView& means, const TensorView& standard_deviations,
                             const TensorView& gamma,
                             const TensorView& gamma_gradient, const TensorView& beta_gradient,
                             TensorView& input_delta)
{
    const int rows = to_int(input.size() / input.shape.back());
    const int cols = to_int(input.shape.back());

    input.dispatch([&](auto tag) {
        using T = decltype(tag);
        layernorm_backward_cuda<T>(rows, cols,
                                   output_delta.as<T>(), input.as<T>(),
                                   means.as<float>(), standard_deviations.as<float>(),
                                   gamma.as<float>(),
                                   input_delta.as<T>(),
                                   gamma_gradient.as<float>(), beta_gradient.as<float>());
    });
}

static void embedding_lookup_forward_gpu(const TensorView& indices, const TensorView& weights,
                                  const TensorView& positional_encoding, TensorView& output,
                                  Index sequence_length, Index embedding_dimension, Index vocabulary_size,
                                  bool scale_embedding, bool add_positional_encoding)
{
    output.dispatch([&](auto tag) {
        using T = decltype(tag);
        embedding_forward_cuda<T>(
            output.size(),
            indices.as<float>(),
            weights.as<float>(),
            add_positional_encoding ? positional_encoding.as<float>() : nullptr,
            output.as<T>(),
            to_int(sequence_length), to_int(embedding_dimension), to_int(vocabulary_size),
            scale_embedding);
    });
}

static void embedding_lookup_backward_gpu(const TensorView& indices, const TensorView& output_delta,
                                   const TensorView& weight_gradient,
                                   Index embedding_dimension, Index vocabulary_size,
                                   bool scale_embedding)
{
    weight_gradient.set_zero_async();

    output_delta.dispatch([&](auto tag) {
        using T = decltype(tag);
        embedding_backward_cuda<T>(
            output_delta.size(),
            indices.as<float>(),
            output_delta.as<T>(),
            weight_gradient.as<float>(),
            to_int(embedding_dimension), to_int(vocabulary_size), scale_embedding);
    });
}

static void max_pooling_3d_forward_gpu(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool /* is_training */)
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

static void average_pooling_3d_forward_gpu(const TensorView& input, TensorView& output)
{
    output.dispatch([&](auto tag) {
        using T = decltype(tag);
        average_pooling_3d_forward_cuda<T>(to_int(input.shape[0]) * to_int(input.shape[2]),
                                           input.as<T>(), output.as<T>(),
                                           to_int(input.shape[1]),
                                           to_int(input.shape[2]));
    });
}

static void max_pooling_3d_backward_gpu(const TensorView& maximal_indices, const TensorView& output_delta, TensorView& input_delta)
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

static void average_pooling_3d_backward_gpu(const TensorView& input,
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

static void split_heads_gpu(const TensorView& source, TensorView& destination)
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

static void merge_heads_gpu(const TensorView& source, TensorView& destination)
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
