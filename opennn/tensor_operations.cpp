//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   O P E R A T I O N S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensor_operations.h"
#include "cpu_math_backend.h"
#include "device_backend.h"
#include "random_utilities.h"
#include "profiler.h"

namespace opennn
{

const EnumMap<ActivationFunction>& activation_function_map()
{
    static const vector<pair<ActivationFunction, string>> entries = {
        {ActivationFunction::Identity, "Identity"},
        {ActivationFunction::Sigmoid,  "Sigmoid"},
        {ActivationFunction::Tanh,     "Tanh"},
        {ActivationFunction::ReLU,     "ReLU"},
        {ActivationFunction::Softmax,  "Softmax"}
    };
    static const EnumMap<ActivationFunction> instance{entries};
    return instance;
}

const string& activation_function_to_string(ActivationFunction function)
{
    return activation_function_map().to_string(function);
}

ActivationFunction activation_function_from_string(const string& name)
{
    return activation_function_map().from_string(name, ActivationFunction::Identity);
}

#define OPENNN_GPU_OPS(X) \
    X(bound_gpu, (const TensorView&, const TensorView&, const TensorView&, TensorView&)) \
    X(scale_gpu, (const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, float, float, TensorView&)) \
    X(unscale_gpu, (const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, float, float, TensorView&)) \
    X(copy_gpu, (const TensorView&, TensorView&)) \
    X(add_gpu, (const TensorView&, const TensorView&, TensorView&)) \
    X(multiply_gpu, (const TensorView&, bool, const TensorView&, bool, TensorView&, float, float)) \
    X(softmax_gpu, (TensorView&)) \
    X(activation_forward_gpu, (TensorView&, ActivationFunction)) \
    X(activation_backward_gpu, (const TensorView&, TensorView&, ActivationFunction)) \
    X(dropout_forward_gpu, (TensorView&, Buffer&, float)) \
    X(dropout_backward_gpu, (TensorView&, const Buffer&, float)) \
    X(linear_forward_gpu, (const TensorView&, const TensorView&, const TensorView&, TensorView&, cublasLtEpilogue_t)) \
    X(linear_backward_gpu, (const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, TensorView&, bool)) \
    X(layer_norm_forward_gpu, (const TensorView&, const TensorView&, const TensorView&, TensorView&, TensorView&, TensorView&)) \
    X(layer_norm_backward_gpu, (const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, TensorView&)) \
    X(embedding_lookup_forward_gpu, (const TensorView&, const TensorView&, const TensorView&, TensorView&, Index, Index, Index, bool, bool)) \
    X(embedding_lookup_backward_gpu, (const TensorView&, const TensorView&, const TensorView&, Index, Index, bool)) \
    X(max_pooling_3d_forward_gpu, (const TensorView&, TensorView&, TensorView&, bool)) \
    X(average_pooling_3d_forward_gpu, (const TensorView&, TensorView&)) \
    X(max_pooling_3d_backward_gpu, (const TensorView&, const TensorView&, TensorView&)) \
    X(average_pooling_3d_backward_gpu, (const TensorView&, const TensorView&, TensorView&)) \
    X(split_heads_gpu, (const TensorView&, TensorView&)) \
    X(merge_heads_gpu, (const TensorView&, TensorView&))

#define OPENNN_DECLARE_GPU_OP(name, sig) static void name sig;
OPENNN_GPU_OPS(OPENNN_DECLARE_GPU_OP)
#undef OPENNN_DECLARE_GPU_OP

static void bound_cpu(const TensorView& input,
               const TensorView& lower_bounds,
               const TensorView& upper_bounds,
               TensorView& output)
{
    const Index features = lower_bounds.size();

    const MatrixMap input_matrix = input.as_flat_matrix();
    const VectorMap lower_bounds_vector = lower_bounds.as_vector();
    const VectorMap upper_bounds_vector = upper_bounds.as_vector();

    MatrixMap output_matrix = output.as_flat_matrix();

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
    if (input.is_cuda()) { bound_gpu(input, lower_bounds, upper_bounds, output); return; }
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
        case 1:
        {
            const float range = maximums_vector(feature_index) - minimums_vector(feature_index);
            if (range < EPSILON)
                column.setZero();
            else
                column = (column - minimums_vector(feature_index)) / range
                       * (max_range - min_range) + min_range;
            break;
        }
        case 2:
        {
            const float sd = standard_deviations_vector(feature_index);
            if (sd > EPSILON)
                column = (column - means_vector(feature_index)) / sd;
            else
                column.setZero();
            break;
        }
        case 3:
        {
            const float sd = standard_deviations_vector(feature_index);
            column *= (sd > EPSILON) ? (1.0f / sd) : 0.0f;
            break;
        }
        case 4:
            column = column.log();
            break;
        case 5:
            column /= 255.0f;
            break;
        default:
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
    if (input.is_cuda())
    {
        scale_gpu(input, minimums, maximums, means, standard_deviations, scalers,
                  min_range, max_range, output);
        return;
    }
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
        case 1:
            column = (column - min_range) / (max_range - min_range)
                   * (maximums_vector(feature_index) - minimums_vector(feature_index)) + minimums_vector(feature_index);
            break;
        case 2:
            column = means_vector(feature_index) + column * standard_deviations_vector(feature_index);
            break;
        case 3:
            column *= standard_deviations_vector(feature_index);
            break;
        case 4:
            column = column.exp();
            break;
        case 5:
            column *= 255.0f;
            break;
        default:
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
    if (input.is_cuda())
    {
        unscale_gpu(input, minimums, maximums, means, standard_deviations, scalers,
                    min_range, max_range, output);
        return;
    }
    
    unscale_cpu(input, minimums, maximums, means, standard_deviations, scalers,
                min_range, max_range, output);
}

static void copy_cpu(const TensorView& source, TensorView& destination)
{
    memcpy(destination.data, source.data, source.size() * sizeof(float));
}

void copy(const TensorView& source, TensorView& destination)
{
    throw_if(source.size() != destination.size(),
             "Tensor sizes mismatch in copy operation.");

    if (source.is_cuda()) { copy_gpu(source, destination); return; }
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
    throw_if(input_1.size() != input_2.size() || input_1.size() != output.size(),
             "Tensor dimensions do not match.");

    if (input_1.is_cuda()) { add_gpu(input_1, input_2, output); return; }
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
    if (input_a.is_cuda()) { multiply_gpu(input_a, transpose_a, input_b, transpose_b, output, alpha, beta); return; }
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

    if (output.is_cuda()) { softmax_gpu(output); return; }
    softmax_cpu(output);
}

static void activation_forward_cpu(TensorView& output, ActivationFunction function)
{
    if (cpu_math::try_activation_forward(output, function)) return;

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

    if (output.is_cuda()) { activation_forward_gpu(output, function); return; }
    activation_forward_cpu(output, function);
}

void activation_backward(const TensorView& outputs, TensorView& delta, ActivationFunction function)
{
    if (function == ActivationFunction::Identity
        || function == ActivationFunction::Softmax
        || outputs.empty()) return;

    if (outputs.is_cuda()) { activation_backward_gpu(outputs, delta, function); return; }
    activation_backward_cpu(outputs, delta, function);
}

static void dropout_forward_cpu(TensorView& output, Buffer& mask, float rate)
{
    const Index element_count = output.size();
    mask.resize_bytes(element_count * Index(sizeof(float)), Device::CPU);
    if (element_count == 0) return;

    const float keep_scale = 1.0f / (1.0f - rate);
    float* output_data = output.as<float>();
    float* mask_values = mask.as<float>();

    set_random_uniform(VectorMap(mask_values, element_count), 0.0f, 1.0f);

    #pragma omp parallel for
    for (Index i = 0; i < element_count; ++i)
    {
        const float keep_value = mask_values[i] < rate ? 0.0f : keep_scale;
        mask_values[i] = keep_value;
        output_data[i] *= keep_value;
    }
}

static void dropout_backward_cpu(TensorView& delta, const Buffer& mask)
{
    const Index element_count = delta.size();
    Map<const VectorR, AlignedMax> mask_view(mask.as<float>(), element_count);
    delta.as_vector().array() *= mask_view.array();
}

void dropout_forward(TensorView& output, Buffer& mask, float rate)
{
    if (rate <= 0.0f) return;
    if (output.is_cuda()) { dropout_forward_gpu(output, mask, rate); return; }
    dropout_forward_cpu(output, mask, rate);
}

void dropout_backward(TensorView& delta, const Buffer& mask, float rate)
{
    if (rate <= 0.0f) return;
    if (delta.is_cuda()) { dropout_backward_gpu(delta, mask, rate); return; }
    dropout_backward_cpu(delta, mask);
}

static void linear_forward_cpu(const TensorView& input, const TensorView& weights, const TensorView& bias,
                        TensorView& output, cublasLtEpilogue_t epilogue)
{
    const bool fuse_relu = epilogue == CUBLASLT_EPILOGUE_RELU_BIAS;

    if (cpu_math::try_linear_forward(input, weights, bias, output, fuse_relu)) return;

    output.as_flat_matrix().noalias() = (input.as_flat_matrix() * weights.as_matrix()).rowwise()
                                      + bias.as_vector().transpose();

    if (fuse_relu)
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
    if (input.is_cuda()) { linear_forward_gpu(input, weights, bias, output, epilogue); return; }
    linear_forward_cpu(input, weights, bias, output, epilogue);
}

void linear_backward(const TensorView& output_delta, const TensorView& input, const TensorView& weights,
                     const TensorView& weight_gradient, const TensorView& bias_gradient,
                     TensorView& input_delta, bool accumulate_input_delta)
{
    if (output_delta.is_cuda())
    {
        linear_backward_gpu(output_delta, input, weights, weight_gradient, bias_gradient,
                            input_delta, accumulate_input_delta);
        return;
    }
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

        const Map<const Array<float, Dynamic, 1>> input_map(input_row, embedding_dimension);
        const float sum    = input_map.sum();
        const float sum_sq = input_map.square().sum();

        const float mean    = sum * inv_D;
        // Variance via E[x^2] - E[x]^2 can go slightly negative from catastrophic
        // cancellation when activations are large (high embedding dimension); clamp
        // to >= 0 before the sqrt so the layer norm cannot produce NaN.
        const float variance = max(sum_sq * inv_D - mean * mean, 0.0f);
        const float std_val = sqrt(variance + EPSILON);
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

    if (input_delta.empty()) return;

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

        const Map<const Array<float, Dynamic, 1>> gamma_map(gamma_data, embedding_dimension);
        const Map<const Array<float, Dynamic, 1>> output_delta_map(output_delta_row, embedding_dimension);
        const Map<const Array<float, Dynamic, 1>> norm_map(norm_row, embedding_dimension);
        Map<Array<float, Dynamic, 1>> input_delta_map(input_delta_row, embedding_dimension);

        const float sum_scaled_gradient      = (gamma_map * output_delta_map).sum() * inv_D;
        const float sum_scaled_gradient_norm = (gamma_map * output_delta_map * norm_map).sum() * inv_D;

        input_delta_map = (gamma_map * output_delta_map - sum_scaled_gradient
                          - norm_map * sum_scaled_gradient_norm) * inv_std;
    }
}

void layer_norm_forward(const TensorView& input, const TensorView& gamma, const TensorView& beta,
                        TensorView& means, TensorView& standard_deviations,
                        TensorView& normalized, TensorView& output)
{
    if (input.is_cuda()) { layer_norm_forward_gpu(input, gamma, beta, means, standard_deviations, output); return; }
    layer_norm_forward_cpu(input, gamma, beta, means, standard_deviations, normalized, output);
}

void layer_norm_backward(const TensorView& input, const TensorView& output_delta,
                         const TensorView& means, const TensorView& standard_deviations,
                         const TensorView& normalized, const TensorView& gamma,
                         const TensorView& gamma_gradient, const TensorView& beta_gradient,
                         TensorView& input_delta)
{
    if (input.is_cuda())
    {
        layer_norm_backward_gpu(input, output_delta, means, standard_deviations, gamma,
                                gamma_gradient, beta_gradient, input_delta);
        return;
    }
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

        if (add_positional_encoding)
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
    if (output.is_cuda())
    {
        embedding_lookup_forward_gpu(indices, weights, positional_encoding, output,
                                     sequence_length, embedding_dimension, vocabulary_size,
                                     scale_embedding, add_positional_encoding);
        return;
    }
    embedding_lookup_forward_cpu(indices, weights, positional_encoding, output,
                                 sequence_length, embedding_dimension, vocabulary_size,
                                 scale_embedding, add_positional_encoding);
}

void embedding_lookup_backward(const TensorView& indices, const TensorView& output_delta,
                               const TensorView& weight_gradient,
                               Index embedding_dimension, Index vocabulary_size,
                               bool scale_embedding)
{
    if (output_delta.is_cuda())
    {
        embedding_lookup_backward_gpu(indices, output_delta, weight_gradient,
                                      embedding_dimension, vocabulary_size, scale_embedding);
        return;
    }
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
    if (output_delta.is_cuda()) { average_pooling_3d_backward_gpu(input, output_delta, input_delta); return; }
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
    if (source.is_cuda()) { split_heads_gpu(source, destination); return; }
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
    if (source.is_cuda()) { merge_heads_gpu(source, destination); return; }
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
    device::copy_async(destination.data, source.data, source.byte_size(),
                       device::CopyKind::DeviceToDevice,
                       Backend::get_compute_stream());
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
                                  input_b.data, input_b.cuda_dtype(), cols_b, stride_b,
                                  input_a.data, input_a.cuda_dtype(), cols_a, stride_a,
                                  output.data,  output.cuda_dtype(), output_columns, stride_output,
                                  batch_count,
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
    device::check_last_error();
}

static void activation_backward_gpu(const TensorView& outputs, TensorView& delta, ActivationFunction function)
{
    delta.dispatch([&](auto tag)
    {
        using T = decltype(tag);
        activation_backward_cuda<T>(delta.size(), outputs.as<T>(), delta.as<T>(), static_cast<int>(function));
    });
    device::check_last_error();
}

static void dropout_forward_gpu(TensorView& output, Buffer& mask, float rate)
{
    const Index element_count = output.size();
    if (mask.device_type != Device::CUDA || mask.bytes < element_count)
        mask.resize_bytes(element_count, Device::CUDA);

    const unsigned long long seed = static_cast<unsigned long long>(random_integer(0, 1 << 30));

    output.dispatch([&](auto tag)
    {
        using T = decltype(tag);
        dropout_forward_cuda<T>(element_count, output.as<T>(), mask.as<uint8_t>(), rate, seed);
    });
}

static void dropout_backward_gpu(TensorView& delta, const Buffer& mask, float rate)
{
    const Index element_count = delta.size();

    delta.dispatch([&](auto tag)
    {
        using T = decltype(tag);
        dropout_backward_cuda<T>(element_count, delta.as<T>(), delta.as<T>(), mask.as<uint8_t>(), rate);
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

    // cuBLASLt's fused BIAS epilogue requires the bias data type to match the
    // matmul I/O type. The bias is stored fp32, so for a bf16 matmul we must
    // cast it to bf16 first; a bf16-I/O + fp32-bias fused epilogue is rejected
    // by the heuristic (no algorithm) and the matmul then fails (cuBLAS 14).
    const void* bias_for_gemm = (bias.data && output.type == Type::BF16 && bias.type == Type::FP32)
        ? bias_for_gemm_bf16(bias)
        : bias.data;

    try
    {
        run_lt_matmul_cached(
            output_columns, total_rows, input_columns,
            CUBLAS_OP_N, CUBLAS_OP_N,
            epilogue,
            weights.data, input_for_gemm, output.data, bias_for_gemm,
            io_type, io_type);
    }
    catch (const runtime_error& e)
    {
        const bool unsupported_bf16_lt = output.type == Type::BF16
                                      && (epilogue == CUBLASLT_EPILOGUE_BIAS
                                          || epilogue == CUBLASLT_EPILOGUE_RELU_BIAS)
                                      && string(e.what()).find("CuBLAS Error: 15") != string::npos;

        if (!unsupported_bf16_lt)
            throw;

        cudaStream_t stream = Backend::get_compute_stream();

        VectorR input_host(input.size());
        VectorR weights_host(weights.size());
        VectorR bias_host(bias.size());
        VectorR output_host(output.size());

        copy_device_to_host_float(input.data, input.type, input.size(), input_host.data(), stream);
        copy_device_to_host_float(weights.data, weights.type, weights.size(), weights_host.data(), stream);
        if (bias.data)
            copy_device_to_host_float(bias.data, bias.type, bias.size(), bias_host.data(), stream);

        TensorView input_cpu(input_host.data(), input.shape, Type::FP32, Device::CPU);
        TensorView weights_cpu(weights_host.data(), weights.shape, Type::FP32, Device::CPU);
        TensorView bias_cpu(bias_host.data(), bias.shape, Type::FP32, Device::CPU);
        TensorView output_cpu(output_host.data(), output.shape, Type::FP32, Device::CPU);

        linear_forward_cpu(input_cpu, weights_cpu, bias_cpu, output_cpu, epilogue);

        if (output.type == Type::FP32)
        {
            device::copy_async(output.data,
                               output_host.data(),
                               output.byte_size(),
                               device::CopyKind::HostToDevice,
                               stream);
        }
        else
        {
            Buffer output_fp32(Device::CUDA);
            output_fp32.resize_bytes(output.size() * Index(sizeof(float)), Device::CUDA);
            device::copy_async(output_fp32.data,
                               output_host.data(),
                               output_fp32.bytes,
                               device::CopyKind::HostToDevice,
                               stream);
            cast_fp32_to_bf16_cuda(output.size(),
                                   output_fp32.as<float>(),
                                   output.as<bfloat16>(),
                                   stream);
        }

        device::synchronize(stream);
    }
}

static void linear_backward_gpu(const TensorView& output_delta, const TensorView& input, const TensorView& weights,
                         const TensorView& weight_gradient, const TensorView& bias_gradient,
                         TensorView& input_delta, bool accumulate_input_delta)
{
    const int input_columns  = to_int(input.shape.back());
    const int output_columns = to_int(output_delta.shape.back());
    const int total_rows     = to_int(input.size() / input.shape.back());

    const void* input_for_gemm = data_for_gemm_dtype(input, weights.type);

    run_lt_matmul_cached(
        output_columns, input_columns, total_rows,
        CUBLAS_OP_N, CUBLAS_OP_T,
        CUBLASLT_EPILOGUE_BGRADA,
        output_delta.data, input_for_gemm, weight_gradient.data, bias_gradient.as<float>(),
        output_delta.cuda_dtype(),
        CUDA_R_32F);

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
        T* input_delta_data = input_delta.empty() ? nullptr : input_delta.as<T>();

        layernorm_backward_cuda<T>(rows, cols,
                                   output_delta.as<T>(), input.as<T>(),
                                   means.as<float>(), standard_deviations.as<float>(),
                                   gamma.as<float>(),
                                   input_delta_data,
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

#else

#define OPENNN_STUB_GPU_OP(name, sig) static void name sig { throw runtime_error(#name ": CUDA support not compiled in."); }
OPENNN_GPU_OPS(OPENNN_STUB_GPU_OP)
#undef OPENNN_STUB_GPU_OP

#endif // OPENNN_HAS_CUDA


MatrixR append_rows(const MatrixR& starting_matrix, const MatrixR& block)
{
    if (starting_matrix.size() == 0)
        return block;
    if (block.size() == 0)
        return starting_matrix;

    throw_if(starting_matrix.cols() != block.cols(),
             format("append_rows: Column mismatch ({} vs {})",
                    starting_matrix.cols(), block.cols()));

    MatrixR final_matrix(starting_matrix.rows() + block.rows(), starting_matrix.cols());

    final_matrix.topRows(starting_matrix.rows()) = starting_matrix;
    final_matrix.bottomRows(block.rows()) = block;

    return final_matrix;
}


MatrixR append_columns(const MatrixR& first_matrix, const MatrixR& second_matrix)
{
    MatrixR result(first_matrix.rows(), first_matrix.cols() + second_matrix.cols());
    result.leftCols(first_matrix.cols()) = first_matrix;
    result.rightCols(second_matrix.cols()) = second_matrix;
    return result;
}


VectorI get_nearest_points(const MatrixR& matrix, const VectorR& point, int neighbors_number)
{
    const Index rows = matrix.rows();

    const VectorR distances = (matrix.rowwise() - point.transpose()).rowwise().norm();

    vector<pair<float, Index>> pairs(rows);

    for (Index i = 0; i < rows; ++i)
        pairs[i] = {distances(i), i};

    if (neighbors_number > rows)
        neighbors_number = rows;

    partial_sort(pairs.begin(), pairs.begin() + neighbors_number, pairs.end());

    VectorI result(neighbors_number);
    transform(pairs.begin(), pairs.begin() + neighbors_number, result.data(),
              [](const auto& p) { return p.second; });
    return result;
}


vector<Index> filter_selected_indices_by_column(const MatrixR& matrix,
                                                const vector<Index>& selected_indices,
                                                const Index column_index,
                                                const float minimum,
                                                const float maximum)
{
    vector<Index> filtered;
    filtered.reserve(selected_indices.size());
    for (const Index row_index : selected_indices)
    {
        const float value = matrix(row_index, column_index);
        if (isfinite(value) && value >= (minimum - 1e-6f) && value <= (maximum + 1e-6f))
            filtered.push_back(row_index);
    }
    return filtered;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
