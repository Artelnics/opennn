//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   O P E R A T I O N S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensor_operations.h"
#include "device_backend.h"
#include "operator.h"
#include "random_utilities.h"
#include "profiler.h"
#include "kernel.cuh"

#include <atomic>

#ifdef EIGEN_USE_MKL_ALL
#include <mkl_cblas.h>
#include <mkl_vml.h>
#endif

namespace opennn
{

#ifdef EIGEN_USE_MKL_ALL

static void add_bias(TensorView& output, const TensorView& bias, Index rows, Index columns, bool fuse_relu)
{
    float* y = output.as<float>();
    const float* b = bias.as<float>();

    if (!fuse_relu && columns > 1)
    {
        static thread_local vector<float> ones;
        if (ssize(ones) < rows) ones.assign(size_t(rows), 1.0f);
        cblas_sger(CblasRowMajor,
                   to_int(rows),
                   to_int(columns),
                   1.0f,
                   ones.data(),
                   1,
                   b,
                   1,
                   y,
                   to_int(columns));
        return;
    }

    const bool parallel_bias = rows * columns >= 65536;

    #pragma omp parallel for schedule(static) if(parallel_bias)
    for (Index i = 0; i < rows; ++i)
    {
        float* row = y + i * columns;
        for (Index j = 0; j < columns; ++j)
        {
            const float value = row[j] + b[j];
            row[j] = fuse_relu ? max(value, 0.0f) : value;
        }
    }
}

static bool try_activation_forward(TensorView& output, ActivationFunction function)
{
    if (function != ActivationFunction::Tanh || !output.is_fp32()) return false;

    float* values = output.as<float>();
    const int size = to_int(output.size());

    vsTanh(size, values, values);

    return true;
}

static bool try_linear_forward(const TensorView& input,
                                const TensorView& weights,
                                const TensorView& bias,
                                TensorView& output,
                                bool fuse_relu)
{
    if (!input.is_fp32()
        || !weights.is_fp32()
        || !bias.is_fp32()
        || !output.is_fp32()
        || input.shape.rank == 0
        || weights.shape.rank != 2
        || bias.shape.rank != 1)
        return false;

    const Index input_columns = input.shape.back();
    const Index output_columns = weights.shape.back();

    if (input_columns <= 0
        || output_columns <= 0
        || input.size() % input_columns != 0
        || weights.shape[0] != input_columns
        || bias.size() != output_columns)
        return false;

    const Index rows = input.size() / input_columns;

    if (rows <= 0 || output.size() != rows * output_columns)
        return false;

    const int m = to_int(rows);
    const int n = to_int(output_columns);
    const int k = to_int(input_columns);

    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                m,
                n,
                k,
                1.0f,
                input.as<float>(),
                k,
                weights.as<float>(),
                n,
                0.0f,
                output.as<float>(),
                n);

    add_bias(output, bias, rows, output_columns, fuse_relu);
    return true;
}

#else

static bool try_activation_forward(TensorView&, ActivationFunction)  { return false; }
static bool try_linear_forward(const TensorView&, const TensorView&,
                               const TensorView&, TensorView&, bool) { return false; }

#endif

const EnumMap<ActivationFunction>& activation_function_map()
{
    static const vector<pair<ActivationFunction, string>> entries = {
        {ActivationFunction::Identity,  "Identity"},
        {ActivationFunction::Sigmoid,   "Sigmoid"},
        {ActivationFunction::Tanh,      "Tanh"},
        {ActivationFunction::ReLU,      "ReLU"},
        {ActivationFunction::Softmax,   "Softmax"},
        {ActivationFunction::LeakyReLU, "LeakyReLU"},
        {ActivationFunction::GELU,      "GELU"},
        {ActivationFunction::GELUTanh,  "GELUTanh"},
        {ActivationFunction::SiLU,      "SiLU"},
        {ActivationFunction::SiLU,      "Swish"},

        {ActivationFunction::Identity,  "Linear"},
        {ActivationFunction::Sigmoid,   "Logistic"},
        {ActivationFunction::Tanh,      "HyperbolicTangent"},
        {ActivationFunction::ReLU,      "RectifiedLinear"},
        {ActivationFunction::ReLU,      "ScaledExponentialLinear"}
    };

    static const EnumMap<ActivationFunction> instance{entries};
    return instance;
}

bool activation_needs_input(ActivationFunction function)
{
    return is_one_of(function, ActivationFunction::GELU,
                     ActivationFunction::GELUTanh, ActivationFunction::SiLU);
}

const string& activation_function_to_string(ActivationFunction function)
{
    return activation_function_map().to_string(function);
}

ActivationFunction activation_function_from_string(const string& name)
{
    return activation_function_map().from_string(name);
}

#define OPENNN_GPU_OPS(X) \
    X(bound_gpu, (const TensorView&, const TensorView&, const TensorView&, TensorView&)) \
    X(scale_gpu, (const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, float, float, TensorView&, bool)) \
    X(copy_gpu, (const TensorView&, TensorView&)) \
    X(add_gpu, (const TensorView&, const TensorView&, TensorView&)) \
    X(multiply_gpu, (const TensorView&, bool, const TensorView&, bool, TensorView&, float, float)) \
    X(softmax_gpu, (TensorView&)) \
    X(activation_forward_gpu, (TensorView&, ActivationFunction)) \
    X(activation_backward_gpu, (const TensorView&, TensorView&, ActivationFunction)) \
    X(dropout_forward_gpu, (TensorView&, Buffer&, float)) \
    X(dropout_backward_gpu, (TensorView&, const Buffer&, float)) \
    X(linear_forward_gpu, (const TensorView&, const TensorView&, const TensorView&, TensorView&, cublasLtEpilogue_t, TensorView*, const TensorView&)) \
    X(linear_backward_gpu, (const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, TensorView&, bool, const TensorView*)) \
    X(layer_normalization_forward_gpu, (const TensorView&, const TensorView&, const TensorView&, TensorView&, TensorView&, TensorView&)) \
    X(layer_normalization_backward_gpu, (const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, TensorView&)) \
    X(rms_normalization_forward_gpu, (const TensorView&, const TensorView&, TensorView&, TensorView&, float)) \
    X(rms_normalization_backward_gpu, (const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, TensorView&)) \
    X(rope_forward_gpu, (const TensorView&, const TensorView&, const TensorView&, TensorView&, Index, Index, Index)) \
    X(rope_backward_gpu, (const TensorView&, const TensorView&, const TensorView&, TensorView&, Index, Index, Index)) \
    X(swiglu_forward_gpu, (const TensorView&, const TensorView&, TensorView&)) \
    X(swiglu_backward_gpu, (const TensorView&, const TensorView&, const TensorView&, TensorView&, TensorView&)) \
    X(grouped_attention_gpu, (const TensorView&, const TensorView&, const TensorView&, TensorView&, Index, Index, Index, bool, float, Index, float*, const int*)) \
    X(qk_norm_gpu, (const TensorView&, const TensorView&, TensorView&, Index, float)) \
    X(embedding_lookup_forward_gpu, (const TensorView&, const TensorView&, const TensorView&, TensorView&, Index, Index, Index, bool, bool, const TensorView&)) \
    X(embedding_lookup_backward_gpu, (const TensorView&, const TensorView&, const TensorView&, const TensorView&, Index, Index, Index, bool)) \
    X(max_pooling_3d_forward_gpu, (const TensorView&, TensorView&, TensorView&, bool)) \
    X(average_pooling_3d_forward_gpu, (const TensorView&, TensorView&)) \
    X(max_pooling_3d_backward_gpu, (const TensorView&, const TensorView&, TensorView&)) \
    X(average_pooling_3d_backward_gpu, (const TensorView&, const TensorView&, TensorView&)) \
    X(first_token_3d_forward_gpu, (const TensorView&, TensorView&)) \
    X(first_token_3d_backward_gpu, (const TensorView&, TensorView&)) \
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
               TensorView& output, bool inverse)
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
            if (!inverse)
            {
                const float range = maximums_vector(feature_index) - minimums_vector(feature_index);
                if (range < EPSILON)
                    column.setZero();
                else
                    column = (column - minimums_vector(feature_index)) / range
                           * (max_range - min_range) + min_range;
            }
            else
            {
                throw_if(max_range - min_range < EPSILON, "The range values are not valid.");
                column = (column - min_range) / (max_range - min_range)
                       * (maximums_vector(feature_index) - minimums_vector(feature_index)) + minimums_vector(feature_index);
            }
            break;
        case 2:
            if (!inverse)
            {
                const float sd = standard_deviations_vector(feature_index);
                if (sd > EPSILON)
                    column = (column - means_vector(feature_index)) / sd;
                else
                    column.setZero();
            }
            else
                column = means_vector(feature_index) + column * standard_deviations_vector(feature_index);
            break;
        case 3:
            if (!inverse)
            {
                const float sd = standard_deviations_vector(feature_index);
                column *= (sd > EPSILON) ? (1.0f / sd) : 0.0f;
            }
            else
            {
                const float sd = standard_deviations_vector(feature_index);
                column *= (abs(sd) < EPSILON) ? 1.0f : sd;
            }
            break;
        case 4:
            if (inverse) column = column.exp();
            else         column = column.max(EPSILON).log();
            break;
        case 5:
            if (inverse) column *= 255.0f;
            else         column /= 255.0f;
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
                  min_range, max_range, output, false);
        return;
    }
    scale_cpu(input, minimums, maximums, means, standard_deviations, scalers,
              min_range, max_range, output, false);
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
        scale_gpu(input, minimums, maximums, means, standard_deviations, scalers,
                  min_range, max_range, output, true);
        return;
    }

    scale_cpu(input, minimums, maximums, means, standard_deviations, scalers,
              min_range, max_range, output, true);
}

void copy(const TensorView& source, TensorView& destination)
{
    throw_if(source.size() != destination.size(),
             "Tensor sizes mismatch in copy operation.");
    throw_if(source.type != destination.type,
             "Tensor dtypes mismatch in copy operation.");

    if (source.is_cuda()) { copy_gpu(source, destination); return; }
    memcpy(destination.data, source.data, source.byte_size());
}

void add(const TensorView& input_1,
         const TensorView& input_2,
         TensorView& output)
{
    throw_if(input_1.size() != input_2.size() || input_1.size() != output.size(),
             "Tensor dimensions do not match.");

    if (input_1.is_cuda()) { add_gpu(input_1, input_2, output); return; }
    output.as_vector().noalias() = input_1.as_vector() + input_2.as_vector();
}

static void multiply_cpu(const TensorView& input_a, bool transpose_a,
                  const TensorView& input_b, bool transpose_b,
                  TensorView& output,
                  float alpha, float beta)
{
    const size_t rank = input_a.get_rank();
    const Index batch_count = input_a.size() / (input_a.shape[rank - 2] * input_a.shape[rank - 1]);

    const bool parallel = output.size() >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
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

    const bool parallel = output_matrix.size() >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
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
    if (try_activation_forward(output, function)) return;

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
    case LeakyReLU:
        a = (a >= 0.0f).select(a, a * LEAKY_RELU_SLOPE);
        return;
    case GELU:
        a = a.unaryExpr([](float x) { return gelu_value(x); });
        return;
    case GELUTanh:

        a = 0.5f * a * (1.0f + (SQRT_2_OVER_PI * (a + GELU_TANH_CUBIC * a * a * a)).tanh());
        return;
    case SiLU:

        a = a / (1.0f + (-a).exp());
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
    case LeakyReLU:
        d = (y >= 0.0f).select(d, d * LEAKY_RELU_SLOPE);
        return;

    case GELU:
        d *= y.unaryExpr([](float x) { return gelu_derivative(x); });
        return;
    case GELUTanh:
        d *= y.unaryExpr([](float x) { return gelu_tanh_derivative(x); });
        return;
    case SiLU:
        d *= y.unaryExpr([](float x) { return silu_derivative(x); });
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
    if (is_one_of(function, ActivationFunction::Identity, ActivationFunction::Softmax)
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

    const bool parallel = element_count >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index i = 0; i < element_count; ++i)
    {
        const float keep_value = mask_values[i] < rate ? 0.0f : keep_scale;
        mask_values[i] = keep_value;
        output_data[i] *= keep_value;
    }
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
    Map<const VectorR, AlignedMax> mask_view(mask.as<float>(), delta.size());
    delta.as_vector().array() *= mask_view.array();
}

static void linear_forward_cpu(const TensorView& input, const TensorView& weights, const TensorView& bias,
                        TensorView& output, cublasLtEpilogue_t epilogue)
{
    const bool fuse_relu = epilogue == CUBLASLT_EPILOGUE_RELU_BIAS;

    if (try_linear_forward(input, weights, bias, output, fuse_relu)) return;

    auto output_matrix = output.as_flat_matrix();
    output_matrix.noalias() = input.as_flat_matrix() * weights.as_matrix();
    if (!bias.empty())
        output_matrix.rowwise() += bias.as_vector().transpose();

    if (fuse_relu)
        output.as_vector().array() = output.as_vector().array().cwiseMax(0.0f);
}

static void linear_backward_cpu(const TensorView& output_delta, const TensorView& input, const TensorView& weights,
                         const TensorView& weight_gradient, const TensorView& bias_gradient,
                         TensorView& input_delta, bool accumulate)
{
    weight_gradient.as_matrix().noalias() = input.as_flat_matrix().transpose() * output_delta.as_flat_matrix();
    if (!bias_gradient.empty())
        bias_gradient.as_vector().noalias() = output_delta.as_flat_matrix().colwise().sum();

    if (!input_delta.data || input_delta.empty()) return;

    auto input_delta_mat = input_delta.as_flat_matrix();
    const auto product   = output_delta.as_flat_matrix() * weights.as_matrix().transpose();

    if (accumulate) input_delta_mat.noalias() += product;
    else            input_delta_mat.noalias()  = product;
}

void linear_forward(const TensorView& input, const TensorView& weights, const TensorView& bias,
                    TensorView& output, cublasLtEpilogue_t epilogue, TensorView* pre_activation,
                    const TensorView& weight_scale)
{
    if (input.is_cuda()) { linear_forward_gpu(input, weights, bias, output, epilogue, pre_activation, weight_scale); return; }

    throw_if(weights.is_int8(), "linear_forward: INT8 weights are CUDA-only.");

    throw_if(epilogue == CUBLASLT_EPILOGUE_GELU_AUX_BIAS,
             "linear_forward: the GELU_AUX_BIAS epilogue is CUDA-only.");

    linear_forward_cpu(input, weights, bias, output, epilogue);
}

void linear_backward(const TensorView& output_delta, const TensorView& input, const TensorView& weights,
                     const TensorView& weight_gradient, const TensorView& bias_gradient,
                     TensorView& input_delta, bool accumulate_input_delta,
                     const TensorView* drelu_mask)
{
    throw_if(drelu_mask && (!output_delta.is_cuda() || output_delta.type == Type::BF16
                            || accumulate_input_delta),
             "linear_backward: the DRELU fused input-delta path is CUDA fp32, non-accumulating only.");

    if (output_delta.is_cuda())
    {
        linear_backward_gpu(output_delta, input, weights, weight_gradient, bias_gradient,
                            input_delta, accumulate_input_delta, drelu_mask);
        return;
    }
    linear_backward_cpu(output_delta, input, weights, weight_gradient, bias_gradient,
                        input_delta, accumulate_input_delta);
}

static void layer_normalization_forward_cpu(const TensorView& input, const TensorView& gamma, const TensorView& beta,
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

    #pragma omp parallel for schedule(static)
    for (Index row = 0; row < total_rows; ++row)
    {
        const float* input_row = input_data + row * embedding_dimension;
        float* norm_row        = normalized_data + row * embedding_dimension;
        float* out_row         = output_data + row * embedding_dimension;

        const Map<const Array<float, Dynamic, 1>> input_map(input_row, embedding_dimension);
        const float sum    = input_map.sum();
        const float sum_sq = input_map.square().sum();

        const float mean    = sum * inv_D;

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

static void layer_normalization_backward_cpu(const TensorView& output_delta,
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

    #pragma omp parallel for schedule(static)
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

        input_delta_map = gamma_map * output_delta_map;

        const float sum_scaled_gradient      = input_delta_map.sum() * inv_D;
        const float sum_scaled_gradient_norm = (input_delta_map * norm_map).sum() * inv_D;

        input_delta_map = (input_delta_map - sum_scaled_gradient
                          - norm_map * sum_scaled_gradient_norm) * inv_std;
    }
}

void layer_normalization_forward(const TensorView& input, const TensorView& gamma, const TensorView& beta,
                        TensorView& means, TensorView& standard_deviations,
                        TensorView& normalized, TensorView& output)
{
    if (input.is_cuda()) { layer_normalization_forward_gpu(input, gamma, beta, means, standard_deviations, output); return; }
    layer_normalization_forward_cpu(input, gamma, beta, means, standard_deviations, normalized, output);
}

void layer_normalization_add_forward(const TensorView& input, const TensorView& residual,
                            const TensorView& gamma, const TensorView& beta,
                            TensorView& means, TensorView& standard_deviations,
                            TensorView& normalized, TensorView& sum, TensorView& output)
{
#ifdef OPENNN_HAS_CUDA
    if (input.is_cuda())
    {
        const int rows = to_int(input.size() / input.shape.back());
        const int cols = to_int(input.shape.back());
        output.dispatch([&](auto tag) {
            using T = decltype(tag);
            layernorm_add_forward_cuda<T>(rows, cols,
                                          input.as<T>(), residual.as<T>(),
                                          sum.as<T>(), output.as<T>(),
                                          means.as<float>(), standard_deviations.as<float>(),
                                          gamma.as<float>(), beta.as<float>(), EPSILON);
        });
        return;
    }
#endif
    add(input, residual, sum);
    layer_normalization_forward_cpu(sum, gamma, beta, means, standard_deviations, normalized, output);
}

void layer_normalization_backward(const TensorView& input, const TensorView& output_delta,
                         const TensorView& means, const TensorView& standard_deviations,
                         const TensorView& normalized, const TensorView& gamma,
                         const TensorView& gamma_gradient, const TensorView& beta_gradient,
                         TensorView& input_delta)
{
    if (input.is_cuda())
    {
        layer_normalization_backward_gpu(input, output_delta, means, standard_deviations, gamma,
                                gamma_gradient, beta_gradient, input_delta);
        return;
    }
    layer_normalization_backward_cpu(output_delta, standard_deviations, normalized, gamma,
                            gamma_gradient, beta_gradient, input_delta);
}

static void rms_normalization_forward_cpu(const TensorView& input, const TensorView& weight,
                            TensorView& inverse_rms, TensorView& normalized, TensorView& output,
                            float epsilon)
{
    const Index embedding_dimension = input.shape.back();
    const Index total_rows = input.size() / embedding_dimension;
    const float inv_D = 1.0f / to_type(embedding_dimension);

    const float* input_data      = input.as<float>();
    float* inverse_rms_data      = inverse_rms.as<float>();
    float* normalized_data       = normalized.as<float>();
    float* output_data           = output.as<float>();
    const float* weight_data     = weight.as<float>();

    #pragma omp parallel for schedule(static)
    for (Index row = 0; row < total_rows; ++row)
    {
        const float* input_row = input_data + row * embedding_dimension;
        float* norm_row        = normalized_data + row * embedding_dimension;
        float* out_row         = output_data + row * embedding_dimension;

        const Map<const Array<float, Dynamic, 1>> input_map(input_row, embedding_dimension);

        const float mean_square = input_map.square().sum() * inv_D;
        const float inverse     = 1.0f / sqrt(mean_square + epsilon);

        inverse_rms_data[row] = inverse;

        for (Index dim = 0; dim < embedding_dimension; ++dim)
        {
            const float x_hat = input_row[dim] * inverse;
            norm_row[dim] = x_hat;
            out_row[dim]  = weight_data[dim] * x_hat;
        }
    }
}

static void rms_normalization_backward_cpu(const TensorView& output_delta,
                             const TensorView& inverse_rms,
                             const TensorView& normalized,
                             const TensorView& weight,
                             const TensorView& weight_gradient,
                             TensorView& input_delta)
{
    const Index embedding_dimension = output_delta.shape.back();
    const Index total_rows = output_delta.size() / embedding_dimension;
    const float inv_D = 1.0f / to_type(embedding_dimension);

    const MatrixMap output_delta_flat = output_delta.as_flat_matrix();
    const MatrixMap norm_flat         = normalized.as_flat_matrix();

    weight_gradient.as_vector().noalias() = (output_delta_flat.array() * norm_flat.array()).matrix().colwise().sum();

    if (input_delta.empty()) return;

    const float* output_delta_data = output_delta.as<float>();
    const float* norm_data         = normalized.as<float>();
    const float* inverse_rms_data  = inverse_rms.as<float>();
    const float* weight_data       = weight.as<float>();
    float* input_delta_data        = input_delta.as<float>();

    #pragma omp parallel for schedule(static)
    for (Index row = 0; row < total_rows; ++row)
    {
        const float* output_delta_row = output_delta_data + row * embedding_dimension;
        const float* norm_row         = norm_data + row * embedding_dimension;
        float* input_delta_row        = input_delta_data + row * embedding_dimension;
        const float inverse = inverse_rms_data[row];

        const Map<const Array<float, Dynamic, 1>> weight_map(weight_data, embedding_dimension);
        const Map<const Array<float, Dynamic, 1>> output_delta_map(output_delta_row, embedding_dimension);
        const Map<const Array<float, Dynamic, 1>> norm_map(norm_row, embedding_dimension);
        Map<Array<float, Dynamic, 1>> input_delta_map(input_delta_row, embedding_dimension);

        input_delta_map = weight_map * output_delta_map;

        const float mean_d_norm = (input_delta_map * norm_map).sum() * inv_D;

        input_delta_map = (input_delta_map - norm_map * mean_d_norm) * inverse;
    }
}

void rms_normalization_forward(const TensorView& input, const TensorView& weight,
                      TensorView& inverse_rms, TensorView& normalized, TensorView& output,
                      float epsilon)
{
    if (input.is_cuda()) { rms_normalization_forward_gpu(input, weight, inverse_rms, output, epsilon); return; }
    rms_normalization_forward_cpu(input, weight, inverse_rms, normalized, output, epsilon);
}

void rms_normalization_backward(const TensorView& input, const TensorView& output_delta,
                       const TensorView& inverse_rms, const TensorView& normalized,
                       const TensorView& weight, const TensorView& weight_gradient,
                       TensorView& input_delta)
{
    if (input.is_cuda())
    {
        rms_normalization_backward_gpu(input, output_delta, inverse_rms, weight,
                              weight_gradient, input_delta);
        return;
    }
    rms_normalization_backward_cpu(output_delta, inverse_rms, normalized, weight,
                          weight_gradient, input_delta);
}

void rotary_build_tables(TensorView& cos_table, TensorView& sin_table,
                         Index sequence_length, Index rotary_dim, float base)
{
    float* cos_data = cos_table.as<float>();
    float* sin_data = sin_table.as<float>();
    const Index half = rotary_dim / 2;

    #pragma omp parallel for schedule(static)
    for (Index pos = 0; pos < sequence_length; ++pos)
        for (Index i = 0; i < half; ++i)
        {
            const float inv_freq = 1.0f / powf(base, (2.0f * float(i)) / float(rotary_dim));
            const float angle    = float(pos) * inv_freq;
            const float c = cosf(angle);
            const float s = sinf(angle);
            cos_data[pos * rotary_dim + i]        = c;
            cos_data[pos * rotary_dim + i + half] = c;
            sin_data[pos * rotary_dim + i]        = s;
            sin_data[pos * rotary_dim + i + half] = s;
        }
}

static void rotary_forward_cpu(const TensorView& input, const TensorView& cos_table, const TensorView& sin_table,
                        TensorView& output, Index head_dim, Index rotary_dim, Index position_offset)
{
    const Index seq       = input.shape[1];
    const Index model_dim = input.shape.back();
    const Index num_heads = model_dim / head_dim;
    const Index rows      = input.size() / model_dim;
    const Index half      = rotary_dim / 2;

    const float* in       = input.as<float>();
    float* out            = output.as<float>();
    const float* cos_data = cos_table.as<float>();
    const float* sin_data = sin_table.as<float>();

    const bool parallel = rows * model_dim >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index row = 0; row < rows; ++row)
    {
        const Index pos = (row % seq) + position_offset;
        const float* cr = cos_data + pos * rotary_dim;
        const float* sr = sin_data + pos * rotary_dim;

        for (Index h = 0; h < num_heads; ++h)
        {
            const Index base = row * model_dim + h * head_dim;

            for (Index j = 0; j < rotary_dim; ++j)
            {
                const float rotated = (j < half) ? -in[base + j + half] : in[base + j - half];
                out[base + j] = in[base + j] * cr[j] + rotated * sr[j];
            }
            for (Index j = rotary_dim; j < head_dim; ++j)
                out[base + j] = in[base + j];
        }
    }
}

static void rotary_backward_cpu(const TensorView& output_delta, const TensorView& cos_table, const TensorView& sin_table,
                         TensorView& input_delta, Index head_dim, Index rotary_dim, Index position_offset)
{
    const Index seq       = output_delta.shape[1];
    const Index model_dim = output_delta.shape.back();
    const Index num_heads = model_dim / head_dim;
    const Index rows      = output_delta.size() / model_dim;
    const Index half      = rotary_dim / 2;

    const float* dout     = output_delta.as<float>();
    float* din            = input_delta.as<float>();
    const float* cos_data = cos_table.as<float>();
    const float* sin_data = sin_table.as<float>();

    const bool parallel = rows * model_dim >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index row = 0; row < rows; ++row)
    {
        const Index pos = (row % seq) + position_offset;
        const float* cr = cos_data + pos * rotary_dim;
        const float* sr = sin_data + pos * rotary_dim;

        for (Index h = 0; h < num_heads; ++h)
        {
            const Index base = row * model_dim + h * head_dim;

            for (Index j = 0; j < rotary_dim; ++j)
            {
                const float rotated = (j < half) ? -dout[base + j + half] : dout[base + j - half];
                din[base + j] = dout[base + j] * cr[j] - rotated * sr[j];
            }
            for (Index j = rotary_dim; j < head_dim; ++j)
                din[base + j] = dout[base + j];
        }
    }
}

void rotary_forward(const TensorView& input, const TensorView& cos_table, const TensorView& sin_table,
                    TensorView& output, Index head_dim, Index rotary_dim, Index position_offset)
{
    if (input.is_cuda()) { rope_forward_gpu(input, cos_table, sin_table, output, head_dim, rotary_dim, position_offset); return; }
    rotary_forward_cpu(input, cos_table, sin_table, output, head_dim, rotary_dim, position_offset);
}

void rotary_backward(const TensorView& output_delta, const TensorView& cos_table, const TensorView& sin_table,
                     TensorView& input_delta, Index head_dim, Index rotary_dim, Index position_offset)
{
    if (output_delta.is_cuda()) { rope_backward_gpu(output_delta, cos_table, sin_table, input_delta, head_dim, rotary_dim, position_offset); return; }
    rotary_backward_cpu(output_delta, cos_table, sin_table, input_delta, head_dim, rotary_dim, position_offset);
}

static void swiglu_forward_cpu(const TensorView& gate, const TensorView& up, TensorView& output)
{
    const Index n = gate.size();
    const float* g = gate.as<float>();
    const float* u = up.as<float>();
    float* o       = output.as<float>();

    const bool parallel = n >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index i = 0; i < n; ++i)
    {
        const float gi = g[i];
        const float silu = gi / (1.0f + expf(-gi));
        o[i] = silu * u[i];
    }
}

static void swiglu_backward_cpu(const TensorView& output_delta, const TensorView& gate, const TensorView& up,
                         TensorView& gate_delta, TensorView& up_delta)
{
    const Index n = output_delta.size();
    const float* d = output_delta.as<float>();
    const float* g = gate.as<float>();
    const float* u = up.as<float>();
    float* dg = gate_delta.empty() ? nullptr : gate_delta.as<float>();
    float* du = up_delta.empty()   ? nullptr : up_delta.as<float>();

    const bool parallel = n >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index i = 0; i < n; ++i)
    {
        const float gi  = g[i];
        const float sig = 1.0f / (1.0f + expf(-gi));
        const float silu = gi * sig;
        if (du) du[i] = d[i] * silu;

        if (dg) dg[i] = d[i] * u[i] * sig * (1.0f + gi * (1.0f - sig));
    }
}

void swiglu_forward(const TensorView& gate, const TensorView& up, TensorView& output)
{
    if (gate.is_cuda()) { swiglu_forward_gpu(gate, up, output); return; }
    swiglu_forward_cpu(gate, up, output);
}

void swiglu_backward(const TensorView& output_delta, const TensorView& gate, const TensorView& up,
                     TensorView& gate_delta, TensorView& up_delta)
{
    if (output_delta.is_cuda()) { swiglu_backward_gpu(output_delta, gate, up, gate_delta, up_delta); return; }
    swiglu_backward_cpu(output_delta, gate, up, gate_delta, up_delta);
}

void grouped_attention_forward(const TensorView& query, const TensorView& key, const TensorView& value,
                               TensorView& output, Index n_query_heads, Index n_kv_heads, Index head_dim,
                               bool causal, float scale, Index query_position_offset,
                               float* decode_partials, const int* position_device)
{
    if (query.is_cuda()) {
        grouped_attention_gpu(query, key, value, output, n_query_heads, n_kv_heads, head_dim,
                              causal, scale, query_position_offset, decode_partials, position_device);
        return;
    }

    const Index batch     = query.shape[0];
    const Index query_seq = query.shape[1];
    const Index key_seq   = key.shape[1];
    const Index group     = n_query_heads / n_kv_heads;

    const float* Q = query.as<float>();
    const float* K = key.as<float>();
    const float* V = value.as<float>();
    float* O       = output.as<float>();

    auto q_off = [&](Index b, Index t, Index h) { return ((b * query_seq + t) * n_query_heads + h) * head_dim; };
    auto kv_off = [&](Index b, Index t, Index h) { return ((b * key_seq + t) * n_kv_heads + h) * head_dim; };

    #pragma omp parallel for collapse(2) schedule(static)
    for (Index b = 0; b < batch; ++b)
        for (Index hq = 0; hq < n_query_heads; ++hq)
        {
            const Index hkv = hq / group;

            thread_local vector<float> scores;
            if (scores.size() < size_t(key_seq)) scores.resize(size_t(key_seq));

            for (Index i = 0; i < query_seq; ++i)
            {

                const Index valid = causal ? min(query_position_offset + i + 1, key_seq) : key_seq;
                const float* q_vec = Q + q_off(b, i, hq);

                float max_score = -numeric_limits<float>::infinity();
                for (Index j = 0; j < valid; ++j)
                {
                    const float* k_vec = K + kv_off(b, j, hkv);
                    float dot = 0.0f;
                    for (Index d = 0; d < head_dim; ++d) dot += q_vec[d] * k_vec[d];
                    dot *= scale;
                    scores[size_t(j)] = dot;
                    max_score = max(max_score, dot);
                }

                float sum = 0.0f;
                for (Index j = 0; j < valid; ++j)
                {
                    const float e = expf(scores[size_t(j)] - max_score);
                    scores[size_t(j)] = e;
                    sum += e;
                }
                const float inv_sum = 1.0f / sum;

                float* o_vec = O + q_off(b, i, hq);
                for (Index d = 0; d < head_dim; ++d) o_vec[d] = 0.0f;
                for (Index j = 0; j < valid; ++j)
                {
                    const float p = scores[size_t(j)] * inv_sum;
                    const float* v_vec = V + kv_off(b, j, hkv);
                    for (Index d = 0; d < head_dim; ++d) o_vec[d] += p * v_vec[d];
                }
            }
        }
}

void qk_norm_forward(const TensorView& input, const TensorView& weight, TensorView& output,
                     Index head_dim, float epsilon)
{
    if (input.is_cuda()) { qk_norm_gpu(input, weight, output, head_dim, epsilon); return; }

    const Index rows  = input.size() / head_dim;
    const float inv_D = 1.0f / to_type(head_dim);

    const float* x = input.as<float>();
    float* o       = output.as<float>();
    const float* w = weight.as<float>();

    #pragma omp parallel for schedule(static)
    for (Index r = 0; r < rows; ++r)
    {
        const float* x_row = x + r * head_dim;
        float* o_row       = o + r * head_dim;

        const Map<const Array<float, Dynamic, 1>> x_map(x_row, head_dim);
        const float mean_square = x_map.square().sum() * inv_D;
        const float inverse     = 1.0f / sqrt(mean_square + epsilon);

        for (Index d = 0; d < head_dim; ++d)
            o_row[d] = w[d] * x_row[d] * inverse;
    }
}

#ifdef OPENNN_HAS_CUDA

constexpr Index int8_dequant_budget_bytes = Index(32) * 1024 * 1024;

static void w8a16_linear_rows(Index rows, Index in_features, Index out_features,
                              bool weights_out_major,
                              const bfloat16* x, const int8_t* weights, const float* scales,
                              const bfloat16* bias, bfloat16* y)
{
    for (Index row = 0; row < rows; row += W8A16_MAX_M)
        w8a16_linear_cuda<bfloat16>(to_int(min(Index(W8A16_MAX_M), rows - row)),
                                    to_int(in_features), to_int(out_features), weights_out_major,
                                    x + row * in_features, weights, scales, bias,
                                    y + row * out_features);
}

#endif

void tied_lm_head_forward(const TensorView& input, const TensorView& embed_weight, TensorView& output,
                          const TensorView& weight_scale)
{
#ifdef OPENNN_HAS_CUDA
    if (input.is_cuda() && embed_weight.is_int8())
    {
        throw_if(weight_scale.empty() || !input.is_bf16() || !output.is_bf16(),
                 "tied_lm_head_forward: INT8 weights require BF16 activations and a per-channel scale vector.");

        const Index in_features  = embed_weight.shape.back();
        const Index out_features = embed_weight.size() / in_features;
        const Index rows = input.size() / in_features;

        if (rows <= W8A16_MAX_M)
        {
            w8a16_linear_rows(rows, in_features, out_features, true,
                              input.as<bfloat16>(), embed_weight.as<int8_t>(),
                              weight_scale.as<float>(), nullptr, output.as<bfloat16>());
            return;
        }

        const Index tile_rows = min(out_features,
            max(Index(1), int8_dequant_budget_bytes / (in_features * Index(sizeof(bfloat16)))));
        bfloat16* dequantized = ensure_int8_dequant_workspace(tile_rows * in_features);

        for (Index j0 = 0; j0 < out_features; j0 += tile_rows)
        {
            const Index tile = min(tile_rows, out_features - j0);
            w8_dequant_cuda<bfloat16>(tile, in_features, true,
                                      embed_weight.as<int8_t>() + j0 * in_features,
                                      weight_scale.as<float>() + j0, dequantized);
            gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
                                      to_int(tile), to_int(rows), to_int(in_features),
                                      dequantized, CUDA_R_16BF, to_int(in_features), 0,
                                      input.data, CUDA_R_16BF, to_int(in_features), 0,
                                      output.as<bfloat16>() + j0, CUDA_R_16BF, to_int(out_features), 0,
                                      1);
        }
        return;
    }
#endif

    if (input.is_cuda()) { multiply(input, false, embed_weight, true, output, 1.0f, 0.0f); return; }
    output.as_flat_matrix().noalias() =
        input.as_flat_matrix() * embed_weight.as_matrix().transpose();
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

    #pragma omp parallel for schedule(static)
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
                cerr << format("EmbeddingLookup warning: token id {} out of range [0, {}); zeroing row. Further warnings suppressed.\n", token_id, vocabulary_size);
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
                                   const TensorView& weight_gradient, const TensorView& positional_gradient,
                                   Index sequence_length, Index embedding_dimension, Index vocabulary_size,
                                   bool scale_embedding)
{
    const Index total_elements = indices.size();

    MatrixMap output_delta_map = output_delta.as_flat_matrix();
    MatrixMap weight_gradients = weight_gradient.as_matrix().setZero();
    const float scale = scale_embedding ? sqrt(to_type(embedding_dimension)) : 1.0f;

    const bool accumulate_positional = !positional_gradient.empty() && positional_gradient.data != nullptr;

    for (Index token_index = 0; token_index < total_elements; ++token_index)
    {
        const Index vocabulary_index = static_cast<Index>(indices.as<float>()[token_index]);

        if (vocabulary_index <= 0 || vocabulary_index >= vocabulary_size)
            continue;

        weight_gradients.row(vocabulary_index).noalias() += scale * output_delta_map.row(token_index);
    }

    if (accumulate_positional)
    {
        MatrixMap positional_gradients = positional_gradient.as_matrix();
        positional_gradients.setZero();
        for (Index token_index = 0; token_index < total_elements; ++token_index)
        {
            const Index vocabulary_index = static_cast<Index>(indices.as<float>()[token_index]);
            if (vocabulary_index <= 0 || vocabulary_index >= vocabulary_size)
                continue;
            positional_gradients.row(token_index % sequence_length).noalias() += output_delta_map.row(token_index);
        }
    }
}

void embedding_lookup_forward(const TensorView& indices, const TensorView& weights,
                              const TensorView& positional_encoding, TensorView& output,
                              Index sequence_length, Index embedding_dimension, Index vocabulary_size,
                              bool scale_embedding, bool add_positional_encoding,
                              const TensorView& weight_scale)
{
    if (output.is_cuda())
    {
        embedding_lookup_forward_gpu(indices, weights, positional_encoding, output,
                                     sequence_length, embedding_dimension, vocabulary_size,
                                     scale_embedding, add_positional_encoding, weight_scale);
        return;
    }
    throw_if(weights.is_int8(), "embedding_lookup_forward: INT8 weights are CUDA-only.");
    embedding_lookup_forward_cpu(indices, weights, positional_encoding, output,
                                 sequence_length, embedding_dimension, vocabulary_size,
                                 scale_embedding, add_positional_encoding);
}

void embedding_lookup_backward(const TensorView& indices, const TensorView& output_delta,
                               const TensorView& weight_gradient, const TensorView& positional_gradient,
                               Index sequence_length, Index embedding_dimension, Index vocabulary_size,
                               bool scale_embedding)
{
    if (output_delta.is_cuda())
    {
        embedding_lookup_backward_gpu(indices, output_delta, weight_gradient, positional_gradient,
                                      sequence_length, embedding_dimension, vocabulary_size, scale_embedding);
        return;
    }
    embedding_lookup_backward_cpu(indices, output_delta, weight_gradient, positional_gradient,
                                  sequence_length,
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
    #pragma omp parallel for collapse(2) schedule(static)
    for (Index b = 0; b < batch_size; ++b)
        for (Index c = 0; c < input_channels; ++c)
            for (Index out_row = 0; out_row < output_height; ++out_row)
            {
                const Index in_row_start = out_row * row_stride - padding_height;
                const Index pr_start = max(Index(0), -in_row_start);
                const Index pr_end   = min(pool_height, input_height - in_row_start);

                for (Index out_col = 0; out_col < output_width; ++out_col)
                {
                    const Index in_col_start = out_col * column_stride - padding_width;
                    const Index pc_start = max(Index(0), -in_col_start);
                    const Index pc_end   = min(pool_width, input_width - in_col_start);

                    visit(PoolWindow{b, c, out_row, out_col,
                                     in_row_start, pr_start, pr_end,
                                     in_col_start, pc_start, pc_end});
                }
            }
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

    if (max_pooling)
    {
        const bool write_indices = !maximal_indices.empty();
        TensorMap4 indices_map = write_indices ? maximal_indices.as_tensor<4>() : TensorMap4(nullptr, 0, 0, 0, 0);
        for_each_pool_window(batch_size, input_channels, input_height, input_width,
                             output_height, output_width, pool_height, pool_width,
                             row_stride, column_stride, padding_height, padding_width,
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
    for_each_pool_window(batch_size, input_channels, input_height, input_width,
                         output_height, output_width, pool_height, pool_width,
                         row_stride, column_stride, padding_height, padding_width,
        [&](const PoolWindow& window) {
            float sum = 0;
            for (Index pr = window.pr_start; pr < window.pr_end; ++pr)
                for (Index pc = window.pc_start; pc < window.pc_end; ++pc)
                    sum += inputs(window.batch, window.in_row_start + pr,
                                  window.in_col_start + pc, window.channel);
            outputs(window.batch, window.out_row, window.out_col, window.channel) = sum * inv_pool_size;
        });
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

        #pragma omp parallel for collapse(2) schedule(static)
        for (Index b = 0; b < batch_size; ++b)
            for (Index c = 0; c < input_channels; ++c)
                for (Index out_row = 0; out_row < output_height; ++out_row)
                {
                    const Index in_row_start = out_row * row_stride - padding_height;
                    for (Index out_col = 0; out_col < output_width; ++out_col)
                    {
                        const Index in_col_start = out_col * column_stride - padding_width;
                        const Index argmax = static_cast<Index>(max_indices(b, out_row, out_col, c));
                        const Index in_row = in_row_start + argmax / pool_width;
                        const Index in_col = in_col_start + argmax % pool_width;
                        if (in_row < 0 || in_row >= input_height || in_col < 0 || in_col >= input_width)
                            continue;
                        input_deltas(b, in_row, in_col, c)
                            += output_deltas(b, out_row, out_col, c);
                    }
                }
        return;
    }

    const float inv_pool_size = 1.0f / (pool_height * pool_width);
    for_each_pool_window(batch_size, input_channels, input_height, input_width,
                         output_height, output_width, pool_height, pool_width,
                         row_stride, column_stride, padding_height, padding_width,
        [&](const PoolWindow& window) {
            const float avg_delta = output_deltas(window.batch, window.out_row, window.out_col, window.channel) * inv_pool_size;
            for (Index pr = window.pr_start; pr < window.pr_end; ++pr)
                for (Index pc = window.pc_start; pc < window.pc_end; ++pc)
                    input_deltas(window.batch, window.in_row_start + pr,
                                 window.in_col_start + pc, window.channel) += avg_delta;
        });
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

void compute_token_valid_lengths(const TensorView& indices, Index sequence_length, vector<Index>& valid_lengths)
{
    const Index total = indices.size();
    const Index batch_size = sequence_length > 0 ? total / sequence_length : 0;

    valid_lengths.assign(batch_size, sequence_length);
    if (batch_size == 0) return;

    const float* ids = indices.as<float>();
    vector<float> host;
#ifdef OPENNN_HAS_CUDA
    if (indices.is_cuda())
    {
        host.resize(size_t(total));
        copy_device_to_host_float(indices.data, indices.type, total, host.data(), Backend::get_compute_stream());
        ids = host.data();
    }
#endif

    for (Index b = 0; b < batch_size; ++b)
    {
        Index count = 0;
        const float* row = ids + b * sequence_length;
        for (Index s = 0; s < sequence_length; ++s)
            if (static_cast<Index>(row[s]) != 0) ++count;
        valid_lengths[b] = count;
    }
}

static void transpose_middle_axes(const float* src, float* dst,
                                  Index batch_size, Index src_m1, Index src_m2, Index D)
{
    #pragma omp parallel for collapse(3) schedule(static)
    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for (Index i = 0; i < src_m2; ++i)
            for (Index j = 0; j < src_m1; ++j)
                memcpy(dst + ((batch_index * src_m2 + i) * src_m1 + j) * D,
                       src + ((batch_index * src_m1 + j) * src_m2 + i) * D,
                       D * sizeof(float));
}

void split_heads(const TensorView& source, TensorView& destination)
{
    if (source.is_cuda()) { split_heads_gpu(source, destination); return; }
    transpose_middle_axes(source.as<float>(), destination.as<float>(),
                          source.shape[0], source.shape[1], source.shape[2], source.shape[3]);
}

void merge_heads(const TensorView& source, TensorView& destination)
{
    if (source.is_cuda()) { merge_heads_gpu(source, destination); return; }
    transpose_middle_axes(source.as<float>(), destination.as<float>(),
                          source.shape[0], source.shape[1], source.shape[2], source.shape[3]);
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
               TensorView& output, bool inverse)
{
    const Index features = scalers.size();

    visit_type_pair<Type::FP32, Type::BF16>(input.type, output.type, [&](auto in, auto out) {
        using TIn  = typename decltype(in)::type;
        using TOut = typename decltype(out)::type;
        if (inverse)
        {
            unscale_cuda<TIn, TOut>(output.size(), to_int(features),
                                    input.as<TIn>(),
                                    minimums.as_float(),
                                    maximums.as_float(),
                                    means.as_float(),
                                    standard_deviations.as_float(),
                                    scalers.as_float(),
                                    min_range, max_range,
                                    output.as<TOut>());
            return;
        }
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

    if (input_1.is_fp32() && input_2.is_fp32() && output.is_fp32())
    {
        add_relu_cuda(output.size(), input_1.as<float>(), input_2.as<float>(),
                       false, output.as<float>());
        return;
    }

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
    const int cols_a = to_int(input_a.shape[rank_a - 1]);
    const int rows_b = to_int(input_b.shape[rank_b - 2]);
    const int cols_b = to_int(input_b.shape[rank_b - 1]);

    if (rank_b == 2 && rank_a > 2)
    {
        rows_a = to_int(input_a.size() / cols_a);
    }

    const int cols_out = transpose_b ? rows_b : cols_b;
    const int rows_out = transpose_a ? cols_a : rows_a;
    const int inner_dim = transpose_a ? rows_a : cols_a;

    const cublasOperation_t operation_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t operation_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    const int batch_count = to_int(input_a.size() / (rows_a * cols_a));
    const long long stride_a = rows_a * cols_a;
    const long long stride_b = rows_b * cols_b;
    const long long stride_output = output.shape[output.get_rank() - 2] * output.shape[output.get_rank() - 1];

    gemm_strided_batched_cuda(operation_b, operation_a,
                              cols_out, rows_out, inner_dim,
                              input_b.data, input_b.cuda_dtype(), cols_b, stride_b,
                              input_a.data, input_a.cuda_dtype(), cols_a, stride_a,
                              output.data,  output.cuda_dtype(), cols_out, stride_output,
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
                        TensorView& output, cublasLtEpilogue_t epilogue, TensorView* pre_activation,
                        const TensorView& weight_scale)
{
    const int input_columns  = to_int(input.shape.back());
    const int output_columns = to_int(weights.shape.back());
    const int total_rows     = to_int(input.size() / input.shape.back());

    if (weights.is_int8())
    {
        throw_if(weight_scale.empty() || !input.is_bf16() || !output.is_bf16(),
                 "linear_forward: INT8 weights require BF16 activations and a per-channel scale vector.");

        const bool gemv_path = (total_rows <= W8A16_MAX_M
                                || weights.byte_size() > int8_dequant_budget_bytes)
            && (epilogue == CUBLASLT_EPILOGUE_DEFAULT || epilogue == CUBLASLT_EPILOGUE_BIAS)
            && (!bias.data || bias.is_bf16());

        if (gemv_path)
        {
            w8a16_linear_rows(total_rows, input_columns, output_columns, false,
                              input.as<bfloat16>(), weights.as<int8_t>(), weight_scale.as<float>(),
                              epilogue == CUBLASLT_EPILOGUE_BIAS && bias.data
                                  ? bias.as<bfloat16>() : nullptr,
                              output.as<bfloat16>());
            return;
        }

        bfloat16* dequantized = ensure_int8_dequant_workspace(weights.size());
        w8_dequant_cuda<bfloat16>(input_columns, output_columns, false, weights.as<int8_t>(),
                                  weight_scale.as<float>(), dequantized);
        const TensorView dequantized_weights(dequantized, weights.shape, Type::BF16, Device::CUDA);
        linear_forward_gpu(input, dequantized_weights, bias, output, epilogue, pre_activation, {});
        return;
    }

    const void* input_for_gemm = data_for_gemm_dtype(input, weights.type);
    const cudaDataType_t io_type = output.cuda_dtype();

    const void* bias_for_gemm = (bias.data && output.is_bf16() && bias.is_fp32())
        ? bias_for_gemm_bf16(bias)
        : bias.data;

    try
    {
        run_lt_matmul_cached(
            output_columns, total_rows, input_columns,
            CUBLAS_OP_N, CUBLAS_OP_N,
            epilogue,
            weights.data, input_for_gemm, output.data, bias_for_gemm,
            io_type, io_type,
            pre_activation ? pre_activation->data : nullptr);
    }
    catch (const runtime_error& e)
    {
        if (epilogue == CUBLASLT_EPILOGUE_GELU_AUX_BIAS && pre_activation)
        {
            linear_forward_gpu(input, weights, bias, *pre_activation, CUBLASLT_EPILOGUE_BIAS, nullptr, {});
            copy_gpu(*pre_activation, output);
            activation_forward_gpu(output, ActivationFunction::GELUTanh);
            return;
        }

        throw runtime_error(format("cuBLASLt GEMM {}x{}x{} ({}) failed: {}",
                                   output_columns, total_rows, input_columns,
                                   output.is_bf16() ? "bf16" : "fp32", e.what()));
    }
}

static void linear_backward_gpu(const TensorView& output_delta, const TensorView& input, const TensorView& weights,
                         const TensorView& weight_gradient, const TensorView& bias_gradient,
                         TensorView& input_delta, bool accumulate_input_delta,
                         const TensorView* drelu_mask)
{
    const int input_columns  = to_int(input.shape.back());
    const int output_columns = to_int(output_delta.shape.back());
    const int total_rows     = to_int(input.size() / input.shape.back());

    const void* input_for_gemm = data_for_gemm_dtype(input, weights.type);

    if (output_delta.type == Type::BF16)
    {

        bfloat16* dw_bf16 = ensure_bf16_gradient_workspace(weight_gradient.size());
        run_lt_matmul_cached(
            output_columns, input_columns, total_rows,
            CUBLAS_OP_N, CUBLAS_OP_T,
            CUBLASLT_EPILOGUE_DEFAULT,
            output_delta.data, input_for_gemm, dw_bf16, nullptr,
            output_delta.cuda_dtype(),
            CUDA_R_16BF);
        cast_bf16_to_fp32(weight_gradient.size(), dw_bf16, weight_gradient.as<float>());

        if (bias_gradient.size() > 0)
        {
            device::set_zero_async(bias_gradient.data, bias_gradient.size() * Index(sizeof(float)),
                                   Backend::get_compute_stream());
            bias_grad_sum_cuda<bfloat16>(total_rows, output_columns,
                                         output_delta.as<bfloat16>(), bias_gradient.as<float>());
        }
    }
    else
    {
        const bool has_bias = bias_gradient.size() > 0;
        run_lt_matmul_cached(
            output_columns, input_columns, total_rows,
            CUBLAS_OP_N, CUBLAS_OP_T,
            has_bias ? CUBLASLT_EPILOGUE_BGRADA : CUBLASLT_EPILOGUE_DEFAULT,
            output_delta.data, input_for_gemm, weight_gradient.data,
            has_bias ? bias_gradient.as<float>() : nullptr,
            output_delta.cuda_dtype(),
            CUDA_R_32F);
    }

    if (!input_delta.data || input_delta.empty()) return;

    if (drelu_mask)
    {

        run_lt_matmul_cached(
            input_columns, total_rows, output_columns,
            CUBLAS_OP_T, CUBLAS_OP_N,
            CUBLASLT_EPILOGUE_DRELU,
            weights.data, output_delta.data, input_delta.data, nullptr,
            output_delta.cuda_dtype(), input_delta.cuda_dtype(),
            drelu_mask->data);
        return;
    }

    multiply(output_delta, false, weights, true, input_delta, 1.0f,
             accumulate_input_delta ? 1.0f : 0.0f);
}

static void layer_normalization_forward_gpu(const TensorView& input, const TensorView& gamma, const TensorView& beta,
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

static void layer_normalization_backward_gpu(const TensorView& input, const TensorView& output_delta,
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

static void rms_normalization_forward_gpu(const TensorView& input, const TensorView& weight,
                            TensorView& inverse_rms, TensorView& output, float epsilon)
{
    const int rows = to_int(input.size() / input.shape.back());
    const int cols = to_int(input.shape.back());

    output.dispatch([&](auto tag) {
        using T = decltype(tag);
        rmsnorm_forward_cuda<T>(rows, cols,
                                input.as<T>(), output.as<T>(),
                                inverse_rms.as<float>(),
                                weight.as<float>(), epsilon);
    });
}

static void rms_normalization_backward_gpu(const TensorView& input, const TensorView& output_delta,
                             const TensorView& inverse_rms, const TensorView& weight,
                             const TensorView& weight_gradient, TensorView& input_delta)
{
    const int rows = to_int(input.size() / input.shape.back());
    const int cols = to_int(input.shape.back());

    input.dispatch([&](auto tag) {
        using T = decltype(tag);
        T* input_delta_data = input_delta.empty() ? nullptr : input_delta.as<T>();

        rmsnorm_backward_cuda<T>(rows, cols,
                                 output_delta.as<T>(), input.as<T>(),
                                 inverse_rms.as<float>(), weight.as<float>(),
                                 input_delta_data, weight_gradient.as<float>());
    });
}

static void rope_forward_gpu(const TensorView& input, const TensorView& cos_table, const TensorView& sin_table,
                             TensorView& output, Index head_dim, Index rotary_dim, Index position_offset)
{
    const int seq       = to_int(input.shape[1]);
    const int model_dim = to_int(input.shape.back());
    const int rows      = to_int(input.size() / input.shape.back());

    output.dispatch([&](auto tag) {
        using T = decltype(tag);
        rope_forward_cuda<T>(rows, seq, model_dim, to_int(head_dim), to_int(rotary_dim), to_int(position_offset),
                             input.as<T>(), output.as<T>(),
                             cos_table.as<float>(), sin_table.as<float>());
    });
}

static void rope_backward_gpu(const TensorView& output_delta, const TensorView& cos_table, const TensorView& sin_table,
                              TensorView& input_delta, Index head_dim, Index rotary_dim, Index position_offset)
{
    const int seq       = to_int(output_delta.shape[1]);
    const int model_dim = to_int(output_delta.shape.back());
    const int rows      = to_int(output_delta.size() / output_delta.shape.back());

    input_delta.dispatch([&](auto tag) {
        using T = decltype(tag);
        rope_backward_cuda<T>(rows, seq, model_dim, to_int(head_dim), to_int(rotary_dim), to_int(position_offset),
                              output_delta.as<T>(), input_delta.as<T>(),
                              cos_table.as<float>(), sin_table.as<float>());
    });
}

static void swiglu_forward_gpu(const TensorView& gate, const TensorView& up, TensorView& output)
{
    const int n = to_int(gate.size());
    output.dispatch([&](auto tag) {
        using T = decltype(tag);
        swiglu_forward_cuda<T>(n, gate.as<T>(), up.as<T>(), output.as<T>());
    });
}

static void swiglu_backward_gpu(const TensorView& output_delta, const TensorView& gate, const TensorView& up,
                                TensorView& gate_delta, TensorView& up_delta)
{
    const int n = to_int(output_delta.size());
    output_delta.dispatch([&](auto tag) {
        using T = decltype(tag);
        T* gate_delta_data = gate_delta.empty() ? nullptr : gate_delta.as<T>();
        T* up_delta_data   = up_delta.empty()   ? nullptr : up_delta.as<T>();
        swiglu_backward_cuda<T>(n, output_delta.as<T>(), gate.as<T>(), up.as<T>(),
                                gate_delta_data, up_delta_data);
    });
}

Index grouped_attention_decode_scratch_floats(Index n_query_heads, Index head_dim)
{
    return n_query_heads * GROUPED_ATTENTION_DECODE_SPLITS * (head_dim + 2);
}

static cublasHandle_t grouped_attention_cublas()
{
    thread_local Buffer cublas_workspace{Device::CUDA};
    thread_local cublasHandle_t handle = nullptr;
    if (!handle)
    {
        constexpr Index workspace_bytes = Index(4) << 20;
        cublas_workspace.grow_to(workspace_bytes);
        CHECK_CUBLAS(cublasCreate(&handle));
        CHECK_CUBLAS(cublasSetStream(handle, device::get_compute_stream()));
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        CHECK_CUBLAS(cublasSetWorkspace(handle, cublas_workspace.as<char>(), size_t(workspace_bytes)));
    }
    return handle;
}

static void grouped_attention_gemm(cublasOperation_t transa, cublasOperation_t transb,
                                   int m, int n, int k, float alpha,
                                   const void* A, cudaDataType_t a_type, int lda, long long stride_a,
                                   const void* B, cudaDataType_t b_type, int ldb, long long stride_b,
                                   void* C, cudaDataType_t c_type, int ldc, long long stride_c,
                                   int batch_count)
{
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(grouped_attention_cublas(),
                                            transa, transb, m, n, k,
                                            &alpha,
                                            A, a_type, lda, stride_a,
                                            B, b_type, ldb, stride_b,
                                            &beta,
                                            C, c_type, ldc, stride_c,
                                            batch_count,
                                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template<typename T>
static bool grouped_attention_gemm_gpu(const int batch, const int query_seq, const int key_seq,
                                       const int n_query_heads, const int n_kv_heads, const int head_dim,
                                       const float scale, const int query_position_offset, const bool causal,
                                       const T* Q, const T* K, const T* V, T* O)
{
    const int group = n_kv_heads > 0 ? n_query_heads / n_kv_heads : 0;
    if (group < 1 || group * n_kv_heads != n_query_heads || key_seq <= 0 || head_dim <= 0)
        return false;

    constexpr bool is_fp32 = std::is_same_v<T, float>;
    const cudaDataType_t dtype = is_fp32 ? CUDA_R_32F : CUDA_R_16BF;

    const Index q_elems  = Index(query_seq) * n_query_heads * head_dim;
    const Index kv_elems = Index(key_seq) * n_kv_heads * head_dim;
    const Index s_elems  = Index(n_query_heads) * query_seq * key_seq;

    const Index per_batch_bytes = s_elems * Index(sizeof(float))
                                + (is_fp32 ? 0 : s_elems * Index(sizeof(T)))
                                + 2 * (q_elems + kv_elems) * Index(sizeof(T));

    constexpr Index budget_bytes = Index(256) << 20;
    const int chunk = to_int(max(Index(1), min(Index(batch), budget_bytes / per_batch_bytes)));

    auto aligned = [](Index bytes) { return (bytes + 15) & ~Index(15); };
    const Index scores_bytes = aligned(Index(chunk) * s_elems * Index(sizeof(float)));
    const Index probs_bytes  = is_fp32 ? 0 : aligned(Index(chunk) * s_elems * Index(sizeof(T)));
    const Index q_bytes      = aligned(Index(chunk) * q_elems * Index(sizeof(T)));
    const Index kv_bytes     = aligned(Index(chunk) * kv_elems * Index(sizeof(T)));

    thread_local Buffer workspace{Device::CUDA};

    try
    {
        workspace.grow_to(scores_bytes + probs_bytes + 2 * q_bytes + 2 * kv_bytes);

        char* base    = workspace.as<char>();
        float* scores = reinterpret_cast<float*>(base);
        T* probs      = is_fp32 ? reinterpret_cast<T*>(scores)
                                : reinterpret_cast<T*>(base + scores_bytes);
        T* Qt         = reinterpret_cast<T*>(base + scores_bytes + probs_bytes);
        T* Ot         = reinterpret_cast<T*>(base + scores_bytes + probs_bytes + q_bytes);
        T* Kt         = reinterpret_cast<T*>(base + scores_bytes + probs_bytes + 2 * q_bytes);
        T* Vt         = reinterpret_cast<T*>(base + scores_bytes + probs_bytes + 2 * q_bytes + kv_bytes);

        const int mq = group * query_seq;

        for (int b0 = 0; b0 < batch; b0 += chunk)
        {
            const int bc = min(chunk, batch - b0);
            const int batch_count = bc * n_kv_heads;

            split_heads_cuda<T>(Index(bc) * q_elems, Q + Index(b0) * q_elems, Qt,
                                query_seq, n_query_heads, head_dim);
            split_heads_cuda<T>(Index(bc) * kv_elems, K + Index(b0) * kv_elems, Kt,
                                key_seq, n_kv_heads, head_dim);
            split_heads_cuda<T>(Index(bc) * kv_elems, V + Index(b0) * kv_elems, Vt,
                                key_seq, n_kv_heads, head_dim);

            const int kv_valid = causal ? min(query_position_offset + query_seq, key_seq) : key_seq;
            if (kv_valid < key_seq)
            {
                const size_t tail_bytes = size_t(key_seq - kv_valid) * head_dim * sizeof(T);
                for (int i = 0; i < bc * n_kv_heads; ++i)
                    CHECK_CUDA(cudaMemsetAsync(Vt + (Index(i) * key_seq + kv_valid) * head_dim,
                                               0, tail_bytes, device::get_compute_stream()));
            }

            grouped_attention_gemm(CUBLAS_OP_T, CUBLAS_OP_N, key_seq, mq, head_dim, scale,
                                   Kt, dtype, head_dim, Index(key_seq) * head_dim,
                                   Qt, dtype, head_dim, Index(mq) * head_dim,
                                   scores, CUDA_R_32F, key_seq, Index(mq) * key_seq,
                                   batch_count);

            grouped_attention_softmax_cuda<T>(bc * n_query_heads * query_seq, query_seq, key_seq,
                                              query_position_offset, causal, scores, probs);

            grouped_attention_gemm(CUBLAS_OP_N, CUBLAS_OP_N, head_dim, mq, key_seq, 1.0f,
                                   Vt, dtype, head_dim, Index(key_seq) * head_dim,
                                   probs, dtype, key_seq, Index(mq) * key_seq,
                                   Ot, dtype, head_dim, Index(mq) * head_dim,
                                   batch_count);

            merge_heads_cuda<T>(Index(bc) * q_elems, Ot, O + Index(b0) * q_elems,
                                query_seq, n_query_heads, head_dim);
        }
    }
    catch (...)
    {
        device::reset_last_error();
        return false;
    }

    return true;
}

static void grouped_attention_gpu(const TensorView& query, const TensorView& key, const TensorView& value,
                                  TensorView& output, Index n_query_heads, Index n_kv_heads, Index head_dim,
                                  bool causal, float scale, Index query_position_offset,
                                  float* decode_partials, const int* kv_length_device)
{
    const int batch     = to_int(query.shape[0]);
    const int query_seq = to_int(query.shape[1]);
    const int key_seq   = to_int(key.shape[1]);
    const int group     = to_int(n_kv_heads) > 0 ? to_int(n_query_heads / n_kv_heads) : 0;

    const bool decode = batch == 1 && query_seq == 1 && causal && decode_partials
                     && grouped_attention_decode_supported(to_int(head_dim), group);

    output.dispatch([&](auto tag) {
        using T = decltype(tag);

        if (batch * query_seq * to_int(n_query_heads) > 0 && !decode
            && grouped_attention_gemm_gpu<T>(batch, query_seq, key_seq,
                                             to_int(n_query_heads), to_int(n_kv_heads), to_int(head_dim),
                                             scale, to_int(query_position_offset), causal,
                                             query.as<T>(), key.as<T>(), value.as<T>(), output.as<T>()))
            return;

        grouped_attention_cuda<T>(batch, query_seq, key_seq, to_int(n_query_heads), to_int(n_kv_heads),
                                  to_int(head_dim), scale, to_int(query_position_offset), causal,
                                  kv_length_device, decode_partials,
                                  query.as<T>(), key.as<T>(), value.as<T>(), output.as<T>());
    });
}

void qk_rope_cache_append(const TensorView& qkv_row, const TensorView& q_norm_weight,
                          const TensorView& k_norm_weight, const TensorView& cos_table,
                          const TensorView& sin_table, TensorView& q_out,
                          TensorView& key_cache, TensorView& value_cache,
                          Index n_query_heads, Index n_kv_heads, Index head_dim,
                          float epsilon, const int* position_device)
{
    throw_if(!qkv_row.is_cuda() || !position_device, "qk_rope_cache_append: GPU tensors and a device position are required.");

    q_out.dispatch([&](auto tag) {
        using T = decltype(tag);
        qk_rope_cache_append_cuda<T>(to_int(n_query_heads), to_int(n_kv_heads), to_int(head_dim),
                                     epsilon, position_device, qkv_row.as<T>(),
                                     q_norm_weight.empty() ? nullptr : q_norm_weight.as<float>(),
                                     k_norm_weight.empty() ? nullptr : k_norm_weight.as<float>(),
                                     cos_table.as<float>(), sin_table.as<float>(),
                                     q_out.as<T>(), key_cache.as<T>(), value_cache.as<T>());
    });
}

void sample_logits_row(const TensorView& logits_row, float temperature, Index top_k, float top_p,
                       unsigned long long seed, unsigned long long step,
                       void* candidates_scratch, int* id_device, float* token_device)
{
    throw_if(!logits_row.is_cuda() || !candidates_scratch || !id_device,
             "sample_logits_row: a GPU logits row, device scratch and a device id are required.");

    logits_row.dispatch([&](auto tag) {
        using T = decltype(tag);
        sample_logits_row_cuda<T>(to_int(logits_row.size()), temperature, to_int(top_k), top_p,
                                  seed, step, logits_row.as<T>(),
                                  static_cast<float2*>(candidates_scratch), id_device, token_device);
    });
}

Index sample_logits_scratch_floats()
{
    return Index(LOGITS_SAMPLE_BLOCKS) * 32 * 2;
}

static void qk_norm_gpu(const TensorView& input, const TensorView& weight, TensorView& output,
                        Index head_dim, float epsilon)
{
    const int rows = to_int(input.size() / head_dim);
    output.dispatch([&](auto tag) {
        using T = decltype(tag);
        rmsnorm_forward_cuda<T>(rows, to_int(head_dim), input.as<T>(), output.as<T>(),
                                nullptr, weight.as<float>(), epsilon);
    });
}

static void embedding_lookup_forward_gpu(const TensorView& indices, const TensorView& weights,
                                  const TensorView& positional_encoding, TensorView& output,
                                  Index sequence_length, Index embedding_dimension, Index vocabulary_size,
                                  bool scale_embedding, bool add_positional_encoding,
                                  const TensorView& weight_scale)
{
    if (weights.is_int8())
    {
        throw_if(weight_scale.empty(),
                 "embedding_lookup_forward: INT8 weights require a per-row scale vector.");
        output.dispatch([&](auto out_tag) {
            using T = decltype(out_tag);
            embedding_forward_w8_cuda<T>(
                output.size(),
                indices.as<float>(),
                weights.as<int8_t>(),
                weight_scale.as<float>(),
                add_positional_encoding ? positional_encoding.as<float>() : nullptr,
                output.as<T>(),
                to_int(sequence_length), to_int(embedding_dimension), to_int(vocabulary_size),
                scale_embedding);
        });
        return;
    }

    output.dispatch([&](auto out_tag) {
        using T = decltype(out_tag);
        weights.dispatch([&](auto weight_tag) {
            using TW = decltype(weight_tag);
            embedding_forward_cuda<TW, T>(
                output.size(),
                indices.as<float>(),
                weights.as<TW>(),
                add_positional_encoding ? positional_encoding.as<float>() : nullptr,
                output.as<T>(),
                to_int(sequence_length), to_int(embedding_dimension), to_int(vocabulary_size),
                scale_embedding);
        });
    });
}

static void embedding_lookup_backward_gpu(const TensorView& indices, const TensorView& output_delta,
                                   const TensorView& weight_gradient, const TensorView& positional_gradient,
                                   Index sequence_length, Index embedding_dimension, Index vocabulary_size,
                                   bool scale_embedding)
{
    weight_gradient.set_zero_async();

    const bool accumulate_positional = !positional_gradient.empty() && positional_gradient.data != nullptr;
    if (accumulate_positional) positional_gradient.set_zero_async();

    output_delta.dispatch([&](auto tag) {
        using T = decltype(tag);
        embedding_backward_cuda<T>(
            output_delta.size(),
            indices.as<float>(),
            output_delta.as<T>(),
            weight_gradient.as<float>(),
            accumulate_positional ? positional_gradient.as<float>() : nullptr,
            to_int(sequence_length), to_int(embedding_dimension), to_int(vocabulary_size), scale_embedding);
    });
}

static void max_pooling_3d_forward_gpu(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool  )
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

static void first_token_3d_forward_gpu(const TensorView& input, TensorView& output)
{
    output.dispatch([&](auto tag) {
        using T = decltype(tag);
        first_token_3d_forward_cuda<T>(to_int(input.shape[0]), to_int(input.shape[1]), to_int(input.shape[2]),
                                       input.as<T>(), output.as<T>());
    });
}

static void first_token_3d_backward_gpu(const TensorView& output_delta, TensorView& input_delta)
{
    input_delta.dispatch([&](auto tag) {
        using T = decltype(tag);
        input_delta.set_zero_async();
        first_token_3d_backward_cuda<T>(to_int(input_delta.shape[0]), to_int(input_delta.shape[1]), to_int(input_delta.shape[2]),
                                        output_delta.as<T>(), input_delta.as<T>());
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

Index grouped_attention_decode_scratch_floats(Index, Index)
{
    return 0;
}

void qk_rope_cache_append(const TensorView&, const TensorView&, const TensorView&, const TensorView&,
                          const TensorView&, TensorView&, TensorView&, TensorView&,
                          Index, Index, Index, float, const int*)
{
    throw runtime_error("qk_rope_cache_append: CUDA support not compiled in.");
}

void sample_logits_row(const TensorView&, float, Index, float, unsigned long long, unsigned long long,
                       void*, int*, float*)
{
    throw runtime_error("sample_logits_row: CUDA support not compiled in.");
}

Index sample_logits_scratch_floats()
{
    return 0;
}

#endif

MatrixR append_rows(const MatrixR& starting_matrix, const MatrixR& block)
{
    if (starting_matrix.size() == 0)
        return block;
    if (block.size() == 0)
        return starting_matrix;

    throw_if(starting_matrix.cols() != block.cols(),
             "append_rows: Column mismatch ({} vs {})",
                    starting_matrix.cols(), block.cols());

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

VectorR slice_rows(const VectorR& values, const vector<Index>& indices)
{
    VectorR result(ssize(indices));

    for (Index i = 0; i < ssize(indices); ++i)
        result(i) = values(indices[i]);

    return result;
}

MatrixR slice_rows(const MatrixR& matrix, const vector<Index>& indices)
{
    MatrixR result(ssize(indices), matrix.cols());

    for (Index i = 0; i < ssize(indices); ++i)
        result.row(i) = matrix.row(indices[i]);

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

MatrixR calculate_distances(const MatrixR& points)
{
    const VectorR squared_norms = points.rowwise().squaredNorm();

    MatrixR squared_distances = -2.0f * points * points.transpose();
    squared_distances.colwise() += squared_norms;
    squared_distances.rowwise() += squared_norms.transpose();

    return squared_distances.cwiseMax(0.0f).cwiseSqrt();
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
