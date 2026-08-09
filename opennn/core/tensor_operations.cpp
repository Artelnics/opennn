//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   O P E R A T I O N S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/tensor_operations.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/scaling.h"
#include "opennn/core/profiler.h"
#include "opennn/core/cuda/kernel.cuh"

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

template<typename Column>
static void scale_column_cpu(Column& column, ScalerMethod method,
                             const Descriptives& descriptives,
                             float min_range, float max_range)
{
    using enum ScalerMethod;

    switch (method)
    {
    case MinimumMaximum:
        if (descriptives.maximum - descriptives.minimum < EPSILON)
            column.setZero();
        else
            column = scale_minimum_maximum_formula(column, descriptives, min_range, max_range);
        break;
    case MeanStandardDeviation:
        if (descriptives.standard_deviation > EPSILON)
            column = scale_mean_standard_deviation_formula(column, descriptives);
        else
            column.setZero();
        break;
    case StandardDeviation:
        column *= descriptives.standard_deviation > EPSILON
                ? 1.0f / descriptives.standard_deviation
                : 0.0f;
        break;
    case Logarithm:
        column = column.max(EPSILON).log();
        break;
    case ImageMinMax:
        column /= 255.0f;
        break;
    case None:
    default:
        break;
    }
}

template<typename Column>
static void unscale_column_cpu(Column& column, ScalerMethod method,
                               const Descriptives& descriptives,
                               float min_range, float max_range)
{
    using enum ScalerMethod;

    switch (method)
    {
    case MinimumMaximum:
        throw_if(max_range - min_range < EPSILON, "The range values are not valid.");
        column = unscale_minimum_maximum_formula(column, descriptives, min_range, max_range);
        break;
    case MeanStandardDeviation:
        column = unscale_mean_standard_deviation_formula(column, descriptives);
        break;
    case StandardDeviation:
        if (descriptives.standard_deviation > EPSILON)
            column *= descriptives.standard_deviation;
        else
            column.setConstant(descriptives.mean);
        break;
    case Logarithm:
        column = column.exp();
        break;
    case ImageMinMax:
        column *= 255.0f;
        break;
    case None:
    default:
        break;
    }
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
        const auto method = static_cast<ScalerMethod>(static_cast<int>(scalers_vector(feature_index)));
        auto column = output_matrix.col(col).array();

        const Descriptives descriptives(minimums_vector(feature_index),
                                        maximums_vector(feature_index),
                                        means_vector(feature_index),
                                        standard_deviations_vector(feature_index));

        if (inverse)
            unscale_column_cpu(column, method, descriptives, min_range, max_range);
        else
            scale_column_cpu(column, method, descriptives, min_range, max_range);
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
    VectorMap mask_values = mask.as_vector();

    set_random_uniform(mask_values, 0.0f, 1.0f);

    const bool parallel = element_count >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index i = 0; i < element_count; ++i)
    {
        const float keep_value = mask_values(i) < rate ? 0.0f : keep_scale;
        mask_values(i) = keep_value;
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
    delta.as_vector().array() *= mask.as_vector().array();
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

    visit_type_pair<Type::FP32, Type::BF16>(input.type, output.type, [&]<typename TIn, typename TOut>() {
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

    visit_type_pair<Type::FP32, Type::BF16>(input.type, output.type, [&]<typename TIn, typename TOut>() {
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
                              Backend::get_op_tensor_add_descriptor(),
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
    output.dispatch([&]<typename T>()
    {
        activation_forward_cuda<T>(output.size(), output.as<T>(), static_cast<int>(function));
    });
    device::check_last_error();
}

static void activation_backward_gpu(const TensorView& outputs, TensorView& delta, ActivationFunction function)
{
    delta.dispatch([&]<typename T>()
    {
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

    output.dispatch([&]<typename T>()
    {
        dropout_forward_cuda<T>(element_count, output.as<T>(), mask.as<uint8_t>(), rate, seed);
    });
}

static void dropout_backward_gpu(TensorView& delta, const Buffer& mask, float rate)
{
    const Index element_count = delta.size();

    delta.dispatch([&]<typename T>()
    {
        dropout_backward_cuda<T>(element_count, delta.as<T>(), delta.as<T>(), mask.as<uint8_t>(), rate);
    });
}

static void linear_forward_lt_gpu(const TensorView& input, const TensorView& weights, const TensorView& bias,
                                  TensorView& output, cublasLtEpilogue_t epilogue,
                                  TensorView* pre_activation)
{
    const int input_columns  = to_int(input.shape.back());
    const int output_columns = to_int(weights.shape.back());
    const int total_rows     = to_int(input.size() / input.shape.back());

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
            linear_forward_lt_gpu(input, weights, bias, *pre_activation,
                                  CUBLASLT_EPILOGUE_BIAS, nullptr);
            copy_gpu(*pre_activation, output);
            activation_forward_gpu(output, ActivationFunction::GELUTanh);
            return;
        }

        throw runtime_error(format("cuBLASLt GEMM {}x{}x{} ({}) failed: {}",
                                   output_columns, total_rows, input_columns,
                                   output.is_bf16() ? "bf16" : "fp32", e.what()));
    }
}

static void linear_forward_gpu(const TensorView& input, const TensorView& weights, const TensorView& bias,
                               TensorView& output, cublasLtEpilogue_t epilogue,
                               TensorView* pre_activation, const TensorView& weight_scale)
{
    if (!weights.is_int8())
    {
        linear_forward_lt_gpu(input, weights, bias, output, epilogue, pre_activation);
        return;
    }

    throw_if(weight_scale.empty() || !input.is_bf16() || !output.is_bf16(),
             "linear_forward: INT8 weights require BF16 activations and a per-channel scale vector.");

    const int input_columns  = to_int(input.shape.back());
    const int output_columns = to_int(weights.shape.back());
    const int total_rows     = to_int(input.size() / input.shape.back());

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
    linear_forward_lt_gpu(input, dequantized_weights, bias, output, epilogue, pre_activation);
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


void sample_logits_row(const TensorView& logits_row, float temperature, Index top_k, float top_p,
                       unsigned long long seed, unsigned long long step,
                       void* candidates_scratch, int* id_device, float* token_device)
{
    throw_if(!logits_row.is_cuda() || !candidates_scratch || !id_device,
             "sample_logits_row: a GPU logits row, device scratch and a device id are required.");

    logits_row.dispatch([&]<typename T>() {
        sample_logits_row_cuda<T>(to_int(logits_row.size()), temperature, to_int(top_k), top_p,
                                  seed, step, logits_row.as<T>(),
                                  static_cast<float2*>(candidates_scratch), id_device, token_device);
    });
}

Index sample_logits_scratch_floats()
{
    return Index(LOGITS_SAMPLE_BLOCKS) * 32 * 2;
}


static void split_heads_gpu(const TensorView& source, TensorView& destination)
{
    const Index sequence_length = source.shape[1];
    const Index heads_number = source.shape[2];
    const Index head_dimension = source.shape[3];

    destination.dispatch([&]<typename T>() {
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

    destination.dispatch([&]<typename T>() {
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

VectorI get_nearest_points(const MatrixR& matrix, const VectorR& point, int neighbors_number)
{
    const Index rows = matrix.rows();

    const VectorR distances = (matrix.rowwise() - point.transpose()).rowwise().norm();

    vector<Index> indices(rows);
    iota(indices.begin(), indices.end(), Index(0));

    neighbors_number = std::min(neighbors_number, to_int(rows));

    partial_sort(indices.begin(), indices.begin() + neighbors_number, indices.end(),
                 [&distances](Index i, Index j) {
                     return pair{distances(i), i} < pair{distances(j), j};
                 });

    return Map<VectorI>(indices.data(), neighbors_number);
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
