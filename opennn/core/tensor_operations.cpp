//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   O P E R A T I O N S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/tensor_operations.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/profiler.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/cuda/kernel_activation.cuh"
#include "opennn/core/cuda/kernel_normalization.cuh"
#include "opennn/core/cuda/kernel_cast.cuh"
#include "opennn/core/cuda/kernel_quantization.cuh"
#include "opennn/core/cuda/kernel_tensor.cuh"

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
        return cblas_sger(CblasRowMajor,
                          to_int(rows),
                          to_int(columns),
                          1.0f,
                          ones.data(),
                          1,
                          b,
                          1,
                          y,
                          to_int(columns));
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
        || input.get_shape().get_rank() == 0
        || weights.get_shape().get_rank() != 2
        || bias.get_shape().get_rank() != 1)
        return false;

    const Index input_columns = input.get_shape().back();
    const Index output_columns = weights.get_shape().back();

    if (input_columns <= 0
        || output_columns <= 0
        || input.size() % input_columns != 0
        || weights.get_shape()[0] != input_columns
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
    X(copy_gpu, (const TensorView&, TensorView&)) \
    X(add_gpu, (const TensorView&, const TensorView&, TensorView&)) \
    X(multiply_gpu, (const TensorView&, bool, const TensorView&, bool, TensorView&, float, float)) \
    X(softmax_gpu, (TensorView&)) \
    X(activation_forward_gpu, (TensorView&, ActivationFunction)) \
    X(activation_backward_gpu, (const TensorView&, TensorView&, ActivationFunction)) \
    X(linear_forward_gpu, (const TensorView&, const TensorView&, const TensorView&, TensorView&, cublasLtEpilogue_t, TensorView*, const TensorView&)) \
    X(linear_backward_gpu, (const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, TensorView&, bool, const TensorView*, const TensorView*))

#define OPENNN_DECLARE_GPU_OP(name, sig) static void name sig;
OPENNN_GPU_OPS(OPENNN_DECLARE_GPU_OP)
#undef OPENNN_DECLARE_GPU_OP

static void require_tensor(const TensorView& tensor, string_view operation, string_view role)
{
    const Shape& shape = tensor.get_shape();
    throw_if(shape.empty(), "{}: {} must have a shape.", operation, role);
    throw_if(any_of(shape.begin(), shape.end(), [](Index dim) { return dim < 0; }),
             "{}: {} has a negative dimension.", operation, role);
    throw_if(tensor.get_device() == Device::Auto, "{}: {} has unresolved device metadata.", operation, role);
    throw_if(tensor.get_type() == Type::Auto, "{}: {} has unresolved dtype metadata.", operation, role);
    throw_if(tensor.size() > 0 && !tensor.get_data(), "{}: {} has no storage.", operation, role);
}

static void require_same_device(const TensorView& reference, const TensorView& tensor,
                                string_view operation)
{
    throw_if(reference.get_device() != tensor.get_device(),
             "{}: all tensors must be on the same device.", operation);
}

static void require_same_type(const TensorView& reference, const TensorView& tensor,
                              string_view operation)
{
    throw_if(reference.get_type() != tensor.get_type(),
             "{}: tensor dtypes are incompatible.", operation);
}

static void require_same_shape(const TensorView& reference, const TensorView& tensor,
                               string_view operation)
{
    throw_if(reference.get_shape() != tensor.get_shape(),
             "{}: tensor shapes are incompatible.", operation);
}

static void require_optional_tensor(const TensorView& reference, const TensorView& tensor,
                                    string_view operation, string_view role)
{
    if (tensor.empty()) return;
    require_tensor(tensor, operation, role);
    require_same_device(reference, tensor, operation);
}

static void require_fp32_or_bf16(const TensorView& tensor, string_view operation, string_view role)
{
    throw_if(!tensor.is_fp32() && !tensor.is_bf16(),
             "{}: {} must use FP32 or BF16 storage.", operation, role);
}

static void require_cpu_fp32(const TensorView& tensor, string_view operation, string_view role)
{
    throw_if(tensor.get_device() != Device::CPU || !tensor.is_fp32(),
             "{}: CPU {} must use FP32 storage.", operation, role);
}

static Index matrix_count(const TensorView& tensor)
{
    const Shape& shape = tensor.get_shape();
    const size_t rank = shape.get_rank();
    return tensor.size() / (shape[rank - 2] * shape[rank - 1]);
}

static void require_matching_linear_prefix(const TensorView& input, const TensorView& output,
                                           string_view operation)
{
    const Shape& input_shape = input.get_shape();
    const Shape& output_shape = output.get_shape();

    throw_if(input_shape.get_rank() != output_shape.get_rank(),
             "{}: input and output ranks do not match.", operation);
    for (size_t i = 0; i + 1 < input_shape.get_rank(); ++i)
        throw_if(input_shape[i] != output_shape[i],
                 "{}: input and output leading dimensions do not match.", operation);
}

static void validate_linear_io(const TensorView& input, const TensorView& weights,
                               const TensorView& output, bool transposed_weights,
                               string_view operation)
{
    require_tensor(input, operation, "input");
    require_tensor(weights, operation, "weights");
    require_tensor(output, operation, "output");
    require_same_device(input, weights, operation);
    require_same_device(input, output, operation);

    const Shape& input_shape = input.get_shape();
    const Shape& weights_shape = weights.get_shape();
    const Shape& output_shape = output.get_shape();

    throw_if(input_shape.get_rank() < 1, "{}: input rank must be at least one.", operation);
    throw_if(weights_shape.get_rank() != 2, "{}: weights must be a matrix.", operation);
    throw_if(output_shape.get_rank() < 1, "{}: output rank must be at least one.", operation);

    const Index input_features  = input_shape.back();
    const Index weight_inputs   = transposed_weights ? weights_shape[1] : weights_shape[0];
    const Index output_features = transposed_weights ? weights_shape[0] : weights_shape[1];
    throw_if(input_features <= 0 || output_features <= 0,
             "{}: feature dimensions must be positive.", operation);
    throw_if(weight_inputs != input_features,
             "{}: input and weight feature dimensions do not match.", operation);
    throw_if(output_shape.back() != output_features,
             "{}: output feature dimension does not match the weights.", operation);
    require_matching_linear_prefix(input, output, operation);
}

static void validate_linear_types(const TensorView& input, const TensorView& weights,
                                  const TensorView& output, string_view operation)
{
    if (!input.is_cuda())
    {
        require_cpu_fp32(input, operation, "input");
        require_cpu_fp32(weights, operation, "weights");
        return require_cpu_fp32(output, operation, "output");
    }

    require_fp32_or_bf16(input, operation, "input");
    require_fp32_or_bf16(output, operation, "output");
    throw_if(!weights.is_int8() && weights.get_type() != output.get_type(),
             "{}: non-quantized weights and output must use the same dtype.", operation);
}

void copy(const TensorView& source, TensorView& destination)
{
    require_tensor(source, "copy", "source");
    require_tensor(destination, "copy", "destination");
    require_same_shape(source, destination, "copy");
    require_same_device(source, destination, "copy");
    require_same_type(source, destination, "copy");

    if (source.is_cuda()) { copy_gpu(source, destination); return; }
    memcpy(destination.get_data(), source.get_data(), source.byte_size());
}

void add(const TensorView& input_1,
         const TensorView& input_2,
         TensorView& output)
{
    require_tensor(input_1, "add", "first input");
    require_tensor(input_2, "add", "second input");
    require_tensor(output, "add", "output");
    require_same_shape(input_1, input_2, "add");
    require_same_shape(input_1, output, "add");
    require_same_device(input_1, input_2, "add");
    require_same_device(input_1, output, "add");
    require_same_type(input_1, input_2, "add");
    require_same_type(input_1, output, "add");
    if (!input_1.is_cuda()) require_cpu_fp32(input_1, "add", "input");

    if (input_1.is_cuda()) { add_gpu(input_1, input_2, output); return; }
    output.as_vector().noalias() = input_1.as_vector() + input_2.as_vector();
}

static void multiply_cpu(const TensorView& input_a, bool transpose_a,
                  const TensorView& input_b, bool transpose_b,
                  TensorView& output,
                  float alpha, float beta)
{
    const Shape& shape = input_a.get_shape();
    const size_t rank = shape.get_rank();
    const Index batch_count = input_a.size() / (shape[rank - 2] * shape[rank - 1]);

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
    require_tensor(input_a, "multiply", "first input");
    require_tensor(input_b, "multiply", "second input");
    require_tensor(output, "multiply", "output");
    require_same_device(input_a, input_b, "multiply");
    require_same_device(input_a, output, "multiply");

    const Shape& shape_a = input_a.get_shape();
    const Shape& shape_b = input_b.get_shape();
    const Shape& output_shape = output.get_shape();

    throw_if(shape_a.get_rank() < 2 || shape_b.get_rank() < 2 || output_shape.get_rank() < 2,
             "multiply: all tensors must have rank two or greater.");
    require_fp32_or_bf16(input_a, "multiply", "first input");
    require_fp32_or_bf16(input_b, "multiply", "second input");
    require_fp32_or_bf16(output, "multiply", "output");
    if (!input_a.is_cuda())
    {
        require_cpu_fp32(input_a, "multiply", "first input");
        require_same_type(input_a, input_b, "multiply");
        require_same_type(input_a, output, "multiply");
    }

    const size_t rank_a = shape_a.get_rank();
    const size_t rank_b = shape_b.get_rank();
    const Index rows_a = shape_a[rank_a - 2];
    const Index cols_a = shape_a[rank_a - 1];
    const Index rows_b = shape_b[rank_b - 2];
    const Index cols_b = shape_b[rank_b - 1];
    const Index rows_output = output_shape[output_shape.get_rank() - 2];
    const Index cols_output = output_shape.back();
    throw_if(rows_a <= 0 || cols_a <= 0 || rows_b <= 0 || cols_b <= 0
             || rows_output <= 0 || cols_output <= 0,
             "multiply: matrix dimensions must be positive.");

    const Index inner_a = transpose_a ? rows_a : cols_a;
    const Index inner_b = transpose_b ? cols_b : rows_b;
    const Index result_rows = transpose_a ? cols_a : rows_a;
    const Index result_columns = transpose_b ? rows_b : cols_b;
    throw_if(inner_a != inner_b, "multiply: inner matrix dimensions do not match.");

    const bool flattened_cuda_rhs = input_a.is_cuda() && rank_a > 2 && rank_b == 2;
    if (flattened_cuda_rhs)
    {
        throw_if(transpose_a, "multiply: a flattened CUDA left operand cannot be transposed.");
        const Index flat_rows = input_a.size() / cols_a;
        throw_if(output_shape.back() != result_columns
                 || output.size() != flat_rows * result_columns,
                 "multiply: output shape does not match the flattened matrix product.");
    }
    else
    {
        throw_if(rank_a != rank_b || rank_a != output_shape.get_rank(),
                 "multiply: batched operands and output must have matching ranks.");
        for (size_t i = 0; i + 2 < rank_a; ++i)
            throw_if(shape_a[i] != shape_b[i] || shape_a[i] != output_shape[i],
                     "multiply: batch dimensions do not match.");
        throw_if(matrix_count(input_a) != matrix_count(input_b)
                 || matrix_count(input_a) != matrix_count(output),
                 "multiply: matrix batch counts do not match.");
        throw_if(output_shape[output_shape.get_rank() - 2] != result_rows
                 || output_shape.back() != result_columns,
                 "multiply: output matrix dimensions do not match the product.");
    }

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

    require_tensor(output, "softmax", "output");
    throw_if(output.get_shape().back() <= 0, "softmax: the channel dimension must be positive.");
    require_fp32_or_bf16(output, "softmax", "output");
    if (!output.is_cuda()) require_cpu_fp32(output, "softmax", "output");

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

    require_tensor(output, "activation_forward", "output");
    require_fp32_or_bf16(output, "activation_forward", "output");
    if (!output.is_cuda()) require_cpu_fp32(output, "activation_forward", "output");

    if (output.is_cuda()) { activation_forward_gpu(output, function); return; }
    activation_forward_cpu(output, function);
}

void activation_backward(const TensorView& outputs, TensorView& delta, ActivationFunction function)
{
    if (is_one_of(function, ActivationFunction::Identity, ActivationFunction::Softmax)
        || outputs.empty()) return;

    require_tensor(outputs, "activation_backward", "outputs");
    require_tensor(delta, "activation_backward", "delta");
    require_same_shape(outputs, delta, "activation_backward");
    require_same_device(outputs, delta, "activation_backward");
    require_same_type(outputs, delta, "activation_backward");
    require_fp32_or_bf16(outputs, "activation_backward", "outputs");
    if (!outputs.is_cuda()) require_cpu_fp32(outputs, "activation_backward", "outputs");

    if (outputs.is_cuda()) { activation_backward_gpu(outputs, delta, function); return; }
    activation_backward_cpu(outputs, delta, function);
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
                         TensorView& input_delta, bool accumulate, const TensorView* addend)
{
    weight_gradient.as_matrix().noalias() = input.as_flat_matrix().transpose() * output_delta.as_flat_matrix();
    if (!bias_gradient.empty())
        bias_gradient.as_vector().noalias() = output_delta.as_flat_matrix().colwise().sum();

    if (!input_delta.get_data() || input_delta.empty()) return;

    auto input_delta_mat = input_delta.as_flat_matrix();
    const auto product   = output_delta.as_flat_matrix() * weights.as_matrix().transpose();

    if (accumulate)   input_delta_mat.noalias() += product;
    else if (addend)  input_delta_mat.noalias()  = product + addend->as_flat_matrix();
    else              input_delta_mat.noalias()  = product;
}

void linear_forward(const TensorView& input, const TensorView& weights, const TensorView& bias,
                    TensorView& output, cublasLtEpilogue_t epilogue, TensorView* pre_activation,
                    const TensorView& weight_scale)
{
    constexpr string_view operation = "linear_forward";
    validate_linear_io(input, weights, output, false, operation);
    validate_linear_types(input, weights, output, operation);

    require_optional_tensor(input, bias, operation, "bias");
    if (!bias.empty())
    {
        throw_if(bias.get_shape().get_rank() != 1 || bias.size() != output.get_shape().back(),
                 "linear_forward: bias shape does not match the output features.");
        throw_if(bias.get_type() != output.get_type() && !(output.is_bf16() && bias.is_fp32()),
                 "linear_forward: bias dtype is incompatible with the output.");
    }

    if (pre_activation)
    {
        require_tensor(*pre_activation, operation, "pre-activation output");
        require_same_device(input, *pre_activation, operation);
        if (epilogue == CUBLASLT_EPILOGUE_GELU_AUX_BIAS)
        {
            require_same_shape(output, *pre_activation, operation);
            require_same_type(output, *pre_activation, operation);
        }
        else if (epilogue == CUBLASLT_EPILOGUE_RELU_AUX_BIAS)
        {
            const Index rows = output.size() / output.get_shape().back();
            throw_if(!pre_activation->is_int8() || pre_activation->get_shape().get_rank() != 2
                     || pre_activation->get_shape()[0] != rows
                     || pre_activation->get_shape()[1] * 8 != output.get_shape().back(),
                     "linear_forward: ReLU mask shape or dtype is incompatible with the output.");
        }
        else
        {
            throw runtime_error("linear_forward: auxiliary output requires an auxiliary epilogue.");
        }
    }

    require_optional_tensor(input, weight_scale, operation, "weight scale");
    if (weights.is_int8())
    {
        throw_if(!input.is_cuda() || !input.is_bf16() || !output.is_bf16(),
                 "linear_forward: INT8 weights require CUDA BF16 activations.");
        throw_if(weight_scale.empty() || !weight_scale.is_fp32()
                 || weight_scale.get_shape().get_rank() != 1 || weight_scale.size() != output.get_shape().back(),
                 "linear_forward: INT8 weights require one FP32 scale per output feature.");
    }

    if (input.is_cuda()) { linear_forward_gpu(input, weights, bias, output, epilogue, pre_activation, weight_scale); return; }

    throw_if(weights.is_int8(), "linear_forward: INT8 weights are CUDA-only.");

    throw_if(epilogue == CUBLASLT_EPILOGUE_GELU_AUX_BIAS,
             "linear_forward: the GELU_AUX_BIAS epilogue is CUDA-only.");

    linear_forward_cpu(input, weights, bias, output, epilogue);
}

void linear_backward(const TensorView& output_delta, const TensorView& input, const TensorView& weights,
                     const TensorView& weight_gradient, const TensorView& bias_gradient,
                     TensorView& input_delta, bool accumulate_input_delta,
                     const TensorView* drelu_mask, const TensorView* addend)
{
    constexpr string_view operation = "linear_backward";
    validate_linear_io(input, weights, output_delta, false, operation);
    require_fp32_or_bf16(output_delta, operation, "output delta");
    require_fp32_or_bf16(input, operation, "input");
    throw_if(weights.get_type() != output_delta.get_type(),
             "linear_backward: weights and output delta must use the same dtype.");
    throw_if(weights.is_int8(), "linear_backward: INT8 weights are inference-only.");

    require_tensor(weight_gradient, operation, "weight gradient");
    require_same_device(input, weight_gradient, operation);
    require_same_shape(weights, weight_gradient, operation);
    throw_if(!weight_gradient.is_fp32(), "linear_backward: weight gradient must use FP32 storage.");

    require_optional_tensor(input, bias_gradient, operation, "bias gradient");
    if (!bias_gradient.empty())
        throw_if(!bias_gradient.is_fp32() || bias_gradient.get_shape().get_rank() != 1
                 || bias_gradient.size() != output_delta.get_shape().back(),
                 "linear_backward: bias gradient must be an FP32 output-feature vector.");

    require_optional_tensor(input, input_delta, operation, "input delta");
    if (!input_delta.empty())
    {
        require_same_shape(input, input_delta, operation);
        require_same_type(input, input_delta, operation);
    }

    if (!output_delta.is_cuda())
    {
        require_cpu_fp32(output_delta, operation, "output delta");
        require_cpu_fp32(input, operation, "input");
        require_cpu_fp32(weights, operation, "weights");
    }

    // The mask was written by the producer's ReLU epilogue over this layer's
    // input, so it is (rows, input_features / 8) bytes.
    if (drelu_mask)
    {
        require_tensor(*drelu_mask, operation, "DReLU mask");
        require_same_device(output_delta, *drelu_mask, operation);
        const Index rows = input.size() / input.get_shape().back();
        throw_if(!drelu_mask->is_int8() || drelu_mask->get_shape().get_rank() != 2
                 || drelu_mask->get_shape()[0] != rows
                 || drelu_mask->get_shape()[1] * 8 != input.get_shape().back(),
                 "linear_backward: DReLU mask shape or dtype is incompatible with the input.");
    }

    throw_if(drelu_mask && (!output_delta.is_cuda() || accumulate_input_delta),
             "linear_backward: the DRELU fused input-delta path is CUDA, non-accumulating only.");

    if (addend && addend->empty()) addend = nullptr;
    if (addend)
    {
        require_tensor(*addend, operation, "input delta addend");
        require_same_shape(input_delta, *addend, operation);
        require_same_type(input_delta, *addend, operation);
        require_same_device(input_delta, *addend, operation);
        throw_if(accumulate_input_delta || input_delta.empty(),
                 "linear_backward: the input delta addend needs a non-accumulating input delta.");
    }

    if (output_delta.is_cuda())
        return linear_backward_gpu(output_delta, input, weights, weight_gradient, bias_gradient,
                                   input_delta, accumulate_input_delta, drelu_mask, addend);
    linear_backward_cpu(output_delta, input, weights, weight_gradient, bias_gradient,
                        input_delta, accumulate_input_delta, addend);
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

void linear_forward_transposed(const TensorView& input, const TensorView& embed_weight, TensorView& output,
                          const TensorView& weight_scale)
{
    constexpr string_view operation = "linear_forward_transposed";
    validate_linear_io(input, embed_weight, output, true, operation);
    validate_linear_types(input, embed_weight, output, operation);
    require_optional_tensor(input, weight_scale, operation, "weight scale");

    if (embed_weight.is_int8())
    {
        throw_if(!input.is_cuda() || !input.is_bf16() || !output.is_bf16(),
                 "linear_forward_transposed: INT8 weights require CUDA BF16 activations.");
        throw_if(weight_scale.empty() || !weight_scale.is_fp32()
                 || weight_scale.get_shape().get_rank() != 1 || weight_scale.size() != output.get_shape().back(),
                 "linear_forward_transposed: INT8 weights require one FP32 scale per output feature.");
    }

#ifdef OPENNN_HAS_CUDA
    if (input.is_cuda() && embed_weight.is_int8())
    {
        throw_if(weight_scale.empty() || !input.is_bf16() || !output.is_bf16(),
                 "linear_forward_transposed: INT8 weights require BF16 activations and a per-channel scale vector.");

        const Index in_features  = embed_weight.get_shape().back();
        const Index out_features = embed_weight.size() / in_features;
        const Index rows = input.size() / in_features;

        if (rows <= W8A16_MAX_M)
            return w8a16_linear_rows(rows, in_features, out_features, true,
                                     input.as<bfloat16>(), embed_weight.as<int8_t>(),
                                     weight_scale.as<float>(), nullptr, output.as<bfloat16>());

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
                                      input.get_data(), CUDA_R_16BF, to_int(in_features), 0,
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


#ifdef OPENNN_HAS_CUDA

static void copy_gpu(const TensorView& source, TensorView& destination)
{
    device::copy_async(destination.get_data(), source.get_data(), source.byte_size(),
                       device::CopyKind::DeviceToDevice,
                       device::get_compute_stream());
}

static void add_gpu(const TensorView& input_1,
             const TensorView& input_2,
             TensorView& output)
{

    if (input_1.is_fp32() && input_2.is_fp32() && output.is_fp32())
        return add_relu_cuda(output.size(), input_1.as<float>(), input_2.as<float>(),
                              false, output.as<float>());

    CHECK_CUDNN(cudnnOpTensor(Backend::get_cudnn_handle(),
                              Backend::get_op_tensor_add_descriptor(),
                              &one, input_1.get_descriptor(), input_1.get_data(),
                              &one, input_2.get_descriptor(), input_2.get_data(),
                              &zero, output.get_descriptor(), output.get_data()));
}

static void multiply_gpu(const TensorView& input_a, bool transpose_a,
                  const TensorView& input_b, bool transpose_b,
                  TensorView& output,
                  float alpha, float beta)
{
    const size_t rank_a = input_a.get_rank();
    const size_t rank_b = input_b.get_rank();

    int rows_a = to_int(input_a.get_shape()[rank_a - 2]);
    const int cols_a = to_int(input_a.get_shape()[rank_a - 1]);
    const int rows_b = to_int(input_b.get_shape()[rank_b - 2]);
    const int cols_b = to_int(input_b.get_shape()[rank_b - 1]);

    if (rank_b == 2 && rank_a > 2)
        rows_a = to_int(input_a.size() / cols_a);

    const int cols_out = transpose_b ? rows_b : cols_b;
    const int rows_out = transpose_a ? cols_a : rows_a;
    const int inner_dim = transpose_a ? rows_a : cols_a;

    const cublasOperation_t operation_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t operation_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    const int batch_count = to_int(input_a.size() / (rows_a * cols_a));
    const long long stride_a = rows_a * cols_a;
    const long long stride_b = rows_b * cols_b;
    const long long stride_output = output.get_shape()[output.get_rank() - 2]
                                  * output.get_shape()[output.get_rank() - 1];

    gemm_strided_batched_cuda(operation_b, operation_a,
                              cols_out, rows_out, inner_dim,
                              input_b.get_data(), input_b.cuda_dtype(), cols_b, stride_b,
                              input_a.get_data(), input_a.cuda_dtype(), cols_a, stride_a,
                              output.get_data(), output.cuda_dtype(), cols_out, stride_output,
                              batch_count,
                              alpha, beta);
}

static void softmax_gpu(TensorView& output)
{
    // cuDNN 4d descriptors hold at most INT32_MAX elements; larger tensors run row-chunked.
    constexpr Index max_descriptor_elements = numeric_limits<int>::max();

    if (output.size() <= max_descriptor_elements)
    {
        CHECK_CUDNN(cudnnSoftmaxForward(Backend::get_cudnn_handle(),
                                        CUDNN_SOFTMAX_ACCURATE,
                                        CUDNN_SOFTMAX_MODE_CHANNEL,
                                        &one,
                                        output.get_descriptor(), output.get_data(),
                                        &zero,
                                        output.get_descriptor(), output.get_data()));
        return;
    }

    const Index channels = output.get_shape().back();
    const Index total_rows = output.size() / channels;
    const Index max_rows = max_descriptor_elements / channels;
    char* const base = static_cast<char*>(output.get_data());
    const Index row_bytes = channels * type_bytes(output.get_type());

    for (Index row = 0; row < total_rows; row += max_rows)
    {
        const Index chunk_rows = min(max_rows, total_rows - row);

        const TensorView chunk(base + row * row_bytes,
                               Shape{chunk_rows, channels},
                               output.get_type(), output.get_device());

        CHECK_CUDNN(cudnnSoftmaxForward(Backend::get_cudnn_handle(),
                                        CUDNN_SOFTMAX_ACCURATE,
                                        CUDNN_SOFTMAX_MODE_CHANNEL,
                                        &one,
                                        chunk.get_descriptor(), chunk.get_data(),
                                        &zero,
                                        chunk.get_descriptor(), chunk.get_data()));
    }
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

static void linear_forward_lt_gpu(const TensorView& input, const TensorView& weights, const TensorView& bias,
                                  TensorView& output, cublasLtEpilogue_t epilogue,
                                  TensorView* pre_activation)
{
    const int input_columns  = to_int(input.get_shape().back());
    const int output_columns = to_int(weights.get_shape().back());
    const int total_rows     = to_int(input.size() / input.get_shape().back());

    const void* input_for_gemm = data_for_gemm_dtype(input, weights.get_type());
    const cudaDataType_t io_type = output.cuda_dtype();

    const void* bias_for_gemm = (bias.get_data() && output.is_bf16() && bias.is_fp32())
        ? bias_for_gemm_bf16(bias)
        : bias.get_data();

    try
    {
        run_lt_matmul_cached(
            output_columns, total_rows, input_columns,
            CUBLAS_OP_N, CUBLAS_OP_N,
            epilogue,
            weights.get_data(), input_for_gemm, output.get_data(), bias_for_gemm,
            io_type, io_type,
            pre_activation ? pre_activation->get_data() : nullptr);
    }
    catch (const runtime_error& e)
    {
        if (epilogue == CUBLASLT_EPILOGUE_GELU_AUX_BIAS && pre_activation)
        {
            linear_forward_lt_gpu(input, weights, bias, *pre_activation,
                                  CUBLASLT_EPILOGUE_BIAS, nullptr);
            copy_gpu(*pre_activation, output);
            return activation_forward_gpu(output, ActivationFunction::GELUTanh);
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
        return linear_forward_lt_gpu(input, weights, bias, output, epilogue, pre_activation);

    throw_if(weight_scale.empty() || !input.is_bf16() || !output.is_bf16(),
             "linear_forward: INT8 weights require BF16 activations and a per-channel scale vector.");

    const int input_columns  = to_int(input.get_shape().back());
    const int output_columns = to_int(weights.get_shape().back());
    const int total_rows     = to_int(input.size() / input.get_shape().back());

    const bool gemv_path = (total_rows <= W8A16_MAX_M
                            || weights.byte_size() > int8_dequant_budget_bytes)
        && (epilogue == CUBLASLT_EPILOGUE_DEFAULT || epilogue == CUBLASLT_EPILOGUE_BIAS)
        && (!bias.get_data() || bias.is_bf16());

    if (gemv_path)
        return w8a16_linear_rows(total_rows, input_columns, output_columns, false,
                                 input.as<bfloat16>(), weights.as<int8_t>(), weight_scale.as<float>(),
                                 epilogue == CUBLASLT_EPILOGUE_BIAS && bias.get_data()
                                     ? bias.as<bfloat16>() : nullptr,
                                 output.as<bfloat16>());

    bfloat16* dequantized = ensure_int8_dequant_workspace(weights.size());
    w8_dequant_cuda<bfloat16>(input_columns, output_columns, false, weights.as<int8_t>(),
                              weight_scale.as<float>(), dequantized);
    const TensorView dequantized_weights(dequantized, weights.get_shape(),
                                         Type::BF16, Device::CUDA);
    linear_forward_lt_gpu(input, dequantized_weights, bias, output, epilogue, pre_activation);
}

static void linear_backward_gpu(const TensorView& output_delta, const TensorView& input, const TensorView& weights,
                         const TensorView& weight_gradient, const TensorView& bias_gradient,
                         TensorView& input_delta, bool accumulate_input_delta,
                         const TensorView* drelu_mask, const TensorView* addend)
{
    const int input_columns  = to_int(input.get_shape().back());
    const int output_columns = to_int(output_delta.get_shape().back());
    const int total_rows     = to_int(input.size() / input.get_shape().back());

    const void* input_for_gemm = data_for_gemm_dtype(input, weights.get_type());

    const bool has_bias = bias_gradient.size() > 0;

    // The weight gradient is FP32 and accumulated in FP32 whatever the IO type,
    // so the GEMM stores it as FP32 directly (BF16 A/B, FP32 D) with the bias
    // gradient in the same epilogue - for BF16 that replaces a BF16 store, a
    // widening cast and a separate zero + column reduction, and stops rounding
    // the gradient to 8 mantissa bits on the way to the optimizer. cuBLASLt
    // support for that BF16-in/FP32-out epilogue is checked once; the first
    // failure pins the old staged path for the rest of the process.
    static atomic<bool> bf16_fp32_store_supported{true};

    // OPENNN_WGRAD_STAGED=1 forces the staged path (BF16 store + cast + separate
    // bias reduction) for A/B measurement of the epilogue's kernel choice.
    static const bool force_staged = env_flag_enabled("OPENNN_WGRAD_STAGED", false);

    const bool direct_fp32_store = !output_delta.is_bf16()
        || (bf16_fp32_store_supported.load(memory_order_relaxed) && !force_staged);

    bool stored = false;
    {
    PROFILE_SCOPE("op:linear_bwd_wgrad " + to_string(output_columns) + "x" + to_string(input_columns) + "x" + to_string(total_rows));

    // A weight gradient with a small output and a long reduction (a first
    // layer's 28 x 1024 over 7,000 rows) is a split-K job cuBLASLt's heuristics
    // handle badly here - measured 17x the time cuBLAS's GEMM takes for the same
    // shape - so those go through cublasGemmEx (multiply), the bias gradient
    // reduced by its own kernel.
    const bool skinny_wgrad = Index(output_columns) * Index(input_columns) <= Index(64) * 1024
                           && Index(total_rows) >= 4 * Index(max(output_columns, input_columns));
    if (skinny_wgrad)
    {
        const TensorView input_2d(const_cast<void*>(input_for_gemm), Shape{total_rows, input_columns},
                                  weights.get_type(), Device::CUDA);
        const TensorView output_delta_2d(output_delta.get_data(), Shape{total_rows, output_columns},
                                         output_delta.get_type(), Device::CUDA);
        TensorView weight_gradient_2d(weight_gradient.get_data(), Shape{input_columns, output_columns},
                                      Type::FP32, Device::CUDA);
        multiply(input_2d, true, output_delta_2d, false, weight_gradient_2d, 1.0f, 0.0f);

        if (has_bias)
        {
            device::set_zero_async(bias_gradient.get_data(),
                                   bias_gradient.size() * Index(sizeof(float)),
                                   device::get_compute_stream());
            output_delta.dispatch([&]<typename T>() {
                bias_grad_sum_cuda<T>(total_rows, output_columns,
                                      output_delta.as<T>(), bias_gradient.as<float>());
            });
        }
        stored = true;
    }
    else if (direct_fp32_store)
    {
        try
        {
            run_lt_matmul_cached(
                output_columns, input_columns, total_rows,
                CUBLAS_OP_N, CUBLAS_OP_T,
                has_bias ? CUBLASLT_EPILOGUE_BGRADA : CUBLASLT_EPILOGUE_DEFAULT,
                output_delta.get_data(), input_for_gemm, weight_gradient.get_data(),
                has_bias ? bias_gradient.as<float>() : nullptr,
                output_delta.cuda_dtype(),
                CUDA_R_32F);
            stored = true;
        }
        catch (const exception&)
        {
            if (!output_delta.is_bf16()) throw;
            bf16_fp32_store_supported.store(false, memory_order_relaxed);
            cerr << "linear_backward: cuBLASLt has no BF16-in/FP32-out weight-gradient "
                    "epilogue here; using BF16 store + cast for the rest of the process.\n";
            device::reset_last_error();
        }
    }

    if (!stored)
    {
        bfloat16* dw_bf16 = ensure_bf16_gradient_workspace(weight_gradient.size());
        run_lt_matmul_cached(
            output_columns, input_columns, total_rows,
            CUBLAS_OP_N, CUBLAS_OP_T,
            CUBLASLT_EPILOGUE_DEFAULT,
            output_delta.get_data(), input_for_gemm, dw_bf16, nullptr,
            output_delta.cuda_dtype(),
            CUDA_R_16BF);
        cast_bf16_to_fp32(weight_gradient.size(), dw_bf16, weight_gradient.as<float>());

        if (has_bias)
        {
            device::set_zero_async(bias_gradient.get_data(),
                                   bias_gradient.size() * Index(sizeof(float)),
                                   device::get_compute_stream());
            bias_grad_sum_cuda<bfloat16>(total_rows, output_columns,
                                         output_delta.as<bfloat16>(), bias_gradient.as<float>());
        }
    }
    }

    if (!input_delta.get_data() || input_delta.empty()) return;

    PROFILE_SCOPE("op:linear_bwd_dx " + to_string(output_columns) + "x" + to_string(input_columns) + "x" + to_string(total_rows));
    if (drelu_mask || addend)
        return run_lt_matmul_cached(
                   input_columns, total_rows, output_columns,
                   CUBLAS_OP_T, CUBLAS_OP_N,
                   drelu_mask ? CUBLASLT_EPILOGUE_DRELU : CUBLASLT_EPILOGUE_DEFAULT,
                   weights.get_data(), output_delta.get_data(), input_delta.get_data(), nullptr,
                   output_delta.cuda_dtype(), input_delta.cuda_dtype(),
                   drelu_mask ? drelu_mask->get_data() : nullptr,
                   addend ? addend->get_data() : nullptr);

    multiply(output_delta, false, weights, true, input_delta, 1.0f,
             accumulate_input_delta ? 1.0f : 0.0f);
}


#else

#define OPENNN_STUB_GPU_OP(name, sig) static void name sig { throw runtime_error(#name ": CUDA support not compiled in."); }
OPENNN_GPU_OPS(OPENNN_STUB_GPU_OP)
#undef OPENNN_STUB_GPU_OP


#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
