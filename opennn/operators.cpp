//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E R A T O R S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "operators.h"
#include "json.h"
#include "random_utilities.h"
#include "math_utilities.h"

#ifdef OPENNN_WITH_CUDA
#include "cuda_gemm.h"
#endif

namespace opennn
{

void Dropout::set_rate(float new_rate)
{
    if (new_rate < 0.0f || new_rate >= 1.0f)
        throw runtime_error("Dropout rate must be in [0, 1).");
    rate = new_rate;
}

void Dropout::apply(TensorView& output)
{
    if (!active()) return;

    Configuration::instance().is_gpu() ? apply_gpu(output) : apply_cpu(output);
}

void Dropout::apply_delta(TensorView& delta) const
{
    if (!active()) return;

    Configuration::instance().is_gpu()
        ? apply_delta_gpu(delta)
        : apply_delta_cpu(delta);
}

void Dropout::apply_cpu(TensorView& output)
{
    const Index total_size = output.size();
    if (mask_cpu.size() != total_size) mask_cpu.resize(total_size);

    const float scale = 1.0f / (1.0f - rate);
    float* data = output.as<float>();
    float* mask_data = mask_cpu.data();

    #pragma omp parallel for
    for (Index i = 0; i < total_size; ++i)
    {
        const float mask_value = random_uniform(0.0f, 1.0f) < rate ? 0.0f : scale;
        mask_data[i] = mask_value;
        data[i] *= mask_value;
    }
}

void Dropout::apply_delta_cpu(TensorView& delta) const
{
    delta.as_vector().array() *= mask_cpu.array();
}

#ifdef OPENNN_WITH_CUDA

void Dropout::apply_gpu(TensorView& output)
{
    const Index n = output.size();
    ensure_mask(n);
    const unsigned long long seed = static_cast<unsigned long long>(random_integer(0, 1 << 30));

    visit_type<Type::FP32, Type::BF16>(output.type, [&](auto info)
    {
        using T = typename decltype(info)::type;
        dropout_forward_cuda<T>(n, output.as<T>(), mask, rate, seed);
    });
}

void Dropout::apply_delta_gpu(TensorView& delta) const
{
    const Index n = delta.size();

    visit_type<Type::FP32, Type::BF16>(delta.type, [&](auto info)
    {
        using T = typename decltype(info)::type;
        dropout_backward_cuda<T>(n, delta.as<T>(), delta.as<T>(), mask, rate);
    });
}

void Dropout::ensure_mask(Index n)
{
    const size_t needed = static_cast<size_t>(n);
    if (needed <= mask_bytes) return;

    if (mask) cudaFree(mask);
    CHECK_CUDA(cudaMalloc(&mask, needed));
    mask_bytes = needed;
}

void Dropout::destroy_cuda()
{
    if (mask) { cudaFree(mask); mask = nullptr; mask_bytes = 0; }
}

#else

void Dropout::apply_gpu(TensorView&)                                 { throw runtime_error("Dropout::apply_gpu: CUDA support not compiled in."); }
void Dropout::apply_delta_gpu(TensorView&) const                     { throw runtime_error("Dropout::apply_delta_gpu: CUDA support not compiled in."); }
void Dropout::ensure_mask(Index)                                     {}
void Dropout::destroy_cuda()                                         {}

#endif

void Dropout::to_JSON(JsonWriter& w) const
{
    if (rate > 0.0f)
        add_json_field(w, "DropoutRate", to_string(rate));
}

void Dropout::from_JSON(const Json* parent)
{
    if (parent && parent->has("DropoutRate"))
        set_rate(float(read_json_type(parent, "DropoutRate")));
}


const EnumMap<Activation::Function>& Activation::map()
{
    static const vector<pair<Function, string>> entries = {
        {Function::Identity, "Identity"},
        {Function::Sigmoid,  "Sigmoid"},
        {Function::Tanh,     "Tanh"},
        {Function::ReLU,     "ReLU"},
        {Function::SELU,     "SELU"},
        {Function::Softmax,  "Softmax"}
    };
    static const EnumMap<Function> instance{entries};
    return instance;
}

Activation::Function Activation::from_string(const string& name)
{
    return map().from_string(name, Function::Identity);
}

const string& Activation::to_string(Function function)
{
    return map().to_string(function);
}

cudnnActivationMode_t Activation::to_cudnn_mode(Function function)
{
    switch (function)
    {
    case Function::Sigmoid: return CUDNN_ACTIVATION_SIGMOID;
    case Function::Tanh:    return CUDNN_ACTIVATION_TANH;
    case Function::ReLU:    return CUDNN_ACTIVATION_RELU;
    case Function::SELU:    return CUDNN_ACTIVATION_ELU;
    default:                return CUDNN_ACTIVATION_IDENTITY;
    }
}

void Activation::set_function(Function new_function)
{
    function = new_function;
#ifdef OPENNN_WITH_CUDA
    if (!descriptor) cudnnCreateActivationDescriptor(&descriptor);
    cudnnSetActivationDescriptor(descriptor, to_cudnn_mode(function), CUDNN_PROPAGATE_NAN, 0.0);
#endif
}

void Activation::set_function(const string& name)
{
    set_function(from_string(name));
}

void Activation::apply(TensorView& output)
{
    if (function == Function::Identity) return;
    if (output.empty()) return;

    Configuration::instance().is_gpu() ? apply_gpu(output) : apply_cpu(output);
}

void Activation::apply_delta(const TensorView& outputs, TensorView& delta) const
{
    if (function == Function::Identity) return;
    if (outputs.empty()) return;

    Configuration::instance().is_gpu()
        ? apply_delta_gpu(outputs, delta)
        : apply_delta_cpu(outputs, delta);
}

static constexpr float SELU_ALPHA  = 1.6732632423543772848170429916717f;
static constexpr float SELU_LAMBDA = 1.0507009873554804934193349852946f;

void Activation::apply_cpu(TensorView& output)
{
    if (function == Activation::Function::Softmax)
    {
        softmax(output);
        return;
    }

    auto a = output.as_vector().array();

    switch (function)
    {
    case Activation::Function::Sigmoid:
        a = (1.0f + (-a).exp()).inverse();
        return;
    case Activation::Function::Tanh:
        a = a.tanh();
        return;
    case Activation::Function::ReLU:
        a = a.cwiseMax(0.0f);
        return;
    case Activation::Function::SELU:
        a = SELU_LAMBDA * (a > 0.0f).select(a, SELU_ALPHA * (a.exp() - 1.0f));
        return;
    default:
        return;
    }
}

void Activation::apply_delta_cpu(const TensorView& outputs, TensorView& delta) const
{
    const auto y = outputs.as_vector().array();
    auto       d = delta.as_vector().array();

    switch (function)
    {
    case Activation::Function::Softmax:
        return;
    case Activation::Function::Sigmoid:
        d *= y * (1.0f - y);
        return;
    case Activation::Function::Tanh:
        d *= (1.0f - y.square());
        return;
    case Activation::Function::ReLU:
        d = (y > 0.0f).select(d, 0.0f);
        return;
    case Activation::Function::SELU:
        d = (y > 0.0f).select(SELU_LAMBDA * d,
                              (y + (SELU_ALPHA * SELU_LAMBDA)) * d);
        return;
    default:
        throw runtime_error("Activation::apply_delta_cpu: unknown activation function.");
    }
}

#ifdef OPENNN_WITH_CUDA

void Activation::apply_gpu(TensorView& output)
{
    if (function == Activation::Function::Softmax)
    {
        softmax(output);
        return;
    }

    CHECK_CUDNN(cudnnActivationForward(Backend::get_cudnn_handle(),
                                       descriptor,
                                       &one,
                                       output.get_descriptor(), output.data,
                                       &zero,
                                       output.get_descriptor(), output.data));
}

void Activation::apply_delta_gpu(const TensorView& outputs, TensorView& delta) const
{
    if (function == Function::Softmax) return;

    CHECK_CUDNN(cudnnActivationBackward(Backend::get_cudnn_handle(),
                                        descriptor,
                                        &one,
                                        outputs.get_descriptor(), outputs.data,
                                        delta.get_descriptor(),   delta.data,
                                        outputs.get_descriptor(), outputs.data,
                                        &zero,
                                        delta.get_descriptor(),   delta.data));
}

void Activation::destroy_cuda()
{
    if (descriptor) { cudnnDestroyActivationDescriptor(descriptor); descriptor = nullptr; }
}

#else

void Activation::apply_gpu(TensorView&)                                                     { throw runtime_error("Activation::apply_gpu: CUDA support not compiled in."); }
void Activation::apply_delta_gpu(const TensorView&, TensorView&) const                      { throw runtime_error("Activation::apply_delta_gpu: CUDA support not compiled in."); }
void Activation::destroy_cuda()                                                             {}

#endif

void Activation::to_JSON(JsonWriter& w) const
{
    add_json_field(w, "Activation", Activation::to_string(function));
}

void Activation::from_JSON(const Json* parent)
{
    if (parent && parent->has("Activation"))
        set_function(read_json_string(parent, "Activation"));
}


void BatchNorm::set(Index new_features, float new_momentum)
{
    if (new_momentum < 0.0f || new_momentum >= 1.0f)
        throw runtime_error("BatchNorm momentum must be in [0, 1).");
    features = new_features;
    momentum = new_momentum;
}

vector<pair<Shape, Type>> BatchNorm::parameter_specs() const
{
    return {
        {{features}, Type::FP32},
        {{features}, Type::FP32},
    };
}

vector<pair<Shape, Type>> BatchNorm::state_specs() const
{
    return {
        {{features}, Type::FP32},
        {{features}, Type::FP32},
    };
}

void BatchNorm::link_parameters(const vector<TensorView>& views)
{
    if (views.size() < 2) return;
    gamma = views[0];
    beta  = views[1];
    invalidate_inference_cache();
}

void BatchNorm::link_states(const vector<TensorView>& views)
{
    if (views.size() < 2) return;
    running_mean     = views[0];
    running_variance = views[1];
    invalidate_inference_cache();
}

void BatchNorm::init_defaults()
{
    if (gamma.data)            VectorMap(gamma.as<float>(),            gamma.size()).setOnes();
    if (beta.data)             VectorMap(beta.as<float>(),             beta.size()).setZero();
    if (running_mean.data)     VectorMap(running_mean.as<float>(),     running_mean.size()).setZero();
    if (running_variance.data) VectorMap(running_variance.as<float>(), running_variance.size()).setOnes();
    invalidate_inference_cache();
}

void BatchNorm::to_JSON(JsonWriter& w) const
{
    add_json_field(w, "Momentum", to_string(momentum));
}

void BatchNorm::from_JSON(const Json* parent)
{
    if (parent && parent->has("Momentum"))
        momentum = float(read_json_type(parent, "Momentum"));
}

void BatchNorm::update_inference_cache()
{
    if (!inference_cache_dirty) return;
    if (!gamma.data || !beta.data || !running_mean.data || !running_variance.data) return;

    inference_scale = gamma.as_vector().array()
                    / (running_variance.as_vector().array() + EPSILON).sqrt();
    inference_shift = beta.as_vector().array()
                    - inference_scale.array() * running_mean.as_vector().array();

    inference_cache_dirty = false;
}

void BatchNorm::apply_inference(const TensorView& input, TensorView& output)
{
    Configuration::instance().is_gpu()
        ? apply_inference_gpu(input, output)
        : apply_inference_cpu(input, output);
}

void BatchNorm::apply_training(const TensorView& input,
                               TensorView& mean, TensorView& inverse_variance,
                               TensorView& output)
{
    Configuration::instance().is_gpu()
        ? apply_training_gpu(input, mean, inverse_variance, output)
        : apply_training_cpu(input, mean, inverse_variance, output);

    invalidate_inference_cache();
}

void BatchNorm::apply_delta(const TensorView& input,
                            const TensorView& mean,
                            const TensorView& inverse_variance,
                            TensorView& gamma_gradient,
                            TensorView& beta_gradient,
                            TensorView& delta) const
{
    Configuration::instance().is_gpu()
        ? apply_delta_gpu(input, mean, inverse_variance, gamma_gradient, beta_gradient, delta)
        : apply_delta_cpu(input, mean, inverse_variance, gamma_gradient, beta_gradient, delta);
}

void BatchNorm::apply_inference_cpu(const TensorView& input, TensorView& output)
{
    update_inference_cache();

    const MatrixMap input_matrix = input.as_flat_matrix();
    MatrixMap output_matrix = output.as_flat_matrix();

    #pragma omp parallel for
    for (Index i = 0; i < input_matrix.rows(); ++i)
        output_matrix.row(i).array() = input_matrix.row(i).array() * inference_scale.transpose().array()
                                     + inference_shift.transpose().array();
}

void BatchNorm::apply_training_cpu(const TensorView& input,
                                   TensorView& mean, TensorView& inverse_variance,
                                   TensorView& output)
{
    const MatrixMap input_matrix = input.as_flat_matrix();
    MatrixMap output_matrix = output.as_flat_matrix();

    VectorMap means = mean.as_vector();
    VectorMap inverse_variances = inverse_variance.as_vector();
    VectorMap running_means = running_mean.as_vector();
    VectorMap running_variances = running_variance.as_vector();

    means.noalias() = input_matrix.colwise().mean();
    output_matrix.noalias() = input_matrix.rowwise() - means.transpose();

    inverse_variances.noalias() = output_matrix.array().square().colwise().mean().matrix();

    running_means     = running_means     * momentum + means             * (1.0f - momentum);
    running_variances = running_variances * momentum + inverse_variances * (1.0f - momentum);

    inverse_variances.array() = 1.0f / (inverse_variances.array() + EPSILON).sqrt();
    const VectorR scale = inverse_variances.array() * gamma.as_vector().array();
    const VectorMap betas = beta.as_vector();

    #pragma omp parallel for
    for (Index i = 0; i < output_matrix.rows(); ++i)
        output_matrix.row(i).array() = output_matrix.row(i).array() * scale.transpose().array() + betas.transpose().array();
}

void BatchNorm::apply_delta_cpu(const TensorView& input,
                                const TensorView& mean,
                                const TensorView& inverse_variance,
                                TensorView& gamma_gradient,
                                TensorView& beta_gradient,
                                TensorView& delta) const
{
    const Index effective_batch_size = input.size() / gamma.size();
    const float inv_N = 1.0f / static_cast<float>(effective_batch_size);
    const float N     = static_cast<float>(effective_batch_size);

    const MatrixMap input_matrix = input.as_flat_matrix();
    MatrixMap deltas             = delta.as_flat_matrix();

    const VectorMap means             = mean.as_vector();
    const VectorMap inverse_variances = inverse_variance.as_vector();
    const VectorMap gammas            = gamma.as_vector();

    VectorMap gamma_gradients = gamma_gradient.as_vector();
    VectorMap beta_gradients  = beta_gradient.as_vector();

    beta_gradients.noalias() = deltas.colwise().sum();

    gamma_gradients.noalias() = (deltas.array()
                                 * ((input_matrix.rowwise() - means.transpose()).array().rowwise()
                                    * inverse_variances.transpose().array())
                                ).matrix().colwise().sum();

    delta_scale_scratch = (gammas.array() * inverse_variances.array() * inv_N).matrix();

    #pragma omp parallel for
    for (Index i = 0; i < effective_batch_size; ++i)
    {
        auto       deltas_row = deltas.row(i).array();
        const auto x_hat_row  = (input_matrix.row(i).array() - means.transpose().array())
                              * inverse_variances.transpose().array();

        deltas_row = delta_scale_scratch.transpose().array()
                   * (N * deltas_row
                      - beta_gradients.transpose().array()
                      - x_hat_row * gamma_gradients.transpose().array());
    }
}

#ifdef OPENNN_WITH_CUDA

void BatchNorm::apply_inference_gpu(const TensorView& input, TensorView& output)
{
    CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
        Backend::get_cudnn_handle(),
        CUDNN_BATCHNORM_SPATIAL,
        &one, &zero,
        input.get_descriptor(),  input.data,
        output.get_descriptor(), output.data,
        gamma.get_descriptor(),  gamma.data, beta.data,
        running_mean.data, running_variance.data,
        EPSILON));
}

void BatchNorm::apply_training_gpu(const TensorView& input,
                                   TensorView& mean, TensorView& inverse_variance,
                                   TensorView& output)
{
    CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
        Backend::get_cudnn_handle(),
        CUDNN_BATCHNORM_SPATIAL,
        &one, &zero,
        input.get_descriptor(),  input.data,
        output.get_descriptor(), output.data,
        gamma.get_descriptor(),  gamma.data, beta.data,
        static_cast<double>(1.0f - momentum),
        running_mean.data, running_variance.data,
        EPSILON,
        mean.data, inverse_variance.data));
}

void BatchNorm::apply_delta_gpu(const TensorView& input,
                                const TensorView& mean,
                                const TensorView& inverse_variance,
                                TensorView& gamma_gradient,
                                TensorView& beta_gradient,
                                TensorView& delta) const
{
    CHECK_CUDNN(cudnnBatchNormalizationBackward(
        Backend::get_cudnn_handle(),
        CUDNN_BATCHNORM_SPATIAL,
        &one, &zero,
        &one, &zero,
        input.get_descriptor(), input.data,
        delta.get_descriptor(), delta.data,
        delta.get_descriptor(), delta.data,
        gamma.get_descriptor(), gamma.data,
        gamma_gradient.data, beta_gradient.data,
        EPSILON,
        mean.data, inverse_variance.data));
}

#else

void BatchNorm::apply_inference_gpu(const TensorView&, TensorView&)                                    { throw runtime_error("BatchNorm::apply_inference_gpu: CUDA support not compiled in."); }
void BatchNorm::apply_training_gpu (const TensorView&, TensorView&, TensorView&, TensorView&)          { throw runtime_error("BatchNorm::apply_training_gpu: CUDA support not compiled in."); }
void BatchNorm::apply_delta_gpu    (const TensorView&, const TensorView&, const TensorView&,
                                    TensorView&, TensorView&, TensorView&) const                       { throw runtime_error("BatchNorm::apply_delta_gpu: CUDA support not compiled in."); }

#endif


void Combination::set(Index new_input_features, Index new_output_features, Type new_weight_type)
{
    input_features  = new_input_features;
    output_features = new_output_features;
    weight_type     = new_weight_type;
}

vector<pair<Shape, Type>> Combination::parameter_specs() const
{
    return {
        {{output_features},                  weight_type},
        {{input_features, output_features},  weight_type},
    };
}

void Combination::link_parameters(const vector<TensorView>& views)
{
    if (views.size() < 2) return;
    bias    = views[0];
    weights = views[1];
}

void Combination::apply(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue)
{
    Configuration::instance().is_gpu()
        ? apply_gpu(input, output, epilogue)
        : apply_cpu(input, output, epilogue);
}

void Combination::apply_delta(const TensorView& output_delta,
                              const TensorView& input,
                              TensorView& input_delta,
                              TensorView& weight_gradient,
                              TensorView& bias_gradient,
                              bool accumulate_input_delta) const
{
    Configuration::instance().is_gpu()
        ? apply_delta_gpu(output_delta, input, input_delta, weight_gradient, bias_gradient, accumulate_input_delta)
        : apply_delta_cpu(output_delta, input, input_delta, weight_gradient, bias_gradient, accumulate_input_delta);
}

void Combination::apply_cpu(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue)
{
    output.as_flat_matrix().noalias() = (input.as_flat_matrix() * weights.as_matrix()).rowwise()
                                      + bias.as_vector().transpose();

    if (epilogue == CUBLASLT_EPILOGUE_RELU_BIAS)
        output.as_vector().array() = output.as_vector().array().cwiseMax(0.0f);
}

void Combination::apply_delta_cpu(const TensorView& output_delta,
                                  const TensorView& input,
                                  TensorView& input_delta, TensorView& weight_gradient, TensorView& bias_gradient,
                                  bool accumulate) const
{
    weight_gradient.as_matrix().noalias() = input.as_flat_matrix().transpose() * output_delta.as_flat_matrix();
    bias_gradient.as_vector().noalias()   = output_delta.as_flat_matrix().colwise().sum();

    if (!input_delta.data || input_delta.size() == 0) return;

    auto input_delta_mat = input_delta.as_flat_matrix();
    const auto product   = output_delta.as_flat_matrix() * weights.as_matrix().transpose();

    if (accumulate) input_delta_mat.noalias() += product;
    else            input_delta_mat.noalias()  = product;
}

#ifdef OPENNN_WITH_CUDA

void Combination::apply_gpu(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue)
{
    const int input_columns  = to_int(input.shape.back());
    const int output_columns = to_int(weights.shape.back());
    const int total_rows     = to_int(input.size() / input.shape.back());

    const void* input_for_gemm = maybe_cast(input, weights.type);

    const cudaDataType_t io_type = output.cuda_dtype();
    const LtMatmulPlan& plan = get_lt_gemm_plan(output_columns, total_rows, input_columns,
                                                CUBLAS_OP_N, CUBLAS_OP_N,
                                                epilogue, io_type, io_type);

    const float* bias_pointer = bias.as_float();
    
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer, sizeof(bias_pointer)));

    CHECK_CUBLAS(cublasLtMatmul(Backend::get_cublas_lt_handle(),
                                plan.op_desc,
                                &one,
                                weights.data,    plan.a_desc,
                                input_for_gemm,  plan.b_desc,
                                &zero,
                                output.data,     plan.c_desc,
                                output.data,     plan.d_desc,
                                plan.algo_valid ? &plan.algo : nullptr,
                                get_cublas_lt_workspace(), cublas_lt_workspace_bytes(),
                                Backend::get_compute_stream()));
}

void Combination::apply_delta_gpu(const TensorView& output_delta, const TensorView& input,
                                  TensorView& input_delta, TensorView& weight_gradient, TensorView& bias_gradient,
                                  bool accumulate_input_delta) const
{
    const int input_columns  = to_int(input.shape.back());
    const int output_columns = to_int(output_delta.shape.back());
    const int total_rows     = to_int(input.size() / input.shape.back());

    const void* input_for_gemm = maybe_cast(input, weights.type);

    const LtMatmulPlan& plan = get_lt_gemm_plan(output_columns, input_columns, total_rows,
                                                CUBLAS_OP_N, CUBLAS_OP_T,
                                                CUBLASLT_EPILOGUE_BGRADA,
                                                output_delta.cuda_dtype(),
                                                CUDA_R_32F);

    float* bias_grad_pointer = bias_gradient.as<float>();
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_grad_pointer, sizeof(bias_grad_pointer)));

    CHECK_CUBLAS(cublasLtMatmul(Backend::get_cublas_lt_handle(),
                                plan.op_desc,
                                &one,
                                output_delta.data,    plan.a_desc,
                                input_for_gemm,       plan.b_desc,
                                &zero,
                                weight_gradient.data, plan.c_desc,
                                weight_gradient.data, plan.d_desc,
                                plan.algo_valid ? &plan.algo : nullptr,
                                get_cublas_lt_workspace(), cublas_lt_workspace_bytes(),
                                Backend::get_compute_stream()));

    if (!input_delta.data || input_delta.size() == 0) return;

    const float beta_in = accumulate_input_delta ? 1.0f : 0.0f;
    multiply(output_delta, false, weights, true, input_delta, 1.0f, beta_in);
}

#else

void Combination::apply_gpu(const TensorView&, TensorView&, cublasLtEpilogue_t)                                             { throw runtime_error("Combination::apply_gpu: CUDA support not compiled in."); }
void Combination::apply_delta_gpu(const TensorView&, const TensorView&, TensorView&, TensorView&, TensorView&, bool) const  { throw runtime_error("Combination::apply_delta_gpu: CUDA support not compiled in."); }

#endif


void Convolution::set(Index new_input_h, Index new_input_w, Index new_input_c,
                      Index new_kernels_n, Index new_kernel_h, Index new_kernel_w, Index new_kernel_c,
                      Index new_row_stride, Index new_column_stride,
                      Index new_padding_h, Index new_padding_w,
                      Type new_activation_dtype)
{
    input_height     = new_input_h;
    input_width      = new_input_w;
    input_channels   = new_input_c;
    kernels_number   = new_kernels_n;
    kernel_height    = new_kernel_h;
    kernel_width     = new_kernel_w;
    kernel_channels  = new_kernel_c;
    row_stride       = new_row_stride;
    column_stride    = new_column_stride;
    padding_height   = new_padding_h;
    padding_width    = new_padding_w;
    activation_dtype = new_activation_dtype;
}

Index Convolution::get_output_height() const
{
    return (input_height + 2 * padding_height - kernel_height) / row_stride + 1;
}

Index Convolution::get_output_width() const
{
    return (input_width + 2 * padding_width - kernel_width) / column_stride + 1;
}

vector<pair<Shape, Type>> Convolution::parameter_specs() const
{
    return {
        /*Bias*/   {{kernels_number}, activation_dtype},
        /*Weight*/ {{kernels_number, kernel_height, kernel_width, kernel_channels}, activation_dtype},
    };
}

void Convolution::link_parameters(const vector<TensorView>& views)
{
    if (views.size() < 2) return;
    bias    = views[0];
    weights = views[1];
}

void Convolution::apply(const TensorView& input, TensorView& output)
{
    Configuration::instance().is_gpu() ? apply_gpu(input, output) : apply_cpu(input, output);
}

void Convolution::apply_delta(const TensorView& input,
                              const TensorView& output_delta,
                              TensorView& weight_gradient,
                              TensorView& bias_gradient,
                              TensorView& input_delta) const
{
    Configuration::instance().is_gpu()
        ? apply_delta_gpu(input, output_delta, weight_gradient, bias_gradient, input_delta)
        : apply_delta_cpu(input, output_delta, weight_gradient, bias_gradient, input_delta);
}

void Convolution::apply_cpu(const TensorView& input, TensorView& output)
{
    const TensorMap4 inputs = input.as_tensor<4>();
    const VectorMap biases = bias.as_vector();

    const Index batch_size = inputs.dimension(0);

    const Eigen::array<Index, 3> conv_dims({1, 2, 3});
    const Eigen::array<Index, 3> out_slice_shape({batch_size, output.shape[1], output.shape[2]});

    TensorMap4 outputs = output.as_tensor<4>();

    for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const TensorMap3 kernel_map = weights.as_tensor<3>(kernel_index);

        outputs.chip(kernel_index, 3).device(get_device()) =
            inputs.convolve(kernel_map, conv_dims).reshape(out_slice_shape) + biases(kernel_index);
    }
}

void Convolution::apply_delta_cpu(const TensorView& input,
                                  const TensorView& output_delta,
                                  TensorView& weight_gradient,
                                  TensorView& bias_gradient,
                                  TensorView& input_delta) const
{
    // Weight + bias gradients
    const TensorMap4 inputs        = input.as_tensor<4>();
    const TensorMap4 output_deltas = output_delta.as_tensor<4>();

    const Index kernel_size = kernel_height * kernel_width * kernel_channels;
    const Index grads_per_kernel = output_deltas.dimension(0)
                                 * output_deltas.dimension(1)
                                 * output_deltas.dimension(2);

    MatrixMap output_grads_mat(output_delta.as<float>(), grads_per_kernel, kernels_number);
    VectorMap(bias_gradient.as<float>(), kernels_number).noalias() = output_grads_mat.colwise().sum();

    float* weight_data = weight_gradient.as<float>();

    #pragma omp parallel for
    for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const auto kernel_convolution_gradients = output_deltas.chip(kernel_index, 3);

        TensorMap4 kernel_weight_gradients(weight_data + kernel_index * kernel_size,
                                           1, kernel_height, kernel_width, kernel_channels);

        kernel_weight_gradients.device(get_device()) =
            inputs.convolve(kernel_convolution_gradients, array<Index, 3>({0, 1, 2}));
    }

    // Input delta (optional)
    if (!input_delta.data || input_delta.size() == 0) return;

    TensorMap4 in_grad = input_delta.as_tensor<4>().setZero();

    const Index batch_size    = output_deltas.dimension(0);
    const Index output_height = output_deltas.dimension(1);
    const Index output_width  = output_deltas.dimension(2);

    const Index pad_height = (input_height + kernel_height - 1) - output_height;
    const Index pad_width  = (input_width  + kernel_width  - 1) - output_width;
    const Index pad_top    = pad_height / 2;
    const Index pad_bottom = pad_height - pad_top;
    const Index pad_left   = pad_width / 2;
    const Index pad_right  = pad_width - pad_left;
    const array<pair<Index, Index>, 2> paddings = {
        make_pair(pad_top, pad_bottom),
        make_pair(pad_left, pad_right)
    };

    const TensorMap4 kernels_4d = weights.as_tensor<4>();
    const Tensor4 rotated_weights = kernels_4d.reverse(array<bool, 4>({false, true, true, false}));

    vector<vector<Tensor2>> precomputed_rotated_slices(kernels_number, vector<Tensor2>(kernel_channels));

    #pragma omp parallel for
    for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        auto kernel_rotated_weights = rotated_weights.chip(kernel_index, 0);

        for (Index channel_index = 0; channel_index < kernel_channels; ++channel_index)
            precomputed_rotated_slices[kernel_index][channel_index] = kernel_rotated_weights.chip(channel_index, 2);
    }

    const array<Index, 2> convolution_dimensions_2d = {0, 1};

    for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        auto kernel_convolution_gradients = output_deltas.chip(kernel_index, 3);

        #pragma omp parallel for
        for (Index image_index = 0; image_index < batch_size; ++image_index)
        {
            const Tensor2 image_kernel_grads_padded = kernel_convolution_gradients.chip(image_index, 0).pad(paddings);

            for (Index channel_index = 0; channel_index < kernel_channels; ++channel_index)
            {
                const Tensor2 convolution_result = image_kernel_grads_padded
                    .convolve(precomputed_rotated_slices[kernel_index][channel_index], convolution_dimensions_2d);

                for (Index h = 0; h < input_height; ++h)
                    for (Index w = 0; w < input_width; ++w)
                        in_grad(image_index, h, w, channel_index) += convolution_result(h, w);
            }
        }
    }
}

#ifdef OPENNN_WITH_CUDA

void Convolution::init_cuda(Index batch_size, bool prefer_relu_algo)
{
    if (!kernel_descriptor)
        cudnnCreateFilterDescriptor(&kernel_descriptor);

    cudnnSetFilter4dDescriptor(kernel_descriptor,
                               to_cudnn(activation_dtype),
                               CUDNN_TENSOR_NHWC,
                               kernels_number, kernel_channels, kernel_height, kernel_width);

    if (!convolution_descriptor)
        cudnnCreateConvolutionDescriptor(&convolution_descriptor);

    cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                    padding_height, padding_width,
                                    row_stride, column_stride,
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);

    cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH);

    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NHWC, to_cudnn(activation_dtype),
                               static_cast<int>(batch_size),
                               static_cast<int>(kernel_channels),
                               static_cast<int>(input_height),
                               static_cast<int>(input_width));

    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NHWC, to_cudnn(activation_dtype),
                               static_cast<int>(batch_size),
                               static_cast<int>(kernels_number),
                               static_cast<int>(get_output_height()),
                               static_cast<int>(get_output_width()));

    int returned_count;

    if (prefer_relu_algo)
    {
        algorithm_forward = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }
    else
    {
        cudnnConvolutionFwdAlgoPerf_t fwd_perf;
        cudnnFindConvolutionForwardAlgorithm(Backend::get_cudnn_handle(),
                                             input_desc, kernel_descriptor, convolution_descriptor, output_desc,
                                             1, &returned_count, &fwd_perf);
        algorithm_forward = fwd_perf.algo;
    }

    cudnnGetConvolutionForwardWorkspaceSize(Backend::get_cudnn_handle(),
                                            input_desc, kernel_descriptor, convolution_descriptor, output_desc,
                                            algorithm_forward, &workspace_size);

    cudnnConvolutionBwdDataAlgoPerf_t data_perf;
    cudnnFindConvolutionBackwardDataAlgorithm(Backend::get_cudnn_handle(),
                                              kernel_descriptor, output_desc, convolution_descriptor, input_desc,
                                              1, &returned_count, &data_perf);
    algorithm_data = data_perf.algo;

    cudnnConvolutionBwdFilterAlgoPerf_t filter_perf;
    cudnnFindConvolutionBackwardFilterAlgorithm(Backend::get_cudnn_handle(),
                                                input_desc, output_desc, convolution_descriptor, kernel_descriptor,
                                                1, &returned_count, &filter_perf);
    algorithm_filter = filter_perf.algo;

    size_t bwd_data_ws = 0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(Backend::get_cudnn_handle(),
                                                 kernel_descriptor, output_desc, convolution_descriptor, input_desc,
                                                 algorithm_data, &bwd_data_ws);

    cudnnGetConvolutionBackwardFilterWorkspaceSize(Backend::get_cudnn_handle(),
                                                   input_desc, output_desc, convolution_descriptor, kernel_descriptor,
                                                   algorithm_filter, &backward_filter_workspace_size);

    workspace_size = max(workspace_size, bwd_data_ws);

    if (workspace) cudaFree(workspace);
    if (workspace_size > 0)
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

    if (backward_filter_workspace) cudaFree(backward_filter_workspace);
    if (backward_filter_workspace_size > 0)
        CHECK_CUDA(cudaMalloc(&backward_filter_workspace, backward_filter_workspace_size));

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
}

void Convolution::destroy_cuda()
{
    if (kernel_descriptor)        { cudnnDestroyFilterDescriptor(kernel_descriptor);            kernel_descriptor = nullptr; }
    if (convolution_descriptor)   { cudnnDestroyConvolutionDescriptor(convolution_descriptor);  convolution_descriptor = nullptr; }
    if (workspace)                { cudaFree(workspace);                                        workspace = nullptr; workspace_size = 0; }
    if (backward_filter_workspace){ cudaFree(backward_filter_workspace);                        backward_filter_workspace = nullptr; backward_filter_workspace_size = 0; }
}

void Convolution::apply_gpu(const TensorView& input, TensorView& output)
{
    CHECK_CUDNN(cudnnConvolutionForward(Backend::get_cudnn_handle(),
                                        &one,
                                        input.get_descriptor(),  input.data,
                                        kernel_descriptor,        weights.data,
                                        convolution_descriptor,
                                        algorithm_forward,
                                        workspace, workspace_size,
                                        &zero,
                                        output.get_descriptor(), output.data));

    CHECK_CUDNN(cudnnAddTensor(Backend::get_cudnn_handle(),
                               &one,
                               bias.get_descriptor(), bias.data,
                               &one,
                               output.get_descriptor(), output.data));
}

void Convolution::apply_delta_gpu(const TensorView& input,
                                  const TensorView& output_delta,
                                  TensorView& weight_gradient,
                                  TensorView& bias_gradient,
                                  TensorView& input_delta) const
{
    const bool bp16 = (input.type == Type::BF16);

    void* dw_dst = weight_gradient.data;
    __nv_bfloat16* dw_scratch = nullptr;

    if (bp16)
    {
        dw_scratch = get_bf16_grad_scratch(weight_gradient.size());
        dw_dst = dw_scratch;
    }

    CHECK_CUDNN(cudnnConvolutionBackwardFilter(Backend::get_cudnn_handle(),
        &one,
        input.get_descriptor(),        input.data,
        output_delta.get_descriptor(), output_delta.data,
        convolution_descriptor,
        algorithm_filter,
        backward_filter_workspace, backward_filter_workspace_size,
        &zero,
        kernel_descriptor, dw_dst));

    if (bp16)
    {
        float* dy_fp32 = get_fp32_upcast_scratch(output_delta.size());
        cast_bf16_to_fp32_cuda(output_delta.size(),
                               reinterpret_cast<const __nv_bfloat16*>(output_delta.data),
                               dy_fp32);

        TensorView dy_fp32_view = output_delta;
        dy_fp32_view.data = dy_fp32;
        dy_fp32_view.type = Type::FP32;
        dy_fp32_view.descriptor_handle.reset();

        CHECK_CUDNN(cudnnConvolutionBackwardBias(Backend::get_cudnn_handle(),
            &one,
            dy_fp32_view.get_descriptor(), dy_fp32_view.data,
            &zero,
            bias_gradient.get_descriptor(), bias_gradient.data));

        cast_bf16_to_fp32_cuda(weight_gradient.size(), dw_scratch, weight_gradient.as_float());
    }
    else
    {
        CHECK_CUDNN(cudnnConvolutionBackwardBias(Backend::get_cudnn_handle(),
            &one,
            output_delta.get_descriptor(), output_delta.data,
            &zero,
            bias_gradient.get_descriptor(), bias_gradient.data));
    }

    if (!input_delta.data || input_delta.size() == 0) return;

    CHECK_CUDNN(cudnnConvolutionBackwardData(Backend::get_cudnn_handle(),
        &one,
        kernel_descriptor, weights.data,
        output_delta.get_descriptor(), output_delta.data,
        convolution_descriptor,
        algorithm_data,
        workspace, workspace_size,
        &zero,
        input_delta.get_descriptor(), input_delta.data));
}

#else

void Convolution::init_cuda(Index, bool)                              {}
void Convolution::destroy_cuda()                                      {}
void Convolution::apply_gpu(const TensorView&, TensorView&)           { throw runtime_error("Convolution::apply_gpu: CUDA support not compiled in."); }
void Convolution::apply_delta_gpu(const TensorView&, const TensorView&,
                                  TensorView&, TensorView&, TensorView&) const { throw runtime_error("Convolution::apply_delta_gpu: CUDA support not compiled in."); }

#endif


void LayerNorm::set(Index new_sequence_length, Index new_embedding_dimension)
{
    sequence_length     = new_sequence_length;
    embedding_dimension = new_embedding_dimension;
}

vector<pair<Shape, Type>> LayerNorm::parameter_specs() const
{
    return {
        /*Gamma*/ {{embedding_dimension}, Type::FP32},
        /*Beta*/  {{embedding_dimension}, Type::FP32},
    };
}

void LayerNorm::link_parameters(const vector<TensorView>& views)
{
    if (views.size() < 2) return;
    gamma = views[0];
    beta  = views[1];
}

void LayerNorm::init_defaults()
{
    if (gamma.data) VectorMap(gamma.as<float>(), gamma.size()).setOnes();
    if (beta.data)  VectorMap(beta.as<float>(),  beta.size()).setZero();
}

void LayerNorm::apply(const TensorView& input,
                      TensorView& means, TensorView& standard_deviations, TensorView& normalized,
                      TensorView& output, Index batch_size)
{
    Configuration::instance().is_gpu()
        ? apply_gpu(input, means, standard_deviations, output, batch_size)
        : apply_cpu(input, means, standard_deviations, normalized, output, batch_size);
}

void LayerNorm::apply_delta(const TensorView& input,
                            const TensorView& output_delta,
                            const TensorView& means, const TensorView& standard_deviations,
                            const TensorView& normalized,
                            TensorView& gamma_gradient, TensorView& beta_gradient,
                            TensorView& input_delta, Index batch_size) const
{
    Configuration::instance().is_gpu()
        ? apply_delta_gpu(input, output_delta, means, standard_deviations,
                          gamma_gradient, beta_gradient, input_delta, batch_size)
        : apply_delta_cpu(output_delta, standard_deviations, normalized,
                          gamma_gradient, beta_gradient, input_delta, batch_size);
}

void LayerNorm::apply_cpu(const TensorView& input,
                          TensorView& means, TensorView& standard_deviations, TensorView& normalized,
                          TensorView& output, Index batch_size)
{
    const float* input_data = input.as<float>();
    float* means_data       = means.as<float>();
    float* stds_data        = standard_deviations.as<float>();
    float* normalized_data  = normalized.as<float>();
    float* output_data      = output.as<float>();
    const float* gamma_data = gamma.as<float>();
    const float* beta_data  = beta.as<float>();

    const Index total_rows = batch_size * sequence_length;
    const float inv_D = float(1) / to_type(embedding_dimension);

    #pragma omp parallel for
    for (Index row = 0; row < total_rows; ++row)
    {
        const float* input_row = input_data + row * embedding_dimension;
        float* norm_row        = normalized_data + row * embedding_dimension;
        float* out_row         = output_data + row * embedding_dimension;

        float sum = 0;
        float sum_sq = 0;
        for (Index dim_index = 0; dim_index < embedding_dimension; ++dim_index)
        {
            const float value = input_row[dim_index];
            sum    += value;
            sum_sq += value * value;
        }

        const float mean     = sum * inv_D;
        const float variance = sum_sq * inv_D - mean * mean;
        const float std_val  = std::sqrt(variance + EPSILON);
        const float inv_std  = float(1) / std_val;

        means_data[row] = mean;
        stds_data[row]  = std_val;

        for (Index dim_index = 0; dim_index < embedding_dimension; ++dim_index)
        {
            const float x_hat = (input_row[dim_index] - mean) * inv_std;
            norm_row[dim_index] = x_hat;
            out_row[dim_index]  = gamma_data[dim_index] * x_hat + beta_data[dim_index];
        }
    }
}

void LayerNorm::apply_delta_cpu(const TensorView& output_delta,
                                const TensorView& standard_deviations,
                                const TensorView& normalized,
                                TensorView& gamma_gradient, TensorView& beta_gradient,
                                TensorView& input_delta, Index batch_size) const
{
    const MatrixMap dy_flat   = output_delta.as_flat_matrix();
    const MatrixMap norm_flat = normalized.as_flat_matrix();

    beta_gradient.as_vector().noalias()  = dy_flat.colwise().sum();
    gamma_gradient.as_vector().noalias() = (dy_flat.array() * norm_flat.array()).matrix().colwise().sum();

    const float* dy_data    = output_delta.as<float>();
    const float* norm_data  = normalized.as<float>();
    const float* std_data   = standard_deviations.as<float>();
    const float* gamma_data = gamma.as<float>();
    float* dx_data          = input_delta.as<float>();

    const Index total_rows = batch_size * sequence_length;
    const float inv_D = float(1) / to_type(embedding_dimension);

    #pragma omp parallel for
    for (Index row = 0; row < total_rows; ++row)
    {
        const float* output_delta_row = dy_data + row * embedding_dimension;
        const float* norm_row         = norm_data + row * embedding_dimension;
        float* input_delta_row        = dx_data + row * embedding_dimension;
        const float inv_std = float(1) / std_data[row];

        float sum_scaled_grad      = 0;
        float sum_scaled_grad_norm = 0;
        for (Index dim_index = 0; dim_index < embedding_dimension; ++dim_index)
        {
            const float scaled_grad = gamma_data[dim_index] * output_delta_row[dim_index];
            sum_scaled_grad      += scaled_grad;
            sum_scaled_grad_norm += scaled_grad * norm_row[dim_index];
        }
        sum_scaled_grad      *= inv_D;
        sum_scaled_grad_norm *= inv_D;

        for (Index dim_index = 0; dim_index < embedding_dimension; ++dim_index)
        {
            const float scaled_grad = gamma_data[dim_index] * output_delta_row[dim_index];
            input_delta_row[dim_index] = (scaled_grad - sum_scaled_grad - norm_row[dim_index] * sum_scaled_grad_norm) * inv_std;
        }
    }
}

#ifdef OPENNN_WITH_CUDA

void LayerNorm::apply_gpu(const TensorView& input,
                          TensorView& means, TensorView& standard_deviations,
                          TensorView& output, Index batch_size)
{
    output.dispatch([&](auto tag) {
        using T = decltype(tag);
        layernorm_forward_cuda<T>(to_int(batch_size * sequence_length),
                                  to_int(embedding_dimension),
                                  input.as<T>(), output.as<T>(),
                                  means.as<float>(), standard_deviations.as<float>(),
                                  gamma.as<float>(), beta.as<float>(), EPSILON);
    });
}

void LayerNorm::apply_delta_gpu(const TensorView& input,
                                const TensorView& output_delta,
                                const TensorView& means, const TensorView& standard_deviations,
                                TensorView& gamma_gradient, TensorView& beta_gradient,
                                TensorView& input_delta, Index batch_size) const
{
    input.dispatch([&](auto tag) {
        using T = decltype(tag);
        layernorm_backward_cuda<T>(to_int(batch_size * sequence_length),
                                   to_int(embedding_dimension),
                                   output_delta.as<T>(), input.as<T>(),
                                   means.as<float>(), standard_deviations.as<float>(),
                                   gamma.as<float>(),
                                   input_delta.as<T>(),
                                   gamma_gradient.as<float>(), beta_gradient.as<float>());
    });
}

#else

void LayerNorm::apply_gpu(const TensorView&, TensorView&, TensorView&, TensorView&, Index)              { throw runtime_error("LayerNorm::apply_gpu: CUDA support not compiled in."); }
void LayerNorm::apply_delta_gpu(const TensorView&, const TensorView&, const TensorView&, const TensorView&,
                                TensorView&, TensorView&, TensorView&, Index) const                     { throw runtime_error("LayerNorm::apply_delta_gpu: CUDA support not compiled in."); }

#endif


void Pool::set(Index input_h, Index input_w, Index input_c,
               Index pool_h, Index pool_w,
               Index new_row_stride, Index new_column_stride,
               Index padding_h, Index padding_w,
               int new_method)
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
}

Index Pool::get_output_height() const
{
    return (input_height - pool_height + 2 * padding_height) / row_stride + 1;
}

Index Pool::get_output_width() const
{
    return (input_width - pool_width + 2 * padding_width) / column_stride + 1;
}

void Pool::apply(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training)
{
    Configuration::instance().is_gpu()
        ? apply_gpu(input, output)
        : apply_cpu(input, output, maximal_indices, is_training);
}

void Pool::apply_delta(const TensorView& input,
                       const TensorView& output,
                       const TensorView& output_delta,
                       const TensorView& maximal_indices,
                       TensorView& input_delta) const
{
    Configuration::instance().is_gpu()
        ? apply_delta_gpu(input, output, output_delta, input_delta)
        : apply_delta_cpu(output_delta, maximal_indices, input_delta);
}

void Pool::apply_cpu(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training)
{
    const TensorMap4 inputs = input.as_tensor<4>();
    TensorMap4 outputs      = output.as_tensor<4>();

    const Index batch_size    = inputs.dimension(0);
    const Index output_height = outputs.dimension(1);
    const Index output_width  = outputs.dimension(2);

    const bool is_max = (method == 0);
    const float inv_pool_size = float(1) / (pool_height * pool_width);

    TensorMap4 maximal_indices_map = (is_max && is_training)
        ? maximal_indices.as_tensor<4>()
        : TensorMap4(nullptr, 0, 0, 0, 0);

    #pragma omp parallel for collapse(2)
    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for (Index channel_index = 0; channel_index < input_channels; ++channel_index)
            for (Index output_row = 0; output_row < output_height; ++output_row)
            {
                const Index input_row_start = output_row * row_stride - padding_height;
                const Index pool_row_start  = max(Index(0), -input_row_start);
                const Index pool_row_end    = min(pool_height, input_height - input_row_start);

                for (Index output_column = 0; output_column < output_width; ++output_column)
                {
                    const Index input_column_start = output_column * column_stride - padding_width;
                    const Index pool_col_start = max(Index(0), -input_column_start);
                    const Index pool_col_end   = min(pool_width, input_width - input_column_start);

                    if (is_max)
                    {
                        float maximum_value = NEG_INFINITY;
                        Index maximal_index = 0;

                        for (Index pool_row = pool_row_start; pool_row < pool_row_end; ++pool_row)
                            for (Index pool_column = pool_col_start; pool_column < pool_col_end; ++pool_column)
                            {
                                const float value = inputs(batch_index,
                                                           input_row_start + pool_row,
                                                           input_column_start + pool_column,
                                                           channel_index);
                                if (value > maximum_value)
                                {
                                    maximum_value = value;
                                    if (is_training) maximal_index = pool_row * pool_width + pool_column;
                                }
                            }

                        outputs(batch_index, output_row, output_column, channel_index) = maximum_value;
                        if (is_training)
                            maximal_indices_map(batch_index, output_row, output_column, channel_index) = maximal_index;
                    }
                    else
                    {
                        float sum = 0;
                        for (Index pool_row = pool_row_start; pool_row < pool_row_end; ++pool_row)
                            for (Index pool_column = pool_col_start; pool_column < pool_col_end; ++pool_column)
                                sum += inputs(batch_index,
                                              input_row_start + pool_row,
                                              input_column_start + pool_column,
                                              channel_index);
                        outputs(batch_index, output_row, output_column, channel_index) = sum * inv_pool_size;
                    }
                }
            }
}

void Pool::apply_delta_cpu(const TensorView& output_delta,
                           const TensorView& maximal_indices,
                           TensorView& input_delta) const
{
    const TensorMap4 out_grads = output_delta.as_tensor<4>();
    TensorMap4 in_grads        = input_delta.as_tensor<4>().setZero();

    const Index batch_size    = out_grads.dimension(0);
    const Index output_height = out_grads.dimension(1);
    const Index output_width  = out_grads.dimension(2);

    if (method == 0)  // Max
    {
        const TensorMap4 max_indices = maximal_indices.as_tensor<4>();

        #pragma omp parallel for collapse(2)
        for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
            for (Index channel_index = 0; channel_index < input_channels; ++channel_index)
                for (Index output_row = 0; output_row < output_height; ++output_row)
                {
                    const Index input_row_start = output_row * row_stride - padding_height;
                    for (Index output_column = 0; output_column < output_width; ++output_column)
                    {
                        const Index input_column_start = output_column * column_stride - padding_width;
                        const Index maximal_index = static_cast<Index>(
                            max_indices(batch_index, output_row, output_column, channel_index));
                        const Index pool_row    = maximal_index / pool_width;
                        const Index pool_column = maximal_index % pool_width;
                        in_grads(batch_index,
                                 input_row_start + pool_row,
                                 input_column_start + pool_column,
                                 channel_index)
                            += out_grads(batch_index, output_row, output_column, channel_index);
                    }
                }
        return;
    }

    // Average
    const float inv_pool_size = float(1) / (pool_height * pool_width);

    #pragma omp parallel for collapse(2)
    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
        for (Index channel_index = 0; channel_index < input_channels; ++channel_index)
            for (Index output_row = 0; output_row < output_height; ++output_row)
            {
                const Index input_row_start = output_row * row_stride - padding_height;
                const Index pool_row_start  = max(Index(0), -input_row_start);
                const Index pool_row_end    = min(pool_height, input_height - input_row_start);

                for (Index output_column = 0; output_column < output_width; ++output_column)
                {
                    const float average_gradient =
                        out_grads(batch_index, output_row, output_column, channel_index) * inv_pool_size;
                    const Index input_column_start = output_column * column_stride - padding_width;
                    const Index pool_col_start = max(Index(0), -input_column_start);
                    const Index pool_col_end   = min(pool_width, input_width - input_column_start);

                    for (Index pool_row = pool_row_start; pool_row < pool_row_end; ++pool_row)
                        for (Index pool_column = pool_col_start; pool_column < pool_col_end; ++pool_column)
                            in_grads(batch_index,
                                     input_row_start + pool_row,
                                     input_column_start + pool_column,
                                     channel_index) += average_gradient;
                }
            }
}

#ifdef OPENNN_WITH_CUDA

void Pool::init_cuda()
{
    if (!pooling_descriptor)
        cudnnCreatePoolingDescriptor(&pooling_descriptor);

    const cudnnPoolingMode_t mode = (method == 0)
        ? CUDNN_POOLING_MAX
        : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

    cudnnSetPooling2dDescriptor(pooling_descriptor,
                                mode,
                                CUDNN_PROPAGATE_NAN,
                                pool_height, pool_width,
                                padding_height, padding_width,
                                row_stride, column_stride);
}

void Pool::destroy_cuda()
{
    if (pooling_descriptor) { cudnnDestroyPoolingDescriptor(pooling_descriptor); pooling_descriptor = nullptr; }
}

void Pool::apply_gpu(const TensorView& input, TensorView& output)
{
    CHECK_CUDNN(cudnnPoolingForward(Backend::get_cudnn_handle(),
        pooling_descriptor,
        &one,
        input.get_descriptor(), input.data,
        &zero,
        output.get_descriptor(), output.data));
}

void Pool::apply_delta_gpu(const TensorView& input,
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

void Pool::init_cuda()                                                                              {}
void Pool::destroy_cuda()                                                                           {}
void Pool::apply_gpu(const TensorView&, TensorView&)                                                { throw runtime_error("Pool::apply_gpu: CUDA support not compiled in."); }
void Pool::apply_delta_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&) const { throw runtime_error("Pool::apply_delta_gpu: CUDA support not compiled in."); }

#endif


void EmbeddingLookup::set(Index new_vocabulary_size, Index new_sequence_length, Index new_embedding_dimension)
{
    vocabulary_size     = new_vocabulary_size;
    sequence_length     = new_sequence_length;
    embedding_dimension = new_embedding_dimension;
    embedding_scale     = std::sqrt(static_cast<float>(new_embedding_dimension));
}

vector<pair<Shape, Type>> EmbeddingLookup::parameter_specs() const
{
    return {{{vocabulary_size, embedding_dimension}, Type::FP32}};
}

vector<pair<Shape, Type>> EmbeddingLookup::state_specs() const
{
    if (!add_positional_encoding) return {};
    return {{{sequence_length, embedding_dimension}, Type::FP32}};
}

void EmbeddingLookup::link_parameters(const vector<TensorView>& views)
{
    if (views.empty()) return;
    weights = views[0];
}

void EmbeddingLookup::link_states(const vector<TensorView>& views)
{
    if (views.empty()) return;
    positional_encoding = views[0];
}

void EmbeddingLookup::init_positional_encoding()
{
    if (!add_positional_encoding) return;
    if (positional_encoding.empty() || !positional_encoding.data) return;

    float* table = positional_encoding.as<float>();
    const float half_depth = float(embedding_dimension) / 2;

    VectorR divisors(embedding_dimension);
    for (Index j = 0; j < embedding_dimension; ++j)
        divisors(j) = pow(float(10000),
                          (j < Index(half_depth) ? j : j - Index(half_depth)) / half_depth);

    #pragma omp parallel for collapse(2)
    for (Index i = 0; i < sequence_length; ++i)
        for (Index j = 0; j < embedding_dimension; ++j)
            table[i * embedding_dimension + j] = (j < Index(half_depth))
                ? sin(i / divisors(j))
                : cos(i / divisors(j));
}

void EmbeddingLookup::apply(const TensorView& indices, TensorView& output)
{
    Configuration::instance().is_gpu() ? apply_gpu(indices, output) : apply_cpu(indices, output);
}

void EmbeddingLookup::apply_delta(const TensorView& indices,
                                  const TensorView& output_delta,
                                  TensorView& weight_gradient) const
{
    Configuration::instance().is_gpu()
        ? apply_delta_gpu(indices, output_delta, weight_gradient)
        : apply_delta_cpu(indices, output_delta, weight_gradient);
}

void EmbeddingLookup::apply_cpu(const TensorView& indices, TensorView& output)
{
    const Index batch_size   = output.shape[0];
    const Index total_tokens = batch_size * sequence_length;

    MatrixMap output_mat(output.as<float>(), total_tokens, embedding_dimension);
    const MatrixMap weights_mat(weights.as<float>(), vocabulary_size, embedding_dimension);
    const float* input_indices = indices.as<float>();

    static std::atomic<bool> out_of_range_warned{false};

    #pragma omp parallel for
    for (Index i = 0; i < total_tokens; ++i)
    {
        const Index token_id = static_cast<Index>(input_indices[i]);

        if (token_id < 0 || token_id >= weights_mat.rows())
        {
            if (!out_of_range_warned.exchange(true))
                std::cerr << "EmbeddingLookup warning: token id " << token_id
                          << " out of range [0, " << weights_mat.rows()
                          << "); zeroing row. Further warnings suppressed.\n";
            output_mat.row(i).setZero();
            continue;
        }

        output_mat.row(i).noalias() = weights_mat.row(token_id);

        if (scale_embedding)
            output_mat.row(i) *= embedding_scale;

        if (add_positional_encoding && token_id > 0)
        {
            const MatrixMap pe(positional_encoding.as<float>(), sequence_length, embedding_dimension);
            output_mat.row(i) += pe.row(i % sequence_length);
        }
    }
}

void EmbeddingLookup::apply_delta_cpu(const TensorView& indices,
                                      const TensorView& output_delta,
                                      TensorView& weight_gradient) const
{
    const Index total_elements = indices.size();

    MatrixMap gradients_map = output_delta.as_flat_matrix();

    if (scale_embedding)
        gradients_map *= sqrt(to_type(embedding_dimension));

    MatrixMap weight_gradients = weight_gradient.as_matrix().setZero();

    for (Index token_index = 0; token_index < total_elements; ++token_index)
    {
        const Index vocabulary_index = static_cast<Index>(indices.as<float>()[token_index]);

        if (vocabulary_index < 0 || vocabulary_index >= weight_gradients.rows())
            continue;

        weight_gradients.row(vocabulary_index).noalias() += gradients_map.row(token_index);
    }
}

#ifdef OPENNN_WITH_CUDA

void EmbeddingLookup::apply_gpu(const TensorView& indices, TensorView& output)
{
    const Index batch_size     = output.shape[0];
    const Index total_elements = batch_size * sequence_length * embedding_dimension;

    const float* pe_data = add_positional_encoding ? positional_encoding.as<float>() : nullptr;

    output.dispatch([&](auto tag) {
        using T = decltype(tag);
        embedding_forward_cuda<T>(
            total_elements,
            indices.as<float>(),
            weights.as<float>(),
            pe_data,
            output.as<T>(),
            sequence_length, embedding_dimension, vocabulary_size,
            scale_embedding, add_positional_encoding);
    });
}

void EmbeddingLookup::apply_delta_gpu(const TensorView& indices,
                                      const TensorView& output_delta,
                                      TensorView& weight_gradient) const
{
    const Index batch_size     = output_delta.shape[0];
    const Index total_elements = batch_size * sequence_length * embedding_dimension;

    CHECK_CUDA(cudaMemsetAsync(weight_gradient.data, 0, weight_gradient.byte_size(),
                               Backend::get_compute_stream()));

    output_delta.dispatch([&](auto tag) {
        using T = decltype(tag);
        embedding_backward_cuda<T>(
            total_elements,
            indices.as<float>(),
            output_delta.as<T>(),
            weight_gradient.as<float>(),
            embedding_dimension, vocabulary_size, scale_embedding);
    });
}

#else

void EmbeddingLookup::apply_gpu(const TensorView&, TensorView&)                                                { throw runtime_error("EmbeddingLookup::apply_gpu: CUDA support not compiled in."); }
void EmbeddingLookup::apply_delta_gpu(const TensorView&, const TensorView&, TensorView&) const                  { throw runtime_error("EmbeddingLookup::apply_delta_gpu: CUDA support not compiled in."); }

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
