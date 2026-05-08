//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E R A T O R S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifdef OPENNN_HAS_CUDNN_FRONTEND
#include <cuda_fp16.h>
#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;
#endif

#include "operators.h"
#include "json.h"
#include "random_utilities.h"
#include "math_utilities.h"
#include "cuda_dispatch.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"

#ifdef OPENNN_HAS_CUDA
#include "cuda_gemm.h"
#endif

namespace opennn
{

void Add::forward_propagate(ForwardPropagation& fp, size_t layer, bool) noexcept
{
    auto& fv = fp.views[layer];
    const vector<TensorView>& inputs = fv[input_slots[0]];
    TensorView& output               = fv[output_slots[0]][0];

    check(inputs, output);

    IF_GPU({
        add_gpu(inputs[0], inputs[1], output);
        for (size_t i = 2; i < inputs.size(); ++i)
            add_gpu(output, inputs[i], output);
        return;
    });

    add_cpu(inputs[0], inputs[1], output);
    for (size_t i = 2; i < inputs.size(); ++i)
        add_cpu(output, inputs[i], output);
}

void Add::back_propagate(ForwardPropagation&, BackPropagation& bp, size_t layer) const noexcept
{
    auto& dv = bp.delta_views[layer];
    const TensorView& output_delta = dv[output_delta_slots[0]][0];

    for (size_t s : input_delta_slots)
        copy(output_delta, dv[s][0]);
}

void Add::check(const vector<TensorView>& inputs, const TensorView& output) const
{
    if (inputs.size() < 2)
        throw runtime_error("Add: needs at least 2 inputs.");

    for (const TensorView& input : inputs)
        if (input.size() != output.size())
            throw runtime_error("Add: tensor dimensions do not match.");
}

void Dropout::set_rate(float new_rate)
{
    if (new_rate < 0.0f || new_rate >= 1.0f)
        throw runtime_error("Dropout rate must be in [0, 1).");
    rate = new_rate;
}

void Dropout::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept
{
    if (!is_training || !active()) return;

    auto& fv = fp.views[layer];
    TensorView& output = fv[output_slots[0]][0];

    if (!save_slots.empty())
        copy(output, fv[save_slots[0]][0]);

    IF_GPU({ apply_gpu(output); return; });
    apply_cpu(output);
}

void Dropout::apply_delta(TensorView& delta) const
{
    if (!active()) return;

    IF_GPU({ apply_delta_gpu(delta); return; });
    apply_delta_cpu(delta);
}

void Dropout::back_propagate(ForwardPropagation&, BackPropagation& bp, size_t layer) const noexcept
{
    if (!active()) return;
    apply_delta(bp.delta_views[layer][output_delta_slots[0]][0]);
}

void Dropout::apply_cpu(TensorView& output)
{
    const Index total_size = output.size();
    mask.resize_bytes(total_size * Index(sizeof(float)), Device::CPU);

    const float scale = 1.0f / (1.0f - rate);
    float* data = output.as<float>();
    float* mask_data = mask.as<float>();

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
    const Index n = delta.size();
    Map<const VectorR, AlignedMax> mask_view(mask.as<float>(), n);
    delta.as_vector().array() *= mask_view.array();
}

#ifdef OPENNN_HAS_CUDA

void Dropout::apply_gpu(TensorView& output)
{
    const Index n = output.size();
    ensure_mask(n);
    const unsigned long long seed = static_cast<unsigned long long>(random_integer(0, 1 << 30));

    visit_type<Type::FP32, Type::BF16>(output.type, [&](auto info)
    {
        using T = typename decltype(info)::type;
        dropout_forward_cuda<T>(n, output.as<T>(), mask.as<uint8_t>(), rate, seed);
    });
}

void Dropout::apply_delta_gpu(TensorView& delta) const
{
    const Index n = delta.size();

    visit_type<Type::FP32, Type::BF16>(delta.type, [&](auto info)
    {
        using T = typename decltype(info)::type;
        dropout_backward_cuda<T>(n, delta.as<T>(), delta.as<T>(), mask.as<uint8_t>(), rate);
    });
}

void Dropout::ensure_mask(Index n)
{
    if (mask.device_type != Device::CUDA || mask.bytes < n)
        mask.resize_bytes(n, Device::CUDA);
}

void Dropout::destroy_cuda()
{
    if (mask.device_type == Device::CUDA)
        mask.resize_bytes(0, Device::CUDA);
}

#else

void Dropout::apply_gpu(TensorView&)             { throw runtime_error("Dropout::apply_gpu: CUDA support not compiled in."); }
void Dropout::apply_delta_gpu(TensorView&) const { throw runtime_error("Dropout::apply_delta_gpu: CUDA support not compiled in."); }
void Dropout::ensure_mask(Index)                 {}
void Dropout::destroy_cuda()                     {}

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
    default:                return CUDNN_ACTIVATION_IDENTITY;
    }
}

void Activation::set_function(Function new_function)
{
    function = new_function;
#ifdef OPENNN_HAS_CUDA
    if (!descriptor) cudnnCreateActivationDescriptor(&descriptor);
    cudnnSetActivationDescriptor(descriptor, to_cudnn_mode(function), CUDNN_PROPAGATE_NAN, 0.0);
#endif
}

void Activation::set_function(const string& name)
{
    set_function(from_string(name));
}

void Activation::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    TensorView& output = fp.views[layer][output_slots[0]][0];

    if (function == Function::Identity || output.empty()) return;
    if (function == Function::Softmax) { softmax(output); return; }

    IF_GPU({ apply_gpu(output); return; });
    apply_cpu(output);
}

void Activation::apply_delta(const TensorView& outputs, TensorView& delta) const
{
    // Softmax delta is folded into Attention's softmax-backward step.
    if (function == Function::Identity || function == Function::Softmax || outputs.empty()) return;

    IF_GPU({ apply_delta_gpu(outputs, delta); return; });
    apply_delta_cpu(outputs, delta);
}

void Activation::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    const auto& slots = output_slots_backward.empty() ? output_slots : output_slots_backward;
    const TensorView& outputs = fp.views[layer][slots[0]][0];
    TensorView& delta         = bp.delta_views[layer][output_delta_slots[0]][0];

    apply_delta(outputs, delta);
}

void Activation::apply_cpu(TensorView& output)
{
    auto a = output.as_vector().array();

    switch (function)
    {
    case Function::Sigmoid:
        a = (1.0f + (-a).exp()).inverse();
        return;
    case Function::Tanh:
        a = a.tanh();
        return;
    case Function::ReLU:
        a = a.cwiseMax(0.0f);
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
    case Function::Sigmoid:
        d *= y * (1.0f - y);
        return;
    case Function::Tanh:
        d *= (1.0f - y.square());
        return;
    case Function::ReLU:
        d = (y > 0.0f).select(d, 0.0f);
        return;
    default:
        return;
    }
}

#ifdef OPENNN_HAS_CUDA

void Activation::apply_gpu(TensorView& output)
{
    CHECK_CUDNN(cudnnActivationForward(Backend::get_cudnn_handle(),
                                       descriptor,
                                       &one,
                                       output.get_descriptor(), output.data,
                                       &zero,
                                       output.get_descriptor(), output.data));
}

void Activation::apply_delta_gpu(const TensorView& outputs, TensorView& delta) const
{
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
    return vector<pair<Shape, Type>>(2, {Shape{features}, Type::FP32});
}

vector<pair<Shape, Type>> BatchNorm::state_specs() const
{
    return vector<pair<Shape, Type>>(2, {Shape{features}, Type::FP32});
}

void BatchNorm::link_parameters(const vector<TensorView>& views)
{
    if (views.size() < 2) return;
    gamma = views[0];
    beta  = views[1];
    invalidate_inference_cache();
}

void BatchNorm::link_gradients(const vector<TensorView>& views)
{
    if (views.size() < 2) return;
    gamma_gradient = views[0];
    beta_gradient  = views[1];
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
    if (gamma.data)            gamma.as_vector().setOnes();
    if (beta.data)             beta.as_vector().setZero();
    if (running_mean.data)     running_mean.as_vector().setZero();
    if (running_variance.data) running_variance.as_vector().setOnes();
    invalidate_inference_cache();
}

void BatchNorm::to_JSON(JsonWriter& w) const
{
    add_json_field(w, "Momentum", to_string(momentum));

    if (running_mean.data)
        add_json_field(w, "RunningMeans", vector_to_string(running_mean.as_vector()));
    if (running_variance.data)
        add_json_field(w, "RunningVariances", vector_to_string(running_variance.as_vector()));
}

void BatchNorm::from_JSON(const Json* parent)
{
    if (parent && parent->has("Momentum"))
        momentum = float(read_json_type(parent, "Momentum"));
}

void BatchNorm::load_state_from_JSON(const Json* parent)
{
    if (!parent) return;

    VectorR tmp;
    if (parent->has("RunningMeans"))
    {
        string_to_vector(read_json_string(parent, "RunningMeans"), tmp);
        if (running_mean.data && tmp.size() == running_mean.size())
            running_mean.as_vector() = tmp;
    }
    if (parent->has("RunningVariances"))
    {
        string_to_vector(read_json_string(parent, "RunningVariances"), tmp);
        if (running_variance.data && tmp.size() == running_variance.size())
            running_variance.as_vector() = tmp;
    }

    invalidate_inference_cache();
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

void BatchNorm::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept
{
    if (!active()) return;

    auto& fv = fp.views[layer];
    const TensorView& input = fv[input_slots[0]][0];
    TensorView& output      = fv[output_slots[0]][0];

    if (is_training)
    {
        TensorView& mean         = fv[output_slots[1]][0];
        TensorView& inv_variance = fv[output_slots[2]][0];

        IF_GPU({ apply_training_gpu(input, mean, inv_variance, output); invalidate_inference_cache(); return; });
        apply_training_cpu(input, mean, inv_variance, output);
        invalidate_inference_cache();
    }
    else
    {
        IF_GPU({ apply_inference_gpu(input, output); return; });
        apply_inference_cpu(input, output);
    }
}

void BatchNorm::apply_delta(const TensorView& input,
                            const TensorView& mean,
                            const TensorView& inverse_variance,
                            TensorView& delta) const
{
    IF_GPU({ apply_delta_gpu(input, mean, inverse_variance, delta); return; });
    apply_delta_cpu(input, mean, inverse_variance, delta);
}

void BatchNorm::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    if (!active()) return;

    auto& fv = fp.views[layer];
    const TensorView& input            = fv[input_slots[0]][0];
    const TensorView& mean             = fv[output_slots[1]][0];
    const TensorView& inverse_variance = fv[output_slots[2]][0];
    TensorView& delta                  = bp.delta_views[layer][output_delta_slots[0]][0];

    apply_delta(input, mean, inverse_variance, delta);
}

void BatchNorm::apply_inference_cpu(const TensorView& input, TensorView& output)
{
    update_inference_cache();

    const MatrixMap input_matrix = input.as_flat_matrix();
    MatrixMap output_matrix = output.as_flat_matrix();

    const auto scale_t = inference_scale.transpose().array();
    const auto shift_t = inference_shift.transpose().array();

    #pragma omp parallel for
    for (Index i = 0; i < input_matrix.rows(); ++i)
        output_matrix.row(i).array() = input_matrix.row(i).array() * scale_t + shift_t;
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

    const auto scale_t = scale.transpose().array();
    const auto betas_t = betas.transpose().array();

    #pragma omp parallel for
    for (Index i = 0; i < output_matrix.rows(); ++i)
        output_matrix.row(i).array() = output_matrix.row(i).array() * scale_t + betas_t;
}

void BatchNorm::apply_delta_cpu(const TensorView& input,
                                const TensorView& mean,
                                const TensorView& inverse_variance,
                                TensorView& delta) const
{
    const Index effective_batch_size = input.size() / gamma.size();
    const float N     = static_cast<float>(effective_batch_size);
    const float inv_N = 1.0f / N;

    const MatrixMap input_matrix = input.as_flat_matrix();
    MatrixMap deltas             = delta.as_flat_matrix();

    const VectorMap means             = mean.as_vector();
    const VectorMap inverse_variances = inverse_variance.as_vector();
    const VectorMap gammas            = gamma.as_vector();

    VectorMap gamma_gradients = gamma_gradient.as_vector();
    VectorMap beta_gradients  = beta_gradient.as_vector();

    beta_gradients.noalias() = deltas.colwise().sum();

    const auto means_t            = means.transpose().array();
    const auto inverse_variances_t = inverse_variances.transpose().array();

    gamma_gradients.noalias() = (deltas.array()
                                 * ((input_matrix.rowwise() - means.transpose()).array().rowwise()
                                    * inverse_variances_t)
                                ).matrix().colwise().sum();

    delta_scale_scratch = (gammas.array() * inverse_variances.array() * inv_N).matrix();

    const auto delta_scale_t   = delta_scale_scratch.transpose().array();
    const auto beta_gradient_t     = beta_gradients.transpose().array();
    const auto gamma_gradient_t    = gamma_gradients.transpose().array();

    #pragma omp parallel for
    for (Index i = 0; i < effective_batch_size; ++i)
    {
        auto       deltas_row = deltas.row(i).array();
        const auto x_hat_row  = (input_matrix.row(i).array() - means_t) * inverse_variances_t;

        deltas_row = delta_scale_t * (N * deltas_row - beta_gradient_t - x_hat_row * gamma_gradient_t);
    }
}

#ifdef OPENNN_HAS_CUDA

void BatchNorm::apply_inference_gpu(const TensorView& input, TensorView& output)
{
    CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
        Backend::get_cudnn_handle(),
        CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
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
        CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
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
                                TensorView& delta) const
{
    CHECK_CUDNN(cudnnBatchNormalizationBackward(
        Backend::get_cudnn_handle(),
        CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
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
                                    TensorView&) const                                                  { throw runtime_error("BatchNorm::apply_delta_gpu: CUDA support not compiled in."); }

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

void Combination::link_gradients(const vector<TensorView>& views)
{
    if (views.size() < 2) return;
    bias_gradient   = views[0];
    weight_gradient = views[1];
}

void Combination::set_parameters_random()
{
    if (weights.empty()) return;
    set_random_uniform(weights.as_vector());
    if (!bias.empty()) bias.fill(0.0f);
}

void Combination::set_parameters_glorot()
{
    if (weights.empty()) return;
    const float limit = sqrt(6.0f / static_cast<float>(input_features + output_features));
    set_random_uniform(weights.as_vector(), -limit, limit);
    if (!bias.empty()) bias.fill(0.0f);
}

void Combination::forward_propagate(ForwardPropagation& fp, size_t layer, bool) noexcept
{
    auto& fv = fp.views[layer];
    apply(fv[input_slots[0]][0], fv[output_slots[0]][0], CUBLASLT_EPILOGUE_BIAS);
}

void Combination::apply(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue)
{
    IF_GPU({ apply_gpu(input, output, epilogue); return; });
    apply_cpu(input, output, epilogue);
}

void Combination::apply_delta(const TensorView& output_delta,
                              const TensorView& input,
                              TensorView& input_delta,
                              bool accumulate_input_delta) const
{
    IF_GPU({ apply_delta_gpu(output_delta, input, input_delta, accumulate_input_delta); return; });
    apply_delta_cpu(output_delta, input, input_delta, accumulate_input_delta);
}

void Combination::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& fv = fp.views[layer];
    auto& dv = bp.delta_views[layer];

    const TensorView& input        = fv[input_slots[0]][0];
    const TensorView& output_delta = dv[output_delta_slots[0]][0];

    TensorView empty_input_delta;
    TensorView& input_delta = input_delta_slots.empty()
        ? empty_input_delta
        : dv[input_delta_slots[0]][0];

    apply_delta(output_delta, input, input_delta, false);
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
                                  TensorView& input_delta,
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

#ifdef OPENNN_HAS_CUDA

void Combination::apply_gpu(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue)
{
    const int input_columns  = to_int(input.shape.back());
    const int output_columns = to_int(weights.shape.back());
    const int total_rows     = to_int(input.size() / input.shape.back());

    const void* input_for_gemm = maybe_cast(input, weights.type);

    const cudaDataType_t io_type = output.cuda_dtype();
    if (!fwd_plan_ || fwd_total_rows_ != total_rows || fwd_epilogue_ != epilogue)
    {
        fwd_plan_         = &get_lt_gemm_plan(output_columns, total_rows, input_columns,
                                              CUBLAS_OP_N, CUBLAS_OP_N,
                                              epilogue, io_type, io_type);
        fwd_total_rows_   = total_rows;
        fwd_epilogue_     = epilogue;
    }
    const LtMatmulPlan& plan = *fwd_plan_;

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
                                ensure_cublas_lt_workspace(plan.workspace_size), plan.workspace_size,
                                Backend::get_compute_stream()));
}

void Combination::apply_delta_gpu(const TensorView& output_delta, const TensorView& input,
                                  TensorView& input_delta,
                                  bool accumulate_input_delta) const
{
    const int input_columns  = to_int(input.shape.back());
    const int output_columns = to_int(output_delta.shape.back());
    const int total_rows     = to_int(input.size() / input.shape.back());

    const void* input_for_gemm = maybe_cast(input, weights.type);

    const int io_dtype = static_cast<int>(output_delta.cuda_dtype());
    if (!bwd_plan_ || bwd_total_rows_ != total_rows || bwd_io_dtype_ != io_dtype)
    {
        bwd_plan_         = &get_lt_gemm_plan(output_columns, input_columns, total_rows,
                                              CUBLAS_OP_N, CUBLAS_OP_T,
                                              CUBLASLT_EPILOGUE_BGRADA,
                                              output_delta.cuda_dtype(),
                                              CUDA_R_32F);
        bwd_total_rows_   = total_rows;
        bwd_io_dtype_     = io_dtype;
    }
    const LtMatmulPlan& plan = *bwd_plan_;

    float* bias_gradient_pointer = bias_gradient.as<float>();
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_gradient_pointer, sizeof(bias_gradient_pointer)));

    CHECK_CUBLAS(cublasLtMatmul(Backend::get_cublas_lt_handle(),
                                plan.op_desc,
                                &one,
                                output_delta.data,    plan.a_desc,
                                input_for_gemm,       plan.b_desc,
                                &zero,
                                weight_gradient.data, plan.c_desc,
                                weight_gradient.data, plan.d_desc,
                                plan.algo_valid ? &plan.algo : nullptr,
                                ensure_cublas_lt_workspace(plan.workspace_size), plan.workspace_size,
                                Backend::get_compute_stream()));

    if (!input_delta.data || input_delta.size() == 0) return;

    multiply(output_delta, false, weights, true, input_delta, 1.0f,
             accumulate_input_delta ? 1.0f : 0.0f);
}

#else

void Combination::apply_gpu(const TensorView&, TensorView&, cublasLtEpilogue_t)                                             { throw runtime_error("Combination::apply_gpu: CUDA support not compiled in."); }
void Combination::apply_delta_gpu(const TensorView&, const TensorView&, TensorView&, bool) const  { throw runtime_error("Combination::apply_delta_gpu: CUDA support not compiled in."); }

#endif

void CombinationRelu::set(Index input_features, Index output_features, Type weight_type)
{
    combination.set(input_features, output_features, weight_type);
    activation.set_function(Activation::Function::ReLU);
}

void CombinationRelu::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    auto& fv = fp.views[layer];
    const TensorView& input = fv[input_slots[0]][0];
    TensorView& output      = fv[output_slots[0]][0];

    combination.apply(input, output, CUBLASLT_EPILOGUE_RELU_BIAS);
}

void CombinationRelu::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& fv = fp.views[layer];
    auto& dv = bp.delta_views[layer];

    const TensorView& output = fv[output_slots[0]][0];
    TensorView& output_delta = dv[output_delta_slots[0]][0];

    activation.apply_delta(output, output_delta);

    const TensorView& input = fv[input_slots[0]][0];

    TensorView empty_input_delta;
    TensorView& input_delta = input_delta_slots.empty()
        ? empty_input_delta
        : dv[input_delta_slots[0]][0];

    combination.apply_delta(output_delta, input, input_delta, false);
}

void Convolution::set(Index new_input_h, Index new_input_w,
                      Index new_kernels_n, Index new_kernel_h, Index new_kernel_w, Index new_kernel_c,
                      Index new_row_stride, Index new_column_stride,
                      Index new_padding_h, Index new_padding_w,
                      Type new_compute_dtype)
{
    input_height     = new_input_h;
    input_width      = new_input_w;
    kernels_number   = new_kernels_n;
    kernel_height    = new_kernel_h;
    kernel_width     = new_kernel_w;
    kernel_channels  = new_kernel_c;
    padding_height   = new_padding_h;
    padding_width    = new_padding_w;
    compute_dtype = new_compute_dtype;

#ifdef OPENNN_HAS_CUDA

    planned_batch_size = 0;

    if (kernels_number <= 0) return;

    if (!kernel_descriptor) CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                           to_cudnn(compute_dtype),
                                           CUDNN_TENSOR_NHWC,
                                           to_int(kernels_number), to_int(kernel_channels),
                                           to_int(kernel_height),  to_int(kernel_width)));

    if (!convolution_descriptor) CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                                to_int(new_padding_h), to_int(new_padding_w),
                                                to_int(new_row_stride), to_int(new_column_stride),
                                                1, 1,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    CHECK_CUDNN(cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
#endif
}

vector<pair<Shape, Type>> Convolution::parameter_specs() const
{
    return {
        {{kernels_number}, compute_dtype},                                                       // Bias
        {{kernels_number, kernel_height, kernel_width, kernel_channels}, compute_dtype},         // Weight
    };
}

void Convolution::link_parameters(const vector<TensorView>& views)
{
    if (views.size() < 2) return;
    bias    = views[0];
    weights = views[1];
}

void Convolution::link_gradients(const vector<TensorView>& views)
{
    if (views.size() < 2) return;
    bias_gradient   = views[0];
    weight_gradient = views[1];
}

void Convolution::set_parameters_random()
{
    if (weights.empty()) return;
    set_random_uniform(weights.as_vector());
    if (!bias.empty()) bias.fill(0.0f);
}

void Convolution::set_parameters_glorot()
{
    if (weights.empty()) return;
    const Index kernel_area = kernel_height * kernel_width;
    const Index fan_in  = kernel_area * kernel_channels;
    const Index fan_out = kernel_area * kernels_number;
    const float limit = sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    set_random_uniform(weights.as_vector(), -limit, limit);
    if (!bias.empty()) bias.fill(0.0f);
}

void Convolution::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    auto& fv = fp.views[layer];
    const TensorView& input = fv[input_slots[0]][0];
    TensorView& output      = fv[output_slots[0]][0];

    IF_GPU({ apply_gpu(input, output, nullptr); return; });
    apply_cpu(input, output);
}

void Convolution::apply_delta(const TensorView& input,
                              const TensorView& output_delta,
                              TensorView& input_delta) const
{
    IF_GPU({ apply_delta_gpu(input, output_delta, input_delta); return; });
    apply_delta_cpu(input, output_delta, input_delta);
}

void Convolution::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& fv = fp.views[layer];
    auto& dv = bp.delta_views[layer];

    const TensorView& input        = fv[input_slots[0]][0];
    const TensorView& output_delta = dv[output_delta_slots[0]][0];

    TensorView empty_input_delta;
    TensorView& input_delta = input_delta_slots.empty()
        ? empty_input_delta
        : dv[input_delta_slots[0]][0];

    apply_delta(input, output_delta, input_delta);
}

void Convolution::apply_cpu(const TensorView& input, TensorView& output)
{
    const TensorMap4 inputs = input.as_tensor<4>();
    const VectorMap biases = bias.as_vector();

    const Index batch_size = inputs.dimension(0);

    const array<Index, 3> conv_dims({1, 2, 3});
    const array<Index, 3> out_slice_shape({batch_size, output.shape[1], output.shape[2]});

    const array<pair<Index, Index>, 4> input_paddings = {
        make_pair(Index(0), Index(0)),
        make_pair(padding_height, padding_height),
        make_pair(padding_width,  padding_width),
        make_pair(Index(0), Index(0))
    };

    TensorMap4 outputs = output.as_tensor<4>();

    for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const TensorMap3 kernel_map = weights.as_tensor<3>(kernel_index);

        outputs.chip(kernel_index, 3).device(get_device()) =
            inputs.pad(input_paddings).convolve(kernel_map, conv_dims).reshape(out_slice_shape) + biases(kernel_index);
    }
}

void Convolution::apply_delta_cpu(const TensorView& input,
                                  const TensorView& output_delta,
                                  TensorView& input_delta) const
{
    const TensorMap4 inputs        = input.as_tensor<4>();
    const TensorMap4 output_deltas = output_delta.as_tensor<4>();

    const Index kernel_size = kernel_height * kernel_width * kernel_channels;

    MatrixMap output_gradients_mat = output_delta.as_flat_matrix();
    bias_gradient.as_vector().noalias() = output_gradients_mat.colwise().sum();

    float* weight_data = weight_gradient.as<float>();

    const array<pair<Index, Index>, 4> input_paddings = {
        make_pair(Index(0), Index(0)),
        make_pair(padding_height, padding_height),
        make_pair(padding_width,  padding_width),
        make_pair(Index(0), Index(0))
    };
    const Tensor4 padded_inputs = inputs.pad(input_paddings);

    #pragma omp parallel for
    for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const auto kernel_convolution_gradients = output_deltas.chip(kernel_index, 3);

        TensorMap4 kernel_weight_gradients(weight_data + kernel_index * kernel_size,
                                           1, kernel_height, kernel_width, kernel_channels);

        kernel_weight_gradients.device(get_device()) =
            padded_inputs.convolve(kernel_convolution_gradients, array<Index, 3>({0, 1, 2}));
    }

    if (!input_delta.data || input_delta.size() == 0) return;

    TensorMap4 in_gradient = input_delta.as_tensor<4>().setZero();

    const Index batch_size = output_deltas.dimension(0);

    const Index pad_height = input_height + kernel_height - 1 - output_deltas.dimension(1);
    const Index pad_width  = input_width  + kernel_width  - 1 - output_deltas.dimension(2);

    const array<pair<Index, Index>, 2> paddings = {
        make_pair(pad_height / 2, pad_height - pad_height / 2),
        make_pair(pad_width  / 2, pad_width  - pad_width  / 2)
    };

    const TensorMap4 kernels_4d = weights.as_tensor<4>();
    const Tensor4 rotated_weights = kernels_4d.reverse(array<bool, 4>({false, true, true, false}));

    vector<vector<Tensor2>> precomputed_rotated_slices(kernels_number, vector<Tensor2>(kernel_channels));

    #pragma omp parallel for
    for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const auto kernel_rotated_weights = rotated_weights.chip(kernel_index, 0);

        for (Index channel_index = 0; channel_index < kernel_channels; ++channel_index)
            precomputed_rotated_slices[kernel_index][channel_index] = kernel_rotated_weights.chip(channel_index, 2);
    }

    const array<Index, 2> convolution_dimensions_2d = {0, 1};

    for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const auto kernel_convolution_gradients = output_deltas.chip(kernel_index, 3);

        #pragma omp parallel for
        for (Index image_index = 0; image_index < batch_size; ++image_index)
        {
            const Tensor2 image_kernel_grads_padded = kernel_convolution_gradients.chip(image_index, 0).pad(paddings);

            for (Index channel_index = 0; channel_index < kernel_channels; ++channel_index)
                in_gradient.chip(image_index, 0).chip(channel_index, 2) +=
                    image_kernel_grads_padded.convolve(
                        precomputed_rotated_slices[kernel_index][channel_index],
                        convolution_dimensions_2d);
        }
    }
}

#ifdef OPENNN_HAS_CUDA

void Convolution::destroy_cuda()
{
    if (kernel_descriptor)      { cudnnDestroyFilterDescriptor(kernel_descriptor);           kernel_descriptor = nullptr; }
    if (convolution_descriptor) { cudnnDestroyConvolutionDescriptor(convolution_descriptor); convolution_descriptor = nullptr; }
    workspace.resize_bytes(0, Device::CUDA);
    backward_filter_workspace.resize_bytes(0, Device::CUDA);
    planned_batch_size = 0;
}
void Convolution::plan_convolution_algorithms(const TensorView& input, const TensorView& output)
{
    cudnnHandle_t handle = Backend::get_cudnn_handle();
    cudnnTensorDescriptor_t input_desc  = input.get_descriptor();
    cudnnTensorDescriptor_t output_desc = output.get_descriptor();

    // Picks the first SUCCESS entry from a Find perfs array.
    auto pick_algo = [](auto* perfs, int count, auto fallback) {
        for (int i = 0; i < count; ++i)
            if (perfs[i].status == CUDNN_STATUS_SUCCESS)
                return perfs[i].algo;
        return fallback;
    };

    int count = 0;

    cudnnConvolutionFwdAlgoPerf_t fwd_perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
        handle, input_desc, kernel_descriptor, convolution_descriptor, output_desc,
        CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &count, fwd_perf));
    algorithm_forward = pick_algo(fwd_perf, count, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);

    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
    CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
        handle, kernel_descriptor, output_desc, convolution_descriptor, input_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT, &count, bwd_data_perf));
    algorithm_data = pick_algo(bwd_data_perf, count, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1);

    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
    CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
        handle, input_desc, output_desc, convolution_descriptor, kernel_descriptor,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT, &count, bwd_filter_perf));
    algorithm_filter = pick_algo(bwd_filter_perf, count, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1);

    size_t fwd_ws = 0, bwd_data_ws = 0, bwd_filter_ws = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle, input_desc, kernel_descriptor, convolution_descriptor, output_desc,
        algorithm_forward, &fwd_ws));
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle, kernel_descriptor, output_desc, convolution_descriptor, input_desc,
        algorithm_data, &bwd_data_ws));
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle, input_desc, output_desc, convolution_descriptor, kernel_descriptor,
        algorithm_filter, &bwd_filter_ws));

    workspace.resize_bytes(Index(max(fwd_ws, bwd_data_ws)), Device::CUDA);
    backward_filter_workspace.resize_bytes(Index(bwd_filter_ws), Device::CUDA);

    planned_batch_size = input.shape[0];
}

void Convolution::apply_gpu(const TensorView& input,
                            TensorView& output,
                            cudnnActivationDescriptor_t fused_activation)
{
    if (input.shape[0] > planned_batch_size)
        plan_convolution_algorithms(input, output);

    if (fused_activation)
    {
        CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
            Backend::get_cudnn_handle(),
            &one,
            input.get_descriptor(),  input.data,
            kernel_descriptor,        weights.data,
            convolution_descriptor,
            algorithm_forward,
            workspace.data, size_t(workspace.bytes),
            &zero,
            output.get_descriptor(), output.data,
            bias.get_descriptor(),   bias.data,
            fused_activation,
            output.get_descriptor(), output.data));
        return;
    }

    CHECK_CUDNN(cudnnConvolutionForward(Backend::get_cudnn_handle(),
                                        &one,
                                        input.get_descriptor(),  input.data,
                                        kernel_descriptor,        weights.data,
                                        convolution_descriptor,
                                        algorithm_forward,
                                        workspace.data, size_t(workspace.bytes),
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
                                  TensorView& input_delta) const
{
    assert(output_delta.type == input.type);
    assert(weight_gradient.type == Type::FP32);

    const bool bf16 = (input.type == Type::BF16);

    void* weight_gradient_buffer = weight_gradient.data;
    __nv_bfloat16* weight_gradient_bf16_scratch = nullptr;

    if (bf16)
    {
        weight_gradient_bf16_scratch = ensure_bf16_gradient_scratch(weight_gradient.size());
        weight_gradient_buffer = weight_gradient_bf16_scratch;
    }

    CHECK_CUDNN(cudnnConvolutionBackwardFilter(Backend::get_cudnn_handle(),
        &one,
        input.get_descriptor(),        input.data,
        output_delta.get_descriptor(), output_delta.data,
        convolution_descriptor,
        algorithm_filter,
        backward_filter_workspace.data, size_t(backward_filter_workspace.bytes),
        &zero,
        kernel_descriptor, weight_gradient_buffer));

    if (bf16)
    {
        float* output_delta_fp32 = ensure_fp32_upcast_scratch(output_delta.size());
        cast_bf16_to_fp32_cuda(output_delta.size(),
                               reinterpret_cast<const __nv_bfloat16*>(output_delta.data),
                               output_delta_fp32);

        TensorView output_delta_fp32_view(output_delta_fp32, output_delta.shape, Type::FP32);

        CHECK_CUDNN(cudnnConvolutionBackwardBias(Backend::get_cudnn_handle(),
            &one,
            output_delta_fp32_view.get_descriptor(), output_delta_fp32_view.data,
            &zero,
            bias_gradient.get_descriptor(), bias_gradient.data));

        cast_bf16_to_fp32_cuda(weight_gradient.size(), weight_gradient_bf16_scratch, weight_gradient.as_float());
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
        workspace.data, size_t(workspace.bytes),
        &zero,
        input_delta.get_descriptor(), input_delta.data));
}

#else

void Convolution::destroy_cuda()                                                                  {}
void Convolution::apply_gpu(const TensorView&, TensorView&, cudnnActivationDescriptor_t)          { throw runtime_error("Convolution::apply_gpu: CUDA support not compiled in."); }
void Convolution::apply_delta_gpu(const TensorView&, const TensorView&, TensorView&) const { throw runtime_error("Convolution::apply_delta_gpu: CUDA support not compiled in."); }

#endif


void ConvolutionRelu::set(Index input_h, Index input_w,
                          Index kernels_n, Index kernel_h, Index kernel_w, Index kernel_c,
                          Index row_stride, Index column_stride,
                          Index padding_h, Index padding_w,
                          Type compute_dtype)
{
    convolution.set(input_h, input_w,
                    kernels_n, kernel_h, kernel_w, kernel_c,
                    row_stride, column_stride,
                    padding_h, padding_w,
                    compute_dtype);

    activation.set_function(Activation::Function::ReLU);
}

void ConvolutionRelu::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    auto& fv = fp.views[layer];
    const TensorView& input = fv[input_slots[0]][0];
    TensorView& output      = fv[output_slots[0]][0];

    IF_GPU({ convolution.apply_gpu(input, output, activation.descriptor); return; });
    convolution.apply_cpu(input, output);
    activation.apply_cpu(output);
}

void ConvolutionRelu::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& fv = fp.views[layer];
    auto& dv = bp.delta_views[layer];

    const TensorView& output = fv[output_slots[0]][0];
    TensorView& output_delta = dv[output_delta_slots[0]][0];

    activation.apply_delta(output, output_delta);

    const TensorView& input = fv[input_slots[0]][0];

    TensorView empty_input_delta;
    TensorView& input_delta = input_delta_slots.empty()
        ? empty_input_delta
        : dv[input_delta_slots[0]][0];

    convolution.apply_delta(input, output_delta, input_delta);
}


void LayerNorm::set(Index new_sequence_length, Index new_embedding_dimension)
{
    sequence_length     = new_sequence_length;
    embedding_dimension = new_embedding_dimension;
}

vector<pair<Shape, Type>> LayerNorm::parameter_specs() const
{
    // Gamma, Beta
    return vector<pair<Shape, Type>>(2, {Shape{embedding_dimension}, Type::FP32});
}

void LayerNorm::link_parameters(const vector<TensorView>& views)
{
    if (views.size() < 2) return;
    gamma = views[0];
    beta  = views[1];
}

void LayerNorm::link_gradients(const vector<TensorView>& views)
{
    if (views.size() < 2) return;
    gamma_gradient = views[0];
    beta_gradient  = views[1];
}

void LayerNorm::init_defaults()
{
    if (gamma.data) gamma.as_vector().setOnes();
    if (beta.data)  beta.as_vector().setZero();
}

void LayerNorm::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    auto& fv = fp.views[layer];
    const TensorView& input = fv[input_slots[0]][0];
    TensorView& means       = fv[output_slots[0]][0];
    TensorView& stds        = fv[output_slots[1]][0];
    TensorView& normalized  = fv[output_slots[2]][0];
    TensorView& output      = fv[output_slots[3]][0];

    IF_GPU({ apply_gpu(input, means, stds, output); return; });
    apply_cpu(input, means, stds, normalized, output);
}

void LayerNorm::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& fv = fp.views[layer];
    auto& dv = bp.delta_views[layer];

    const TensorView& input        = fv[input_slots[0]][0];
    const TensorView& means        = fv[output_slots[0]][0];
    const TensorView& stds         = fv[output_slots[1]][0];
    const TensorView& normalized   = fv[output_slots[2]][0];
    const TensorView& output_delta = dv[output_delta_slots[0]][0];
    TensorView& input_delta        = dv[input_delta_slots[0]][0];

    IF_GPU({ apply_delta_gpu(input, output_delta, means, stds, input_delta); return; });
    apply_delta_cpu(output_delta, stds, normalized, input_delta);
}

void LayerNorm::apply_cpu(const TensorView& input,
                          TensorView& means, TensorView& standard_deviations, TensorView& normalized,
                          TensorView& output)
{
    const float* input_data = input.as<float>();
    float* means_data       = means.as<float>();
    float* stds_data        = standard_deviations.as<float>();
    float* normalized_data  = normalized.as<float>();
    float* output_data      = output.as<float>();
    const float* gamma_data = gamma.as<float>();
    const float* beta_data  = beta.as<float>();

    const Index total_rows = input.shape[0] * sequence_length;
    const float inv_D = 1.0f / to_type(embedding_dimension);

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
        const float inv_std  = 1.0f / std_val;

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
                                TensorView& input_delta) const
{
    const MatrixMap output_delta_flat = output_delta.as_flat_matrix();
    const MatrixMap norm_flat         = normalized.as_flat_matrix();

    beta_gradient.as_vector().noalias()  = output_delta_flat.colwise().sum();
    gamma_gradient.as_vector().noalias() = (output_delta_flat.array() * norm_flat.array()).matrix().colwise().sum();

    const float* output_delta_data = output_delta.as<float>();
    const float* norm_data         = normalized.as<float>();
    const float* std_data          = standard_deviations.as<float>();
    const float* gamma_data        = gamma.as<float>();
    float* input_delta_data        = input_delta.as<float>();

    const Index total_rows = output_delta.shape[0] * sequence_length;
    const float inv_D = 1.0f / to_type(embedding_dimension);

    #pragma omp parallel for
    for (Index row = 0; row < total_rows; ++row)
    {
        const float* output_delta_row = output_delta_data + row * embedding_dimension;
        const float* norm_row         = norm_data + row * embedding_dimension;
        float* input_delta_row        = input_delta_data + row * embedding_dimension;
        const float inv_std = 1.0f / std_data[row];

        float sum_scaled_gradient      = 0;
        float sum_scaled_gradient_norm = 0;
        for (Index dim_index = 0; dim_index < embedding_dimension; ++dim_index)
        {
            const float scaled_gradient = gamma_data[dim_index] * output_delta_row[dim_index];
            sum_scaled_gradient      += scaled_gradient;
            sum_scaled_gradient_norm += scaled_gradient * norm_row[dim_index];
        }
        sum_scaled_gradient      *= inv_D;
        sum_scaled_gradient_norm *= inv_D;

        for (Index dim_index = 0; dim_index < embedding_dimension; ++dim_index)
        {
            const float scaled_gradient = gamma_data[dim_index] * output_delta_row[dim_index];
            input_delta_row[dim_index] = (scaled_gradient - sum_scaled_gradient - norm_row[dim_index] * sum_scaled_gradient_norm) * inv_std;
        }
    }
}

#ifdef OPENNN_HAS_CUDA

void LayerNorm::apply_gpu(const TensorView& input,
                          TensorView& means, TensorView& standard_deviations,
                          TensorView& output)
{
    output.dispatch([&](auto tag) {
        using T = decltype(tag);
        layernorm_forward_cuda<T>(to_int(input.shape[0] * sequence_length),
                                  to_int(embedding_dimension),
                                  input.as<T>(), output.as<T>(),
                                  means.as<float>(), standard_deviations.as<float>(),
                                  gamma.as<float>(), beta.as<float>(), EPSILON);
    });
}

void LayerNorm::apply_delta_gpu(const TensorView& input,
                                const TensorView& output_delta,
                                const TensorView& means, const TensorView& standard_deviations,
                                TensorView& input_delta) const
{
    input.dispatch([&](auto tag) {
        using T = decltype(tag);
        layernorm_backward_cuda<T>(to_int(input.shape[0] * sequence_length),
                                   to_int(embedding_dimension),
                                   output_delta.as<T>(), input.as<T>(),
                                   means.as<float>(), standard_deviations.as<float>(),
                                   gamma.as<float>(),
                                   input_delta.as<T>(),
                                   gamma_gradient.as<float>(), beta_gradient.as<float>());
    });
}

#else

void LayerNorm::apply_gpu(const TensorView&, TensorView&, TensorView&, TensorView&)              { throw runtime_error("LayerNorm::apply_gpu: CUDA support not compiled in."); }
void LayerNorm::apply_delta_gpu(const TensorView&, const TensorView&, const TensorView&, const TensorView&,
                                TensorView&) const                                                  { throw runtime_error("LayerNorm::apply_delta_gpu: CUDA support not compiled in."); }

#endif

void MultiHeadProjection::set(Index new_input_features, Index new_heads_number,
                              Index new_head_dimension, Type new_compute_dtype)
{
    input_features   = new_input_features;
    heads_number     = new_heads_number;
    head_dimension   = new_head_dimension;
    compute_dtype = new_compute_dtype;

    combination.set(input_features, heads_number * head_dimension, compute_dtype);
}

void MultiHeadProjection::apply(const TensorView& input, TensorView& head_output, float* scratch)
{
    const Index batch_size = input.shape[0];
    const Index seq_len    = input.shape[1];
    const Index rows       = batch_size * seq_len;

    TensorView input_2d   = input.reshape({rows, input_features});
    TensorView scratch_2d(scratch, {rows, input_features}, head_output.type);
    TensorView scratch_4d(scratch, {batch_size, seq_len, heads_number, head_dimension}, head_output.type);

    combination.apply(input_2d, scratch_2d);
    split_heads(scratch_4d, head_output);
}

void MultiHeadProjection::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    auto& fv = fp.views[layer];
    const auto& input_views = fv[input_slots[0]];
    const TensorView& input = input_views[min(input_view_index, input_views.size() - 1)];
    TensorView& output = fv[output_slots[0]][0];
    float* scratch = fv[scratch_slots[0]][0].as<float>();
    apply(input, output, scratch);
}

void MultiHeadProjection::apply_delta(const TensorView& head_gradient,
                                      const TensorView& input,
                                      TensorView& input_gradient,
                                      bool accumulate,
                                      float* scratch) const
{
    const Index batch_size = input.shape[0];
    const Index seq_len    = input.shape[1];
    const Index rows       = batch_size * seq_len;

    TensorView scratch_4d(scratch, {batch_size, seq_len, heads_number, head_dimension}, head_gradient.type);
    merge_heads(head_gradient, scratch_4d);

    TensorView scratch_2d(scratch, {rows, input_features}, head_gradient.type);
    TensorView input_2d          = input.reshape({rows, input_features});
    TensorView input_gradient_2d = input_gradient.reshape({rows, input_features});

    combination.apply_delta(scratch_2d, input_2d, input_gradient_2d, accumulate);
}

void MultiHeadProjection::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& fv = fp.views[layer];
    auto& dv = bp.delta_views[layer];

    const auto& input_views = fv[input_slots[0]];
    const TensorView& input = input_views[min(input_view_index, input_views.size() - 1)];
    const bool self_attention = (input_views.size() == 1);

    const TensorView head_gradient(dv[output_delta_slots[0]][0].as<float>(),
                                   {fp.batch_size, heads_number, input.shape[1], head_dimension},
                                   compute_dtype);

    TensorView& input_delta = dv[(self_attention ? input_delta_slots_self : input_delta_slots_cross)[0]][0];
    const bool accumulate   = self_attention ? accumulate_input_delta_self : accumulate_input_delta_cross;

    apply_delta(head_gradient, input, input_delta, accumulate,
                fv[scratch_slots[0]][0].as<float>());
}


void Attention::set(Index new_heads_number, Index new_head_dimension,
                    Index new_query_sequence_length, Index new_source_sequence_length,
                    bool new_use_causal_mask, Type new_compute_dtype)
{
    heads_number = new_heads_number;
    head_dimension = new_head_dimension;
    query_sequence_length = new_query_sequence_length;
    source_sequence_length = new_source_sequence_length;
    use_causal_mask = new_use_causal_mask;
    compute_dtype = new_compute_dtype;

    if (use_causal_mask && query_sequence_length > 0 && source_sequence_length > 0)
    {
        causal_mask.resize(query_sequence_length, source_sequence_length);
        for (Index row = 0; row < query_sequence_length; ++row)
            for (Index column = 0; column < source_sequence_length; ++column)
                causal_mask(row, column) = (column > row) ? NEG_INFINITY : 0.0f;
    }
    else
    {
        causal_mask.resize(0, 0);
    }
}

float Attention::scaling_factor() const
{
    return (head_dimension == 0) ? 0.25f : 1.0f / float(sqrt(head_dimension));
}

vector<pair<Shape, Type>> Attention::forward_scratch_specs(Index batch_size) const
{
    bool sdpa_will_be_used = false;
#ifdef OPENNN_HAS_CUDNN_FRONTEND
    sdpa_will_be_used =
            Configuration::instance().is_gpu()
         && compute_dtype == Type::BF16
         && !dropout.active();
#endif

    if (sdpa_will_be_used)
        return vector<pair<Shape, Type>>(2, {Shape{}, compute_dtype});

    const Shape attention_shape = {batch_size, heads_number,
                                   query_sequence_length, source_sequence_length};
    const Shape dropout_shape = dropout.active() ? attention_shape : Shape{};
    
    return {
        {attention_shape, compute_dtype}, // AttentionWeights
        {dropout_shape,   compute_dtype}, // AttentionWeightsDropped
    };
}

#ifdef OPENNN_HAS_CUDNN_FRONTEND

namespace
{

fe::DataType_t to_fe_dtype(Type t)
{
    switch (t)
    {
        case Type::FP32: return fe::DataType_t::FLOAT;
        case Type::BF16: return fe::DataType_t::BFLOAT16;
        default:         return fe::DataType_t::FLOAT;
    }
}

}  // namespace

#endif

struct Attention::SDPACache
{
    struct CacheKey
    {
        Index batch_size = 0;
        Index q_seq      = 0;
        Index src_seq    = 0;
        Index heads      = 0;
        Index head_dim   = 0;
        Type  dtype      = Type::FP32;
        bool  dropout_active = false;
        bool  causal         = false;
        bool  is_training    = false;

        bool operator==(const CacheKey& other) const
        {
            return batch_size == other.batch_size && q_seq == other.q_seq
                && src_seq == other.src_seq && heads == other.heads
                && head_dim == other.head_dim && dtype == other.dtype
                && dropout_active == other.dropout_active
                && causal == other.causal && is_training == other.is_training;
        }
    };

    struct CacheKeyHash
    {
        size_t operator()(const CacheKey& k) const
        {
            return hash_combine(k.batch_size, k.q_seq, k.src_seq, k.heads, k.head_dim,
                                Index(k.dtype),
                                Index(k.dropout_active), Index(k.causal), Index(k.is_training));
        }
    };

#ifdef OPENNN_HAS_CUDNN_FRONTEND
    struct Entry
    {
        // Forward graph
        std::shared_ptr<fe::graph::Graph> fwd_graph;
        std::shared_ptr<fe::graph::Tensor_attributes> fwd_Q, fwd_K, fwd_V, fwd_O, fwd_Stats;
        std::shared_ptr<fe::graph::Tensor_attributes> fwd_Seed, fwd_Offset;
        void* fwd_workspace_buf = nullptr;

        // Backward graph (built lazily on first apply_delta_gpu)
        std::shared_ptr<fe::graph::Graph> bwd_graph;
        std::shared_ptr<fe::graph::Tensor_attributes> bwd_Q, bwd_K, bwd_V, bwd_O, bwd_dO, bwd_Stats;
        std::shared_ptr<fe::graph::Tensor_attributes> bwd_dQ, bwd_dK, bwd_dV;
        void* bwd_workspace_buf = nullptr;

        // Shared (forward writes, backward reads). LSE stats from softmax.
        void* stats_buf = nullptr;
    };

    std::unordered_map<CacheKey, Entry, CacheKeyHash> entries;

    // 1-element shortcut: Attention is typically called with the same shape
    // and dtype across all training/inference steps. After the first call this
    // skips the unordered_map hash lookup entirely.
    mutable Entry*   last_entry_ = nullptr;
    mutable CacheKey last_key_;
    mutable bool     last_valid_ = false;

    Entry& get_or_create_entry(const CacheKey& key)
    {
        if (last_valid_ && key == last_key_) return *last_entry_;
        Entry& e = entries[key];
        last_entry_ = &e;
        last_key_   = key;
        last_valid_ = true;
        return e;
    }

    Entry* find_entry(const CacheKey& key) const
    {
        if (last_valid_ && key == last_key_) return last_entry_;
        const auto it = entries.find(key);
        if (it == entries.end()) return nullptr;
        last_entry_ = const_cast<Entry*>(&it->second);
        last_key_   = key;
        last_valid_ = true;
        return last_entry_;
    }

    ~SDPACache()
    {
#ifdef OPENNN_HAS_CUDA
        for (auto& [_, e] : entries)
        {
            if (e.fwd_workspace_buf) cudaFree(e.fwd_workspace_buf);
            if (e.bwd_workspace_buf) cudaFree(e.bwd_workspace_buf);
            if (e.stats_buf)         cudaFree(e.stats_buf);
        }
#endif
    }
#endif  // OPENNN_HAS_CUDNN_FRONTEND
};

#ifdef OPENNN_HAS_CUDNN_FRONTEND

namespace
{

auto sdpa_check = [](auto s, const string& what) {
    if (s.is_bad())
        throw runtime_error("SDPA " + what + ": " + s.get_message());
};

// {B, H, S, D} contiguous tensor input.
std::shared_ptr<fe::graph::Tensor_attributes>
bhsd_input(fe::graph::Graph& graph, const char* name, int64_t B, int64_t H, int64_t S, int64_t D)
{
    return graph.tensor(fe::graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim   ({B, H, S, D})
                        .set_stride({H * S * D, S * D, D, 1}));
}
void bhsd_output(std::shared_ptr<fe::graph::Tensor_attributes>& T,
                 int64_t B, int64_t H, int64_t S, int64_t D)
{
    T->set_output(true).set_dim({B, H, S, D}).set_stride({H * S * D, S * D, D, 1});
}

void build_sdpa_graph_common(fe::graph::Graph& graph, Type dtype)
{
    graph.set_io_data_type(to_fe_dtype(dtype))
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);
}
void require_attention_scratch(const TensorView& attention_weights, const string& context)
{
    if (attention_weights.empty())
        throw runtime_error("Attention: " + context +
            " — set_dropout_rate must be called before compiling the network on GPU "
            "(see Attention::forward_scratch_specs).");
}

void finalize_sdpa_graph(fe::graph::Graph& graph, cudnnHandle_t handle, const string& tag)
{
    sdpa_check(graph.validate(),                                                tag + " validate");
    sdpa_check(graph.build_operation_graph(handle),                             tag + " build_operation_graph");
    sdpa_check(graph.create_execution_plans({fe::HeurMode_t::A}),               tag + " create_execution_plans");
    sdpa_check(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE), tag + " build_plans");
}

}  // namespace

static void build_sdpa_forward_graph(Attention::SDPACache::Entry& entry,
                                      const Attention::SDPACache::CacheKey& k,
                                      cudnnHandle_t handle,
                                      float dropout_rate)
{
    const auto graph = make_shared<fe::graph::Graph>();
    build_sdpa_graph_common(*graph, k.dtype);

    entry.fwd_Q = bhsd_input(*graph, "Q", k.batch_size, k.heads, k.q_seq,   k.head_dim);
    entry.fwd_K = bhsd_input(*graph, "K", k.batch_size, k.heads, k.src_seq, k.head_dim);
    entry.fwd_V = bhsd_input(*graph, "V", k.batch_size, k.heads, k.src_seq, k.head_dim);

    auto sdpa_options = fe::graph::SDPA_attributes()
                        .set_name("flash_attn_fwd")
                        .set_is_inference(!k.is_training)
                        .set_causal_mask(k.causal)
                        .set_attn_scale(1.0f / std::sqrt(float(k.head_dim)));

    if (k.dropout_active)
    {
        entry.fwd_Seed   = graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("Seed").set_dim({1,1,1,1}).set_stride({1,1,1,1})
                                         .set_data_type(fe::DataType_t::INT64));
        entry.fwd_Offset = graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("Offset").set_dim({1,1,1,1}).set_stride({1,1,1,1})
                                         .set_data_type(fe::DataType_t::INT64));
        sdpa_options.set_dropout(dropout_rate, entry.fwd_Seed, entry.fwd_Offset);
    }

    auto [O, Stats] = graph->sdpa(entry.fwd_Q, entry.fwd_K, entry.fwd_V, sdpa_options);

    bhsd_output(O, k.batch_size, k.heads, k.q_seq, k.head_dim);
    entry.fwd_O = O;

    if (k.is_training && Stats)
    {
        Stats->set_output(true)
              .set_data_type(fe::DataType_t::FLOAT)
              .set_dim({k.batch_size, k.heads, k.q_seq, 1})
              .set_stride({k.heads * k.q_seq, k.q_seq, 1, 1});
        entry.fwd_Stats = Stats;
    }

    finalize_sdpa_graph(*graph, handle, "fwd");

    int64_t ws = 0;
    graph->get_workspace_size(ws);
    if (ws > 0)
        CHECK_CUDA(cudaMalloc(&entry.fwd_workspace_buf, size_t(ws)));

    if (k.is_training)
    {
        const size_t stats_bytes = size_t(k.batch_size * k.heads * k.q_seq) * sizeof(float);
        CHECK_CUDA(cudaMalloc(&entry.stats_buf, stats_bytes));
    }

    entry.fwd_graph = graph;
}

static void build_sdpa_backward_graph(Attention::SDPACache::Entry& entry,
                                       const Attention::SDPACache::CacheKey& k,
                                       cudnnHandle_t handle)
{
    const auto graph = make_shared<fe::graph::Graph>();
    build_sdpa_graph_common(*graph, k.dtype);

    entry.bwd_Q  = bhsd_input(*graph, "Q_bwd",  k.batch_size, k.heads, k.q_seq,   k.head_dim);
    entry.bwd_K  = bhsd_input(*graph, "K_bwd",  k.batch_size, k.heads, k.src_seq, k.head_dim);
    entry.bwd_V  = bhsd_input(*graph, "V_bwd",  k.batch_size, k.heads, k.src_seq, k.head_dim);
    entry.bwd_dO = bhsd_input(*graph, "dO_bwd", k.batch_size, k.heads, k.q_seq,   k.head_dim);

    // O is read from the layer's ConcatenatedAttentionOutputs buffer, which is
    // physically laid out as {B, Q_seq, H, D} (post merge_heads). Logical shape
    // is {B, H, Q_seq, D} with non-contiguous strides:
    //   stride[B]=Q*H*D, stride[H]=D, stride[Q]=H*D, stride[D]=1
    entry.bwd_O = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("O_bwd")
                                .set_dim({k.batch_size, k.heads, k.q_seq, k.head_dim})
                                .set_stride({k.q_seq * k.heads * k.head_dim,
                                             k.head_dim,
                                             k.heads * k.head_dim,
                                             1}));

    entry.bwd_Stats = graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("Stats_bwd")
                                    .set_data_type(fe::DataType_t::FLOAT)
                                    .set_dim   ({k.batch_size, k.heads, k.q_seq, 1})
                                    .set_stride({k.heads * k.q_seq, k.q_seq, 1, 1}));

    const auto sdpa_bwd_options = fe::graph::SDPA_backward_attributes()
                            .set_name("flash_attn_bwd")
                            .set_causal_mask(k.causal)
                            .set_attn_scale(1.0f / std::sqrt(float(k.head_dim)));

    auto [dQ, dK, dV] = graph->sdpa_backward(entry.bwd_Q, entry.bwd_K, entry.bwd_V,
                                              entry.bwd_O, entry.bwd_dO, entry.bwd_Stats,
                                              sdpa_bwd_options);

    bhsd_output(dQ, k.batch_size, k.heads, k.q_seq,   k.head_dim);
    bhsd_output(dK, k.batch_size, k.heads, k.src_seq, k.head_dim);
    bhsd_output(dV, k.batch_size, k.heads, k.src_seq, k.head_dim);

    entry.bwd_dQ = dQ;
    entry.bwd_dK = dK;
    entry.bwd_dV = dV;

    finalize_sdpa_graph(*graph, handle, "bwd");

    int64_t ws = 0;
    graph->get_workspace_size(ws);
    if (ws > 0)
        CHECK_CUDA(cudaMalloc(&entry.bwd_workspace_buf, size_t(ws)));

    entry.bwd_graph = graph;
}

#endif  // OPENNN_HAS_CUDNN_FRONTEND

Attention::Attention() = default;
Attention::~Attention() { destroy_cuda(); }
Attention::Attention(Attention&&) noexcept = default;
Attention& Attention::operator=(Attention&&) noexcept = default;

void Attention::destroy_cuda()
{
    sdpa_cache.reset();
}

void Attention::apply(const TensorView& query,
                      const TensorView& key,
                      const TensorView& value,
                      const TensorView& source_input,
                      TensorView& attention_weights,
                      TensorView& attention_weights_dropped,
                      TensorView& output,
                      float* mask_scratch,
                      bool is_training)
{
    IF_GPU({
        apply_gpu(query, key, value, source_input,
                  attention_weights, attention_weights_dropped,
                  output, mask_scratch, is_training);
        return;
    });
    apply_cpu(query, key, value, source_input,
              attention_weights, attention_weights_dropped,
              output, mask_scratch, is_training);
}

void Attention::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept
{
    auto& fv = fp.views[layer];

    const auto& src_views = fv[input_slots[3]];
    const TensorView& source_input = src_views[min(source_view_index, src_views.size() - 1)];

    float* mask_scratch = fv[scratch_slots[0]][0].as<float>();
    TensorView attention_out(mask_scratch,
                             {fp.batch_size, heads_number, query_sequence_length, head_dimension},
                             compute_dtype);

    apply(fv[input_slots[0]][0], fv[input_slots[1]][0], fv[input_slots[2]][0], source_input,
          fv[output_slots[0]][0], fv[output_slots[1]][0],
          attention_out, mask_scratch, is_training);
}

void Attention::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& fv = fp.views[layer];
    auto& dv = bp.delta_views[layer];
    const Index batch_size = fp.batch_size;

    const TensorView& query             = fv[input_slots[0]][0];
    const TensorView& key               = fv[input_slots[1]][0];
    const TensorView& value             = fv[input_slots[2]][0];
    const TensorView& attention_output  = fv[attention_output_slots[0]][0];
    const TensorView& attention_weights = fv[output_slots[0]][0];
    const TensorView& attention_weights_dropped = fv[output_slots[1]][0];

    const TensorView output_gradient = fv[scratch_slots[0]][0]
        .reshape({batch_size, heads_number, query_sequence_length, head_dimension});

    TensorView& attention_weight_gradient = dv[output_delta_slots[0]][0];
    TensorView  query_gradient(dv[output_delta_slots[1]][0].as<float>(),
                               {batch_size, heads_number, query_sequence_length, head_dimension},
                               compute_dtype);
    TensorView  key_gradient(dv[output_delta_slots[2]][0].as<float>(),
                             {batch_size, heads_number, source_sequence_length, head_dimension},
                             compute_dtype);
    TensorView& value_gradient = dv[output_delta_slots[3]][0];

    apply_delta(query, key, value, attention_output,
                attention_weights, attention_weights_dropped,
                output_gradient,
                attention_weight_gradient,
                query_gradient, key_gradient, value_gradient);
}

void Attention::apply_cpu(const TensorView& query,
                          const TensorView& key,
                          const TensorView& value,
                          const TensorView& source_input,
                          TensorView& attention_weights,
                          TensorView& attention_weights_dropped,
                          TensorView& output,
                          float* mask_scratch,
                          bool is_training)
{
    multiply(query, false, key, true, attention_weights, scaling_factor(), 0.0f);

    {
        const Index batch_size = source_input.shape[0];
        const Index source_sequence_length = source_input.shape[1];
        const Index embedding_dimension = source_input.shape[2];
        const Index query_sequence_length = attention_weights.shape[2];

        if (!TRY_GPU_DISPATCH(attention_weights, [&](auto tag) {
            using T = decltype(tag);
            attention_masks_cuda<T>(to_int(batch_size),
                                    to_int(heads_number),
                                    to_int(query_sequence_length),
                                    to_int(source_sequence_length),
                                    to_int(embedding_dimension),
                                    source_input.as<T>(),
                                    attention_weights.as<T>(),
                                    reinterpret_cast<T*>(mask_scratch),
                                    use_causal_mask);
        }))
        {
            const Index att_rows_per_batch = heads_number * query_sequence_length;

            #pragma omp parallel for
            for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
            {
                const float* source_batch = source_input.as<float>() + batch_index * source_sequence_length * embedding_dimension;
                float*       attention_batch = attention_weights.as<float>() + batch_index * att_rows_per_batch * source_sequence_length;

                for (Index source_index = 0; source_index < source_sequence_length; ++source_index)
                {
                    const float* source_row = source_batch + source_index * embedding_dimension;
                    float max_abs = 0.0f;
                    for (Index k = 0; k < embedding_dimension; ++k)
                    {
                        const float abs_value = abs(source_row[k]);
                        if (abs_value > max_abs) max_abs = abs_value;
                    }
                    if (max_abs > EPSILON) continue;

                    for (Index row_index = 0; row_index < att_rows_per_batch; ++row_index)
                        attention_batch[row_index * source_sequence_length + source_index] = SOFTMAX_MASK_VALUE;
                }
            }

            if (use_causal_mask)
            {
                const Index batch_heads = batch_size * heads_number;
                MatrixMap attention_flat = attention_weights.as_flat_matrix();
                attention_flat += causal_mask.replicate(batch_heads, 1);
            }
        }
    }

    softmax(attention_weights);

    const bool apply_dropout = is_training && dropout.active();
    TensorView& used = apply_dropout ? attention_weights_dropped : attention_weights;

    if (apply_dropout)
    {
        copy(attention_weights, used);
        dropout.apply_cpu(used);
    }

    multiply(used, false, value, false, output);
}

void Attention::apply_gpu(const TensorView& query,
                          const TensorView& key,
                          const TensorView& value,
                          const TensorView& source_input,
                          TensorView& attention_weights,
                          TensorView& attention_weights_dropped,
                          TensorView& output,
                          float* mask_scratch,
                          bool is_training)
{
#ifdef OPENNN_HAS_CUDNN_FRONTEND
    // SDPA Flash Attention requires BF16 or FP16 in current cuDNN 9.x.
    // For FP32 we fall back to the manual GPU path (still correct, no fused kernel).
    // Padding-mask handling not yet wired through the graph (would need a per-batch
    // seq-len tensor); for now SDPA path assumes no padding tokens.
    // Dropout in SDPA needs device-resident seed/offset tensors (TODO); for now
    // we fall back to CPU when training-time dropout is active.
    const bool sdpa_supported =
            query.type == Type::BF16
         && !(dropout.active() && is_training);
    if (!sdpa_supported)
    {
        require_attention_scratch(attention_weights, "SDPA fallback triggered (FP32 or training-time dropout)");
        apply_cpu(query, key, value, source_input,
                  attention_weights, attention_weights_dropped,
                  output, mask_scratch, is_training);
        return;
    }

    if (!sdpa_cache) sdpa_cache = make_unique<SDPACache>();

    SDPACache::CacheKey ck{
        query.shape[0],          // batch_size
        query.shape[2],          // q_seq
        key.shape[2],            // src_seq
        heads_number,
        head_dimension,
        query.type,
        dropout.active() && is_training,
        use_causal_mask,
        is_training
    };

    auto& entry = sdpa_cache->get_or_create_entry(ck);
    if (!entry.fwd_graph)
        build_sdpa_forward_graph(entry, ck, Backend::get_cudnn_handle(), dropout.rate);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> tp;
    tp[entry.fwd_Q] = query.data;
    tp[entry.fwd_K] = key.data;
    tp[entry.fwd_V] = value.data;
    tp[entry.fwd_O] = output.data;
    if (is_training && entry.fwd_Stats) tp[entry.fwd_Stats] = entry.stats_buf;
    // Dropout in SDPA: deferred. When wired, requires device-resident seed/offset
    // tensors (see Seed/Offset graph nodes); current support gate above never
    // sets ck.dropout_active to true.

    const auto status = entry.fwd_graph->execute(Backend::get_cudnn_handle(), tp, entry.fwd_workspace_buf);
    if (status.is_bad())
        throw runtime_error("SDPA forward execute: " + status.get_message());
#else
    // No cudnn-frontend: fall back to the manual softmax+matmul GPU path.
    apply_cpu(query, key, value, source_input,
              attention_weights, attention_weights_dropped,
              output, mask_scratch, is_training);
#endif
}

void Attention::apply_delta(const TensorView& query,
                            const TensorView& key,
                            const TensorView& value,
                            const TensorView& attention_output,
                            const TensorView& attention_weights,
                            const TensorView& attention_weights_dropped,
                            const TensorView& output_gradient,
                            TensorView& attention_weight_gradient,
                            TensorView& query_gradient,
                            TensorView& key_gradient,
                            TensorView& value_gradient) const
{
    IF_GPU({
        apply_delta_gpu(query, key, value, attention_output,
                        attention_weights, attention_weights_dropped,
                        output_gradient, attention_weight_gradient,
                        query_gradient, key_gradient, value_gradient);
        return;
    });
    apply_delta_cpu(query, key, value, attention_output,
                    attention_weights, attention_weights_dropped,
                    output_gradient, attention_weight_gradient,
                    query_gradient, key_gradient, value_gradient);
}

void Attention::apply_delta_cpu(const TensorView& query,
                                const TensorView& key,
                                const TensorView& value,
                                const TensorView& /*attention_output*/,
                                const TensorView& attention_weights,
                                const TensorView& attention_weights_dropped,
                                const TensorView& output_gradient,
                                TensorView& attention_weight_gradient,
                                TensorView& query_gradient,
                                TensorView& key_gradient,
                                TensorView& value_gradient) const
{
    const TensorView& attention_used = dropout.active()
        ? attention_weights_dropped
        : attention_weights;

    multiply(attention_used, true, output_gradient, false, value_gradient);
    multiply(output_gradient, false, value, true, attention_weight_gradient);

    if (dropout.active())
        dropout.apply_delta(attention_weight_gradient);

    if (!attention_weight_gradient.empty())
    {
        const MatrixMap y  = attention_weights.as_flat_matrix();
        MatrixMap       dY = attention_weight_gradient.as_flat_matrix();

        const VectorR dot = (y.array() * dY.array()).rowwise().sum();
        dY.array() = y.array() * (dY.colwise() - dot).array();
    }

    const float scale = scaling_factor();
    multiply(attention_weight_gradient, false, key,   false, query_gradient, scale, 0.0f);
    multiply(attention_weight_gradient, true,  query, false, key_gradient,   scale, 0.0f);
}

#ifdef OPENNN_HAS_CUDA

void Attention::apply_delta_gpu_unfused(const TensorView& query,
                                        const TensorView& key,
                                        const TensorView& value,
                                        const TensorView& attention_weights,
                                        const TensorView& attention_weights_dropped,
                                        const TensorView& output_gradient,
                                        TensorView& attention_weight_gradient,
                                        TensorView& query_gradient,
                                        TensorView& key_gradient,
                                        TensorView& value_gradient) const
{
    const TensorView& attention_used = dropout.active()
        ? attention_weights_dropped
        : attention_weights;

    multiply(attention_used, true, output_gradient, false, value_gradient);
    multiply(output_gradient, false, value, true, attention_weight_gradient);

    if (dropout.active())
        dropout.apply_delta(attention_weight_gradient);

    if (!attention_weight_gradient.empty())
    {
        CHECK_CUDNN(cudnnSoftmaxBackward(Backend::get_cudnn_handle(),
                                         CUDNN_SOFTMAX_ACCURATE,
                                         CUDNN_SOFTMAX_MODE_CHANNEL,
                                         &one,
                                         attention_weights.get_descriptor(),     attention_weights.data,
                                         attention_weight_gradient.get_descriptor(), attention_weight_gradient.data,
                                         &zero,
                                         attention_weight_gradient.get_descriptor(), attention_weight_gradient.data));
    }

    const float scale = scaling_factor();
    multiply(attention_weight_gradient, false, key,   false, query_gradient, scale, 0.0f);
    multiply(attention_weight_gradient, true,  query, false, key_gradient,   scale, 0.0f);
}

#endif

void Attention::apply_delta_gpu(const TensorView& query,
                                const TensorView& key,
                                const TensorView& value,
                                const TensorView& attention_output,
                                const TensorView& attention_weights,
                                const TensorView& attention_weights_dropped,
                                const TensorView& output_gradient,
                                TensorView& attention_weight_gradient,
                                TensorView& query_gradient,
                                TensorView& key_gradient,
                                TensorView& value_gradient) const
{
#ifdef OPENNN_HAS_CUDNN_FRONTEND
    // Same support gates as forward. The forward path that produced this
    // backward call established the cache entry; if we fell back to CPU on
    // forward, we must also fall back here (no LSE stats to use).
    const bool sdpa_supported =
            query.type == Type::BF16
         && !dropout.active();
    if (!sdpa_supported || !sdpa_cache)
    {
        require_attention_scratch(attention_weights, "SDPA backward fallback triggered");
        apply_delta_gpu_unfused(query, key, value,
                                attention_weights, attention_weights_dropped,
                                output_gradient, attention_weight_gradient,
                                query_gradient, key_gradient, value_gradient);
        return;
    }

    SDPACache::CacheKey ck{
        query.shape[0],
        query.shape[2],
        key.shape[2],
        heads_number,
        head_dimension,
        query.type,
        false,                  // dropout falls back, never reaches here
        use_causal_mask,
        true                    // backward implies training
    };

    SDPACache::Entry* entry_ptr = sdpa_cache->find_entry(ck);
    if (!entry_ptr || !entry_ptr->fwd_graph)
    {
        // Forward never built an SDPA entry for this shape (e.g. fell back).
        require_attention_scratch(attention_weights, "SDPA backward without matching forward entry");
        apply_delta_gpu_unfused(query, key, value,
                                attention_weights, attention_weights_dropped,
                                output_gradient, attention_weight_gradient,
                                query_gradient, key_gradient, value_gradient);
        return;
    }

    auto& entry = *entry_ptr;
    if (!entry.bwd_graph)
        build_sdpa_backward_graph(entry, ck, Backend::get_cudnn_handle());

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> tp;
    tp[entry.bwd_Q]     = const_cast<float*>(query.as<float>());
    tp[entry.bwd_K]     = const_cast<float*>(key.as<float>());
    tp[entry.bwd_V]     = const_cast<float*>(value.as<float>());
    tp[entry.bwd_O]     = const_cast<float*>(attention_output.as<float>());
    tp[entry.bwd_dO]    = const_cast<float*>(output_gradient.as<float>());
    tp[entry.bwd_Stats] = entry.stats_buf;
    tp[entry.bwd_dQ]    = query_gradient.data;
    tp[entry.bwd_dK]    = key_gradient.data;
    tp[entry.bwd_dV]    = value_gradient.data;

    const auto status = entry.bwd_graph->execute(Backend::get_cudnn_handle(), tp, entry.bwd_workspace_buf);
    if (status.is_bad())
        throw runtime_error("SDPA backward execute: " + status.get_message());
#elif defined(OPENNN_HAS_CUDA)
    apply_delta_gpu_unfused(query, key, value,
                            attention_weights, attention_weights_dropped,
                            output_gradient, attention_weight_gradient,
                            query_gradient, key_gradient, value_gradient);
#else
    apply_delta_cpu(query, key, value, attention_output,
                    attention_weights, attention_weights_dropped,
                    output_gradient, attention_weight_gradient,
                    query_gradient, key_gradient, value_gradient);
#endif
}

void Attention::to_JSON(JsonWriter&) const
{
    // Per-call config (heads/seqs/causal_mask/dtype) lives at the layer level.
    // Dropout rate is a runtime knob, not serialized — preserves existing schema.
}

void Attention::from_JSON(const Json*)
{
    // No-op for the same reason.
}


void Merge::set(Index new_heads_number, Index new_query_sequence_length, Index new_head_dimension, Type new_compute_dtype)
{
    heads_number          = new_heads_number;
    query_sequence_length = new_query_sequence_length;
    head_dimension        = new_head_dimension;
    compute_dtype         = new_compute_dtype;
}

void Merge::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    auto& fv = fp.views[layer];
    const Index batch_size = fp.batch_size;

    const TensorView source_4d(fv[input_slots[0]][0].as<float>(),
                               {batch_size, heads_number, query_sequence_length, head_dimension},
                               compute_dtype);
    TensorView dest_4d = fv[output_slots[0]][0].reshape({batch_size, query_sequence_length, heads_number, head_dimension});

    merge_heads(source_4d, dest_4d);
}

void Merge::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& fv = fp.views[layer];
    auto& dv = bp.delta_views[layer];
    const Index batch_size = fp.batch_size;

    const TensorView concat_gradient_4d = dv[output_delta_slots[0]][0]
        .reshape({batch_size, query_sequence_length, heads_number, head_dimension});
    TensorView heads_gradient_4d(fv[input_slots[0]][0].as<float>(),
                                 {batch_size, heads_number, query_sequence_length, head_dimension},
                                 compute_dtype);

    split_heads(concat_gradient_4d, heads_gradient_4d);
}


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

#ifdef OPENNN_HAS_CUDA
    if (pool_height <= 0 || pool_width <= 0) return;

    if (!pooling_descriptor) CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));

    const cudnnPoolingMode_t mode = (method == 0)
        ? CUDNN_POOLING_MAX
        : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

    CHECK_CUDNN(cudnnSetPooling2dDescriptor(pooling_descriptor,
                                            mode,
                                            CUDNN_PROPAGATE_NAN,
                                            to_int(pool_height), to_int(pool_width),
                                            to_int(padding_height), to_int(padding_width),
                                            to_int(row_stride), to_int(column_stride)));
#endif
}

void Pool::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept
{
    auto& fv = fp.views[layer];
    const TensorView& input = fv[input_slots[0]][0];
    TensorView& output      = fv[output_slots[0]][0];

    TensorView empty_indices;
    TensorView& indices = (output_slots.size() > 1)
        ? fv[output_slots[1]][0]
        : empty_indices;

    IF_GPU({ apply_gpu(input, output); return; });
    apply_cpu(input, output, indices, is_training);
}

void Pool::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& fv = fp.views[layer];
    auto& dv = bp.delta_views[layer];

    const TensorView& input        = fv[input_slots[0]][0];
    const TensorView& output       = fv[output_slots[0]][0];
    const TensorView& output_delta = dv[output_delta_slots[0]][0];
    TensorView& input_delta        = dv[input_delta_slots[0]][0];

    TensorView empty_indices;
    const TensorView& indices = (output_slots.size() > 1)
        ? fv[output_slots[1]][0]
        : empty_indices;

    IF_GPU({ apply_delta_gpu(input, output, output_delta, input_delta); return; });
    apply_delta_cpu(output_delta, indices, input_delta);
}

void Pool3d::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept
{
    auto& fv = fp.views[layer];
    const TensorView& input = fv[input_slots[0]][0];
    TensorView& output      = fv[output_slots[0]][0];
    TensorView& indices     = fv[output_slots[1]][0];

    if (method == 0)
        max_pooling_3d_forward(input, output, indices, is_training);
    else
        average_pooling_3d_forward(input, output);
}

void Pool3d::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& fv = fp.views[layer];
    auto& dv = bp.delta_views[layer];

    const TensorView& output_delta = dv[output_delta_slots[0]][0];
    TensorView& input_delta        = dv[input_delta_slots[0]][0];

    if (method == 0)
        max_pooling_3d_backward(fv[output_slots[1]][0], output_delta, input_delta);
    else
        average_pooling_3d_backward(fv[input_slots[0]][0], output_delta, input_delta);
}

void Pool::apply_cpu(const TensorView& input, TensorView& output, TensorView& maximal_indices, bool is_training)
{
    const TensorMap4 inputs = input.as_tensor<4>();
    TensorMap4 outputs      = output.as_tensor<4>();

    const Index batch_size    = inputs.dimension(0);
    const Index output_height = outputs.dimension(1);
    const Index output_width  = outputs.dimension(2);

    if (method == 0 && is_training)  // Max with argmax (training)
    {
        TensorMap4 maximal_indices_map = maximal_indices.as_tensor<4>();

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
                                    maximal_index = pool_row * pool_width + pool_column;
                                }
                            }

                        outputs(batch_index, output_row, output_column, channel_index) = maximum_value;
                        maximal_indices_map(batch_index, output_row, output_column, channel_index) = maximal_index;
                    }
                }
    }
    else if (method == 0)  // Max (inference, no argmax)
    {
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

                        float maximum_value = NEG_INFINITY;

                        for (Index pool_row = pool_row_start; pool_row < pool_row_end; ++pool_row)
                            for (Index pool_column = pool_col_start; pool_column < pool_col_end; ++pool_column)
                            {
                                const float value = inputs(batch_index,
                                                           input_row_start + pool_row,
                                                           input_column_start + pool_column,
                                                           channel_index);
                                if (value > maximum_value) maximum_value = value;
                            }

                        outputs(batch_index, output_row, output_column, channel_index) = maximum_value;
                    }
                }
    }
    else  // Average
    {
        const float inv_pool_size = 1.0f / (pool_height * pool_width);

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
    TensorMap4 in_gradients        = input_delta.as_tensor<4>().setZero();

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
                        in_gradients(batch_index,
                                 input_row_start + pool_row,
                                 input_column_start + pool_column,
                                 channel_index)
                            += out_grads(batch_index, output_row, output_column, channel_index);
                    }
                }
        return;
    }

    // Average
    const float inv_pool_size = 1.0f / (pool_height * pool_width);

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
                            in_gradients(batch_index,
                                     input_row_start + pool_row,
                                     input_column_start + pool_column,
                                     channel_index) += average_gradient;
                }
            }
}

#ifdef OPENNN_HAS_CUDA

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

void EmbeddingLookup::link_gradients(const vector<TensorView>& views)
{
    if (views.empty()) return;
    weight_gradient = views[0];
}

void EmbeddingLookup::link_states(const vector<TensorView>& views)
{
    if (views.empty()) return;
    const bool needs_init = positional_encoding.data == nullptr;
    positional_encoding = views[0];
    if (needs_init) init_positional_encoding();
}

void EmbeddingLookup::set_parameters_random()
{
    if (weights.empty()) return;
    MatrixMap weights_matrix = weights.as_matrix();
    set_random_normal(weights_matrix, 0.0f, 1.0f);
    weights_matrix.row(0).setZero();
}

void EmbeddingLookup::set_parameters_glorot()
{
    if (weights.empty()) return;
    const float limit = sqrt(6.0f / (vocabulary_size + embedding_dimension));
    MatrixMap weights_matrix = weights.as_matrix();
    weights_matrix.setRandom();
    weights_matrix *= limit;
    weights_matrix.row(0).setZero();
}

void EmbeddingLookup::init_positional_encoding()
{
    if (!add_positional_encoding) return;
    if (positional_encoding.empty() || !positional_encoding.data) return;

    float* table = positional_encoding.as<float>();
    const float half_depth = float(embedding_dimension) / 2;

    VectorR divisors(embedding_dimension);
    for (Index j = 0; j < embedding_dimension; ++j)
        divisors(j) = pow(10000.0f,
                          (j < Index(half_depth) ? j : j - Index(half_depth)) / half_depth);

    #pragma omp parallel for collapse(2)
    for (Index i = 0; i < sequence_length; ++i)
        for (Index j = 0; j < embedding_dimension; ++j)
            table[i * embedding_dimension + j] = (j < Index(half_depth))
                ? sin(i / divisors(j))
                : cos(i / divisors(j));
}

void EmbeddingLookup::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    auto& fv = fp.views[layer];
    const TensorView& indices = fv[input_slots[0]][0];
    TensorView& output        = fv[output_slots[0]][0];

    IF_GPU({ apply_gpu(indices, output); return; });
    apply_cpu(indices, output);
}

void EmbeddingLookup::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& fv = fp.views[layer];
    auto& dv = bp.delta_views[layer];

    const TensorView& indices      = fv[input_slots[0]][0];
    const TensorView& output_delta = dv[output_delta_slots[0]][0];

    IF_GPU({ apply_delta_gpu(indices, output_delta); return; });
    apply_delta_cpu(indices, output_delta);
}

void EmbeddingLookup::apply_cpu(const TensorView& indices, TensorView& output)
{
    const Index total_tokens = indices.size();

    MatrixMap output_mat              = output.as_flat_matrix();
    const MatrixMap weights_mat       = weights.as_matrix();
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
            const MatrixMap pe = positional_encoding.as_matrix();
            output_mat.row(i) += pe.row(i % sequence_length);
        }
    }
}

void EmbeddingLookup::apply_delta_cpu(const TensorView& indices,
                                      const TensorView& output_delta) const
{
    const Index total_elements = indices.size();

    MatrixMap output_delta_map = output_delta.as_flat_matrix();

    if (scale_embedding)
        output_delta_map *= sqrt(to_type(embedding_dimension));

    MatrixMap weight_gradients = weight_gradient.as_matrix().setZero();

    for (Index token_index = 0; token_index < total_elements; ++token_index)
    {
        const Index vocabulary_index = static_cast<Index>(indices.as<float>()[token_index]);

        if (vocabulary_index < 0 || vocabulary_index >= weight_gradients.rows())
            continue;

        weight_gradients.row(vocabulary_index).noalias() += output_delta_map.row(token_index);
    }
}

#ifdef OPENNN_HAS_CUDA

void EmbeddingLookup::apply_gpu(const TensorView& indices, TensorView& output)
{
    output.dispatch([&](auto tag) {
        using T = decltype(tag);
        embedding_forward_cuda<T>(
            output.size(),
            indices.as<float>(),
            weights.as<float>(),
            add_positional_encoding ? positional_encoding.as<float>() : nullptr,
            output.as<T>(),
            sequence_length, embedding_dimension, vocabulary_size,
            scale_embedding);
    });
}

void EmbeddingLookup::apply_delta_gpu(const TensorView& indices,
                                      const TensorView& output_delta) const
{
    weight_gradient.set_zero_async();

    output_delta.dispatch([&](auto tag) {
        using T = decltype(tag);
        embedding_backward_cuda<T>(
            output_delta.size(),
            indices.as<float>(),
            output_delta.as<T>(),
            weight_gradient.as<float>(),
            embedding_dimension, vocabulary_size, scale_embedding);
    });
}

#else

void EmbeddingLookup::apply_gpu(const TensorView&, TensorView&)                                                { throw runtime_error("EmbeddingLookup::apply_gpu: CUDA support not compiled in."); }
void EmbeddingLookup::apply_delta_gpu(const TensorView&, const TensorView&) const                              { throw runtime_error("EmbeddingLookup::apply_delta_gpu: CUDA support not compiled in."); }

#endif

void Flat::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    auto& fv = fp.views[layer];
    copy(fv[input_slots[0]][0], fv[output_slots[0]][0]);
}

void Flat::back_propagate(ForwardPropagation&, BackPropagation& bp, size_t layer) const noexcept
{
    auto& dv = bp.delta_views[layer];
    copy(dv[output_delta_slots[0]][0], dv[input_delta_slots[0]][0]);
}

vector<pair<Shape, Type>> Bound::state_specs() const
{
    if (method == Method::NoBounding || features == 0) return {};
    return vector<pair<Shape, Type>>(2, {Shape{features}, Type::FP32});
}

void Bound::link_states(const vector<TensorView>& views)
{
    if (views.size() < 2) return;

    const bool needs_defaults = (lower.data == nullptr);

    lower = views[0];
    upper = views[1];

    if (!needs_defaults) return;

    if (lower.data) lower.as_vector().setConstant(-numeric_limits<float>::max());
    if (upper.data) upper.as_vector().setConstant( numeric_limits<float>::max());
}

void Bound::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    auto& fv = fp.views[layer];
    const TensorView& input = fv[input_slots[0]][0];
    TensorView& output      = fv[output_slots[0]][0];

    if (method == Method::NoBounding)
    {
        copy(input, output);
        return;
    }

    bound(input, lower, upper, output);
}

void Bound::load_state_from_JSON(const Json* parent)
{
    if (!parent || method == Method::NoBounding || !lower.data) return;

    VectorR tmp;
    if (parent->has("LowerBounds"))
    {
        string_to_vector(read_json_string(parent, "LowerBounds"), tmp);
        if (tmp.size() == lower.size()) lower.as_vector() = tmp;
    }
    
    if (parent->has("UpperBounds"))
    {
        string_to_vector(read_json_string(parent, "UpperBounds"), tmp);
        if (tmp.size() == upper.size()) upper.as_vector() = tmp;
    }
}

vector<pair<Shape, Type>> Scale::state_specs() const
{
    if (features == 0) return {};
    return vector<pair<Shape, Type>>(5, {Shape{features}, Type::FP32});
}

void Scale::link_states(const vector<TensorView>& views)
{
    if (views.size() < 5) return;

    const bool needs_defaults = (means.data == nullptr);

    minimums            = views[0];
    maximums            = views[1];
    means               = views[2];
    standard_deviations = views[3];
    scalers             = views[4];

    if (!needs_defaults) return;

    if (means.data)               means.as_vector().setZero();
    if (standard_deviations.data) standard_deviations.as_vector().setOnes();
    if (minimums.data)            minimums.as_vector().setConstant(-1.0f);
    if (maximums.data)            maximums.as_vector().setOnes();
}

void Scale::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    auto& fv = fp.views[layer];
    const TensorView& input = fv[input_slots[0]][0];
    TensorView& output      = fv[output_slots[0]][0];

    if (!minimums.data)
    {
        copy(input, output);
        return;
    }

    scale(input, minimums, maximums, means, standard_deviations, scalers,
          min_range, max_range, output);
}

void Scale::load_state_from_JSON(const Json* parent)
{
    if (!parent || !means.data) return;

    VectorR tmp;
    if (parent->has("Means"))
    {
        string_to_vector(read_json_string(parent, "Means"), tmp);
        if (tmp.size() == means.size()) means.as_vector() = tmp;
    }
    if (parent->has("StandardDeviations"))
    {
        string_to_vector(read_json_string(parent, "StandardDeviations"), tmp);
        if (tmp.size() == standard_deviations.size()) standard_deviations.as_vector() = tmp;
    }
    if (parent->has("Minimums"))
    {
        string_to_vector(read_json_string(parent, "Minimums"), tmp);
        if (tmp.size() == minimums.size()) minimums.as_vector() = tmp;
    }
    if (parent->has("Maximums"))
    {
        string_to_vector(read_json_string(parent, "Maximums"), tmp);
        if (tmp.size() == maximums.size()) maximums.as_vector() = tmp;
    }
}

vector<pair<Shape, Type>> Unscale::state_specs() const
{
    if (features == 0) return {};
    return vector<pair<Shape, Type>>(5, {Shape{features}, Type::FP32});
}

void Unscale::link_states(const vector<TensorView>& views)
{
    if (views.size() < 5) return;

    const bool needs_defaults = (means.data == nullptr);

    minimums            = views[0];
    maximums            = views[1];
    means               = views[2];
    standard_deviations = views[3];
    scalers             = views[4];

    if (!needs_defaults) return;

    if (means.data)               means.as_vector().setZero();
    if (standard_deviations.data) standard_deviations.as_vector().setOnes();
    if (minimums.data)            minimums.as_vector().setConstant(-1.0f);
    if (maximums.data)            maximums.as_vector().setOnes();
}

void Unscale::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    auto& fv = fp.views[layer];
    const TensorView& input = fv[input_slots[0]][0];
    TensorView& output      = fv[output_slots[0]][0];

    if (!minimums.data)
    {
        copy(input, output);
        return;
    }

    unscale(input, minimums, maximums, means, standard_deviations, scalers,
            min_range, max_range, output);
}

void Unscale::load_state_from_JSON(const Json* parent)
{
    if (!parent || !means.data) return;

    const Json* neurons_array = parent->find("Neurons");
    if (!neurons_array || !neurons_array->is_array()) return;

    for (size_t i = 0; i < neurons_array->array_value.size() && Index(i) < minimums.size(); ++i)
    {
        const Json* neuron = &neurons_array->array_value[i];
        const string descriptives = read_json_string(neuron, "Descriptives");
        const vector<string> tokens = get_tokens(descriptives, " ");
        if (tokens.size() >= 4)
        {
            minimums.as<float>()[i]            = float(stof(tokens[0]));
            maximums.as<float>()[i]            = float(stof(tokens[1]));
            means.as<float>()[i]               = float(stof(tokens[2]));
            standard_deviations.as<float>()[i] = float(stof(tokens[3]));
        }
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
