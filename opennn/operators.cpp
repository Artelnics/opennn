//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E R A T O R S   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifdef OPENNN_HAS_CUDA
#include <cuda_fp16.h>
#ifdef HAVE_CUDNN_FRONTEND
#include <cudnn_frontend.h>
#endif
#include "cuda_gemm.h"
#endif

#include "operators.h"
#include "json.h"
#include "random_utilities.h"
#include "math_utilities.h"
#include "cuda_dispatch.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

void AddOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool) noexcept
{
    const vector<TensorView>& inputs = get_inputs(fp, layer);
    TensorView& output               = get_output(fp, layer);

    check(inputs, output);

    add(inputs[0], inputs[1], output);
    for (size_t i = 2; i < inputs.size(); ++i)
        add(output, inputs[i], output);
}

void AddOp::back_propagate(ForwardPropagation&, BackPropagation& bp, size_t layer) const noexcept
{
    const TensorView& output_delta = get_output_delta(bp, layer);

    for (size_t s : input_delta_slots)
        copy(output_delta, bp.delta_views[layer][s]);
}

void AddOp::check(const vector<TensorView>& inputs, const TensorView& output) const
{
    if (inputs.size() < 2)
        throw runtime_error("Add: needs at least 2 inputs.");

    for (const TensorView& input : inputs)
        if (input.size() != output.size())
            throw runtime_error("Add: tensor dimensions do not match.");
}

void DropoutOp::set_rate(float new_rate)
{
    if (new_rate < 0.0f || new_rate >= 1.0f)
        throw runtime_error("Dropout rate must be in [0, 1).");

    rate = new_rate;
}

void DropoutOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept
{
    if (!is_training || !active()) return;

    auto& fv = fp.views[layer];
    TensorView& output = get_output(fp, layer);

    if (!save_slots.empty())
        copy(output, fv[save_slots[0]][0]);

    dropout_forward(output, mask, rate);
}

void DropoutOp::back_propagate(ForwardPropagation&, BackPropagation& bp, size_t layer) const noexcept
{
    if (!active()) return;
    dropout_backward(get_output_delta(bp, layer), mask, rate);
}

void DropoutOp::destroy_cuda()
{
#ifdef OPENNN_HAS_CUDA
    if (mask.device_type == Device::CUDA)
        mask.resize_bytes(0, Device::CUDA);
#endif
}

void DropoutOp::to_JSON(JsonWriter& w) const
{
    if (rate > 0.0f)
        add_json_field(w, "DropoutRate", to_string(rate));
}

void DropoutOp::from_JSON(const Json* parent)
{
    if (parent && parent->has("DropoutRate"))
        set_rate(float(read_json_type(parent, "DropoutRate")));
}

cudnnActivationMode_t ActivationOp::to_cudnn_mode(Function function)
{
    using enum Function;
    switch (function)
    {
    case Sigmoid: return CUDNN_ACTIVATION_SIGMOID;
    case Tanh:    return CUDNN_ACTIVATION_TANH;
    case ReLU:    return CUDNN_ACTIVATION_RELU;
    case Identity:
    case Softmax: return CUDNN_ACTIVATION_IDENTITY;
    }

    return CUDNN_ACTIVATION_IDENTITY;
}

void ActivationOp::set_function(Function new_function)
{
    function = new_function;
#ifdef OPENNN_HAS_CUDA
    if (!descriptor) cudnnCreateActivationDescriptor(&descriptor);
    cudnnSetActivationDescriptor(descriptor, to_cudnn_mode(function), CUDNN_PROPAGATE_NAN, 0.0);
#endif
}

void ActivationOp::set_function(const string& name)
{
    set_function(from_string(name));
}

void ActivationOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    TensorView& output = fp.views[layer][output_slots[0]][0];
    if (output.empty()) return;

    // Standalone Activation layer: input and output live in different slots.
    // Seed `output` with the upstream value before applying the elementwise
    // function in place. When input_slots is empty (the fused path inside
    // Dense / Convolutional / etc.), `output` already holds the pre-activation
    // value written by the previous operator in the chain.
    if (!input_slots.empty() && input_slots[0] != output_slots[0])
        copy(fp.views[layer][input_slots[0]][0], output);

    activation_forward(output, function);
}

void ActivationOp::apply_delta(const TensorView& outputs, TensorView& delta) const
{
    activation_backward(outputs, delta, function);
}

void ActivationOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    const auto& slots = output_slots_backward.empty() ? output_slots : output_slots_backward;
    const TensorView& outputs = fp.views[layer][slots[0]][0];

    // Standalone Activation layer: write input_delta = output_delta * sigma'(output)
    // into a distinct InputDelta slot. In the fused path (Dense/Conv/...),
    // input_slots is empty (in-place) and we modify the layer's OutputDelta
    // directly — the next operator in the chain reads it as its own input.
    const bool standalone = !input_slots.empty() && input_slots[0] != output_slots[0];
    if (standalone)
    {
        const TensorView& output_delta = get_output_delta(bp, layer);
        TensorView& input_delta        = get_input_delta(bp, layer);
        copy(output_delta, input_delta);
        apply_delta(outputs, input_delta);
    }
    else
    {
        TensorView& delta = get_output_delta(bp, layer);
        apply_delta(outputs, delta);
    }
}

void ActivationOp::destroy_cuda()
{
#ifdef OPENNN_HAS_CUDA
    if (descriptor) { cudnnDestroyActivationDescriptor(descriptor); descriptor = nullptr; }
#endif
}

void ActivationOp::to_JSON(JsonWriter& w) const
{
    add_json_field(w, "Activation", ActivationOp::to_string(function));
}

void ActivationOp::from_JSON(const Json* parent)
{
    if (parent && parent->has("Activation"))
        set_function(read_json_string(parent, "Activation"));
}

void BatchNormOp::set(Index new_features, float new_momentum)
{
    if (new_momentum < 0.0f || new_momentum >= 1.0f)
        throw runtime_error("BatchNorm momentum must be in [0, 1).");
    features = new_features;
    momentum = new_momentum;
}

vector<TensorSpec> BatchNormOp::parameter_specs() const
{
    if (!active()) return {};
    return vector<TensorSpec>(2, {Shape{features}, Type::FP32});
}

vector<TensorSpec> BatchNormOp::state_specs() const
{
    return parameter_specs();
}

void BatchNormOp::link_parameters(span<const TensorView> views)
{
    if (views.size() < 2) return;
    gamma = views[0];
    beta  = views[1];
    invalidate_inference_cache();
}

void BatchNormOp::link_gradients(span<const TensorView> views)
{
    if (views.size() < 2) return;
    gamma_gradient = views[0];
    beta_gradient  = views[1];
}

void BatchNormOp::link_states(span<const TensorView> views)
{
    if (views.size() < 2) return;
    running_mean     = views[0];
    running_variance = views[1];
    invalidate_inference_cache();
}

void BatchNormOp::init_defaults()
{
    if (gamma.data)            gamma.as_vector().setOnes();
    if (beta.data)             beta.as_vector().setZero();
    if (running_mean.data)     running_mean.as_vector().setZero();
    if (running_variance.data) running_variance.as_vector().setOnes();
    invalidate_inference_cache();
}

void BatchNormOp::to_JSON(JsonWriter& w) const
{
    if (!active()) return;

    add_json_field(w, "Momentum", to_string(momentum));

    if (running_mean.data)
        add_json_field(w, "RunningMeans", vector_to_string(running_mean.as_vector()));
    if (running_variance.data)
        add_json_field(w, "RunningVariances", vector_to_string(running_variance.as_vector()));
}

void BatchNormOp::from_JSON(const Json* parent)
{
    if (parent && parent->has("Momentum"))
        momentum = float(read_json_type(parent, "Momentum"));
}

void BatchNormOp::load_state_from_JSON(const Json* parent)
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

void BatchNormOp::update_inference_cache()
{
    if (!inference_cache_dirty) return;
    if (!gamma.data || !beta.data || !running_mean.data || !running_variance.data) return;

    inference_scale = gamma.as_vector().array()
                    / (running_variance.as_vector().array() + EPSILON).sqrt();
    inference_shift = beta.as_vector().array()
                    - inference_scale.array() * running_mean.as_vector().array();

    inference_cache_dirty = false;
}

void BatchNormOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept
{
    if (!active()) return;

    const TensorView& input = get_input(fp, layer);
    TensorView& output      = get_output(fp, layer);

    if (is_training)
    {
        TensorView& mean         = get_output(fp, layer, 1);
        TensorView& inv_variance = get_output(fp, layer, 2);

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

void BatchNormOp::apply_delta(const TensorView& input,
                            const TensorView& mean,
                            const TensorView& inverse_variance,
                            TensorView& delta) const
{
    IF_GPU({ apply_delta_gpu(input, mean, inverse_variance, delta); return; });
    apply_delta_cpu(input, mean, inverse_variance, delta);
}

void BatchNormOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    if (!active()) return;

    const TensorView& input            = get_input(fp, layer);
    const TensorView& mean             = get_output(fp, layer, 1);
    const TensorView& inverse_variance = get_output(fp, layer, 2);
    TensorView& delta                  = get_output_delta(bp, layer);

    apply_delta(input, mean, inverse_variance, delta);
}

void BatchNormOp::apply_inference_cpu(const TensorView& input, TensorView& output)
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

void BatchNormOp::apply_training_cpu(const TensorView& input,
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

    running_means     = running_means     * (1.0f - momentum) + means             * momentum;
    running_variances = running_variances * (1.0f - momentum) + inverse_variances * momentum;

    inverse_variances.array() = 1.0f / (inverse_variances.array() + EPSILON).sqrt();
    const VectorR scale = inverse_variances.array() * gamma.as_vector().array();
    const VectorMap betas = beta.as_vector();

    const auto scale_t = scale.transpose().array();
    const auto betas_t = betas.transpose().array();

    #pragma omp parallel for
    for (Index i = 0; i < output_matrix.rows(); ++i)
        output_matrix.row(i).array() = output_matrix.row(i).array() * scale_t + betas_t;
}

void BatchNormOp::apply_delta_cpu(const TensorView& input,
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

void BatchNormOp::apply_inference_gpu(const TensorView& input, TensorView& output)
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

void BatchNormOp::apply_training_gpu(const TensorView& input,
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
        static_cast<double>(momentum),
        running_mean.data, running_variance.data,
        EPSILON,
        mean.data, inverse_variance.data));
}

void BatchNormOp::apply_delta_gpu(const TensorView& input,
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

void BatchNormOp::apply_inference_gpu(const TensorView&, TensorView&)                                    { throw runtime_error("BatchNorm::apply_inference_gpu: CUDA support not compiled in."); }
void BatchNormOp::apply_training_gpu (const TensorView&, TensorView&, TensorView&, TensorView&)          { throw runtime_error("BatchNorm::apply_training_gpu: CUDA support not compiled in."); }
void BatchNormOp::apply_delta_gpu    (const TensorView&, const TensorView&, const TensorView&,
                                    TensorView&) const                                                  { throw runtime_error("BatchNorm::apply_delta_gpu: CUDA support not compiled in."); }

#endif

void CombinationOp::set(Index new_input_features, Index new_output_features, Type new_weight_type)
{
    input_features  = new_input_features;
    output_features = new_output_features;
    weight_type     = new_weight_type;
}

vector<TensorSpec> CombinationOp::parameter_specs() const
{
    return {
        {{output_features},                  weight_type},
        {{input_features, output_features},  weight_type},
    };
}

void CombinationOp::link_parameters(span<const TensorView> views)
{
    if (views.size() < 2) return;
    bias    = views[0];
    weights = views[1];
}

void CombinationOp::link_gradients(span<const TensorView> views)
{
    if (views.size() < 2) return;
    bias_gradient   = views[0];
    weight_gradient = views[1];
}

void CombinationOp::set_parameters_random()
{
    if (weights.empty()) return;
    set_random_uniform(weights.as_vector());
    if (!bias.empty()) bias.setZero();
}

void CombinationOp::set_parameters_glorot()
{
    if (weights.empty()) return;
    const float limit = glorot_limit(input_features, output_features);
    set_random_uniform(weights.as_vector(), -limit, limit);
    if (!bias.empty()) bias.setZero();
}

void CombinationOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool) noexcept
{
    apply(get_input(fp, layer), get_output(fp, layer), CUBLASLT_EPILOGUE_BIAS);
}

void CombinationOp::apply(const TensorView& input, TensorView& output, cublasLtEpilogue_t epilogue)
{
    linear_forward(input, weights, bias, output, epilogue);
}

void CombinationOp::apply_delta(const TensorView& output_delta,
                              const TensorView& input,
                              TensorView& input_delta,
                              bool accumulate_input_delta) const
{
    linear_backward(output_delta, input, weights, weight_gradient, bias_gradient,
                    input_delta, accumulate_input_delta);
}

void CombinationOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& dv = bp.delta_views[layer];

    const TensorView& input        = get_input(fp, layer);
    const TensorView& output_delta = get_output_delta(bp, layer);

    TensorView empty;
    TensorView& input_delta = view_at_slot_or(dv, input_delta_slots, 0, empty);

    apply_delta(output_delta, input, input_delta, false);
}

void CombinationReluOp::set(Index input_features, Index output_features, Type weight_type)
{
    combination.set(input_features, output_features, weight_type);
    activation.set_function(ActivationOp::Function::ReLU);
}

void CombinationReluOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    const TensorView& input = get_input(fp, layer);
    TensorView& output      = get_output(fp, layer);

    combination.apply(input, output, CUBLASLT_EPILOGUE_RELU_BIAS);
}

void CombinationReluOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& dv = bp.delta_views[layer];

    const TensorView& output = get_output(fp, layer);
    TensorView& output_delta = get_output_delta(bp, layer);

    activation.apply_delta(output, output_delta);

    const TensorView& input = get_input(fp, layer);

    TensorView empty;
    TensorView& input_delta = view_at_slot_or(dv, input_delta_slots, 0, empty);

    combination.apply_delta(output_delta, input, input_delta, false);
}

void RecurrentOp::set(Index new_input_features,
                      Index new_time_steps,
                      Index new_output_features,
                      ActivationOp::Function new_activation,
                      Type new_weight_type)
{
    input_features  = new_input_features;
    time_steps      = new_time_steps;
    output_features = new_output_features;
    activation      = new_activation;
    weight_type     = new_weight_type;
}

vector<TensorSpec> RecurrentOp::parameter_specs() const
{
    return {
        {{output_features},                   weight_type},  // bias
        {{input_features, output_features},   weight_type},  // input weights
        {{output_features, output_features},  weight_type},  // recurrent weights
    };
}

void RecurrentOp::link_parameters(span<const TensorView> views)
{
    if (views.size() < 3) return;
    bias              = views[0];
    input_weights     = views[1];
    recurrent_weights = views[2];
}

void RecurrentOp::link_gradients(span<const TensorView> views)
{
    if (views.size() < 3) return;
    bias_gradient              = views[0];
    input_weight_gradient      = views[1];
    recurrent_weight_gradient  = views[2];
}

void RecurrentOp::set_parameters_random()
{
    if (!input_weights.empty())     set_random_uniform(input_weights.as_vector());
    if (!recurrent_weights.empty()) set_random_uniform(recurrent_weights.as_vector());
    if (!bias.empty())              bias.setZero();
}

void RecurrentOp::set_parameters_glorot()
{
    if (!input_weights.empty())
    {
        const float limit = glorot_limit(input_features, output_features);
        set_random_uniform(input_weights.as_vector(), -limit, limit);
    }
    if (!recurrent_weights.empty())
    {
        const float limit = glorot_limit(output_features, output_features);
        set_random_uniform(recurrent_weights.as_vector(), -limit, limit);
    }
    if (!bias.empty()) bias.setZero();
}

void RecurrentOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept
{
    // Slot convention (set by the hosting layer):
    //   output_slots = {Output, HiddenStates, ActivationDerivatives}
    auto& fv = fp.views[layer];
    const TensorView& input             = get_input(fp, layer);
    TensorView& output                  = fv[output_slots[0]][0];
    TensorView& hidden_states           = fv[output_slots[1]][0];
    TensorView& activation_derivatives  = fv[output_slots[2]][0];

    IF_GPU({ throw runtime_error("Recurrent GPU path is not implemented yet. Call Configuration::instance().set(Device::CPU, ...) before training a network that contains a Recurrent layer."); });
    apply(input, hidden_states, activation_derivatives, output, is_training);
}

void RecurrentOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& fv = fp.views[layer];
    auto& dv = bp.delta_views[layer];

    const TensorView& input                    = get_input(fp, layer);
    const TensorView& hidden_states            = fv[output_slots[1]][0];
    const TensorView& activation_derivatives   = fv[output_slots[2]][0];
    const TensorView& output_delta             = get_output_delta(bp, layer);

    TensorView empty;
    TensorView& input_delta = view_at_slot_or(dv, input_delta_slots, 0, empty);

    IF_GPU({ throw runtime_error("Recurrent GPU path is not implemented yet. Call Configuration::instance().set(Device::CPU, ...) before training a network that contains a Recurrent layer."); });
    apply_delta(input, hidden_states, activation_derivatives, output_delta, input_delta);
}

void RecurrentOp::apply(const TensorView& input,
                            TensorView& hidden_states,
                            TensorView& activation_derivatives,
                            TensorView& output,
                            bool is_training)
{
    const Index batch_size = input.shape[0];

    const VectorMap bias_map  = bias.as_vector();
    const MatrixMap w_in_map  = input_weights.as_matrix();
    const MatrixMap w_rec_map = recurrent_weights.as_matrix();

    const float* input_data  = input.as<float>();
    float*       hidden_data = hidden_states.as<float>();
    float*       derivs_data = (is_training && !activation_derivatives.empty())
                               ? activation_derivatives.as<float>() : nullptr;

    const Index in_stride_t = input_features;             // stride between consecutive time steps (one batch row)
    const Index in_stride_b = time_steps * input_features; // stride between batch rows
    const Index h_stride_t  = output_features;
    const Index h_stride_b  = time_steps * output_features;

    MatrixR step_input  (batch_size, input_features);
    MatrixR step_hidden (batch_size, output_features);
    MatrixR prev_hidden (batch_size, output_features);
    MatrixR step_derivs (batch_size, output_features);

    for (Index t = 0; t < time_steps; ++t)
    {
        for (Index i = 0; i < batch_size; ++i)
            std::memcpy(step_input.data() + i * input_features,
                        input_data + i * in_stride_b + t * in_stride_t,
                        input_features * sizeof(float));

        step_hidden.noalias() = step_input * w_in_map;
        step_hidden.rowwise() += bias_map.transpose();

        if (t > 0)
            step_hidden.noalias() += prev_hidden * w_rec_map;

        using F = ActivationOp::Function;
        switch (activation)
        {
        case F::Tanh:
            step_hidden = step_hidden.array().tanh();
            if (is_training)
                step_derivs = 1.0f - step_hidden.array().square();
            break;
        case F::Sigmoid:
            step_hidden = (1.0f / (1.0f + (-step_hidden.array()).exp())).matrix();
            if (is_training)
                step_derivs = step_hidden.array() * (1.0f - step_hidden.array());
            break;
        case F::ReLU:
            if (is_training)
                step_derivs = (step_hidden.array() > 0.0f).cast<float>();
            step_hidden = step_hidden.array().cwiseMax(0.0f);
            break;
        case F::Identity:
        case F::Softmax: // Softmax over time is degenerate for an RNN; treat as Identity.
            if (is_training)
                step_derivs.setOnes();
            break;
        }

        for (Index i = 0; i < batch_size; ++i)
            std::memcpy(hidden_data + i * h_stride_b + t * h_stride_t,
                        step_hidden.data() + i * output_features,
                        output_features * sizeof(float));

        if (derivs_data)
            for (Index i = 0; i < batch_size; ++i)
                std::memcpy(derivs_data + i * h_stride_b + t * h_stride_t,
                            step_derivs.data() + i * output_features,
                            output_features * sizeof(float));

        prev_hidden = step_hidden;
    }

    output.as_matrix() = prev_hidden;
}

void RecurrentOp::apply_delta(const TensorView& input,
                              const TensorView& hidden_states,
                              const TensorView& activation_derivatives,
                              const TensorView& output_delta,
                              TensorView& input_delta) const
{
    const Index batch_size = input.shape[0];

    const MatrixMap w_in_map  = input_weights.as_matrix();
    const MatrixMap w_rec_map = recurrent_weights.as_matrix();

    VectorMap bias_grad   = bias_gradient.as_vector();
    MatrixMap w_in_grad   = input_weight_gradient.as_matrix();
    MatrixMap w_rec_grad  = recurrent_weight_gradient.as_matrix();

    bias_grad.setZero();
    w_in_grad.setZero();
    w_rec_grad.setZero();

    const float* input_data  = input.as<float>();
    const float* hidden_data = hidden_states.as<float>();
    const float* derivs_data = activation_derivatives.as<float>();
    const MatrixMap out_delta = output_delta.as_matrix();

    const bool write_input_delta = !input_delta.empty() && input_delta.data != nullptr;
    float* input_delta_data = write_input_delta ? input_delta.as<float>() : nullptr;

    const Index in_stride_t = input_features;
    const Index in_stride_b = time_steps * input_features;
    const Index h_stride_t  = output_features;
    const Index h_stride_b  = time_steps * output_features;

    MatrixR delta        (batch_size, output_features);
    MatrixR next_carry   = MatrixR::Zero(batch_size, output_features);
    MatrixR step_input   (batch_size, input_features);
    MatrixR step_prev_h  (batch_size, output_features);
    MatrixR step_derivs  (batch_size, output_features);
    MatrixR step_in_delta(batch_size, input_features);

    for (Index t = time_steps - 1; t >= 0; --t)
    {
        delta = (t == time_steps - 1) ? out_delta : next_carry;

        for (Index i = 0; i < batch_size; ++i)
            std::memcpy(step_derivs.data() + i * output_features,
                        derivs_data + i * h_stride_b + t * h_stride_t,
                        output_features * sizeof(float));

        delta.array() *= step_derivs.array();

        for (Index i = 0; i < batch_size; ++i)
            std::memcpy(step_input.data() + i * input_features,
                        input_data + i * in_stride_b + t * in_stride_t,
                        input_features * sizeof(float));

        w_in_grad.noalias() += step_input.transpose() * delta;
        bias_grad.noalias() += delta.colwise().sum().transpose();

        if (t > 0)
        {
            for (Index i = 0; i < batch_size; ++i)
                std::memcpy(step_prev_h.data() + i * output_features,
                            hidden_data + i * h_stride_b + (t - 1) * h_stride_t,
                            output_features * sizeof(float));

            w_rec_grad.noalias() += step_prev_h.transpose() * delta;
            next_carry.noalias()  = delta * w_rec_map.transpose();
        }

        if (write_input_delta)
        {
            step_in_delta.noalias() = delta * w_in_map.transpose();

            for (Index i = 0; i < batch_size; ++i)
                std::memcpy(input_delta_data + i * in_stride_b + t * in_stride_t,
                            step_in_delta.data() + i * input_features,
                            input_features * sizeof(float));
        }
    }
}

void ConvolutionOp::set(Index new_input_h, Index new_input_w,
                      Index new_kernels_n, Index new_kernel_h, Index new_kernel_w, Index new_kernel_c,
                      [[maybe_unused]] Index new_row_stride, [[maybe_unused]] Index new_column_stride,
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

vector<TensorSpec> ConvolutionOp::parameter_specs() const
{
    return {
        {{kernels_number}, compute_dtype},                                                       // Bias
        {{kernels_number, kernel_height, kernel_width, kernel_channels}, compute_dtype},         // Weight
    };
}

void ConvolutionOp::link_parameters(span<const TensorView> views)
{
    if (views.size() < 2) return;
    bias    = views[0];
    weights = views[1];
}

void ConvolutionOp::link_gradients(span<const TensorView> views)
{
    if (views.size() < 2) return;
    bias_gradient   = views[0];
    weight_gradient = views[1];
}

void ConvolutionOp::set_parameters_random()
{
    if (weights.empty()) return;
    set_random_uniform(weights.as_vector());
    if (!bias.empty()) bias.setZero();
}

void ConvolutionOp::set_parameters_glorot()
{
    if (weights.empty()) return;
    const Index kernel_area = kernel_height * kernel_width;
    const float limit = glorot_limit(kernel_area * kernel_channels, kernel_area * kernels_number);
    set_random_uniform(weights.as_vector(), -limit, limit);
    if (!bias.empty()) bias.setZero();
}

void ConvolutionOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    const TensorView& input = get_input(fp, layer);
    TensorView& output      = get_output(fp, layer);

    apply(input, output);
}

void ConvolutionOp::apply_delta(const TensorView& input,
                                const TensorView& output_delta,
                                TensorView& input_delta) const
{
    IF_GPU({ apply_delta_gpu(input, output_delta, input_delta); return; });
    apply_delta_cpu(input, output_delta, input_delta);
}

void ConvolutionOp::apply(const TensorView& input, TensorView& output, cudnnActivationDescriptor_t fused_activation)
{
    IF_GPU({ apply_gpu(input, output, fused_activation); return; });
    apply_cpu(input, output);
}

void ConvolutionOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& dv = bp.delta_views[layer];

    const TensorView& input        = get_input(fp, layer);
    const TensorView& output_delta = get_output_delta(bp, layer);

    TensorView empty;
    TensorView& input_delta = view_at_slot_or(dv, input_delta_slots, 0, empty);

    apply_delta(input, output_delta, input_delta);
}

array<pair<Index, Index>, 4> ConvolutionOp::nhwc_padding() const
{
    return {
        make_pair(Index(0), Index(0)),
        make_pair(padding_height, padding_height),
        make_pair(padding_width,  padding_width),
        make_pair(Index(0), Index(0))
    };
}

void ConvolutionOp::apply_cpu(const TensorView& input, TensorView& output)
{
    const TensorMap4 inputs = input.as_tensor<4>();
    const VectorMap biases = bias.as_vector();

    const Index batch_size = inputs.dimension(0);

    const array<Index, 3> conv_dims({1, 2, 3});
    const array<Index, 3> out_slice_shape({batch_size, output.shape[1], output.shape[2]});

    const auto input_paddings = nhwc_padding();

    TensorMap4 outputs = output.as_tensor<4>();

    // Pad inputs ONCE before the kernel loop. Doing inputs.pad() inside the
    // loop would re-materialize the padded tensor per kernel (kernels_number
    // times) and was the main forward-pass perf gap vs. master.
    const bool needs_padding = padding_height > 0 || padding_width > 0;
    const Tensor4 padded_inputs_storage = needs_padding
        ? Tensor4(inputs.pad(input_paddings))
        : Tensor4();
    const TensorMap4 padded_inputs = needs_padding
        ? TensorMap4(const_cast<float*>(padded_inputs_storage.data()),
                     padded_inputs_storage.dimensions())
        : inputs;

    for (Index kernel_index = 0; kernel_index < kernels_number; ++kernel_index)
    {
        const TensorMap3 kernel_map = weights.as_tensor<3>(kernel_index);

        outputs.chip(kernel_index, 3).device(get_device()) =
            padded_inputs.convolve(kernel_map, conv_dims).reshape(out_slice_shape) + biases(kernel_index);
    }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
void ConvolutionOp::apply_delta_cpu(const TensorView& input,
                                  const TensorView& output_delta,
                                  TensorView& input_delta) const
{
    const TensorMap4 inputs        = input.as_tensor<4>();
    const TensorMap4 output_deltas = output_delta.as_tensor<4>();

    const Index kernel_size = kernel_height * kernel_width * kernel_channels;

    MatrixMap output_deltas_mat = output_delta.as_flat_matrix();
    bias_gradient.as_vector().noalias() = output_deltas_mat.colwise().sum();

    float* weight_data = weight_gradient.as<float>();

    const Tensor4 padded_inputs = inputs.pad(nhwc_padding());

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
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#ifdef OPENNN_HAS_CUDA

void ConvolutionOp::destroy_cuda()
{
    if (kernel_descriptor)      { cudnnDestroyFilterDescriptor(kernel_descriptor);           kernel_descriptor = nullptr; }
    if (convolution_descriptor) { cudnnDestroyConvolutionDescriptor(convolution_descriptor); convolution_descriptor = nullptr; }
    cudnn_workspace_size_ = 0;
    planned_batch_size = 0;
}
void ConvolutionOp::plan_convolution_algorithms(const TensorView& input, const TensorView& output)
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

    cudnn_workspace_size_ = max({fwd_ws, bwd_data_ws, bwd_filter_ws});
    scratch::ensure_cudnn_conv_workspace(cudnn_workspace_size_);

    planned_batch_size = input.shape[0];
}

void ConvolutionOp::apply_gpu(const TensorView& input,
                            TensorView& output,
                            cudnnActivationDescriptor_t fused_activation)
{
    if (input.shape[0] > planned_batch_size)
        plan_convolution_algorithms(input, output);

    void* ws_ptr = scratch::ensure_cudnn_conv_workspace(cudnn_workspace_size_);

    if (fused_activation)
    {
        CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
            Backend::get_cudnn_handle(),
            &one,
            input.get_descriptor(),  input.data,
            kernel_descriptor,        weights.data,
            convolution_descriptor,
            algorithm_forward,
            ws_ptr, cudnn_workspace_size_,
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
                                        ws_ptr, cudnn_workspace_size_,
                                        &zero,
                                        output.get_descriptor(), output.data));

    CHECK_CUDNN(cudnnAddTensor(Backend::get_cudnn_handle(),
                               &one,
                               bias.get_descriptor(), bias.data,
                               &one,
                               output.get_descriptor(), output.data));
}

void ConvolutionOp::apply_delta_gpu(const TensorView& input,
                                  const TensorView& output_delta,
                                  TensorView& input_delta) const
{
    assert(output_delta.type == input.type);
    assert(weight_gradient.type == Type::FP32);

    const bool bf16 = (input.type == Type::BF16);

    void* weight_gradient_buffer = weight_gradient.data;
    bfloat16* weight_gradient_bf16_scratch = nullptr;

    if (bf16)
    {
        weight_gradient_bf16_scratch = scratch::ensure_bf16_gradient_scratch(weight_gradient.size());
        weight_gradient_buffer = weight_gradient_bf16_scratch;
    }

    void* ws_ptr = scratch::ensure_cudnn_conv_workspace(cudnn_workspace_size_);

    CHECK_CUDNN(cudnnConvolutionBackwardFilter(Backend::get_cudnn_handle(),
        &one,
        input.get_descriptor(),        input.data,
        output_delta.get_descriptor(), output_delta.data,
        convolution_descriptor,
        algorithm_filter,
        ws_ptr, cudnn_workspace_size_,
        &zero,
        kernel_descriptor, weight_gradient_buffer));

    TensorView output_delta_for_bias = output_delta;

    if (bf16)
    {
        float* const output_delta_fp32 = scratch::ensure_fp32_upcast_scratch(output_delta.size());
        cast_bf16_to_fp32_cuda(output_delta.size(),
                               output_delta.as<bfloat16>(),
                               output_delta_fp32);

        output_delta_for_bias = TensorView(output_delta_fp32, output_delta.shape, Type::FP32);
    }

    CHECK_CUDNN(cudnnConvolutionBackwardBias(Backend::get_cudnn_handle(),
        &one,
        output_delta_for_bias.get_descriptor(), output_delta_for_bias.data,
        &zero,
        bias_gradient.get_descriptor(), bias_gradient.data));

    if (bf16)
        cast_bf16_to_fp32_cuda(weight_gradient.size(), weight_gradient_bf16_scratch, weight_gradient.as_float());

    if (!input_delta.data || input_delta.size() == 0) return;

    CHECK_CUDNN(cudnnConvolutionBackwardData(Backend::get_cudnn_handle(),
        &one,
        kernel_descriptor, weights.data,
        output_delta.get_descriptor(), output_delta.data,
        convolution_descriptor,
        algorithm_data,
        ws_ptr, cudnn_workspace_size_,
        &zero,
        input_delta.get_descriptor(), input_delta.data));
}

#else

void ConvolutionOp::destroy_cuda()                                                                  {}
void ConvolutionOp::apply_gpu(const TensorView&, TensorView&, cudnnActivationDescriptor_t)          { throw runtime_error("Convolution::apply_gpu: CUDA support not compiled in."); }
void ConvolutionOp::apply_delta_gpu(const TensorView&, const TensorView&, TensorView&) const { throw runtime_error("Convolution::apply_delta_gpu: CUDA support not compiled in."); }

#endif


void ConvolutionReluOp::set(Index input_h, Index input_w,
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

    activation.set_function(ActivationOp::Function::ReLU);
}

void ConvolutionReluOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    const TensorView& input = get_input(fp, layer);
    TensorView& output      = get_output(fp, layer);

    IF_GPU({ convolution.apply(input, output, activation.descriptor); return; });
    convolution.apply(input, output);
    activation_forward(output, activation.function);
}

void ConvolutionReluOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& dv = bp.delta_views[layer];

    const TensorView& output = get_output(fp, layer);
    TensorView& output_delta = get_output_delta(bp, layer);

    activation.apply_delta(output, output_delta);

    const TensorView& input = get_input(fp, layer);

    TensorView empty;
    TensorView& input_delta = view_at_slot_or(dv, input_delta_slots, 0, empty);

    convolution.apply_delta(input, output_delta, input_delta);
}


void LayerNormOp::set(Index new_sequence_length, Index new_embedding_dimension)
{
    sequence_length     = new_sequence_length;
    embedding_dimension = new_embedding_dimension;
}

vector<TensorSpec> LayerNormOp::parameter_specs() const
{
    // Gamma, Beta
    return vector<TensorSpec>(2, {Shape{embedding_dimension}, Type::FP32});
}

void LayerNormOp::link_parameters(span<const TensorView> views)
{
    if (views.size() < 2) return;
    gamma = views[0];
    beta  = views[1];
}

void LayerNormOp::link_gradients(span<const TensorView> views)
{
    if (views.size() < 2) return;
    gamma_gradient = views[0];
    beta_gradient  = views[1];
}

void LayerNormOp::init_defaults()
{
    if (gamma.data) gamma.as_vector().setOnes();
    if (beta.data)  beta.as_vector().setZero();
}

void LayerNormOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    const TensorView& input = get_input(fp, layer);
    TensorView& means       = get_output(fp, layer);
    TensorView& stds        = get_output(fp, layer, 1);
    TensorView& normalized  = get_output(fp, layer, 2);
    TensorView& output      = get_output(fp, layer, 3);

    layer_norm_forward(input, gamma, beta, means, stds, normalized, output);
}

void LayerNormOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    const TensorView& stds         = get_output(fp, layer, 1);
    const TensorView& normalized   = get_output(fp, layer, 2);
    const TensorView& output_delta = get_output_delta(bp, layer);
    TensorView& input_delta        = get_input_delta(bp, layer);

    layer_norm_backward(get_input(fp, layer), output_delta, get_output(fp, layer),
                        stds, normalized, gamma, gamma_gradient, beta_gradient,
                        input_delta);
}

void MultiHeadProjectionOp::set(Index new_input_features, Index new_heads_number,
                              Index new_head_dimension, Type new_compute_dtype)
{
    input_features = new_input_features;
    compute_dtype  = new_compute_dtype;

    combination.set(input_features, new_heads_number * new_head_dimension, compute_dtype);
}

void MultiHeadProjectionOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    auto& fv = fp.views[layer];
    const auto& input_views = get_inputs(fp, layer);
    const TensorView& input = input_views[min(input_view_index, input_views.size() - 1)];
    TensorView& head_output = get_output(fp, layer);

    const Index batch_size     = input.shape[0];
    const Index seq_len        = input.shape[1];
    const Index rows           = batch_size * seq_len;
    const Index heads_number   = head_output.shape[1];
    const Index head_dimension = head_output.shape[3];

    TensorView& scratch     = fv[scratch_slots[0]][0];
    TensorView  scratch_2d  = scratch.reshape({rows, input_features});
    TensorView  scratch_4d  = scratch.reshape({batch_size, seq_len, heads_number, head_dimension});
    TensorView  input_2d    = input.reshape({rows, input_features});

    combination.apply(input_2d, scratch_2d);
    split_heads(scratch_4d, head_output);
}

void MultiHeadProjectionOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& fv = fp.views[layer];
    auto& dv = bp.delta_views[layer];

    const auto& input_views = get_inputs(fp, layer);
    const TensorView& input = input_views[min(input_view_index, input_views.size() - 1)];
    const bool self_attention = (input_views.size() == 1);

    const TensorView& head_delta = get_output_delta(bp, layer);

    const Index batch_size     = input.shape[0];
    const Index seq_len        = input.shape[1];
    const Index rows           = batch_size * seq_len;
    const Index heads_number   = head_delta.shape[1];
    const Index head_dimension = head_delta.shape[3];

    TensorView& scratch     = fv[scratch_slots[0]][0];
    TensorView  scratch_4d  = scratch.reshape({batch_size, seq_len, heads_number, head_dimension});
    TensorView  scratch_2d  = scratch.reshape({rows, input_features});
    TensorView  input_2d    = input.reshape({rows, input_features});

    merge_heads(head_delta, scratch_4d);

    TensorView& input_delta    = dv[(self_attention ? input_delta_slots_self : input_delta_slots_cross)[0]];
    TensorView  input_delta_2d = input_delta.reshape({rows, input_features});
    const bool  accumulate     = self_attention ? accumulate_input_delta_self : accumulate_input_delta_cross;

    combination.apply_delta(scratch_2d, input_2d, input_delta_2d, accumulate);
}

void AttentionOp::set(Index new_heads_number, Index new_head_dimension,
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

float AttentionOp::scaling_factor() const
{
    return (head_dimension == 0) ? 0.25f : 1.0f / float(sqrt(head_dimension));
}

bool AttentionOp::get_contiguous_source_lengths(const TensorView& source_input,
                                                vector<Index>& lengths,
                                                bool& has_padding)
{
    if (source_input.shape.rank != 3 || source_input.type != Type::FP32)
        return false;

    const Index batch_size          = source_input.shape[0];
    const Index sequence_length     = source_input.shape[1];
    const Index embedding_dimension = source_input.shape[2];
    const float* source_data        = source_input.as<float>();

    auto row_nonzero = [embedding_dimension](const float* row) {
        for (Index j = 0; j < embedding_dimension; ++j)
            if (abs(row[j]) > EPSILON) return true;
        return false;
    };

    lengths.assign(batch_size, sequence_length);
    has_padding = false;

    for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        const float* batch = source_data + batch_index * sequence_length * embedding_dimension;

        Index valid_length = 0;
        while (valid_length < sequence_length
               && row_nonzero(batch + valid_length * embedding_dimension))
            ++valid_length;

        if (valid_length == 0) return false;

        for (Index i = valid_length; i < sequence_length; ++i)
            if (row_nonzero(batch + i * embedding_dimension))
                return false;

        if (valid_length < sequence_length) has_padding = true;
        lengths[batch_index] = valid_length;
    }

    return true;
}

void AttentionOp::softmax_rows_prefix(float* matrix, Index rows, Index cols, Index length)
{
    for (Index row = 0; row < rows; ++row)
    {
        Eigen::Map<Eigen::VectorXf> v(matrix + row * cols, length);
        v = (v.array() - v.maxCoeff()).exp();
        v /= v.sum();
    }
}

Index AttentionOp::infer_attention_prefix_length(const TensorView& attention_weights,
                                                 Index batch_index)
{
    const auto& shape = attention_weights.shape;
    const float* first_row = attention_weights.as<float>()
        + batch_index * shape[1] * shape[2] * shape[3];

    Index length = shape[3];
    while (length > 0 && first_row[length - 1] == 0.0f)
        --length;

    return length;
}

vector<TensorSpec> AttentionOp::forward_scratch_specs(Index batch_size) const
{
    // SDPA does not materialise the attention matrix and does not use these
    // scratch slots for dropout either, so they collapse to empty placeholders.
    // Unfused path needs a (B, H, Q, K) scratch for the attention weights and,
    // when dropout is active, a matching one for the dropped mask.
    if (use_sdpa && !dropout.active())
        return vector<TensorSpec>(2, {Shape{}, compute_dtype});

    const Shape attention_shape = {batch_size, heads_number,
                                   query_sequence_length, source_sequence_length};
    const Shape dropout_shape = dropout.active() ? attention_shape : Shape{};

    return {
        {attention_shape, compute_dtype}, // AttentionWeights
        {dropout_shape,   compute_dtype}, // AttentionWeightsDropped
    };
}

TensorSpec AttentionOp::backward_scratch_spec(Index batch_size) const
{
    // SDPA backward (cuDNN frontend) computes dQ/dK/dV without materialising
    // the attention weight matrix, so the (B,H,Q,K) scratch can be skipped.
    // Unfused backward still needs it for the softmax-derivative reduction.
    if (use_sdpa)
        return {Shape{}, compute_dtype};

    return {{batch_size, heads_number, query_sequence_length, source_sequence_length},
            compute_dtype};
}

bool AttentionOp::sdpa_supported(Type dtype)
{
#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)
    return is_gpu() && dtype == Type::BF16;
#else
    (void)dtype;
    return false;
#endif
}

#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)

namespace
{

cudnn_frontend::DataType_t to_cudnn_frontend_dtype(Type t)
{
    using enum Type;
    switch (t)
    {
        case FP32: return cudnn_frontend::DataType_t::FLOAT;
        case BF16: return cudnn_frontend::DataType_t::BFLOAT16;
        default:         return cudnn_frontend::DataType_t::FLOAT;
    }
}

}  // namespace

#endif

struct AttentionOp::SDPACache
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

        bool operator==(const CacheKey&) const = default;
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

#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)
    struct Entry
    {
        // Forward graph
        shared_ptr<cudnn_frontend::graph::Graph> fwd_graph;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_Q, fwd_K, fwd_V, fwd_O, fwd_Stats;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_SeqLenQ, fwd_SeqLenKV;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_Seed, fwd_Offset;
        void* fwd_workspace_buf = nullptr;

        // Backward graph (built lazily on first apply_delta_gpu)
        shared_ptr<cudnn_frontend::graph::Graph> bwd_graph;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_Q, bwd_K, bwd_V, bwd_O, bwd_dO, bwd_Stats;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_dQ, bwd_dK, bwd_dV;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_SeqLenQ, bwd_SeqLenKV;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_Seed, bwd_Offset;
        void* bwd_workspace_buf = nullptr;

        // Shared (forward writes, backward reads). LSE stats from softmax.
        void* stats_buf = nullptr;

        // Device-resident scalars for SDPA dropout RNG (1 INT64 each).
        // Allocated only when the cache key has dropout_active=true. Forward
        // writes the per-step (seed, offset); backward writes the same values
        // so cuDNN regenerates an identical mask.
        void* seed_buf   = nullptr;
        void* offset_buf = nullptr;

        // Per-batch INT32 sequence lengths for cuDNN padding masks. Forward
        // refreshes them from source_input; backward reuses the same batch.
        void* seq_len_q_buf  = nullptr;
        void* seq_len_kv_buf = nullptr;
    };

    unordered_map<CacheKey, Entry, CacheKeyHash> entries;

    // 1-element shortcut: AttentionOp is typically called with the same shape
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
            if (e.seed_buf)          cudaFree(e.seed_buf);
            if (e.offset_buf)        cudaFree(e.offset_buf);
            if (e.seq_len_q_buf)     cudaFree(e.seq_len_q_buf);
            if (e.seq_len_kv_buf)    cudaFree(e.seq_len_kv_buf);
        }
#endif
    }
#endif  // OPENNN_HAS_CUDA && HAVE_CUDNN_FRONTEND
};

#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)

namespace
{

auto sdpa_check = [](auto s, const string& what) {
    if (s.is_bad())
        throw runtime_error(format("SDPA {}: {}", what, s.get_message()));
};

// {B, H, S, D} contiguous tensor input.
shared_ptr<cudnn_frontend::graph::Tensor_attributes>
bhsd_input(cudnn_frontend::graph::Graph& graph, const char* name, int64_t B, int64_t H, int64_t S, int64_t D)
{
    return graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim   ({B, H, S, D})
                        .set_stride({H * S * D, S * D, D, 1}));
}

shared_ptr<cudnn_frontend::graph::Tensor_attributes>
seq_len_input(cudnn_frontend::graph::Graph& graph, const char* name, int64_t batch_size)
{
    return graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim({batch_size, 1, 1, 1})
                        .set_stride({1, 1, 1, 1})
                        .set_data_type(cudnn_frontend::DataType_t::INT32));
}

void bhsd_output(shared_ptr<cudnn_frontend::graph::Tensor_attributes>& T,
                 int64_t B, int64_t H, int64_t S, int64_t D)
{
    T->set_output(true).set_dim({B, H, S, D}).set_stride({H * S * D, S * D, D, 1});
}

void build_sdpa_graph_common(cudnn_frontend::graph::Graph& graph, Type dtype)
{
    graph.set_io_data_type(to_cudnn_frontend_dtype(dtype))
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
}
void require_attention_scratch(const TensorView& attention_weights, const string& context)
{
    if (attention_weights.empty())
        throw runtime_error(format("Attention: {} — set_dropout_rate must be called before compiling the network on GPU "
                                   "(see Attention::forward_scratch_specs).",
                                   context));
}

void finalize_sdpa_graph(cudnn_frontend::graph::Graph& graph, cudnnHandle_t handle, const string& tag)
{
    sdpa_check(graph.validate(),                                                tag + " validate");
    sdpa_check(graph.build_operation_graph(handle),                             tag + " build_operation_graph");
    sdpa_check(graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}),               tag + " create_execution_plans");
    sdpa_check(graph.build_plans(handle, cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_CHOICE), tag + " build_plans");
}

void refresh_sdpa_sequence_lengths(AttentionOp::SDPACache::Entry& entry,
                                   const AttentionOp::SDPACache::CacheKey& k,
                                   const TensorView& source_input)
{
    const bool ok = source_input.shape.rank == 3 && source_input.shape[0] == k.batch_size && source_input.shape[1] == k.src_seq
        && TRY_GPU_DISPATCH(source_input, [&](auto tag) {
            using T = decltype(tag);
            attention_sequence_lengths_cuda<T>(to_int(k.batch_size),
                                               to_int(k.q_seq),
                                               to_int(k.src_seq),
                                               to_int(source_input.shape[2]),
                                               source_input.as<T>(),
                                               static_cast<int32_t*>(entry.seq_len_q_buf),
                                               static_cast<int32_t*>(entry.seq_len_kv_buf));
        });

    if (!ok)
        throw runtime_error("SDPA padding mask: source_input must be a rank-3 CUDA tensor with supported dtype.");
}

}  // namespace

static void build_sdpa_forward_graph(AttentionOp::SDPACache::Entry& entry,
                                      const AttentionOp::SDPACache::CacheKey& k,
                                      cudnnHandle_t handle,
                                      float dropout_rate)
{
    const auto graph = make_shared<cudnn_frontend::graph::Graph>();
    build_sdpa_graph_common(*graph, k.dtype);

    entry.fwd_Q = bhsd_input(*graph, "Q", k.batch_size, k.heads, k.q_seq,   k.head_dim);
    entry.fwd_K = bhsd_input(*graph, "K", k.batch_size, k.heads, k.src_seq, k.head_dim);
    entry.fwd_V = bhsd_input(*graph, "V", k.batch_size, k.heads, k.src_seq, k.head_dim);
    entry.fwd_SeqLenQ  = seq_len_input(*graph, "SeqLenQ",  k.batch_size);
    entry.fwd_SeqLenKV = seq_len_input(*graph, "SeqLenKV", k.batch_size);

    auto sdpa_options = cudnn_frontend::graph::SDPA_attributes()
                        .set_name("flash_attn_fwd")
                        .set_is_inference(!k.is_training)
                        .set_padding_mask(true)
                        .set_seq_len_q(entry.fwd_SeqLenQ)
                        .set_seq_len_kv(entry.fwd_SeqLenKV)
                        .set_causal_mask(k.causal)
                        .set_attn_scale(1.0f / sqrt(float(k.head_dim)));

    if (!entry.seq_len_q_buf)  CHECK_CUDA(cudaMalloc(&entry.seq_len_q_buf,  size_t(k.batch_size) * sizeof(int32_t)));
    if (!entry.seq_len_kv_buf) CHECK_CUDA(cudaMalloc(&entry.seq_len_kv_buf, size_t(k.batch_size) * sizeof(int32_t)));

    if (k.dropout_active)
    {
        entry.fwd_Seed   = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                         .set_name("Seed").set_dim({1,1,1,1}).set_stride({1,1,1,1})
                                         .set_data_type(cudnn_frontend::DataType_t::INT64));
        entry.fwd_Offset = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                         .set_name("Offset").set_dim({1,1,1,1}).set_stride({1,1,1,1})
                                         .set_data_type(cudnn_frontend::DataType_t::INT64));
        sdpa_options.set_dropout(dropout_rate, entry.fwd_Seed, entry.fwd_Offset);

        if (!entry.seed_buf)   CHECK_CUDA(cudaMalloc(&entry.seed_buf,   sizeof(int64_t)));
        if (!entry.offset_buf) CHECK_CUDA(cudaMalloc(&entry.offset_buf, sizeof(int64_t)));
    }

    auto [O, Stats] = graph->sdpa(entry.fwd_Q, entry.fwd_K, entry.fwd_V, sdpa_options);

    bhsd_output(O, k.batch_size, k.heads, k.q_seq, k.head_dim);
    entry.fwd_O = O;

    if (k.is_training && Stats)
    {
        Stats->set_output(true)
              .set_data_type(cudnn_frontend::DataType_t::FLOAT)
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

static void build_sdpa_backward_graph(AttentionOp::SDPACache::Entry& entry,
                                       const AttentionOp::SDPACache::CacheKey& k,
                                       cudnnHandle_t handle,
                                       float dropout_rate)
{
    const auto graph = make_shared<cudnn_frontend::graph::Graph>();
    build_sdpa_graph_common(*graph, k.dtype);

    entry.bwd_Q  = bhsd_input(*graph, "Q_bwd",  k.batch_size, k.heads, k.q_seq,   k.head_dim);
    entry.bwd_K  = bhsd_input(*graph, "K_bwd",  k.batch_size, k.heads, k.src_seq, k.head_dim);
    entry.bwd_V  = bhsd_input(*graph, "V_bwd",  k.batch_size, k.heads, k.src_seq, k.head_dim);
    entry.bwd_dO = bhsd_input(*graph, "dO_bwd", k.batch_size, k.heads, k.q_seq,   k.head_dim);
    entry.bwd_SeqLenQ  = seq_len_input(*graph, "SeqLenQ_bwd",  k.batch_size);
    entry.bwd_SeqLenKV = seq_len_input(*graph, "SeqLenKV_bwd", k.batch_size);

    // O is read from the layer's ConcatenatedAttentionOutputs buffer, which is
    // physically laid out as {B, Q_seq, H, D} (post merge_heads). Logical shape
    // is {B, H, Q_seq, D} with non-contiguous strides:
    //   stride[B]=Q*H*D, stride[H]=D, stride[Q]=H*D, stride[D]=1
    entry.bwd_O = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                .set_name("O_bwd")
                                .set_dim({k.batch_size, k.heads, k.q_seq, k.head_dim})
                                .set_stride({k.q_seq * k.heads * k.head_dim,
                                             k.head_dim,
                                             k.heads * k.head_dim,
                                             1}));

    entry.bwd_Stats = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                    .set_name("Stats_bwd")
                                    .set_data_type(cudnn_frontend::DataType_t::FLOAT)
                                    .set_dim   ({k.batch_size, k.heads, k.q_seq, 1})
                                    .set_stride({k.heads * k.q_seq, k.q_seq, 1, 1}));

    auto sdpa_bwd_options = cudnn_frontend::graph::SDPA_backward_attributes()
                            .set_name("flash_attn_bwd")
                            .set_padding_mask(true)
                            .set_seq_len_q(entry.bwd_SeqLenQ)
                            .set_seq_len_kv(entry.bwd_SeqLenKV)
                            .set_causal_mask(k.causal)
                            .set_attn_scale(1.0f / sqrt(float(k.head_dim)));

    if (k.dropout_active)
    {
        entry.bwd_Seed   = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                         .set_name("Seed_bwd").set_dim({1,1,1,1}).set_stride({1,1,1,1})
                                         .set_data_type(cudnn_frontend::DataType_t::INT64));
        entry.bwd_Offset = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                         .set_name("Offset_bwd").set_dim({1,1,1,1}).set_stride({1,1,1,1})
                                         .set_data_type(cudnn_frontend::DataType_t::INT64));
        sdpa_bwd_options.set_dropout(dropout_rate, entry.bwd_Seed, entry.bwd_Offset);
    }

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

#endif  // OPENNN_HAS_CUDA && HAVE_CUDNN_FRONTEND

AttentionOp::AttentionOp() = default;
AttentionOp::~AttentionOp() { destroy_cuda(); }
AttentionOp::AttentionOp(AttentionOp&&) noexcept = default;
AttentionOp& AttentionOp::operator=(AttentionOp&&) noexcept = default;

void AttentionOp::destroy_cuda()
{
    sdpa_cache.reset();
}

void AttentionOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept
{
    auto& fv = fp.views[layer];

    const auto& src_views = get_inputs(fp, layer, 3);
    const TensorView& source_input = src_views[min(source_view_index, src_views.size() - 1)];

    const TensorView& query = get_input(fp, layer);

    // TransposeScratch is reused as the per-head attention output {B,H,Q,D}.
    // CPU path also dereferences it as float* for the padding mask scratch
    // — it does that inside apply_cpu without leaking the pointer up here.
    TensorView attention_out = fv[scratch_slots[0]][0].reshape(
        {fp.batch_size, query.shape[1], query.shape[2], query.shape[3]});

    IF_GPU({
        apply_gpu(query, get_input(fp, layer, 1), get_input(fp, layer, 2), source_input,
                  get_output(fp, layer), get_output(fp, layer, 1),
                  attention_out, fv[scratch_slots[0]][0].as<float>(), is_training);
        return;
    });
    apply_cpu(query, get_input(fp, layer, 1), get_input(fp, layer, 2), source_input,
              get_output(fp, layer), get_output(fp, layer, 1),
              attention_out, fv[scratch_slots[0]][0].as<float>(), is_training);
}

void AttentionOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& fv = fp.views[layer];

    const TensorView& query             = get_input(fp, layer);
    const TensorView& key               = get_input(fp, layer, 1);
    const TensorView& value             = get_input(fp, layer, 2);
    const TensorView& attention_output  = fv[attention_output_slots[0]][0];
    const TensorView& attention_weights = get_output(fp, layer);
    const TensorView& attention_weights_dropped = get_output(fp, layer, 1);

    // TransposeScratch reused as output_delta {B,H,Q,D}. Dims read from `query`
    // instead of duplicating member state in the hot path.
    const TensorView output_delta = fv[scratch_slots[0]][0]
        .reshape({fp.batch_size, query.shape[1], query.shape[2], query.shape[3]});

    // delta_views are preallocated by MHA::get_backward_specs with the right
    // shape and dtype; no need to reconstruct them from raw float pointers.
    TensorView& attention_weight_delta = get_output_delta(bp, layer);
    TensorView& query_delta            = get_output_delta(bp, layer, 1);
    TensorView& key_delta              = get_output_delta(bp, layer, 2);
    TensorView& value_delta            = get_output_delta(bp, layer, 3);

    IF_GPU({
        apply_delta_gpu(query, key, value, attention_output,
                        attention_weights, attention_weights_dropped,
                        output_delta,
                        attention_weight_delta,
                        query_delta, key_delta, value_delta);
        return;
    });
    apply_delta_cpu(query, key, value, attention_output,
                    attention_weights, attention_weights_dropped,
                    output_delta,
                    attention_weight_delta,
                    query_delta, key_delta, value_delta);
}

void AttentionOp::apply_cpu(const TensorView& query,
                          const TensorView& key,
                          const TensorView& value,
                          const TensorView& source_input,
                          TensorView& attention_weights,
                          TensorView& attention_weights_dropped,
                          TensorView& output,
                          [[maybe_unused]] void* scratch,
                          bool is_training)
{
    // CPU-only padding-aware fast path: uses Eigen MatrixMap directly on raw
    // pointers, which requires host memory. Gated on !is_gpu() so that the
    // FP32 fallback from apply_gpu (where buffers live on device) falls
    // through to the generic GPU-dispatched path below.
    if (!is_gpu()
        && !use_causal_mask
        && !dropout.active()
        && compute_dtype == Type::FP32
        && query.type == Type::FP32
        && key.type == Type::FP32
        && value.type == Type::FP32
        && attention_weights.type == Type::FP32
        && output.type == Type::FP32
        && source_input.shape.rank == 3
        && attention_weights.shape.rank == 4)
    {
        vector<Index> valid_lengths;
        bool has_padding = false;

        if (get_contiguous_source_lengths(source_input, valid_lengths, has_padding) && has_padding)
        {
            const Index batch_size = source_input.shape[0];
            const Index query_sequence_length = query.shape[2];
            const Index source_sequence_length = key.shape[2];
            const Index batch_heads = batch_size * heads_number;
            const float scale = scaling_factor();

            #pragma omp parallel for
            for (Index batch_head = 0; batch_head < batch_heads; ++batch_head)
            {
                const Index batch_index = batch_head / heads_number;
                const Index valid_length = valid_lengths[batch_index];

                const MatrixMap query_matrix = query.as_matrix(batch_head);
                const MatrixMap key_matrix = key.as_matrix(batch_head);
                const MatrixMap value_matrix = value.as_matrix(batch_head);
                MatrixMap attention_matrix = attention_weights.as_matrix(batch_head);
                MatrixMap output_matrix = output.as_matrix(batch_head);

                auto attention_valid = attention_matrix.leftCols(valid_length);
                attention_valid.noalias() = scale * (query_matrix * key_matrix.topRows(valid_length).transpose());
                if (valid_length < source_sequence_length)
                    attention_matrix.rightCols(source_sequence_length - valid_length).setZero();
                softmax_rows_prefix(attention_matrix.data(), query_sequence_length, source_sequence_length, valid_length);
                output_matrix.noalias() = attention_valid * value_matrix.topRows(valid_length);
            }

            return;
        }
    }

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
                                    reinterpret_cast<T*>(scratch),
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
        dropout_forward(used, dropout.mask, dropout.rate);
    }

    multiply(used, false, value, false, output);
}

void AttentionOp::apply_gpu(const TensorView& query,
                          const TensorView& key,
                          const TensorView& value,
                          const TensorView& source_input,
                          TensorView& attention_weights,
                          TensorView& attention_weights_dropped,
                          TensorView& output,
                          void* scratch,
                          bool is_training)
{
#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)
    // The layer's policy decides whether this AttentionOp runs SDPA. The
    // operator itself just executes — no implicit fallback, because the
    // forward_scratch_specs allocation already committed to one path.
    if (!use_sdpa)
    {
        apply_cpu(query, key, value, source_input,
                  attention_weights, attention_weights_dropped,
                  output, scratch, is_training);
        return;
    }

    if (!sdpa_supported(query.type))
        throw runtime_error("AttentionOp: SDPA backend selected by the layer "
                            "but not supported (build without HAVE_CUDNN_FRONTEND, "
                            "non-BF16 dtype, or CPU runtime).");

    if (!sdpa_cache) sdpa_cache = make_unique<SDPACache>();

    const bool dropout_in_graph = dropout.active() && is_training;

    SDPACache::CacheKey ck{
        query.shape[0],          // batch_size
        query.shape[2],          // q_seq
        key.shape[2],            // src_seq
        heads_number,
        head_dimension,
        query.type,
        dropout_in_graph,
        use_causal_mask,
        is_training
    };

    auto& entry = sdpa_cache->get_or_create_entry(ck);
    if (!entry.fwd_graph)
        build_sdpa_forward_graph(entry, ck, Backend::get_cudnn_handle(), dropout.rate);

    refresh_sdpa_sequence_lengths(entry, ck, source_input);

    if (dropout_in_graph)
    {
        sdpa_last_used_offset = sdpa_dropout_offset;
        const int64_t seed_value   = static_cast<int64_t>(sdpa_dropout_seed);
        const int64_t offset_value = static_cast<int64_t>(sdpa_last_used_offset);
        CHECK_CUDA(cudaMemcpyAsync(entry.seed_buf,   &seed_value,   sizeof(int64_t),
                                   cudaMemcpyHostToDevice, Backend::get_compute_stream()));
        CHECK_CUDA(cudaMemcpyAsync(entry.offset_buf, &offset_value, sizeof(int64_t),
                                   cudaMemcpyHostToDevice, Backend::get_compute_stream()));
        ++sdpa_dropout_offset;
    }

    unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tp;
    tp[entry.fwd_Q] = query.data;
    tp[entry.fwd_K] = key.data;
    tp[entry.fwd_V] = value.data;
    tp[entry.fwd_O] = output.data;
    tp[entry.fwd_SeqLenQ]  = entry.seq_len_q_buf;
    tp[entry.fwd_SeqLenKV] = entry.seq_len_kv_buf;
    if (is_training && entry.fwd_Stats) tp[entry.fwd_Stats] = entry.stats_buf;
    if (dropout_in_graph)
    {
        tp[entry.fwd_Seed]   = entry.seed_buf;
        tp[entry.fwd_Offset] = entry.offset_buf;
    }

    auto status = entry.fwd_graph->execute(Backend::get_cudnn_handle(), tp, entry.fwd_workspace_buf);
    if (status.is_bad())
        throw runtime_error(format("SDPA forward execute: {}", status.get_message()));
#else
    // No cudnn-frontend: fall back to the manual softmax+matmul GPU path.
    apply_cpu(query, key, value, source_input,
              attention_weights, attention_weights_dropped,
              output, scratch, is_training);
#endif
}

template<typename SoftmaxBwd>
void AttentionOp::apply_delta_unfused(const TensorView& query,
                                     const TensorView& key,
                                     const TensorView& value,
                                     const TensorView& attention_weights,
                                     const TensorView& attention_weights_dropped,
                                     const TensorView& output_delta,
                                     TensorView& attention_weight_delta,
                                     TensorView& query_delta,
                                     TensorView& key_delta,
                                     TensorView& value_delta,
                                     SoftmaxBwd&& softmax_bwd) const
{
    const TensorView& attention_used = dropout.active()
        ? attention_weights_dropped
        : attention_weights;

    multiply(attention_used, true, output_delta, false, value_delta);
    multiply(output_delta, false, value, true, attention_weight_delta);

    if (dropout.active())
        dropout_backward(attention_weight_delta, dropout.mask, dropout.rate);

    if (!attention_weight_delta.empty())
        softmax_bwd();

    const float scale = scaling_factor();
    multiply(attention_weight_delta, false, key,   false, query_delta, scale, 0.0f);
    multiply(attention_weight_delta, true,  query, false, key_delta,   scale, 0.0f);
}

void AttentionOp::apply_delta_cpu(const TensorView& query,
                                const TensorView& key,
                                const TensorView& value,
                                const TensorView& /*attention_output*/,
                                const TensorView& attention_weights,
                                const TensorView& attention_weights_dropped,
                                const TensorView& output_delta,
                                TensorView& attention_weight_delta,
                                TensorView& query_delta,
                                TensorView& key_delta,
                                TensorView& value_delta) const
{
    // CPU-only padding-aware fast path: uses Eigen MatrixMap directly on raw
    // pointers, which requires host memory. Gated on !is_gpu() so that the
    // FP32 fallback from apply_delta_gpu falls through to the unfused path
    // below (which uses GPU-dispatched multiply/softmax).
    if (!is_gpu()
        && !use_causal_mask
        && !dropout.active()
        && compute_dtype == Type::FP32
        && query.type == Type::FP32
        && key.type == Type::FP32
        && value.type == Type::FP32
        && attention_weights.type == Type::FP32
        && output_delta.type == Type::FP32
        && attention_weight_delta.type == Type::FP32
        && query_delta.type == Type::FP32
        && key_delta.type == Type::FP32
        && value_delta.type == Type::FP32
        && attention_weights.shape.rank == 4
        && attention_weight_delta.shape.rank == 4)
    {
        const Index batch_size = query.shape[0];
        const Index source_sequence_length = key.shape[2];
        vector<Index> valid_lengths(batch_size);
        bool has_padding = false;
        bool valid_prefixes = true;

        for (Index batch_index = 0; batch_index < batch_size; ++batch_index)
        {
            const Index valid_length = infer_attention_prefix_length(attention_weights, batch_index);
            if (valid_length <= 0 || valid_length > source_sequence_length)
            {
                valid_prefixes = false;
                break;
            }

            valid_lengths[batch_index] = valid_length;
            if (valid_length < source_sequence_length)
                has_padding = true;
        }

        if (valid_prefixes && has_padding)
        {
            const Index batch_heads = batch_size * heads_number;
            const float scale = scaling_factor();

            #pragma omp parallel for
            for (Index batch_head = 0; batch_head < batch_heads; ++batch_head)
            {
                const Index batch_index = batch_head / heads_number;
                const Index valid_length = valid_lengths[batch_index];

                const MatrixMap query_matrix = query.as_matrix(batch_head);
                const MatrixMap key_matrix = key.as_matrix(batch_head);
                const MatrixMap value_matrix = value.as_matrix(batch_head);
                const MatrixMap attention_matrix = attention_weights.as_matrix(batch_head);
                const MatrixMap output_delta_matrix = output_delta.as_matrix(batch_head);

                MatrixMap attention_delta_matrix = attention_weight_delta.as_matrix(batch_head);
                MatrixMap query_delta_matrix = query_delta.as_matrix(batch_head);
                MatrixMap key_delta_matrix = key_delta.as_matrix(batch_head);
                MatrixMap value_delta_matrix = value_delta.as_matrix(batch_head);

                const auto attention_valid = attention_matrix.leftCols(valid_length);
                auto attention_delta_valid = attention_delta_matrix.leftCols(valid_length);

                value_delta_matrix.topRows(valid_length).noalias() =
                    attention_valid.transpose() * output_delta_matrix;

                attention_delta_valid.noalias() =
                    output_delta_matrix * value_matrix.topRows(valid_length).transpose();

                const VectorR dot = (attention_valid.array() * attention_delta_valid.array()).rowwise().sum();
                attention_delta_valid.array() =
                    attention_valid.array() * (attention_delta_valid.colwise() - dot).array();

                query_delta_matrix.noalias() =
                    scale * (attention_delta_valid * key_matrix.topRows(valid_length));
                key_delta_matrix.topRows(valid_length).noalias() =
                    scale * (attention_delta_valid.transpose() * query_matrix);

                if (valid_length < source_sequence_length)
                {
                    attention_delta_matrix.rightCols(source_sequence_length - valid_length).setZero();
                    key_delta_matrix.bottomRows(source_sequence_length - valid_length).setZero();
                    value_delta_matrix.bottomRows(source_sequence_length - valid_length).setZero();
                }
            }

            return;
        }
    }

    apply_delta_unfused(query, key, value,
                        attention_weights, attention_weights_dropped,
                        output_delta, attention_weight_delta,
                        query_delta, key_delta, value_delta,
        [&]() {
            const MatrixMap y  = attention_weights.as_flat_matrix();
            MatrixMap       dY = attention_weight_delta.as_flat_matrix();
            const VectorR dot = (y.array() * dY.array()).rowwise().sum();
            dY.array() = y.array() * (dY.colwise() - dot).array();
        });
}

#ifdef OPENNN_HAS_CUDA

void AttentionOp::apply_delta_gpu_unfused(const TensorView& query,
                                        const TensorView& key,
                                        const TensorView& value,
                                        const TensorView& attention_weights,
                                        const TensorView& attention_weights_dropped,
                                        const TensorView& output_delta,
                                        TensorView& attention_weight_delta,
                                        TensorView& query_delta,
                                        TensorView& key_delta,
                                        TensorView& value_delta) const
{
    apply_delta_unfused(query, key, value,
                        attention_weights, attention_weights_dropped,
                        output_delta, attention_weight_delta,
                        query_delta, key_delta, value_delta,
        [&]() {
            CHECK_CUDNN(cudnnSoftmaxBackward(Backend::get_cudnn_handle(),
                                             CUDNN_SOFTMAX_ACCURATE,
                                             CUDNN_SOFTMAX_MODE_CHANNEL,
                                             &one,
                                             attention_weights.get_descriptor(),         attention_weights.data,
                                             attention_weight_delta.get_descriptor(), attention_weight_delta.data,
                                             &zero,
                                             attention_weight_delta.get_descriptor(), attention_weight_delta.data));
        });
}

#endif

void AttentionOp::apply_delta_gpu(const TensorView& query,
                                const TensorView& key,
                                const TensorView& value,
                                const TensorView& attention_output,
                                const TensorView& attention_weights,
                                const TensorView& attention_weights_dropped,
                                const TensorView& output_delta,
                                TensorView& attention_weight_delta,
                                TensorView& query_delta,
                                TensorView& key_delta,
                                TensorView& value_delta) const
{
#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)
    // Mirrors the forward path: backend is decided by the hosting layer at
    // configuration time and locked in by forward_scratch_specs. The forward
    // populated sdpa_cache iff use_sdpa is true.
    if (!use_sdpa)
    {
        apply_delta_gpu_unfused(query, key, value,
                                attention_weights, attention_weights_dropped,
                                output_delta, attention_weight_delta,
                                query_delta, key_delta, value_delta);
        return;
    }

    if (!sdpa_supported(query.type) || !sdpa_cache)
        throw runtime_error("AttentionOp: SDPA backward called without a live SDPA "
                            "forward graph (use_sdpa set inconsistently between fwd/bwd).");

    const bool dropout_in_graph = dropout.active();   // backward implies training

    SDPACache::CacheKey ck{
        query.shape[0],
        query.shape[2],
        key.shape[2],
        heads_number,
        head_dimension,
        query.type,
        dropout_in_graph,
        use_causal_mask,
        true                    // backward implies training
    };

    SDPACache::Entry* entry_ptr = sdpa_cache->find_entry(ck);
    if (!entry_ptr || !entry_ptr->fwd_graph)
        // Under the current policy (layer-decided, no silent fallback) the
        // forward path always populates the cache when use_sdpa is true, so a
        // missing entry here means the cache key drifted between fwd and bwd
        // (e.g. batch size changed mid-iteration). The unfused scratch is no
        // longer allocated in that case — falling back would just crash later.
        throw runtime_error("AttentionOp::apply_delta_gpu: SDPA forward did not populate "
                            "a cache entry for this shape. Cache key drifted between "
                            "forward and backward (likely batch size changing across "
                            "iterations under use_sdpa).");

    auto& entry = *entry_ptr;
    if (!entry.bwd_graph)
        build_sdpa_backward_graph(entry, ck, Backend::get_cudnn_handle(), dropout.rate);

    if (dropout_in_graph)
    {
        const int64_t seed_value   = static_cast<int64_t>(sdpa_dropout_seed);
        const int64_t offset_value = static_cast<int64_t>(sdpa_last_used_offset);
        CHECK_CUDA(cudaMemcpyAsync(entry.seed_buf,   &seed_value,   sizeof(int64_t),
                                   cudaMemcpyHostToDevice, Backend::get_compute_stream()));
        CHECK_CUDA(cudaMemcpyAsync(entry.offset_buf, &offset_value, sizeof(int64_t),
                                   cudaMemcpyHostToDevice, Backend::get_compute_stream()));
    }

    unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tp;
    tp[entry.bwd_Q]     = const_cast<float*>(query.as<float>());
    tp[entry.bwd_K]     = const_cast<float*>(key.as<float>());
    tp[entry.bwd_V]     = const_cast<float*>(value.as<float>());
    tp[entry.bwd_O]     = const_cast<float*>(attention_output.as<float>());
    tp[entry.bwd_dO]    = const_cast<float*>(output_delta.as<float>());
    tp[entry.bwd_Stats] = entry.stats_buf;
    tp[entry.bwd_dQ]    = query_delta.data;
    tp[entry.bwd_dK]    = key_delta.data;
    tp[entry.bwd_dV]    = value_delta.data;
    tp[entry.bwd_SeqLenQ]  = entry.seq_len_q_buf;
    tp[entry.bwd_SeqLenKV] = entry.seq_len_kv_buf;
    if (dropout_in_graph)
    {
        tp[entry.bwd_Seed]   = entry.seed_buf;
        tp[entry.bwd_Offset] = entry.offset_buf;
    }

    auto status = entry.bwd_graph->execute(Backend::get_cudnn_handle(), tp, entry.bwd_workspace_buf);
    if (status.is_bad())
        throw runtime_error(format("SDPA backward execute: {}", status.get_message()));
#elif defined(OPENNN_HAS_CUDA)
    apply_delta_gpu_unfused(query, key, value,
                            attention_weights, attention_weights_dropped,
                            output_delta, attention_weight_delta,
                            query_delta, key_delta, value_delta);
#else
    apply_delta_cpu(query, key, value, attention_output,
                    attention_weights, attention_weights_dropped,
                    output_delta, attention_weight_delta,
                    query_delta, key_delta, value_delta);
#endif
}

void AttentionOp::to_JSON(JsonWriter&) const
{
    // Per-call config (heads/seqs/causal_mask/dtype) lives at the layer level.
    // DropoutOp rate is a runtime knob, not serialized — preserves existing schema.
}

void AttentionOp::from_JSON(const Json*)
{
    // No-op for the same reason.
}


void MergeOp::set(Index new_heads_number, Index new_query_sequence_length, Index new_head_dimension, Type new_compute_dtype)
{
    heads_number          = new_heads_number;
    query_sequence_length = new_query_sequence_length;
    head_dimension        = new_head_dimension;
    compute_dtype         = new_compute_dtype;
}

void MergeOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    const Index batch_size = fp.batch_size;

    const TensorView source_4d(get_input(fp, layer).as<float>(),
                               {batch_size, heads_number, query_sequence_length, head_dimension},
                               compute_dtype);
    TensorView dest_4d = get_output(fp, layer).reshape({batch_size, query_sequence_length, heads_number, head_dimension});

    merge_heads(source_4d, dest_4d);
}

void MergeOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    const Index batch_size = fp.batch_size;

    const TensorView concat_gradient_4d = get_output_delta(bp, layer)
        .reshape({batch_size, query_sequence_length, heads_number, head_dimension});
    TensorView heads_gradient_4d = get_input(fp, layer)
        .reshape({batch_size, heads_number, query_sequence_length, head_dimension});

    split_heads(concat_gradient_4d, heads_gradient_4d);
}


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
    if (pool_height <= 0 || pool_width <= 0) return;

    if (!pooling_descriptor) CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));

    const cudnnPoolingMode_t mode = (method == Max)
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

void PoolOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept
{
    auto& fv = fp.views[layer];
    const TensorView& input = get_input(fp, layer);
    TensorView& output      = get_output(fp, layer);

    TensorView empty;
    TensorView& indices = view_at_slot_or(fv, output_slots, 1, empty);

    IF_GPU({ apply_gpu(input, output); return; });
    apply_cpu(input, output, indices, is_training);
}

void PoolOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& fv = fp.views[layer];

    const TensorView& output_delta = get_output_delta(bp, layer);
    TensorView& input_delta        = get_input_delta(bp, layer);

    TensorView empty;
    const TensorView& indices = view_at_slot_or(fv, output_slots, 1, empty);

    IF_GPU({
        const TensorView& input = get_input(fp, layer);
        const TensorView& output = get_output(fp, layer);
        apply_delta_gpu(input, output, output_delta, input_delta);
        return;
    });

    apply_delta_cpu(output_delta, indices, input_delta);
}

void Pool3dOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training) noexcept
{
    const TensorView& input = get_input(fp, layer);
    TensorView& output      = get_output(fp, layer);
    TensorView& indices     = get_output(fp, layer, 1);

    if (method == Max)
        max_pooling_3d_forward(input, output, indices, is_training);
    else
        average_pooling_3d_forward(input, output);
}

void Pool3dOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    const TensorView& output_delta = get_output_delta(bp, layer);
    TensorView& input_delta        = get_input_delta(bp, layer);

    if (method == Max)
        max_pooling_3d_backward(get_output(fp, layer, 1), output_delta, input_delta);
    else
        average_pooling_3d_backward(get_input(fp, layer), output_delta, input_delta);
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

    if (method == Max && is_training)
    {
        TensorMap4 indices_map = maximal_indices.as_tensor<4>();
        for_each_pool_window(*this, batch_size, output_height, output_width,
            [&](const PoolWindow& w) {
                float best = NEG_INFINITY;
                Index argmax = 0;
                for (Index pr = w.pr_start; pr < w.pr_end; ++pr)
                    for (Index pc = w.pc_start; pc < w.pc_end; ++pc)
                    {
                        const float v = inputs(w.batch, w.in_row_start + pr,
                                                w.in_col_start + pc, w.channel);
                        if (v > best) { best = v; argmax = pr * pool_width + pc; }
                    }
                outputs(w.batch, w.out_row, w.out_col, w.channel) = best;
                indices_map(w.batch, w.out_row, w.out_col, w.channel) = argmax;
            });
        return;
    }

    if (method == Max)
    {
        for_each_pool_window(*this, batch_size, output_height, output_width,
            [&](const PoolWindow& w) {
                float best = NEG_INFINITY;
                for (Index pr = w.pr_start; pr < w.pr_end; ++pr)
                    for (Index pc = w.pc_start; pc < w.pc_end; ++pc)
                    {
                        const float v = inputs(w.batch, w.in_row_start + pr,
                                                w.in_col_start + pc, w.channel);
                        if (v > best) best = v;
                    }
                outputs(w.batch, w.out_row, w.out_col, w.channel) = best;
            });
        return;
    }

    const float inv_pool_size = 1.0f / (pool_height * pool_width);
    for_each_pool_window(*this, batch_size, output_height, output_width,
        [&](const PoolWindow& w) {
            float sum = 0;
            for (Index pr = w.pr_start; pr < w.pr_end; ++pr)
                for (Index pc = w.pc_start; pc < w.pc_end; ++pc)
                    sum += inputs(w.batch, w.in_row_start + pr,
                                  w.in_col_start + pc, w.channel);
            outputs(w.batch, w.out_row, w.out_col, w.channel) = sum * inv_pool_size;
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
        [&](const PoolWindow& w) {
            const float avg_delta = output_deltas(w.batch, w.out_row, w.out_col, w.channel) * inv_pool_size;
            for (Index pr = w.pr_start; pr < w.pr_end; ++pr)
                for (Index pc = w.pc_start; pc < w.pc_end; ++pc)
                    input_deltas(w.batch, w.in_row_start + pr,
                                 w.in_col_start + pc, w.channel) += avg_delta;
        });
}

#ifdef OPENNN_HAS_CUDA

void PoolOp::destroy_cuda()
{
    if (pooling_descriptor) { cudnnDestroyPoolingDescriptor(pooling_descriptor); pooling_descriptor = nullptr; }
}

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

void PoolOp::destroy_cuda()                                                                           {}
void PoolOp::apply_gpu(const TensorView&, TensorView&)                                                { throw runtime_error("Pool::apply_gpu: CUDA support not compiled in."); }
void PoolOp::apply_delta_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&) const { throw runtime_error("Pool::apply_delta_gpu: CUDA support not compiled in."); }

#endif


void EmbeddingLookupOp::set(Index new_vocabulary_size, Index new_sequence_length, Index new_embedding_dimension)
{
    vocabulary_size     = new_vocabulary_size;
    sequence_length     = new_sequence_length;
    embedding_dimension = new_embedding_dimension;
}

vector<TensorSpec> EmbeddingLookupOp::parameter_specs() const
{
    return {{{vocabulary_size, embedding_dimension}, Type::FP32}};
}

vector<TensorSpec> EmbeddingLookupOp::state_specs() const
{
    if (!add_positional_encoding) return {};
    return {{{sequence_length, embedding_dimension}, Type::FP32}};
}

void EmbeddingLookupOp::link_parameters(span<const TensorView> views)
{
    if (views.empty()) return;
    weights = views[0];
}

void EmbeddingLookupOp::link_gradients(span<const TensorView> views)
{
    if (views.empty()) return;
    weight_gradient = views[0];
}

void EmbeddingLookupOp::link_states(span<const TensorView> views)
{
    if (views.empty()) return;
    const bool needs_init = positional_encoding.data == nullptr;
    positional_encoding = views[0];
    if (needs_init) init_positional_encoding();
}

void EmbeddingLookupOp::set_parameters_random()
{
    if (weights.empty()) return;
    MatrixMap weights_matrix = weights.as_matrix();
    set_random_normal(weights_matrix, 0.0f, 1.0f);
    weights_matrix.row(0).setZero();
}

void EmbeddingLookupOp::set_parameters_glorot()
{
    if (weights.empty()) return;
    const float limit = glorot_limit(vocabulary_size, embedding_dimension);
    // Eigen's setRandom() bypasses the project's RNG (uses std::rand
    // internally), which breaks determinism — see [random_utilities.cpp]
    // for why everything else routes through the global generator.
    set_random_uniform(weights.as_vector(), -limit, limit);
    weights.as_matrix().row(0).setZero();
}

void EmbeddingLookupOp::init_positional_encoding()
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

void EmbeddingLookupOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    const TensorView& indices = get_input(fp, layer);
    TensorView& output        = get_output(fp, layer);

    embedding_lookup_forward(indices, weights, positional_encoding, output,
                             sequence_length, embedding_dimension, vocabulary_size,
                             scale_embedding, add_positional_encoding);
}

void EmbeddingLookupOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    const TensorView& indices      = get_input(fp, layer);
    const TensorView& output_delta = get_output_delta(bp, layer);

    embedding_lookup_backward(indices, output_delta, weight_gradient,
                              embedding_dimension, vocabulary_size, scale_embedding);
}

void FlatOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    copy(get_input(fp, layer), get_output(fp, layer));
}

void FlatOp::back_propagate(ForwardPropagation&, BackPropagation& bp, size_t layer) const noexcept
{
    copy(get_output_delta(bp, layer), get_input_delta(bp, layer));
}

void BoundOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    const TensorView& input = get_input(fp, layer);
    TensorView& output      = get_output(fp, layer);

    if (method == Method::NoBounding || !lower.data)
    {
        copy(input, output);
        return;
    }

    bound(input, lower, upper, output);
}

void ScaleOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    const TensorView& input = get_input(fp, layer);
    TensorView& output      = get_output(fp, layer);

    if (!minimums.data)
    {
        copy(input, output);
        return;
    }

    scale(input, minimums, maximums, means, standard_deviations, scalers,
          min_range, max_range, output);
}

void UnscaleOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool /*is_training*/) noexcept
{
    const TensorView& input = get_input(fp, layer);
    TensorView& output      = get_output(fp, layer);

    if (!minimums.data)
    {
        copy(input, output);
        return;
    }

    unscale(input, minimums, maximums, means, standard_deviations, scalers,
            min_range, max_range, output);
}

namespace
{

float yolo_sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

float yolo_iou_xywh(const array<float, 6>& a, const array<float, 6>& b)
{
    const float a_left = a[0] - 0.5f * a[2];
    const float a_top = a[1] - 0.5f * a[3];
    const float a_right = a[0] + 0.5f * a[2];
    const float a_bottom = a[1] + 0.5f * a[3];

    const float b_left = b[0] - 0.5f * b[2];
    const float b_top = b[1] - 0.5f * b[3];
    const float b_right = b[0] + 0.5f * b[2];
    const float b_bottom = b[1] + 0.5f * b[3];

    const float inter_w = max(0.0f, min(a_right, b_right) - max(a_left, b_left));
    const float inter_h = max(0.0f, min(a_bottom, b_bottom) - max(a_top, b_top));
    const float inter = inter_w * inter_h;
    const float area = a[2] * a[3] + b[2] * b[3] - inter;

    return area > 0.0f ? inter / area : 0.0f;
}

}

void DetectionOp::set(const Shape& input_shape, const vector<array<float, 2>>& new_anchors)
{
    if (input_shape.rank != 3)
        throw runtime_error("DetectionOp: input shape must be rank 3.");
    if (new_anchors.empty())
        throw runtime_error("DetectionOp: anchors are empty.");

    grid_size = input_shape[0];
    boxes_per_cell = ssize(new_anchors);
    anchors = new_anchors;

    const Index channels = input_shape[2];
    if (channels % boxes_per_cell != 0)
        throw runtime_error("DetectionOp: channels must be divisible by boxes_per_cell.");

    classes_number = channels / boxes_per_cell - 5;
    if (classes_number <= 0)
        throw runtime_error("DetectionOp: classes_number must be positive.");
}

void DetectionOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool) noexcept
{
    const TensorView& input = get_input(fp, layer);
    TensorView& output = get_output(fp, layer);

    IF_GPU({ throw runtime_error("DetectionOp GPU path is not implemented yet."); });
    apply(input, output);
}

void DetectionOp::apply(const TensorView& input, TensorView& output) const
{
    const Index batch_size = input.shape[0];
    const Index channels = input.shape[3];
    const Index values_per_box = 5 + classes_number;

    const float* src = input.as<float>();
    float* dst = output.as<float>();

    #pragma omp parallel for collapse(3)
    for (Index b = 0; b < batch_size; ++b)
        for (Index row = 0; row < grid_size; ++row)
            for (Index col = 0; col < grid_size; ++col)
            {
                const Index cell = ((b * grid_size + row) * grid_size + col) * channels;

                for (Index box = 0; box < boxes_per_cell; ++box)
                {
                    const Index base = cell + box * values_per_box;

                    dst[base + 0] = yolo_sigmoid(src[base + 0]);
                    dst[base + 1] = yolo_sigmoid(src[base + 1]);
                    dst[base + 2] = exp(src[base + 2]) * anchors[size_t(box)][0];
                    dst[base + 3] = exp(src[base + 3]) * anchors[size_t(box)][1];
                    dst[base + 4] = yolo_sigmoid(src[base + 4]);

                    float max_logit = src[base + 5];
                    for (Index c = 1; c < classes_number; ++c)
                        max_logit = max(max_logit, src[base + 5 + c]);

                    float sum = 0.0f;
                    for (Index c = 0; c < classes_number; ++c)
                    {
                        const float e = exp(src[base + 5 + c] - max_logit);
                        dst[base + 5 + c] = e;
                        sum += e;
                    }

                    const float inv_sum = 1.0f / (sum + EPSILON);
                    for (Index c = 0; c < classes_number; ++c)
                        dst[base + 5 + c] *= inv_sum;
                }
            }
}

void DetectionOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    const TensorView& output = get_output(fp, layer);
    const TensorView& output_delta = get_output_delta(bp, layer);
    TensorView& input_delta = get_input_delta(bp, layer);

    if (input_delta.empty()) return;

    IF_GPU({ throw runtime_error("DetectionOp GPU path is not implemented yet."); });
    apply_delta(output, output_delta, input_delta);
}

void DetectionOp::apply_delta(const TensorView& output,
                              const TensorView& output_delta,
                              TensorView& input_delta) const
{
    const Index batch_size = output.shape[0];
    const Index channels = output.shape[3];
    const Index values_per_box = 5 + classes_number;

    const float* out = output.as<float>();
    const float* delta = output_delta.as<float>();
    float* in_delta = input_delta.as<float>();

    #pragma omp parallel for collapse(3)
    for (Index b = 0; b < batch_size; ++b)
        for (Index row = 0; row < grid_size; ++row)
            for (Index col = 0; col < grid_size; ++col)
            {
                const Index cell = ((b * grid_size + row) * grid_size + col) * channels;

                for (Index box = 0; box < boxes_per_cell; ++box)
                {
                    const Index base = cell + box * values_per_box;

                    in_delta[base + 0] = delta[base + 0] * out[base + 0] * (1.0f - out[base + 0]);
                    in_delta[base + 1] = delta[base + 1] * out[base + 1] * (1.0f - out[base + 1]);
                    in_delta[base + 2] = delta[base + 2] * out[base + 2];
                    in_delta[base + 3] = delta[base + 3] * out[base + 3];
                    in_delta[base + 4] = delta[base + 4] * out[base + 4] * (1.0f - out[base + 4]);

                    float dot = 0.0f;
                    for (Index c = 0; c < classes_number; ++c)
                        dot += delta[base + 5 + c] * out[base + 5 + c];

                    for (Index c = 0; c < classes_number; ++c)
                        in_delta[base + 5 + c] = out[base + 5 + c] * (delta[base + 5 + c] - dot);
                }
            }
}

void NonMaxSuppressionOp::set(const Shape& input_shape,
                              Index new_boxes_per_cell,
                              float new_confidence_threshold,
                              float new_iou_threshold)
{
    if (input_shape.rank != 3)
        throw runtime_error("NonMaxSuppressionOp: input shape must be rank 3.");
    if (new_boxes_per_cell <= 0)
        throw runtime_error("NonMaxSuppressionOp: boxes_per_cell must be positive.");

    grid_size = input_shape[0];
    boxes_per_cell = new_boxes_per_cell;
    confidence_threshold = new_confidence_threshold;
    iou_threshold = new_iou_threshold;

    const Index channels = input_shape[2];
    if (channels % boxes_per_cell != 0)
        throw runtime_error("NonMaxSuppressionOp: channels must be divisible by boxes_per_cell.");

    classes_number = channels / boxes_per_cell - 5;
    if (classes_number <= 0)
        throw runtime_error("NonMaxSuppressionOp: classes_number must be positive.");
}

void NonMaxSuppressionOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool) noexcept
{
    const TensorView& input = get_input(fp, layer);
    TensorView& output = get_output(fp, layer);

    IF_GPU({ throw runtime_error("NonMaxSuppressionOp GPU path is not implemented yet."); });
    apply(input, output);
}

void NonMaxSuppressionOp::apply(const TensorView& input, TensorView& output) const
{
    const Index batch_size = input.shape[0];
    const Index channels = input.shape[3];
    const Index values_per_box = 5 + classes_number;
    const Index max_boxes = grid_size * grid_size * boxes_per_cell;

    const float* src = input.as<float>();
    float* dst = output.as<float>();
    fill_n(dst, output.size(), 0.0f);

    for (Index b = 0; b < batch_size; ++b)
    {
        vector<array<float, 6>> candidates;
        candidates.reserve(size_t(max_boxes));

        for (Index row = 0; row < grid_size; ++row)
            for (Index col = 0; col < grid_size; ++col)
            {
                const Index cell = ((b * grid_size + row) * grid_size + col) * channels;

                for (Index box = 0; box < boxes_per_cell; ++box)
                {
                    const Index base = cell + box * values_per_box;

                    Index best_class = 0;
                    float best_probability = src[base + 5];
                    for (Index c = 1; c < classes_number; ++c)
                        if (src[base + 5 + c] > best_probability)
                        {
                            best_probability = src[base + 5 + c];
                            best_class = c;
                        }

                    const float score = src[base + 4] * best_probability;
                    if (score < confidence_threshold)
                        continue;

                    candidates.push_back({
                        (float(col) + src[base + 0]) / float(grid_size),
                        (float(row) + src[base + 1]) / float(grid_size),
                        src[base + 2],
                        src[base + 3],
                        score,
                        float(best_class)
                    });
                }
            }

        ranges::sort(candidates, greater<>{}, [](const array<float, 6>& box) { return box[4]; });

        Index kept_count = 0;
        for (const array<float, 6>& candidate : candidates)
        {
            bool suppressed = false;
            for (Index k = 0; k < kept_count; ++k)
            {
                const float* kept = dst + (b * max_boxes + k) * 6;
                const array<float, 6> kept_box{kept[0], kept[1], kept[2], kept[3], kept[4], kept[5]};

                if (Index(kept_box[5]) == Index(candidate[5])
                &&  yolo_iou_xywh(candidate, kept_box) > iou_threshold)
                {
                    suppressed = true;
                    break;
                }
            }

            if (suppressed)
                continue;

            float* out = dst + (b * max_boxes + kept_count) * 6;
            std::copy(candidate.begin(), candidate.end(), out);
            if (++kept_count == max_boxes)
                break;
        }
    }
}

namespace
{

float lstm_activate(ActivationOp::Function function, float x)
{
    using enum ActivationOp::Function;
    switch (function)
    {
        case Identity: return x;
        case Sigmoid:  return 1.0f / (1.0f + exp(-x));
        case Tanh:     return tanh(x);
        case ReLU:     return max(0.0f, x);
        case Softmax:  return x;
    }
    return x;
}

float lstm_derivative_from_output(ActivationOp::Function function, float y)
{
    using enum ActivationOp::Function;
    switch (function)
    {
        case Identity: return 1.0f;
        case Sigmoid:  return y * (1.0f - y);
        case Tanh:     return 1.0f - y * y;
        case ReLU:     return y > 0.0f ? 1.0f : 0.0f;
        case Softmax:  return 1.0f;
    }
    return 1.0f;
}

void zero_if_linked(const TensorView& view)
{
    if (view.data) const_cast<TensorView&>(view).setZero();
}

}

void LongShortTermMemoryOp::set(Index new_input_features,
                                Index new_output_features,
                                Index new_time_steps,
                                ActivationOp::Function new_activation_function,
                                ActivationOp::Function new_recurrent_activation_function)
{
    input_features = new_input_features;
    output_features = new_output_features;
    time_steps = new_time_steps;
    activation_function = new_activation_function;
    recurrent_activation_function = new_recurrent_activation_function;
}

vector<TensorSpec> LongShortTermMemoryOp::parameter_specs() const
{
    if (output_features == 0)
        return {};

    const Shape bias_shape{output_features};
    const Shape input_weight_shape{input_features, output_features};
    const Shape recurrent_weight_shape{output_features, output_features};

    return {
        {bias_shape, Type::FP32},             // forget bias
        {bias_shape, Type::FP32},             // input bias
        {bias_shape, Type::FP32},             // candidate bias
        {bias_shape, Type::FP32},             // output bias
        {input_weight_shape, Type::FP32},     // forget input weights
        {input_weight_shape, Type::FP32},     // input gate input weights
        {input_weight_shape, Type::FP32},     // candidate input weights
        {input_weight_shape, Type::FP32},     // output gate input weights
        {recurrent_weight_shape, Type::FP32}, // forget recurrent weights
        {recurrent_weight_shape, Type::FP32}, // input recurrent weights
        {recurrent_weight_shape, Type::FP32}, // candidate recurrent weights
        {recurrent_weight_shape, Type::FP32}, // output recurrent weights
    };
}

void LongShortTermMemoryOp::link_parameters(span<const TensorView> views)
{
    if (views.size() < 12) return;

    forget_bias = views[0];
    input_bias = views[1];
    candidate_bias = views[2];
    output_bias = views[3];

    forget_weights = views[4];
    input_weights = views[5];
    candidate_weights = views[6];
    output_weights = views[7];

    forget_recurrent_weights = views[8];
    input_recurrent_weights = views[9];
    candidate_recurrent_weights = views[10];
    output_recurrent_weights = views[11];
}

void LongShortTermMemoryOp::link_gradients(span<const TensorView> views)
{
    if (views.size() < 12) return;

    forget_bias_gradient = views[0];
    input_bias_gradient = views[1];
    candidate_bias_gradient = views[2];
    output_bias_gradient = views[3];

    forget_weight_gradient = views[4];
    input_weight_gradient = views[5];
    candidate_weight_gradient = views[6];
    output_weight_gradient = views[7];

    forget_recurrent_weight_gradient = views[8];
    input_recurrent_weight_gradient = views[9];
    candidate_recurrent_weight_gradient = views[10];
    output_recurrent_weight_gradient = views[11];
}

void LongShortTermMemoryOp::set_parameters_random()
{
    set_parameters_glorot();
}

void LongShortTermMemoryOp::set_parameters_glorot()
{
    zero_if_linked(forget_bias);
    zero_if_linked(input_bias);
    zero_if_linked(candidate_bias);
    zero_if_linked(output_bias);

    if (forget_weights.data)
    {
        const float limit = glorot_limit(input_features, output_features);
        set_random_uniform(forget_weights.as_vector(), -limit, limit);
        set_random_uniform(input_weights.as_vector(), -limit, limit);
        set_random_uniform(candidate_weights.as_vector(), -limit, limit);
        set_random_uniform(output_weights.as_vector(), -limit, limit);
    }

    if (forget_recurrent_weights.data)
    {
        const float limit = glorot_limit(output_features, output_features);
        set_random_uniform(forget_recurrent_weights.as_vector(), -limit, limit);
        set_random_uniform(input_recurrent_weights.as_vector(), -limit, limit);
        set_random_uniform(candidate_recurrent_weights.as_vector(), -limit, limit);
        set_random_uniform(output_recurrent_weights.as_vector(), -limit, limit);
    }
}

void LongShortTermMemoryOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool) noexcept
{
    auto& views = fp.views[layer];

    TensorView& input = views[InputSlot][0];
    TensorView& output = views[OutputSlot][0];
    TensorView& forget_gate = views[ForgetGateSlot][0];
    TensorView& input_gate = views[InputGateSlot][0];
    TensorView& candidate_gate = views[CandidateGateSlot][0];
    TensorView& output_gate = views[OutputGateSlot][0];
    TensorView& cell_state = views[CellStateSlot][0];
    TensorView& hidden_state = views[HiddenStateSlot][0];
    TensorView& cell_activation = views[CellActivationSlot][0];

    IF_GPU({ throw runtime_error("LongShortTermMemoryOp GPU path is not implemented yet."); });

    apply(input, output, forget_gate, input_gate, candidate_gate, output_gate,
          cell_state, hidden_state, cell_activation);
}

void LongShortTermMemoryOp::apply(const TensorView& input,
                                      TensorView& output,
                                      TensorView& forget_gate,
                                      TensorView& input_gate,
                                      TensorView& candidate_gate,
                                      TensorView& output_gate,
                                      TensorView& cell_state,
                                      TensorView& hidden_state,
                                      TensorView& cell_activation) const
{
    if (!input.data || output_features == 0 || time_steps == 0) return;

    const Index batch_size = input.shape[0];
    const Index F = input_features;
    const Index H = output_features;
    const Index T = time_steps;

    const float* x = input.as<float>();
    float* y = output.as<float>();
    float* f_gate = forget_gate.as<float>();
    float* i_gate = input_gate.as<float>();
    float* g_gate = candidate_gate.as<float>();
    float* o_gate = output_gate.as<float>();
    float* cells = cell_state.as<float>();
    float* hidden = hidden_state.as<float>();
    float* cell_act = cell_activation.as<float>();

    const float* bf = forget_bias.as<float>();
    const float* bi = input_bias.as<float>();
    const float* bg = candidate_bias.as<float>();
    const float* bo = output_bias.as<float>();

    const float* Wf = forget_weights.as<float>();
    const float* Wi = input_weights.as<float>();
    const float* Wg = candidate_weights.as<float>();
    const float* Wo = output_weights.as<float>();

    const float* Uf = forget_recurrent_weights.as<float>();
    const float* Ui = input_recurrent_weights.as<float>();
    const float* Ug = candidate_recurrent_weights.as<float>();
    const float* Uo = output_recurrent_weights.as<float>();

    #pragma omp parallel for
    for (Index b = 0; b < batch_size; ++b)
    {
        for (Index t = 0; t < T; ++t)
        {
            const float* xt = x + (b * T + t) * F;
            const float* h_prev = t > 0 ? hidden + (b * T + t - 1) * H : nullptr;
            const float* c_prev = t > 0 ? cells + (b * T + t - 1) * H : nullptr;
            const Index step = (b * T + t) * H;

            for (Index h = 0; h < H; ++h)
            {
                float zf = bf[h];
                float zi = bi[h];
                float zg = bg[h];
                float zo = bo[h];

                for (Index k = 0; k < F; ++k)
                {
                    const float xk = xt[k];
                    zf += xk * Wf[k * H + h];
                    zi += xk * Wi[k * H + h];
                    zg += xk * Wg[k * H + h];
                    zo += xk * Wo[k * H + h];
                }

                if (h_prev)
                {
                    for (Index j = 0; j < H; ++j)
                    {
                        const float hp = h_prev[j];
                        zf += hp * Uf[j * H + h];
                        zi += hp * Ui[j * H + h];
                        zg += hp * Ug[j * H + h];
                        zo += hp * Uo[j * H + h];
                    }
                }

                const float f = lstm_activate(recurrent_activation_function, zf);
                const float i = lstm_activate(recurrent_activation_function, zi);
                const float g = lstm_activate(activation_function, zg);
                const float o = lstm_activate(recurrent_activation_function, zo);
                const float c = f * (c_prev ? c_prev[h] : 0.0f) + i * g;
                const float a = lstm_activate(activation_function, c);
                const float h_value = o * a;

                f_gate[step + h] = f;
                i_gate[step + h] = i;
                g_gate[step + h] = g;
                o_gate[step + h] = o;
                cells[step + h] = c;
                cell_act[step + h] = a;
                hidden[step + h] = h_value;
            }
        }

        copy_n(hidden + (b * T + T - 1) * H, H, y + b * H);
    }
}

void LongShortTermMemoryOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const noexcept
{
    auto& deltas = bp.delta_views[layer];
    if (deltas.size() <= OutputDeltaScratchSlot) return;

    const auto& views = fp.views[layer];

    TensorView& input_delta = deltas[InputDeltaSlot];
    TensorView& hidden_delta = deltas[HiddenDeltaScratchSlot];
    TensorView& cell_delta = deltas[CellDeltaScratchSlot];
    TensorView& forget_delta = deltas[ForgetDeltaScratchSlot];
    TensorView& input_gate_delta = deltas[InputDeltaScratchSlot];
    TensorView& candidate_delta = deltas[CandidateDeltaScratchSlot];
    TensorView& output_gate_delta = deltas[OutputDeltaScratchSlot];

    const TensorView& input = views[InputSlot][0];
    const TensorView& output_delta = get_output_delta(bp, layer);
    const TensorView& forget_gate = views[ForgetGateSlot][0];
    const TensorView& input_gate = views[InputGateSlot][0];
    const TensorView& candidate_gate = views[CandidateGateSlot][0];
    const TensorView& output_gate = views[OutputGateSlot][0];
    const TensorView& cell_state = views[CellStateSlot][0];
    const TensorView& hidden_state = views[HiddenStateSlot][0];
    const TensorView& cell_activation = views[CellActivationSlot][0];

    IF_GPU({ throw runtime_error("LongShortTermMemoryOp GPU path is not implemented yet."); });

    apply_delta(input, output_delta, input_delta, hidden_delta, cell_delta,
                forget_delta, input_gate_delta, candidate_delta, output_gate_delta,
                forget_gate, input_gate, candidate_gate, output_gate, cell_state,
                hidden_state, cell_activation);
}

void LongShortTermMemoryOp::apply_delta(const TensorView& input,
                                        const TensorView& output_delta,
                                        TensorView& input_delta,
                                        TensorView& hidden_delta_scratch,
                                        TensorView& cell_delta_scratch,
                                        TensorView& forget_delta_scratch,
                                        TensorView& input_delta_scratch,
                                        TensorView& candidate_delta_scratch,
                                        TensorView& output_delta_scratch,
                                        const TensorView& forget_gate,
                                            const TensorView& input_gate,
                                            const TensorView& candidate_gate,
                                            const TensorView& output_gate,
                                            const TensorView& cell_state,
                                            const TensorView& hidden_state,
                                            const TensorView& cell_activation) const
{
    if (!input.data || !output_delta.data || output_features == 0 || time_steps == 0) return;

    zero_if_linked(forget_bias_gradient);
    zero_if_linked(input_bias_gradient);
    zero_if_linked(candidate_bias_gradient);
    zero_if_linked(output_bias_gradient);
    zero_if_linked(forget_weight_gradient);
    zero_if_linked(input_weight_gradient);
    zero_if_linked(candidate_weight_gradient);
    zero_if_linked(output_weight_gradient);
    zero_if_linked(forget_recurrent_weight_gradient);
    zero_if_linked(input_recurrent_weight_gradient);
    zero_if_linked(candidate_recurrent_weight_gradient);
    zero_if_linked(output_recurrent_weight_gradient);

    const Index batch_size = input.shape[0];
    const Index F = input_features;
    const Index H = output_features;
    const Index T = time_steps;

    const float* x = input.as<float>();
    const float* out_delta = output_delta.as<float>();
    float* in_delta = input_delta.as<float>();

    const float* f_gate = forget_gate.as<float>();
    const float* i_gate = input_gate.as<float>();
    const float* g_gate = candidate_gate.as<float>();
    const float* o_gate = output_gate.as<float>();
    const float* cells = cell_state.as<float>();
    const float* hidden = hidden_state.as<float>();
    const float* cell_act = cell_activation.as<float>();

    const float* Wf = forget_weights.as<float>();
    const float* Wi = input_weights.as<float>();
    const float* Wg = candidate_weights.as<float>();
    const float* Wo = output_weights.as<float>();

    const float* Uf = forget_recurrent_weights.as<float>();
    const float* Ui = input_recurrent_weights.as<float>();
    const float* Ug = candidate_recurrent_weights.as<float>();
    const float* Uo = output_recurrent_weights.as<float>();

    float* gbf = forget_bias_gradient.as<float>();
    float* gbi = input_bias_gradient.as<float>();
    float* gbg = candidate_bias_gradient.as<float>();
    float* gbo = output_bias_gradient.as<float>();

    float* gWf = forget_weight_gradient.as<float>();
    float* gWi = input_weight_gradient.as<float>();
    float* gWg = candidate_weight_gradient.as<float>();
    float* gWo = output_weight_gradient.as<float>();

    float* gUf = forget_recurrent_weight_gradient.as<float>();
    float* gUi = input_recurrent_weight_gradient.as<float>();
    float* gUg = candidate_recurrent_weight_gradient.as<float>();
    float* gUo = output_recurrent_weight_gradient.as<float>();

    float* dh_next_all = hidden_delta_scratch.as<float>();
    float* dc_next_all = cell_delta_scratch.as<float>();
    float* df_all = forget_delta_scratch.as<float>();
    float* di_all = input_delta_scratch.as<float>();
    float* dg_all = candidate_delta_scratch.as<float>();
    float* do_all = output_delta_scratch.as<float>();

    for (Index b = 0; b < batch_size; ++b)
    {
        float* dh_next = dh_next_all + b * H;
        float* dc_next = dc_next_all + b * H;
        float* df = df_all + b * H;
        float* di = di_all + b * H;
        float* dg = dg_all + b * H;
        float* do_gate = do_all + b * H;

        copy_n(out_delta + b * H, H, dh_next);
        fill_n(dc_next, H, 0.0f);

        for (Index t = T; t-- > 0;)
        {
            const Index step = (b * T + t) * H;
            const float* xt = x + (b * T + t) * F;
            const float* h_prev = t > 0 ? hidden + (b * T + t - 1) * H : nullptr;
            const float* c_prev = t > 0 ? cells + (b * T + t - 1) * H : nullptr;

            for (Index h = 0; h < H; ++h)
            {
                const float f = f_gate[step + h];
                const float i = i_gate[step + h];
                const float g = g_gate[step + h];
                const float o = o_gate[step + h];
                const float a = cell_act[step + h];

                const float dc = dh_next[h] * o * lstm_derivative_from_output(activation_function, a) + dc_next[h];

                do_gate[h] = dh_next[h] * a * lstm_derivative_from_output(recurrent_activation_function, o);
                df[h] = dc * (c_prev ? c_prev[h] : 0.0f) * lstm_derivative_from_output(recurrent_activation_function, f);
                di[h] = dc * g * lstm_derivative_from_output(recurrent_activation_function, i);
                dg[h] = dc * i * lstm_derivative_from_output(activation_function, g);
                dc_next[h] = dc * f;

                gbf[h] += df[h];
                gbi[h] += di[h];
                gbg[h] += dg[h];
                gbo[h] += do_gate[h];
            }

            for (Index k = 0; k < F; ++k)
            {
                float dx = 0.0f;
                const float xk = xt[k];

                for (Index h = 0; h < H; ++h)
                {
                    const Index wh = k * H + h;
                    gWf[wh] += xk * df[h];
                    gWi[wh] += xk * di[h];
                    gWg[wh] += xk * dg[h];
                    gWo[wh] += xk * do_gate[h];

                    dx += df[h] * Wf[wh]
                        + di[h] * Wi[wh]
                        + dg[h] * Wg[wh]
                        + do_gate[h] * Wo[wh];
                }

                in_delta[(b * T + t) * F + k] = dx;
            }

            for (Index j = 0; j < H; ++j)
            {
                float dh_prev = 0.0f;
                const float hp = h_prev ? h_prev[j] : 0.0f;

                for (Index h = 0; h < H; ++h)
                {
                    const Index uh = j * H + h;

                    if (h_prev)
                    {
                        gUf[uh] += hp * df[h];
                        gUi[uh] += hp * di[h];
                        gUg[uh] += hp * dg[h];
                        gUo[uh] += hp * do_gate[h];
                    }

                    dh_prev += df[h] * Uf[uh]
                             + di[h] * Ui[uh]
                             + dg[h] * Ug[uh]
                             + do_gate[h] * Uo[uh];
                }

                dh_next[j] = dh_prev;
            }
        }
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
