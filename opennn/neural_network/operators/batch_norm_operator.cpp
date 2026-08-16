//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   N O R M   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/operators/batch_norm_operator.h"

#ifdef OPENNN_HAS_CUDA
#include <cudnn_frontend.h>
#endif

#include "opennn/core/cuda/cudnn_frontend_utilities.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/core/cuda/kernel_cast.cuh"
#include "opennn/core/cuda/kernel_normalization.cuh"
#endif
#include "opennn/core/device_backend.h"
#include "opennn/core/json.h"
#include "opennn/core/profiler.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/neural_network/forward_propagation.h"

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

namespace
{

// Fallback staging for the backward pass: when cuDNN has no engine config for a
// BF16-IO batchnorm *backward* graph, X/DY go through an FP32 workspace. The
// backward prefers a native BF16 graph and only stages when that finds no plan.
// The forward always runs with native BF16 IO (stats and scale/bias stay FP32
// either way).
struct Fp32Staging
{
    Fp32Staging(Index count, Index slices)
        : elements(count),
          workspace(ensure_bf16_to_fp32_workspace(slices * count), size_t(slices * count))
    {
    }

    span<float> read(const void* bfloat16_source)
    {
        const span<float> slice = next();
        cast_bf16_to_fp32(elements, static_cast<const bfloat16*>(bfloat16_source), slice.data());
        return slice;
    }

    Index elements = 0;

private:

    span<float> next()
    {
        const size_t offset = size_t(used++) * size_t(elements);
        return workspace.subspan(offset, size_t(elements));
    }

    span<float> workspace;
    Index used = 0;
};

void store_as_bfloat16(const Fp32Staging& staging,
                       const span<const float> slice,
                       void* bfloat16_target)
{
    cast_fp32_to_bf16(staging.elements, slice.data(), static_cast<bfloat16*>(bfloat16_target),
                      device::get_compute_stream());
}

}

#endif

// One epsilon for both paths. Batch norm is only meaningful if inference
// reproduces what training computed, and that identity holds only when the two
// use the same epsilon; a larger one at inference silently rescales every
// channel whose variance is near or below it.
static constexpr float BN_EPSILON = 1e-5f;

void BatchNormalizationOperator::set(Index new_features, float new_momentum)
{
    throw_if(new_momentum < 0.0f || new_momentum >= 1.0f,
             "BatchNorm momentum must be in [0, 1).");
    features = new_features;
    momentum = new_momentum;
}

vector<TensorSpec> BatchNormalizationOperator::parameter_specs() const
{
    if (!active()) return {};
    return vector<TensorSpec>(2, {Shape{features}, Type::FP32});
}

vector<TensorSpec> BatchNormalizationOperator::state_specs() const
{
    return parameter_specs();
}

void BatchNormalizationOperator::link_parameters(span<const TensorView> views)
{
    if (link_views(views, {&gamma, &beta}))
        invalidate_inference_cache();
}

void BatchNormalizationOperator::link_gradients(span<const TensorView> views)
{
    link_views(views, {&gamma_gradient, &beta_gradient});
}

void BatchNormalizationOperator::link_states(span<const TensorView> views)
{
    if (views.size() < 2) return;
    running_mean     = views[0];
    running_variance = views[1];
    invalidate_inference_cache();
}

void BatchNormalizationOperator::init_defaults()
{
    if (gamma.get_data())            gamma.as_vector().setOnes();
    if (beta.get_data())             beta.as_vector().setZero();
    initialize_states();
}

void BatchNormalizationOperator::initialize_states()
{
    if (running_mean.get_data())     running_mean.as_vector().setZero();
    if (running_variance.get_data()) running_variance.as_vector().setOnes();
    invalidate_inference_cache();
}

void BatchNormalizationOperator::to_JSON(JsonWriter& w) const
{
    if (!active()) return;

    add_json_field(w, "Momentum", momentum);

    if (running_mean.get_data())
        add_json_field(w, "RunningMeans", vector_to_string(running_mean.as_vector()));
    if (running_variance.get_data())
        add_json_field(w, "RunningVariances", vector_to_string(running_variance.as_vector()));
}

void BatchNormalizationOperator::from_JSON(const Json* parent)
{
    if (parent && parent->has("Momentum"))
        momentum = read_json_float(parent, "Momentum");
}

void BatchNormalizationOperator::load_state_from_JSON(const Json* parent)
{
    if (!parent) return;

    VectorR tmp;
    if (parent->has("RunningMeans"))
    {
        string_to_vector(read_json_string(parent, "RunningMeans"), tmp);
        if (running_mean.get_data() && tmp.size() == running_mean.size())
            running_mean.as_vector() = tmp;
    }
    if (parent->has("RunningVariances"))
    {
        string_to_vector(read_json_string(parent, "RunningVariances"), tmp);
        if (running_variance.get_data() && tmp.size() == running_variance.size())
            running_variance.as_vector() = tmp;
    }

    invalidate_inference_cache();
}

void BatchNormalizationOperator::update_inference_cache()
{
    if (!inference_cache_dirty || !gamma.get_data() || !beta.get_data() || !running_mean.get_data() || !running_variance.get_data()) return;

    inference_scale = gamma.as_vector().array()
                    / (running_variance.as_vector().array().max(0.0f) + BN_EPSILON).sqrt();
    inference_shift = beta.as_vector().array()
                    - inference_scale.array() * running_mean.as_vector().array();

    inference_cache_dirty = false;
}

void BatchNormalizationOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training)
{
    if (!active()) return;

    static TensorView empty;

    const TensorView& input    = get_input(forward_propagation, layer);
    TensorView& output         = get_output(forward_propagation, layer);
    const TensorView& residual = fuse_add ? forward_propagation.inputs[layer][1] : empty;

    if (!is_training)
    {
        if (input.is_cuda()) apply_inference_gpu(input, output, residual);
        else
        {
            apply_inference_cpu(input, output);
            if (fuse_add) add(output, residual, output);
        }
        return;
    }

    TensorView& mean         = get_output(forward_propagation, layer, 1);
    TensorView& inv_variance = get_output(forward_propagation, layer, 2);

    if (input.is_cuda()) apply_training_gpu(input, mean, inv_variance, output, residual,
                                            relu_mask(forward_propagation, layer));
    else
    {
        apply_training_cpu(input, mean, inv_variance, output);
        if (fuse_add) add(output, residual, output);
    }

    invalidate_inference_cache();
}

TensorView& BatchNormalizationOperator::relu_mask(ForwardPropagation& forward_propagation, size_t layer) const noexcept
{
    static TensorView empty;
    return output_slots.size() > 3 ? get_output(forward_propagation, layer, 3) : empty;
}

bool BatchNormalizationOperator::own_forward_kernel(const TensorView& mask) const noexcept
{
    switch (device::rung<device::BatchNormForwardRung>())
    {
    case device::BatchNormForwardRung::CudnnGraph: return false;
    case device::BatchNormForwardRung::OwnKernel:  return features % 8 == 0;
    case device::BatchNormForwardRung::Auto:       break;
    }
    // Where the mask pays: the backward reads it in place of Y.
    return fuse_relu && !mask.empty();
}

void BatchNormalizationOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    if (!active()) return;

    static TensorView empty;

    const TensorView& input            = get_input(forward_propagation, layer);
    const TensorView& output           = get_output(forward_propagation, layer);
    const TensorView& mean             = get_output(forward_propagation, layer, 1);
    const TensorView& inverse_variance = get_output(forward_propagation, layer, 2);
    TensorView& delta                  = get_output_delta(back_propagation, layer);
    TensorView& residual_delta         = residual_delta_slot
        ? back_propagation.slots[layer][residual_delta_slot] : empty;

    if (!delta.is_cuda())
    {

        if (!residual_delta.empty()) copy(delta, residual_delta);
        apply_delta_cpu(input, mean, inverse_variance, delta);
        return;
    }

    apply_delta_gpu(input, output, mean, inverse_variance, relu_mask(forward_propagation, layer),
                    delta, residual_delta);
}

void BatchNormalizationOperator::apply_inference_cpu(const TensorView& input, TensorView& output)
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

void BatchNormalizationOperator::apply_training_cpu(const TensorView& input,
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

    inverse_variances.array() = 1.0f / (inverse_variances.array().max(0.0f) + BN_EPSILON).sqrt();
    const VectorR scale = inverse_variances.array() * gamma.as_vector().array();
    const VectorMap betas = beta.as_vector();

    const auto scale_t = scale.transpose().array();
    const auto betas_t = betas.transpose().array();

    #pragma omp parallel for
    for (Index i = 0; i < output_matrix.rows(); ++i)
        output_matrix.row(i).array() = output_matrix.row(i).array() * scale_t + betas_t;
}

void BatchNormalizationOperator::apply_delta_cpu(const TensorView& input,
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

    const VectorR delta_scale =
        (gammas.array() * inverse_variances.array() * inv_N).matrix();

    const auto delta_scale_t       = delta_scale.transpose().array();
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

BatchNormalizationOperator::BatchNormalizationOperator() = default;
BatchNormalizationOperator::~BatchNormalizationOperator() = default;

#ifndef OPENNN_HAS_CUDA

struct BatchNormalizationOperator::BatchNormalizationGraphCache {};
#endif

#ifdef OPENNN_HAS_CUDA

struct BatchNormalizationOperator::BatchNormalizationGraphCache
{
    struct Entry
    {
        cudnn_frontend::GraphSlot fwd, bwd;

        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_X, fwd_Scale, fwd_Bias,
            fwd_PrevMean, fwd_PrevVar, fwd_Eps, fwd_Mom, fwd_Residual,
            fwd_Y, fwd_Mean, fwd_InvVar, fwd_NextMean, fwd_NextVar;

        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_DY, bwd_Y, bwd_X, bwd_Scale,
            bwd_Bias, bwd_Mean, bwd_InvVar, bwd_DPre, bwd_DX, bwd_DScale, bwd_DBias;

        // The backward attempt that won (see apply_delta_gpu); nullopt until
        // the first backward, and, with no cuDNN engine, own_kernel.
        struct BackwardChoice { Type dtype; bool fuse_relu; bool fork; bool own_kernel; };
        optional<BackwardChoice> bwd_choice;
    };

    unordered_map<Index, Entry> entries;
    bool disabled = false;
};

namespace cudnn_frontend
{
using namespace ::cudnn_frontend;

namespace
{

shared_ptr<graph::Tensor_attributes>
per_channel_tensor(graph::Graph& graph, const char* name, int64_t channels)
{

    return graph.tensor(graph::Tensor_attributes()
                        .set_name(name)
                        .set_data_type(DataType_t::FLOAT)
                        .set_dim({1, channels, 1, 1})
                        .set_stride({channels, 1, channels, channels}));
}

shared_ptr<graph::Tensor_attributes>
scalar_tensor(graph::Graph& graph, const char* name)
{
    return graph.tensor(graph::Tensor_attributes()
                        .set_name(name)
                        .set_data_type(DataType_t::FLOAT)
                        .set_dim({1, 1, 1, 1})
                        .set_stride({1, 1, 1, 1})
                        .set_is_pass_by_value(true));
}

void set_per_channel_output(shared_ptr<graph::Tensor_attributes>& tensor, int64_t channels)
{
    tensor->set_output(true)
           .set_data_type(DataType_t::FLOAT)
           .set_dim({1, channels, 1, 1})
           .set_stride({channels, 1, channels, channels});
}

void build_bn_forward(BatchNormalizationOperator::BatchNormalizationGraphCache::Entry& entry,
                      int64_t batch, int64_t channels, int64_t spatial,
                      bool fuse_relu, bool fuse_add, Type dtype)
{
    auto graph = new_graph(dtype);

    entry.fwd_X        = nhwc_tensor(*graph, "X", batch, channels, spatial, 1);
    entry.fwd_Scale    = per_channel_tensor(*graph, "SCALE", channels);
    entry.fwd_Bias     = per_channel_tensor(*graph, "BIAS", channels);
    entry.fwd_PrevMean = per_channel_tensor(*graph, "PREV_MEAN", channels);
    entry.fwd_PrevVar  = per_channel_tensor(*graph, "PREV_VAR", channels);
    entry.fwd_Eps      = scalar_tensor(*graph, "BN_EPSILON");
    entry.fwd_Mom      = scalar_tensor(*graph, "MOMENTUM");

    auto attributes = graph::Batchnorm_attributes()
                      .set_epsilon(entry.fwd_Eps)
                      .set_previous_running_stats(entry.fwd_PrevMean, entry.fwd_PrevVar, entry.fwd_Mom);

    auto [Y, mean, inv_variance, next_mean, next_var] =
        graph->batchnorm(entry.fwd_X, entry.fwd_Scale, entry.fwd_Bias, attributes);

    if (fuse_add)
    {
        entry.fwd_Residual = nhwc_tensor(*graph, "RESIDUAL", batch, channels, spatial, 1);
        Y = graph->pointwise(Y, entry.fwd_Residual,
                             graph::Pointwise_attributes()
                             .set_mode(PointwiseMode_t::ADD));
    }

    if (fuse_relu)
        Y = graph->pointwise(Y, graph::Pointwise_attributes()
                                .set_mode(PointwiseMode_t::RELU_FWD));

    set_nhwc_output(Y, batch, channels, spatial, 1);
    set_per_channel_output(mean, channels);
    set_per_channel_output(inv_variance, channels);
    set_per_channel_output(next_mean, channels);
    set_per_channel_output(next_var, channels);

    entry.fwd_Y        = Y;
    entry.fwd_Mean     = mean;
    entry.fwd_InvVar   = inv_variance;
    entry.fwd_NextMean = next_mean;
    entry.fwd_NextVar  = next_var;

    entry.fwd.autotune_pending = finalize(
        *graph, entry.fwd.workspace_bytes, "batchnorm forward",
        device::conv_autotune_enabled());
    entry.fwd.graph = graph;
}

void build_bn_backward(BatchNormalizationOperator::BatchNormalizationGraphCache::Entry& entry,
                       int64_t batch, int64_t channels, int64_t spatial,
                       bool fuse_relu, Type dtype, bool fork_residual_delta = false)
{
    auto graph = new_graph(dtype);

    entry.bwd_DY     = nhwc_tensor(*graph, "DY", batch, channels, spatial, 1);
    entry.bwd_X      = nhwc_tensor(*graph, "X", batch, channels, spatial, 1);
    entry.bwd_Scale  = per_channel_tensor(*graph, "SCALE", channels);
    entry.bwd_Mean   = per_channel_tensor(*graph, "MEAN", channels);
    entry.bwd_InvVar = per_channel_tensor(*graph, "INV_VARIANCE", channels);

    auto delta_in = entry.bwd_DY;

    if (fuse_relu)
    {
        shared_ptr<graph::Tensor_attributes> relu_reference;
        if (fork_residual_delta)
        {

            entry.bwd_Y = nhwc_tensor(*graph, "Y", batch, channels, spatial, 1);
            relu_reference = entry.bwd_Y;
        }
        else
        {
            entry.bwd_Bias = per_channel_tensor(*graph, "BIAS", channels);
            relu_reference =
                graph->batchnorm_inference(entry.bwd_X, entry.bwd_Mean, entry.bwd_InvVar,
                                           entry.bwd_Scale, entry.bwd_Bias,
                                           graph::Batchnorm_inference_attributes());
        }

        delta_in = graph->pointwise(entry.bwd_DY, relu_reference,
                                    graph::Pointwise_attributes()
                                    .set_mode(PointwiseMode_t::RELU_BWD));

        if (fork_residual_delta)
        {
            set_nhwc_output(delta_in, batch, channels, spatial, 1);
            entry.bwd_DPre = delta_in;
        }
    }

    auto attributes = graph::Batchnorm_backward_attributes()
                      .set_saved_mean_and_inv_variance(entry.bwd_Mean, entry.bwd_InvVar);

    auto [DX, dscale, dbias] = graph->batchnorm_backward(delta_in, entry.bwd_X, entry.bwd_Scale, attributes);

    set_nhwc_output(DX, batch, channels, spatial, 1);
    set_per_channel_output(dscale, channels);
    set_per_channel_output(dbias, channels);

    entry.bwd_DX     = DX;
    entry.bwd_DScale = dscale;
    entry.bwd_DBias  = dbias;

    entry.bwd.autotune_pending = finalize(
        *graph, entry.bwd.workspace_bytes, "batchnorm backward",
        device::conv_autotune_enabled());
    entry.bwd.graph = graph;
}

}

}

void BatchNormalizationOperator::apply_inference_gpu(const TensorView& input, TensorView& output,
                                    const TensorView& residual)
{
    PROFILE_SCOPE("op:bn_infer_fwd");

    input.dispatch([&]<typename T>()
    {
        batchnorm_inference_cuda<T>(input.size(), features,
                                    input.as<T>(),
                                    fuse_add ? residual.as<T>() : nullptr,
                                    gamma.as<float>(), beta.as<float>(),
                                    running_mean.as<float>(), running_variance.as<float>(),
                                    BN_EPSILON, fuse_relu,
                                    output.as<T>());
    });
}

void BatchNormalizationOperator::apply_training_gpu(const TensorView& input,
                                   TensorView& mean, TensorView& inverse_variance,
                                   TensorView& output,
                                   const TensorView& residual,
                                   TensorView& mask)
{
    PROFILE_SCOPE("op:bn_fwd");

    const bool bf16 = input.is_bf16();
    const Type graph_dtype = input.get_type();

    throw_if(!input.is_fp32() && !bf16,
             "BatchNormalizationOperator: GPU training forward requires FP32 or BF16.");

    if (own_forward_kernel(mask))
    {
        const Index rows = input.size() / features;
        float* partials = ensure_bf16_to_fp32_workspace(
            (2 * batchnorm_partial_rows(rows) + 2) * features);
        uint8_t* mask_bits = fuse_relu && !mask.empty() ? mask.as<uint8_t>() : nullptr;

        input.dispatch([&]<typename T>()
        {
            batchnorm_forward_fused_cuda<T>(
                rows, features,
                input.as<T>(), fuse_add ? residual.as<T>() : nullptr,
                gamma.as<float>(), beta.as<float>(), BN_EPSILON, momentum,
                output.as<T>(), mean.as<float>(), inverse_variance.as<float>(),
                running_mean.as<float>(), running_variance.as<float>(),
                fuse_relu, mask_bits, partials);
        });
        return;
    }

    const bool ran = cudnn_frontend::bn_frontend_enabled()
        && cudnn_frontend::run_frontend(bn_graph_cache, "BatchNormalizationOperator", [&](BatchNormalizationGraphCache& cache)
    {
        auto& entry = cache.entries[input.get_shape()[0]];
        if (!entry.fwd.graph)
        {
            const int64_t batch    = input.get_shape()[0];
            const int64_t spatial  = int64_t(input.size()) / (batch * features);
            cudnn_frontend::build_bn_forward(entry, batch, features, spatial, fuse_relu, fuse_add, graph_dtype);
        }

        float epsilon_value = BN_EPSILON;
        float momentum_value = momentum;

        void* x_ptr        = input.get_data();
        void* residual_ptr = fuse_add ? residual.get_data() : nullptr;
        void* y_ptr        = output.get_data();

        unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensors;
        if (fuse_add) tensors[entry.fwd_Residual] = residual_ptr;
        tensors[entry.fwd_X]        = x_ptr;
        tensors[entry.fwd_Scale]    = gamma.get_data();
        tensors[entry.fwd_Bias]     = beta.get_data();
        tensors[entry.fwd_PrevMean] = running_mean.get_data();
        tensors[entry.fwd_PrevVar]  = running_variance.get_data();
        tensors[entry.fwd_Eps]      = &epsilon_value;
        tensors[entry.fwd_Mom]      = &momentum_value;
        tensors[entry.fwd_Y]        = y_ptr;
        tensors[entry.fwd_Mean]     = mean.get_data();
        tensors[entry.fwd_InvVar]   = inverse_variance.get_data();
        tensors[entry.fwd_NextMean] = running_mean.get_data();
        tensors[entry.fwd_NextVar]  = running_variance.get_data();

        cudnn_frontend::run_slot(entry.fwd, tensors, "BatchNormOperator fwd",
                                 cudnn_frontend::graph_timing_enabled()
                                 ? format("bn_fwd c{} r{}", features, input.size() / features) : string(),
                                 true);
    });

    if (!ran)
    {

        CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
            Backend::get_cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one, &zero,
            input.get_descriptor(),  input.get_data(),
            output.get_descriptor(), output.get_data(),
            gamma.get_descriptor(),  gamma.get_data(), beta.get_data(),
            double(momentum),
            running_mean.get_data(), running_variance.get_data(),
            BN_EPSILON,
            mean.get_data(),
            inverse_variance.get_data()));
        if (fuse_add)  add(output, residual, output);
        if (fuse_relu) activation_forward(output, ActivationFunction::ReLU);
    }
}

void BatchNormalizationOperator::apply_delta_gpu(const TensorView& input,
                                const TensorView& output,
                                const TensorView& mean,
                                const TensorView& inverse_variance,
                                const TensorView& mask,
                                TensorView& delta,
                                TensorView& residual_delta) const
{
    PROFILE_SCOPE("op:bn_bwd");

    const bool bf16 = input.is_bf16();
    const bool fork_capable =
        fuse_add
        && fuse_relu
        && !residual_delta.empty();

    throw_if(!input.is_fp32() && !bf16,
             "BatchNormalizationOperator: GPU backward requires FP32 or BF16.");

    const bool ran = cudnn_frontend::bn_frontend_enabled()
        && cudnn_frontend::run_frontend(bn_graph_cache, "BatchNormalizationOperator", [&](BatchNormalizationGraphCache& cache)
    {
        auto& entry = cache.entries[input.get_shape()[0]];
        if (!entry.bwd_choice)
        {
            const int64_t batch   = input.get_shape()[0];
            const int64_t spatial = int64_t(input.size()) / (batch * features);

            // cuDNN's BF16 fused-backward coverage is partial. Auto tries the
            // exact fused graph once, then uses the library kernel to retain the
            // dReLU and residual fork without FP32 staging. The other rungs remain
            // available for direct comparison in the GPU gradient test.
            using Attempt = BatchNormalizationGraphCache::Entry::BackwardChoice;

            const device::BatchNormBackwardRung rung = device::rung<device::BatchNormBackwardRung>();

            vector<Attempt> attempts;
            switch (rung)
            {
            case device::BatchNormBackwardRung::StagedFp32:
                attempts.push_back({Type::FP32, fuse_relu && !fuse_add, false, false});
                break;
            case device::BatchNormBackwardRung::PlainNative:
                attempts.push_back({input.get_type(), false, false, false});
                break;
            case device::BatchNormBackwardRung::Auto:
                attempts.push_back({input.get_type(), fuse_relu, fork_capable, false});
                break;
            case device::BatchNormBackwardRung::OwnKernel:
                break;
            }

            // The first attempt with an engine wins; with none, Auto and OwnKernel
            // take the library kernel and a pinned cuDNN rung reports its failure.
            exception_ptr last_failure;
            for (const Attempt& attempt : attempts)
            {
                try
                {
                    cudnn_frontend::build_bn_backward(entry, batch, features, spatial,
                                                      attempt.fuse_relu, attempt.dtype, attempt.fork);
                    entry.bwd_choice = attempt;
                    break;
                }
                catch (...)
                {
                    // build_bn_backward assigns entry.bwd.graph last, so a throw leaves the
                    // tensor handles it already overwrote pointing at a dead graph.
                    entry.bwd_Y    = nullptr;
                    entry.bwd_DPre = nullptr;
                    last_failure = current_exception();
                }
            }

            if (!entry.bwd_choice)
            {
                if (rung == device::BatchNormBackwardRung::StagedFp32
                    || rung == device::BatchNormBackwardRung::PlainNative)
                    rethrow_exception(last_failure);
                entry.bwd_choice = Attempt{input.get_type(), fuse_relu, fuse_add, true};
            }

            // Once per shape: what this backward runs on when it is not the fully
            // fused cuDNN graph. Anything else costs extra full-tensor passes
            // (a standalone dReLU, a residual-delta copy, three FP32 staging
            // casts), and this line is the only signal of it.
            const Attempt& chosen = *entry.bwd_choice;
            const bool fully_fused = !chosen.own_kernel && chosen.dtype == input.get_type()
                && chosen.fuse_relu == fuse_relu && (chosen.fork || !fuse_add);
            if (!fully_fused)
                cerr << "BatchNormalizationOperator backward c" << features
                     << " r" << input.size() / features << " batch " << batch << ": "
                     << (chosen.own_kernel
                            ? (fuse_relu && own_forward_kernel(mask)
                                   ? "own fused kernel, ReLU from the forward mask (no fused cuDNN engine)"
                                   : "own fused kernel (no fused cuDNN engine)")
                         : chosen.dtype != input.get_type() ? "FP32-staged cuDNN graph"
                         : chosen.fuse_relu ? "cuDNN graph, fused ReLU, no residual fork"
                         : "plain cuDNN graph; ReLU/copy run separately")
                     << ".\n";
        }
        const auto& chosen = *entry.bwd_choice;

        if (chosen.own_kernel)
        {
            const Index rows = input.size() / features;
            float* partials = ensure_bf16_to_fp32_workspace(
                2 * batchnorm_partial_rows(rows) * features);
            // The ReLU gate comes from the packed mask the library's own forward
            // left, when it ran; otherwise from Y.
            const uint8_t* mask_bits = fuse_relu && own_forward_kernel(mask) && !mask.empty()
                ? mask.as<uint8_t>() : nullptr;

            // Without a mask, for a ReLU output that is BN(x) itself the reduce
            // pass rebuilds x_hat from Y and skips X (six passes, not seven).
            // Kept to FP32: in BF16 the (y - beta) / gamma reconstruction
            // amplifies y's rounding by 1/gamma.
            const bool xhat_from_y = fuse_relu && !fuse_add && !bf16;

            input.dispatch([&]<typename T>()
            {
                batchnorm_backward_fused_cuda<T>(
                    rows, features,
                    input.as<T>(), delta.as<T>(),
                    fuse_relu ? output.as<T>() : nullptr, mask_bits,
                    gamma.as<float>(), beta.as<float>(), mean.as<float>(), inverse_variance.as<float>(),
                    xhat_from_y,
                    fuse_add && !residual_delta.empty() ? residual_delta.as<T>() : nullptr,
                    gamma_gradient.as<float>(), beta_gradient.as<float>(),
                    partials);
            });
            return;
        }

        // Whatever the chosen graph does not fuse runs here instead, ahead of it.
        if (fuse_relu && !chosen.fuse_relu)
            activation_backward(output, delta, ActivationFunction::ReLU);

        if (fuse_add && !chosen.fork && !residual_delta.empty())
            copy(delta, residual_delta);

        const bool stage_fp32 = bf16 && chosen.dtype != input.get_type();

        void* x_ptr    = input.get_data();
        void* dy_ptr   = delta.get_data();
        span<float> dx_fp32;
        optional<Fp32Staging> staging;
        if (stage_fp32)
        {

            staging.emplace(delta.size(), 2);
            x_ptr   = staging->read(input.get_data()).data();
            dx_fp32 = staging->read(delta.get_data());
            dy_ptr  = dx_fp32.data();
        }

        unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensors;
        tensors[entry.bwd_DY]     = dy_ptr;
        if (entry.bwd_Bias)     tensors[entry.bwd_Bias]     = beta.get_data();
        if (chosen.fork)
        {
            tensors[entry.bwd_Y]    = output.get_data();
            tensors[entry.bwd_DPre] = residual_delta.get_data();
        }
        tensors[entry.bwd_X]      = x_ptr;
        tensors[entry.bwd_Scale]  = gamma.get_data();
        tensors[entry.bwd_Mean]   = mean.get_data();
        tensors[entry.bwd_InvVar] = inverse_variance.get_data();
        tensors[entry.bwd_DX]     = dy_ptr;
        tensors[entry.bwd_DScale] = gamma_gradient.get_data();
        tensors[entry.bwd_DBias]  = beta_gradient.get_data();

        cudnn_frontend::run_slot(entry.bwd, tensors, "BatchNormOperator bwd",
                                 cudnn_frontend::graph_timing_enabled()
                                 ? format("bn_bwd c{} r{}", features, input.size() / features) : string(),
                                 true);

        if (stage_fp32) store_as_bfloat16(*staging, dx_fp32, delta.get_data());
    });

    if (!ran)
    {

        if (fuse_relu) activation_backward(output, delta, ActivationFunction::ReLU);
        if (fuse_add && !residual_delta.empty()) copy(delta, residual_delta);
        CHECK_CUDNN(cudnnBatchNormalizationBackward(
            Backend::get_cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one, &zero, &one, &zero,
            input.get_descriptor(),  input.get_data(),
            delta.get_descriptor(),  delta.get_data(),
            delta.get_descriptor(),  delta.get_data(),
            gamma.get_descriptor(),  gamma.get_data(),
            gamma_gradient.get_data(),
            beta_gradient.get_data(),
            BN_EPSILON,
            mean.get_data(),
            inverse_variance.get_data()));
    }
}

#else

void BatchNormalizationOperator::apply_inference_gpu(const TensorView&, TensorView&, const TensorView&)                 OPENNN_CUDA_STUB_BODY(apply_inference_gpu)
void BatchNormalizationOperator::apply_training_gpu (const TensorView&, TensorView&, TensorView&, TensorView&,
                                                     const TensorView&, TensorView&)                             OPENNN_CUDA_STUB_BODY(apply_training_gpu)
void BatchNormalizationOperator::apply_delta_gpu    (const TensorView&, const TensorView&,
                                                     const TensorView&, const TensorView&, const TensorView&,
                                                     TensorView&, TensorView&) const                            OPENNN_CUDA_STUB_BODY(apply_delta_gpu)

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
