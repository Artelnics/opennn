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
                      Backend::get_compute_stream());
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
    if (gamma.data)            gamma.as_vector().setOnes();
    if (beta.data)             beta.as_vector().setZero();
    if (running_mean.data)     running_mean.as_vector().setZero();
    if (running_variance.data) running_variance.as_vector().setOnes();
    invalidate_inference_cache();
}

void BatchNormalizationOperator::to_JSON(JsonWriter& w) const
{
    if (!active()) return;

    add_json_field(w, "Momentum", momentum);

    if (running_mean.data)
        add_json_field(w, "RunningMeans", vector_to_string(running_mean.as_vector()));
    if (running_variance.data)
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

void BatchNormalizationOperator::update_inference_cache()
{
    if (!inference_cache_dirty || !gamma.data || !beta.data || !running_mean.data || !running_variance.data) return;

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

    if (input.is_cuda()) apply_training_gpu(input, mean, inv_variance, output, residual);
    else
    {
        apply_training_cpu(input, mean, inv_variance, output);
        if (fuse_add) add(output, residual, output);
    }

    invalidate_inference_cache();
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

    apply_delta_gpu(input, output, mean, inverse_variance, delta, residual_delta);
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
        shared_ptr<cudnn_frontend::graph::Graph> fwd, bwd;

        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_X, fwd_Scale, fwd_Bias,
            fwd_PrevMean, fwd_PrevVar, fwd_Eps, fwd_Mom, fwd_Residual,
            fwd_Y, fwd_Mean, fwd_InvVar, fwd_NextMean, fwd_NextVar;

        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_DY, bwd_Y, bwd_X, bwd_Scale,
            bwd_Bias, bwd_Mean, bwd_InvVar, bwd_DPre, bwd_DX, bwd_DScale, bwd_DBias;

        int64_t fwd_workspace_bytes = 0;
        int64_t bwd_workspace_bytes = 0;

        bool bwd_forked = false;
        bool bwd_native_dtype = false;
        bool bwd_fused_relu = false;
        bool bwd_own_kernel = false;   // batchnorm_backward_fused_cuda, no cuDNN graph
        bool fwd_autotune = false;
        bool bwd_autotune = false;
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

    entry.fwd_autotune = finalize(
        *graph, entry.fwd_workspace_bytes, "batchnorm forward",
        device::conv_autotune_enabled());
    entry.fwd = graph;
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

    entry.bwd_autotune = finalize(
        *graph, entry.bwd_workspace_bytes, "batchnorm backward",
        device::conv_autotune_enabled());
    entry.bwd = graph;
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
                                   const TensorView& residual)
{
    PROFILE_SCOPE("op:bn_fwd");

    const bool bf16 = input.is_bf16();
    const Type graph_dtype = input.type;

    throw_if(!input.is_fp32() && !bf16,
             "BatchNormalizationOperator: GPU training forward requires FP32 or BF16.");

    const bool ran = cudnn_frontend::bn_frontend_enabled()
        && cudnn_frontend::run_frontend(bn_graph_cache, "BatchNormalizationOperator", [&](BatchNormalizationGraphCache& cache)
    {
        auto& entry = cache.entries[input.shape[0]];
        if (!entry.fwd)
        {
            const int64_t batch    = input.shape[0];
            const int64_t spatial  = int64_t(input.size()) / (batch * features);
            cudnn_frontend::build_bn_forward(entry, batch, features, spatial, fuse_relu, fuse_add, graph_dtype);
        }

        float epsilon_value = BN_EPSILON;
        float momentum_value = momentum;

        void* x_ptr        = input.data;
        void* residual_ptr = fuse_add ? residual.data : nullptr;
        void* y_ptr        = output.data;

        unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensors;
        if (fuse_add) tensors[entry.fwd_Residual] = residual_ptr;
        tensors[entry.fwd_X]        = x_ptr;
        tensors[entry.fwd_Scale]    = gamma.data;
        tensors[entry.fwd_Bias]     = beta.data;
        tensors[entry.fwd_PrevMean] = running_mean.data;
        tensors[entry.fwd_PrevVar]  = running_variance.data;
        tensors[entry.fwd_Eps]      = &epsilon_value;
        tensors[entry.fwd_Mom]      = &momentum_value;
        tensors[entry.fwd_Y]        = y_ptr;
        tensors[entry.fwd_Mean]     = mean.data;
        tensors[entry.fwd_InvVar]   = inverse_variance.data;
        tensors[entry.fwd_NextMean] = running_mean.data;
        tensors[entry.fwd_NextVar]  = running_variance.data;

        cudnn_frontend::autotune_with_scratch(entry.fwd_autotune, *entry.fwd, tensors,
                                              entry.fwd_workspace_bytes, "BatchNormOperator fwd");

        cudnn_frontend::execute_graph(*entry.fwd, tensors, cudnn_frontend::shared_workspace(entry.fwd_workspace_bytes),
                                "batchnorm forward execute",
                                cudnn_frontend::graph_timing_enabled()
                                ? format("bn_fwd c{} r{}", features, input.size() / features)
                                : string());
    });

    if (!ran)
    {

        CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
            Backend::get_cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one, &zero,
            input.get_descriptor(),  input.data,
            output.get_descriptor(), output.data,
            gamma.get_descriptor(),  gamma.data, beta.data,
            double(momentum),
            running_mean.data, running_variance.data,
            BN_EPSILON,
            mean.data,
            inverse_variance.data));
        if (fuse_add)  add(output, residual, output);
        if (fuse_relu) activation_forward(output, ActivationFunction::ReLU);
    }
}

void BatchNormalizationOperator::apply_delta_gpu(const TensorView& input,
                                const TensorView& output,
                                const TensorView& mean,
                                const TensorView& inverse_variance,
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
        auto& entry = cache.entries[input.shape[0]];
        if (!entry.bwd && !entry.bwd_own_kernel)
        {
            const int64_t batch   = input.shape[0];
            const int64_t spatial = int64_t(input.size()) / (batch * features);

            // cuDNN's BF16 fused-backward coverage is partial. Auto tries the
            // exact fused graph once, then uses the library kernel to retain the
            // dReLU and residual fork without FP32 staging. The other rungs remain
            // available for direct comparison in the GPU gradient test.
            struct Attempt { Type dtype; bool fuse_relu; bool fork; };

            const device::BatchNormBackwardRung rung = device::batch_norm_backward_rung();

            vector<Attempt> attempts;
            switch (rung)
            {
            case device::BatchNormBackwardRung::StagedFp32:
                attempts.push_back({Type::FP32, fuse_relu && !fuse_add, false});
                break;
            case device::BatchNormBackwardRung::PlainNative:
                attempts.push_back({input.type, false, false});
                break;
            case device::BatchNormBackwardRung::Auto:
                attempts.push_back({input.type, fuse_relu, fork_capable});
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
                    entry.bwd_forked       = attempt.fork;
                    entry.bwd_fused_relu   = attempt.fuse_relu;
                    entry.bwd_native_dtype = attempt.dtype == input.type;
                    break;
                }
                catch (...)
                {
                    // build_bn_backward assigns entry.bwd last, so a throw leaves the
                    // tensor handles it already overwrote pointing at a dead graph.
                    entry.bwd_Y    = nullptr;
                    entry.bwd_DPre = nullptr;
                    last_failure = current_exception();
                }
            }

            if (!entry.bwd)
            {
                if (rung == device::BatchNormBackwardRung::StagedFp32
                    || rung == device::BatchNormBackwardRung::PlainNative)
                    rethrow_exception(last_failure);
                entry.bwd_own_kernel = true;
            }

            // Once per shape: what this backward runs on when it is not the fully
            // fused cuDNN graph. Anything else costs extra full-tensor passes
            // (a standalone dReLU, a residual-delta copy, three FP32 staging
            // casts), and this line is the only signal of it.
            const bool fully_fused = entry.bwd && entry.bwd_native_dtype
                && entry.bwd_fused_relu == fuse_relu && (entry.bwd_forked || !fuse_add);
            if (!fully_fused)
                cerr << "BatchNormalizationOperator backward c" << features
                     << " r" << input.size() / features << " batch " << batch << ": "
                     << (entry.bwd_own_kernel ? "own fused kernel (no fused cuDNN engine)"
                         : !entry.bwd_native_dtype ? "FP32-staged cuDNN graph"
                         : entry.bwd_fused_relu ? "cuDNN graph, fused ReLU, no residual fork"
                         : "plain cuDNN graph; ReLU/copy run separately")
                     << ".\n";
        }

        if (entry.bwd_own_kernel)
        {
            const Index rows = input.size() / features;
            float* partials = ensure_bf16_to_fp32_workspace(
                2 * batchnorm_backward_partial_rows(rows) * features);
            void* dpre = fuse_add && !residual_delta.empty() ? residual_delta.data : nullptr;
            const void* y = fuse_relu ? output.data : nullptr;

            // For a ReLU output that is BN(x) itself the reduce pass rebuilds
            // x_hat from Y and skips X (six passes, not seven). Kept to FP32: in
            // BF16 the (y - beta) / gamma reconstruction amplifies y's rounding by
            // 1/gamma.
            const bool xhat_from_y = fuse_relu && !fuse_add && !bf16;

            if (bf16)
                batchnorm_backward_fused_cuda<bfloat16>(
                    rows, features,
                    input.as<bfloat16>(), delta.as<bfloat16>(), static_cast<const bfloat16*>(y),
                    gamma.as<float>(), beta.as<float>(), mean.as<float>(), inverse_variance.as<float>(),
                    xhat_from_y,
                    static_cast<bfloat16*>(dpre), gamma_gradient.as<float>(), beta_gradient.as<float>(),
                    partials);
            else
                batchnorm_backward_fused_cuda<float>(
                    rows, features,
                    input.as<float>(), delta.as<float>(), static_cast<const float*>(y),
                    gamma.as<float>(), beta.as<float>(), mean.as<float>(), inverse_variance.as<float>(),
                    xhat_from_y,
                    static_cast<float*>(dpre), gamma_gradient.as<float>(), beta_gradient.as<float>(),
                    partials);
            return;
        }

        // Whatever the chosen graph does not fuse runs here instead, ahead of it.
        if (fuse_relu && !entry.bwd_fused_relu)
            activation_backward(output, delta, ActivationFunction::ReLU);

        if (fuse_add && !entry.bwd_forked && !residual_delta.empty())
            copy(delta, residual_delta);

        const bool stage_fp32 = bf16 && !entry.bwd_native_dtype;

        void* x_ptr    = input.data;
        void* dy_ptr   = delta.data;
        span<float> dx_fp32;
        optional<Fp32Staging> staging;
        if (stage_fp32)
        {

            staging.emplace(delta.size(), 2);
            x_ptr   = staging->read(input.data).data();
            dx_fp32 = staging->read(delta.data);
            dy_ptr  = dx_fp32.data();
        }

        unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensors;
        tensors[entry.bwd_DY]     = dy_ptr;
        if (entry.bwd_Bias)     tensors[entry.bwd_Bias]     = beta.data;
        if (entry.bwd_forked)
        {
            tensors[entry.bwd_Y]    = output.data;
            tensors[entry.bwd_DPre] = residual_delta.data;
        }
        tensors[entry.bwd_X]      = x_ptr;
        tensors[entry.bwd_Scale]  = gamma.data;
        tensors[entry.bwd_Mean]   = mean.data;
        tensors[entry.bwd_InvVar] = inverse_variance.data;
        tensors[entry.bwd_DX]     = dy_ptr;
        tensors[entry.bwd_DScale] = gamma_gradient.data;
        tensors[entry.bwd_DBias]  = beta_gradient.data;

        cudnn_frontend::autotune_with_scratch(entry.bwd_autotune, *entry.bwd, tensors,
                                              entry.bwd_workspace_bytes, "BatchNormOperator bwd");

        cudnn_frontend::execute_graph(*entry.bwd, tensors, cudnn_frontend::shared_workspace(entry.bwd_workspace_bytes),
                                "batchnorm backward execute",
                                cudnn_frontend::graph_timing_enabled()
                                ? format("bn_bwd c{} r{}", features, input.size() / features)
                                : string());

        if (stage_fp32) store_as_bfloat16(*staging, dx_fp32, delta.data);
    });

    if (!ran)
    {

        if (fuse_relu) activation_backward(output, delta, ActivationFunction::ReLU);
        if (fuse_add && !residual_delta.empty()) copy(delta, residual_delta);
        CHECK_CUDNN(cudnnBatchNormalizationBackward(
            Backend::get_cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one, &zero, &one, &zero,
            input.get_descriptor(),  input.data,
            delta.get_descriptor(),  delta.data,
            delta.get_descriptor(),  delta.data,
            gamma.get_descriptor(),  gamma.data,
            gamma_gradient.data,
            beta_gradient.data,
            BN_EPSILON,
            mean.data,
            inverse_variance.data));
    }
}

#else

void BatchNormalizationOperator::apply_inference_gpu(const TensorView&, TensorView&, const TensorView&)                 OPENNN_CUDA_STUB_BODY(apply_inference_gpu)
void BatchNormalizationOperator::apply_training_gpu (const TensorView&, TensorView&, TensorView&, TensorView&,
                                    const TensorView&)                                                   OPENNN_CUDA_STUB_BODY(apply_training_gpu)
void BatchNormalizationOperator::apply_delta_gpu    (const TensorView&, const TensorView&,
                                    const TensorView&, const TensorView&, TensorView&,
                                    TensorView&) const                                                  OPENNN_CUDA_STUB_BODY(apply_delta_gpu)

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
