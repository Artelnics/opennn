//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   N O R M   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifdef OPENNN_HAS_CUDA
#include <cudnn_frontend.h>
#endif

#include "batch_norm_operator.h"
#include "device_backend.h"
#include "kernel.cuh"
#include "json.h"
#include "tensor_operations.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"
#include "cudnn_frontend_utilities.h"

namespace opennn
{

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
    if (views.size() < 2) return;
    gamma = views[0];
    beta  = views[1];
    invalidate_inference_cache();
}

void BatchNormalizationOperator::link_gradients(span<const TensorView> views)
{
    if (views.size() < 2) return;
    gamma_gradient = views[0];
    beta_gradient  = views[1];
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

    add_json_field(w, "Momentum", to_string(momentum));

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
                    / (running_variance.as_vector().array().max(0.0f) + EPSILON).sqrt();
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
    const TensorView& residual = fuse_add ? forward_propagation.input_views[layer][1] : empty;

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
        ? back_propagation.backward_slots[layer][residual_delta_slot] : empty;

    if (!delta.is_cuda())
    {
        // The activation has already been undone in place, so the residual
        // branch gets its delta before the in-place transform destroys it.
        if (!residual_delta.empty()) copy(delta, residual_delta);
        apply_delta_cpu(input, mean, inverse_variance, delta);
        return;
    }

    const TensorView& residual = fuse_add ? forward_propagation.input_views[layer][1] : empty;

    apply_delta_gpu(input, output, residual, mean, inverse_variance, delta, residual_delta);
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

    inverse_variances.array() = 1.0f / (inverse_variances.array().max(0.0f) + EPSILON).sqrt();
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

BatchNormalizationOperator::BatchNormalizationOperator() = default;
BatchNormalizationOperator::~BatchNormalizationOperator() = default;

#ifdef OPENNN_HAS_CUDA

struct BatchNormalizationOperator::BatchNormalizationGraphCache
{
    struct Entry
    {
        shared_ptr<cudnn_frontend::graph::Graph> fwd, bwd;

        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_X, fwd_Scale, fwd_Bias,
            fwd_PrevMean, fwd_PrevVar, fwd_Eps, fwd_Mom, fwd_Residual,
            fwd_Y, fwd_Mean, fwd_InvVar, fwd_NextMean, fwd_NextVar;
            
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_DY, bwd_X, bwd_Scale, bwd_Bias,
            bwd_Mean, bwd_InvVar, bwd_Residual, bwd_DPre, bwd_DX, bwd_DScale, bwd_DBias;

        int64_t fwd_workspace_bytes = 0;
        int64_t bwd_workspace_bytes = 0;
        
        bool bwd_forked = false;
        bool fwd_autotune = false;
        bool bwd_autotune = false;
    };

    unordered_map<Index, Entry> entries;
    bool disabled = false;
};

namespace cudnn_frontend
{
using namespace ::cudnn_frontend;

shared_ptr<graph::Tensor_attributes>
per_channel_tensor(graph::Graph& graph, const char* name, int64_t channels)
{
    // BN scale/bias/stats stay FP32 in every mode (cuDNN mixed-precision BN
    // requires FP32 per-channel tensors even with BF16 io).
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
    entry.fwd_Eps      = scalar_tensor(*graph, "EPSILON");
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

    entry.fwd_autotune = finalize(*graph, entry.fwd_workspace_bytes, "batchnorm forward", autotune_enabled());
    entry.fwd = graph;
}

void build_bn_backward(BatchNormalizationOperator::BatchNormalizationGraphCache::Entry& entry,
                       int64_t batch, int64_t channels, int64_t spatial,
                       bool fuse_relu, Type dtype, bool fork_residual = false)
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
        entry.bwd_Bias = per_channel_tensor(*graph, "BIAS", channels);
        auto pre_activation = graph->batchnorm_inference(entry.bwd_X, entry.bwd_Mean, entry.bwd_InvVar,
                                                         entry.bwd_Scale, entry.bwd_Bias,
                                                         graph::Batchnorm_inference_attributes());

        if (fork_residual)
        {
            entry.bwd_Residual = nhwc_tensor(*graph, "RESIDUAL", batch, channels, spatial, 1);
            pre_activation = graph->pointwise(pre_activation, entry.bwd_Residual,
                                              graph::Pointwise_attributes()
                                              .set_mode(PointwiseMode_t::ADD));
        }

        delta_in = graph->pointwise(entry.bwd_DY, pre_activation,
                                    graph::Pointwise_attributes()
                                    .set_mode(PointwiseMode_t::RELU_BWD));

        if (fork_residual)
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

    entry.bwd_autotune = finalize(*graph, entry.bwd_workspace_bytes, "batchnorm backward", autotune_enabled());
    entry.bwd = graph;
}

}

void BatchNormalizationOperator::apply_inference_gpu(const TensorView& input, TensorView& output,
                                    const TensorView& residual)
{
    // One fused NHWC pass (BN + optional residual + optional ReLU). The legacy
    // cudnnBatchNormalizationForwardInference call had no NHWC kernel for
    // several ResNet shapes and silently inserted NCHW layout converts, which
    // cost ~40% of a ResNet-50 inference forward.
    input.dispatch([&](auto tag)
    {
        using T = decltype(tag);
        batchnorm_inference_cuda<T>(input.size(), features,
                                    input.as<T>(),
                                    fuse_add ? residual.as<T>() : nullptr,
                                    gamma.as<float>(), beta.as<float>(),
                                    running_mean.as<float>(), running_variance.as<float>(),
                                    EPSILON, fuse_relu,
                                    output.as<T>());
    });
}

void BatchNormalizationOperator::apply_training_gpu(const TensorView& input,
                                   TensorView& mean, TensorView& inverse_variance,
                                   TensorView& output,
                                   const TensorView& residual)
{
    PROFILE_SCOPE("op:bn_fwd");

    // cuDNN's BF16 batchnorm-backward has no engine, so BN runs entirely in FP32
    // (the PyTorch/TF AMP convention): cast X/residual up, run FP32, cast Y back.
    const bool bf16 = input.is_bf16();
    const Type graph_dtype = bf16 ? Type::FP32 : input.type;

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

        float epsilon_value = EPSILON;
        float momentum_value = momentum;

        void* x_ptr        = input.data;
        void* residual_ptr = fuse_add ? residual.data : nullptr;
        void* y_ptr        = output.data;
        float* y_fp32      = nullptr;
        if (bf16)
        {
            const Index n = input.size();
            float* scratch = ensure_bf16_to_fp32_workspace((fuse_add ? 3 : 2) * n);
            y_fp32 = scratch + n;
            cast_bf16_to_fp32(n, static_cast<const bfloat16*>(input.data), scratch);
            x_ptr = scratch;
            y_ptr = y_fp32;
            if (fuse_add)
            {
                float* r_fp32 = scratch + 2 * n;
                cast_bf16_to_fp32(n, static_cast<const bfloat16*>(residual.data), r_fp32);
                residual_ptr = r_fp32;
            }
        }

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

        cudnn_frontend::autotune_with_scratch(entry.fwd_autotune, *entry.fwd, tensors, entry.fwd_workspace_bytes);

        cudnn_frontend::execute_graph(*entry.fwd, tensors, cudnn_frontend::shared_workspace(entry.fwd_workspace_bytes),
                                "batchnorm forward execute",
                                cudnn_frontend::graph_timing_enabled()
                                ? format("bn_fwd c{} r{}", features, input.size() / features)
                                : string());

        if (bf16)
            cast_fp32_to_bf16(output.size(), y_fp32, static_cast<bfloat16*>(output.data),
                              Backend::get_compute_stream());
    });

    if (!ran)
    {
        // Legacy cuDNN path for GPUs older than SM 8.0 (e.g. RTX 2080 = SM 7.5).
        CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
            Backend::get_cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one, &zero,
            input.get_descriptor(),  input.data,
            output.get_descriptor(), output.data,
            gamma.get_descriptor(),  gamma.data, beta.data,
            double(momentum),
            running_mean.data, running_variance.data,
            EPSILON,
            mean.data,
            inverse_variance.data));
        if (fuse_add)  add(output, residual, output);
        if (fuse_relu) activation_forward(output, ActivationFunction::ReLU);
    }
}

void BatchNormalizationOperator::apply_delta_gpu(const TensorView& input,
                                const TensorView& output,
                                const TensorView& residual,
                                const TensorView& mean,
                                const TensorView& inverse_variance,
                                TensorView& delta,
                                TensorView& residual_delta) const
{
    PROFILE_SCOPE("op:bn_bwd");

    // Residual blocks try the forked graph (BN_infer + add -> dReLU, whose
    // result is both a real output for the skip branch and the DBN input);
    // every unforked path undoes the activation and copies the delta to the
    // skip branch by hand before the plain batch-norm backward.
    // cuDNN has no BF16 batchnorm-backward engine, so in BF16 the backward runs
    // FP32: X/DY are cast up at the boundary and DX cast back. Forced-unforked
    // there to keep the FP32 scratch to a single 2N buffer.
    const bool bf16 = input.is_bf16();
    const Type graph_dtype = bf16 ? Type::FP32 : input.type;
    const bool want_fork = fuse_add && fuse_relu && !residual_delta.empty() && !bf16;

    throw_if(!input.is_fp32() && !bf16,
             "BatchNormalizationOperator: GPU backward requires FP32 or BF16.");

    const bool ran = cudnn_frontend::bn_frontend_enabled()
        && cudnn_frontend::run_frontend(bn_graph_cache, "BatchNormalizationOperator", [&](BatchNormalizationGraphCache& cache)
    {
        auto& entry = cache.entries[input.shape[0]];
        if (!entry.bwd)
        {
            const int64_t batch   = input.shape[0];
            const int64_t spatial = int64_t(input.size()) / (batch * features);

            if (want_fork)
                try
                {
                    cudnn_frontend::build_bn_backward(entry, batch, features, spatial, true, graph_dtype, true);
                    entry.bwd_forked = true;
                }
                catch (const exception&)
                {
                    entry.bwd_Bias     = nullptr;
                    entry.bwd_Residual = nullptr;
                    entry.bwd_DPre     = nullptr;
                }

            if (!entry.bwd)
                cudnn_frontend::build_bn_backward(entry, batch, features, spatial, fuse_relu && !fuse_add, graph_dtype);
        }

        if (fuse_add && !entry.bwd_forked)
        {
            if (fuse_relu) activation_backward(output, delta, ActivationFunction::ReLU);
            if (!residual_delta.empty()) copy(delta, residual_delta);
        }

        // BF16: cast X and DY (DX shares the buffer, in place) into FP32 scratch.
        void* x_ptr    = input.data;
        void* dy_ptr   = delta.data;
        float* dx_fp32 = nullptr;
        if (bf16)
        {
            const Index n = delta.size();
            float* scratch = ensure_bf16_to_fp32_workspace(2 * n);
            dx_fp32 = scratch + n;
            cast_bf16_to_fp32(n, static_cast<const bfloat16*>(input.data), scratch);
            cast_bf16_to_fp32(n, static_cast<const bfloat16*>(delta.data), dx_fp32);
            x_ptr  = scratch;
            dy_ptr = dx_fp32;
        }

        unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensors;
        tensors[entry.bwd_DY]     = dy_ptr;
        if (entry.bwd_Bias)     tensors[entry.bwd_Bias]     = beta.data;
        if (entry.bwd_forked)
        {
            tensors[entry.bwd_Residual] = residual.data;
            tensors[entry.bwd_DPre]     = residual_delta.data;
        }
        tensors[entry.bwd_X]      = x_ptr;
        tensors[entry.bwd_Scale]  = gamma.data;
        tensors[entry.bwd_Mean]   = mean.data;
        tensors[entry.bwd_InvVar] = inverse_variance.data;
        tensors[entry.bwd_DX]     = dy_ptr;
        tensors[entry.bwd_DScale] = gamma_gradient.data;
        tensors[entry.bwd_DBias]  = beta_gradient.data;

        cudnn_frontend::autotune_with_scratch(entry.bwd_autotune, *entry.bwd, tensors, entry.bwd_workspace_bytes);

        cudnn_frontend::execute_graph(*entry.bwd, tensors, cudnn_frontend::shared_workspace(entry.bwd_workspace_bytes),
                                "batchnorm backward execute",
                                cudnn_frontend::graph_timing_enabled()
                                ? format("bn_bwd c{} r{}", features, input.size() / features)
                                : string());

        if (bf16)
            cast_fp32_to_bf16(delta.size(), dx_fp32, static_cast<bfloat16*>(delta.data),
                              Backend::get_compute_stream());
    });

    if (!ran)
    {
        // Legacy cuDNN backward path for GPUs older than SM 8.0.
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
            EPSILON,
            mean.data,
            inverse_variance.data));
    }
}

#else

void BatchNormalizationOperator::apply_inference_gpu(const TensorView&, TensorView&, const TensorView&)                 { throw runtime_error("apply_inference_gpu requires CUDA."); }
void BatchNormalizationOperator::apply_training_gpu (const TensorView&, TensorView&, TensorView&, TensorView&,
                                    const TensorView&)                                                   { throw runtime_error("apply_training_gpu requires CUDA."); }
void BatchNormalizationOperator::apply_delta_gpu    (const TensorView&, const TensorView&, const TensorView&,
                                    const TensorView&, const TensorView&, TensorView&,
                                    TensorView&) const                                                  { throw runtime_error("apply_delta_gpu requires CUDA."); }

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
