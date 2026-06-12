//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   N O R M   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifdef HAVE_CUDNN_FRONTEND
#include <cudnn_frontend.h>
#endif

#include "batch_norm_operator.h"
#include "device_backend.h"
#include "json.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"
#include "cudnn_frontend_utilities.h"

namespace opennn
{

void BatchNormOp::set(Index new_features, float new_momentum)
{
    throw_if(new_momentum < 0.0f || new_momentum >= 1.0f,
             "BatchNorm momentum must be in [0, 1).");
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
        momentum = float(read_json_float(parent, "Momentum"));
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

void BatchNormOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training)
{
    if (!active()) return;

    const TensorView& input = get_input(fp, layer);
    TensorView& output      = get_output(fp, layer);

    if (!is_training)
    {
        if (input.is_cuda()) apply_inference_gpu(input, output);
        else                 apply_inference_cpu(input, output);
        return;
    }

    TensorView& mean         = get_output(fp, layer, 1);
    TensorView& inv_variance = get_output(fp, layer, 2);

    if (input.is_cuda()) apply_training_gpu(input, mean, inv_variance, output);
    else                 apply_training_cpu(input, mean, inv_variance, output);

    invalidate_inference_cache();
}

void BatchNormOp::apply_delta(const TensorView& input,
                            const TensorView& output,
                            const TensorView& mean,
                            const TensorView& inverse_variance,
                            TensorView& delta) const
{
    if (delta.is_cuda())
    {
        apply_delta_gpu(input, output, mean, inverse_variance, delta);
        return;
    }
    apply_delta_cpu(input, mean, inverse_variance, delta);
}

void BatchNormOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const
{
    if (!active()) return;

    const TensorView& input            = get_input(fp, layer);
    const TensorView& output           = get_output(fp, layer);
    const TensorView& mean             = get_output(fp, layer, 1);
    const TensorView& inverse_variance = get_output(fp, layer, 2);
    TensorView& delta                  = get_output_delta(bp, layer);

    apply_delta(input, output, mean, inverse_variance, delta);
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

struct BatchNormOp::BnGraphCache
{
#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)
    struct Entry
    {
        shared_ptr<cudnn_frontend::graph::Graph> fwd, bwd;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> fwd_X, fwd_Scale, fwd_Bias,
            fwd_PrevMean, fwd_PrevVar, fwd_Eps, fwd_Mom,
            fwd_Y, fwd_Mean, fwd_InvVar, fwd_NextMean, fwd_NextVar;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> bwd_DY, bwd_X, bwd_Scale, bwd_Bias,
            bwd_Mean, bwd_InvVar, bwd_DX, bwd_DScale, bwd_DBias;
        void* fwd_workspace = nullptr;
        void* bwd_workspace = nullptr;
    };

    map<Index, Entry> entries;

    ~BnGraphCache()
    {
        for (auto& [_, entry] : entries)
        {
            device::deallocate(Device::CUDA, entry.fwd_workspace, 0);
            device::deallocate(Device::CUDA, entry.bwd_workspace, 0);
        }
    }
#endif

    bool disabled = false;
};

BatchNormOp::BatchNormOp() = default;
BatchNormOp::~BatchNormOp() = default;

void BatchNormOp::destroy_cuda()
{
    bn_graph_cache.reset();
}

#if defined(OPENNN_HAS_CUDA) && defined(HAVE_CUDNN_FRONTEND)

namespace cudnn_fe
{

struct BnDims
{
    int64_t batch, channels, spatial;
};

BnDims bn_dims(const TensorView& input, Index features)
{
    const int64_t batch = input.shape[0];
    const int64_t channels = features;
    return {batch, channels, int64_t(input.size()) / (batch * channels)};
}

shared_ptr<cudnn_frontend::graph::Tensor_attributes>
per_channel_tensor(cudnn_frontend::graph::Graph& graph, const char* name, int64_t channels)
{
    return graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim({1, channels, 1, 1})
                        .set_stride({channels, 1, channels, channels}));
}

shared_ptr<cudnn_frontend::graph::Tensor_attributes>
scalar_tensor(cudnn_frontend::graph::Graph& graph, const char* name)
{
    return graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim({1, 1, 1, 1})
                        .set_stride({1, 1, 1, 1})
                        .set_is_pass_by_value(true));
}

void set_per_channel_output(shared_ptr<cudnn_frontend::graph::Tensor_attributes>& tensor, int64_t channels)
{
    tensor->set_output(true)
           .set_dim({1, channels, 1, 1})
           .set_stride({channels, 1, channels, channels});
}

void build_bn_forward(BatchNormOp::BnGraphCache::Entry& entry, const BnDims& d, bool fuse_relu)
{
    auto graph = new_graph();

    entry.fwd_X        = nhwc_tensor(*graph, "X", d.batch, d.channels, d.spatial, 1);
    entry.fwd_Scale    = per_channel_tensor(*graph, "SCALE", d.channels);
    entry.fwd_Bias     = per_channel_tensor(*graph, "BIAS", d.channels);
    entry.fwd_PrevMean = per_channel_tensor(*graph, "PREV_MEAN", d.channels);
    entry.fwd_PrevVar  = per_channel_tensor(*graph, "PREV_VAR", d.channels);
    entry.fwd_Eps      = scalar_tensor(*graph, "EPSILON");
    entry.fwd_Mom      = scalar_tensor(*graph, "MOMENTUM");

    auto attributes = cudnn_frontend::graph::Batchnorm_attributes()
                      .set_epsilon(entry.fwd_Eps)
                      .set_previous_running_stats(entry.fwd_PrevMean, entry.fwd_PrevVar, entry.fwd_Mom);

    auto [Y, mean, inv_variance, next_mean, next_var] =
        graph->batchnorm(entry.fwd_X, entry.fwd_Scale, entry.fwd_Bias, attributes);

    if (fuse_relu)
        Y = graph->pointwise(Y, cudnn_frontend::graph::Pointwise_attributes()
                                .set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD));

    set_nhwc_output(Y, d.batch, d.channels, d.spatial, 1);
    set_per_channel_output(mean, d.channels);
    set_per_channel_output(inv_variance, d.channels);
    set_per_channel_output(next_mean, d.channels);
    set_per_channel_output(next_var, d.channels);

    entry.fwd_Y        = Y;
    entry.fwd_Mean     = mean;
    entry.fwd_InvVar   = inv_variance;
    entry.fwd_NextMean = next_mean;
    entry.fwd_NextVar  = next_var;

    finalize(*graph, entry.fwd_workspace, "batchnorm forward");
    entry.fwd = graph;
}

void build_bn_backward(BatchNormOp::BnGraphCache::Entry& entry, const BnDims& d, bool fuse_relu)
{
    auto graph = new_graph();

    entry.bwd_DY     = nhwc_tensor(*graph, "DY", d.batch, d.channels, d.spatial, 1);
    entry.bwd_X      = nhwc_tensor(*graph, "X", d.batch, d.channels, d.spatial, 1);
    entry.bwd_Scale  = per_channel_tensor(*graph, "SCALE", d.channels);
    entry.bwd_Mean   = per_channel_tensor(*graph, "MEAN", d.channels);
    entry.bwd_InvVar = per_channel_tensor(*graph, "INV_VARIANCE", d.channels);

    auto delta_in = entry.bwd_DY;

    if (fuse_relu)
    {
        // cuDNN only fuses DReLU->DBN when the ReLU input is a virtual tensor,
        // so the pre-ReLU batch-norm output is recomputed in-graph.
        entry.bwd_Bias = per_channel_tensor(*graph, "BIAS", d.channels);
        auto bn_y = graph->batchnorm_inference(entry.bwd_X, entry.bwd_Mean, entry.bwd_InvVar,
                                               entry.bwd_Scale, entry.bwd_Bias,
                                               cudnn_frontend::graph::Batchnorm_inference_attributes());
        delta_in = graph->pointwise(entry.bwd_DY, bn_y,
                                    cudnn_frontend::graph::Pointwise_attributes()
                                    .set_mode(cudnn_frontend::PointwiseMode_t::RELU_BWD));
    }

    auto attributes = cudnn_frontend::graph::Batchnorm_backward_attributes()
                      .set_saved_mean_and_inv_variance(entry.bwd_Mean, entry.bwd_InvVar);

    auto [DX, dscale, dbias] = graph->batchnorm_backward(delta_in, entry.bwd_X, entry.bwd_Scale, attributes);

    set_nhwc_output(DX, d.batch, d.channels, d.spatial, 1);
    set_per_channel_output(dscale, d.channels);
    set_per_channel_output(dbias, d.channels);

    entry.bwd_DX     = DX;
    entry.bwd_DScale = dscale;
    entry.bwd_DBias  = dbias;

    finalize(*graph, entry.bwd_workspace, "batchnorm backward");
    entry.bwd = graph;
}

}  // namespace cudnn_fe

#endif


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

    if (fuse_relu) activation_forward(output, ActivationFunction::ReLU);
}

void BatchNormOp::apply_training_gpu(const TensorView& input,
                                   TensorView& mean, TensorView& inverse_variance,
                                   TensorView& output)
{
#ifdef HAVE_CUDNN_FRONTEND
    if (input.type == Type::FP32 && cudnn_fe::frontend_enabled()
        && cudnn_fe::run_frontend(bn_graph_cache, "BatchNormOp", [&](BnGraphCache& cache)
    {
        auto& entry = cache.entries[input.shape[0]];
        if (!entry.fwd)
            cudnn_fe::build_bn_forward(entry, cudnn_fe::bn_dims(input, features), fuse_relu);

        float epsilon_value = EPSILON;
        float momentum_value = momentum;

        unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensors;
        tensors[entry.fwd_X]        = input.data;
        tensors[entry.fwd_Scale]    = gamma.data;
        tensors[entry.fwd_Bias]     = beta.data;
        tensors[entry.fwd_PrevMean] = running_mean.data;
        tensors[entry.fwd_PrevVar]  = running_variance.data;
        tensors[entry.fwd_Eps]      = &epsilon_value;
        tensors[entry.fwd_Mom]      = &momentum_value;
        tensors[entry.fwd_Y]        = output.data;
        tensors[entry.fwd_Mean]     = mean.data;
        tensors[entry.fwd_InvVar]   = inverse_variance.data;
        tensors[entry.fwd_NextMean] = running_mean.data;
        tensors[entry.fwd_NextVar]  = running_variance.data;

        cudnn_fe::check_status(entry.fwd->execute(Backend::get_cudnn_handle(), tensors, entry.fwd_workspace),
                               "batchnorm forward execute");
    }))
        return;
#endif

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

    if (fuse_relu) activation_forward(output, ActivationFunction::ReLU);
}

void BatchNormOp::apply_delta_gpu(const TensorView& input,
                                const TensorView& output,
                                const TensorView& mean,
                                const TensorView& inverse_variance,
                                TensorView& delta) const
{
#ifdef HAVE_CUDNN_FRONTEND
    if (input.type == Type::FP32 && cudnn_fe::frontend_enabled()
        && cudnn_fe::run_frontend(bn_graph_cache, "BatchNormOp", [&](BnGraphCache& cache)
    {
        auto& entry = cache.entries[input.shape[0]];
        if (!entry.bwd)
            cudnn_fe::build_bn_backward(entry, cudnn_fe::bn_dims(input, features), fuse_relu);

        unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensors;
        tensors[entry.bwd_DY]     = delta.data;
        if (fuse_relu) tensors[entry.bwd_Bias] = beta.data;
        tensors[entry.bwd_X]      = input.data;
        tensors[entry.bwd_Scale]  = gamma.data;
        tensors[entry.bwd_Mean]   = mean.data;
        tensors[entry.bwd_InvVar] = inverse_variance.data;
        tensors[entry.bwd_DX]     = delta.data;
        tensors[entry.bwd_DScale] = gamma_gradient.data;
        tensors[entry.bwd_DBias]  = beta_gradient.data;

        cudnn_fe::check_status(entry.bwd->execute(Backend::get_cudnn_handle(), tensors, entry.bwd_workspace),
                               "batchnorm backward execute");
    }))
        return;
#endif

    if (fuse_relu) activation_backward(output, delta, ActivationFunction::ReLU);

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
                                    const TensorView&, TensorView&) const                               { throw runtime_error("BatchNorm::apply_delta_gpu: CUDA support not compiled in."); }

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
