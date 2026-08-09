//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   N O R M   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/operators/layer_normalization_operator.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/core/cuda/kernel.cuh"
#endif

namespace opennn
{

// Defined below: against the CUDA kernels, or as throwing stubs.
static void layer_normalization_forward_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&, TensorView&, TensorView&);
static void layer_normalization_backward_gpu(const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, TensorView&);
static void rms_normalization_forward_gpu(const TensorView&, const TensorView&, TensorView&, TensorView&, float);
static void rms_normalization_backward_gpu(const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, TensorView&);

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

        Map<Array<float, Dynamic, 1>> norm_map(norm_row, embedding_dimension);
        norm_map = (input_map - mean) * inv_std;

        Map<Array<float, Dynamic, 1>>(out_row, embedding_dimension) =
            Map<const Array<float, Dynamic, 1>>(gamma_data, embedding_dimension) * norm_map
            + Map<const Array<float, Dynamic, 1>>(beta_data, embedding_dimension);
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
        output.dispatch([&]<typename T>() {
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

        Map<Array<float, Dynamic, 1>> norm_map(norm_row, embedding_dimension);
        norm_map = input_map * inverse;

        Map<Array<float, Dynamic, 1>>(out_row, embedding_dimension) =
            Map<const Array<float, Dynamic, 1>>(weight_data, embedding_dimension) * norm_map;
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

#ifdef OPENNN_HAS_CUDA

static void layer_normalization_forward_gpu(const TensorView& input, const TensorView& gamma, const TensorView& beta,
                            TensorView& means, TensorView& standard_deviations, TensorView& output)
{
    const int rows = to_int(input.size() / input.shape.back());
    const int cols = to_int(input.shape.back());

    output.dispatch([&]<typename T>() {
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

    input.dispatch([&]<typename T>() {
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

    output.dispatch([&]<typename T>() {
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

    input.dispatch([&]<typename T>() {
        T* input_delta_data = input_delta.empty() ? nullptr : input_delta.as<T>();

        rmsnorm_backward_cuda<T>(rows, cols,
                                 output_delta.as<T>(), input.as<T>(),
                                 inverse_rms.as<float>(), weight.as<float>(),
                                 input_delta_data, weight_gradient.as<float>());
    });
}

#else

static void layer_normalization_forward_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&, TensorView&, TensorView&) { throw runtime_error("layer_normalization_forward_gpu: CUDA support not compiled in."); }
static void layer_normalization_backward_gpu(const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, TensorView&) { throw runtime_error("layer_normalization_backward_gpu: CUDA support not compiled in."); }
static void rms_normalization_forward_gpu(const TensorView&, const TensorView&, TensorView&, TensorView&, float) { throw runtime_error("rms_normalization_forward_gpu: CUDA support not compiled in."); }
static void rms_normalization_backward_gpu(const TensorView&, const TensorView&, const TensorView&, const TensorView&, const TensorView&, TensorView&) { throw runtime_error("rms_normalization_backward_gpu: CUDA support not compiled in."); }

#endif


void LayerNormalizationOperator::set(Index new_sequence_length, Index new_embedding_dimension)
{
    sequence_length     = new_sequence_length;
    embedding_dimension = new_embedding_dimension;
}

vector<TensorSpec> LayerNormalizationOperator::parameter_specs() const
{

    const size_t count = (method == NormalizationMethod::RMS) ? 1 : 2;
    return vector<TensorSpec>(count, {Shape{embedding_dimension}, Type::FP32});
}

void LayerNormalizationOperator::link_parameters(span<const TensorView> views)
{
    if (method == NormalizationMethod::RMS)
    {
        beta = {};
        link_views(views, {&gamma});
        return;
    }
    link_views(views, {&gamma, &beta});
}

void LayerNormalizationOperator::link_gradients(span<const TensorView> views)
{
    if (method == NormalizationMethod::RMS)
    {
        beta_gradient = {};
        link_views(views, {&gamma_gradient});
        return;
    }
    link_views(views, {&gamma_gradient, &beta_gradient});
}

void LayerNormalizationOperator::init_defaults()
{
    if (gamma.data) gamma.as_vector().setOnes();
    if (beta.data)  beta.as_vector().setZero();
}

void LayerNormalizationOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool  )
{
    const TensorView& input = get_input(forward_propagation, layer);

    if (method == NormalizationMethod::RMS)
    {

        TensorView& inverse_rms = get_output(forward_propagation, layer);
        TensorView& normalized  = get_output(forward_propagation, layer, 2);
        TensorView& output      = get_output(forward_propagation, layer, 3);

        rms_normalization_forward(input, gamma, inverse_rms, normalized, output, epsilon);
        return;
    }

    TensorView& means       = get_output(forward_propagation, layer);
    TensorView& stds        = get_output(forward_propagation, layer, 1);
    TensorView& normalized  = get_output(forward_propagation, layer, 2);
    TensorView& output      = get_output(forward_propagation, layer, 3);

    if (fuse_add)
    {

        const TensorView& residual = forward_propagation.inputs[layer][1];
        layer_normalization_add_forward(input, residual, gamma, beta, means, stds, normalized, normalized, output);
        return;
    }

    layer_normalization_forward(input, gamma, beta, means, stds, normalized, output);
}

void LayerNormalizationOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    const TensorView& output_delta = get_output_delta(back_propagation, layer);
    TensorView& input_delta        = get_input_delta(back_propagation, layer);

    if (method == NormalizationMethod::RMS)
    {
        const TensorView& inverse_rms = get_output(forward_propagation, layer);
        const TensorView& normalized  = get_output(forward_propagation, layer, 2);

        rms_normalization_backward(get_input(forward_propagation, layer), output_delta,
                                   inverse_rms, normalized, gamma, gamma_gradient, input_delta);
        return;
    }

    const TensorView& stds       = get_output(forward_propagation, layer, 1);
    const TensorView& normalized = get_output(forward_propagation, layer, 2);

    const TensorView& norm_input = fuse_add ? normalized : get_input(forward_propagation, layer);

    layer_normalization_backward(norm_input, output_delta, get_output(forward_propagation, layer),
                        stds, normalized, gamma, gamma_gradient, beta_gradient,
                        input_delta);

    if (fuse_add)
    {
        TensorView& residual_delta = back_propagation.slots[layer][2];
        if (residual_delta.data && residual_delta.data != input_delta.data)
            copy(input_delta, residual_delta);
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
