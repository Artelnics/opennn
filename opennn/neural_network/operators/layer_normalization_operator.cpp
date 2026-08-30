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
#include "opennn/core/cuda/kernel_normalization.cuh"
#endif

namespace opennn
{

#ifdef OPENNN_HAS_CUDA

static void layer_normalization_forward_gpu(const TensorView& input, const TensorView& gamma, const TensorView& beta,
                            TensorView& means, TensorView& standard_deviations, TensorView& output, float epsilon)
{
    const int rows = to_int(input.flat_rows());
    const int cols = to_int(input.flat_columns());

    output.dispatch([&]<typename T>() {
        layernorm_forward_cuda<T>(rows, cols,
                                  input.as<T>(), output.as<T>(),
                                  means.as<float>(), standard_deviations.as<float>(),
                                  gamma.as<float>(), beta.as<float>(), epsilon);
    });
}

static void layer_normalization_backward_gpu(const TensorView& input, const TensorView& output_delta,
                             const TensorView& means, const TensorView& standard_deviations,
                             const TensorView& gamma,
                             const TensorView& gamma_gradient, const TensorView& beta_gradient,
                             TensorView& input_delta, TensorView* residual_delta)
{
    const int rows = to_int(input.flat_rows());
    const int cols = to_int(input.flat_columns());

    input.dispatch([&]<typename T>() {
        T* input_delta_data = input_delta.empty() ? nullptr : input_delta.as<T>();
        T* residual_delta_data = residual_delta && input_delta_data ? residual_delta->as<T>() : nullptr;

        layernorm_backward_cuda<T>(rows, cols,
                                   output_delta.as<T>(), input.as<T>(),
                                   means.as<float>(), standard_deviations.as<float>(),
                                   gamma.as<float>(),
                                   input_delta_data, residual_delta_data,
                                   gamma_gradient.as<float>(), beta_gradient.as<float>());
    });
}

static void rms_normalization_forward_gpu(const TensorView& input, const TensorView& weight,
                            TensorView& inverse_rms, TensorView& output, float epsilon)
{
    const int rows = to_int(input.flat_rows());
    const int cols = to_int(input.flat_columns());

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
    const int rows = to_int(input.flat_rows());
    const int cols = to_int(input.flat_columns());

    input.dispatch([&]<typename T>() {
        T* input_delta_data = input_delta.empty() ? nullptr : input_delta.as<T>();

        rmsnorm_backward_cuda<T>(rows, cols,
                                 output_delta.as<T>(), input.as<T>(),
                                 inverse_rms.as<float>(), weight.as<float>(),
                                 input_delta_data, weight_gradient.as<float>());
    });
}

#else

OPENNN_CUDA_TEMPLATE_STUB(layer_normalization_forward_gpu)
OPENNN_CUDA_TEMPLATE_STUB(layer_normalization_backward_gpu)
OPENNN_CUDA_TEMPLATE_STUB(rms_normalization_forward_gpu)
OPENNN_CUDA_TEMPLATE_STUB(rms_normalization_backward_gpu)

#endif

// LayerNorm and RMS norm are one row pass: RMS is LayerNorm with the mean held
// at zero and no beta. They were written out twice, ~85 lines that differed by
// about four. The per-row scalar each one saves for its backward still differs
// -- LayerNorm stores the standard deviation, RMS stores its reciprocal -- so
// `scales` holds whichever the method wants. The two output expressions stay
// spelled out separately rather than sharing one with a zero folded in, which
// would let the compiler contract them differently and move the last bit.
static TensorView unused_means;

static void normalization_forward_cpu(const TensorView& input, const TensorView& gamma, const TensorView& beta,
                            TensorView& means, TensorView& scales,
                            TensorView& normalized, TensorView& output,
                            float epsilon, NormalizationMethod method)
{
    const bool centered = method == NormalizationMethod::LayerNorm;

    const Index embedding_dimension = input.get_shape().back();
    const Index total_rows = input.size() / embedding_dimension;
    const float inv_D = 1.0f / to_type(embedding_dimension);

    const float* input_data = input.as<float>();
    float* means_data       = centered ? means.as<float>() : nullptr;
    float* scales_data      = scales.as<float>();
    float* normalized_data  = normalized.as<float>();
    float* output_data      = output.as<float>();
    const float* gamma_data = gamma.as<float>();
    const float* beta_data  = centered ? beta.as<float>() : nullptr;

    using ArrayMap = Map<Array<float, Dynamic, 1>>;
    using ConstArrayMap = Map<const Array<float, Dynamic, 1>>;

    #pragma omp parallel for schedule(static)
    for (Index row = 0; row < total_rows; ++row)
    {
        const float* input_row = input_data + row * embedding_dimension;
        float* norm_row        = normalized_data + row * embedding_dimension;
        float* out_row         = output_data + row * embedding_dimension;

        const ConstArrayMap input_map(input_row, embedding_dimension);

        const float mean = centered ? input_map.sum() * inv_D : 0.0f;

        const float variance = centered
            ? max((input_map - mean).square().sum() * inv_D, 0.0f)
            : input_map.square().sum() * inv_D;

        const float deviation = sqrt(variance + epsilon);
        const float inv_std   = 1.0f / deviation;

        if (centered) means_data[row] = mean;
        scales_data[row] = centered ? deviation : inv_std;

        ArrayMap norm_map(norm_row, embedding_dimension);
        norm_map = (input_map - mean) * inv_std;

        const ConstArrayMap gamma_map(gamma_data, embedding_dimension);
        ArrayMap out_map(out_row, embedding_dimension);

        if (centered) out_map = gamma_map * norm_map + ConstArrayMap(beta_data, embedding_dimension);
        else          out_map = gamma_map * norm_map;
    }
}

static void normalization_backward_cpu(const TensorView& output_delta,
                             const TensorView& scales,
                             const TensorView& normalized,
                             const TensorView& gamma,
                             const TensorView& gamma_gradient,
                             const TensorView& beta_gradient,
                             TensorView& input_delta,
                             NormalizationMethod method)
{
    const bool centered = method == NormalizationMethod::LayerNorm;

    const Index embedding_dimension = output_delta.get_shape().back();
    const Index total_rows = output_delta.size() / embedding_dimension;
    const float inv_D = 1.0f / to_type(embedding_dimension);

    const MatrixMap output_delta_flat = output_delta.as_flat_matrix();
    const MatrixMap norm_flat         = normalized.as_flat_matrix();

    if (centered)
        beta_gradient.as_vector().noalias() = output_delta_flat.colwise().sum();

    gamma_gradient.as_vector().noalias() = (output_delta_flat.array() * norm_flat.array()).matrix().colwise().sum();

    if (input_delta.empty()) return;

    const float* output_delta_data = output_delta.as<float>();
    const float* norm_data         = normalized.as<float>();
    const float* scales_data       = scales.as<float>();
    const float* gamma_data        = gamma.as<float>();
    float* input_delta_data        = input_delta.as<float>();

    using ArrayMap = Map<Array<float, Dynamic, 1>>;
    using ConstArrayMap = Map<const Array<float, Dynamic, 1>>;

    #pragma omp parallel for schedule(static)
    for (Index row = 0; row < total_rows; ++row)
    {
        const float* output_delta_row = output_delta_data + row * embedding_dimension;
        const float* norm_row         = norm_data + row * embedding_dimension;
        float* input_delta_row        = input_delta_data + row * embedding_dimension;

        // LayerNorm saved the deviation, RMS saved its reciprocal.
        const float inv_std = centered ? 1.0f / scales_data[row] : scales_data[row];

        const ConstArrayMap gamma_map(gamma_data, embedding_dimension);
        const ConstArrayMap output_delta_map(output_delta_row, embedding_dimension);
        const ConstArrayMap norm_map(norm_row, embedding_dimension);
        ArrayMap input_delta_map(input_delta_row, embedding_dimension);

        input_delta_map = gamma_map * output_delta_map;

        const float sum_scaled_gradient      = centered ? input_delta_map.sum() * inv_D : 0.0f;
        const float sum_scaled_gradient_norm = (input_delta_map * norm_map).sum() * inv_D;

        if (centered)
            input_delta_map = (input_delta_map - sum_scaled_gradient
                              - norm_map * sum_scaled_gradient_norm) * inv_std;
        else
            input_delta_map = (input_delta_map - norm_map * sum_scaled_gradient_norm) * inv_std;
    }
}

void layer_normalization_forward(const TensorView& input, const TensorView& gamma, const TensorView& beta,
                        TensorView& means, TensorView& standard_deviations,
                        TensorView& normalized, TensorView& output, float epsilon)
{
    if (input.is_cuda()) { layer_normalization_forward_gpu(input, gamma, beta, means, standard_deviations, output, epsilon); return; }
    normalization_forward_cpu(input, gamma, beta, means, standard_deviations, normalized, output, epsilon,
                              NormalizationMethod::LayerNorm);
}

void layer_normalization_add_forward(const TensorView& input, const TensorView& residual,
                            const TensorView& gamma, const TensorView& beta,
                            TensorView& means, TensorView& standard_deviations,
                            TensorView& normalized, TensorView& sum, TensorView& output, float epsilon)
{
#ifdef OPENNN_HAS_CUDA
    if (input.is_cuda())
    {
        const int rows = to_int(input.flat_rows());
        const int cols = to_int(input.flat_columns());
        output.dispatch([&]<typename T>() {
            // The slot is absent in inference: nothing reads x + residual once
            // there is no backward pass to read it.
            layernorm_add_forward_cuda<T>(rows, cols,
                                          input.as<T>(), residual.as<T>(),
                                          sum.empty() ? nullptr : sum.as<T>(), output.as<T>(),
                                          means.as<float>(), standard_deviations.as<float>(),
                                          gamma.as<float>(), beta.as<float>(), epsilon);
        });
        return;
    }
#endif
    add(input, residual, sum);
    normalization_forward_cpu(sum, gamma, beta, means, standard_deviations, normalized, output, epsilon,
                              NormalizationMethod::LayerNorm);
}

void layer_normalization_backward(const TensorView& input, const TensorView& output_delta,
                         const TensorView& means, const TensorView& standard_deviations,
                         const TensorView& normalized, const TensorView& gamma,
                         const TensorView& gamma_gradient, const TensorView& beta_gradient,
                         TensorView& input_delta, TensorView* residual_delta)
{
    if (input.is_cuda())
        return layer_normalization_backward_gpu(input, output_delta, means, standard_deviations, gamma,
                                       gamma_gradient, beta_gradient, input_delta, residual_delta);
    normalization_backward_cpu(output_delta, standard_deviations, normalized, gamma,
                               gamma_gradient, beta_gradient, input_delta,
                               NormalizationMethod::LayerNorm);
    if (residual_delta) copy(input_delta, *residual_delta);
}

void rms_normalization_forward(const TensorView& input, const TensorView& weight,
                      TensorView& inverse_rms, TensorView& normalized, TensorView& output,
                      float epsilon)
{
    if (input.is_cuda()) { rms_normalization_forward_gpu(input, weight, inverse_rms, output, epsilon); return; }
    normalization_forward_cpu(input, weight, TensorView{}, unused_means, inverse_rms, normalized, output, epsilon,
                              NormalizationMethod::RMS);
}

void rms_normalization_backward(const TensorView& input, const TensorView& output_delta,
                       const TensorView& inverse_rms, const TensorView& normalized,
                       const TensorView& weight, const TensorView& weight_gradient,
                       TensorView& input_delta)
{
    if (input.is_cuda())
        return rms_normalization_backward_gpu(input, output_delta, inverse_rms, weight,
                                     weight_gradient, input_delta);
    normalization_backward_cpu(output_delta, inverse_rms, normalized, weight,
                               weight_gradient, TensorView{}, input_delta,
                               NormalizationMethod::RMS);
}

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

vector<Operator::ParameterSlot> LayerNormalizationOperator::parameter_slots()
{
    return {
        {&gamma, &gamma_gradient},
        {&beta,  &beta_gradient, method != NormalizationMethod::RMS},
    };
}

void LayerNormalizationOperator::init_defaults()
{
    if (gamma.get_data()) gamma.as_vector().setOnes();
    if (beta.get_data())  beta.as_vector().setZero();
}

void LayerNormalizationOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, ForwardPropagationMode  )
{
    const TensorView& input = get_input(forward_propagation, layer);

    if (method == NormalizationMethod::RMS)
    {

        TensorView& inverse_rms = get_output(forward_propagation, layer);
        TensorView& normalized  = get_output(forward_propagation, layer, 2);
        TensorView& output      = get_output(forward_propagation, layer, 3);

        return rms_normalization_forward(input, gamma, inverse_rms, normalized, output, epsilon);
    }

    TensorView& means       = get_output(forward_propagation, layer);
    TensorView& stds        = get_output(forward_propagation, layer, 1);
    TensorView& normalized  = get_output(forward_propagation, layer, 2);
    TensorView& output      = get_output(forward_propagation, layer, 3);

    if (fuse_add)
    {

        const TensorView& residual = forward_propagation.inputs[layer][1];
        return layer_normalization_add_forward(input, residual, gamma, beta, means, stds, normalized, normalized, output, epsilon);
    }

    layer_normalization_forward(input, gamma, beta, means, stds, normalized, output, epsilon);
}

void LayerNormalizationOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    const TensorView& output_delta = get_output_delta(back_propagation, layer);
    TensorView& input_delta        = get_input_delta(back_propagation, layer);

    if (method == NormalizationMethod::RMS)
    {
        const TensorView& inverse_rms = get_output(forward_propagation, layer);
        const TensorView& normalized  = get_output(forward_propagation, layer, 2);

        return rms_normalization_backward(get_input(forward_propagation, layer), output_delta,
                                          inverse_rms, normalized, gamma, gamma_gradient, input_delta);
    }

    const TensorView& stds       = get_output(forward_propagation, layer, 1);
    const TensorView& normalized = get_output(forward_propagation, layer, 2);

    const TensorView& norm_input = fuse_add ? normalized : get_input(forward_propagation, layer);

    TensorView* residual_delta = nullptr;
    if (fuse_add)
    {
        TensorView& residual = back_propagation.slots[layer][2];
        if (residual.get_data() && residual.get_data() != input_delta.get_data())
            residual_delta = &residual;
    }

    layer_normalization_backward(norm_input, output_delta, get_output(forward_propagation, layer),
                        stds, normalized, gamma, gamma_gradient, beta_gradient,
                        input_delta, residual_delta);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
