//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/layers/recurrent_layer.h"
#include "opennn/registry.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"

#include "opennn/core/device_backend.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/layers/kernel_recurrent.cuh"
#include "opennn/core/cuda/kernel_tensor.cuh"

namespace opennn
{

void RecurrentOperator::set(Index new_input_features,
                      Index new_time_steps,
                      Index new_output_features,
                      ActivationFunction new_activation,
                      Type new_compute_dtype)
{
    input_features  = new_input_features;
    time_steps      = new_time_steps;
    output_features = new_output_features;
    activation      = new_activation;
    compute_dtype   = new_compute_dtype;
}

vector<TensorSpec> RecurrentOperator::parameter_specs() const
{
    return {
        {{output_features},                   compute_dtype},
        {{input_features, output_features},   compute_dtype},
        {{output_features, output_features},  compute_dtype},
    };
}

void RecurrentOperator::link_parameters(span<const TensorView> views)
{
    link_views(views, {&bias, &input_weights, &recurrent_weights});
}

void RecurrentOperator::link_gradients(span<const TensorView> views)
{
    link_views(views, {&bias_gradient, &input_weight_gradient, &recurrent_weight_gradient});
}

void RecurrentOperator::set_parameters_random()
{
    if (!input_weights.empty())     set_random_uniform(input_weights.as_vector());
    if (!recurrent_weights.empty()) set_random_uniform(recurrent_weights.as_vector());
    if (!bias.empty())              bias.setZero();
}

void RecurrentOperator::set_parameters_glorot()
{
    if (!input_weights.empty())
    {
        const float limit = glorot_limit(input_features, output_features);
        set_random_uniform(input_weights.as_vector(), -limit, limit);
    }
    if (!recurrent_weights.empty())
        set_random_orthogonal(recurrent_weights.as_matrix());
    if (!bias.empty()) bias.setZero();
}

void RecurrentOperator::set_parameters_pytorch()
{
    const float limit = 1.0f / sqrt(float(output_features > 0 ? output_features : 1));
    if (!input_weights.empty())     set_random_uniform(input_weights.as_vector(), -limit, limit);
    if (!recurrent_weights.empty()) set_random_uniform(recurrent_weights.as_vector(), -limit, limit);
    if (!bias.empty())              set_random_uniform(bias.as_vector(), -limit, limit);
}

void RecurrentOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training)
{
    auto& forward_slots = forward_propagation.slots[layer];
    const TensorView& input             = get_input(forward_propagation, layer);
    TensorView& output                  = forward_slots[output_slots[0]];
    TensorView& hidden_states           = forward_slots[output_slots[1]];
    TensorView& activation_derivatives  = forward_slots[output_slots[2]];

    if (input.is_cuda())
        return apply_gpu(input, hidden_states, activation_derivatives, output,
                         forward_slots[StepInputForwardSlot],
                         forward_slots[StepHiddenForwardSlot],
                         forward_slots[PreviousHiddenForwardSlot],
                         forward_slots[StepDerivativesForwardSlot],
                         forward_propagation.layer_state_storage[layer],
                         is_training);
    apply(input, hidden_states, activation_derivatives, output, is_training);
}

void RecurrentOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    auto& forward_slots = forward_propagation.slots[layer];
    auto& backward_slots = back_propagation.slots[layer];

    const TensorView& input                    = get_input(forward_propagation, layer);
    const TensorView& hidden_states            = forward_slots[output_slots[1]];
    const TensorView& activation_derivatives   = forward_slots[output_slots[2]];
    const TensorView& output_delta             = get_output_delta(back_propagation, layer);

    TensorView empty_input_delta;
    TensorView& input_delta = slot_or(backward_slots, input_delta_slots, 0,
                                      empty_input_delta);

    if (output_delta.is_cuda())
    {
        TensorView& step_input_scratch    = backward_slots[StepInputScratchSlot];
        TensorView& step_prev_h_scratch   = backward_slots[StepPrevHScratchSlot];
        TensorView& delta_scratch         = backward_slots[DeltaScratchSlot];
        TensorView& next_carry_scratch    = backward_slots[NextCarryScratchSlot];
        TensorView& step_in_delta_scratch = backward_slots[StepInDeltaScratchSlot];
        TensorView& sequence_delta_scratch = backward_slots[SequenceDeltaScratchSlot];
        TensorView& cudnn_input_delta_scratch = backward_slots[CudnnInputDeltaScratchSlot];
        return apply_delta_gpu(input, hidden_states, activation_derivatives,
                               output_delta, input_delta,
                               step_input_scratch, step_prev_h_scratch,
                               delta_scratch, next_carry_scratch, step_in_delta_scratch,
                               sequence_delta_scratch, cudnn_input_delta_scratch,
                               forward_propagation.layer_state_storage[layer],
                               back_propagation.layer_scratch_storage[layer]);
    }
    apply_delta(input, hidden_states, activation_derivatives, output_delta, input_delta);
}

namespace
{

using StridedMap      = Eigen::Map<MatrixR, 0, Eigen::OuterStride<>>;
using ConstStridedMap = Eigen::Map<const MatrixR, 0, Eigen::OuterStride<>>;

constexpr bool supports_recurrent_activation(const ActivationFunction activation) noexcept
{
    using enum ActivationFunction;
    return is_one_of(activation, Identity, Sigmoid, Tanh, ReLU);
}

void activate_in_place(ActivationFunction activation,
                       StridedMap& values, StridedMap* derivatives)
{
    using enum ActivationFunction;
    switch (activation)
    {
    case Tanh:
        values.array() = values.array().tanh();
        if (derivatives) derivatives->array() = 1.0f - values.array().square();
        break;
    case Sigmoid:
        values.array() = (1.0f + (-values.array()).exp()).inverse();
        if (derivatives) derivatives->array() = values.array() * (1.0f - values.array());
        break;
    case ReLU:
        values.array() = values.array().max(0.0f);
        if (derivatives) derivatives->array() = (values.array() > 0.0f).cast<float>();
        break;
    case Identity:
        if (derivatives) derivatives->setOnes();
        break;
    case Softmax:
    case LeakyReLU:
    case GELU:
    case GELUTanh:
    case SiLU:
        throw runtime_error("RecurrentOperator: unsupported activation.");
    }
}

}

void RecurrentOperator::apply(const TensorView& input,
                            TensorView& hidden_states,
                            TensorView& activation_derivatives,
                            TensorView& output,
                            bool is_training) const
{
    const Index batch_size = input.get_shape()[0];
    const Index BT = batch_size * time_steps;

    const VectorMap bias_map  = bias.as_vector();
    const MatrixMap w_in_map  = input_weights.as_matrix();
    const MatrixMap w_rec_map = recurrent_weights.as_matrix();

    const float* input_data  = input.as<float>();
    float*       hidden_data = hidden_states.as<float>();
    float*       derivs_data = (is_training && !activation_derivatives.empty())
                               ? activation_derivatives.as<float>() : nullptr;

    const Index h_stride_b = time_steps * output_features;

    Eigen::Map<const MatrixR> all_input(input_data, BT, input_features);
    MatrixMap all_hidden(hidden_data, BT, output_features);

    all_hidden.noalias() = all_input * w_in_map;
    all_hidden.rowwise() += bias_map.transpose();

    MatrixR h_c(batch_size, output_features);
    MatrixR rec_acc(batch_size, output_features);

    const int eigen_threads = Eigen::nbThreads();
    Eigen::setNbThreads(1);

    for (Index t = 0; t < time_steps; ++t)
    {
        StridedMap h_t(hidden_data + t * output_features,
                       batch_size, output_features, Eigen::OuterStride<>(h_stride_b));

        if (t > 0)
        {
            rec_acc.noalias() = h_c * w_rec_map;
            h_t += rec_acc;
        }

        if (derivs_data)
        {
            StridedMap d_t(derivs_data + t * output_features,
                           batch_size, output_features, Eigen::OuterStride<>(h_stride_b));
            activate_in_place(activation, h_t, &d_t);
        }
        else
            activate_in_place(activation, h_t, nullptr);

        h_c = h_t;
    }

    Eigen::setNbThreads(eigen_threads);

    if (return_sequences)
        memcpy(output.as<float>(), hidden_data,
               size_t(BT) * output_features * sizeof(float));
    else
        output.as_matrix() = ConstStridedMap(hidden_data + (time_steps - 1) * output_features,
                                             batch_size, output_features,
                                             Eigen::OuterStride<>(h_stride_b));
}

void RecurrentOperator::apply_delta(const TensorView& input,
                              const TensorView& hidden_states,
                              const TensorView& activation_derivatives,
                              const TensorView& output_delta,
                              TensorView& input_delta) const
{
    const Index batch_size = input.get_shape()[0];

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

    const bool write_input_delta = !input_delta.empty() && input_delta.get_data() != nullptr;
    float* input_delta_data = write_input_delta ? input_delta.as<float>() : nullptr;

    const Index h_stride_b = time_steps * output_features;

    const float* seq_delta_data = return_sequences ? output_delta.as<float>()
                                                   : nullptr;
    const float* final_delta_data = return_sequences ? nullptr
                                                     : output_delta.as<float>();

    const Index BT = batch_size * time_steps;

    MatrixR all_delta(BT, output_features);
    MatrixR d_c(batch_size, output_features);
    MatrixR h_prev_c(batch_size, output_features);
    MatrixR next_carry = MatrixR::Zero(batch_size, output_features);

    const int eigen_threads = Eigen::nbThreads();
    Eigen::setNbThreads(1);

    for (Index t = time_steps - 1; t >= 0; --t)
    {
        const ConstStridedMap derivs_t(derivs_data + t * output_features,
                                       batch_size, output_features,
                                       Eigen::OuterStride<>(h_stride_b));

        if (return_sequences)
        {
            const ConstStridedMap out_delta_t(seq_delta_data + t * output_features,
                                              batch_size, output_features,
                                              Eigen::OuterStride<>(h_stride_b));
            d_c.array() = (next_carry.array() + out_delta_t.array()) * derivs_t.array();
        }
        else if (t == time_steps - 1)
        {
            d_c.array() = Eigen::Map<const MatrixR>(final_delta_data, batch_size, output_features)
                              .array() * derivs_t.array();
        }
        else
        {
            d_c.array() = next_carry.array() * derivs_t.array();
        }

        StridedMap(all_delta.data() + t * output_features,
                   batch_size, output_features, Eigen::OuterStride<>(h_stride_b)) = d_c;

        if (t > 0)
        {
            h_prev_c = ConstStridedMap(hidden_data + (t - 1) * output_features,
                                       batch_size, output_features,
                                       Eigen::OuterStride<>(h_stride_b));
            w_rec_grad.noalias() += h_prev_c.transpose() * d_c;
            next_carry.noalias()  = d_c * w_rec_map.transpose();
        }
    }

    Eigen::setNbThreads(eigen_threads);

    const Eigen::Map<const MatrixR> all_input(input_data, BT, input_features);
    const Eigen::Map<const MatrixR> all_delta_map(all_delta.data(), BT, output_features);

    w_in_grad.noalias() = all_input.transpose() * all_delta_map;
    bias_grad.noalias() = all_delta_map.colwise().sum().transpose();

    if (write_input_delta)
        Eigen::Map<MatrixR>(input_delta_data, BT, input_features).noalias()
            = all_delta_map * w_in_map.transpose();
}

#ifdef OPENNN_HAS_CUDA

static void zero_device_view(const TensorView& view)
{
    if (!view.get_data() || view.empty()) return;
    device::set_zero_async(view.get_data(), view.byte_size(), device::get_compute_stream());
}

static void require_same_recurrent_dtype(const TensorView& reference,
                                         initializer_list<pair<const TensorView*, const char*>> views)
{
    for (const auto& [view, name] : views)
        throw_if(view->get_data() && !view->empty() && view->get_type() != reference.get_type(),
                 "RecurrentOperator CUDA: {} dtype does not match recurrent compute dtype.", name);
}

bool RecurrentOperator::cudnn_rnn_eligible_(const TensorView& reference) const
{
    return is_one_of(activation, ActivationFunction::Tanh, ActivationFunction::ReLU)
        && reference.is_fp32();
}

static CudnnRnnConfig recurrent_cudnn_config(ActivationFunction activation)
{
    return {activation == ActivationFunction::ReLU ? CUDNN_RNN_RELU
                                                   : CUDNN_RNN_TANH};
}

void RecurrentOperator::ensure_cudnn_setup_(Index batch_size, bool for_training) const
{
    cudnn_setup_(recurrent_cudnn_config(activation),
                 input_features, output_features, time_steps,
                 batch_size, for_training);
}

void RecurrentOperator::pack_weights_to_cudnn_(Buffer& forward_state) const
{
    const TensorView* weights[2] = {&input_weights, &recurrent_weights};
    const TensorView* biases[2]  = {&bias, nullptr};
    cudnn_pack_weights_(2, input_features, output_features,
                        weights, biases, forward_state);
}

void RecurrentOperator::unpack_gradients_from_cudnn_(Buffer& backward_scratch) const
{
    const TensorView* weight_gradients[2] = {&input_weight_gradient, &recurrent_weight_gradient};
    const TensorView* bias_gradients[2]   = {&bias_gradient, nullptr};
    cudnn_unpack_gradients_(2, input_features, output_features,
                            weight_gradients, bias_gradients,
                            backward_scratch);
}

void RecurrentOperator::apply_gpu_cudnn_(const TensorView& input,
                                         TensorView& hidden_states,
                                         TensorView& output,
                                         Buffer& forward_state,
                                         bool is_training) const
{
    const Index batch_size = input.get_shape()[0];

    ensure_cudnn_setup_(batch_size, is_training);
    prepare_cudnn_forward_state_(forward_state, is_training);
    pack_weights_to_cudnn_(forward_state);

    cudnn_rnn_forward_(is_training,  false,
                       input.get_data(), hidden_states.get_data(),
                       forward_state,
                       [&] {
                           ensure_cudnn_setup_(batch_size, is_training);
                           prepare_cudnn_forward_state_(forward_state, is_training);
                           pack_weights_to_cudnn_(forward_state);
                       });

    if (return_sequences)
        copy(hidden_states, output);
    else
        gather_time_slice_cuda<float>(
            batch_size, time_steps, output_features, time_steps - 1,
            hidden_states.as<float>(), output.as<float>());
}

void RecurrentOperator::apply_delta_gpu_cudnn_(const TensorView& input,
                                               const TensorView& hidden_states,
                                               const TensorView& output_delta,
                                               TensorView& input_delta,
                                               TensorView& sequence_delta_scratch,
                                               TensorView& input_delta_scratch,
                                               const Buffer& forward_state,
                                               Buffer& backward_scratch) const
{
    const Index batch_size = input.get_shape()[0];
    const Index H = output_features;
    const Index T = time_steps;

    ensure_cudnn_setup_(batch_size, true);

    const float* dy_data = output_delta.as<float>();
    if (!return_sequences)
    {
        device::set_zero_async(sequence_delta_scratch.get_data(),
                               sequence_delta_scratch.byte_size(),
                               device::get_compute_stream());
        scatter_time_slice_cuda<float>(
            batch_size, T, H, T - 1,
            output_delta.as<float>(),
            sequence_delta_scratch.as<float>());
        dy_data = sequence_delta_scratch.as<float>();
    }

    void* dx_data = input_delta.get_data()
        ? input_delta.get_data()
        : input_delta_scratch.get_data();

    cudnn_rnn_backward_( false,
                        input.get_data(), hidden_states.get_data(), dy_data, dx_data,
                        forward_state, backward_scratch);

    unpack_gradients_from_cudnn_(backward_scratch);
}

void RecurrentOperator::apply_gpu(const TensorView& input,
                            TensorView& hidden_states,
                            TensorView& activation_derivatives,
                            TensorView& output,
                            TensorView& step_input_scratch,
                            TensorView& step_hidden_scratch,
                            TensorView& previous_hidden_scratch,
                            TensorView& step_derivatives_scratch,
                            Buffer& forward_state,
                            bool is_training) const
{
    if (!input.get_data() || output_features == 0 || time_steps == 0) return;

    if (cudnn_rnn_eligible_(output))
        return apply_gpu_cudnn_(input, hidden_states, output,
                                forward_state, is_training);

    require_same_recurrent_dtype(output, {
        {&input, "input"},
        {&hidden_states, "hidden_states"},
        {&activation_derivatives, "activation_derivatives"},
        {&bias, "bias"},
        {&input_weights, "input_weights"},
        {&recurrent_weights, "recurrent_weights"}
    });

    output.dispatch([&]<typename Scalar>()
    {
        const Index batch_size = input.get_shape()[0];

        TensorView* step_hidden = &step_hidden_scratch;
        TensorView* previous_hidden = &previous_hidden_scratch;

        for (Index t = 0; t < time_steps; ++t)
        {
            gather_time_slice_cuda<Scalar>(batch_size, time_steps, input_features, t,
                                           input.as<Scalar>(), step_input_scratch.as<Scalar>());

            const Scalar* prev_h_ptr = (t > 0)
                ? previous_hidden->as<Scalar>()
                : nullptr;

            Scalar* derivs = nullptr;
            if (is_training && !activation_derivatives.empty())
                derivs = step_derivatives_scratch.as<Scalar>();

            rnn_step_fused_forward_cuda<Scalar>(batch_size,
                                                input_features,
                                                output_features,
                                                step_input_scratch.as<Scalar>(),
                                                prev_h_ptr,
                                                input_weights.as<Scalar>(),
                                                recurrent_weights.as<Scalar>(),
                                                bias.as<Scalar>(),
                                                step_hidden->as<Scalar>(),
                                                derivs,
                                                static_cast<int>(activation));

            scatter_time_slice_cuda<Scalar>(batch_size, time_steps, output_features, t,
                                            step_hidden->as<Scalar>(), hidden_states.as<Scalar>());

            if (derivs)
                scatter_time_slice_cuda<Scalar>(batch_size, time_steps, output_features, t,
                                                derivs, activation_derivatives.as<Scalar>());

            swap(step_hidden, previous_hidden);
        }

        if (return_sequences)
            copy(hidden_states, output);
        else
        {
            copy(*previous_hidden, output);
        }
    });
}

void RecurrentOperator::apply_delta_gpu(const TensorView& input,
                                  const TensorView& hidden_states,
                                  const TensorView& activation_derivatives,
                                  const TensorView& output_delta,
                                  TensorView& input_delta,
                                  TensorView& step_input_scratch,
                                  TensorView& step_prev_h_scratch,
                                  TensorView& delta_scratch,
                                  TensorView& next_carry_scratch,
                                  TensorView& step_in_delta_scratch,
                                  TensorView& sequence_delta_scratch,
                                  TensorView& cudnn_input_delta_scratch,
                                  const Buffer& forward_state,
                                  Buffer& backward_scratch) const
{
    if (!input.get_data() || !output_delta.get_data() || output_features == 0 || time_steps == 0) return;

    if (cudnn_rnn_eligible_(output_delta))
        return apply_delta_gpu_cudnn_(input, hidden_states, output_delta,
                                      input_delta, sequence_delta_scratch,
                                      cudnn_input_delta_scratch,
                                      forward_state, backward_scratch);

    require_same_recurrent_dtype(output_delta, {
        {&input, "input"},
        {&hidden_states, "hidden_states"},
        {&activation_derivatives, "activation_derivatives"},
        {&input_weights, "input_weights"},
        {&recurrent_weights, "recurrent_weights"},
        {&input_delta, "input_delta"},
        {&step_input_scratch, "step_input_scratch"},
        {&step_prev_h_scratch, "step_prev_h_scratch"},
        {&delta_scratch, "delta_scratch"},
        {&next_carry_scratch, "next_carry_scratch"},
        {&step_in_delta_scratch, "step_in_delta_scratch"}
    });

    output_delta.dispatch([&]<typename Scalar>()
    {
        const Index batch_size = input.get_shape()[0];

        zero_device_view(bias_gradient);
        zero_device_view(input_weight_gradient);
        zero_device_view(recurrent_weight_gradient);

        const cudaDataType_t axpy_dtype = output_delta.cuda_dtype();

        for (Index t = time_steps; t-- > 0;)
        {
            const bool first_iter = (t == time_steps - 1);

            const Scalar* delta_src = nullptr;
            const Scalar* carry_src = nullptr;

            if (return_sequences)
            {
                gather_time_slice_cuda<Scalar>(batch_size, time_steps,
                                               output_features, t,
                                               output_delta.as<Scalar>(),
                                               sequence_delta_scratch.as<Scalar>());
                if (!first_iter)
                {
                    const float alpha = 1.0f;
                    const int   n     = to_int(batch_size * output_features);
                    CHECK_CUBLAS(cublasAxpyEx(device::get_cublas_handle(), n,
                                              &alpha, CUDA_R_32F,
                                              next_carry_scratch.get_data(), axpy_dtype, 1,
                                              sequence_delta_scratch.get_data(), axpy_dtype, 1,
                                              CUDA_R_32F));
                }
                delta_src = sequence_delta_scratch.as<Scalar>();
                carry_src = nullptr;
            }
            else
            {
                delta_src = first_iter ? output_delta.as<Scalar>() : nullptr;
                carry_src = first_iter ? nullptr : next_carry_scratch.as<Scalar>();
            }

            rnn_step_fused_backward_pre_cuda<Scalar>(
                batch_size, output_features, time_steps, t,
                delta_src, carry_src,
                activation_derivatives.as<Scalar>(),
                delta_scratch.as<Scalar>());

            bias_grad_sum_cuda<Scalar>(
                batch_size, output_features,
                delta_scratch.as<Scalar>(),
                bias_gradient.as<float>());

            gather_time_slice_cuda<Scalar>(batch_size, time_steps, input_features, t,
                                           input.as<Scalar>(), step_input_scratch.as<Scalar>());

            multiply(step_input_scratch, true, delta_scratch, false,
                     const_cast<TensorView&>(input_weight_gradient), 1.0f, 1.0f);

            if (t > 0)
            {
                gather_time_slice_cuda<Scalar>(batch_size, time_steps, output_features, t - 1,
                                               hidden_states.as<Scalar>(), step_prev_h_scratch.as<Scalar>());

                multiply(step_prev_h_scratch, true, delta_scratch, false,
                         const_cast<TensorView&>(recurrent_weight_gradient), 1.0f, 1.0f);

                multiply(delta_scratch, false, recurrent_weights, true,
                         next_carry_scratch, 1.0f, 0.0f);
            }

            if (input_delta.get_data() && !input_delta.empty())
            {
                multiply(delta_scratch, false, input_weights, true,
                         step_in_delta_scratch, 1.0f, 0.0f);

                scatter_time_slice_cuda<Scalar>(batch_size, time_steps, input_features, t,
                                                step_in_delta_scratch.as<Scalar>(), input_delta.as<Scalar>());
            }
        }
    });
}

#else

void RecurrentOperator::apply_gpu(const TensorView&, TensorView&, TensorView&, TensorView&,
                                  TensorView&, TensorView&, TensorView&, TensorView&,
                                  Buffer&, bool) const OPENNN_CUDA_STUB_BODY(apply_gpu)

void RecurrentOperator::apply_delta_gpu(const TensorView&, const TensorView&, const TensorView&,
                                  const TensorView&, TensorView&,
                                  TensorView&, TensorView&, TensorView&,
                                  TensorView&, TensorView&, TensorView&, TensorView&,
                                  const Buffer&, Buffer&) const OPENNN_CUDA_STUB_BODY(apply_delta_gpu)

#endif

Recurrent::Recurrent(const Shape& new_input_shape,
                     const Shape& new_output_shape,
                     const string& new_activation_function,
                     const string& new_label)
    : Layer(LayerType::Recurrent)
{
    operators = {&recurrent_op};
    set(new_input_shape, new_output_shape, new_activation_function, new_label);
}

vector<TensorSpec> Recurrent::get_forward_specs(Index batch_size) const
{
    const Shape state_history {batch_size, time_steps, output_features};
    const Shape step_input {batch_size, input_features};
    const Shape step_hidden {batch_size, output_features};

    return {
        {state_history, compute_dtype},
        {state_history, compute_dtype},
        {step_input, compute_dtype},
        {step_hidden, compute_dtype},
        {step_hidden, compute_dtype},
        {step_hidden, compute_dtype},
        {return_sequences ? state_history : Shape{batch_size, output_features}, compute_dtype},
    };
}

vector<TensorSpec> Recurrent::get_backward_specs(Index batch_size) const
{
    if (!is_trainable) return {};

    const Shape step_in_shape  {batch_size, input_features};
    const Shape step_out_shape {batch_size, output_features};

    return {
        {Shape{batch_size}.append(get_input_shape()), compute_dtype},
        {step_in_shape,     compute_dtype},
        {step_out_shape,    compute_dtype},
        {step_out_shape,    compute_dtype},
        {step_out_shape,    compute_dtype},
        {step_in_shape,     compute_dtype},
        {{batch_size, time_steps, output_features}, compute_dtype},
        {Shape{batch_size}.append(get_input_shape()), compute_dtype},
    };
}

void Recurrent::configure_operators()
{
    recurrent_op.set(input_features, time_steps, output_features,
                     recurrent_op.activation, compute_dtype);

    recurrent_op.return_sequences = return_sequences;
    recurrent_op.input_slots  = {Input};
    recurrent_op.output_slots = {Output, HiddenStates, ActivationDerivatives};
}

void Recurrent::set_return_sequences(bool value)
{
    if (return_sequences == value) return;
    return_sequences = value;
    configure_operators();
}

void Recurrent::set(const Shape& new_input_shape,
                    const Shape& new_output_shape,
                    const string& new_activation_function,
                    const string& new_label)
{
    if (new_input_shape.empty() && new_output_shape.empty())
    {
        time_steps      = 0;
        input_features  = 0;
        output_features = 0;
        return;
    }

    check_rank(new_input_shape,  {2}, "Recurrent", "input");
    check_rank(new_output_shape, {1}, "Recurrent", "output");

    time_steps      = new_input_shape[0];
    input_features  = new_input_shape[1];
    output_features = new_output_shape[0];

    set_activation_function(new_activation_function);
    set_label(new_label);

    configure_operators();
}

void Recurrent::apply_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {2}, "Recurrent", "input");
    time_steps     = new_input_shape[0];
    input_features = new_input_shape[1];
    configure_operators();
}

void Recurrent::set_output_shape(const Shape& new_output_shape)
{
    check_rank(new_output_shape, {1, 2}, "Recurrent", "output");
    output_features = new_output_shape[new_output_shape.get_rank() - 1];
    configure_operators();
}

void Recurrent::set_activation_function(const string& name)
{
    const ActivationFunction fn = ActivationOperator::from_string(name);
    throw_if(!supports_recurrent_activation(fn),
             "Recurrent: unsupported activation (use Tanh, Sigmoid, ReLU or Identity).");
    recurrent_op.activation = fn;
}

void Recurrent::read_JSON_body(const Json* recurrent_layer_element)
{
    set_activation_function(read_json_string(recurrent_layer_element, "Activation"));
    return_sequences = read_json_bool(recurrent_layer_element, "ReturnSequences");
    configure_operators();
}

void Recurrent::write_JSON_body(JsonWriter& printer) const
{
    add_json_field(printer, "Activation", get_activation_function());
    add_json_field(printer, "ReturnSequences", return_sequences);
}

string Recurrent::write_expression(const vector<string>& feature_names,
                                   const vector<string>& output_names) const
{
    if (parameters.size() < 3 || !parameters[0].get_data() || !parameters[1].get_data() || !parameters[2].get_data())
        return {};

    VectorMap biases_map        = parameters[0].as_vector();
    MatrixMap input_w_map       = parameters[1].as_matrix();
    MatrixMap recurrent_w_map   = parameters[2].as_matrix();

    const string& activation_name = ActivationOperator::to_string(recurrent_op.activation);

    const auto step_var = [&](Index t, Index j) -> string {
        const string internal = format("recurrent_hidden_step_{}_neuron_{}", t, j);
        if (return_sequences)
        {
            const Index linear = t * output_features + j;
            if (linear < ssize(output_names)) return output_names[linear];
            return internal;
        }
        if (t == time_steps - 1)
        {
            if (j < ssize(output_names)) return output_names[j];
            return format("recurrent_output_{}", j);
        }
        return internal;
    };

    ostringstream buffer;
    buffer.precision(10);

    for (Index time_step = 0; time_step < time_steps; ++time_step)
    {
        for (Index j = 0; j < output_features; ++j)
        {
            const string current_var = step_var(time_step, j);
            buffer << current_var << " = " << activation_name << "( " << biases_map(j);

            for (Index i = 0; i < input_features; ++i)
            {
                const Index feature_index = time_step * input_features + i;
                if (feature_index < ssize(feature_names))
                    buffer << " + (" << feature_names[feature_index] << "*" << input_w_map(i, j) << ")";
            }

            if (time_step > 0)
                for (Index prev_j = 0; prev_j < output_features; ++prev_j)
                    buffer << " + (" << step_var(time_step - 1, prev_j)
                           << "*" << recurrent_w_map(prev_j, j) << ")";

            buffer << " );\n";
        }
    }

    return buffer.str();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
