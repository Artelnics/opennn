//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "recurrent_layer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

#include "device_backend.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "kernel.cuh"

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
    {
        apply_gpu(input, hidden_states, activation_derivatives, output, is_training);
        return;
    }
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

    TensorView& input_delta = slot_or(backward_slots, input_delta_slots, 0);

    if (output_delta.is_cuda())
    {
        TensorView& step_input_scratch    = backward_slots[StepInputScratchSlot];
        TensorView& step_prev_h_scratch   = backward_slots[StepPrevHScratchSlot];
        TensorView& delta_scratch         = backward_slots[DeltaScratchSlot];
        TensorView& next_carry_scratch    = backward_slots[NextCarryScratchSlot];
        TensorView& step_in_delta_scratch = backward_slots[StepInDeltaScratchSlot];
        apply_delta_gpu(input, hidden_states, activation_derivatives,
                        output_delta, input_delta,
                        step_input_scratch, step_prev_h_scratch,
                        delta_scratch, next_carry_scratch, step_in_delta_scratch);
        return;
    }
    apply_delta(input, hidden_states, activation_derivatives, output_delta, input_delta);
}

namespace
{

using StridedMap      = Eigen::Map<MatrixR, 0, Eigen::OuterStride<>>;
using ConstStridedMap = Eigen::Map<const MatrixR, 0, Eigen::OuterStride<>>;

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
    default:
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
    const Index batch_size = input.shape[0];
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

    const bool write_input_delta = !input_delta.empty() && input_delta.data != nullptr;
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
    if (!view.data || view.empty()) return;
    device::set_zero_async(view.data, view.byte_size(), Backend::get_compute_stream());
}

static void require_same_recurrent_dtype(const TensorView& reference,
                                         initializer_list<pair<const TensorView*, const char*>> views)
{
    for (const auto& [view, name] : views)
        throw_if(view->data && !view->empty() && view->type != reference.type,
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
                                                   : CUDNN_RNN_TANH,
            2, "OPENNN_RNN_PERSIST"};
}

void RecurrentOperator::ensure_cudnn_setup_(Index batch_size, bool for_training) const
{
    cudnn_setup_(recurrent_cudnn_config(activation),
                 input_features, output_features, time_steps,
                 batch_size, for_training);
}

void RecurrentOperator::pack_weights_to_cudnn_() const
{
    const TensorView* weights[2] = {&input_weights, &recurrent_weights};
    const TensorView* biases[2]  = {&bias, nullptr};
    cudnn_pack_weights_(2, input_features, output_features, weights, biases);
}

void RecurrentOperator::unpack_gradients_from_cudnn_() const
{
    const TensorView* weight_gradients[2] = {&input_weight_gradient, &recurrent_weight_gradient};
    const TensorView* bias_gradients[2]   = {&bias_gradient, nullptr};
    cudnn_unpack_gradients_(2, input_features, output_features, weight_gradients, bias_gradients);
}

void RecurrentOperator::apply_gpu_cudnn_(const TensorView& input,
                                         TensorView& hidden_states,
                                         TensorView& output,
                                         bool is_training) const
{
    const Index batch_size = input.shape[0];

    ensure_cudnn_setup_(batch_size, is_training);
    pack_weights_to_cudnn_();

    cudnn_rnn_forward_(is_training,  false,
                       input.data, hidden_states.data,
                       [&] {
                           ensure_cudnn_setup_(batch_size, is_training);
                           pack_weights_to_cudnn_();
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
                                               TensorView& input_delta) const
{
    const Index batch_size = input.shape[0];
    const Index H = output_features;
    const Index T = time_steps;

    ensure_cudnn_setup_(batch_size, true);

    const float* dy_data = output_delta.as<float>();
    if (!return_sequences)
    {
        scatter_time_slice_fill_cuda(
            batch_size, T, H, T - 1,
            output_delta.as<float>(),
            static_cast<float*>(dy_buf.data));
        dy_data = static_cast<const float*>(dy_buf.data);
    }

    void* dx_data = input_delta.data;
    if (!dx_data || input_delta.empty())
    {
        dx_scratch_buf.grow_to(batch_size * T * input_features * Index(sizeof(float)));
        dx_data = dx_scratch_buf.data;
    }

    cudnn_rnn_backward_( false,
                        input.data, hidden_states.data, dy_data, dx_data);

    unpack_gradients_from_cudnn_();
}

void RecurrentOperator::apply_gpu(const TensorView& input,
                            TensorView& hidden_states,
                            TensorView& activation_derivatives,
                            TensorView& output,
                            bool is_training) const
{
    if (!input.data || output_features == 0 || time_steps == 0) return;

    if (cudnn_rnn_eligible_(output))
    {
        apply_gpu_cudnn_(input, hidden_states, output, is_training);
        return;
    }

    require_same_recurrent_dtype(output, {
        {&input, "input"},
        {&hidden_states, "hidden_states"},
        {&activation_derivatives, "activation_derivatives"},
        {&bias, "bias"},
        {&input_weights, "input_weights"},
        {&recurrent_weights, "recurrent_weights"}
    });

    output.dispatch([&](auto tag)
    {
        using Scalar = decltype(tag);

        const Index batch_size = input.shape[0];
        const Shape step_input_shape{batch_size, input_features};
        const Shape step_hidden_shape{batch_size, output_features};

        step_input_buf.grow_to(batch_size * input_features * Index(sizeof(Scalar)));
        step_hidden_buf.grow_to(batch_size * output_features * Index(sizeof(Scalar)));
        prev_hidden_buf.grow_to(batch_size * output_features * Index(sizeof(Scalar)));

        if (is_training && !activation_derivatives.empty())
            step_derivs_buf.grow_to(batch_size * output_features * Index(sizeof(Scalar)));

        for (Index t = 0; t < time_steps; ++t)
        {
            TensorView step_input(step_input_buf.data, step_input_shape, input.type, Device::CUDA);
            TensorView step_hidden(step_hidden_buf.data, step_hidden_shape, output.type, Device::CUDA);

            gather_time_slice_cuda<Scalar>(batch_size, time_steps, input_features, t,
                                           input.as<Scalar>(), step_input.as<Scalar>());

            const Scalar* prev_h_ptr = (t > 0)
                ? static_cast<const Scalar*>(prev_hidden_buf.data)
                : nullptr;

            Scalar* derivs = nullptr;
            TensorView step_derivs;
            if (is_training && !activation_derivatives.empty())
            {
                step_derivs = TensorView(step_derivs_buf.data, step_hidden_shape, output.type, Device::CUDA);
                derivs = step_derivs.as<Scalar>();
            }

            rnn_step_fused_forward_cuda<Scalar>(batch_size,
                                                input_features,
                                                output_features,
                                                step_input.as<Scalar>(),
                                                prev_h_ptr,
                                                input_weights.as<Scalar>(),
                                                recurrent_weights.as<Scalar>(),
                                                bias.as<Scalar>(),
                                                step_hidden.as<Scalar>(),
                                                derivs,
                                                static_cast<int>(activation));

            scatter_time_slice_cuda<Scalar>(batch_size, time_steps, output_features, t,
                                            step_hidden.as<Scalar>(), hidden_states.as<Scalar>());

            if (derivs)
                scatter_time_slice_cuda<Scalar>(batch_size, time_steps, output_features, t,
                                                step_derivs.as<Scalar>(), activation_derivatives.as<Scalar>());

            step_hidden_buf.swap(prev_hidden_buf);
        }

        if (return_sequences)
            copy(hidden_states, output);
        else
        {
            TensorView final_hidden(prev_hidden_buf.data, step_hidden_shape, output.type, Device::CUDA);
            copy(final_hidden, output);
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
                                  TensorView& step_in_delta_scratch) const
{
    if (!input.data || !output_delta.data || output_features == 0 || time_steps == 0) return;

    if (cudnn_rnn_eligible_(output_delta))
    {
        apply_delta_gpu_cudnn_(input, hidden_states, output_delta, input_delta);
        return;
    }

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

    output_delta.dispatch([&](auto tag)
    {
        using Scalar = decltype(tag);

        const Index batch_size = input.shape[0];

        zero_device_view(bias_gradient);
        zero_device_view(input_weight_gradient);
        zero_device_view(recurrent_weight_gradient);

        if (return_sequences)
            step_seq_delta_buf.grow_to(batch_size * output_features *
                                       Index(sizeof(Scalar)));

        const cudaDataType_t axpy_dtype = output_delta.cuda_dtype();

        for (Index t = time_steps; t-- > 0;)
        {
            const bool first_iter = (t == time_steps - 1);

            const Scalar* delta_src = nullptr;
            const Scalar* carry_src = nullptr;
            bool kernel_first_iter  = first_iter;

            if (return_sequences)
            {
                gather_time_slice_cuda<Scalar>(batch_size, time_steps,
                                               output_features, t,
                                               output_delta.as<Scalar>(),
                                               static_cast<Scalar*>(step_seq_delta_buf.data));
                if (!first_iter)
                {
                    const float alpha = 1.0f;
                    const int   n     = to_int(batch_size * output_features);
                    CHECK_CUBLAS(cublasAxpyEx(Backend::get_cublas_handle(), n,
                                              &alpha, CUDA_R_32F,
                                              next_carry_scratch.data, axpy_dtype, 1,
                                              step_seq_delta_buf.data, axpy_dtype, 1,
                                              CUDA_R_32F));
                }
                delta_src        = static_cast<const Scalar*>(step_seq_delta_buf.data);
                carry_src        = nullptr;
                kernel_first_iter = true;
            }
            else
            {
                delta_src = first_iter ? output_delta.as<Scalar>() : nullptr;
                carry_src = first_iter ? nullptr : next_carry_scratch.as<Scalar>();
            }

            rnn_step_fused_backward_pre_cuda<Scalar>(
                batch_size, output_features, time_steps, t, kernel_first_iter,
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

            if (input_delta.data && !input_delta.empty())
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

void RecurrentOperator::apply_gpu(const TensorView&, TensorView&, TensorView&, TensorView&, bool) const OPENNN_CUDA_STUB_BODY(apply_gpu)

void RecurrentOperator::apply_delta_gpu(const TensorView&, const TensorView&, const TensorView&,
                                  const TensorView&, TensorView&,
                                  TensorView&, TensorView&, TensorView&,
                                  TensorView&, TensorView&) const OPENNN_CUDA_STUB_BODY(apply_delta_gpu)

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

    return {
        {state_history, compute_dtype},
        {state_history, compute_dtype},
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
    output_features = new_output_shape[new_output_shape.rank - 1];
    configure_operators();
}

void Recurrent::set_activation_function(const string& name)
{
    const ActivationFunction fn = ActivationOperator::from_string(name);
    using enum ActivationFunction;
    throw_if(fn != Identity && fn != Sigmoid && fn != Tanh && fn != ReLU,
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
    if (parameters.size() < 3 || !parameters[0].data || !parameters[1].data || !parameters[2].data)
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
