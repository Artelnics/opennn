//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "recurrent_operator.h"
#include "device_backend.h"
#include "json.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "string_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include "profiler.h"

namespace opennn
{

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
        {{output_features},                   weight_type},
        {{input_features, output_features},   weight_type},
        {{output_features, output_features},  weight_type},
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

void RecurrentOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool is_training)
{
    auto& forward_slots = fp.forward_slots[layer];
    const TensorView& input             = get_input(fp, layer);
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

void RecurrentOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const
{
    auto& forward_slots = fp.forward_slots[layer];
    auto& backward_slots = bp.backward_slots[layer];

    const TensorView& input                    = get_input(fp, layer);
    const TensorView& hidden_states            = forward_slots[output_slots[1]];
    const TensorView& activation_derivatives   = forward_slots[output_slots[2]];
    const TensorView& output_delta             = get_output_delta(bp, layer);

    TensorView empty;
    TensorView& input_delta = view_at_slot_or(backward_slots, input_delta_slots, 0, empty);

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

    const Index in_stride_t = input_features;
    const Index in_stride_b = time_steps * input_features;
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
            if (is_training)
                step_derivs.setOnes();
            break;
        case F::Softmax:
        case F::LeakyReLU: throw runtime_error("RecurrentOp: unsupported activation.");
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

    if (return_sequences)
    {
        std::memcpy(output.as<float>(), hidden_data,
                    batch_size * time_steps * output_features * sizeof(float));
    }
    else
    {
        output.as_matrix() = prev_hidden;
    }
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

    const bool write_input_delta = !input_delta.empty() && input_delta.data != nullptr;
    float* input_delta_data = write_input_delta ? input_delta.as<float>() : nullptr;

    const Index in_stride_t = input_features;
    const Index in_stride_b = time_steps * input_features;
    const Index h_stride_t  = output_features;
    const Index h_stride_b  = time_steps * output_features;

    const float* seq_delta_data = return_sequences ? output_delta.as<float>()
                                                   : nullptr;
    const float* final_delta_data = return_sequences ? nullptr
                                                     : output_delta.as<float>();

    MatrixR delta        (batch_size, output_features);
    MatrixR next_carry   = MatrixR::Zero(batch_size, output_features);
    MatrixR step_input   (batch_size, input_features);
    MatrixR step_prev_h  (batch_size, output_features);
    MatrixR step_derivs  (batch_size, output_features);
    MatrixR step_in_delta(batch_size, input_features);
    MatrixR step_out_delta(batch_size, output_features);

    for (Index t = time_steps - 1; t >= 0; --t)
    {
        if (return_sequences)
        {
            for (Index i = 0; i < batch_size; ++i)
                std::memcpy(step_out_delta.data() + i * output_features,
                            seq_delta_data + i * h_stride_b + t * h_stride_t,
                            output_features * sizeof(float));
            delta = next_carry + step_out_delta;
        }
        else if (t == time_steps - 1)
        {
            delta = Eigen::Map<const MatrixR>(final_delta_data, batch_size, output_features);
        }
        else
        {
            delta = next_carry;
        }

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

#ifdef OPENNN_HAS_CUDA

namespace
{

void zero_device_view(const TensorView& view)
{
    if (!view.data || view.empty()) return;
    device::set_zero_async(view.data, view.byte_size(), Backend::get_compute_stream());
}

void require_same_recurrent_dtype(const TensorView& reference,
                                  initializer_list<pair<const TensorView*, const char*>> views)
{
    for (const auto& [view, name] : views)
        throw_if(view->data && !view->empty() && view->type != reference.type,
                 format("RecurrentOp CUDA: {} dtype does not match recurrent compute dtype.", name));
}

}

void RecurrentOp::apply_gpu(const TensorView& input,
                            TensorView& hidden_states,
                            TensorView& activation_derivatives,
                            TensorView& output,
                            bool is_training)
{
    if (!input.data || output_features == 0 || time_steps == 0) return;

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
        {
            copy(hidden_states, output);
        }
        else
        {
            TensorView final_hidden(prev_hidden_buf.data, step_hidden_shape, output.type, Device::CUDA);
            copy(final_hidden, output);
        }
    });
}

void RecurrentOp::apply_delta_gpu(const TensorView& input,
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

        const cudaDataType_t axpy_dtype =
            (output_delta.type == Type::FP32) ? CUDA_R_32F :
            (output_delta.type == Type::BF16) ? CUDA_R_16BF :
                                                 CUDA_R_32F;

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

            rnn_accumulate_bias_grad_cuda<Scalar>(
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

void RecurrentOp::apply_gpu(const TensorView&, TensorView&, TensorView&, TensorView&, bool)
{
    throw runtime_error("RecurrentOp::apply_gpu: CUDA support not compiled in.");
}

void RecurrentOp::apply_delta_gpu(const TensorView&, const TensorView&, const TensorView&,
                                  const TensorView&, TensorView&,
                                  TensorView&, TensorView&, TensorView&,
                                  TensorView&, TensorView&) const
{
    throw runtime_error("RecurrentOp::apply_delta_gpu: CUDA support not compiled in.");
}

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
