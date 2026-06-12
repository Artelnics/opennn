//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "long_short_term_memory_operator.h"
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
        {bias_shape, Type::FP32},
        {bias_shape, Type::FP32},
        {bias_shape, Type::FP32},
        {bias_shape, Type::FP32},
        {input_weight_shape, Type::FP32},
        {input_weight_shape, Type::FP32},
        {input_weight_shape, Type::FP32},
        {input_weight_shape, Type::FP32},
        {recurrent_weight_shape, Type::FP32},
        {recurrent_weight_shape, Type::FP32},
        {recurrent_weight_shape, Type::FP32},
        {recurrent_weight_shape, Type::FP32},
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
    if (forget_bias.data) forget_bias.fill(1.0f);
    zero_if_linked(input_bias);
    zero_if_linked(candidate_bias);
    zero_if_linked(output_bias);

    if (forget_weights.data)
    {
        set_random_uniform(forget_weights.as_vector(),    -0.1f, 0.1f);
        set_random_uniform(input_weights.as_vector(),     -0.1f, 0.1f);
        set_random_uniform(candidate_weights.as_vector(), -0.1f, 0.1f);
        set_random_uniform(output_weights.as_vector(),    -0.1f, 0.1f);
    }

    if (forget_recurrent_weights.data)
    {
        set_random_uniform(forget_recurrent_weights.as_vector(),    -0.1f, 0.1f);
        set_random_uniform(input_recurrent_weights.as_vector(),     -0.1f, 0.1f);
        set_random_uniform(candidate_recurrent_weights.as_vector(), -0.1f, 0.1f);
        set_random_uniform(output_recurrent_weights.as_vector(),    -0.1f, 0.1f);
    }
}

void LongShortTermMemoryOp::set_parameters_glorot()
{
    if (forget_bias.data) forget_bias.fill(1.0f);
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

void LongShortTermMemoryOp::forward_propagate(ForwardPropagation& fp, size_t layer, bool)
{
    auto& forward_slots = fp.forward_slots[layer];

    TensorView& input = get_input(fp, layer);
    TensorView& output = forward_slots[OutputSlot];
    TensorView& forget_gate = forward_slots[ForgetGateSlot];
    TensorView& input_gate = forward_slots[InputGateSlot];
    TensorView& candidate_gate = forward_slots[CandidateGateSlot];
    TensorView& output_gate = forward_slots[OutputGateSlot];
    TensorView& cell_state = forward_slots[CellStateSlot];
    TensorView& hidden_state = forward_slots[HiddenStateSlot];
    TensorView& cell_activation = forward_slots[CellActivationSlot];

    if (input.is_cuda())
    {
        apply_gpu(input, output, return_sequences);
        return;
    }

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

    static const bool lstm_scalar = (std::getenv("OPENNN_LSTM_SCALAR") != nullptr);
    if (!lstm_scalar && H >= 64)
    {
        const MatrixMap Wf_m = forget_weights.as_matrix();
        const MatrixMap Wi_m = input_weights.as_matrix();
        const MatrixMap Wg_m = candidate_weights.as_matrix();
        const MatrixMap Wo_m = output_weights.as_matrix();
        const MatrixMap Uf_m = forget_recurrent_weights.as_matrix();
        const MatrixMap Ui_m = input_recurrent_weights.as_matrix();
        const MatrixMap Ug_m = candidate_recurrent_weights.as_matrix();
        const MatrixMap Uo_m = output_recurrent_weights.as_matrix();
        const VectorMap bf_m = forget_bias.as_vector();
        const VectorMap bi_m = input_bias.as_vector();
        const VectorMap bg_m = candidate_bias.as_vector();
        const VectorMap bo_m = output_bias.as_vector();

        MatrixR step_in(batch_size, F);
        MatrixR prev_h(batch_size, H);
        MatrixR Zf(batch_size, H), Zi(batch_size, H), Zg(batch_size, H), Zo(batch_size, H);

        for (Index t = 0; t < T; ++t)
        {
            for (Index b = 0; b < batch_size; ++b)
                std::memcpy(step_in.data() + b * F, x + (b * T + t) * F, F * sizeof(float));

            Zf.noalias() = step_in * Wf_m;  Zf.rowwise() += bf_m.transpose();
            Zi.noalias() = step_in * Wi_m;  Zi.rowwise() += bi_m.transpose();
            Zg.noalias() = step_in * Wg_m;  Zg.rowwise() += bg_m.transpose();
            Zo.noalias() = step_in * Wo_m;  Zo.rowwise() += bo_m.transpose();

            if (t > 0)
            {
                Zf.noalias() += prev_h * Uf_m;
                Zi.noalias() += prev_h * Ui_m;
                Zg.noalias() += prev_h * Ug_m;
                Zo.noalias() += prev_h * Uo_m;
            }

            for (Index b = 0; b < batch_size; ++b)
            {
                const Index step = (b * T + t) * H;
                const float* c_prev = t > 0 ? cells + (b * T + t - 1) * H : nullptr;

                for (Index h = 0; h < H; ++h)
                {
                    const float f = lstm_activate(recurrent_activation_function, Zf(b, h));
                    const float i = lstm_activate(recurrent_activation_function, Zi(b, h));
                    const float g = lstm_activate(activation_function, Zg(b, h));
                    const float o = lstm_activate(recurrent_activation_function, Zo(b, h));
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
                    if (return_sequences) y[step + h] = h_value;
                }
            }

            for (Index b = 0; b < batch_size; ++b)
                std::memcpy(prev_h.data() + b * H, hidden + (b * T + t) * H, H * sizeof(float));
        }

        if (!return_sequences)
            for (Index b = 0; b < batch_size; ++b)
                copy_n(hidden + (b * T + T - 1) * H, H, y + b * H);

        return;
    }

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

                if (return_sequences)
                    y[step + h] = h_value;
            }
        }

        if (!return_sequences)
            copy_n(hidden + (b * T + T - 1) * H, H, y + b * H);
    }
}

void LongShortTermMemoryOp::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const
{
    auto& backward_slots = bp.backward_slots[layer];
    if (backward_slots.size() <= OutputDeltaScratchSlot) return;

    const auto& forward_slots = fp.forward_slots[layer];

    TensorView& input_delta = backward_slots[InputDeltaSlot];
    TensorView& hidden_delta = backward_slots[HiddenDeltaScratchSlot];
    TensorView& cell_delta = backward_slots[CellDeltaScratchSlot];
    TensorView& forget_delta = backward_slots[ForgetDeltaScratchSlot];
    TensorView& input_gate_delta = backward_slots[InputDeltaScratchSlot];
    TensorView& candidate_delta = backward_slots[CandidateDeltaScratchSlot];
    TensorView& output_gate_delta = backward_slots[OutputDeltaScratchSlot];

    const TensorView& input = get_input(fp, layer);
    const TensorView& output_delta = get_output_delta(bp, layer);
    const TensorView& forget_gate = forward_slots[ForgetGateSlot];
    const TensorView& input_gate = forward_slots[InputGateSlot];
    const TensorView& candidate_gate = forward_slots[CandidateGateSlot];
    const TensorView& output_gate = forward_slots[OutputGateSlot];
    const TensorView& cell_state = forward_slots[CellStateSlot];
    const TensorView& hidden_state = forward_slots[HiddenStateSlot];
    const TensorView& cell_activation = forward_slots[CellActivationSlot];

    if (input.is_cuda())
    {
        apply_delta_gpu(input, output_delta, input_delta, return_sequences);
        return;
    }

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
    const bool write_input_delta = !input_delta.empty();
    float* in_delta = write_input_delta ? input_delta.as<float>() : nullptr;

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

    static const bool lstm_scalar_bwd = (std::getenv("OPENNN_LSTM_SCALAR") != nullptr);
    if (!lstm_scalar_bwd && H >= 64)
    {
        const MatrixMap Wf_m = forget_weights.as_matrix();
        const MatrixMap Wi_m = input_weights.as_matrix();
        const MatrixMap Wg_m = candidate_weights.as_matrix();
        const MatrixMap Wo_m = output_weights.as_matrix();
        const MatrixMap Uf_m = forget_recurrent_weights.as_matrix();
        const MatrixMap Ui_m = input_recurrent_weights.as_matrix();
        const MatrixMap Ug_m = candidate_recurrent_weights.as_matrix();
        const MatrixMap Uo_m = output_recurrent_weights.as_matrix();

        MatrixMap gWf_m = forget_weight_gradient.as_matrix();
        MatrixMap gWi_m = input_weight_gradient.as_matrix();
        MatrixMap gWg_m = candidate_weight_gradient.as_matrix();
        MatrixMap gWo_m = output_weight_gradient.as_matrix();
        MatrixMap gUf_m = forget_recurrent_weight_gradient.as_matrix();
        MatrixMap gUi_m = input_recurrent_weight_gradient.as_matrix();
        MatrixMap gUg_m = candidate_recurrent_weight_gradient.as_matrix();
        MatrixMap gUo_m = output_recurrent_weight_gradient.as_matrix();
        VectorMap gbf_v = forget_bias_gradient.as_vector();
        VectorMap gbi_v = input_bias_gradient.as_vector();
        VectorMap gbg_v = candidate_bias_gradient.as_vector();
        VectorMap gbo_v = output_bias_gradient.as_vector();

        MatrixR step_in(batch_size, F), prev_h(batch_size, H), c_prev_m(batch_size, H);
        MatrixR DF(batch_size, H), DI(batch_size, H), DG(batch_size, H), DO(batch_size, H);
        MatrixR DX(batch_size, F);
        MatrixR dh_next(batch_size, H), dc_next(batch_size, H);
        dh_next.setZero();
        dc_next.setZero();

        for (Index t = T; t-- > 0;)
        {
            for (Index b = 0; b < batch_size; ++b)
                std::memcpy(step_in.data() + b * F, x + (b * T + t) * F, F * sizeof(float));

            if (t > 0)
                for (Index b = 0; b < batch_size; ++b)
                {
                    std::memcpy(prev_h.data()   + b * H, hidden + (b * T + t - 1) * H, H * sizeof(float));
                    std::memcpy(c_prev_m.data() + b * H, cells  + (b * T + t - 1) * H, H * sizeof(float));
                }

            if (return_sequences)
                for (Index b = 0; b < batch_size; ++b)
                    for (Index h = 0; h < H; ++h)
                        dh_next(b, h) += out_delta[(b * T + t) * H + h];
            else if (t == T - 1)
                for (Index b = 0; b < batch_size; ++b)
                    for (Index h = 0; h < H; ++h)
                        dh_next(b, h) += out_delta[b * H + h];

            for (Index b = 0; b < batch_size; ++b)
            {
                const Index step = (b * T + t) * H;
                for (Index h = 0; h < H; ++h)
                {
                    const float f = f_gate[step + h];
                    const float i = i_gate[step + h];
                    const float g = g_gate[step + h];
                    const float o = o_gate[step + h];
                    const float a = cell_act[step + h];

                    const float dc = dh_next(b, h) * o * lstm_derivative_from_output(activation_function, a) + dc_next(b, h);

                    DO(b, h) = dh_next(b, h) * a * lstm_derivative_from_output(recurrent_activation_function, o);
                    DF(b, h) = dc * (t > 0 ? c_prev_m(b, h) : 0.0f) * lstm_derivative_from_output(recurrent_activation_function, f);
                    DI(b, h) = dc * g * lstm_derivative_from_output(recurrent_activation_function, i);
                    DG(b, h) = dc * i * lstm_derivative_from_output(activation_function, g);
                    dc_next(b, h) = dc * f;
                }
            }

            gbf_v += DF.colwise().sum().transpose();
            gbi_v += DI.colwise().sum().transpose();
            gbg_v += DG.colwise().sum().transpose();
            gbo_v += DO.colwise().sum().transpose();

            gWf_m.noalias() += step_in.transpose() * DF;
            gWi_m.noalias() += step_in.transpose() * DI;
            gWg_m.noalias() += step_in.transpose() * DG;
            gWo_m.noalias() += step_in.transpose() * DO;

            DX.noalias()  = DF * Wf_m.transpose();
            DX.noalias() += DI * Wi_m.transpose();
            DX.noalias() += DG * Wg_m.transpose();
            DX.noalias() += DO * Wo_m.transpose();
            for (Index b = 0; b < batch_size; ++b)
                std::memcpy(in_delta + (b * T + t) * F, DX.data() + b * F, F * sizeof(float));

            if (t > 0)
            {
                gUf_m.noalias() += prev_h.transpose() * DF;
                gUi_m.noalias() += prev_h.transpose() * DI;
                gUg_m.noalias() += prev_h.transpose() * DG;
                gUo_m.noalias() += prev_h.transpose() * DO;

                dh_next.noalias()  = DF * Uf_m.transpose();
                dh_next.noalias() += DI * Ui_m.transpose();
                dh_next.noalias() += DG * Ug_m.transpose();
                dh_next.noalias() += DO * Uo_m.transpose();
            }
        }

        return;
    }

    float* dh_next_all = hidden_delta_scratch.as<float>();
    float* dc_next_all = cell_delta_scratch.as<float>();
    float* df_all = forget_delta_scratch.as<float>();
    float* di_all = input_delta_scratch.as<float>();
    float* dg_all = candidate_delta_scratch.as<float>();
    float* do_all = output_delta_scratch.as<float>();

    const int nthreads = omp_get_max_threads();

    const Index bias_sz = 4 * H;
    const Index w_sz    = 4 * F * H;
    const Index u_sz    = 4 * H * H;
    const Index per_thread_sz = bias_sz + w_sz + u_sz;

    grad_tls_buf_.assign(size_t(nthreads) * size_t(per_thread_sz), 0.0f);

    #pragma omp parallel for
    for (Index b = 0; b < batch_size; ++b)
    {
        const int tid = omp_get_thread_num();
        float* tls_base = grad_tls_buf_.data() + size_t(tid) * size_t(per_thread_sz);

        float* tls_gbf = tls_base;
        float* tls_gbi = tls_gbf + H;
        float* tls_gbg = tls_gbi + H;
        float* tls_gbo = tls_gbg + H;

        float* tls_gWf = tls_base + bias_sz;
        float* tls_gWi = tls_gWf + F * H;
        float* tls_gWg = tls_gWi + F * H;
        float* tls_gWo = tls_gWg + F * H;

        float* tls_gUf = tls_base + bias_sz + w_sz;
        float* tls_gUi = tls_gUf + H * H;
        float* tls_gUg = tls_gUi + H * H;
        float* tls_gUo = tls_gUg + H * H;

        float* dh_next = dh_next_all + b * H;
        float* dc_next = dc_next_all + b * H;
        float* df = df_all + b * H;
        float* di = di_all + b * H;
        float* dg = dg_all + b * H;
        float* do_gate = do_all + b * H;

        fill_n(dh_next, H, 0.0f);
        fill_n(dc_next, H, 0.0f);

        for (Index t = T; t-- > 0;)
        {
            const Index step = (b * T + t) * H;
            const float* xt = x + (b * T + t) * F;
            const float* h_prev = t > 0 ? hidden + (b * T + t - 1) * H : nullptr;
            const float* c_prev = t > 0 ? cells + (b * T + t - 1) * H : nullptr;

            if (return_sequences)
                for (Index h = 0; h < H; ++h) dh_next[h] += out_delta[step + h];
            else if (t == T - 1)
                for (Index h = 0; h < H; ++h) dh_next[h] += out_delta[b * H + h];

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

                tls_gbf[h] += df[h];
                tls_gbi[h] += di[h];
                tls_gbg[h] += dg[h];
                tls_gbo[h] += do_gate[h];
            }

            for (Index k = 0; k < F; ++k)
            {
                float dx = 0.0f;
                const float xk = xt[k];

                for (Index h = 0; h < H; ++h)
                {
                    const Index wh = k * H + h;
                    tls_gWf[wh] += xk * df[h];
                    tls_gWi[wh] += xk * di[h];
                    tls_gWg[wh] += xk * dg[h];
                    tls_gWo[wh] += xk * do_gate[h];

                    if (write_input_delta)
                        dx += df[h] * Wf[wh]
                            + di[h] * Wi[wh]
                            + dg[h] * Wg[wh]
                            + do_gate[h] * Wo[wh];
                }

                if (write_input_delta)
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
                        tls_gUf[uh] += hp * df[h];
                        tls_gUi[uh] += hp * di[h];
                        tls_gUg[uh] += hp * dg[h];
                        tls_gUo[uh] += hp * do_gate[h];
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

    for (int tid = 0; tid < nthreads; ++tid)
    {
        const float* base = grad_tls_buf_.data() + size_t(tid) * size_t(per_thread_sz);

        const float* t_gbf = base;
        const float* t_gbi = t_gbf + H;
        const float* t_gbg = t_gbi + H;
        const float* t_gbo = t_gbg + H;

        const float* t_gWf = base + bias_sz;
        const float* t_gWi = t_gWf + F * H;
        const float* t_gWg = t_gWi + F * H;
        const float* t_gWo = t_gWg + F * H;

        const float* t_gUf = base + bias_sz + w_sz;
        const float* t_gUi = t_gUf + H * H;
        const float* t_gUg = t_gUi + H * H;
        const float* t_gUo = t_gUg + H * H;

        for (Index h = 0; h < H; ++h)
        {
            gbf[h] += t_gbf[h]; gbi[h] += t_gbi[h];
            gbg[h] += t_gbg[h]; gbo[h] += t_gbo[h];
        }
        for (Index k = 0; k < F * H; ++k)
        {
            gWf[k] += t_gWf[k]; gWi[k] += t_gWi[k];
            gWg[k] += t_gWg[k]; gWo[k] += t_gWo[k];
        }
        for (Index k = 0; k < H * H; ++k)
        {
            gUf[k] += t_gUf[k]; gUi[k] += t_gUi[k];
            gUg[k] += t_gUg[k]; gUo[k] += t_gUo[k];
        }
    }
}

#ifdef OPENNN_HAS_CUDA

void LongShortTermMemoryOp::ensure_cudnn_setup_(Index batch_size) const
{
    using F_ = ActivationOp::Function;
    if (activation_function != F_::Tanh
        || recurrent_activation_function != F_::Sigmoid)
    {
        throw runtime_error(
            "LongShortTermMemoryOp::apply_gpu: cuDNN CUDNN_LSTM only supports "
            "Tanh cell activation + Sigmoid gate activation. "
            "Reconfigure the layer or fall back to CPU.");
    }

    const Index F = input_features;
    const Index H = output_features;
    const Index T = time_steps;

    const bool topology_changed =
        cached_input_features  != F ||
        cached_output_features != H ||
        rnn_desc == nullptr;

    if (topology_changed)
    {
        rnn_desc.reset();
        CHECK_CUDNN(cudnnCreateRNNDescriptor(&rnn_desc.handle));
        rnn_desc.deleter = &cudnnDestroyRNNDescriptor;

        if (!dropout_desc)
        {
            CHECK_CUDNN(cudnnCreateDropoutDescriptor(&dropout_desc.handle));
            dropout_desc.deleter = &cudnnDestroyDropoutDescriptor;
        }
        size_t dropout_states_bytes = 0;
        CHECK_CUDNN(cudnnDropoutGetStatesSize(
            Backend::get_cudnn_handle(), &dropout_states_bytes));
        dropout_states_buf.grow_to(Index(dropout_states_bytes));
        CHECK_CUDNN(cudnnSetDropoutDescriptor(
            dropout_desc, Backend::get_cudnn_handle(),
            /*dropout=*/0.0f,
            dropout_states_buf.data,
            size_t(dropout_states_buf.bytes),
            /*seed=*/0ULL));

        CHECK_CUDNN(cudnnSetRNNDescriptor_v8(
            rnn_desc,
            CUDNN_RNN_ALGO_STANDARD,
            CUDNN_LSTM,
            CUDNN_RNN_SINGLE_INP_BIAS,
            CUDNN_UNIDIRECTIONAL,
            CUDNN_LINEAR_INPUT,
            CUDNN_DATA_FLOAT,
            CUDNN_DATA_FLOAT,
            CUDNN_DEFAULT_MATH,
            int(F),
            int(H),
            /*projSize=*/ int(H),
            1,
            dropout_desc,
            CUDNN_RNN_PADDED_IO_ENABLED));

        size_t weight_bytes = 0;
        CHECK_CUDNN(cudnnGetRNNWeightSpaceSize(
            Backend::get_cudnn_handle(), rnn_desc, &weight_bytes));
        weight_space_buf.grow_to(Index(weight_bytes));
        dweight_space_buf.grow_to(Index(weight_bytes));
    }

    const bool data_shape_changed =
        cached_batch_size != batch_size ||
        cached_time_steps != T;

    if (data_shape_changed || topology_changed)
    {
        x_data_desc.reset();
        y_data_desc.reset();
        CHECK_CUDNN(cudnnCreateRNNDataDescriptor(&x_data_desc.handle));
        x_data_desc.deleter = &cudnnDestroyRNNDataDescriptor;
        CHECK_CUDNN(cudnnCreateRNNDataDescriptor(&y_data_desc.handle));
        y_data_desc.deleter = &cudnnDestroyRNNDataDescriptor;

        seq_lengths_host_buf.grow_to(batch_size * Index(sizeof(int32_t)));
        int32_t* seq_h = seq_lengths_host_buf.as<int32_t>();
        for (Index i = 0; i < batch_size; ++i) seq_h[i] = int32_t(T);

        seq_lengths_dev_buf.grow_to(batch_size * Index(sizeof(int32_t)));
        device::copy_async(seq_lengths_dev_buf.data, seq_h,
                           batch_size * Index(sizeof(int32_t)),
                           device::CopyKind::HostToDevice,
                           Backend::get_compute_stream());

        static float zero_pad_fill = 0.0f;
        CHECK_CUDNN(cudnnSetRNNDataDescriptor(
            x_data_desc, CUDNN_DATA_FLOAT,
            CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
            int(T), int(batch_size), int(F),
            seq_h, &zero_pad_fill));
        CHECK_CUDNN(cudnnSetRNNDataDescriptor(
            y_data_desc, CUDNN_DATA_FLOAT,
            CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
            int(T), int(batch_size), int(H),
            seq_h, &zero_pad_fill));

        if (!h_desc)
        {
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&h_desc.handle));
            h_desc.deleter = &cudnnDestroyTensorDescriptor;
        }
        if (!c_desc)
        {
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&c_desc.handle));
            c_desc.deleter = &cudnnDestroyTensorDescriptor;
        }
        const int dimA[3]    = {1, int(batch_size), int(H)};
        const int strideA[3] = {int(batch_size * H), int(H), 1};
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(h_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(c_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

        size_t work_bytes = 0;
        size_t reserve_bytes = 0;
        CHECK_CUDNN(cudnnGetRNNTempSpaceSizes(
            Backend::get_cudnn_handle(), rnn_desc,
            CUDNN_FWD_MODE_TRAINING, x_data_desc,
            &work_bytes, &reserve_bytes));
        workspace_buf.grow_to(Index(work_bytes));
        reserve_space_buf.grow_to(Index(reserve_bytes));

        const Index yh_bytes = batch_size * T * H * Index(sizeof(float));
        y_buf.grow_to(yh_bytes);
        dy_buf.grow_to(yh_bytes);

        CHECK_CUDNN(cudnnBuildRNNDynamic(
            Backend::get_cudnn_handle(), rnn_desc, int(batch_size)));

#ifdef OPENNN_LSTM_GPU_DEBUG
        std::cerr << "[lstm-gpu] cudnn buffers: weight=" << weight_space_buf.bytes
                  << " work=" << workspace_buf.bytes
                  << " reserve=" << reserve_space_buf.bytes
                  << " y=" << y_buf.bytes
                  << " B=" << batch_size << " T=" << T
                  << " F=" << F << " H=" << H << "\n";
#endif
    }

    cached_batch_size      = batch_size;
    cached_time_steps      = T;
    cached_input_features  = F;
    cached_output_features = H;
}

void LongShortTermMemoryOp::pack_weights_to_cudnn_() const
{
    if (weight_space_buf.data && weight_space_buf.bytes > 0)
        device::set_zero_async(weight_space_buf.data, weight_space_buf.bytes,
                               Backend::get_compute_stream());

    const TensorView* W[8] = {
        &input_weights,
        &forget_weights,
        &candidate_weights,
        &output_weights,
        &input_recurrent_weights,
        &forget_recurrent_weights,
        &candidate_recurrent_weights,
        &output_recurrent_weights
    };
    const TensorView* B[8] = {
        &input_bias,
        &forget_bias,
        &candidate_bias,
        &output_bias,
        nullptr, nullptr, nullptr, nullptr
    };

    const Index F = input_features;
    const Index H = output_features;

    CudnnDescriptor<cudnnTensorDescriptor_t> m_desc;
    CudnnDescriptor<cudnnTensorDescriptor_t> b_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_desc.handle));
    m_desc.deleter = &cudnnDestroyTensorDescriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&b_desc.handle));
    b_desc.deleter = &cudnnDestroyTensorDescriptor;

    for (int lin = 0; lin < 8; ++lin)
    {
        float* m_addr = nullptr;
        float* b_addr = nullptr;
        CHECK_CUDNN(cudnnGetRNNWeightParams(
            Backend::get_cudnn_handle(),
            rnn_desc,
            0,
            size_t(weight_space_buf.bytes),
            weight_space_buf.data,
            lin,
            m_desc, reinterpret_cast<void**>(&m_addr),
            b_desc, reinterpret_cast<void**>(&b_addr)));

        const bool is_input_w = (lin < 4);
        const Index rows_src = is_input_w ? F : H;
        if (m_addr && W[lin] && W[lin]->data)
            transpose_2d_cuda<float>(rows_src, H,
                                     W[lin]->as<float>(), m_addr);

        if (b_addr)
        {
            if (B[lin] && B[lin]->data)
                device::copy_async(b_addr, B[lin]->data,
                                   H * Index(sizeof(float)),
                                   device::CopyKind::DeviceToDevice,
                                   Backend::get_compute_stream());
            else
                device::set_zero_async(b_addr, H * Index(sizeof(float)),
                                       Backend::get_compute_stream());
        }
    }
}

void LongShortTermMemoryOp::unpack_gradients_from_cudnn_() const
{
    const TensorView* gW[8] = {
        &input_weight_gradient,
        &forget_weight_gradient,
        &candidate_weight_gradient,
        &output_weight_gradient,
        &input_recurrent_weight_gradient,
        &forget_recurrent_weight_gradient,
        &candidate_recurrent_weight_gradient,
        &output_recurrent_weight_gradient
    };
    const TensorView* gB[8] = {
        &input_bias_gradient,
        &forget_bias_gradient,
        &candidate_bias_gradient,
        &output_bias_gradient,
        nullptr, nullptr, nullptr, nullptr
    };

    const Index F = input_features;
    const Index H = output_features;

    CudnnDescriptor<cudnnTensorDescriptor_t> m_desc;
    CudnnDescriptor<cudnnTensorDescriptor_t> b_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_desc.handle));
    m_desc.deleter = &cudnnDestroyTensorDescriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&b_desc.handle));
    b_desc.deleter = &cudnnDestroyTensorDescriptor;

    for (int lin = 0; lin < 8; ++lin)
    {
        float* m_addr = nullptr;
        float* b_addr = nullptr;
        CHECK_CUDNN(cudnnGetRNNWeightParams(
            Backend::get_cudnn_handle(),
            rnn_desc,
            0,
            size_t(dweight_space_buf.bytes),
            dweight_space_buf.data,
            lin,
            m_desc, reinterpret_cast<void**>(&m_addr),
            b_desc, reinterpret_cast<void**>(&b_addr)));

        const bool is_input_w = (lin < 4);
        const Index cols_cudnn = is_input_w ? F : H;
        if (m_addr && gW[lin] && gW[lin]->data)
            transpose_2d_cuda<float>(H, cols_cudnn,
                                     m_addr, const_cast<float*>(gW[lin]->as<float>()));

        if (b_addr && gB[lin] && gB[lin]->data)
            device::copy_async(const_cast<void*>(gB[lin]->data), b_addr,
                               H * Index(sizeof(float)),
                               device::CopyKind::DeviceToDevice,
                               Backend::get_compute_stream());
    }
}

void LongShortTermMemoryOp::apply_gpu(const TensorView& input,
                                      TensorView& output,
                                      bool return_seq) const
{
    if (!input.data || output_features == 0 || time_steps == 0) return;

    const Index batch_size = input.shape[0];
    if (batch_size == 0) return;

    device::synchronize(Backend::get_compute_stream());

#ifdef OPENNN_LSTM_GPU_DEBUG
    {
        static bool printed_version = false;
        if (!printed_version)
        {
            std::cerr << "[lstm-gpu] cudnnGetVersion()=" << cudnnGetVersion()
                      << " cudnnGetCudartVersion()=" << cudnnGetCudartVersion() << "\n";
            printed_version = true;
        }
    }
#endif

#ifdef OPENNN_LSTM_GPU_DEBUG
    auto log = [&](const char* tag) {
        device::synchronize(Backend::get_compute_stream());
        std::cerr << "[lstm-gpu] " << tag << " B=" << batch_size
                  << " T=" << time_steps << " F=" << input_features
                  << " H=" << output_features << " ret_seq=" << return_seq << "\n";
    };
    log("enter apply_gpu");
#endif

    ensure_cudnn_setup_(batch_size);
#ifdef OPENNN_LSTM_GPU_DEBUG
    log("after ensure_cudnn_setup_");
#endif

    pack_weights_to_cudnn_();
#ifdef OPENNN_LSTM_GPU_DEBUG
    log("after pack_weights_to_cudnn_");
#endif

#ifdef OPENNN_LSTM_GPU_DEBUG
    auto where = [](const void* p) -> const char* {
        if (!p) return "NULL";
        cudaPointerAttributes a{};
        if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
            cudaGetLastError();
            return "UNKNOWN";
        }
        switch (a.type) {
            case cudaMemoryTypeHost:      return "HOST";
            case cudaMemoryTypeDevice:    return "DEVICE";
            case cudaMemoryTypeManaged:   return "MANAGED";
            case cudaMemoryTypeUnregistered: return "UNREG";
        }
        return "?";
    };
    std::cerr << "[lstm-gpu] pointer kinds  input=" << where(input.data)
              << "  weight=" << where(weight_space_buf.data)
              << "  y=" << where(y_buf.data)
              << "  workspace=" << where(workspace_buf.data)
              << "  reserve=" << where(reserve_space_buf.data)
              << "  seqLenDev=" << where(seq_lengths_dev_buf.data) << "\n";
#endif

    CHECK_CUDNN(cudnnRNNForward(
        Backend::get_cudnn_handle(),
        rnn_desc,
        CUDNN_FWD_MODE_TRAINING,
        seq_lengths_dev_buf.as<int32_t>(),
        x_data_desc, input.data,
        y_data_desc, y_buf.data,
        h_desc, nullptr, nullptr,
        c_desc, nullptr, nullptr,
        size_t(weight_space_buf.bytes), weight_space_buf.data,
        size_t(workspace_buf.bytes), workspace_buf.data,
        size_t(reserve_space_buf.bytes), reserve_space_buf.data));
#ifdef OPENNN_LSTM_GPU_DEBUG
    log("after cudnnRNNForward");
#endif

    const Index H = output_features;
    const Index T = time_steps;

    if (return_seq)
    {
        device::copy_async(output.data, y_buf.data,
                           batch_size * T * H * Index(sizeof(float)),
                           device::CopyKind::DeviceToDevice,
                           Backend::get_compute_stream());
    }
    else
    {
        gather_time_slice_cuda<float>(
            batch_size, T, H, T - 1,
            static_cast<const float*>(y_buf.data),
            output.as<float>());
    }
}

void LongShortTermMemoryOp::apply_delta_gpu(const TensorView& input,
                                            const TensorView& output_delta,
                                            TensorView& input_delta,
                                            bool return_seq) const
{
    if (!input.data || !output_delta.data
        || output_features == 0 || time_steps == 0) return;

    const Index batch_size = input.shape[0];
    if (batch_size == 0) return;

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

#ifdef OPENNN_LSTM_GPU_DEBUG
    auto log = [&](const char* tag) {
        device::synchronize(Backend::get_compute_stream());
        std::cerr << "[lstm-gpu-bwd] " << tag
                  << " B=" << batch_size << " T=" << time_steps
                  << " F=" << input_features << " H=" << output_features
                  << " ret_seq=" << return_seq << "\n";
    };
    log("enter apply_delta_gpu");
#endif

    ensure_cudnn_setup_(batch_size);
#ifdef OPENNN_LSTM_GPU_DEBUG
    log("after ensure_cudnn_setup_");
#endif

    const Index H = output_features;
    const Index T = time_steps;
    const size_t y_bytes = size_t(batch_size * T * H) * sizeof(float);

    if (return_seq)
    {
        device::copy_async(dy_buf.data, output_delta.data, Index(y_bytes),
                           device::CopyKind::DeviceToDevice,
                           Backend::get_compute_stream());
    }
    else
    {
        device::set_zero_async(dy_buf.data, Index(y_bytes),
                               Backend::get_compute_stream());
        scatter_time_slice_cuda<float>(
            batch_size, T, H, T - 1,
            output_delta.as<float>(),
            static_cast<float*>(dy_buf.data));
    }

#ifdef OPENNN_LSTM_GPU_DEBUG
    log("before cudnnRNNBackwardData_v8");
#endif

    // cuDNN always writes dx; when the previous layer needs no gradient
    // (input_delta unlinked) give it a scratch sink sized (B, T, F).
    void* dx_data = input_delta.data;
    if (!dx_data || input_delta.empty())
    {
        dx_scratch_buf.grow_to(batch_size * T * input_features * Index(sizeof(float)));
        dx_data = dx_scratch_buf.data;
    }

    CHECK_CUDNN(cudnnRNNBackwardData_v8(
        Backend::get_cudnn_handle(),
        rnn_desc,
        seq_lengths_dev_buf.as<int32_t>(),
        y_data_desc, y_buf.data, dy_buf.data,
        x_data_desc, dx_data,   // dx
        h_desc, nullptr, nullptr, nullptr,   // hx, dhy, dhx
        c_desc, nullptr, nullptr, nullptr,   // cx, dcy, dcx
        size_t(weight_space_buf.bytes), weight_space_buf.data,
        size_t(workspace_buf.bytes), workspace_buf.data,
        size_t(reserve_space_buf.bytes), reserve_space_buf.data));
#ifdef OPENNN_LSTM_GPU_DEBUG
    log("after cudnnRNNBackwardData_v8");
#endif

    device::set_zero_async(dweight_space_buf.data, dweight_space_buf.bytes,
                           Backend::get_compute_stream());

    CHECK_CUDNN(cudnnRNNBackwardWeights_v8(
        Backend::get_cudnn_handle(),
        rnn_desc,
        CUDNN_WGRAD_MODE_ADD,
        seq_lengths_dev_buf.as<int32_t>(),
        x_data_desc, input.data,
        h_desc, nullptr,
        y_data_desc, y_buf.data,
        size_t(dweight_space_buf.bytes), dweight_space_buf.data,
        size_t(workspace_buf.bytes), workspace_buf.data,
        size_t(reserve_space_buf.bytes), reserve_space_buf.data));
#ifdef OPENNN_LSTM_GPU_DEBUG
    log("after cudnnRNNBackwardWeights_v8");
#endif

    unpack_gradients_from_cudnn_();
#ifdef OPENNN_LSTM_GPU_DEBUG
    log("after unpack_gradients_from_cudnn_");
#endif
}

#else

void LongShortTermMemoryOp::apply_gpu(const TensorView&, TensorView&, bool) const
{
    throw runtime_error("LongShortTermMemoryOp::apply_gpu: CUDA support not compiled in.");
}

void LongShortTermMemoryOp::apply_delta_gpu(const TensorView&, const TensorView&, TensorView&, bool) const
{
    throw runtime_error("LongShortTermMemoryOp::apply_delta_gpu: CUDA support not compiled in.");
}

#endif  // OPENNN_HAS_CUDA

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
