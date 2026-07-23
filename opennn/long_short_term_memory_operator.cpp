//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "long_short_term_memory_operator.h"
#include "device_backend.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"

#include <initializer_list>
#include "kernel.cuh"

namespace opennn
{

namespace
{


void link_views(span<const TensorView> views, initializer_list<TensorView*> targets)
{
    if (views.size() < targets.size()) return;

    size_t index = 0;
    for (TensorView* target : targets) *target = views[index++];
}

void zero_if_linked(TensorView& view)
{
    if (view.data) view.setZero();
}

void zero_if_linked(const TensorView& view)
{
    if (view.data) const_cast<TensorView&>(view).setZero();
}

void zero_linked(initializer_list<const TensorView*> views)
{
    for (const TensorView* view : views) zero_if_linked(*view);
}

void set_random_uniform_linked(initializer_list<const TensorView*> views, float min, float max)
{
    for (const TensorView* view : views) set_random_uniform(view->as_vector(), min, max);
}

inline float lstm_activate(ActivationFunction function, float x)
{
    return activation_forward_value(function, x);
}

inline float lstm_derivative_from_output(ActivationFunction function, float y)
{
    return activation_derivative_from_output_value(function, y);
}

}


void LongShortTermMemoryOperator::set(Index new_input_features,
                                Index new_output_features,
                                Index new_time_steps,
                                ActivationFunction new_activation_function,
                                ActivationFunction new_recurrent_activation_function)
{
    input_features = new_input_features;
    output_features = new_output_features;
    time_steps = new_time_steps;
    activation_function = new_activation_function;
    recurrent_activation_function = new_recurrent_activation_function;
}

vector<TensorSpec> LongShortTermMemoryOperator::parameter_specs() const
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

void LongShortTermMemoryOperator::link_parameters(span<const TensorView> views)
{
    link_views(views, {&forget_bias, &input_bias, &candidate_bias, &output_bias,
                       &forget_weights, &input_weights, &candidate_weights, &output_weights,
                       &forget_recurrent_weights, &input_recurrent_weights,
                       &candidate_recurrent_weights, &output_recurrent_weights});
}

void LongShortTermMemoryOperator::link_gradients(span<const TensorView> views)
{
    link_views(views, {&forget_bias_gradient, &input_bias_gradient,
                       &candidate_bias_gradient, &output_bias_gradient,
                       &forget_weight_gradient, &input_weight_gradient,
                       &candidate_weight_gradient, &output_weight_gradient,
                       &forget_recurrent_weight_gradient, &input_recurrent_weight_gradient,
                       &candidate_recurrent_weight_gradient, &output_recurrent_weight_gradient});
}

void LongShortTermMemoryOperator::set_parameters_random()
{
    if (forget_bias.data) forget_bias.fill(1.0f);
    zero_linked({&input_bias, &candidate_bias, &output_bias});

    if (forget_weights.data)
        set_random_uniform_linked({&forget_weights, &input_weights,
                                   &candidate_weights, &output_weights}, -0.1f, 0.1f);

    if (forget_recurrent_weights.data)
        set_random_uniform_linked({&forget_recurrent_weights, &input_recurrent_weights,
                                   &candidate_recurrent_weights, &output_recurrent_weights}, -0.1f, 0.1f);
}

void LongShortTermMemoryOperator::set_parameters_glorot()
{
    if (forget_bias.data) forget_bias.fill(1.0f);
    zero_linked({&input_bias, &candidate_bias, &output_bias});

    if (forget_weights.data)
    {
        const float limit = glorot_limit(input_features, output_features);
        set_random_uniform_linked({&forget_weights, &input_weights,
                                   &candidate_weights, &output_weights}, -limit, limit);
    }

    if (forget_recurrent_weights.data)
        for (TensorView* recurrent : {&forget_recurrent_weights, &input_recurrent_weights,
                                      &candidate_recurrent_weights, &output_recurrent_weights})
            set_random_orthogonal(recurrent->as_matrix());
}

void LongShortTermMemoryOperator::set_parameters_pytorch()
{
    const float limit = 1.0f / sqrt(float(output_features > 0 ? output_features : 1));

    if (forget_bias.data)
        set_random_uniform_linked({&forget_bias, &input_bias,
                                   &candidate_bias, &output_bias}, -limit, limit);

    if (forget_weights.data)
        set_random_uniform_linked({&forget_weights, &input_weights,
                                   &candidate_weights, &output_weights}, -limit, limit);

    if (forget_recurrent_weights.data)
        set_random_uniform_linked({&forget_recurrent_weights, &input_recurrent_weights,
                                   &candidate_recurrent_weights, &output_recurrent_weights}, -limit, limit);
}

void LongShortTermMemoryOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training)
{
    auto& forward_slots = forward_propagation.forward_slots[layer];

    TensorView& input = get_input(forward_propagation, layer);
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
        apply_gpu(input, output, return_sequences, is_training);
        return;
    }

    apply(input, output, forget_gate, input_gate, candidate_gate, output_gate,
          cell_state, hidden_state, cell_activation);
}

void LongShortTermMemoryOperator::apply(const TensorView& input,
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

    if (H >= 64)
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

        // Gate fusion: stack [f|i|g|o] so each step is 1+1 GEMM instead of 4+4.
        MatrixR Wcat(F, 4 * H);
        Wcat.leftCols(H)          = Wf_m;
        Wcat.middleCols(H, H)     = Wi_m;
        Wcat.middleCols(2 * H, H) = Wg_m;
        Wcat.rightCols(H)         = Wo_m;
        MatrixR Ucat(H, 4 * H);
        Ucat.leftCols(H)          = Uf_m;
        Ucat.middleCols(H, H)     = Ui_m;
        Ucat.middleCols(2 * H, H) = Ug_m;
        Ucat.rightCols(H)         = Uo_m;
        VectorR bcat(4 * H);
        bcat.segment(0, H)        = bf_m;
        bcat.segment(H, H)        = bi_m;
        bcat.segment(2 * H, H)    = bg_m;
        bcat.segment(3 * H, H)    = bo_m;

        const Index BT = batch_size * T;
        MatrixR Zin(BT, 4 * H);
        Zin.noalias() = Eigen::Map<const MatrixR>(x, BT, F) * Wcat;
        Zin.rowwise() += bcat.transpose();

        using StridedZ = Eigen::Map<const MatrixR, 0, Eigen::OuterStride<>>;

        const bool standard_gates =
            recurrent_activation_function == ActivationFunction::Sigmoid
            && activation_function == ActivationFunction::Tanh;

        MatrixR Z_c(batch_size, 4 * H);
        MatrixR h_c(batch_size, H);

        const int eigen_threads = Eigen::nbThreads();
        Eigen::setNbThreads(1);

        #pragma omp parallel
        for (Index t = 0; t < T; ++t)
        {
            #pragma omp single
            {
                Z_c = StridedZ(Zin.data() + t * 4 * H, batch_size, 4 * H,
                               Eigen::OuterStride<>(T * 4 * H));

                if (t > 0)
                    Z_c.noalias() += h_c * Ucat;
            }

            #pragma omp for
            for (Index b = 0; b < batch_size; ++b)
            {
                const Index step = (b * T + t) * H;
                const float* Zrow = Z_c.data() + b * 4 * H;
                float* h_next = h_c.data() + b * H;
                const float* c_prev = t > 0 ? cells + (b * T + t - 1) * H : nullptr;

                for (Index h = 0; h < H; ++h)
                {
                    float f, i, g, o, a, c;
                    if (standard_gates)
                    {
                        f = 1.0f / (1.0f + std::exp(-Zrow[h]));
                        i = 1.0f / (1.0f + std::exp(-Zrow[H + h]));
                        g = std::tanh(Zrow[2 * H + h]);
                        o = 1.0f / (1.0f + std::exp(-Zrow[3 * H + h]));
                        c = f * (c_prev ? c_prev[h] : 0.0f) + i * g;
                        a = std::tanh(c);
                    }
                    else
                    {
                        f = lstm_activate(recurrent_activation_function, Zrow[h]);
                        i = lstm_activate(recurrent_activation_function, Zrow[H + h]);
                        g = lstm_activate(activation_function, Zrow[2 * H + h]);
                        o = lstm_activate(recurrent_activation_function, Zrow[3 * H + h]);
                        c = f * (c_prev ? c_prev[h] : 0.0f) + i * g;
                        a = activation_forward_value(activation_function, c);
                    }
                    const float h_value = o * a;

                    f_gate[step + h] = f;
                    i_gate[step + h] = i;
                    g_gate[step + h] = g;
                    o_gate[step + h] = o;
                    cells[step + h] = c;
                    cell_act[step + h] = a;
                    hidden[step + h] = h_value;
                    h_next[h] = h_value;
                    if (return_sequences) y[step + h] = h_value;
                }
            }
        }

        Eigen::setNbThreads(eigen_threads);

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

                const float f = activation_forward_value(recurrent_activation_function, zf);
                const float i = activation_forward_value(recurrent_activation_function, zi);
                const float g = activation_forward_value(activation_function, zg);
                const float o = activation_forward_value(recurrent_activation_function, zo);
                const float c = f * (c_prev ? c_prev[h] : 0.0f) + i * g;
                const float a = activation_forward_value(activation_function, c);
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

void LongShortTermMemoryOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    auto& backward_slots = back_propagation.backward_slots[layer];
    if (backward_slots.size() <= OutputDeltaScratchSlot) return;

    const auto& forward_slots = forward_propagation.forward_slots[layer];

    TensorView& input_delta = backward_slots[InputDeltaSlot];
    TensorView& hidden_delta = backward_slots[HiddenDeltaScratchSlot];
    TensorView& cell_delta = backward_slots[CellDeltaScratchSlot];
    TensorView& forget_delta = backward_slots[ForgetDeltaScratchSlot];
    TensorView& input_gate_delta = backward_slots[InputDeltaScratchSlot];
    TensorView& candidate_delta = backward_slots[CandidateDeltaScratchSlot];
    TensorView& output_gate_delta = backward_slots[OutputDeltaScratchSlot];

    const TensorView& input = get_input(forward_propagation, layer);
    const TensorView& output_delta = get_output_delta(back_propagation, layer);
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

void LongShortTermMemoryOperator::apply_delta(const TensorView& input,
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

    zero_linked({&forget_bias_gradient, &input_bias_gradient,
                 &candidate_bias_gradient, &output_bias_gradient,
                 &forget_weight_gradient, &input_weight_gradient,
                 &candidate_weight_gradient, &output_weight_gradient,
                 &forget_recurrent_weight_gradient, &input_recurrent_weight_gradient,
                 &candidate_recurrent_weight_gradient, &output_recurrent_weight_gradient});

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

    if (H >= 64)
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

        // Gate fusion: stack [f|i|g|o] so each step is 1+1+1 GEMM instead of 4×4.
        MatrixR Wcat(F, 4 * H);
        Wcat.leftCols(H)          = Wf_m;
        Wcat.middleCols(H, H)     = Wi_m;
        Wcat.middleCols(2 * H, H) = Wg_m;
        Wcat.rightCols(H)         = Wo_m;
        MatrixR Ucat(H, 4 * H);
        Ucat.leftCols(H)          = Uf_m;
        Ucat.middleCols(H, H)     = Ui_m;
        Ucat.middleCols(2 * H, H) = Ug_m;
        Ucat.rightCols(H)         = Uo_m;

        MatrixR gWcat = MatrixR::Zero(F, 4 * H);
        MatrixR gUcat = MatrixR::Zero(H, 4 * H);

        const Index BT = batch_size * T;
        MatrixR Dcat_all(BT, 4 * H);
        MatrixR D_c(batch_size, 4 * H);
        MatrixR h_prev_c(batch_size, H);
        MatrixR dh_next = MatrixR::Zero(batch_size, H);
        MatrixR dc_next = MatrixR::Zero(batch_size, H);

        using StridedD  = Eigen::Map<MatrixR, 0, Eigen::OuterStride<>>;
        using StridedCH = Eigen::Map<const MatrixR, 0, Eigen::OuterStride<>>;

        const int eigen_threads = Eigen::nbThreads();
        Eigen::setNbThreads(1);

        #pragma omp parallel
        for (Index t = T; t-- > 0;)
        {
            #pragma omp for
            for (Index b = 0; b < batch_size; ++b)
            {
                const Index step = (b * T + t) * H;
                float* Drow = D_c.data() + b * 4 * H;
                const float* c_prev = t > 0 ? cells + (b * T + t - 1) * H : nullptr;
                const float* dh_in  = return_sequences ? out_delta + step
                                    : (t == T - 1 ? out_delta + b * H : nullptr);

                for (Index h = 0; h < H; ++h)
                {
                    const float f = f_gate[step + h];
                    const float i = i_gate[step + h];
                    const float g = g_gate[step + h];
                    const float o = o_gate[step + h];
                    const float a = cell_act[step + h];

                    const float dh = dh_next(b, h) + (dh_in ? dh_in[h] : 0.0f);
                    const float dc = dh * o
                                   * activation_derivative_from_output_value(activation_function, a)
                                   + dc_next(b, h);

                    Drow[3 * H + h] = dh * a * lstm_derivative_from_output(recurrent_activation_function, o);
                    Drow[h]         = dc * (c_prev ? c_prev[h] : 0.0f) * lstm_derivative_from_output(recurrent_activation_function, f);
                    Drow[H + h]     = dc * g * lstm_derivative_from_output(recurrent_activation_function, i);
                    Drow[2 * H + h] = dc * i * lstm_derivative_from_output(activation_function, g);
                    dc_next(b, h)   = dc * f;
                }
            }

            #pragma omp single
            {
                StridedD(Dcat_all.data() + t * 4 * H, batch_size, 4 * H,
                         Eigen::OuterStride<>(T * 4 * H)) = D_c;

                if (t > 0)
                {
                    h_prev_c = StridedCH(hidden + (t - 1) * H, batch_size, H,
                                         Eigen::OuterStride<>(T * H));
                    gUcat.noalias() += h_prev_c.transpose() * D_c;
                    dh_next.noalias() = D_c * Ucat.transpose();
                }
            }
        }

        Eigen::setNbThreads(eigen_threads);

        const Eigen::Map<const MatrixR> all_x(x, BT, F);
        gWcat.noalias() = all_x.transpose() * Dcat_all;
        const VectorR gbcat = Dcat_all.colwise().sum().transpose();

        if (write_input_delta)
            Eigen::Map<MatrixR>(in_delta, BT, F).noalias() = Dcat_all * Wcat.transpose();

        gWf_m += gWcat.leftCols(H);          gWi_m += gWcat.middleCols(H, H);
        gWg_m += gWcat.middleCols(2 * H, H); gWo_m += gWcat.rightCols(H);
        gUf_m += gUcat.leftCols(H);          gUi_m += gUcat.middleCols(H, H);
        gUg_m += gUcat.middleCols(2 * H, H); gUo_m += gUcat.rightCols(H);
        gbf_v += gbcat.segment(0, H);        gbi_v += gbcat.segment(H, H);
        gbg_v += gbcat.segment(2 * H, H);    gbo_v += gbcat.segment(3 * H, H);

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

                const float dc = dh_next[h] * o
                               * activation_derivative_from_output_value(activation_function, a)
                               + dc_next[h];

                do_gate[h] = dh_next[h] * a
                            * activation_derivative_from_output_value(recurrent_activation_function, o);
                df[h] = dc * (c_prev ? c_prev[h] : 0.0f)
                      * activation_derivative_from_output_value(recurrent_activation_function, f);
                di[h] = dc * g
                      * activation_derivative_from_output_value(recurrent_activation_function, i);
                dg[h] = dc * i
                      * activation_derivative_from_output_value(activation_function, g);
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

static bool lstm_persist_env_enabled()
{
    static const bool enabled = []() {
        const char* env = std::getenv("OPENNN_RNN_PERSIST");
        return !(env && string(env) == "0");
    }();
    return enabled;
}

void LongShortTermMemoryOperator::ensure_cudnn_setup_(Index batch_size, bool for_training) const
{
    using F_ = ActivationFunction;
    if (activation_function != F_::Tanh
        || recurrent_activation_function != F_::Sigmoid)
    {
        throw runtime_error(
            "LongShortTermMemoryOperator::apply_gpu: cuDNN CUDNN_LSTM only supports "
            "Tanh cell activation + Sigmoid gate activation. "
            "Reconfigure the layer or fall back to CPU.");
    }

    if (!persist_algo_failed_ && lstm_persist_env_enabled())
    {
        try
        {
            ensure_cudnn_setup_attempt_(batch_size, for_training);
            return;
        }
        catch (const std::exception&)
        {
            persist_algo_failed_ = true;
            rnn_desc.reset();
            cached_input_features = -1;
        }
    }
    ensure_cudnn_setup_attempt_(batch_size, for_training);
}

void LongShortTermMemoryOperator::ensure_cudnn_setup_attempt_(Index batch_size, bool for_training) const
{
    persist_algo_active_ = !persist_algo_failed_ && lstm_persist_env_enabled();

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
            persist_algo_active_ ? CUDNN_RNN_ALGO_PERSIST_STATIC
                                 : CUDNN_RNN_ALGO_STANDARD,
            CUDNN_LSTM,
            CUDNN_RNN_SINGLE_INP_BIAS,
            CUDNN_UNIDIRECTIONAL,
            CUDNN_LINEAR_INPUT,
            CUDNN_DATA_FLOAT,
            CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_OP_MATH,
            int(F),
            int(H),
            /*projSize=*/ int(H),
            1,
            dropout_desc,
            persist_algo_active_ ? CUDNN_RNN_PADDED_IO_DISABLED
                                 : CUDNN_RNN_PADDED_IO_ENABLED));

        size_t weight_bytes = 0;
        CHECK_CUDNN(cudnnGetRNNWeightSpaceSize(
            Backend::get_cudnn_handle(), rnn_desc, &weight_bytes));
        weight_space_buf.grow_to(Index(weight_bytes));
        dweight_space_buf.grow_to(Index(weight_bytes));

        device::set_zero_async(weight_space_buf.data, weight_space_buf.bytes,
                               Backend::get_compute_stream());

        CudnnDescriptor<cudnnTensorDescriptor_t> m_desc;
        CudnnDescriptor<cudnnTensorDescriptor_t> b_desc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_desc.handle));
        m_desc.deleter = &cudnnDestroyTensorDescriptor;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&b_desc.handle));
        b_desc.deleter = &cudnnDestroyTensorDescriptor;

        for (int lin = 0; lin < 8; ++lin)
        {
            CHECK_CUDNN(cudnnGetRNNWeightParams(
                Backend::get_cudnn_handle(), rnn_desc, 0,
                size_t(weight_space_buf.bytes), weight_space_buf.data, lin,
                m_desc, reinterpret_cast<void**>(&cudnn_w_ptrs_[lin]),
                b_desc, reinterpret_cast<void**>(&cudnn_b_ptrs_[lin])));
            CHECK_CUDNN(cudnnGetRNNWeightParams(
                Backend::get_cudnn_handle(), rnn_desc, 0,
                size_t(dweight_space_buf.bytes), dweight_space_buf.data, lin,
                m_desc, reinterpret_cast<void**>(&cudnn_gw_ptrs_[lin]),
                b_desc, reinterpret_cast<void**>(&cudnn_gb_ptrs_[lin])));
        }
    }

    if (topology_changed)
        for (CudnnRnnShapeSlot& slot : shape_slots_)
        {
            slot.batch = -1;
            slot.time  = -1;
        }

    int slot_index = -1;
    for (int s = 0; s < RNN_SHAPE_SLOTS; ++s)
        if (shape_slots_[s].batch == batch_size && shape_slots_[s].time == T)
            slot_index = s;

    if (slot_index >= 0 && for_training && !shape_slots_[slot_index].training_ready)
    {
        CudnnRnnShapeSlot& slot = shape_slots_[slot_index];
        size_t work_bytes = 0;
        size_t reserve_bytes = 0;
        CHECK_CUDNN(cudnnGetRNNTempSpaceSizes(
            Backend::get_cudnn_handle(), rnn_desc,
            CUDNN_FWD_MODE_TRAINING, slot.x_desc,
            &work_bytes, &reserve_bytes));
        workspace_buf.grow_to(Index(work_bytes));
        reserve_space_buf.grow_to(Index(reserve_bytes));
        dy_buf.grow_to(batch_size * T * output_features * Index(sizeof(float)));
        slot.training_ready = true;
    }

    if (slot_index < 0)
    {
        slot_index = 0;
        for (int s = 1; s < RNN_SHAPE_SLOTS; ++s)
        {
            if (shape_slots_[slot_index].batch < 0) break;
            if (shape_slots_[s].batch < 0 || shape_slots_[s].stamp < shape_slots_[slot_index].stamp)
                slot_index = s;
        }
        CudnnRnnShapeSlot& slot = shape_slots_[slot_index];
        slot.batch = batch_size;
        slot.time  = T;

        slot.x_desc.reset();
        slot.y_desc.reset();
        CHECK_CUDNN(cudnnCreateRNNDataDescriptor(&slot.x_desc.handle));
        slot.x_desc.deleter = &cudnnDestroyRNNDataDescriptor;
        CHECK_CUDNN(cudnnCreateRNNDataDescriptor(&slot.y_desc.handle));
        slot.y_desc.deleter = &cudnnDestroyRNNDataDescriptor;

        slot.seq_host.grow_to(batch_size * Index(sizeof(int32_t)));
        int32_t* seq_h = slot.seq_host.as<int32_t>();
        for (Index i = 0; i < batch_size; ++i) seq_h[i] = int32_t(T);

        slot.seq_dev.grow_to(batch_size * Index(sizeof(int32_t)));
        device::copy_async(slot.seq_dev.data, seq_h,
                           batch_size * Index(sizeof(int32_t)),
                           device::CopyKind::HostToDevice,
                           Backend::get_compute_stream());

        static float zero_pad_fill = 0.0f;
        CHECK_CUDNN(cudnnSetRNNDataDescriptor(
            slot.x_desc, CUDNN_DATA_FLOAT,
            CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
            int(T), int(batch_size), int(F),
            seq_h, &zero_pad_fill));
        CHECK_CUDNN(cudnnSetRNNDataDescriptor(
            slot.y_desc, CUDNN_DATA_FLOAT,
            CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
            int(T), int(batch_size), int(H),
            seq_h, &zero_pad_fill));

        slot.h_desc.reset();
        slot.c_desc.reset();
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&slot.h_desc.handle));
        slot.h_desc.deleter = &cudnnDestroyTensorDescriptor;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&slot.c_desc.handle));
        slot.c_desc.deleter = &cudnnDestroyTensorDescriptor;
        const int dimA[3]    = {1, int(batch_size), int(H)};
        const int strideA[3] = {int(batch_size * H), int(H), 1};
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(slot.h_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(slot.c_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

        size_t work_bytes = 0;
        size_t reserve_bytes = 0;
        if (for_training)
            CHECK_CUDNN(cudnnGetRNNTempSpaceSizes(
                Backend::get_cudnn_handle(), rnn_desc,
                CUDNN_FWD_MODE_TRAINING, slot.x_desc,
                &work_bytes, &reserve_bytes));

        size_t inference_work_bytes = 0;
        CHECK_CUDNN(cudnnGetRNNTempSpaceSizes(
            Backend::get_cudnn_handle(), rnn_desc,
            CUDNN_FWD_MODE_INFERENCE, slot.x_desc,
            &inference_work_bytes, nullptr));

        slot.training_ready = for_training;

        workspace_buf.grow_to(Index(max(work_bytes, inference_work_bytes)));

        const Index yh_bytes = batch_size * T * H * Index(sizeof(float));
        y_buf.grow_to(yh_bytes);
        if (for_training)
        {
            reserve_space_buf.grow_to(Index(reserve_bytes));
            dy_buf.grow_to(yh_bytes);
        }

    }

    shape_slots_[slot_index].stamp = ++shape_stamp_;
    active_shape_ = slot_index;

    cached_input_features  = F;
    cached_output_features = H;
}

void LongShortTermMemoryOperator::pack_weights_to_cudnn_() const
{
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

    RnnCopySpec specs[RNN_COPY_MAX_REGIONS];
    int count = 0;
    for (int lin = 0; lin < 8; ++lin)
    {
        const bool is_input_w = (lin < 4);
        if (cudnn_w_ptrs_[lin] && W[lin]->data)
            specs[count++] = {W[lin]->as<float>(), cudnn_w_ptrs_[lin],
                              int(is_input_w ? F : H), int(H), 1};

        if (cudnn_b_ptrs_[lin] && B[lin] && B[lin]->data)
            specs[count++] = {B[lin]->as<float>(), cudnn_b_ptrs_[lin],
                              1, int(H), 0};
    }
    rnn_copy_regions_cuda(specs, count);
}

void LongShortTermMemoryOperator::unpack_gradients_from_cudnn_() const
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

    RnnCopySpec specs[RNN_COPY_MAX_REGIONS];
    int count = 0;
    for (int lin = 0; lin < 8; ++lin)
    {
        const bool is_input_w = (lin < 4);
        if (cudnn_gw_ptrs_[lin] && gW[lin]->data)
            specs[count++] = {cudnn_gw_ptrs_[lin],
                              const_cast<float*>(gW[lin]->as<float>()),
                              int(H), int(is_input_w ? F : H), 1};

        if (cudnn_gb_ptrs_[lin] && gB[lin] && gB[lin]->data)
            specs[count++] = {cudnn_gb_ptrs_[lin],
                              const_cast<float*>(gB[lin]->as<float>()),
                              1, int(H), 0};
    }
    rnn_copy_regions_cuda(specs, count);
}

void LongShortTermMemoryOperator::apply_gpu(const TensorView& input,
                                      TensorView& output,
                                      bool return_seq,
                                      bool is_training) const
{
    const Index batch_size = input.shape[0];
    if (!input.data || output_features == 0 || time_steps == 0 || batch_size == 0) return;

    ensure_cudnn_setup_(batch_size, is_training);
    pack_weights_to_cudnn_();

    float* y_target = return_seq ? output.as<float>()
                                 : static_cast<float*>(y_buf.data);
    y_used_ = y_target;

    auto run_forward = [&]() {
        return cudnnRNNForward(
            Backend::get_cudnn_handle(),
            rnn_desc,
            is_training ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE,
            active_shape().seq_dev.as<int32_t>(),
            active_shape().x_desc, input.data,
            active_shape().y_desc, y_target,
            active_shape().h_desc, nullptr, nullptr,
            active_shape().c_desc, nullptr, nullptr,
            size_t(weight_space_buf.bytes), weight_space_buf.data,
            size_t(workspace_buf.bytes), workspace_buf.data,
            is_training ? size_t(reserve_space_buf.bytes) : 0,
            is_training ? reserve_space_buf.data : nullptr);
    };

    cudnnStatus_t forward_status = run_forward();
    if (forward_status == CUDNN_STATUS_NOT_SUPPORTED && persist_algo_active_)
    {
        persist_algo_failed_ = true;
        rnn_desc.reset();
        cached_input_features = -1;
        ensure_cudnn_setup_(batch_size, is_training);
        pack_weights_to_cudnn_();
        forward_status = run_forward();
    }
    CHECK_CUDNN(forward_status);

    if (return_seq) return;

    gather_time_slice_cuda<float>(
        batch_size, time_steps, output_features, time_steps - 1,
        y_target,
        output.as<float>());
}

void LongShortTermMemoryOperator::apply_delta_gpu(const TensorView& input,
                                            const TensorView& output_delta,
                                            TensorView& input_delta,
                                            bool return_seq) const
{
    if (!input.data || !output_delta.data
        || output_features == 0 || time_steps == 0) return;

    const Index batch_size = input.shape[0];
    if (batch_size == 0) return;

    ensure_cudnn_setup_(batch_size, true);

    const Index H = output_features;
    const Index T = time_steps;

    const float* dy_data = output_delta.as<float>();
    if (!return_seq)
    {
        scatter_time_slice_fill_cuda(
            batch_size, T, H, T - 1,
            output_delta.as<float>(),
            static_cast<float*>(dy_buf.data));
        dy_data = static_cast<const float*>(dy_buf.data);
    }

    const float* y_data = y_used_ ? y_used_
                                  : static_cast<const float*>(y_buf.data);

    const CudnnRnnShapeSlot& shape = active_shape();

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
        shape.seq_dev.as<int32_t>(),
        shape.y_desc, y_data, dy_data,
        shape.x_desc, dx_data,   // dx
        shape.h_desc, nullptr, nullptr, nullptr,   // hx, dhy, dhx
        shape.c_desc, nullptr, nullptr, nullptr,   // cx, dcy, dcx
        size_t(weight_space_buf.bytes), weight_space_buf.data,
        size_t(workspace_buf.bytes), workspace_buf.data,
        size_t(reserve_space_buf.bytes), reserve_space_buf.data));

    device::set_zero_async(dweight_space_buf.data, dweight_space_buf.bytes,
                           Backend::get_compute_stream());

    CHECK_CUDNN(cudnnRNNBackwardWeights_v8(
        Backend::get_cudnn_handle(),
        rnn_desc,
        CUDNN_WGRAD_MODE_ADD,
        shape.seq_dev.as<int32_t>(),
        shape.x_desc, input.data,
        shape.h_desc, nullptr,
        shape.y_desc, y_data,
        size_t(dweight_space_buf.bytes), dweight_space_buf.data,
        size_t(workspace_buf.bytes), workspace_buf.data,
        size_t(reserve_space_buf.bytes), reserve_space_buf.data));

    unpack_gradients_from_cudnn_();
}

#else

void LongShortTermMemoryOperator::apply_gpu(const TensorView&, TensorView&, bool, bool) const
{
    throw runtime_error("apply_gpu requires CUDA.");
}

void LongShortTermMemoryOperator::apply_delta_gpu(const TensorView&, const TensorView&, TensorView&, bool) const
{
    throw runtime_error("apply_delta_gpu requires CUDA.");
}

#endif  // OPENNN_HAS_CUDA

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
