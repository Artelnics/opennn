//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/layers/long_short_term_memory_layer.h"
#include "opennn/registry.h"

#include "opennn/core/device_backend.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/profiler.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#include <initializer_list>
#include "opennn/neural_network/layers/kernel_recurrent.cuh"

#ifdef EIGEN_USE_MKL_ALL
#include <mkl_vml.h>
#endif

namespace opennn
{

namespace
{

void zero_if_linked(const TensorView& view)
{
    if (view.get_data()) view.setZero();
}

void zero_linked(initializer_list<const TensorView*> views)
{
    for (const TensorView* view : views) zero_if_linked(*view);
}

void set_random_uniform_linked(initializer_list<const TensorView*> views, float min, float max)
{
    for (const TensorView* view : views) set_random_uniform(view->as_vector(), min, max);
}

}

void LongShortTermMemoryOperator::set(Index new_input_features,
                                Index new_output_features,
                                Index new_time_steps,
                                ActivationFunction new_activation_function,
                                ActivationFunction new_recurrent_activation_function,
                                Type new_compute_dtype)
{
    input_features = new_input_features;
    output_features = new_output_features;
    time_steps = new_time_steps;
    activation_function = new_activation_function;
    recurrent_activation_function = new_recurrent_activation_function;
    compute_dtype = new_compute_dtype;
}

vector<TensorSpec> LongShortTermMemoryOperator::parameter_specs() const
{
    if (output_features == 0)
        return {};

    const Shape bias_shape{output_features};
    const Shape input_weight_shape{input_features, output_features};
    const Shape recurrent_weight_shape{output_features, output_features};

    return {
        {bias_shape, compute_dtype},
        {bias_shape, compute_dtype},
        {bias_shape, compute_dtype},
        {bias_shape, compute_dtype},
        {input_weight_shape, compute_dtype},
        {input_weight_shape, compute_dtype},
        {input_weight_shape, compute_dtype},
        {input_weight_shape, compute_dtype},
        {recurrent_weight_shape, compute_dtype},
        {recurrent_weight_shape, compute_dtype},
        {recurrent_weight_shape, compute_dtype},
        {recurrent_weight_shape, compute_dtype},
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
    if (forget_bias.get_data()) forget_bias.fill(1.0f);
    zero_linked({&input_bias, &candidate_bias, &output_bias});

    if (forget_weights.get_data())
        set_random_uniform_linked({&forget_weights, &input_weights,
                                   &candidate_weights, &output_weights}, -0.1f, 0.1f);

    if (forget_recurrent_weights.get_data())
        set_random_uniform_linked({&forget_recurrent_weights, &input_recurrent_weights,
                                   &candidate_recurrent_weights, &output_recurrent_weights}, -0.1f, 0.1f);
}

void LongShortTermMemoryOperator::set_parameters_glorot()
{
    if (forget_bias.get_data()) forget_bias.fill(1.0f);
    zero_linked({&input_bias, &candidate_bias, &output_bias});

    if (forget_weights.get_data())
    {
        const float limit = glorot_limit(input_features, output_features);
        set_random_uniform_linked({&forget_weights, &input_weights,
                                   &candidate_weights, &output_weights}, -limit, limit);
    }

    if (forget_recurrent_weights.get_data())
        for (TensorView* recurrent : {&forget_recurrent_weights, &input_recurrent_weights,
                                      &candidate_recurrent_weights, &output_recurrent_weights})
            set_random_orthogonal(recurrent->as_matrix());
}

void LongShortTermMemoryOperator::set_parameters_pytorch()
{
    const float limit = 1.0f / sqrt(float(output_features > 0 ? output_features : 1));

    if (forget_bias.get_data())
        set_random_uniform_linked({&forget_bias, &input_bias,
                                   &candidate_bias, &output_bias}, -limit, limit);

    if (forget_weights.get_data())
        set_random_uniform_linked({&forget_weights, &input_weights,
                                   &candidate_weights, &output_weights}, -limit, limit);

    if (forget_recurrent_weights.get_data())
        set_random_uniform_linked({&forget_recurrent_weights, &input_recurrent_weights,
                                   &candidate_recurrent_weights, &output_recurrent_weights}, -limit, limit);
}

void LongShortTermMemoryOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training)
{
    auto& forward_slots = forward_propagation.slots[layer];

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
        return apply_gpu(input, output, hidden_state,
                         forward_slots[CudnnInputSequenceSlot],
                         forward_slots[CudnnOutputSequenceSlot],
                         forward_propagation.layer_state_storage[layer],
                         return_sequences, is_training);

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
    if (!input.get_data() || output_features == 0 || time_steps == 0) return;

    const Index batch_size = input.get_shape()[0];
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

    // Below 96 units the contiguous per-sequence kernel avoids the repeated
    // GEMM/barrier overhead of the matrix path on forecasting-sized cells.
    if (H >= 96)
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
                        f = 1.0f / (1.0f + exp(-Zrow[h]));
                        i = 1.0f / (1.0f + exp(-Zrow[H + h]));
                        g = tanh(Zrow[2 * H + h]);
                        o = 1.0f / (1.0f + exp(-Zrow[3 * H + h]));
                        c = f * (c_prev ? c_prev[h] : 0.0f) + i * g;
                        a = tanh(c);
                    }
                    else
                    {
                        f = activation_forward_value(
                            recurrent_activation_function, Zrow[h]);
                        i = activation_forward_value(
                            recurrent_activation_function, Zrow[H + h]);
                        g = activation_forward_value(
                            activation_function, Zrow[2 * H + h]);
                        o = activation_forward_value(
                            recurrent_activation_function, Zrow[3 * H + h]);
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

    // Keep one preactivation row per sequence.  Iterating features before
    // hidden units makes every weight access contiguous; the previous h-k
    // ordering stepped through the matrices with a stride of H.
    vector<float> gate_preactivations(
        size_t(batch_size) * size_t(4 * H));
    const bool standard_gates =
        recurrent_activation_function == ActivationFunction::Sigmoid
        && activation_function == ActivationFunction::Tanh;

    #pragma omp parallel for
    for (Index b = 0; b < batch_size; ++b)
    {
        float* zf = gate_preactivations.data() + b * 4 * H;
        float* zi = zf + H;
        float* zg = zi + H;
        float* zo = zg + H;

        for (Index t = 0; t < T; ++t)
        {
            const float* xt = x + (b * T + t) * F;
            const float* h_prev = t > 0 ? hidden + (b * T + t - 1) * H : nullptr;
            const float* c_prev = t > 0 ? cells + (b * T + t - 1) * H : nullptr;
            const Index step = (b * T + t) * H;

            copy_n(bf, H, zf);
            copy_n(bi, H, zi);
            copy_n(bg, H, zg);
            copy_n(bo, H, zo);

            for (Index k = 0; k < F; ++k)
            {
                const float xk = xt[k];
                const Index row = k * H;

                #pragma omp simd
                for (Index h = 0; h < H; ++h)
                {
                    zf[h] += xk * Wf[row + h];
                    zi[h] += xk * Wi[row + h];
                    zg[h] += xk * Wg[row + h];
                    zo[h] += xk * Wo[row + h];
                }
            }

            if (h_prev)
            {
                for (Index j = 0; j < H; ++j)
                {
                    const float hp = h_prev[j];
                    const Index row = j * H;

                    #pragma omp simd
                    for (Index h = 0; h < H; ++h)
                    {
                        zf[h] += hp * Uf[row + h];
                        zi[h] += hp * Ui[row + h];
                        zg[h] += hp * Ug[row + h];
                        zo[h] += hp * Uo[row + h];
                    }
                }
            }

            if (standard_gates)
            {
#ifdef EIGEN_USE_MKL_ALL
                #pragma omp simd
                for (Index h = 0; h < H; ++h)
                {
                    zf[h] = -zf[h];
                    zi[h] = -zi[h];
                    zo[h] = -zo[h];
                }
                vsExp(MKL_INT(H), zf, zf);
                vsExp(MKL_INT(H), zi, zi);
                vsExp(MKL_INT(H), zo, zo);
                vsTanh(MKL_INT(H), zg, zg);

                #pragma omp simd
                for (Index h = 0; h < H; ++h)
                {
                    const float f = 1.0f / (1.0f + zf[h]);
                    const float i = 1.0f / (1.0f + zi[h]);
                    const float g = zg[h];
                    const float o = 1.0f / (1.0f + zo[h]);
                    const float c = f * (c_prev ? c_prev[h] : 0.0f) + i * g;
                    f_gate[step + h] = f; i_gate[step + h] = i;
                    g_gate[step + h] = g; o_gate[step + h] = o;
                    cells[step + h] = c;
                }
                vsTanh(MKL_INT(H), cells + step, cell_act + step);

                #pragma omp simd
                for (Index h = 0; h < H; ++h)
                {
                    const float h_value = o_gate[step + h] * cell_act[step + h];
                    hidden[step + h] = h_value;
                    if (return_sequences) y[step + h] = h_value;
                }
#else
                #pragma omp simd
                for (Index h = 0; h < H; ++h)
                {
                    const float f = 1.0f / (1.0f + expf(-zf[h]));
                    const float i = 1.0f / (1.0f + expf(-zi[h]));
                    const float g = tanhf(zg[h]);
                    const float o = 1.0f / (1.0f + expf(-zo[h]));
                    const float c = f * (c_prev ? c_prev[h] : 0.0f) + i * g;
                    const float a = tanhf(c);
                    const float h_value = o * a;

                    f_gate[step + h] = f; i_gate[step + h] = i;
                    g_gate[step + h] = g; o_gate[step + h] = o;
                    cells[step + h] = c; cell_act[step + h] = a;
                    hidden[step + h] = h_value;
                    if (return_sequences) y[step + h] = h_value;
                }
#endif
            }
            else
            {
                for (Index h = 0; h < H; ++h)
                {
                    const float f = activation_forward_value(recurrent_activation_function, zf[h]);
                    const float i = activation_forward_value(recurrent_activation_function, zi[h]);
                    const float g = activation_forward_value(activation_function, zg[h]);
                    const float o = activation_forward_value(recurrent_activation_function, zo[h]);
                    const float c = f * (c_prev ? c_prev[h] : 0.0f) + i * g;
                    const float a = activation_forward_value(activation_function, c);
                    const float h_value = o * a;

                    f_gate[step + h] = f; i_gate[step + h] = i;
                    g_gate[step + h] = g; o_gate[step + h] = o;
                    cells[step + h] = c; cell_act[step + h] = a;
                    hidden[step + h] = h_value;
                    if (return_sequences) y[step + h] = h_value;
                }
            }
        }

        if (!return_sequences)
            copy_n(hidden + (b * T + T - 1) * H, H, y + b * H);
    }
}

void LongShortTermMemoryOperator::back_propagate(ForwardPropagation& forward_propagation, BackPropagation& back_propagation, size_t layer) const
{
    auto& backward_slots = back_propagation.slots[layer];
    if (backward_slots.size() <= OutputDeltaScratchSlot) return;

    const auto& forward_slots = forward_propagation.slots[layer];

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
        return apply_delta_gpu(input,
                               forward_slots[return_sequences ? OutputSlot : HiddenStateSlot],
                               output_delta,
                               forward_slots[CudnnInputSequenceSlot],
                               forward_slots[CudnnOutputSequenceSlot],
                               input_delta,
                               backward_slots[CudnnOutputDeltaScratchSlot],
                               backward_slots[CudnnInputDeltaScratchSlot],
                               forward_propagation.layer_state_storage[layer],
                               back_propagation.layer_scratch_storage[layer],
                               return_sequences);

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
    if (!input.get_data() || !output_delta.get_data() || output_features == 0 || time_steps == 0) return;

    zero_linked({&forget_bias_gradient, &input_bias_gradient,
                 &candidate_bias_gradient, &output_bias_gradient,
                 &forget_weight_gradient, &input_weight_gradient,
                 &candidate_weight_gradient, &output_weight_gradient,
                 &forget_recurrent_weight_gradient, &input_recurrent_weight_gradient,
                 &candidate_recurrent_weight_gradient, &output_recurrent_weight_gradient});

    const Index batch_size = input.get_shape()[0];
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

    if (H >= 96)
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

                    Drow[3 * H + h] = dh * a
                        * activation_derivative_from_output_value(
                            recurrent_activation_function, o);
                    Drow[h] = dc * (c_prev ? c_prev[h] : 0.0f)
                        * activation_derivative_from_output_value(
                            recurrent_activation_function, f);
                    Drow[H + h] = dc * g
                        * activation_derivative_from_output_value(
                            recurrent_activation_function, i);
                    Drow[2 * H + h] = dc * i
                        * activation_derivative_from_output_value(
                            activation_function, g);
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

    const bool standard_gates =
        recurrent_activation_function == ActivationFunction::Sigmoid
        && activation_function == ActivationFunction::Tanh;

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

    vector<float> gradient_thread_scratch(
        size_t(nthreads) * size_t(per_thread_sz), 0.0f);

    #pragma omp parallel for
    for (Index b = 0; b < batch_size; ++b)
    {
        const int tid = omp_get_thread_num();
        float* tls_base = gradient_thread_scratch.data()
                        + size_t(tid) * size_t(per_thread_sz);

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

            #pragma omp simd
            for (Index h = 0; h < H; ++h)
            {
                const float f = f_gate[step + h];
                const float i = i_gate[step + h];
                const float g = g_gate[step + h];
                const float o = o_gate[step + h];
                const float a = cell_act[step + h];

                const float dc = dh_next[h] * o
                               * (standard_gates ? 1.0f - a * a
                                   : activation_derivative_from_output_value(
                                       activation_function, a))
                               + dc_next[h];

                do_gate[h] = dh_next[h] * a
                            * (standard_gates ? o * (1.0f - o)
                                : activation_derivative_from_output_value(
                                    recurrent_activation_function, o));
                df[h] = dc * (c_prev ? c_prev[h] : 0.0f)
                      * (standard_gates ? f * (1.0f - f)
                          : activation_derivative_from_output_value(
                              recurrent_activation_function, f));
                di[h] = dc * g
                      * (standard_gates ? i * (1.0f - i)
                          : activation_derivative_from_output_value(
                              recurrent_activation_function, i));
                dg[h] = dc * i
                      * (standard_gates ? 1.0f - g * g
                          : activation_derivative_from_output_value(
                              activation_function, g));
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

                #pragma omp simd reduction(+:dx)
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

                #pragma omp simd reduction(+:dh_prev)
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
        const float* base = gradient_thread_scratch.data()
                          + size_t(tid) * size_t(per_thread_sz);

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

        #pragma omp simd
        for (Index h = 0; h < H; ++h)
        {
            gbf[h] += t_gbf[h]; gbi[h] += t_gbi[h];
            gbg[h] += t_gbg[h]; gbo[h] += t_gbo[h];
        }
        #pragma omp simd
        for (Index k = 0; k < F * H; ++k)
        {
            gWf[k] += t_gWf[k]; gWi[k] += t_gWi[k];
            gWg[k] += t_gWg[k]; gWo[k] += t_gWo[k];
        }
        #pragma omp simd
        for (Index k = 0; k < H * H; ++k)
        {
            gUf[k] += t_gUf[k]; gUi[k] += t_gUi[k];
            gUg[k] += t_gUg[k]; gUo[k] += t_gUo[k];
        }
    }
}

#ifdef OPENNN_HAS_CUDA

CudnnRnnShapeSlot& LongShortTermMemoryOperator::ensure_cudnn_setup_(
    Index batch_size, bool for_training) const
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

    return cudnn_setup_({CUDNN_LSTM, compute_dtype},
                        input_features, output_features, time_steps,
                        batch_size, for_training);
}

void LongShortTermMemoryOperator::pack_weights_to_cudnn_(Buffer& forward_state) const
{
    const TensorView* weights[8] = {
        &input_weights,
        &forget_weights,
        &candidate_weights,
        &output_weights,
        &input_recurrent_weights,
        &forget_recurrent_weights,
        &candidate_recurrent_weights,
        &output_recurrent_weights
    };
    const TensorView* biases[8] = {
        &input_bias,
        &forget_bias,
        &candidate_bias,
        &output_bias,
        nullptr, nullptr, nullptr, nullptr
    };
    cudnn_pack_weights_(8, input_features, output_features,
                        weights, biases, forward_state);
}

void LongShortTermMemoryOperator::unpack_gradients_from_cudnn_(Buffer& backward_scratch) const
{
    const TensorView* weight_gradients[8] = {
        &input_weight_gradient,
        &forget_weight_gradient,
        &candidate_weight_gradient,
        &output_weight_gradient,
        &input_recurrent_weight_gradient,
        &forget_recurrent_weight_gradient,
        &candidate_recurrent_weight_gradient,
        &output_recurrent_weight_gradient
    };
    const TensorView* bias_gradients[8] = {
        &input_bias_gradient,
        &forget_bias_gradient,
        &candidate_bias_gradient,
        &output_bias_gradient,
        nullptr, nullptr, nullptr, nullptr
    };
    cudnn_unpack_gradients_(8, input_features, output_features,
                            weight_gradients, bias_gradients,
                            backward_scratch);
}

void LongShortTermMemoryOperator::apply_gpu(const TensorView& input,
                                      TensorView& output,
                                      TensorView& sequence_output_scratch,
                                      TensorView& cudnn_input_sequence,
                                      TensorView& cudnn_output_sequence,
                                      Buffer& forward_state,
                                      bool return_seq,
                                      bool is_training) const
{
    const Index batch_size = input.get_shape()[0];
    if (!input.get_data() || output_features == 0 || time_steps == 0 || batch_size == 0) return;

    const auto backend_lock = lock_backend_state();

    CudnnRnnShapeSlot& shape = ensure_cudnn_setup_(batch_size, is_training);
    prepare_cudnn_forward_state_(forward_state, is_training, shape);
    pack_weights_to_cudnn_(forward_state);

    const void* x_data = input.get_data();
    void* y_data = sequence_output_scratch.get_data();
    if (shape.time_major)
    {
        PROFILE_SCOPE("rnn:transpose_input");
        const Index cudnn_input_features = shape.input_features;
        input.dispatch([&]<typename Scalar>()
        {
            batch_time_to_time_batch_padded_cuda<Scalar>(
                batch_size, time_steps, input_features, cudnn_input_features,
                input.as<Scalar>(), cudnn_input_sequence.as<Scalar>());
        });
        x_data = cudnn_input_sequence.get_data();
        y_data = cudnn_output_sequence.get_data();
    }

    cudnn_rnn_forward_(shape, is_training, true,
                       x_data, y_data,
                       forward_state,
                       [&]() -> CudnnRnnShapeSlot& {
                           CudnnRnnShapeSlot& retry_shape =
                               ensure_cudnn_setup_(batch_size, is_training);
                           prepare_cudnn_forward_state_(forward_state, is_training,
                                                        retry_shape);
                           pack_weights_to_cudnn_(forward_state);
                           return retry_shape;
                       });

    if (return_seq && shape.time_major)
    {
        PROFILE_SCOPE("rnn:transpose_output");
        output.dispatch([&]<typename Scalar>()
        {
            time_batch_to_batch_time_cuda<Scalar>(
                batch_size, time_steps, output_features,
                cudnn_output_sequence.as<Scalar>(), output.as<Scalar>());
        });
    }
    else if (return_seq)
        copy(sequence_output_scratch, output);
    else if (shape.time_major)
        output.dispatch([&]<typename Scalar>()
        {
            gather_time_major_slice_cuda<Scalar>(
                batch_size, time_steps, output_features, time_steps - 1,
                cudnn_output_sequence.as<Scalar>(), output.as<Scalar>());
        });
    else
        output.dispatch([&]<typename Scalar>()
        {
            gather_time_slice_cuda<Scalar>(
                batch_size, time_steps, output_features, time_steps - 1,
                sequence_output_scratch.as<Scalar>(), output.as<Scalar>());
        });
}

void LongShortTermMemoryOperator::apply_delta_gpu(const TensorView& input,
                                            const TensorView& sequence_output,
                                            const TensorView& output_delta,
                                            const TensorView& cudnn_input_sequence,
                                            const TensorView& cudnn_output_sequence,
                                            TensorView& input_delta,
                                            TensorView& sequence_delta_scratch,
                                            TensorView& input_delta_scratch,
                                            const Buffer& forward_state,
                                            Buffer& backward_scratch,
                                            bool return_seq) const
{
    if (!input.get_data() || !output_delta.get_data()
        || output_features == 0 || time_steps == 0) return;

    const Index batch_size = input.get_shape()[0];
    if (batch_size == 0) return;
    const auto backend_lock = lock_backend_state();

    CudnnRnnShapeSlot& shape = ensure_cudnn_setup_(batch_size, true);

    const Index H = output_features;
    const Index T = time_steps;

    const void* dy_data = output_delta.get_data();
    if (return_seq && shape.time_major)
    {
        PROFILE_SCOPE("rnn:transpose_delta");
        output_delta.dispatch([&]<typename Scalar>()
        {
            batch_time_to_time_batch_cuda<Scalar>(
                batch_size, T, H,
                output_delta.as<Scalar>(), sequence_delta_scratch.as<Scalar>());
        });
        dy_data = sequence_delta_scratch.get_data();
    }
    else if (!return_seq)
    {
        device::set_zero_async(sequence_delta_scratch.get_data(),
                               sequence_delta_scratch.byte_size(),
                               device::get_compute_stream());
        output_delta.dispatch([&]<typename Scalar>()
        {
            if (shape.time_major)
                scatter_time_major_slice_cuda<Scalar>(
                    batch_size, T, H, T - 1,
                    output_delta.as<Scalar>(), sequence_delta_scratch.as<Scalar>());
            else
                scatter_time_slice_cuda<Scalar>(
                    batch_size, T, H, T - 1,
                    output_delta.as<Scalar>(), sequence_delta_scratch.as<Scalar>());
        });
        dy_data = sequence_delta_scratch.get_data();
    }

    const void* x_data = shape.time_major
        ? cudnn_input_sequence.get_data() : input.get_data();
    const void* y_data = shape.time_major
        ? cudnn_output_sequence.get_data() : sequence_output.get_data();
    void* dx_data = shape.time_major || !input_delta.get_data()
        ? input_delta_scratch.get_data() : input_delta.get_data();

    cudnn_rnn_backward_(shape, true,
                        x_data, y_data, dy_data, dx_data,
                        forward_state, backward_scratch);

    if (shape.time_major && input_delta.get_data())
    {
        PROFILE_SCOPE("rnn:transpose_input_delta");
        input_delta.dispatch([&]<typename Scalar>()
        {
            time_batch_to_batch_time_cropped_cuda<Scalar>(
                batch_size, T, input_features, shape.input_features,
                input_delta_scratch.as<Scalar>(), input_delta.as<Scalar>());
        });
    }

    unpack_gradients_from_cudnn_(backward_scratch);
}

#else

void LongShortTermMemoryOperator::apply_gpu(const TensorView&, TensorView&, TensorView&,
                                            TensorView&, TensorView&, Buffer&, bool, bool) const OPENNN_CUDA_STUB_BODY(apply_gpu)

void LongShortTermMemoryOperator::apply_delta_gpu(
    const TensorView&, const TensorView&, const TensorView&, const TensorView&,
    const TensorView&, TensorView&, TensorView&, TensorView&,
    const Buffer&, Buffer&, bool) const OPENNN_CUDA_STUB_BODY(apply_delta_gpu)

#endif

LongShortTermMemory::LongShortTermMemory(const Shape& new_input_shape,
                                         const Shape& new_output_shape,
                                         const string& new_activation_function,
                                         const string& new_recurrent_activation_function,
                                         const string& new_label)
    : Layer(LayerType::LongShortTermMemory)
{
    operators = {&lstm_op};

    set(new_input_shape,
        new_output_shape,
        new_activation_function,
        new_recurrent_activation_function,
        new_label);
}

vector<TensorSpec> LongShortTermMemory::get_forward_specs(Index batch_size) const
{
    const Index T = get_time_steps();
    const Shape sequence_shape{batch_size, T, output_features};
    const Shape input_sequence_shape{
        batch_size, T, ((get_input_features() + 7) / 8) * 8};

    return {
        {sequence_shape,  compute_dtype},
        {sequence_shape,  compute_dtype},
        {sequence_shape,  compute_dtype},
        {sequence_shape,  compute_dtype},
        {sequence_shape,  compute_dtype},
        {sequence_shape,  compute_dtype},
        {sequence_shape,  compute_dtype},
        {input_sequence_shape, compute_dtype},
        {sequence_shape,  compute_dtype},
        {return_sequences ? sequence_shape : Shape{batch_size, output_features}, compute_dtype},
    };
}

vector<TensorSpec> LongShortTermMemory::get_backward_specs(Index batch_size) const
{
    if (!is_trainable) return {};

    const Shape input_delta_shape = Shape{batch_size}.append(get_input_shape());
    const Shape scratch_shape{batch_size, output_features};

    return {
        {input_delta_shape, compute_dtype},
        {scratch_shape,     compute_dtype},
        {scratch_shape,     compute_dtype},
        {scratch_shape,     compute_dtype},
        {scratch_shape,     compute_dtype},
        {scratch_shape,     compute_dtype},
        {scratch_shape,     compute_dtype},
        {{batch_size, get_time_steps(), output_features}, compute_dtype},
        {{batch_size, get_time_steps(), ((get_input_features() + 7) / 8) * 8}, compute_dtype},
    };
}

void LongShortTermMemory::configure_operators()
{
    lstm_op.set(get_input_features(),
                output_features,
                get_time_steps(),
                lstm_op.activation_function,
                lstm_op.recurrent_activation_function,
                compute_dtype);

    lstm_op.return_sequences = return_sequences;

    using enum LongShortTermMemoryOperator::ForwardSlot;
    using enum LongShortTermMemoryOperator::BackwardSlot;

    lstm_op.input_slots = {InputSlot};
    lstm_op.output_slots = {
        ForgetGateSlot,
        InputGateSlot,
        CandidateGateSlot,
        OutputGateSlot,
        CellStateSlot,
        HiddenStateSlot,
        CellActivationSlot,
        OutputSlot
    };
    lstm_op.output_delta_slots = {OutputDeltaSlot};
    lstm_op.input_delta_slots = {InputDeltaSlot};
}

void LongShortTermMemory::set_return_sequences(bool value)
{
    if (return_sequences == value) return;
    return_sequences = value;
    configure_operators();
}

void LongShortTermMemory::set(const Shape& new_input_shape,
                              const Shape& new_output_shape,
                              const string& new_activation_function,
                              const string& new_recurrent_activation_function,
                              const string& new_label)
{
    set_label(new_label);
    set_activation_function(new_activation_function);
    set_recurrent_activation_function(new_recurrent_activation_function);

    if (new_input_shape.empty() && new_output_shape.empty())
    {
        input_shape = {};
        output_features = 0;
        return configure_operators();
    }

    check_rank(new_input_shape, {2}, "LongShortTermMemory", "input");
    check_rank(new_output_shape, {1}, "LongShortTermMemory", "output");

    input_shape = new_input_shape;
    output_features = new_output_shape[0];

    configure_operators();
}

void LongShortTermMemory::apply_input_shape(const Shape& new_input_shape)
{
    check_rank(new_input_shape, {2}, "LongShortTermMemory", "input");
    input_shape = new_input_shape;
    configure_operators();
}

void LongShortTermMemory::set_output_shape(const Shape& new_output_shape)
{
    check_rank(new_output_shape, {1, 2}, "LongShortTermMemory", "output");
    output_features = new_output_shape[new_output_shape.get_rank() - 1];
    configure_operators();
}

void LongShortTermMemory::set_activation_function(const string& new_activation_function)
{
    const ActivationFunction function = ActivationOperator::from_string(new_activation_function);

    using enum ActivationFunction;
    throw_if(function != Identity && function != Sigmoid && function != Tanh && function != ReLU,
             "LongShortTermMemory: unsupported activation function \"{}\".", new_activation_function);

    lstm_op.activation_function = function;
}

void LongShortTermMemory::set_recurrent_activation_function(const string& new_recurrent_activation_function)
{
    const ActivationFunction function = ActivationOperator::from_string(new_recurrent_activation_function);

    using enum ActivationFunction;
    throw_if(function != Identity && function != Sigmoid && function != Tanh && function != ReLU,
             "LongShortTermMemory: unsupported recurrent activation function \"{}\".",
                    new_recurrent_activation_function);

    lstm_op.recurrent_activation_function = function;
}

void LongShortTermMemory::read_JSON_body(const Json* lstm_layer_element)
{
    set_activation_function(read_json_string(lstm_layer_element, "Activation"));
    set_recurrent_activation_function(read_json_string(lstm_layer_element, "RecurrentActivation"));
    return_sequences = read_json_bool(lstm_layer_element, "ReturnSequences");
    configure_operators();
}

void LongShortTermMemory::write_JSON_body(JsonWriter& printer) const
{
    add_json_field(printer, "Activation", ActivationOperator::to_string(lstm_op.activation_function));
    add_json_field(printer, "RecurrentActivation", ActivationOperator::to_string(lstm_op.recurrent_activation_function));
    add_json_field(printer, "ReturnSequences", return_sequences);
}

string LongShortTermMemory::write_expression(const vector<string>& feature_names,
                                             const vector<string>& output_names) const
{
    if (parameters.size() < 12) return {};
    for (Index p = 0; p < 12; ++p)
        if (!parameters[p].get_data()) return {};

    VectorMap bf = parameters[0].as_vector();
    VectorMap bi = parameters[1].as_vector();
    VectorMap bg = parameters[2].as_vector();
    VectorMap bo = parameters[3].as_vector();

    MatrixMap Wf = parameters[4].as_matrix();
    MatrixMap Wi = parameters[5].as_matrix();
    MatrixMap Wg = parameters[6].as_matrix();
    MatrixMap Wo = parameters[7].as_matrix();

    MatrixMap Uf = parameters[8].as_matrix();
    MatrixMap Ui = parameters[9].as_matrix();
    MatrixMap Ug = parameters[10].as_matrix();
    MatrixMap Uo = parameters[11].as_matrix();

    const string act = ActivationOperator::to_string(lstm_op.activation_function);
    const string ract = ActivationOperator::to_string(lstm_op.recurrent_activation_function);

    const Index T = get_time_steps();
    const Index F = get_input_features();
    const Index H = output_features;

    const auto h_name = [&](Index t, Index j) -> string {
        const string internal = format("lstm_h_{}_{}", t, j);
        if (return_sequences)
        {
            const Index linear = t * H + j;
            if (linear < ssize(output_names)) return output_names[linear];
            return internal;
        }
        if (t == T - 1 && j < ssize(output_names)) return output_names[j];
        return internal;
    };

    ostringstream buf;
    buf.precision(10);

    const auto gate_expr = [&](const string& name,
                         const string& activation,
                         const VectorMap& b,
                         const MatrixMap& W,
                         const MatrixMap& U,
                         Index t, Index j)
    {
        buf << name << " = " << activation << "( " << b(j);
        for (Index k = 0; k < F; ++k)
        {
            const Index feature_index = t * F + k;
            if (feature_index < ssize(feature_names))
                buf << " + (" << feature_names[feature_index] << "*" << W(k, j) << ")";
        }
        if (t > 0)
            for (Index p = 0; p < H; ++p)
                buf << " + (" << h_name(t - 1, p) << "*" << U(p, j) << ")";
        buf << " );\n";
    };

    for (Index t = 0; t < T; ++t)
    {
        for (Index j = 0; j < H; ++j)
        {
            const string f_var = format("lstm_f_{}_{}", t, j);
            const string i_var = format("lstm_i_{}_{}", t, j);
            const string g_var = format("lstm_g_{}_{}", t, j);
            const string o_var = format("lstm_o_{}_{}", t, j);
            const string c_var = format("lstm_c_{}_{}", t, j);

            gate_expr(f_var, ract, bf, Wf, Uf, t, j);
            gate_expr(i_var, ract, bi, Wi, Ui, t, j);
            gate_expr(g_var, act,  bg, Wg, Ug, t, j);
            gate_expr(o_var, ract, bo, Wo, Uo, t, j);

            if (t > 0)
                buf << c_var << " = (" << f_var << " * lstm_c_" << (t - 1) << "_" << j
                    << ") + (" << i_var << " * " << g_var << ");\n";
            else
                buf << c_var << " = " << i_var << " * " << g_var << ";\n";

            buf << h_name(t, j) << " = " << o_var << " * " << act << "( " << c_var << " );\n";
        }
    }

    return buf.str();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
