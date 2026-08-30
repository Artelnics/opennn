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
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <mutex>
#include <tuple>
#include "opennn/neural_network/layers/kernel_recurrent.cuh"

#ifdef EIGEN_USE_MKL_ALL
#include <mkl_vml.h>
#endif

#ifdef OPENNN_HAS_ONEDNN
// MKL's direct-call header exposes the Fortran BLAS entry points as
// function-like macros. oneDNN also declares dnnl::sgemm, so hide the MKL
// spellings only while parsing the oneDNN C++ API and restore them afterward.
#ifdef sgemm
#pragma push_macro("sgemm")
#undef sgemm
#define OPENNN_RESTORE_MKL_SGEMM
#endif
#ifdef sgemm_
#pragma push_macro("sgemm_")
#undef sgemm_
#define OPENNN_RESTORE_MKL_SGEMM_
#endif
#include <oneapi/dnnl/dnnl.hpp>
#ifdef OPENNN_RESTORE_MKL_SGEMM_
#pragma pop_macro("sgemm_")
#undef OPENNN_RESTORE_MKL_SGEMM_
#endif
#ifdef OPENNN_RESTORE_MKL_SGEMM
#pragma pop_macro("sgemm")
#undef OPENNN_RESTORE_MKL_SGEMM
#endif
#endif

namespace opennn
{

namespace
{

MatrixR concat_gate_columns(span<const TensorView* const> gates)
{
    const MatrixMap first = gates[0]->as_matrix();
    const Index rows = first.rows();
    const Index columns = first.cols();

    MatrixR concatenated(rows, Index(gates.size()) * columns);

    for (size_t gate = 0; gate < gates.size(); ++gate)
        concatenated.middleCols(Index(gate) * columns, columns) = gates[gate]->as_matrix();

    return concatenated;
}

VectorR concat_gate_biases(span<const TensorView* const> gates)
{
    const Index size = gates[0]->as_vector().size();

    VectorR concatenated(Index(gates.size()) * size);

    for (size_t gate = 0; gate < gates.size(); ++gate)
        concatenated.segment(Index(gate) * size, size) = gates[gate]->as_vector();

    return concatenated;
}

void zero_if_linked(const TensorView& view)
{
    if (view.get_data()) view.setZero();
}

void zero_linked(span<const TensorView* const> views)
{
    for (const TensorView* view : views) zero_if_linked(*view);
}

void set_random_uniform_linked(span<const TensorView* const> views, float min, float max)
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

    // Four gates each, in the order the gate accessors report: biases, then
    // input weights, then recurrent weights.
    vector<TensorSpec> specs;
    specs.reserve(3 * GateCount);

    for (const Shape& shape : {Shape{output_features},
                               Shape{input_features, output_features},
                               Shape{output_features, output_features}})
        specs.insert(specs.end(), GateCount, {shape, compute_dtype});

    return specs;
}

vector<Operator::ParameterSlot> LongShortTermMemoryOperator::parameter_slots()
{
    return {
        {&forget_bias,    &forget_bias_gradient},
        {&input_bias,     &input_bias_gradient},
        {&candidate_bias, &candidate_bias_gradient},
        {&output_bias,    &output_bias_gradient},

        {&forget_weights,    &forget_weight_gradient},
        {&input_weights,     &input_weight_gradient},
        {&candidate_weights, &candidate_weight_gradient},
        {&output_weights,    &output_weight_gradient},

        {&forget_recurrent_weights,    &forget_recurrent_weight_gradient},
        {&input_recurrent_weights,     &input_recurrent_weight_gradient},
        {&candidate_recurrent_weights, &candidate_recurrent_weight_gradient},
        {&output_recurrent_weights,    &output_recurrent_weight_gradient},
    };
}

void LongShortTermMemoryOperator::set_parameters_random()
{
    // The forget gate starts biased open; the other three start at zero.
    if (forget_bias.get_data()) forget_bias.fill(1.0f);
    const GateViews biases = gate_biases();
    zero_linked(span(biases).subspan(1));

    if (forget_weights.get_data())
        set_random_uniform_linked(gate_weights(), -0.1f, 0.1f);

    if (forget_recurrent_weights.get_data())
        set_random_uniform_linked(gate_recurrent_weights(), -0.1f, 0.1f);
}

void LongShortTermMemoryOperator::set_parameters_glorot()
{
    if (forget_bias.get_data()) forget_bias.fill(1.0f);
    const GateViews biases = gate_biases();
    zero_linked(span(biases).subspan(1));

    if (forget_weights.get_data())
    {
        const float limit = glorot_limit(input_features, output_features);
        set_random_uniform_linked(gate_weights(), -limit, limit);
    }

    if (forget_recurrent_weights.get_data())
        for (const TensorView* recurrent : gate_recurrent_weights())
            set_random_orthogonal(recurrent->as_matrix());
}

void LongShortTermMemoryOperator::set_parameters_pytorch()
{
    const float limit = 1.0f / sqrt(float(output_features > 0 ? output_features : 1));

    if (forget_bias.get_data())
        set_random_uniform_linked(gate_biases(), -limit, limit);

    if (forget_weights.get_data())
        set_random_uniform_linked(gate_weights(), -limit, limit);

    if (forget_recurrent_weights.get_data())
        set_random_uniform_linked(gate_recurrent_weights(), -limit, limit);
}

#ifdef OPENNN_HAS_ONEDNN

namespace
{

using namespace dnnl;

constexpr uint64_t ONEDNN_LSTM_STATE_MAGIC = 0x4f4e4e4c53544d31ULL;
constexpr size_t ONEDNN_LSTM_STATE_HEADER = 64;

size_t align_onednn_offset(size_t value)
{
    return (value + 63U) & ~size_t(63U);
}

class ScopedStaticOmpTeams
{
public:
    ScopedStaticOmpTeams() : previous_dynamic(omp_get_dynamic())
    {
        if (previous_dynamic) omp_set_dynamic(0);
    }

    ~ScopedStaticOmpTeams()
    {
        if (previous_dynamic) omp_set_dynamic(previous_dynamic);
    }

private:
    int previous_dynamic;
};

struct OneDnnLstmPlan
{
    engine cpu_engine{engine::kind::cpu, 0};
    memory::desc src_layer_desc;
    memory::desc dst_layer_desc;
    memory::desc iter_desc;
    memory::desc user_weights_layer_desc;
    memory::desc user_weights_iter_desc;
    memory::desc bias_desc;
    lstm_forward::primitive_desc forward_desc;
    lstm_forward forward;
    unique_ptr<lstm_backward::primitive_desc> backward_desc;
    unique_ptr<lstm_backward> backward;

    size_t weights_layer_offset = 0;
    size_t weights_iter_offset = 0;
    size_t workspace_offset = 0;
    size_t scratchpad_offset = 0;
    size_t iter_state_offset = 0;
    size_t iter_state_bytes = 0;
    size_t state_bytes = 0;

    OneDnnLstmPlan(Index batch, Index time, Index inputs, Index hidden, bool training)
    {
        const memory::dims src_dims{time, batch, inputs};
        const memory::dims dst_dims{time, batch, hidden};

        // oneDNN 2.x drops to its reference RNN for batch-time-channel
        // strides. Feed its native time-major layout so the JIT recurrent
        // kernel is selected; the explicit input transpose is small beside
        // the hidden-state traffic it removes.
        src_layer_desc = memory::desc(src_dims, memory::data_type::f32,
                                      memory::format_tag::tnc);
        dst_layer_desc = memory::desc(dst_dims, memory::data_type::f32,
                                      memory::format_tag::tnc);
        if (!training)
            iter_desc = memory::desc({1, 1, batch, hidden},
                                     memory::data_type::f32,
                                     memory::format_tag::ldnc);

        user_weights_layer_desc = memory::desc(
            {1, 1, inputs, 4, hidden}, memory::data_type::f32,
            memory::format_tag::ldigo);
        user_weights_iter_desc = memory::desc(
            {1, 1, hidden, 4, hidden}, memory::data_type::f32,
            memory::format_tag::ldigo);
        bias_desc = memory::desc({1, 1, 4, hidden}, memory::data_type::f32,
                                 memory::format_tag::ldgo);

        const memory::desc any_weights_layer(
            {1, 1, inputs, 4, hidden}, memory::data_type::f32,
            memory::format_tag::any);
        const memory::desc any_weights_iter(
            {1, 1, hidden, 4, hidden}, memory::data_type::f32,
            memory::format_tag::any);
        const memory::desc& forward_weights_layer_desc =
            training ? user_weights_layer_desc : any_weights_layer;
        const memory::desc& forward_weights_iter_desc =
            training ? user_weights_iter_desc : any_weights_iter;
        const memory::desc none;
        primitive_attr attributes;
        attributes.set_scratchpad_mode(scratchpad_mode::user);

#if DNNL_VERSION_MAJOR >= 3
        forward_desc = lstm_forward::primitive_desc(
            cpu_engine,
            training ? prop_kind::forward_training : prop_kind::forward_inference,
            rnn_direction::unidirectional_left2right,
            src_layer_desc, iter_desc, iter_desc,
            forward_weights_layer_desc, forward_weights_iter_desc, bias_desc,
            dst_layer_desc, iter_desc, iter_desc,
            attributes);
#else
        const lstm_forward::desc descriptor(
            training ? prop_kind::forward_training : prop_kind::forward_inference,
            rnn_direction::unidirectional_left2right,
            src_layer_desc, iter_desc, iter_desc,
            forward_weights_layer_desc, forward_weights_iter_desc, bias_desc,
            dst_layer_desc, iter_desc, iter_desc);

        forward_desc = lstm_forward::primitive_desc(
            descriptor, attributes, cpu_engine);
#endif
        forward = lstm_forward(forward_desc);

        weights_layer_offset = ONEDNN_LSTM_STATE_HEADER;
        weights_iter_offset = align_onednn_offset(
            weights_layer_offset + forward_desc.weights_layer_desc().get_size());
        workspace_offset = align_onednn_offset(
            weights_iter_offset + forward_desc.weights_iter_desc().get_size());
        scratchpad_offset = align_onednn_offset(
            workspace_offset + (training ? forward_desc.workspace_desc().get_size() : 0));
        iter_state_offset = align_onednn_offset(
            scratchpad_offset + forward_desc.scratchpad_desc().get_size());
        iter_state_bytes = training ? 0 : iter_desc.get_size();
        state_bytes = align_onednn_offset(
            iter_state_offset + 4 * iter_state_bytes);

        if (!training) return;

        // oneDNN's f32 backward reference kernel consumes weights with the
        // input channel innermost (ldgoi), while its forward kernel uses
        // ldigo. The backward call therefore has its own packed views.
        const memory::desc backward_weights_layer_desc(
            {1, 1, inputs, 4, hidden}, memory::data_type::f32,
            memory::format_tag::ldgoi);
        const memory::desc backward_weights_iter_desc(
            {1, 1, hidden, 4, hidden}, memory::data_type::f32,
            memory::format_tag::ldgoi);

#if DNNL_VERSION_MAJOR >= 3
        backward_desc = make_unique<lstm_backward::primitive_desc>(
            cpu_engine,
            prop_kind::backward,
            rnn_direction::unidirectional_left2right,
            src_layer_desc, none, none,
            backward_weights_layer_desc,
            backward_weights_iter_desc, bias_desc,
            dst_layer_desc, none, none,
            src_layer_desc, none, none,
            user_weights_layer_desc, user_weights_iter_desc, bias_desc,
            dst_layer_desc, none, none,
            forward_desc,
            attributes);
#else
        const lstm_backward::desc backward_descriptor(
            prop_kind::backward,
            rnn_direction::unidirectional_left2right,
            src_layer_desc, none, none,
            backward_weights_layer_desc,
            backward_weights_iter_desc, bias_desc,
            dst_layer_desc, none, none,
            src_layer_desc, none, none,
            user_weights_layer_desc, user_weights_iter_desc, bias_desc,
            dst_layer_desc, none, none);

        backward_desc = make_unique<lstm_backward::primitive_desc>(
            backward_descriptor, attributes, cpu_engine, forward_desc);
#endif
        backward = make_unique<lstm_backward>(*backward_desc);
    }
};

shared_ptr<OneDnnLstmPlan> onednn_lstm_plan(Index batch, Index time,
                                            Index inputs, Index hidden,
                                            bool training)
{
    using Key = tuple<Index, Index, Index, Index, bool>;
    // Keep descriptors and JIT kernels alive across batches. A weak cache
    // recreated the primitive on every call because no other long-lived
    // owner exists after apply_onednn() returns.
    static map<Key, shared_ptr<OneDnnLstmPlan>> plans;
    static mutex plans_mutex;

    const Key key{batch, time, inputs, hidden, training};
    const lock_guard lock(plans_mutex);

    if (const auto found = plans.find(key); found != plans.end())
        return found->second;

    auto plan = make_shared<OneDnnLstmPlan>(batch, time, inputs, hidden, training);
    plans[key] = plan;
    return plan;
}

void pack_onednn_gate_matrix(vector<float>& packed,
                             initializer_list<const TensorView*> gates,
                             Index rows, Index hidden)
{
    packed.resize(size_t(rows * 4 * hidden));
    Index gate = 0;

    // oneDNN's native LSTM order is input, forget, cell, output. OpenNN keeps
    // forget first, so callers pass the views in oneDNN order here.
    for (const TensorView* view : gates)
    {
        const float* source = view->as<float>();
        for (Index row = 0; row < rows; ++row)
            memcpy(packed.data() + size_t((row * 4 + gate) * hidden),
                   source + row * hidden, size_t(hidden) * sizeof(float));
        ++gate;
    }
}

void pack_onednn_bias(vector<float>& packed,
                      initializer_list<const TensorView*> gates,
                      Index hidden)
{
    packed.resize(size_t(4 * hidden));
    Index gate = 0;
    for (const TensorView* view : gates)
    {
        memcpy(packed.data() + size_t(gate * hidden), view->as<float>(),
               size_t(hidden) * sizeof(float));
        ++gate;
    }
}

void unpack_onednn_gate_matrix(const vector<float>& packed,
                               initializer_list<const TensorView*> gates,
                               Index rows, Index hidden)
{
    Index gate = 0;
    for (const TensorView* view : gates)
    {
        float* destination = view->as<float>();
        for (Index row = 0; row < rows; ++row)
            memcpy(destination + row * hidden,
                   packed.data() + size_t((row * 4 + gate) * hidden),
                   size_t(hidden) * sizeof(float));
        ++gate;
    }
}

void unpack_onednn_bias(const vector<float>& packed,
                        initializer_list<const TensorView*> gates,
                        Index hidden)
{
    Index gate = 0;
    for (const TensorView* view : gates)
    {
        memcpy(view->as<float>(), packed.data() + size_t(gate * hidden),
               size_t(hidden) * sizeof(float));
        ++gate;
    }
}

bool onednn_lstm_supported(const TensorView& input,
                           Index hidden_size,
                           Type compute_dtype,
                           ActivationFunction activation,
                           ActivationFunction recurrent_activation)
{
    return !input.is_cuda()
        && input.is_fp32()
        && hidden_size >= 128
        && compute_dtype == Type::FP32
        && activation == ActivationFunction::Tanh
        && recurrent_activation == ActivationFunction::Sigmoid
        && getenv("OPENNN_NO_ONEDNN_LSTM") == nullptr;
}

}

#endif

bool LongShortTermMemoryOperator::apply_onednn(
    const TensorView& input,
    TensorView& output,
    TensorView& sequence_output,
    TensorView& input_sequence_scratch,
    TensorView& output_sequence_scratch,
    Buffer& forward_state,
    bool is_training) const
{
#ifdef OPENNN_HAS_ONEDNN
    if (!onednn_lstm_supported(input, output_features, compute_dtype, activation_function,
                               recurrent_activation_function))
        return false;

    const Index batch = input.get_shape()[0];
    if (batch == 0 || time_steps == 0 || output_features == 0) return false;

    try
    {
        shared_ptr<OneDnnLstmPlan> plan = onednn_lstm_plan(
            batch, time_steps, input_features, output_features, is_training);

        forward_state.resize_bytes(Index(plan->state_bytes), Device::CPU);
        *forward_state.as<uint64_t>() = 0;
        char* const state = forward_state.as<char>();

        vector<float> layer_weights;
        vector<float> iter_weights;
        vector<float> bias;
        pack_onednn_gate_matrix(layer_weights,
                                {&input_weights, &forget_weights,
                                 &candidate_weights, &output_weights},
                                input_features, output_features);
        pack_onednn_gate_matrix(iter_weights,
                                {&input_recurrent_weights, &forget_recurrent_weights,
                                 &candidate_recurrent_weights, &output_recurrent_weights},
                                output_features, output_features);
        pack_onednn_bias(bias,
                         {&input_bias, &forget_bias, &candidate_bias, &output_bias},
                         output_features);

        const float* batch_major_input = input.as<float>();
        float* time_major_input = input_sequence_scratch.as<float>();

        {
            PROFILE_SCOPE_HOST("rnn:onednn_transpose_input");
            #pragma omp parallel for schedule(static)
            for (Index time = 0; time < time_steps; ++time)
                for (Index sample = 0; sample < batch; ++sample)
                    memcpy(time_major_input + (time * batch + sample) * input_features,
                           batch_major_input + (sample * time_steps + time) * input_features,
                           size_t(input_features) * sizeof(float));
        }

        stream execution_stream(plan->cpu_engine);
        memory user_layer(plan->user_weights_layer_desc, plan->cpu_engine,
                          layer_weights.data());
        memory packed_layer(plan->forward_desc.weights_layer_desc(), plan->cpu_engine,
                            state + plan->weights_layer_offset);
        memory user_iter(plan->user_weights_iter_desc, plan->cpu_engine,
                         iter_weights.data());
        memory packed_iter(plan->forward_desc.weights_iter_desc(), plan->cpu_engine,
                           state + plan->weights_iter_offset);

        reorder(user_layer, packed_layer).execute(execution_stream, user_layer, packed_layer);
        reorder(user_iter, packed_iter).execute(execution_stream, user_iter, packed_iter);

        unordered_map<int, memory> arguments{
            {DNNL_ARG_SRC_LAYER,
             memory(plan->src_layer_desc, plan->cpu_engine,
                    time_major_input)},
            {DNNL_ARG_WEIGHTS_LAYER, packed_layer},
            {DNNL_ARG_WEIGHTS_ITER, packed_iter},
            {DNNL_ARG_BIAS,
             memory(plan->bias_desc, plan->cpu_engine, bias.data())},
            {DNNL_ARG_DST_LAYER,
             memory(plan->dst_layer_desc, plan->cpu_engine,
                    output_sequence_scratch.get_data())},
            {DNNL_ARG_SCRATCHPAD,
             memory(plan->forward_desc.scratchpad_desc(), plan->cpu_engine,
                    state + plan->scratchpad_offset)}
        };

        if (!is_training)
        {
            void* const initial_hidden = state + plan->iter_state_offset;
            void* const initial_cell = state + plan->iter_state_offset
                                     + plan->iter_state_bytes;
            memset(initial_hidden, 0, 2 * plan->iter_state_bytes);
            arguments.emplace(DNNL_ARG_SRC_ITER,
                memory(plan->iter_desc, plan->cpu_engine, initial_hidden));
            arguments.emplace(DNNL_ARG_SRC_ITER_C,
                memory(plan->iter_desc, plan->cpu_engine, initial_cell));
            arguments.emplace(DNNL_ARG_DST_ITER,
                memory(plan->iter_desc, plan->cpu_engine,
                       state + plan->iter_state_offset + 2 * plan->iter_state_bytes));
            arguments.emplace(DNNL_ARG_DST_ITER_C,
                memory(plan->iter_desc, plan->cpu_engine,
                       state + plan->iter_state_offset + 3 * plan->iter_state_bytes));
        }

        if (is_training)
            arguments.emplace(DNNL_ARG_WORKSPACE,
                memory(plan->forward_desc.workspace_desc(), plan->cpu_engine,
                       state + plan->workspace_offset));

        {
            PROFILE_SCOPE_HOST("rnn:onednn_forward");
            const ScopedStaticOmpTeams static_teams;
            plan->forward.execute(execution_stream, arguments);
            execution_stream.wait();
        }

        if (!return_sequences)
        {
            const float* sequence = output_sequence_scratch.as<float>();
            float* result = output.as<float>();
            memcpy(result,
                   sequence + (time_steps - 1) * batch * output_features,
                   size_t(batch * output_features) * sizeof(float));
        }
        else
        {
            const float* time_major_output = output_sequence_scratch.as<float>();
            float* batch_major_output = sequence_output.as<float>();

            PROFILE_SCOPE_HOST("rnn:onednn_transpose_output");
            #pragma omp parallel for schedule(static)
            for (Index time = 0; time < time_steps; ++time)
                for (Index sample = 0; sample < batch; ++sample)
                    memcpy(batch_major_output + (sample * time_steps + time) * output_features,
                           time_major_output + (time * batch + sample) * output_features,
                           size_t(output_features) * sizeof(float));
        }

        *forward_state.as<uint64_t>() = ONEDNN_LSTM_STATE_MAGIC;
        return true;
    }
    catch (const dnnl::error& error)
    {
        if (!forward_state.empty()) *forward_state.as<uint64_t>() = 0;
        if (getenv("OPENNN_ONEDNN_REPORT"))
            cerr << "oneDNN LSTM forward unavailable: " << error.what() << '\n';
        return false;
    }
#else
    (void)input; (void)output; (void)sequence_output;
    (void)input_sequence_scratch; (void)output_sequence_scratch;
    (void)forward_state; (void)is_training;
    return false;
#endif
}

bool LongShortTermMemoryOperator::apply_delta_onednn(
    const TensorView& input,
    const TensorView& input_sequence,
    const TensorView& sequence_output,
    const TensorView& output_delta,
    TensorView& input_delta,
    TensorView& sequence_delta_scratch,
    TensorView& input_delta_scratch,
    const Buffer& forward_state,
    Buffer& backward_scratch,
    bool return_seq) const
{
#ifdef OPENNN_HAS_ONEDNN
    if (!onednn_lstm_supported(input, output_features, compute_dtype, activation_function,
                               recurrent_activation_function)
        || forward_state.byte_size() < Index(ONEDNN_LSTM_STATE_HEADER)
        || *forward_state.as<uint64_t>() != ONEDNN_LSTM_STATE_MAGIC)
        return false;

    const Index batch = input.get_shape()[0];
    shared_ptr<OneDnnLstmPlan> plan = onednn_lstm_plan(
        batch, time_steps, input_features, output_features, true);
    throw_if(!plan->backward || forward_state.byte_size() < Index(plan->state_bytes),
             "oneDNN LSTM backward has no matching training state.");

    float* diff_destination = nullptr;

    if (return_seq)
    {
        const float* batch_major_delta = output_delta.as<float>();
        diff_destination = sequence_delta_scratch.as<float>();

        PROFILE_SCOPE_HOST("rnn:onednn_transpose_output_delta");
        #pragma omp parallel for schedule(static)
        for (Index time = 0; time < time_steps; ++time)
            for (Index sample = 0; sample < batch; ++sample)
                memcpy(diff_destination + (time * batch + sample) * output_features,
                       batch_major_delta + (sample * time_steps + time) * output_features,
                       size_t(output_features) * sizeof(float));
    }
    else
    {
        sequence_delta_scratch.setZero();
        diff_destination = sequence_delta_scratch.as<float>();
        const float* final_delta = output_delta.as<float>();

        memcpy(diff_destination + (time_steps - 1) * batch * output_features,
               final_delta, size_t(batch * output_features) * sizeof(float));
    }

    vector<float> discarded_input_delta;
    float* diff_source = nullptr;
    if (!input_delta.empty())
        diff_source = input_delta_scratch.as<float>();
    else
    {
        discarded_input_delta.resize(size_t(batch * time_steps * input_features));
        diff_source = discarded_input_delta.data();
    }

    vector<float> bias;
    vector<float> diff_layer_weights(size_t(input_features * 4 * output_features));
    vector<float> diff_iter_weights(size_t(output_features * 4 * output_features));
    vector<float> diff_bias(size_t(4 * output_features));
    vector<char> packed_diff_layer(
        plan->backward_desc->diff_weights_layer_desc().get_size());
    vector<char> packed_diff_iter(
        plan->backward_desc->diff_weights_iter_desc().get_size());

    pack_onednn_bias(bias,
                     {&input_bias, &forget_bias, &candidate_bias, &output_bias},
                     output_features);

    vector<float> layer_weights;
    vector<float> iter_weights;
    pack_onednn_gate_matrix(layer_weights,
                            {&input_weights, &forget_weights,
                             &candidate_weights, &output_weights},
                            input_features, output_features);
    pack_onednn_gate_matrix(iter_weights,
                            {&input_recurrent_weights, &forget_recurrent_weights,
                             &candidate_recurrent_weights, &output_recurrent_weights},
                            output_features, output_features);

    backward_scratch.resize_bytes(
        Index(plan->backward_desc->scratchpad_desc().get_size()), Device::CPU);

    char* const state = static_cast<char*>(const_cast<void*>(forward_state.data()));
    stream execution_stream(plan->cpu_engine);
    vector<char> backward_layer_weights(
        plan->backward_desc->weights_layer_desc().get_size());
    vector<char> backward_iter_weights(
        plan->backward_desc->weights_iter_desc().get_size());
    memory user_layer(plan->user_weights_layer_desc, plan->cpu_engine,
                      layer_weights.data());
    memory backward_layer(plan->backward_desc->weights_layer_desc(),
                          plan->cpu_engine, backward_layer_weights.data());
    memory user_iter(plan->user_weights_iter_desc, plan->cpu_engine,
                     iter_weights.data());
    memory backward_iter(plan->backward_desc->weights_iter_desc(),
                         plan->cpu_engine, backward_iter_weights.data());
    reorder(user_layer, backward_layer).execute(
        execution_stream, user_layer, backward_layer);
    reorder(user_iter, backward_iter).execute(
        execution_stream, user_iter, backward_iter);
    execution_stream.wait();

    unordered_map<int, memory> arguments{
        {DNNL_ARG_SRC_LAYER,
         memory(plan->src_layer_desc, plan->cpu_engine,
                const_cast<void*>(input_sequence.get_data()))},
        {DNNL_ARG_WEIGHTS_LAYER, backward_layer},
        {DNNL_ARG_WEIGHTS_ITER, backward_iter},
        {DNNL_ARG_BIAS,
         memory(plan->bias_desc, plan->cpu_engine, bias.data())},
        {DNNL_ARG_DST_LAYER,
         memory(plan->dst_layer_desc, plan->cpu_engine,
                const_cast<void*>(sequence_output.get_data()))},
        {DNNL_ARG_WORKSPACE,
         memory(plan->forward_desc.workspace_desc(), plan->cpu_engine,
                state + plan->workspace_offset)},
        {DNNL_ARG_DIFF_SRC_LAYER,
         memory(plan->src_layer_desc, plan->cpu_engine, diff_source)},
        {DNNL_ARG_DIFF_WEIGHTS_LAYER,
         memory(plan->backward_desc->diff_weights_layer_desc(), plan->cpu_engine,
                packed_diff_layer.data())},
        {DNNL_ARG_DIFF_WEIGHTS_ITER,
         memory(plan->backward_desc->diff_weights_iter_desc(), plan->cpu_engine,
                packed_diff_iter.data())},
        {DNNL_ARG_DIFF_BIAS,
         memory(plan->bias_desc, plan->cpu_engine, diff_bias.data())},
        {DNNL_ARG_DIFF_DST_LAYER,
         memory(plan->dst_layer_desc, plan->cpu_engine, diff_destination)},
        {DNNL_ARG_SCRATCHPAD,
         memory(plan->backward_desc->scratchpad_desc(), plan->cpu_engine,
                backward_scratch.data())}
    };

    {
        PROFILE_SCOPE_HOST("rnn:onednn_backward");
        const ScopedStaticOmpTeams static_teams;
        plan->backward->execute(execution_stream, arguments);
        execution_stream.wait();
    }

    memory packed_diff_layer_memory(
        plan->backward_desc->diff_weights_layer_desc(), plan->cpu_engine,
        packed_diff_layer.data());
    memory user_diff_layer_memory(
        plan->user_weights_layer_desc, plan->cpu_engine,
        diff_layer_weights.data());
    memory packed_diff_iter_memory(
        plan->backward_desc->diff_weights_iter_desc(), plan->cpu_engine,
        packed_diff_iter.data());
    memory user_diff_iter_memory(
        plan->user_weights_iter_desc, plan->cpu_engine,
        diff_iter_weights.data());

    reorder(packed_diff_layer_memory, user_diff_layer_memory).execute(
        execution_stream, packed_diff_layer_memory, user_diff_layer_memory);
    reorder(packed_diff_iter_memory, user_diff_iter_memory).execute(
        execution_stream, packed_diff_iter_memory, user_diff_iter_memory);
    execution_stream.wait();

    if (!input_delta.empty())
    {
        const float* time_major_delta = diff_source;
        float* batch_major_delta = input_delta.as<float>();

        PROFILE_SCOPE_HOST("rnn:onednn_transpose_input_delta");
        #pragma omp parallel for schedule(static)
        for (Index time = 0; time < time_steps; ++time)
            for (Index sample = 0; sample < batch; ++sample)
                memcpy(batch_major_delta + (sample * time_steps + time) * input_features,
                       time_major_delta + (time * batch + sample) * input_features,
                       size_t(input_features) * sizeof(float));
    }

    unpack_onednn_gate_matrix(diff_layer_weights,
                              {&input_weight_gradient, &forget_weight_gradient,
                               &candidate_weight_gradient, &output_weight_gradient},
                              input_features, output_features);
    unpack_onednn_gate_matrix(diff_iter_weights,
                              {&input_recurrent_weight_gradient, &forget_recurrent_weight_gradient,
                               &candidate_recurrent_weight_gradient, &output_recurrent_weight_gradient},
                              output_features, output_features);
    unpack_onednn_bias(diff_bias,
                       {&input_bias_gradient, &forget_bias_gradient,
                        &candidate_bias_gradient, &output_bias_gradient},
                       output_features);

    return true;
#else
    (void)input; (void)input_sequence; (void)sequence_output; (void)output_delta;
    (void)input_delta; (void)sequence_delta_scratch; (void)input_delta_scratch;
    (void)forward_state; (void)backward_scratch; (void)return_seq;
    return false;
#endif
}

void LongShortTermMemoryOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, ForwardPropagationMode pass)
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
                         return_sequences, is_training(pass),
                         forward_propagation.get_parameters_version());

    if (apply_onednn(input, output,
                     return_sequences ? output : hidden_state,
                     forward_slots[CudnnInputSequenceSlot],
                     forward_slots[CudnnOutputSequenceSlot],
                     forward_propagation.layer_state_storage[layer],
                     is_training(pass)))
        return;

    apply(input, output, forget_gate, input_gate, candidate_gate, output_gate,
          cell_state, hidden_state, cell_activation, is_training(pass));
}

void LongShortTermMemoryOperator::apply(const TensorView& input,
                                      TensorView& output,
                                      TensorView& forget_gate,
                                      TensorView& input_gate,
                                      TensorView& candidate_gate,
                                      TensorView& output_gate,
                                      TensorView& cell_state,
                                      TensorView& hidden_state,
                                      TensorView& cell_activation,
                                      const bool training) const
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

    if (H >= 96)
    {
        const MatrixR Wcat = concat_gate_columns(gate_weights());
        const MatrixR Ucat = concat_gate_columns(gate_recurrent_weights());
        const VectorR bcat = concat_gate_biases(gate_biases());

        const Index BT = batch_size * T;
        MatrixR Zin(BT, 4 * H);
        Zin.noalias() = Eigen::Map<const MatrixR>(x, BT, F) * Wcat;
        Zin.rowwise() += bcat.transpose();

        using StridedZ = Eigen::Map<const MatrixR, 0, Eigen::OuterStride<>>;
        using Row = Eigen::Map<Eigen::ArrayXf>;
        using ConstRow = Eigen::Map<const Eigen::ArrayXf>;

        const bool standard_gates =
            recurrent_activation_function == ActivationFunction::Sigmoid
            && activation_function == ActivationFunction::Tanh;

        MatrixR Z_c(batch_size, 4 * H);
        MatrixR h_c(batch_size, H);

        // Two different shapes of parallelism, one per phase.
        //
        // The recurrent GEMM is one 256x512x128 product per step and Eigen
        // threads it well, so it is left whole: splitting the batch across
        // threads first made every thread re-pack the 256 KiB Ucat, 384 times
        // a call. The elementwise tail is row-independent and transcendental
        // heavy, so that gets the batch split instead.
        //
        // Both used to be serial -- the GEMM sat in an `omp single` with Eigen
        // pinned to one thread, and it is the layer: 805 Mflop per call at
        // batch 256 and H 128.
        for (Index t = 0; t < T; ++t)
        {
            {
            PROFILE_SCOPE("rnn:strided_copy");
            Z_c = StridedZ(Zin.data() + t * 4 * H, batch_size, 4 * H,
                           Eigen::OuterStride<>(T * 4 * H));
            }
            {
            PROFILE_SCOPE("rnn:recurrent_gemm");
            if (t > 0)
                Z_c.noalias() += h_c * Ucat;
            }

            PROFILE_SCOPE("rnn:gates");

            if (standard_gates)
            {
                // Parallel over the batch, vectorised over hidden units.
                //
                // Both matter and neither is enough alone. The scalar form
                // called libm five times per element -- 3.9M transcendentals a
                // call at batch 256, T 24, H 128 -- and a scalar loop
                // vectorises neither `exp` nor `tanh`. But hoisting the whole
                // step into Eigen array expressions over the batch-by-H block
                // gave up the thread split and cost 7x more than it saved.
                //
                // A row of a step is contiguous in all four gates and in the
                // outputs, so each thread can take rows and let Eigen vectorise
                // along H. Written as two assignments rather than six named
                // temporaries, so Eigen fuses each into one pass and allocates
                // nothing.
                #pragma omp parallel for schedule(static)
                for (Index b = 0; b < batch_size; ++b)
                {
                    const Index step = (b * T + t) * H;
                    const float* Zrow = Z_c.data() + b * 4 * H;

                    const ConstRow zf(Zrow, H);
                    const ConstRow zi(Zrow + H, H);
                    const ConstRow zg(Zrow + 2 * H, H);
                    const ConstRow zo(Zrow + 3 * H, H);

                    Row cell_now(cells + step, H);
                    Row hidden_now(hidden + step, H);

                    const auto f = 1.0f / (1.0f + (-zf).exp());
                    const auto i = 1.0f / (1.0f + (-zi).exp());
                    const auto g = zg.tanh();
                    const auto o = 1.0f / (1.0f + (-zo).exp());

                    if (t > 0)
                        cell_now = f * ConstRow(cells + step - H, H) + i * g;
                    else
                        cell_now = i * g;

                    hidden_now = o * cell_now.tanh();

                    Row(h_c.data() + b * H, H) = hidden_now;

                    if (training)
                    {
                        Row(f_gate + step, H) = f;
                        Row(i_gate + step, H) = i;
                        Row(g_gate + step, H) = g;
                        Row(o_gate + step, H) = o;
                        Row(cell_act + step, H) = cell_now.tanh();
                    }

                    if (return_sequences) Row(y + step, H) = hidden_now;
                }
            }
            else
            {
            #pragma omp parallel for schedule(static)
            for (Index b = 0; b < batch_size; ++b)
            {
                const Index step = (b * T + t) * H;
                const float* Zrow = Z_c.data() + b * 4 * H;
                float* h_next = h_c.data() + b * H;
                const float* c_prev = t > 0 ? cells + (b * T + t - 1) * H : nullptr;

                for (Index h = 0; h < H; ++h)
                {
                    const float f = activation_forward_value(
                        recurrent_activation_function, Zrow[h]);
                    const float i = activation_forward_value(
                        recurrent_activation_function, Zrow[H + h]);
                    const float g = activation_forward_value(
                        activation_function, Zrow[2 * H + h]);
                    const float o = activation_forward_value(
                        recurrent_activation_function, Zrow[3 * H + h]);
                    const float c = f * (c_prev ? c_prev[h] : 0.0f) + i * g;
                    const float a = activation_forward_value(activation_function, c);
                    const float h_value = o * a;

                    if (training)
                    {
                        f_gate[step + h] = f;
                        i_gate[step + h] = i;
                        g_gate[step + h] = g;
                        o_gate[step + h] = o;
                        cell_act[step + h] = a;
                    }

                    cells[step + h] = c;
                    hidden[step + h] = h_value;
                    h_next[h] = h_value;
                    if (return_sequences) y[step + h] = h_value;
                }
            }
            }
        }

        if (!return_sequences)
            for (Index b = 0; b < batch_size; ++b)
                copy_n(hidden + (b * T + T - 1) * H, H, y + b * H);

        return;
    }

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

    if (apply_delta_onednn(input,
                           forward_slots[CudnnInputSequenceSlot],
                           forward_slots[CudnnOutputSequenceSlot],
                           output_delta,
                           input_delta,
                           backward_slots[CudnnOutputDeltaScratchSlot],
                           backward_slots[CudnnInputDeltaScratchSlot],
                           forward_propagation.layer_state_storage[layer],
                           back_propagation.layer_scratch_storage[layer],
                           return_sequences))
        return;

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

    zero_linked(gate_bias_gradients());
    zero_linked(gate_weight_gradients());
    zero_linked(gate_recurrent_weight_gradients());

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

        const MatrixR Wcat = concat_gate_columns(gate_weights());
        const MatrixR Ucat = concat_gate_columns(gate_recurrent_weights());

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

void LongShortTermMemoryOperator::pack_weights_to_cudnn_(Buffer& forward_state,
                                                        uint64_t parameters_version) const
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
                        weights, biases, forward_state, parameters_version);
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
                                      bool is_training,
                                      uint64_t parameters_version) const
{
    const Index batch_size = input.get_shape()[0];
    if (!input.get_data() || output_features == 0 || time_steps == 0 || batch_size == 0) return;

    drive_cudnn_forward_({input_features, output_features, time_steps, return_seq, true},
                         input, sequence_output_scratch, output,
                         cudnn_input_sequence, cudnn_output_sequence,
                         forward_state, is_training, parameters_version);
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

    if (input.get_shape()[0] == 0) return;

    drive_cudnn_backward_({input_features, output_features, time_steps, return_seq, true},
                          input, sequence_output, output_delta,
                          cudnn_input_sequence, cudnn_output_sequence,
                          input_delta, sequence_delta_scratch, input_delta_scratch,
                          forward_state, backward_scratch);
}

#else

void LongShortTermMemoryOperator::apply_gpu(const TensorView&, TensorView&, TensorView&,
                                            TensorView&, TensorView&, Buffer&, bool, bool,
                                            uint64_t) const OPENNN_CUDA_STUB_BODY(apply_gpu)

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
