//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "recurrent_operator.h"
#include "device_backend.h"
#include "random_utilities.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"

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
    if (views.size() < 3) return;
    bias              = views[0];
    input_weights     = views[1];
    recurrent_weights = views[2];
}

void RecurrentOperator::link_gradients(span<const TensorView> views)
{
    if (views.size() < 3) return;
    bias_gradient              = views[0];
    input_weight_gradient      = views[1];
    recurrent_weight_gradient  = views[2];
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
    auto& forward_slots = forward_propagation.forward_slots[layer];
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
    auto& forward_slots = forward_propagation.forward_slots[layer];
    auto& backward_slots = back_propagation.backward_slots[layer];

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

} // namespace

void RecurrentOperator::apply(const TensorView& input,
                            TensorView& hidden_states,
                            TensorView& activation_derivatives,
                            TensorView& output,
                            bool is_training)
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
                 format("RecurrentOperator CUDA: {} dtype does not match recurrent compute dtype.", name));
}

bool RecurrentOperator::cudnn_rnn_eligible_(const TensorView& reference) const
{
    return (activation == ActivationFunction::Tanh
            || activation == ActivationFunction::ReLU)
        && reference.is_fp32();
}

static bool rnn_persist_env_enabled()
{
    static const bool enabled = []() {
        const char* env = std::getenv("OPENNN_RNN_PERSIST");
        return !(env && string(env) == "0");
    }();
    return enabled;
}

void RecurrentOperator::ensure_cudnn_setup_(Index batch_size) const
{
    if (!persist_algo_failed_ && rnn_persist_env_enabled())
    {
        try
        {
            ensure_cudnn_setup_attempt_(batch_size);
            return;
        }
        catch (const std::exception&)
        {
            persist_algo_failed_ = true;
            rnn_desc.reset();
            cached_input_features = -1;
        }
    }
    ensure_cudnn_setup_attempt_(batch_size);
}

void RecurrentOperator::ensure_cudnn_setup_attempt_(Index batch_size) const
{
    persist_algo_active_ = !persist_algo_failed_ && rnn_persist_env_enabled();

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
            activation == ActivationFunction::ReLU ? CUDNN_RNN_RELU
                                                   : CUDNN_RNN_TANH,
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

        for (int lin = 0; lin < 2; ++lin)
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
    for (int s = 0; s < 2; ++s)
        if (shape_slots_[s].batch == batch_size && shape_slots_[s].time == T)
            slot_index = s;

    if (slot_index < 0)
    {
        slot_index = (shape_slots_[0].batch < 0) ? 0
                   : (shape_slots_[1].batch < 0) ? 1
                   : (shape_slots_[0].stamp <= shape_slots_[1].stamp ? 0 : 1);
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
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&slot.h_desc.handle));
        slot.h_desc.deleter = &cudnnDestroyTensorDescriptor;
        const int dimA[3]    = {1, int(batch_size), int(H)};
        const int strideA[3] = {int(batch_size * H), int(H), 1};
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(slot.h_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

        size_t work_bytes = 0;
        size_t reserve_bytes = 0;
        CHECK_CUDNN(cudnnGetRNNTempSpaceSizes(
            Backend::get_cudnn_handle(), rnn_desc,
            CUDNN_FWD_MODE_TRAINING, slot.x_desc,
            &work_bytes, &reserve_bytes));

        size_t inference_work_bytes = 0;
        CHECK_CUDNN(cudnnGetRNNTempSpaceSizes(
            Backend::get_cudnn_handle(), rnn_desc,
            CUDNN_FWD_MODE_INFERENCE, slot.x_desc,
            &inference_work_bytes, nullptr));

        workspace_buf.grow_to(Index(max(work_bytes, inference_work_bytes)));
        reserve_space_buf.grow_to(Index(reserve_bytes));
        dy_buf.grow_to(batch_size * T * H * Index(sizeof(float)));

    }

    shape_slots_[slot_index].stamp = ++shape_stamp_;
    active_shape_ = slot_index;

    cached_input_features  = F;
    cached_output_features = H;
}

void RecurrentOperator::pack_weights_to_cudnn_() const
{
    const Index F = input_features;
    const Index H = output_features;

    RnnCopySpec specs[RNN_COPY_MAX_REGIONS];
    int count = 0;
    if (cudnn_w_ptrs_[0] && input_weights.data)
        specs[count++] = {input_weights.as<float>(), cudnn_w_ptrs_[0], int(F), int(H), 1};
    if (cudnn_w_ptrs_[1] && recurrent_weights.data)
        specs[count++] = {recurrent_weights.as<float>(), cudnn_w_ptrs_[1], int(H), int(H), 1};
    if (cudnn_b_ptrs_[0] && bias.data)
        specs[count++] = {bias.as<float>(), cudnn_b_ptrs_[0], 1, int(H), 0};
    rnn_copy_regions_cuda(specs, count);
}

void RecurrentOperator::unpack_gradients_from_cudnn_() const
{
    const Index F = input_features;
    const Index H = output_features;

    RnnCopySpec specs[RNN_COPY_MAX_REGIONS];
    int count = 0;
    if (cudnn_gw_ptrs_[0] && input_weight_gradient.data)
        specs[count++] = {cudnn_gw_ptrs_[0],
                          const_cast<float*>(input_weight_gradient.as<float>()),
                          int(H), int(F), 1};
    if (cudnn_gw_ptrs_[1] && recurrent_weight_gradient.data)
        specs[count++] = {cudnn_gw_ptrs_[1],
                          const_cast<float*>(recurrent_weight_gradient.as<float>()),
                          int(H), int(H), 1};
    if (cudnn_gb_ptrs_[0] && bias_gradient.data)
        specs[count++] = {cudnn_gb_ptrs_[0],
                          const_cast<float*>(bias_gradient.as<float>()),
                          1, int(H), 0};
    rnn_copy_regions_cuda(specs, count);
}

void RecurrentOperator::apply_gpu_cudnn_(const TensorView& input,
                                         TensorView& hidden_states,
                                         TensorView& output,
                                         bool is_training)
{
    const Index batch_size = input.shape[0];

    ensure_cudnn_setup_(batch_size);
    pack_weights_to_cudnn_();

    auto run_forward = [&]() {
        const CudnnRnnShapeSlot& shape = active_shape();
        return cudnnRNNForward(
            Backend::get_cudnn_handle(),
            rnn_desc,
            is_training ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE,
            shape.seq_dev.as<int32_t>(),
            shape.x_desc, input.data,
            shape.y_desc, hidden_states.data,
            shape.h_desc, nullptr, nullptr,
            shape.h_desc, nullptr, nullptr,
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
        ensure_cudnn_setup_(batch_size);
        pack_weights_to_cudnn_();
        forward_status = run_forward();
    }
    CHECK_CUDNN(forward_status);

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

    ensure_cudnn_setup_(batch_size);

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

    const CudnnRnnShapeSlot& shape = active_shape();

    CHECK_CUDNN(cudnnRNNBackwardData_v8(
        Backend::get_cudnn_handle(),
        rnn_desc,
        shape.seq_dev.as<int32_t>(),
        shape.y_desc, hidden_states.data, dy_data,
        shape.x_desc, dx_data,
        shape.h_desc, nullptr, nullptr, nullptr,
        shape.h_desc, nullptr, nullptr, nullptr,
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
        shape.y_desc, hidden_states.data,
        size_t(dweight_space_buf.bytes), dweight_space_buf.data,
        size_t(workspace_buf.bytes), workspace_buf.data,
        size_t(reserve_space_buf.bytes), reserve_space_buf.data));

    unpack_gradients_from_cudnn_();
}

void RecurrentOperator::apply_gpu(const TensorView& input,
                            TensorView& hidden_states,
                            TensorView& activation_derivatives,
                            TensorView& output,
                            bool is_training)
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

        const cudaDataType_t axpy_dtype =
            (output_delta.is_fp32()) ? CUDA_R_32F :
            (output_delta.is_bf16()) ? CUDA_R_16BF :
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

void RecurrentOperator::apply_gpu(const TensorView&, TensorView&, TensorView&, TensorView&, bool)
{
    throw runtime_error("apply_gpu requires CUDA.");
}

void RecurrentOperator::apply_delta_gpu(const TensorView&, const TensorView&, const TensorView&,
                                  const TensorView&, TensorView&,
                                  TensorView&, TensorView&, TensorView&,
                                  TensorView&, TensorView&) const
{
    throw runtime_error("apply_delta_gpu requires CUDA.");
}

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
