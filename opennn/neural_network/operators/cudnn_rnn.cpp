//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D N N   R N N   S T A T E   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/operators/cudnn_rnn.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/device_backend.h"
#include "opennn/neural_network/layers/kernel_recurrent.cuh"

#ifdef OPENNN_HAS_CUDA

namespace opennn
{

// OPENNN_RNN_PERSIST=0 turns the persistent cuDNN RNN algorithm off.
static bool persist_env_enabled()
{
    static const bool enabled = env_flag_enabled("OPENNN_RNN_PERSIST", true);
    return enabled;
}

CudnnRnnShapeSlot& CudnnRnnState::cudnn_setup_(const CudnnRnnConfig& config,
                                               Index input_features,
                                               Index output_features,
                                               Index time_steps,
                                               Index batch_size,
                                               bool for_training) const
{
    BackendState& state = backend_state;
    if (!state.persist_algo_failed && persist_env_enabled())
    {
        try
        {
            return cudnn_setup_attempt_(config, input_features, output_features, time_steps,
                                        batch_size, for_training);
        }
        catch (const exception&)
        {
            state.persist_algo_failed = true;
            state.rnn_desc.reset();
            state.cached_input_features = -1;
        }
    }
    return cudnn_setup_attempt_(config, input_features, output_features, time_steps,
                                batch_size, for_training);
}

CudnnRnnShapeSlot& CudnnRnnState::cudnn_setup_attempt_(const CudnnRnnConfig& config,
                                                       Index input_features,
                                                       Index output_features,
                                                       Index time_steps,
                                                       Index batch_size,
                                                       bool for_training) const
{
    BackendState& state = backend_state;
    state.persist_algo_active = !state.persist_algo_failed && persist_env_enabled();

    const Index F = input_features;
    const Index H = output_features;
    const Index T = time_steps;
    const bool is_lstm = (config.cell_mode == CUDNN_LSTM);

    const bool topology_changed =
        state.cached_input_features  != F ||
        state.cached_output_features != H ||
        state.rnn_desc == nullptr;

    if (topology_changed)
    {
        state.rnn_desc.reset();
        CHECK_CUDNN(cudnnCreateRNNDescriptor(&state.rnn_desc.handle));
        state.rnn_desc.deleter = &cudnnDestroyRNNDescriptor;

        if (!state.dropout_desc)
        {
            CHECK_CUDNN(cudnnCreateDropoutDescriptor(&state.dropout_desc.handle));
            state.dropout_desc.deleter = &cudnnDestroyDropoutDescriptor;
        }
        size_t dropout_states_bytes = 0;
        CHECK_CUDNN(cudnnDropoutGetStatesSize(
            device::get_cudnn_handle(), &dropout_states_bytes));
        state.dropout_states.grow_to(Index(dropout_states_bytes));
        CHECK_CUDNN(cudnnSetDropoutDescriptor(
            state.dropout_desc, device::get_cudnn_handle(),
             0.0f,
            state.dropout_states.data(),
            size_t(state.dropout_states.byte_size()),
             0ULL));

        CHECK_CUDNN(cudnnSetRNNDescriptor_v8(
            state.rnn_desc,
            state.persist_algo_active ? CUDNN_RNN_ALGO_PERSIST_STATIC
                                      : CUDNN_RNN_ALGO_STANDARD,
            config.cell_mode,
            CUDNN_RNN_SINGLE_INP_BIAS,
            CUDNN_UNIDIRECTIONAL,
            CUDNN_LINEAR_INPUT,
            CUDNN_DATA_FLOAT,
            CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_OP_MATH,
            int(F),
            int(H),
              int(H),
            1,
            state.dropout_desc,
            state.persist_algo_active ? CUDNN_RNN_PADDED_IO_DISABLED
                                      : CUDNN_RNN_PADDED_IO_ENABLED));

        size_t weight_bytes = 0;
        CHECK_CUDNN(cudnnGetRNNWeightSpaceSize(
            device::get_cudnn_handle(), state.rnn_desc, &weight_bytes));
        state.weight_space_bytes = Index(weight_bytes);
    }

    if (topology_changed)
        for (CudnnRnnShapeSlot& slot : state.shape_slots)
        {
            slot.batch = -1;
            slot.time  = -1;
        }

    int slot_index = -1;
    for (int s = 0; s < RNN_SHAPE_SLOTS; ++s)
        if (state.shape_slots[s].batch == batch_size && state.shape_slots[s].time == T)
            slot_index = s;

    if (slot_index >= 0 && for_training && !state.shape_slots[slot_index].training_ready)
    {
        CudnnRnnShapeSlot& slot = state.shape_slots[slot_index];
        size_t work_bytes = 0;
        size_t reserve_bytes = 0;
        CHECK_CUDNN(cudnnGetRNNTempSpaceSizes(
            device::get_cudnn_handle(), state.rnn_desc,
            CUDNN_FWD_MODE_TRAINING, slot.x_desc,
            &work_bytes, &reserve_bytes));
        slot.workspace_bytes = max(slot.workspace_bytes, Index(work_bytes));
        slot.reserve_space_bytes = Index(reserve_bytes);
        slot.training_ready = true;
    }

    if (slot_index < 0)
    {
        slot_index = 0;
        for (int s = 1; s < RNN_SHAPE_SLOTS; ++s)
        {
            if (state.shape_slots[slot_index].batch < 0) break;
            if (state.shape_slots[s].batch < 0
                || state.shape_slots[s].stamp < state.shape_slots[slot_index].stamp)
                slot_index = s;
        }
        CudnnRnnShapeSlot& slot = state.shape_slots[slot_index];
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
        device::copy_async(slot.seq_dev.data(), seq_h,
                           batch_size * Index(sizeof(int32_t)),
                           device::CopyKind::HostToDevice,
                           device::get_compute_stream());

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
        if (is_lstm) slot.c_desc.reset();
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&slot.h_desc.handle));
        slot.h_desc.deleter = &cudnnDestroyTensorDescriptor;
        if (is_lstm)
        {
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&slot.c_desc.handle));
            slot.c_desc.deleter = &cudnnDestroyTensorDescriptor;
        }
        const int dimA[3]    = {1, int(batch_size), int(H)};
        const int strideA[3] = {int(batch_size * H), int(H), 1};
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(slot.h_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        if (is_lstm)
            CHECK_CUDNN(cudnnSetTensorNdDescriptor(slot.c_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

        size_t work_bytes = 0;
        size_t reserve_bytes = 0;
        if (for_training)
            CHECK_CUDNN(cudnnGetRNNTempSpaceSizes(
                device::get_cudnn_handle(), state.rnn_desc,
                CUDNN_FWD_MODE_TRAINING, slot.x_desc,
                &work_bytes, &reserve_bytes));

        size_t inference_work_bytes = 0;
        CHECK_CUDNN(cudnnGetRNNTempSpaceSizes(
            device::get_cudnn_handle(), state.rnn_desc,
            CUDNN_FWD_MODE_INFERENCE, slot.x_desc,
            &inference_work_bytes, nullptr));

        slot.training_ready = for_training;
        slot.workspace_bytes = Index(max(work_bytes, inference_work_bytes));
        slot.reserve_space_bytes = Index(reserve_bytes);

    }

    state.shape_slots[slot_index].stamp = ++state.shape_stamp;

    state.cached_input_features  = F;
    state.cached_output_features = H;
    return state.shape_slots[slot_index];
}

void CudnnRnnState::cudnn_copy_weight_regions_(int num_linear_layers,
                                               Index input_features,
                                               Index output_features,
                                               const TensorView* const* matrices,
                                               const TensorView* const* vectors,
                                               Buffer& packed_weights,
                                               bool to_cudnn) const
{
    BackendState& state = backend_state;
    const int F = int(input_features);
    const int H = int(output_features);
    const int input_layers = num_linear_layers / 2;

    CudnnDescriptor<cudnnTensorDescriptor_t> matrix_desc;
    CudnnDescriptor<cudnnTensorDescriptor_t> bias_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&matrix_desc.handle));
    matrix_desc.deleter = &cudnnDestroyTensorDescriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc.handle));
    bias_desc.deleter = &cudnnDestroyTensorDescriptor;

    RnnCopySpec specs[RNN_COPY_MAX_REGIONS];
    int count = 0;
    for (int lin = 0; lin < num_linear_layers; ++lin)
    {
        float* cudnn_matrix = nullptr;
        float* cudnn_vector = nullptr;
        CHECK_CUDNN(cudnnGetRNNWeightParams(
            device::get_cudnn_handle(), state.rnn_desc, 0,
            size_t(state.weight_space_bytes), packed_weights.data(), lin,
            matrix_desc, reinterpret_cast<void**>(&cudnn_matrix),
            bias_desc, reinterpret_cast<void**>(&cudnn_vector)));

        const int rows = lin < input_layers ? F : H;
        if (cudnn_matrix && matrices[lin]->get_data())
        {
            float* ours = const_cast<float*>(matrices[lin]->as<float>());
            specs[count++] = to_cudnn ? RnnCopySpec{ours, cudnn_matrix, rows, H, 1}
                                      : RnnCopySpec{cudnn_matrix, ours, H, rows, 1};
        }
        if (cudnn_vector && vectors[lin] && vectors[lin]->get_data())
        {
            float* ours = const_cast<float*>(vectors[lin]->as<float>());
            specs[count++] = to_cudnn ? RnnCopySpec{ours, cudnn_vector, 1, H, 0}
                                      : RnnCopySpec{cudnn_vector, ours, 1, H, 0};
        }
    }
    rnn_copy_regions_cuda(specs, count);
}

void CudnnRnnState::cudnn_pack_weights_(int num_linear_layers,
                                        Index input_features,
                                        Index output_features,
                                        const TensorView* const* weights,
                                        const TensorView* const* biases,
                                        Buffer& forward_state) const
{
    const Index weight_space_bytes = backend_state.weight_space_bytes;
    forward_state.grow_to(get_aligned_bytes(weight_space_bytes));
    device::set_zero_async(forward_state.data(), weight_space_bytes,
                           device::get_compute_stream());
    cudnn_copy_weight_regions_(num_linear_layers, input_features, output_features,
                               weights, biases, forward_state, true);
}

void CudnnRnnState::cudnn_unpack_gradients_(int num_linear_layers,
                                            Index input_features,
                                            Index output_features,
                                            const TensorView* const* weight_gradients,
                                            const TensorView* const* bias_gradients,
                                            Buffer& backward_scratch) const
{
    cudnn_copy_weight_regions_(num_linear_layers, input_features, output_features,
                               weight_gradients, bias_gradients,
                               backward_scratch, false);
}

void CudnnRnnState::prepare_cudnn_forward_state_(Buffer& forward_state,
                                                 bool is_training,
                                                 const CudnnRnnShapeSlot& shape) const
{
    const Index reserve_offset = get_aligned_bytes(backend_state.weight_space_bytes);
    const Index reserve_bytes = is_training ? shape.reserve_space_bytes : 0;
    forward_state.grow_to(detail::checked_index_add(
        reserve_offset, reserve_bytes, "cuDNN RNN forward state"));
}

void CudnnRnnState::cudnn_rnn_forward_(const CudnnRnnShapeSlot& initial_shape,
                                       bool is_training, bool has_cell_state,
                                       const void* x, void* y,
                                       Buffer& forward_state,
                                       const function<CudnnRnnShapeSlot&()>& reconfigure) const
{
    BackendState& state = backend_state;
    const CudnnRnnShapeSlot* selected_shape = &initial_shape;
    auto run_forward = [&]() {
        const CudnnRnnShapeSlot& shape = *selected_shape;
        void* workspace = shape.workspace_bytes > 0
            ? ensure_shared_scratch(size_t(shape.workspace_bytes))
            : nullptr;
        void* reserve = is_training && shape.reserve_space_bytes > 0
            ? forward_state.as<uint8_t>() + get_aligned_bytes(state.weight_space_bytes)
            : nullptr;
        return cudnnRNNForward(
            device::get_cudnn_handle(),
            state.rnn_desc,
            is_training ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE,
            shape.seq_dev.as<int32_t>(),
            shape.x_desc, x,
            shape.y_desc, y,
            shape.h_desc, nullptr, nullptr,
            has_cell_state ? shape.c_desc : shape.h_desc, nullptr, nullptr,
            size_t(state.weight_space_bytes), forward_state.data(),
            size_t(shape.workspace_bytes), workspace,
            is_training ? size_t(shape.reserve_space_bytes) : 0,
            reserve);
    };

    cudnnStatus_t forward_status = run_forward();
    if (forward_status == CUDNN_STATUS_NOT_SUPPORTED && state.persist_algo_active)
    {
        state.persist_algo_failed = true;
        state.rnn_desc.reset();
        state.cached_input_features = -1;
        selected_shape = &reconfigure();
        forward_status = run_forward();
    }
    CHECK_CUDNN(forward_status);
}

void CudnnRnnState::cudnn_rnn_backward_(const CudnnRnnShapeSlot& shape,
                                        bool has_cell_state,
                                        const void* x, const void* y, const void* dy,
                                        void* dx,
                                        const Buffer& forward_state,
                                        Buffer& backward_scratch) const
{
    BackendState& state = backend_state;
    const cudnnTensorDescriptor_t second_state_desc =
        has_cell_state ? shape.c_desc.handle : shape.h_desc.handle;
    void* workspace = shape.workspace_bytes > 0
        ? ensure_shared_scratch(size_t(shape.workspace_bytes))
        : nullptr;
    void* reserve = shape.reserve_space_bytes > 0
        ? static_cast<uint8_t*>(forward_state.data()) + get_aligned_bytes(state.weight_space_bytes)
        : nullptr;

    CHECK_CUDNN(cudnnRNNBackwardData_v8(
        device::get_cudnn_handle(),
        state.rnn_desc,
        shape.seq_dev.as<int32_t>(),
        shape.y_desc, y, dy,
        shape.x_desc, dx,
        shape.h_desc, nullptr, nullptr, nullptr,
        second_state_desc, nullptr, nullptr, nullptr,
        size_t(state.weight_space_bytes), forward_state.data(),
        size_t(shape.workspace_bytes), workspace,
        size_t(shape.reserve_space_bytes), reserve));

    backward_scratch.grow_to(state.weight_space_bytes);
    device::set_zero_async(backward_scratch.data(), state.weight_space_bytes,
                           device::get_compute_stream());

    CHECK_CUDNN(cudnnRNNBackwardWeights_v8(
        device::get_cudnn_handle(),
        state.rnn_desc,
        CUDNN_WGRAD_MODE_ADD,
        shape.seq_dev.as<int32_t>(),
        shape.x_desc, x,
        shape.h_desc, nullptr,
        shape.y_desc, y,
        size_t(state.weight_space_bytes), backward_scratch.data(),
        size_t(shape.workspace_bytes), workspace,
        size_t(shape.reserve_space_bytes), reserve));
}

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
