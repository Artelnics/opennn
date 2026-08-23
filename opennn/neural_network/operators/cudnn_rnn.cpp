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
#include "opennn/core/profiler.h"
#include "opennn/neural_network/layers/kernel_recurrent.cuh"

#ifdef OPENNN_HAS_CUDA

namespace opennn
{

// cuDNN's small-hidden-state persistent algorithm is the fastest FP32 path for
// the forecasting topology (H=64). Unsupported topologies fall back to the
// standard algorithm in cudnn_setup_().
static bool persist_env_enabled()
{
    static const bool enabled = env_flag_enabled("OPENNN_RNN_PERSIST", true);
    return enabled;
}

static bool time_major_env_enabled()
{
    static const bool enabled = env_flag_enabled("OPENNN_RNN_TIME_MAJOR", true);
    return enabled;
}

static Index cudnn_input_features(Index logical_features, bool persistent)
{
    static const bool pad_features = env_flag_enabled("OPENNN_RNN_PAD_FEATURES", true);
    return !persistent && time_major_env_enabled() && pad_features
        ? ((logical_features + 7) / 8) * 8
        : logical_features;
}

static bool tensor_math_env_enabled()
{
    static const bool enabled = env_flag_enabled("OPENNN_RNN_TENSOR_MATH", false);
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
        catch (const exception& error)
        {
            if (env_flag_enabled("OPENNN_RNN_DEBUG", false))
                cerr << "OpenNN cuDNN RNN: persistent algorithm unavailable: "
                     << error.what() << '\n';
            state.persist_algo_failed = true;
            state.rnn_desc.reset();
            state.cached_input_features = -1;
            state.cached_data_type = Type::Auto;
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
    const Index logical_F = input_features;
    const Index H = output_features;
    const Index T = time_steps;
    const bool is_lstm = (config.cell_mode == CUDNN_LSTM);
    const bool bf16 = config.data_type == Type::BF16;
    throw_if(!is_one_of(config.data_type, Type::FP32, Type::BF16),
             "cuDNN RNN supports FP32 or BF16 data.");
    const cudnnDataType_t cudnn_data_type = bf16 ? CUDNN_DATA_BFLOAT16
                                                 : CUDNN_DATA_FLOAT;
    // PERSIST_STATIC_SMALL_H is an FP32-specific cuDNN specialization. Keep
    // BF16 on the standard tensor-core path and cap the hidden size so larger
    // models do not repeatedly probe an ineligible algorithm.
    state.persist_algo_active = !state.persist_algo_failed
                             && persist_env_enabled()
                             && !bf16 && H <= 128
                             && (!is_lstm || batch_size < 512);
    const Index F = cudnn_input_features(logical_F,
                                         state.persist_algo_active);

    const bool topology_changed =
        state.cached_input_features  != F ||
        state.cached_output_features != H ||
        state.cached_data_type       != config.data_type ||
        state.rnn_desc == nullptr;

    if (topology_changed)
    {
        // cuDNN's double-bias/default-math kernel is decisively faster for
        // small LSTMs (and slower for the simple RNN and larger batches).
        // Environment variables remain explicit overrides for reproducible
        // tuning runs.
        const bool small_lstm = is_lstm && batch_size <= 64;
        const bool large_lstm = is_lstm && batch_size >= 512 && !bf16;
        state.double_bias = env_flag_enabled(
            "OPENNN_RNN_DOUBLE_BIAS",
            state.persist_algo_active || small_lstm || large_lstm);
        state.packed_layout = env_flag_enabled(
            "OPENNN_RNN_PACKED_LAYOUT", false) && !state.persist_algo_active;
        const bool use_default_math = env_flag_enabled(
            "OPENNN_RNN_DEFAULT_MATH", state.persist_algo_active || small_lstm);
        state.rnn_desc.reset();
        CHECK_CUDNN(cudnnCreateRNNDescriptor(&state.rnn_desc.handle));
        state.rnn_desc.deleter = &cudnnDestroyRNNDescriptor;

        if (!state.dropout_desc)
        {
            CHECK_CUDNN(cudnnCreateDropoutDescriptor(&state.dropout_desc.handle));
            state.dropout_desc.deleter = &cudnnDestroyDropoutDescriptor;
        }
        // A zero-rate descriptor has no RNG state. Supplying a null/zero state
        // avoids allocating and initializing cuDNN's otherwise large dropout
        // buffer during recurrent setup.
        state.dropout_states.resize_bytes(0, Device::CUDA);
        CHECK_CUDNN(cudnnSetDropoutDescriptor(
            state.dropout_desc, device::get_cudnn_handle(),
             0.0f,
            nullptr,
            0,
             0ULL));

        CHECK_CUDNN(cudnnSetRNNDescriptor_v8(
            state.rnn_desc,
            state.persist_algo_active ? CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H
                                      : CUDNN_RNN_ALGO_STANDARD,
            config.cell_mode,
            state.double_bias ? CUDNN_RNN_DOUBLE_BIAS
                              : CUDNN_RNN_SINGLE_INP_BIAS,
            CUDNN_UNIDIRECTIONAL,
            CUDNN_LINEAR_INPUT,
            cudnn_data_type,
            CUDNN_DATA_FLOAT,
            use_default_math ? CUDNN_DEFAULT_MATH
            : tensor_math_env_enabled() ? CUDNN_TENSOR_OP_MATH
                                        : CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION,
            int(F),
            int(H),
              int(H),
            1,
            state.dropout_desc,
            state.packed_layout ? CUDNN_RNN_PADDED_IO_DISABLED
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
        slot.input_features = F;
        slot.time_major = time_major_env_enabled();

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

        // Every sequence in an OpenNN batch has the full logical length, so
        // there are no padded positions to fill. A null fill matches cuDNN's
        // fastest full-sequence path (and PyTorch's descriptor setup).
        void* const padding_fill = nullptr;
        const cudnnRNNDataLayout_t layout = state.packed_layout
            ? CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED
            : (slot.time_major ? CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED
                               : CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED);
        CHECK_CUDNN(cudnnSetRNNDataDescriptor(
            slot.x_desc, cudnn_data_type,
            layout,
            int(T), int(batch_size), int(F),
            seq_h, padding_fill));
        CHECK_CUDNN(cudnnSetRNNDataDescriptor(
            slot.y_desc, cudnn_data_type,
            layout,
            int(T), int(batch_size), int(H),
            seq_h, padding_fill));

        slot.h_desc.reset();
        if (is_lstm) slot.c_desc.reset();
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&slot.h_desc.handle));
        slot.h_desc.deleter = &cudnnDestroyTensorDescriptor;
        if (is_lstm)
        {
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&slot.c_desc.handle));
            slot.c_desc.deleter = &cudnnDestroyTensorDescriptor;
        }
        // Keep the two singleton spatial dimensions used by cuDNN's native
        // framework integrations. They select the same logical tensor but
        // avoid an internal descriptor normalization on recurrent entry.
        const int dimA[5]    = {1, int(batch_size), int(H), 1, 1};
        const int strideA[5] = {int(batch_size * H), int(H), 1, 1, 1};
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(slot.h_desc, cudnn_data_type, 5, dimA, strideA));
        if (is_lstm)
            CHECK_CUDNN(cudnnSetTensorNdDescriptor(slot.c_desc, cudnn_data_type, 5, dimA, strideA));

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
    state.cached_data_type       = config.data_type;
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
    const int cudnn_F = int(state.cached_input_features);
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
        void* cudnn_matrix = nullptr;
        void* cudnn_vector = nullptr;
        CHECK_CUDNN(cudnnGetRNNWeightParams(
            device::get_cudnn_handle(), state.rnn_desc, 0,
            size_t(state.weight_space_bytes), packed_weights.data(), lin,
            matrix_desc, &cudnn_matrix,
            bias_desc, &cudnn_vector));

        const int rows = lin < input_layers ? F : H;
        const int cudnn_rows = lin < input_layers ? cudnn_F : H;
        if (cudnn_matrix && matrices[lin]->get_data())
        {
            void* ours = const_cast<void*>(matrices[lin]->get_data());
            const int ours_bf16 = matrices[lin]->get_type() == Type::BF16;
            const int packed_bf16 = state.cached_data_type == Type::BF16;
            specs[count++] = to_cudnn
                ? RnnCopySpec{ours, cudnn_matrix, rows, H, 1, H, cudnn_rows,
                              ours_bf16, packed_bf16}
                : RnnCopySpec{cudnn_matrix, ours, H, rows, 1, cudnn_rows, H,
                              packed_bf16, ours_bf16};
        }
        if (cudnn_vector && vectors[lin] && vectors[lin]->get_data())
        {
            void* ours = const_cast<void*>(vectors[lin]->get_data());
            const int ours_bf16 = vectors[lin]->get_type() == Type::BF16;
            const int packed_bf16 = state.cached_data_type == Type::BF16;
            specs[count++] = to_cudnn
                ? RnnCopySpec{ours, cudnn_vector, 1, H, 0, 0, 0,
                              ours_bf16, packed_bf16}
                : RnnCopySpec{cudnn_vector, ours, 1, H, 0, 0, 0,
                              packed_bf16, ours_bf16};
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
    PROFILE_SCOPE("rnn:pack_weights");
    const Index weight_space_bytes = backend_state.weight_space_bytes;
    forward_state.grow_to(get_aligned_bytes(weight_space_bytes));

    // Double-bias mode has recurrent bias regions that OpenNN intentionally
    // leaves at zero, and feature padding has unused matrix rows. Nothing
    // below writes those holes -- only the live regions are copied, on every
    // step -- so they have to be zeroed whenever this weight space has not
    // already been zeroed for this exact layout.
    //
    // This used to be conditioned on the buffer having grown, which is not the
    // same question: a buffer handed back by the device pool already large
    // enough skipped the zeroing and left a previous tenant's bytes sitting in
    // the recurrent bias. The RNN then trained and inferred against garbage
    // biases, a few percent off, and only when some earlier work had dirtied
    // the pool -- so it reproduced in a full suite run and never in isolation.
    const bool has_holes = backend_state.double_bias
                        || backend_state.cached_input_features != input_features;

    const bool already_zeroed =
        backend_state.zeroed_weight_space == forward_state.data()
        && backend_state.zeroed_weight_space_bytes == weight_space_bytes
        && backend_state.zeroed_input_features == input_features
        && backend_state.zeroed_double_bias == backend_state.double_bias;

    if (has_holes && !already_zeroed)
    {
        device::set_zero_async(forward_state.data(), weight_space_bytes,
                               device::get_compute_stream());

        backend_state.zeroed_weight_space = forward_state.data();
        backend_state.zeroed_weight_space_bytes = weight_space_bytes;
        backend_state.zeroed_input_features = input_features;
        backend_state.zeroed_double_bias = backend_state.double_bias;
    }
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
    PROFILE_SCOPE("rnn:unpack_gradients");
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
    PROFILE_SCOPE("rnn:cudnn_forward");
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
        state.cached_data_type = Type::Auto;
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
    PROFILE_SCOPE("rnn:cudnn_backward");
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
