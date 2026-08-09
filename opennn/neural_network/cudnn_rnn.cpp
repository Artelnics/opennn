//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D N N   R N N   S T A T E   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/cudnn_rnn.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/cuda/kernel.cuh"

#ifdef OPENNN_HAS_CUDA

namespace opennn
{

static bool persist_env_enabled(const char* env_var)
{
    static const bool enabled = [env_var]() {
        const char* env = getenv(env_var);
        return !(env && string(env) == "0");
    }();
    return enabled;
}

void CudnnRnnState::cudnn_setup_(const CudnnRnnConfig& config,
                                 Index input_features,
                                 Index output_features,
                                 Index time_steps,
                                 Index batch_size,
                                 bool for_training) const
{
    if (!persist_algo_failed_ && persist_env_enabled(config.persist_env_var))
    {
        try
        {
            cudnn_setup_attempt_(config, input_features, output_features, time_steps,
                                 batch_size, for_training);
            return;
        }
        catch (const exception&)
        {
            persist_algo_failed_ = true;
            rnn_desc.reset();
            cached_input_features = -1;
        }
    }
    cudnn_setup_attempt_(config, input_features, output_features, time_steps,
                         batch_size, for_training);
}

void CudnnRnnState::cudnn_setup_attempt_(const CudnnRnnConfig& config,
                                         Index input_features,
                                         Index output_features,
                                         Index time_steps,
                                         Index batch_size,
                                         bool for_training) const
{
    persist_algo_active_ = !persist_algo_failed_ && persist_env_enabled(config.persist_env_var);

    const Index F = input_features;
    const Index H = output_features;
    const Index T = time_steps;
    const bool is_lstm = (config.cell_mode == CUDNN_LSTM);

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
             0.0f,
            dropout_states_buf.data,
            size_t(dropout_states_buf.bytes),
             0ULL));

        CHECK_CUDNN(cudnnSetRNNDescriptor_v8(
            rnn_desc,
            persist_algo_active_ ? CUDNN_RNN_ALGO_PERSIST_STATIC
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

        for (int lin = 0; lin < config.num_linear_layers; ++lin)
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
        if (is_lstm)
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

void CudnnRnnState::cudnn_pack_weights_(int num_linear_layers,
                                        Index input_features,
                                        Index output_features,
                                        const TensorView* const* weights,
                                        const TensorView* const* biases) const
{
    const Index F = input_features;
    const Index H = output_features;
    const int input_layers = num_linear_layers / 2;

    RnnCopySpec specs[RNN_COPY_MAX_REGIONS];
    int count = 0;
    for (int lin = 0; lin < num_linear_layers; ++lin)
    {
        const bool is_input_w = (lin < input_layers);
        if (cudnn_w_ptrs_[lin] && weights[lin]->data)
            specs[count++] = {weights[lin]->as<float>(), cudnn_w_ptrs_[lin],
                              int(is_input_w ? F : H), int(H), 1};

        if (cudnn_b_ptrs_[lin] && biases[lin] && biases[lin]->data)
            specs[count++] = {biases[lin]->as<float>(), cudnn_b_ptrs_[lin],
                              1, int(H), 0};
    }
    rnn_copy_regions_cuda(specs, count);
}

void CudnnRnnState::cudnn_unpack_gradients_(int num_linear_layers,
                                            Index input_features,
                                            Index output_features,
                                            const TensorView* const* weight_gradients,
                                            const TensorView* const* bias_gradients) const
{
    const Index F = input_features;
    const Index H = output_features;
    const int input_layers = num_linear_layers / 2;

    RnnCopySpec specs[RNN_COPY_MAX_REGIONS];
    int count = 0;
    for (int lin = 0; lin < num_linear_layers; ++lin)
    {
        const bool is_input_w = (lin < input_layers);
        if (cudnn_gw_ptrs_[lin] && weight_gradients[lin]->data)
            specs[count++] = {cudnn_gw_ptrs_[lin],
                              const_cast<float*>(weight_gradients[lin]->as<float>()),
                              int(H), int(is_input_w ? F : H), 1};

        if (cudnn_gb_ptrs_[lin] && bias_gradients[lin] && bias_gradients[lin]->data)
            specs[count++] = {cudnn_gb_ptrs_[lin],
                              const_cast<float*>(bias_gradients[lin]->as<float>()),
                              1, int(H), 0};
    }
    rnn_copy_regions_cuda(specs, count);
}

void CudnnRnnState::cudnn_rnn_forward_(bool is_training, bool has_cell_state,
                                       const void* x, void* y,
                                       const function<void()>& reconfigure) const
{
    auto run_forward = [&]() {
        const CudnnRnnShapeSlot& shape = active_shape();
        return cudnnRNNForward(
            Backend::get_cudnn_handle(),
            rnn_desc,
            is_training ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE,
            shape.seq_dev.as<int32_t>(),
            shape.x_desc, x,
            shape.y_desc, y,
            shape.h_desc, nullptr, nullptr,
            has_cell_state ? shape.c_desc : shape.h_desc, nullptr, nullptr,
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
        reconfigure();
        forward_status = run_forward();
    }
    CHECK_CUDNN(forward_status);
}

void CudnnRnnState::cudnn_rnn_backward_(bool has_cell_state,
                                        const void* x, const void* y, const void* dy,
                                        void* dx) const
{
    const CudnnRnnShapeSlot& shape = active_shape();
    const cudnnTensorDescriptor_t second_state_desc =
        has_cell_state ? shape.c_desc.handle : shape.h_desc.handle;

    CHECK_CUDNN(cudnnRNNBackwardData_v8(
        Backend::get_cudnn_handle(),
        rnn_desc,
        shape.seq_dev.as<int32_t>(),
        shape.y_desc, y, dy,
        shape.x_desc, dx,
        shape.h_desc, nullptr, nullptr, nullptr,
        second_state_desc, nullptr, nullptr, nullptr,
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
        shape.x_desc, x,
        shape.h_desc, nullptr,
        shape.y_desc, y,
        size_t(dweight_space_buf.bytes), dweight_space_buf.data,
        size_t(workspace_buf.bytes), workspace_buf.data,
        size_t(reserve_space_buf.bytes), reserve_space_buf.data));
}

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
