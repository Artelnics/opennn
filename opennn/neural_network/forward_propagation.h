//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R W A R D   P R O P A G A T I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/tensor_types.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/memory_pool.h"

#include <tuple>

namespace opennn
{

struct SequenceLengths
{
    const vector<Index>* host = nullptr;
    const int* device = nullptr;

    explicit operator bool() const noexcept { return host || device; }
};

template<size_t... Kind>
array<Buffer, sizeof...(Kind)> cuda_workspace_buffers(index_sequence<Kind...>)
{
    return {(static_cast<void>(Kind), Buffer{Device::CUDA})...};
}

class NeuralNetwork;

enum class ForwardPropagationMode
{
    Training,
    Inference
};

inline bool is_training(ForwardPropagationMode pass) noexcept
{
    return pass == ForwardPropagationMode::Training;
}

struct InferenceShapePolicy
{
    Index sequence_capacity = 0;
    Index final_output_capacity = 0;
    vector<Index> retained_output_layers;
};

struct ForwardPropagation
{
    ForwardPropagation() = default;

    ForwardPropagation(Index, NeuralNetwork*,
                       ForwardPropagationMode = ForwardPropagationMode::Training,
                       InferenceShapePolicy = {},
                       bool inputs_pre_scaled = false,
                       span<const MemoryPoolEntry> co_planned_lifetimes = {},
                       bool exhaustive_training_plan = false);

    ~ForwardPropagation();

    ForwardPropagation(const ForwardPropagation&) = delete;
    ForwardPropagation& operator=(const ForwardPropagation&) = delete;

    void set(Index, NeuralNetwork*, Buffer* external_storage = nullptr,
             ForwardPropagationMode = ForwardPropagationMode::Training,
             InferenceShapePolicy = {},
             bool inputs_pre_scaled = false,
             span<const MemoryPoolEntry> co_planned_lifetimes = {},
             bool exhaustive_training_plan = false);

    vector<Index> co_planned_offsets;

    void stage_position(cudaStream_t stream);

    void set_active_sequence_length(Index length);

    void share_session_state_from(const ForwardPropagation& source);

    void set_output_sequence_window(Index start, Index count);
    void gather_output_window();

    Index get_sequence_capacity() const noexcept { return sequence_capacity; }
    Index get_final_output_capacity() const noexcept { return final_output_capacity; }
    Index get_final_output_layer() const noexcept { return final_output_layer; }
    Index get_execution_start_layer() const noexcept { return execution_start_layer; }
    bool needs_position_staging() const noexcept { return position_staging_required; }

    TensorView get_last_trainable_layer_outputs() const;

    TensorView get_outputs() const;

    void recompute_for_backward(Index layer_index);

    void set_cuda_graph(bool enabled);
    void reset_cuda_graph() noexcept;
    void prepare_cuda_graph_workspaces();
    bool cuda_graph_workspaces_need_growth() const noexcept;
    device::GraphWorkspaceViews get_cuda_graph_workspace_views() const noexcept;

    Index batch_size = 0;
    ForwardPropagationMode mode = ForwardPropagationMode::Training;

    Index past_length = 0;

    Buffer arena;
    vector<Buffer> layer_state_storage;
    shared_ptr<vector<Buffer>> layer_session_state_storage;
    vector<device::PinnedBuffer> layer_pinned_storage;
    vector<Buffer> staged_input_storage;
    vector<TensorView> staged_inputs;

    mutable Buffer loss_workspace{Device::CUDA};
    mutable Buffer loss_target_workspace{Device::CUDA};

    vector<vector<uint16_t>> host_bf16_input_scratch;
    vector<uint16_t> host_bf16_output_scratch;

    Buffer position_device{Device::CUDA};
    device::PinnedBuffer position_pinned;

    vector<vector<TensorView>> inputs;
    vector<vector<TensorView>> slots;
    vector<uint8_t> drelu_fused_by_layer;
    vector<tuple<size_t, size_t, size_t>> passthrough_overrides;

    vector<vector<Index>> valid_lengths;
    vector<const int*> device_valid_lengths;
    vector<Buffer> device_valid_length_storage;

    const vector<Index>* input_valid_lengths(size_t layer, size_t input_ordinal) const;
    const int* input_device_valid_lengths(size_t layer, size_t input_ordinal) const;
    SequenceLengths input_sequence_lengths(size_t layer, size_t input_ordinal) const;

    int* device_valid_lengths_slot(size_t layer, Index batch_size);

    void inherit_valid_lengths(size_t layer);

    Index valid_lengths_source(size_t layer, size_t input_ordinal) const;

    bool use_cuda_graph = false;
    bool cuda_graph_failed = false;
    Index cuda_graph_warmup_calls = 0;
    device::GraphExecHandle inference_graph_exec;
    vector<const void*> captured_input_pointers;

    device::GraphWorkspaceRequirements inference_graph_workspace_requirements{};
    array<Buffer, size_t(device::GraphWorkspaceKind::Count)> inference_graph_workspaces
        = cuda_workspace_buffers(make_index_sequence<size_t(device::GraphWorkspaceKind::Count)>{});

private:

    struct OutputWindow
    {
        Buffer input;
        Index start = 0;
        Index count = 0;
    };

    Index bind_slots(const vector<vector<TensorSpec>>& forward_specs,
                     const vector<vector<Index>>& slot_offsets,
                     const vector<vector<Index>>& transient_slot_offsets);

    TensorView get_layer_outputs(Index layer) const;

    NeuralNetwork* neural_network = nullptr;
    vector<vector<TensorView>> capacity_inputs;
    vector<vector<TensorView>> capacity_slots;
    vector<size_t> recomputable_slots;
    Index sequence_capacity = 0;
    Index active_sequence_length = 0;
    Index execution_start_layer = 0;
    bool position_staging_required = false;
    Index final_output_capacity = 0;
    Index final_output_layer = -1;
    optional<OutputWindow> output_window;
};

}
