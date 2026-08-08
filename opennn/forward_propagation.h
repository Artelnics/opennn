//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R W A R D   P R O P A G A T I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_types.h"
#include "device_backend.h"
#include "memory_pool.h"

#include <tuple>

namespace opennn
{

class NeuralNetwork;

enum class ForwardPropagationMode
{
    Training,
    Inference
};

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
                       const vector<MemoryPoolEntry>* co_planned_lifetimes = nullptr);

    ~ForwardPropagation();

    ForwardPropagation(const ForwardPropagation&) = delete;
    ForwardPropagation& operator=(const ForwardPropagation&) = delete;

    void set(Index, NeuralNetwork*, Buffer* external_storage = nullptr,
             ForwardPropagationMode = ForwardPropagationMode::Training,
             InferenceShapePolicy = {},
             bool inputs_pre_scaled = false,
             const vector<MemoryPoolEntry>* co_planned_lifetimes = nullptr);

    Index bind_slot_views(const vector<vector<TensorSpec>>& forward_specs,
                          const vector<vector<Index>>& slot_offsets,
                          const vector<vector<Index>>& transient_slot_offsets);

    // Byte offsets, inside `data`, for lifetimes a caller asked to be planned
    // alongside the forward activations. ForwardPropagation neither knows nor
    // cares what they are; BackPropagation supplies its delta lifetimes here.
    struct CoPlannedBlock
    {
        vector<Index> offsets;
        Index bytes = 0;
        bool valid = false;
    };
    CoPlannedBlock co_planned_block;

    void stage_position(cudaStream_t stream);

    void set_active_sequence_length(Index length);

    void set_output_sequence_window(Index start, Index count);
    void gather_output_window();

    Index get_sequence_capacity() const noexcept { return sequence_capacity; }
    Index get_final_output_capacity() const noexcept { return final_output_capacity; }

    TensorView get_last_trainable_layer_outputs() const;

    TensorView get_outputs() const;

    void recompute_for_backward(Index layer_index);

    void set_cuda_graph(bool);
    void reset_cuda_graph() noexcept;
    void prepare_cuda_graph_workspaces();
    bool cuda_graph_workspaces_need_growth() const noexcept;
    device::GraphWorkspaceViews get_cuda_graph_workspace_views() const noexcept;

    Index batch_size = 0;
    ForwardPropagationMode mode = ForwardPropagationMode::Training;

    Index past_length = 0;

    bool inputs_pre_scaled = false;

    NeuralNetwork* neural_network = nullptr;

    Buffer data;
    vector<Buffer> device_input_buffers;
    vector<TensorView> device_input_views;

    vector<vector<uint16_t>> host_bf16_input_scratch;

    Buffer position_device{Device::CUDA};
    void* position_pinned = nullptr;

    vector<vector<TensorView>> input_views;
    vector<vector<TensorView>> forward_slots;
    vector<vector<TensorView>> capacity_input_views;
    vector<vector<TensorView>> capacity_forward_slots;
    vector<tuple<size_t, size_t, size_t>> passthrough_overrides;
    vector<Index> attention_valid_lengths;
    vector<size_t> recomputable_forward_slots;

    InferenceShapePolicy inference_shape_policy;
    Index sequence_capacity = 0;
    Index active_sequence_length = 0;
    Index final_output_capacity = 0;
    Index final_output_layer = -1;

    Buffer output_window_input;
    Index output_window_start = 0;
    Index output_window_count = 0;

    bool use_cuda_graph = false;
    bool cuda_graph_failed = false;
    Index cuda_graph_warmup_calls = 0;
    device::GraphExecHandle inference_graph_exec;
    vector<const void*> captured_input_pointers;

    device::GraphWorkspaceRequirements inference_graph_workspace_requirements{};
    array<Buffer, size_t(device::GraphWorkspaceKind::Count)> inference_graph_workspaces{
        Buffer{Device::CUDA}, Buffer{Device::CUDA}, Buffer{Device::CUDA},
        Buffer{Device::CUDA}, Buffer{Device::CUDA}};
};

}
