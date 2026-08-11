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
                       span<const MemoryPoolEntry> co_planned_lifetimes = {});

    ~ForwardPropagation();

    ForwardPropagation(const ForwardPropagation&) = delete;
    ForwardPropagation& operator=(const ForwardPropagation&) = delete;

    void set(Index, NeuralNetwork*, Buffer* external_storage = nullptr,
             ForwardPropagationMode = ForwardPropagationMode::Training,
             InferenceShapePolicy = {},
             bool inputs_pre_scaled = false,
             span<const MemoryPoolEntry> co_planned_lifetimes = {});

    // Byte offsets inside `arena` for lifetimes planned alongside the forward
    // activations. An empty vector means that no joint plan is active.
    vector<Index> co_planned_offsets;

    void stage_position(cudaStream_t stream);

    void set_active_sequence_length(Index length);

    void set_output_sequence_window(Index start, Index count);
    void gather_output_window();

    Index get_sequence_capacity() const noexcept { return sequence_capacity; }
    Index get_final_output_capacity() const noexcept { return final_output_capacity; }
    Index get_final_output_layer() const noexcept { return final_output_layer; }

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

    Buffer arena;
    vector<Buffer> staged_input_storage;
    vector<TensorView> staged_inputs;

    vector<vector<uint16_t>> host_bf16_input_scratch;

    Buffer position_device{Device::CUDA};
    void* position_pinned = nullptr;

    vector<vector<TensorView>> inputs;
    vector<vector<TensorView>> slots;
    vector<tuple<size_t, size_t, size_t>> passthrough_overrides;
    vector<Index> attention_valid_lengths;

    bool use_cuda_graph = false;
    bool cuda_graph_failed = false;
    Index cuda_graph_warmup_calls = 0;
    device::GraphExecHandle inference_graph_exec;
    vector<const void*> captured_input_pointers;

    device::GraphWorkspaceRequirements inference_graph_workspace_requirements{};
    array<Buffer, size_t(device::GraphWorkspaceKind::Count)> inference_graph_workspaces{
        Buffer{Device::CUDA}, Buffer{Device::CUDA}, Buffer{Device::CUDA},
        Buffer{Device::CUDA}, Buffer{Device::CUDA}};

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
    Index final_output_capacity = 0;
    Index final_output_layer = -1;
    optional<OutputWindow> output_window;
};

}
