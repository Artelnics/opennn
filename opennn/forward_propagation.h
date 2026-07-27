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

#include <tuple>

namespace opennn
{

class NeuralNetwork;

enum class ForwardPropagationMode
{
    Training,
    Inference
};

struct ForwardPropagation
{
    ForwardPropagation() = default;

    ForwardPropagation(Index, NeuralNetwork*,
                       ForwardPropagationMode = ForwardPropagationMode::Training);

    ~ForwardPropagation();

    ForwardPropagation(const ForwardPropagation&) = delete;
    ForwardPropagation& operator=(const ForwardPropagation&) = delete;

    void set(Index, NeuralNetwork*, Buffer* external_storage = nullptr,
             ForwardPropagationMode = ForwardPropagationMode::Training);

    void stage_position(cudaStream_t stream);

    void set_active_sequence_length(Index length);

    TensorView get_last_trainable_layer_outputs() const;

    TensorView get_outputs() const;

    void set_cuda_graph(bool);
    bool get_cuda_graph() const noexcept { return use_cuda_graph; }
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
    vector<tuple<size_t, size_t, size_t>> passthrough_overrides;
    vector<Index> attention_valid_lengths;

    bool use_cuda_graph = false;
    bool cuda_graph_failed = false;
    Index cuda_graph_warmup_calls = 0;
    device::GraphExecHandle inference_graph_exec;
    vector<const void*> captured_input_pointers;

    device::GraphWorkspaceRequirements inference_graph_workspace_requirements;
    Buffer inference_graph_shared_scratch{Device::CUDA};
    Buffer inference_graph_bf16_input{Device::CUDA};
    Buffer inference_graph_bf16_gradient{Device::CUDA};
    Buffer inference_graph_bf16_to_fp32{Device::CUDA};
};

}
