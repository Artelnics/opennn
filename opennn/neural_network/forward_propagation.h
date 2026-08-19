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

// One length per sample of a batch, as an Embedding exports it: on the host for
// CPU runs, as int32 on the device for CUDA runs. Either side may be absent.
struct SequenceLengths
{
    const vector<Index>* host = nullptr;
    const int* device = nullptr;

    explicit operator bool() const noexcept { return host || device; }
};

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

    // Reuse persistent session state (such as an autoregressive KV cache)
    // across propagation shapes that belong to the same inference session.
    // The source and destination must execute the same network.
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

    void set_cuda_graph(bool);
    void reset_cuda_graph() noexcept;
    void prepare_cuda_graph_workspaces();
    bool cuda_graph_workspaces_need_growth() const noexcept;
    device::GraphWorkspaceViews get_cuda_graph_workspace_views() const noexcept;

    Index batch_size = 0;
    ForwardPropagationMode mode = ForwardPropagationMode::Training;

    Index past_length = 0;

    Buffer arena;
    // Opaque execution-local storage whose size is known only after a backend
    // configures an operation (for example, cuDNN RNN state).
    vector<Buffer> layer_state_storage;
    // Persistent state shared by the propagation shapes of one inference
    // session (for example, an autoregressive KV cache).
    shared_ptr<vector<Buffer>> layer_session_state_storage;
    // Host mirrors used only while an execution stages data across a device
    // boundary. Kept per layer so independent propagation contexts never share
    // staging addresses.
    vector<device::PinnedBuffer> layer_pinned_storage;
    vector<Buffer> staged_input_storage;
    vector<TensorView> staged_inputs;

    // Loss evaluation scratch belongs to this execution, not to the reusable
    // Loss configuration. YOLO keeps its assembled targets separate because
    // the target and reduction buffers are live at the same time.
    mutable Buffer loss_workspace{Device::CUDA};
    mutable Buffer loss_target_workspace{Device::CUDA};

    vector<vector<uint16_t>> host_bf16_input_scratch;

    Buffer position_device{Device::CUDA};
    device::PinnedBuffer position_pinned;

    vector<vector<TensorView>> inputs;
    vector<vector<TensorView>> slots;
    vector<uint8_t> drelu_fused_by_layer;
    vector<tuple<size_t, size_t, size_t>> passthrough_overrides;

    // Where each sequence in the batch ends, one record per layer, describing
    // the sequence that layer outputs. Empty means no record: the Embedding
    // behind that sequence exported none, and whoever needs to tell padding
    // from data falls back to reading it off the data.
    //
    // Per layer rather than one for the whole pass, because a network can carry
    // more than one sequence at a time. An encoder-decoder holds two, of
    // different lengths and padded differently, and cross-attention reads the
    // encoder's while decoder self-attention reads the decoder's. A single
    // record cannot say which is which, and the two would overwrite each other.
    //
    // A CPU run keeps the record on the host; a CUDA run keeps it on the device
    // (one int32 per sample) so the masks that read it never wait on a host
    // round trip and the step can be captured into a CUDA graph.
    vector<vector<Index>> valid_lengths;
    vector<const int*> device_valid_lengths;
    vector<Buffer> device_valid_length_storage;

    // The record for whatever feeds one of a layer's inputs. Null when that
    // input carries no record.
    const vector<Index>* input_valid_lengths(size_t layer, size_t input_ordinal) const;
    const int* input_device_valid_lengths(size_t layer, size_t input_ordinal) const;
    SequenceLengths input_sequence_lengths(size_t layer, size_t input_ordinal) const;

    // The device record a layer is about to write for the sequence it outputs
    // (`batch_size` int32 values), owned here so a graph replay reads the same
    // addresses.
    int* device_valid_lengths_slot(size_t layer, Index batch_size);

    // Carries a record forward: a layer that keeps the sequence dimension keeps
    // its first input's record. Called once per layer, after it has run.
    void inherit_valid_lengths(size_t layer);

    Index valid_lengths_source(size_t layer, size_t input_ordinal) const;

    bool use_cuda_graph = false;
    bool cuda_graph_failed = false;
    Index cuda_graph_warmup_calls = 0;
    device::GraphExecHandle inference_graph_exec;
    vector<const void*> captured_input_pointers;

    device::GraphWorkspaceRequirements inference_graph_workspace_requirements{};
    array<Buffer, size_t(device::GraphWorkspaceKind::Count)> inference_graph_workspaces{
        Buffer{Device::CUDA}, Buffer{Device::CUDA}, Buffer{Device::CUDA},
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
    Index execution_start_layer = 0;
    bool position_staging_required = false;
    Index final_output_capacity = 0;
    Index final_output_layer = -1;
    optional<OutputWindow> output_window;
};

}
