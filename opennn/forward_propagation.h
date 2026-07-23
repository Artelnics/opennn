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

namespace opennn
{

class NeuralNetwork;

struct ForwardPropagation
{
    ForwardPropagation() = default;

    ForwardPropagation(Index, NeuralNetwork*);

    void set(Index, NeuralNetwork*, Buffer* external_storage = nullptr);

    TensorView get_last_trainable_layer_outputs() const;

    TensorView get_outputs() const;

    void print() const;

    void set_cuda_graph(bool);
    bool get_cuda_graph() const noexcept { return use_cuda_graph; }
    void reset_cuda_graph() noexcept;

    Index batch_size = 0;

    bool inputs_pre_scaled = false;

    NeuralNetwork* neural_network = nullptr;

    Buffer data;
    vector<Buffer> device_input_buffers;

    vector<vector<TensorView>> input_views;
    vector<vector<TensorView>> forward_slots;
    vector<tuple<size_t, size_t, size_t>> passthrough_overrides;
    vector<Index> attention_valid_lengths;

    bool use_cuda_graph = false;
    bool cuda_graph_failed = false;
    Index cuda_graph_warmup_calls = 0;
    device::GraphExecHandle inference_graph_exec;
    vector<const void*> captured_input_pointers;
};

}
