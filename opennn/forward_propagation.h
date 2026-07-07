//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R W A R D   P R O P A G A T I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_types.h"

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

    Index batch_size = 0;

    // True when the batches fed through this propagation come from a dataset
    // that Optimizer::set_scaling() already scaled in place (training and
    // in-loop validation). NeuralNetwork::forward_propagate then skips the
    // leading Scaling layers even with is_training == false, so validation
    // inputs are not scaled twice.
    bool inputs_pre_scaled = false;

    NeuralNetwork* neural_network = nullptr;

    Buffer data;
    vector<Buffer> device_input_buffers;
    
    vector<vector<TensorView>> input_views;
    vector<vector<TensorView>> forward_slots;
    vector<tuple<size_t, size_t, size_t>> passthrough_overrides;
    vector<Index> attention_valid_lengths;
};

}
