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

    void set(Index, NeuralNetwork*);

    TensorView get_last_trainable_layer_outputs() const;

    TensorView get_outputs() const;

    void print() const;

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;

    Buffer data;
    vector<Buffer> device_input_buffers;
    vector<Buffer> device_fp32_input_buffers;
    vector<vector<TensorView>> input_views;
    vector<vector<TensorView>> forward_slots;
};

}
