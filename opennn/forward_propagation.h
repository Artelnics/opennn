//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R W A R D   P R O P A G A T I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_utilities.h"

namespace opennn
{

class NeuralNetwork;

struct ForwardPropagation
{
    ForwardPropagation(const Index = 0, NeuralNetwork* = nullptr);

    void set(const Index = 0, NeuralNetwork* = nullptr);

    void allocate_device();

    TensorView get_last_trainable_layer_outputs() const;

    vector<vector<TensorView>> get_layer_input_views(const vector<TensorView>&, bool) const;

    TensorView get_outputs() const;

    void print() const;

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;

    Memory data;
    vector<vector<vector<TensorView>>> views;
};

}
