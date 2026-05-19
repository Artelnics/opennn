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

/// @brief Workspace holding the activations of every layer during a forward pass.
struct ForwardPropagation
{
    /// @brief Constructs a workspace for the given batch size and network.
    /// @param batch_size Maximum number of samples per forward pass.
    /// @param neural_network Network whose layer specs drive buffer sizing (non-owning).
    ForwardPropagation(const Index = 0, NeuralNetwork* = nullptr);

    /// @brief Reconfigures the workspace for a new batch size or network; reuses allocations when possible.
    /// @param batch_size Maximum number of samples per forward pass.
    /// @param neural_network Network whose layer specs drive buffer sizing (non-owning).
    void set(const Index = 0, NeuralNetwork* = nullptr);

    /// @brief Returns the output tensor of the last trainable layer.
    TensorView get_last_trainable_layer_outputs() const;

    /// @brief Returns the final output tensor of the network.
    TensorView get_outputs() const;

    /// @brief Prints a human-readable summary of the workspace contents.
    void print() const;

    Index batch_size = 0;

    NeuralNetwork* neural_network = nullptr;

    Buffer data;
    vector<vector<vector<TensorView>>> views;
};

}
