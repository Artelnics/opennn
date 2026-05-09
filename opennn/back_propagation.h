//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A C K   P R O P A G A T I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_utilities.h"

namespace opennn
{

class Loss;
class NeuralNetwork;

struct BackPropagation
{
    struct BackwardEdge
    {
        size_t consumer_index;
        size_t port;
    };

    BackPropagation(const Index = 0, Loss* = nullptr);

    virtual ~BackPropagation() = default;

    void set(const Index = 0, Loss* = nullptr);

    void accumulate_output_deltas(size_t layer_index);

    NeuralNetwork* neural_network = nullptr;

    Buffer gradient;
    vector<vector<TensorView>> gradient_views;

    Buffer delta_pool;
    vector<vector<TensorView>> delta_views;

    vector<vector<BackwardEdge>> backward_edges;

    TensorView get_output_deltas() const;

    void print() const;

    Index batch_size = 0;

    Loss* loss = nullptr;

    float error = 0.0f;
    float accuracy = 0.0f;
    float loss_value = 0.0f;
    Index active_tokens_count = 0;

    Shape output_delta_dimensions;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.