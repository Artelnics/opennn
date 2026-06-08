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
    BackPropagation(const Index = 0, Loss* = nullptr);

    virtual ~BackPropagation() = default;

    void set(const Index = 0, Loss* = nullptr);

    void accumulate_output_deltas(size_t layer_index);

    const NeuralNetwork* neural_network = nullptr;

    Buffer gradient;
    vector<vector<TensorView>> gradient_views;

    Buffer delta_pool;
    vector<TensorView> layer_output_deltas;
    vector<vector<TensorView>> backward_slots;

    vector<vector<pair<size_t, size_t>>> consumer_edges;

    TensorView& get_output_delta();
    const TensorView& get_output_delta() const;

    void print() const;

    Index batch_size = 0;

    Loss* loss_pointer = nullptr;

    float error = 0.0f;
    float accuracy = 0.0f;
    float regularization = 0.0f;
    float loss = 0.0f;
    Index active_tokens_count = 0;

private:

    void setup_delta_pool(const vector<vector<TensorSpec>>& backward_specs);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
