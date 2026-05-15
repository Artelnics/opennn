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
        size_t consumer_layer_index;
        size_t consumer_input_index;
    };

    BackPropagation(const Index = 0, Loss* = nullptr);

    virtual ~BackPropagation() = default;

    void set(const Index = 0, Loss* = nullptr);

    void accumulate_output_deltas(size_t layer_index);

    const NeuralNetwork* neural_network = nullptr;

    Buffer gradient;
    vector<vector<TensorView>> gradient_views;

    Buffer delta_pool;
    vector<vector<TensorView>> delta_views;

    vector<vector<BackwardEdge>> backward_edges;

    TensorView& get_output_delta();
    const TensorView& get_output_delta() const;

    void print() const;

    Index batch_size = 0;

    Loss* loss = nullptr;

    float error = 0.0f;
    float accuracy = 0.0f;
    float loss_value = 0.0f;
    Index active_tokens_count = 0;

private:

    struct DeltaPoolEntry
    {
        Index layer;
        size_t slot;
        Index offset;
        Shape shape;
        Type dtype;
        Index bytes;
        Index birth;
        Index death;
    };

    struct DeltaPoolPlan
    {
        Index peak_bytes = 0;
        vector<DeltaPoolEntry> entries;
        vector<pair<size_t, size_t>> alias_target;
    };

    DeltaPoolPlan compute_delta_pool_plan(const vector<vector<Shape>>& backward_shapes,
                                          const vector<vector<Type>>& backward_dtypes) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
