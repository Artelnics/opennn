//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A C K   P R O P A G A T I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_types.h"
#include "memory_pool.h"

#include <utility>

namespace opennn
{

class Loss;
class NeuralNetwork;
struct ForwardPropagation;

struct BackPropagation
{

    struct DeltaEntry
    {
        Index      layer;
        size_t     slot;
        TensorSpec spec;
        Index      first_step;
        Index      last_step;
    };

    struct DeltaLayout
    {
        vector<DeltaEntry> entries;
        vector<bool> aliases_residual_delta;
        Index aliased_residual_delta_bytes = 0;
        vector<pair<size_t, size_t>> reusable_consumer_deltas;
    };

    BackPropagation(const Index = 0, Loss* = nullptr,
                    ForwardPropagation* forward_propagation = nullptr);

    virtual ~BackPropagation() = default;

    void set(const Index = 0, Loss* = nullptr,
             ForwardPropagation* forward_propagation = nullptr);

    static vector<vector<pair<size_t, size_t>>> make_consumer_edges(const NeuralNetwork&);

    static DeltaLayout build_delta_entries(const NeuralNetwork&, const Loss&,
                                           Index batch_size,
                                           const vector<vector<TensorSpec>>&,
                                           const vector<vector<pair<size_t, size_t>>>&);

    static vector<MemoryPoolEntry> to_pool_entries(const vector<DeltaEntry>&,
                                                   Index step_offset = 0);

    // Delta lifetimes expressed on the forward timeline, so ForwardPropagation can
    // co-plan them without knowing what they are.
    static vector<MemoryPoolEntry> make_co_planned_lifetimes(const NeuralNetwork&,
                                                             const Loss&,
                                                             Index batch_size);

    void accumulate_output_deltas(size_t);

    const NeuralNetwork* neural_network = nullptr;

    Buffer gradient;
    vector<vector<TensorView>> gradient_views;

    Buffer delta_pool;
    vector<TensorView> layer_output_deltas;
    vector<vector<TensorView>> backward_slots;

    vector<vector<pair<size_t, size_t>>> consumer_edges;

    TensorView& get_output_delta();
    const TensorView& get_output_delta() const;

    Index batch_size = 0;

    Loss* loss = nullptr;

    float error = 0.0f;
    float accuracy = 0.0f;
    float regularization = 0.0f;
    float loss_value = 0.0f;
    Index active_tokens_count = 0;

private:

    void setup_delta_pool(const vector<vector<TensorSpec>>&, const DeltaLayout&);

    void bind_delta_views(const DeltaLayout&, const vector<Index>& byte_offsets,
                          uint8_t* base, Device device,
                          const vector<vector<TensorSpec>>&);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
