//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A C K   P R O P A G A T I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/tensor_types.h"
#include "opennn/core/memory_pool.h"

#include <utility>

namespace opennn
{

class Loss;
class NeuralNetwork;

struct BackPropagation
{
    // The deltas either live in memory somebody else planned - pass that arena
    // and the byte offsets for this layout - or in an arena of our own. This is
    // the same choice ForwardPropagation::set offers through its external_storage,
    // and deliberately not spelled as a ForwardPropagation: what is wanted here is
    // a region and a list of offsets, not a forward pass.
    // Loss is required only while building that layout; neither it nor its network
    // is retained by BackPropagation.
    BackPropagation(Index, const Loss&,
                    Buffer* external_arena = nullptr,
                    span<const Index> arena_offsets = {});

    virtual ~BackPropagation() = default;

    void set(Index, const Loss&,
             Buffer* external_arena = nullptr,
             span<const Index> arena_offsets = {});

    // Delta lifetimes expressed on the forward timeline, so ForwardPropagation can
    // co-plan them without knowing what they are.
    static vector<MemoryPoolEntry> make_co_planned_lifetimes(const Loss&,
                                                             Index batch_size);

    void accumulate_output_deltas(size_t);

    TensorView& get_output_delta();
    const TensorView& get_output_delta() const;

    Buffer gradient;

    Buffer arena;
    vector<TensorView> output_deltas;
    vector<vector<TensorView>> slots;

    Index batch_size = 0;

    float error = 0.0f;
    float accuracy = 0.0f;
    float regularization = 0.0f;
    float loss_value = 0.0f;
    Index active_tokens_count = 0;

private:

    struct DeltaEntry
    {
        Index layer;
        size_t slot;
        TensorSpec spec;
        Index first_step;
        Index last_step;
    };

    struct DeltaLayout
    {
        vector<DeltaEntry> entries;
        vector<bool> passthrough_layers;
        vector<bool> aliases_residual_delta;
        Index aliased_residual_delta_bytes = 0;
        vector<pair<size_t, size_t>> reusable_consumer_deltas;
    };

    struct DeltaPlan
    {
        vector<vector<TensorSpec>> backward_specs;
        vector<vector<pair<size_t, size_t>>> consumer_edges;
        DeltaLayout layout;
    };

    static vector<vector<pair<size_t, size_t>>> make_consumer_edges(const NeuralNetwork&);
    static DeltaLayout build_delta_layout(const Loss&, Index batch_size,
                                          const vector<vector<TensorSpec>>&,
                                          const vector<vector<pair<size_t, size_t>>>&);
    static DeltaPlan build_delta_plan(const Loss&, Index batch_size);
    static vector<MemoryPoolEntry> to_pool_entries(const vector<DeltaEntry>&,
                                                   Index step_offset = 0);

    void setup_arena(const NeuralNetwork&,
                     const vector<vector<TensorSpec>>&,
                     const DeltaLayout&);

    void bind_deltas(const NeuralNetwork&,
                     const DeltaLayout&, span<const Index> byte_offsets,
                     uint8_t* base, Device device,
                     const vector<vector<TensorSpec>>&);

    vector<vector<pair<size_t, size_t>>> consumer_edges;
    Index output_delta_layer_index = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
