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
    // The Loss is retained, mirroring the NeuralNetwork that ForwardPropagation
    // keeps, so holders of a BackPropagation can reach the network without being
    // handed it separately. It is a borrowed pointer: a BackPropagation must not
    // outlive the Loss it was built from, and must be re-set if that Loss is
    // pointed at a different network.
    BackPropagation(Index, Loss&,
                    Buffer* external_arena = nullptr,
                    span<const Index> arena_offsets = {});

    virtual ~BackPropagation() = default;

    void set(Index, Loss&,
             Buffer* external_arena = nullptr,
             span<const Index> arena_offsets = {});

    Loss* get_loss() const noexcept { return loss; }
    NeuralNetwork* get_neural_network() const;

    // Delta lifetimes expressed on the forward timeline, so ForwardPropagation can
    // co-plan them without knowing what they are.
    static vector<MemoryPoolEntry> make_co_planned_lifetimes(Loss&, Index batch_size);

    void accumulate_output_deltas(size_t);

    TensorView& get_output_delta();
    const TensorView& get_output_delta() const;

    Buffer gradient;

    Buffer arena;
    vector<TensorView> output_deltas;
    vector<vector<TensorView>> slots;

    Index batch_size = 0;

    // What the last batch produced. Nothing here describes the delta layout or
    // the arena: these are outputs of running a batch, reset before each one,
    // and they are grouped so that is visible from the declaration.
    struct Metrics
    {
        float error = 0.0f;
        float accuracy = 0.0f;
        float regularization = 0.0f;
        float loss_value = 0.0f;
        Index active_tokens_count = 0;

        void reset() { *this = Metrics{}; }
    };

    Metrics metrics;

private:

    struct DeltaEntry
    {
        Index layer;
        // Slot 0 is the layer's output delta; slot 1+i is the delta for its i-th
        // input. That offset is why slots[] is sized backward_specs[i].size() + 1.
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
        DeltaLayout layout;
    };

    // Planning-only instance: make_co_planned_lifetimes must produce the delta
    // layout before any BackPropagation is built into the arena that layout sizes,
    // so it borrows a Loss and a batch size into an object that allocates nothing.
    BackPropagation() = default;

    vector<vector<pair<size_t, size_t>>> make_consumer_edges() const;
    DeltaLayout build_delta_layout(const vector<vector<TensorSpec>>&) const;
    DeltaPlan build_delta_plan();
    static vector<MemoryPoolEntry> to_pool_entries(const vector<DeltaEntry>&,
                                                   Index step_offset = 0);

    const NeuralNetwork& require_network() const;

    void setup_gradient();

    void setup_arena(const vector<vector<TensorSpec>>&,
                     const DeltaLayout&);

    void bind_deltas(const DeltaLayout&, span<const Index> byte_offsets,
                     uint8_t* base, Device device,
                     const vector<vector<TensorSpec>>&);

    // For each layer, the (consumer layer, input position) pairs that read its
    // output. Built by build_delta_plan, then read while binding the deltas and
    // again by accumulate_output_deltas on every backward pass, so it outlives the
    // plan that produced it and is stored here rather than passed around.
    vector<vector<pair<size_t, size_t>>> consumer_edges;

    Index output_delta_layer_index = 0;

    Loss* loss = nullptr;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
