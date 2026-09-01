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
    struct GradientSlice
    {
        TensorView values;
        Index parameter_offset = 0;
    };

    BackPropagation(Index, Loss&,
                    Buffer* external_arena = nullptr,
                    span<const Index> arena_offsets = {},
                    Buffer* external_gradient = nullptr,
                    span<const Index> gradient_arena_offsets = {});

    virtual ~BackPropagation() = default;

    void set(Index, Loss&,
             Buffer* external_arena = nullptr,
             span<const Index> arena_offsets = {},
             Buffer* external_gradient = nullptr,
             span<const Index> gradient_arena_offsets = {});

    Loss* get_loss() const noexcept { return loss; }
    NeuralNetwork* get_neural_network() const;

    static vector<MemoryPoolEntry> make_co_planned_lifetimes(Loss&, Index batch_size);
    static vector<MemoryPoolEntry> make_gradient_co_planned_lifetimes(Loss&);

    bool has_joint_gradient_arena() const noexcept
    {
        return joint_gradient_arena;
    }

    Index gradient_logical_bytes() const noexcept
    {
        return gradient_bytes;
    }

    const vector<GradientSlice>& get_gradient_slices() const noexcept
    {
        return gradient_slices;
    }

    void link_parameter_gradients();

    void accumulate_output_deltas(size_t);

    const TensorView& input_delta_addend(size_t layer, size_t input) const noexcept;

    TensorView& get_output_delta();
    const TensorView& get_output_delta() const;

    Buffer gradient;

    Buffer arena;
    vector<Buffer> layer_scratch_storage;
    Buffer execution_workspace{Device::CUDA};
    vector<TensorView> output_deltas;
    vector<vector<TensorView>> slots;

    Index batch_size = 0;

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

    BackPropagation() = default;

    friend struct TrainingContext;

    DeltaLayout build_delta_layout(const vector<vector<TensorSpec>>&) const;
    DeltaPlan build_delta_plan();
    static vector<MemoryPoolEntry> to_pool_entries(const vector<DeltaEntry>&,
                                                   Index step_offset = 0);

    const NeuralNetwork& require_network() const;

    void setup_gradient(Buffer* external_gradient,
                        Buffer* external_arena,
                        span<const Index> gradient_arena_offsets);

    void setup_arena(const vector<vector<TensorSpec>>&,
                     const DeltaLayout&);

    void bind_deltas(const DeltaLayout&, span<const Index> byte_offsets,
                     uint8_t* base, Device device,
                     const vector<vector<TensorSpec>>&);

    vector<vector<pair<size_t, size_t>>> consumer_edges;

    vector<vector<TensorView>> input_delta_addends;
    vector<pair<size_t, size_t>> folded_consumer_edge;

    vector<GradientSlice> gradient_slices;
    vector<TensorView> layer_gradient_views;
    Index gradient_bytes = 0;
    bool joint_gradient_arena = false;

    void plan_delta_addends();

    Index output_delta_layer_index = 0;

    Loss* loss = nullptr;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
