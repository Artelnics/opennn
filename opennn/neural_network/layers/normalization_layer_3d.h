//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/layers/layer.h"
#include "opennn/neural_network/operators/layer_normalization_operator.h"

namespace opennn
{

class Normalization3d final : public Layer
{
    enum Forward {Input, Means, StandardDeviations, NormalizedInput, Output};

public:

    Normalization3d(const Shape& = Shape({0,0}),
                    const string& = "normalization_layer_3d");

    Shape get_input_shape() const noexcept override;
    Shape get_output_shape() const override;

    Index get_sequence_length() const { return sequence_length; }
    Index get_embedding_dimension() const { return embedding_dimension; }

    NormalizationMethod get_method() const { return layer_normalization.method; }

    void set_method(NormalizationMethod);

    float get_epsilon() const { return layer_normalization.epsilon; }
    void set_epsilon(float new_epsilon) { layer_normalization.epsilon = new_epsilon; }

    Index get_sources_number() const noexcept override { return layer_normalization.fuse_add ? 2 : 1; }

    vector<TensorSpec> get_forward_specs(Index) const override;
    vector<TensorSpec> get_backward_specs(Index) const override;
    bool backward_uses_input(size_t) const noexcept override { return !layer_normalization.fuse_add; }
    bool allows_input_delta_alias() const noexcept override { return layer_normalization.fuse_add; }

    void set(Index = 0, Index = 0, const string& = "normalization_layer_3d");

    void set_fuse_add(bool);

    // With fuse_add the forward writes x + residual to NormalizedInput, and only
    // back_propagate reads it. Inference plans the same slots as training, so
    // that store cost a full tensor's write per norm for a pass that never runs.
    // CUDA only: the CPU path uses the slot as a real intermediate, normalising
    // out of it rather than out of the input.
    ForwardSlotKind get_forward_slot_kind(size_t slot) const override
    {
        return slot == NormalizedInput
            && layer_normalization.fuse_add
            && get_compute_device() == Device::CUDA
                ? ForwardSlotKind::TrainingOnly
                : ForwardSlotKind::Pooled;
    }

    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 2, 3); }

    void apply_input_shape(const Shape&) override;

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    Index sequence_length = 0;
    Index embedding_dimension = 0;

    LayerNormalizationOperator layer_normalization;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
