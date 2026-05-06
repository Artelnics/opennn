//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

class Normalization3d final : public Layer
{
public:

    Normalization3d(const Shape& = Shape({0,0}),
                    const string& = "normalization_layer_3d");

    Shape get_input_shape() const override;
    Shape get_output_shape() const override;

    Index get_sequence_length() const { return sequence_length; }
    Index get_embedding_dimension() const { return embedding_dimension; }

    vector<Operator*> get_operators() override;

    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;

    void set(Index = 0, Index = 0, const string& = "normalization_layer_3d");

    void set_input_shape(const Shape& new_input_shape) override
    {
        if (new_input_shape.rank >= 2)
        {
            sequence_length     = new_input_shape[0];
            embedding_dimension = new_input_shape[1];
        }
    }

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    void read_JSON_body(const Json*) override;

private:

    Index sequence_length = 0;
    Index embedding_dimension = 0;

    LayerNorm layer_norm;

    enum Forward {Input, Means, StandardDeviations, NormalizedInput, Output};
    enum Backward {OutputDelta, InputDelta};
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
