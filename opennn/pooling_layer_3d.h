//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "pooling_layer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

class Pooling3d final : public Layer
{
public:

    Pooling3d(const Shape& = {0, 0},
              const PoolingMethod& = PoolingMethod::MaxPooling,
              const string& = "sequence_pooling_layer");

    Shape get_input_shape() const override { return {sequence_length, input_features}; }
    Shape get_output_shape() const override;

    Index get_sequence_length() const { return sequence_length; }
    Index get_input_features() const { return input_features; }

    PoolingMethod get_pooling_method() const { return pooling_method; }

    string write_pooling_method() const;

    vector<pair<Shape, Type>> get_forward_specs(Index batch_size) const override;
    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override;

    void set(const Shape&, const PoolingMethod&, const string&);

    void set_input_shape(const Shape& new_input_shape) override
    {
        sequence_length = new_input_shape[0];
        input_features  = new_input_shape[1];
    }

    void set_pooling_method(const PoolingMethod& new_pooling_method) { pooling_method = new_pooling_method; }
    void set_pooling_method(const string&);

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    void from_JSON(const JsonDocument&) override;
    void to_JSON(JsonWriter&) const override;

private:

    Index sequence_length = 0;
    Index input_features = 0;

    PoolingMethod pooling_method = PoolingMethod::MaxPooling;

    enum Forward { Input, MaximalIndices, Output };
    enum Backward { OutputDelta, InputDelta };
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
