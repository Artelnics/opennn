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
#include "pool3d_operator.h"

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

    vector<TensorSpec> get_forward_specs(Index) const override;

    void set(const Shape&, const PoolingMethod&, const string&);

    void set_input_shape(const Shape& new_input_shape) override
    {
        set(new_input_shape, pooling_method, get_label());
    }

    void set_pooling_method(PoolingMethod);
    void set_pooling_method(const string&);

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    Index sequence_length = 0;
    Index input_features = 0;

    PoolingMethod pooling_method = PoolingMethod::MaxPooling;

    Pool3dOperator pool3d;

    enum Forward { Input, MaximalIndices, Output };
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
