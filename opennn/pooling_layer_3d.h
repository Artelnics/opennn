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

    // Getters

    Shape get_input_shape() const override { return {sequence_length, input_features}; }
    Shape get_output_shape() const override;

    PoolingMethod get_pooling_method() const { return pooling_method; }

    string write_pooling_method() const;

    // Setters

    void set(const Shape&, const PoolingMethod&, const string&);
    void set_input_shape(const Shape& shape) override { sequence_length = shape[0]; input_features = shape[1]; }
    void set_pooling_method(const PoolingMethod& new_pooling_method) { pooling_method = new_pooling_method; }
    void set_pooling_method(const string&);

    // Forward / back propagation

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    // Serialization

    void from_XML(const XmlDocument&) override;
    void to_XML(XmlPrinter&) const override;

private:

    enum Forward { Input, MaximalIndices, Output };
    enum Backward { OutputDelta, InputDelta };

    vector<pair<Shape, Type>> get_forward_specs(const Index batch_size) const override
    {
        const Type act = activation_dtype;
        return {
            /*MaximalIndices*/ {(pooling_method == PoolingMethod::MaxPooling)
                                    ? Shape{batch_size, input_features}
                                    : Shape{},
                                Type::FP32},
            /*Output*/         {{batch_size, input_features}, act}, // must be last
        };
    }

    vector<pair<Shape, Type>> get_backward_specs(Index batch_size) const override
    {
        return {{{batch_size, sequence_length, input_features}, activation_dtype}};
    }

    Index sequence_length = 0;
    Index input_features = 0;

    PoolingMethod pooling_method = PoolingMethod::MaxPooling;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
