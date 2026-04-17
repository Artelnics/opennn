//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pooling_layer.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

class Pooling3d final : public Layer
{

public:

    Pooling3d(const Shape& = {0, 0}, // Input shape {sequence_length, input_features}
              const PoolingMethod& = PoolingMethod::MaxPooling,
              const string& = "sequence_pooling_layer");

    Shape get_input_shape() const override { return {sequence_length, input_features}; }

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {

        vector<Shape> shapes;

        if (pooling_method == PoolingMethod::MaxPooling)
            shapes.push_back({ batch_size, input_features }); // MaximalIndices

        shapes.push_back({ batch_size, input_features }); // Output (must be last — wiring convention)

        return shapes;
    }

    enum Forward { Input, MaximalIndices, Output };
    enum Backward { OutputGradient, InputGradient };

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        return {{ batch_size, sequence_length, input_features }};
    }

    Shape get_output_shape() const override;
    PoolingMethod get_pooling_method() const { return pooling_method; }
    string write_pooling_method() const;

    void set(const Shape&, const PoolingMethod&, const string&);
    void set_input_shape(const Shape& s) override { sequence_length = s[0]; input_features = s[1]; }
    void set_pooling_method(const PoolingMethod& m) { pooling_method = m; }
    void set_pooling_method(const string&);

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    void back_propagate(ForwardPropagation&,
                        BackPropagation&,
                        size_t) const noexcept override;

    void from_XML(const XmlDocument&) override;
    void to_XML(XmlPrinter&) const override;

private:

    Index sequence_length = 0;
    Index input_features = 0;

    PoolingMethod pooling_method;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
