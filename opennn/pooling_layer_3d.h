//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pooling_layer.h"

namespace opennn
{

class Pooling3d final : public Layer
{

public:

    Pooling3d(const Shape& = {0, 0}, // Input shape {sequence_length, features}
              const PoolingMethod& = PoolingMethod::MaxPooling,
              const string& = "sequence_pooling_layer");

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        const Index features = input_shape[1];

        vector<Shape> shapes;

        if (pooling_method == PoolingMethod::MaxPooling)
            shapes.push_back({ batch_size, features }); // MaximalIndices

        shapes.push_back({ batch_size, features }); // Outputs (must be last — wiring convention)

        return shapes;
    }

    enum Forward { Inputs, MaximalIndices, Outputs };
    enum Backward { OutputGradients, InputGradients };

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        const Index seq_len = input_shape[0];
        const Index features = input_shape[1];

        // Input Gradients (dX): {batch, seq_len, features}
        return {{ batch_size, seq_len, features }};
    }

    Shape get_output_shape() const override;
    PoolingMethod get_pooling_method() const { return pooling_method; }
    string write_pooling_method() const;

    void set(const Shape&, const PoolingMethod&, const string&);
    void set_input_shape(const Shape& s) override { input_shape = s; }
    void set_pooling_method(const PoolingMethod& m) { pooling_method = m; }
    void set_pooling_method(const string&);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;

    void back_propagate(ForwardPropagation&,
                        BackPropagation&,
                        size_t) const override;

    void from_XML(const XmlDocument&) override;
    void to_XML(XmlPrinter&) const override;

private:

    PoolingMethod pooling_method;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
