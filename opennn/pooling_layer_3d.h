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
private:

    PoolingMethod pooling_method = PoolingMethod::MaxPooling;

    enum Forward {Inputs, MaximalIndices, Outputs};

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        const Index features = input_shape[1];

        vector<Shape> shapes;

        if (pooling_method == PoolingMethod::MaxPooling)
            shapes.push_back({batch_size, features}); // MaximalIndices

        shapes.push_back({batch_size, features}); // Outputs

        return shapes;
    }

    enum Backward {OutputGradients, InputGradients};

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        // Input Gradients: {batch, seq_len, features}
        return {{batch_size, input_shape[0], input_shape[1]}};
    }

public:

    Pooling3d(const Shape& = {0, 0}, // Input shape {sequence_length, features}
              const PoolingMethod& = PoolingMethod::MaxPooling,
              const string& = "sequence_pooling_layer");

    // Getters

    Shape get_output_shape() const override;

    PoolingMethod get_pooling_method() const { return pooling_method; }

    string write_pooling_method() const;

    // Setters

    void set(const Shape&, const PoolingMethod&, const string&);

    void set_input_shape(const Shape& s) override { input_shape = s; }

    void set_pooling_method(const PoolingMethod& m) { pooling_method = m; }
    void set_pooling_method(const string&);

    // Forward / back propagation

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const noexcept override;

    // Serialization

    void from_XML(const XmlDocument&) override;
    void to_XML(XmlPrinter&) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
