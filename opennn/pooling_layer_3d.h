//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"

namespace opennn
{

class Pooling3d final : public Layer
{

public:

    enum class PoolingMethod{MaxPooling, AveragePooling};

    Pooling3d(const Shape& = {0, 0}, // Input shape {sequence_length, features}
              const PoolingMethod& = PoolingMethod::MaxPooling,
              const string& = "sequence_pooling_layer");

    vector<Shape> get_forward_shapes(const Index batch_size) const override
    {
        const Index features = input_shape[1];

        vector<Shape> shapes;
        shapes.push_back({ batch_size, features }); // Outputs

        if (pooling_method == PoolingMethod::MaxPooling)
            shapes.push_back({ batch_size, features }); // MaximalIndices

        return shapes;
    }

    vector<Shape> get_backward_shapes(Index batch_size) const override
    {
        const Index seq_len = input_shape[0];
        const Index features = input_shape[1];

        // Input Gradients (dX): {batch, seq_len, features}
        return {{ batch_size, seq_len, features }};
    }

    Shape get_output_shape() const override;
    PoolingMethod get_pooling_method() const;
    string write_pooling_method() const;

    void set(const Shape&, const PoolingMethod&, const string&);
    void set_input_shape(const Shape&) override;
    void set_pooling_method(const PoolingMethod&);
    void set_pooling_method(const string&);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;

    void back_propagate(ForwardPropagation&,
                        BackPropagation&,
                        size_t) const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    void print() const override;

private:

    Shape input_shape;
    PoolingMethod pooling_method;
};


struct Pooling3dForwardPropagation final : LayerForwardPropagation
{
    MatrixI maximal_indices;
};



#ifdef CUDA

struct Pooling3dForwardPropagationCuda : public LayerForwardPropagationCuda
{
    TensorCuda maximal_indices_device;
};

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
