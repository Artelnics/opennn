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

    vector<Shape> get_forward_shapes() const override
    {
        /*
    const Index features = pooling_layer->get_output_shape()[0];
    outputs.shape = {batch_size, features};

    if (pooling_layer->get_pooling_method() == Pooling3d::PoolingMethod::MaxPooling)
        maximal_indices.resize(batch_size, features);

*/
        return {};
    }

    vector<Shape> get_backward_shapes() const override
    {
        /*
    const Shape layer_input_dimensions = pooling_layer->get_input_shape();
    const Index sequence_length = layer_input_dimensions[0];
    const Index features = layer_input_dimensions[1];

    input_gradients = {{nullptr, {batch_size, sequence_length, features}}};

*/
        return {};
    }


    Shape get_input_shape() const override;
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
