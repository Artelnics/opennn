//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S E Q U E N C E   P O O L I N G   L A Y E R   C L A S S   H E A D E R
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

    Shape get_input_shape() const override;
    Shape get_output_shape() const override;
    PoolingMethod get_pooling_method() const;
    string write_pooling_method() const;

    void set(const Shape&, const PoolingMethod&, const string&);
    void set_pooling_method(const PoolingMethod&);
    void set_pooling_method(const string&);

    void forward_propagate(const vector<TensorView>&,
                           unique_ptr<LayerForwardPropagation>&,
                           bool) override;

    void back_propagate(const vector<TensorView>&,
                        const vector<TensorView>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    void print() const override;

private:
    Shape input_shape;
    PoolingMethod pooling_method;
};


struct Pooling3dForwardPropagation final : LayerForwardPropagation
{
    Pooling3dForwardPropagation(const Index = 0, Layer* = nullptr);

    void initialize() override;

    MatrixI maximal_indices;
};


struct Pooling3dBackPropagation final : LayerBackPropagation
{
    Pooling3dBackPropagation(const Index = 0, Layer* = nullptr);

    void initialize() override;

    Tensor3 input_derivatives;
};

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
