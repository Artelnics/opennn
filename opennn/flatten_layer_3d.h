/*
//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F L A T T E N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef FLATTEN_LAYER_3D_H
#define FLATTEN_LAYER_3D_H

#include "layer.h"

namespace opennn
{

class Flatten3d : public Layer
{

public:

    Flatten3d(const dimensions& = {0,0});

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    Index get_input_height() const;
    Index get_input_width() const;

    void set(const dimensions & = {0,0});

    void forward_propagate(const vector<TensorView>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

    void back_propagate(const vector<TensorView>&,
                        const vector<TensorView>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

    void print() const override;

private:

    dimensions input_dimensions;
};


struct Flatten3dForwardPropagation : LayerForwardPropagation
{
    Flatten3dForwardPropagation(const Index& = 0, Layer* = nullptr);

    TensorView get_output_pair() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    Tensor<type, 2> outputs;
};


struct Flatten3dBackPropagation : LayerBackPropagation
{
    Flatten3dBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<TensorView> get_input_derivative_views() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    Tensor<type, 3> input_deltas;
};


}

#endif // FLATTEN_LAYER_3D_H

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software

// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/
