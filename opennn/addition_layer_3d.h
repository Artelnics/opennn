//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef ADDITIONLAYER3D_H
#define ADDITIONLAYER3D_H

#include <iostream>
#include <string>

#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"

namespace opennn
{

#ifdef OPENNN_CUDA
    struct AdditionLayer3DForwardPropagationCuda;
    struct AdditionLayer3DBackPropagationCuda;
#endif

class AdditionLayer3D : public Layer
{

public:

    explicit AdditionLayer3D(const Index& = 0, const Index& = 0);

    Index get_inputs_number() const;
    Index get_inputs_depth() const;

    dimensions get_output_dimensions() const final;

    void set(const Index& = 0, const Index& = 0);

    void set_inputs_number(const Index&);
    void set_inputs_depth(const Index&);

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) final;

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const final;

    void from_XML(const XMLDocument&) final;
    void to_XML(XMLPrinter&) const final;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/addition_layer_3d_cuda.h"
    #endif

private:

    Index inputs_number = 0;

    Index inputs_depth = 0;
};


struct AdditionLayer3DForwardPropagation : LayerForwardPropagation
{
    explicit AdditionLayer3DForwardPropagation(const Index& = 0, Layer* new_layer = nullptr);

    pair<type*, dimensions> get_outputs_pair() const final;

    void set(const Index& = 0, Layer* = nullptr) final;

    void print() const;

    Tensor<type, 3> outputs;
};


struct AdditionLayer3DBackPropagation : LayerBackPropagation
{
    explicit AdditionLayer3DBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const;

    void set(const Index& = 0, Layer* = nullptr) final;

    void print() const;

    Tensor<type, 3> input_1_derivatives;
    Tensor<type, 3> input_2_derivatives;
};

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/addition_layer_3d_forward_propagation_cuda.h"
    #include "../../opennn_cuda/opennn_cuda/addition_layer_3d_back_propagation_cuda.h"
#endif
}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
