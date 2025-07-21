//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef ADDITIONLAYER3D_H
#define ADDITIONLAYER3D_H

#include "layer.h"

namespace opennn
{

#ifdef OPENNN_CUDA
    struct AdditionLayer3DForwardPropagationCuda;
    struct AdditionLayer3DBackPropagationCuda;
#endif

class AdditionLayer3D : public Layer
{

public:

    AdditionLayer3D(const Index& = 0, const Index& = 0, const string& = "addition_layer_3d");

    Index get_sequence_length() const;
    Index get_embedding_length() const;

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    void set(const Index& = 0, const Index& = 0, const string & = "addition_layer_3d");

    void set_sequence_length(const Index&);
    void set_embedding_length(const Index&);

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void add_deltas(const vector<pair<type*, dimensions>>&) const;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/addition_layer_3d_cuda.h"
    #endif

private:

    Index sequence_length = 0;

    Index embedding_length = 0;
};


struct AdditionLayer3DForwardPropagation : LayerForwardPropagation
{
    AdditionLayer3DForwardPropagation(const Index& = 0, Layer* new_layer = nullptr);

    pair<type*, dimensions> get_outputs_pair() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 3> outputs;
};


struct AdditionLayer3DBackPropagation : LayerBackPropagation
{
    AdditionLayer3DBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

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
