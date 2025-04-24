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

class Addition3d : public Layer
{

public:

    Addition3d(const Index& = 0, const Index& = 0, const string& = "addition_layer_3d");

    Index get_sequence_length() const;
    Index get_embedding_dimension() const;

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    void set(const Index& = 0, const Index& = 0, const string& = "addition_layer_3d");

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

#ifdef OPENNN_CUDA_test

public:

    void forward_propagate_cuda(const vector<pair<type*, dimensions>>&,
                                unique_ptr<LayerForwardPropagationCuda>&,
                                const bool&) override;

    void back_propagate_cuda(const vector<pair<type*, dimensions>>&,
                             const vector<pair<type*, dimensions>>&,
                             unique_ptr<LayerForwardPropagationCuda>&,
                             unique_ptr<LayerBackPropagationCuda>&) const override;

#endif

private:

    Index sequence_length = 0;

    Index embedding_dimension = 0;
};


struct Addition3dForwardPropagation : LayerForwardPropagation
{
    Addition3dForwardPropagation(const Index& = 0, Layer* new_layer = nullptr);

    pair<type*, dimensions> get_outputs_pair() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 3> outputs;
};


struct Addition3dBackPropagation : LayerBackPropagation
{
    Addition3dBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 3> input_1_derivatives;
    Tensor<type, 3> input_2_derivatives;
};


#ifdef OPENNN_CUDA_test

    struct AdditionLayer3DForwardPropagationCuda : public LayerForwardPropagationCuda
    {
        AdditionLayer3DForwardPropagationCuda(const Index & = 0, Layer* = nullptr);

        pair<type*, dimensions> get_outputs_pair_device() const override;

        void set(const Index & = 0, Layer* = nullptr);

        void print() const override;
    };


    struct AdditionLayer3DBackPropagationCuda : public LayerBackPropagationCuda
    {
        AdditionLayer3DBackPropagationCuda(const Index & = 0, Layer* = nullptr);

        vector<pair<type*, dimensions>> get_input_derivative_pairs_device() const override;

        void set(const Index & = 0, Layer* = nullptr);

        void print() const override;

        type* inputs_1_derivatives = nullptr;
        type* inputs_2_derivatives = nullptr;
    };

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
