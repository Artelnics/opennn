//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NORMALIZATIONLAYER3D_H
#define NORMALIZATIONLAYER3D_H

#include "layer.h"

namespace opennn
{

class Normalization3d : public Layer
{

public:

    Normalization3d(const dimensions& = dimensions({0,0}),
                    const string& = "normalization_layer_3d");

    Index get_sequence_length() const;
    Index get_embedding_dimension() const;

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    vector<pair<type*, Index>> get_parameter_pairs() const override;

    void set(const Index& = 0, const Index& = 0, const string& = "normalization_layer_3d");

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void insert_gradient(unique_ptr<LayerBackPropagation>&,
                         Index&,
                         Tensor<type, 1>&) const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    #ifdef OPENNN_CUDA
        // @todo
    #endif

private:

    Index sequence_length;
        
    Tensor<type, 1> gammas;

    Tensor<type, 1> betas;

    const type epsilon = numeric_limits<type>::epsilon();
};


struct Normalization3dForwardPropagation : LayerForwardPropagation
{        
    Normalization3dForwardPropagation(const Index& = 0, Layer* = nullptr);

    pair<type*, dimensions> get_output_pair() const override;

    void set(const Index& = 0, Layer* = 0);

    void print() const override;

    Tensor<type, 3> outputs;

    Tensor<type, 2> means;
    Tensor<type, 2> standard_deviations;
};


struct Normalization3dBackPropagation : LayerBackPropagation
{
    Normalization3dBackPropagation(const Index& new_batch_size = 0,
                                   Layer* new_layer = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const;

    vector<pair<type*, Index>> get_parameter_delta_pairs() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const;

    Tensor<type, 1> gamma_derivatives;
    Tensor<type, 1> beta_derivatives;

    Tensor<type, 3> input_deltas;

    Tensor<type, 3> scaled_deltas;
    Tensor<type, 3> standard_deviation_derivatives;
    Tensor<type, 2> aux_2d;
        
};


#ifdef OPENNN_CUDA
    // @todo
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
