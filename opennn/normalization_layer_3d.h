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

#ifdef OPENNN_CUDA
struct NormalizationLayer3DForwardPropagationCuda;
struct NormalizationLayer3DBackPropagationCuda;
#endif

class NormalizationLayer3D : public Layer
{

public:

    NormalizationLayer3D(const Index& = 0, const Index& = 0, const string& = "normalization_layer_3d");

    Index get_sequence_length() const;
    Index get_embedding_dimension() const;

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    Index get_parameters_number() const override;
    Tensor<type, 1> get_parameters() const override;

    void set(const Index& = 0, const Index& = 0, const string& = "normalization_layer_3d");

    void set_sequence_length(const Index&);
    void set_embedding_dimension(const Index&);

    void set_parameters(const Tensor<type, 1>&, const Index& index = 0) override;

    void set_gammas_constant(const type&);
    void set_betas_constant(const type&);

    void set_parameters_constant(const type&) override;
    void set_parameters_random() override;

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                            unique_ptr<LayerForwardPropagation>&,
                            const bool&) override;

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void add_deltas(const vector<pair<type*, dimensions>>&) const;

    void insert_gradient(unique_ptr<LayerBackPropagation>&,
                            Index&,
                            Tensor<type, 1>&) const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/normalization_layer_3d_cuda.h"
    #endif

private:

    Index sequence_length;
        
    Tensor<type, 1> gammas;

    Tensor<type, 1> betas;

    const Eigen::array<Index, 1> sum_dimensions_1 = {2};
    const Eigen::array<Index, 2> sum_dimensions_2 = {0, 1};

    const Eigen::array<Index, 1> normalization_axis{{2}};

};


struct NormalizationLayer3DForwardPropagation : LayerForwardPropagation
{        
    NormalizationLayer3DForwardPropagation(const Index& = 0, Layer* = nullptr);

    pair<type*, dimensions> get_outputs_pair() const override;

    void set(const Index& = 0, Layer* = 0);

    void print() const override;

    Tensor<type, 3> outputs;

    Tensor<type, 3> means;
    Tensor<type, 3> standard_deviations;

    type epsilon = type(0.001);
};


struct NormalizationLayer3DBackPropagation : LayerBackPropagation
{

    NormalizationLayer3DBackPropagation(const Index& new_batch_samples_number = 0,
                                                 Layer* new_layer = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const;

    Tensor<type, 1> gamma_derivatives;
    Tensor<type, 1> beta_derivatives;

    Tensor<type, 3> scaled_deltas;
    Tensor<type, 3> standard_deviation_derivatives;
    Tensor<type, 2> aux_2d;
        
    Tensor<type, 3> input_derivatives;
};


#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/normalization_layer_3d_forward_propagation_cuda.h"
    #include "../../opennn_cuda/opennn_cuda/normalization_layer_3d_back_propagation_cuda.h"
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
