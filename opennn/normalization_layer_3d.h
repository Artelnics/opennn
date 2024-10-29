//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NORMALIZATIONLAYER3D_H
#define NORMALIZATIONLAYER3D_H

#include <iostream>
#include <string>

#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"

namespace opennn
{
struct NormalizationLayer3DForwardPropagation;
struct NormalizationLayer3DBackPropagation;

#ifdef OPENNN_CUDA
struct NormalizationLayer3DForwardPropagationCuda;
struct NormalizationLayer3DBackPropagationCuda;
#endif

class NormalizationLayer3D : public Layer
{

public:

    // Constructors

    explicit NormalizationLayer3D();

    explicit NormalizationLayer3D(const Index&, const Index&);

    // Get

    Index get_inputs_number() const final;
    Index get_inputs_depth() const;

    dimensions get_output_dimensions() const final;

    // Parameters

    Index get_gammas_number() const;
    Index get_betas_number() const;
    Index get_parameters_number() const final;
    Tensor<type, 1> get_parameters() const final;

    // Display messages

    const bool& get_display() const;

    // Set

    void set();
    void set(const Index&, const Index&);

    void set_default();

    // Architecture

    void set_inputs_number(const Index&);
    void set_inputs_depth(const Index&);

    // Parameters

    void set_parameters(const Tensor<type, 1>&, const Index& index = 0) final;

    // Display messages

    void set_display(const bool&);

    // Parameters initialization

    void set_gammas_constant(const type&);
    void set_betas_constant(const type&);

    void set_parameters_default();
    void set_parameters_constant(const type&) final;
    void set_parameters_random() final;

    // Forward propagation

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                            unique_ptr<LayerForwardPropagation>&,
                            const bool&) final;

    // Gradient

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const final;

    void add_deltas(const vector<pair<type*, dimensions>>&) const;

    void insert_gradient(unique_ptr<LayerBackPropagation>&,
                            const Index&,
                            Tensor<type, 1>&) const final;

    // Serialization

    void from_XML(const tinyxml2::XMLDocument&) final;
    void to_XML(tinyxml2::XMLPrinter&) const final;


    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/normalization_layer_3d_cuda.h"
    #endif

protected:

    // MEMBERS

    Index inputs_number;

    Index inputs_depth;
        
    Index neurons_number;

    Tensor<type, 1> gammas;

    Tensor<type, 1> betas;

    bool display = true;

    const Eigen::array<Index, 1> sum_dimensions_1 = {2};
    const Eigen::array<Index, 2> sum_dimensions_2 = {0, 1};

    const Eigen::array<Index, 1> normalization_axis{{2}};

};


struct NormalizationLayer3DForwardPropagation : LayerForwardPropagation
{
        
    explicit NormalizationLayer3DForwardPropagation() : LayerForwardPropagation()
    {
    }

    explicit NormalizationLayer3DForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }

    pair<type*, dimensions> get_outputs_pair() const final;

    void set(const Index& new_batch_samples_number, Layer* new_layer) final;

    void print() const
    {
        cout << "Outputs:" << endl
                << outputs << endl;
    }

    Tensor<type, 3> outputs;
    Tensor<type, 3> normalized_inputs;

    Tensor<type, 3> means;
    Tensor<type, 3> standard_deviations;

    type epsilon = type(0.001);
};


struct NormalizationLayer3DBackPropagation : LayerBackPropagation
{
        

    explicit NormalizationLayer3DBackPropagation() : LayerBackPropagation()
    {

    }

    explicit NormalizationLayer3DBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const;

    void set(const Index& new_batch_samples_number, Layer* new_layer) final;

    void print() const
    {
        cout << "Gammas derivatives:" << endl
                << gammas_derivatives << endl
                << "Betas derivatives:" << endl
                << betas_derivatives << endl;
    }

    Tensor<type, 1> gammas_derivatives;
    Tensor<type, 1> betas_derivatives;

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
