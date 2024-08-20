//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NORMALIZATIONLAYER3D_H
#define NORMALIZATIONLAYER3D_H

// System includes

//#include <cstdlib>
#include <iostream>
#include <string>

// OpenNN includes

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

        const Tensor<type, 1>& get_gammas() const;
        const Tensor<type, 1>& get_betas() const;

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
        void set_name(const string&);

        // Architecture

        void set_inputs_number(const Index&);
        void set_inputs_depth(const Index&);

        // Parameters

        void set_gammas(const Tensor<type, 1>&);
        void set_betas(const Tensor<type, 1>&);

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

        void forward_propagate(const Tensor<pair<type*, dimensions>, 1>&,
                               LayerForwardPropagation*,
                               const bool&) final;

        // Gradient

        void back_propagate(const Tensor<pair<type*, dimensions>, 1>&,
                                      const Tensor<pair<type*, dimensions>, 1>&,
                                      LayerForwardPropagation*,
                                      LayerBackPropagation*) const final;

        void add_deltas(const Tensor<pair<type*, dimensions>, 1>& deltas_pair) const;

        void insert_gradient(LayerBackPropagation*,
                             const Index&,
                             Tensor<type, 1>&) const final;

        // Serialization

        void from_XML(const tinyxml2::XMLDocument&) final;
        void write_XML(tinyxml2::XMLPrinter&) const final;


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
    };


    struct NormalizationLayer3DForwardPropagation : LayerForwardPropagation
    {
        // Default constructor

        explicit NormalizationLayer3DForwardPropagation() : LayerForwardPropagation()
        {
        }


        explicit NormalizationLayer3DForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
            : LayerForwardPropagation()
        {
            set(new_batch_samples_number, new_layer);
        }


        virtual ~NormalizationLayer3DForwardPropagation()
        {
        }


        pair<type*, dimensions> get_outputs_pair() const final;


        void set(const Index& new_batch_samples_number, Layer* new_layer) final;


        void print() const
        {
            cout << "Outputs:" << endl;
            cout << outputs << endl;
        }

        Tensor<type, 3> outputs;
        Tensor<type, 3> normalized_inputs;

        Tensor<type, 3> means;
        Tensor<type, 3> standard_deviations;

        type epsilon = type(0.001);
    };


    struct NormalizationLayer3DBackPropagation : LayerBackPropagation
    {
        // Default constructor

        explicit NormalizationLayer3DBackPropagation() : LayerBackPropagation()
        {

        }


        explicit NormalizationLayer3DBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
            : LayerBackPropagation()
        {
            set(new_batch_samples_number, new_layer);
        }


        virtual ~NormalizationLayer3DBackPropagation()
        {
        }

        void set(const Index& new_batch_samples_number, Layer* new_layer) final;


        void print() const
        {
            cout << "Gammas derivatives:" << endl;
            cout << gammas_derivatives << endl;

            cout << "Betas derivatives:" << endl;
            cout << betas_derivatives << endl;
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
