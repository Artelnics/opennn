//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef ADDITIONLAYER3D_H
#define ADDITIONLAYER3D_H

// System includes

#include <cstdlib>
#include <iostream>
#include <string>

// OpenNN includes

#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"

namespace opennn
{

    struct AdditionLayer3DForwardPropagation;
    struct AdditionLayer3DBackPropagation;

    struct NormalizationLayer3DForwardPropagation;
    struct NormalizationLayer3DBackPropagation;

#ifdef OPENNN_CUDA
    struct AdditionLayer3DForwardPropagationCuda;
    struct AdditionLayer3DBackPropagationCuda;
#endif

    // @todo explain

    class AdditionLayer3D : public Layer
    {

    public:

        // Constructors

        explicit AdditionLayer3D();
        explicit AdditionLayer3D(const Index&, const Index&);

        // Get methods

        Index get_inputs_number() const final;
        Index get_inputs_size() const;

        // Parameters
        Index get_parameters_number() const final;
        Tensor<type, 1> get_parameters() const final;

        // Display messages

        const bool& get_display() const;

        // Set methods

        void set();
        void set(const Index&, const Index&);

        void set_default();
        void set_name(const string&);

        void set_inputs_size(const Index&);

        // Parameters

        void set_parameters(const Tensor<type, 1>&, const Index& index = 0) final;

        // Display messages

        void set_display(const bool&);

        // Forward propagation

        void forward_propagate(const Tensor<pair<type*, dimensions>, 1>&,
            LayerForwardPropagation*,
            const bool&) final;

        void forward_propagate(const Tensor<pair<type*, dimensions>, 1>&,
            Tensor<type, 1>&,
            LayerForwardPropagation*);

        // Delta methods

        void calculate_hidden_delta(LayerForwardPropagation*,
            LayerBackPropagation*,
            LayerForwardPropagation*,
            LayerBackPropagation*) const final;

        void calculate_hidden_delta(NormalizationLayer3DForwardPropagation*,
            NormalizationLayer3DBackPropagation*,
            AdditionLayer3DBackPropagation*) const;

        // Gradient methods

        void calculate_error_gradient(const Tensor<pair<type*, dimensions>, 1>&,
            LayerForwardPropagation*,
            LayerBackPropagation*) const final;

        void insert_gradient(LayerBackPropagation*,
            const Index&,
            Tensor<type, 1>&) const final;

        // Expression methods

        string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;

        // Serialization methods

        void from_XML(const tinyxml2::XMLDocument&) final;
        void write_XML(tinyxml2::XMLPrinter&) const final;

        #ifdef OPENNN_CUDA
            #include "../../opennn_cuda/opennn_cuda/addition_layer_3d_cuda.h"
        #endif

    protected:

        // MEMBERS

        /// Inputs number

        Index inputs_number = 0;

        /// Inputs size

        Index inputs_size = 0;

        /// Display messages to screen.

        bool display = true;

    };


    struct AdditionLayer3DForwardPropagation : LayerForwardPropagation
    {
        // Default constructor

        explicit AdditionLayer3DForwardPropagation() : LayerForwardPropagation()
        {
        }


        explicit AdditionLayer3DForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
            : LayerForwardPropagation()
        {
            set(new_batch_samples_number, new_layer);
        }


        virtual ~AdditionLayer3DForwardPropagation()
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
    };


    struct AdditionLayer3DBackPropagation : LayerBackPropagation
    {
        // Default constructor

        explicit AdditionLayer3DBackPropagation() : LayerBackPropagation()
        {

        }


        explicit AdditionLayer3DBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
            : LayerBackPropagation()
        {
            set(new_batch_samples_number, new_layer);
        }


        virtual ~AdditionLayer3DBackPropagation()
        {
        }


        pair<type*, dimensions> get_deltas_pair() const final;


        void set(const Index& new_batch_samples_number, Layer* new_layer) final;


        void print() const
        {
            cout << "Deltas:" << endl;
            cout << deltas << endl;
        }

        Tensor<type, 3> deltas;
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
