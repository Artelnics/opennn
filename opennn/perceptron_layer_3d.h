//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   3 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PERCEPTRONLAYER3D_H
#define PERCEPTRONLAYER3D_H



#include <iostream>
#include <string>



#include "config.h"
#include "layer.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"

namespace opennn
{

struct PerceptronLayer3DForwardPropagation;
struct PerceptronLayer3DBackPropagation;

#ifdef OPENNN_CUDA
struct PerceptronLayer3DForwardPropagationCuda;
struct PerceptronLayer3DBackPropagationCuda;
#endif


class PerceptronLayer3D : public Layer
{

public:

    enum class ActivationFunction{HyperbolicTangent,
                                  Linear,
                                  RectifiedLinear};

   // Constructors

   explicit PerceptronLayer3D();

   explicit PerceptronLayer3D(const Index&,
                              const Index&,
                              const Index&,
                              const ActivationFunction& = PerceptronLayer3D::ActivationFunction::HyperbolicTangent);

   // Get

   Index get_inputs_number() const final;
   Index get_inputs_depth() const;
   Index get_neurons_number() const final;

   dimensions get_output_dimensions() const final;

   // Parameters

   const Tensor<type, 1>& get_biases() const;
   const Tensor<type, 2>& get_synaptic_weights() const;

   Index get_biases_number() const;
   Index get_synaptic_weights_number() const;
   Index get_parameters_number() const final;
   type get_dropout_rate() const;
   Tensor<type, 1> get_parameters() const final;

   // Activation functions

   const PerceptronLayer3D::ActivationFunction& get_activation_function() const;

   string write_activation_function() const;

   // Display messages

   const bool& get_display() const;

   // Set

   void set();
   void set(const Index&,
            const Index&,
            const Index&,
            const PerceptronLayer3D::ActivationFunction& = PerceptronLayer3D::ActivationFunction::HyperbolicTangent);

   void set_default();

   // Architecture

   void set_inputs_number(const Index&) final;
   void set_inputs_depth(const Index&);
   void set_neurons_number(const Index&) final;

   // Parameters

   void set_biases(const Tensor<type, 1>&);
   void set_synaptic_weights(const Tensor<type, 2>&);

   void set_parameters(const Tensor<type, 1>&, const Index& index = 0) final;

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);
   void set_dropout_rate(const type&);

   // Display messages

   void set_display(const bool&);

   // Parameters initialization

   void set_parameters_constant(const type&) final;
   void set_parameters_random() final;
   void set_parameters_glorot();

   // Forward propagation

   void calculate_combinations(const Tensor<type, 3>&,
                               Tensor<type, 3>&) const;

   void dropout(Tensor<type, 3>&) const;

   void calculate_activations(Tensor<type, 3>&,
                              Tensor<type, 3>&) const;

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          unique_ptr<LayerForwardPropagation>,
                          const bool&) final;

   // Gradient

   void back_propagate(const vector<pair<type*, dimensions>>&,
                                 const vector<pair<type*, dimensions>>&,
                                 unique_ptr<LayerForwardPropagation>,
                                 unique_ptr<LayerBackPropagation>) const final;

   void add_deltas(const vector<pair<type*, dimensions>>&) const;

   void insert_gradient(unique_ptr<LayerBackPropagation>,
                        const Index&,
                        Tensor<type, 1>&) const final;

   // Serialization

   void from_XML(const tinyxml2::XMLDocument&) final;
   void to_XML(tinyxml2::XMLPrinter&) const final;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/perceptron_layer_3d_cuda.h"
    #endif

protected:

   // MEMBERS

   Index inputs_number;

   Tensor<type, 1> biases;

   Tensor<type, 2> synaptic_weights;

   ActivationFunction activation_function;

   type dropout_rate = type(0);

   bool display = true;

   Tensor<type, 3> empty;
};


struct PerceptronLayer3DForwardPropagation : LayerForwardPropagation
{
    

    explicit PerceptronLayer3DForwardPropagation() : LayerForwardPropagation()
    {
    }


    explicit PerceptronLayer3DForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
     : LayerForwardPropagation()
    {
    set(new_batch_samples_number, new_layer);
    }


    virtual ~PerceptronLayer3DForwardPropagation()
    {
    }


    pair<type*, dimensions> get_outputs_pair() const final;


    void set(const Index& new_batch_samples_number, Layer* new_layer) final;


    void print() const;

    Tensor<type, 3> outputs;

    Tensor<type, 3> activations_derivatives;
};


struct PerceptronLayer3DBackPropagation : LayerBackPropagation
{
    

    explicit PerceptronLayer3DBackPropagation() : LayerBackPropagation()
    {

    }

    explicit PerceptronLayer3DBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }


    virtual ~PerceptronLayer3DBackPropagation()
    {
    }

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const;

    void set(const Index& new_batch_samples_number, Layer* new_layer) final;


    void print() const
    {
        cout << "Biases derivatives:" << endl;
        cout << biases_derivatives << endl;

        cout << "Synaptic weights derivatives:" << endl;
        cout << synaptic_weights_derivatives << endl;
    }

    Tensor<type, 1> biases_derivatives;
    Tensor<type, 2> synaptic_weights_derivatives;

    Tensor<type, 3> combinations_derivatives;
    Tensor<type, 3> input_derivatives;
};


#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/perceptron_layer_3d_forward_propagation_cuda.h"
    #include "../../opennn_cuda/opennn_cuda/perceptron_layer_3d_back_propagation_cuda.h"
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
