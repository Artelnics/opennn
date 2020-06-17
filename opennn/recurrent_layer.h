//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef RECURRENTLAYER_H
#define RECURRENTLAYER_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
#include "layer.h"


#include "perceptron_layer.h"

namespace OpenNN
{

/// This class represents a layer of neurons.
/// Layers of neurons will be used to construct multilayer neurons.

class RecurrentLayer : public Layer
{

public:

    /// Enumeration of the available activation functions for the recurrent layer.

    enum ActivationFunction{Threshold, SymmetricThreshold, Logistic, HyperbolicTangent, Linear, RectifiedLinear, ExponentialLinear, ScaledExponentialLinear, SoftPlus, SoftSign, HardSigmoid};

    struct RecurrentLayerForwardPropagation : ForwardPropagation
    {
        /// Default constructor.

        explicit RecurrentLayerForwardPropagation() : ForwardPropagation(){}

        virtual ~RecurrentLayerForwardPropagation() {}

        void allocate()
        {
/*
           const RecurrentLayer* recurrent_layer = dynamic_cast<RecurrentLayer*>(trainable_layers_pointers[i]);

           const Index neurons_number = recurrent_layer->get_neurons_number();

           layers[i].combinations_2d = Tensor<type, 2>(batch_instances_number, neurons_number);
           layers[i].activations_2d = Tensor<type, 2>(batch_instances_number, neurons_number);
           layers[i].activations_derivatives = Tensor<type, 2>(batch_instances_number, neurons_number);
*/
        }
    };


   // Constructors

   explicit RecurrentLayer();

   explicit RecurrentLayer(const Index&, const Index&);

   RecurrentLayer(const RecurrentLayer&);

   // Destructor
   
   virtual ~RecurrentLayer();

   // Get methods

   bool is_empty() const;

   Tensor<Index, 1> get_input_variables_dimensions() const;

   Index get_inputs_number() const;
   Index get_neurons_number() const;

   Tensor<type, 1> get_hidden_states() const;

   // Parameters

   Index get_timesteps()const;

   Tensor<type, 2> get_biases() const;
   Tensor<type, 2> get_input_weights() const;
   Tensor<type, 2> get_recurrent_weights() const;

   Index get_biases_number() const;
   Index get_input_weights_number() const;
   Index get_recurrent_weights_number() const;

   Index get_parameters_number() const;
   Tensor<type, 1> get_parameters() const;

   Tensor<type, 2> get_biases(const Tensor<type, 1>&) const;
   Tensor<type, 2> get_input_weights(const Tensor<type, 1>&) const;
   Tensor<type, 2> get_recurrent_weights(const Tensor<type, 1>&) const;

   // Activation functions

   const RecurrentLayer::ActivationFunction& get_activation_function() const;

   string write_activation_function() const;

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&, const Index&);
   void set(const RecurrentLayer&);

   void set_default();

   // Architecture

   void set_inputs_number(const Index&);
   void set_neurons_number(const Index&);
   void set_input_shape(const Tensor<Index, 1>&);

   // Parameters

   void set_timesteps(const Index&);

   void set_biases(const Tensor<type, 2>&);

   void set_input_weights(const Tensor<type, 2>&);

   void set_recurrent_weights(const Tensor<type, 2>&);

   void set_parameters(const Tensor<type, 1>&, const Index& index=0);

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   // Display messages

   void set_display(const bool&);

   // Parameters initialization methods

   void initialize_hidden_states(const type&);

   void set_biases_constant(const type&);

   void initialize_input_weights(const type&);
   void initialize_recurrent_weights(const type&);
   void initialize_input_weights_Glorot(const type&, const type&);

   void set_parameters_constant(const type&);

   void set_parameters_random();

   // neuron layer combinations_2d

//   Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&) const;

//   Tensor<type, 2> calculate_combinations(const Tensor<type, 2>&);

//   Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&) const;

//   Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   void calculate_combinations(const Tensor<type, 2>& inputs,
                               const Tensor<type, 2>& input_weights,
                               const Tensor<type, 2>& biases,
                               const Tensor<type, 2>& recurrent_weights,
                               Tensor<type, 2>& combinations_2d);

   void calculate_combinations(const Tensor<type, 1>& inputs,
                               const Tensor<type, 2>& input_weights,
                               const Tensor<type, 2>& biases,
                               const Tensor<type, 2>& recurrent_weights,
                               Tensor<type, 1>& combinations_1d) const;


   // neuron layer activations_2d

   void calculate_activations(const Tensor<type, 1>& combinations_1d, Tensor<type, 1>& activations_1d) const;
   void calculate_activations(const Tensor<type, 2>& combinations_2d, Tensor<type, 2>& activations_2d) const;

   void calculate_activations_derivatives(const Tensor<type, 1>& combinations_1d,
                                          Tensor<type, 1>& activations_1d,
                                          Tensor<type, 1>& activations_derivatives_1d) const;

   void calculate_activations_derivatives(const Tensor<type, 2>& combinations_2d,
                                          Tensor<type, 2>& activations_2d,
                                          Tensor<type, 2>& activations_derivatives) const;

   // neuron layer outputs

   void update_hidden_states(const Tensor<type, 1>&);

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);
   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 1>&);

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&);

   Tensor<type, 2> calculate_hidden_delta(Layer*, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   void forward_propagate(const Tensor<type, 2>& inputs, ForwardPropagation& forward_propagation)
   {/*

       const Index instances_number = inputs.dimension(0);
       const Index neurons_number = get_neurons_number();

       Tensor<type, 1> combinations_1d(forward_propagation.combinations_2d.dimension(1));
       Tensor<type, 1> activations_derivatives_1d(forward_propagation.combinations_2d.dimension(1));

       for(Index i = 0; i < instances_number; i++)
       {
           if(i%timesteps == 0) hidden_states.setZero();

           const Tensor<type, 1> current_inputs = inputs.chip(i, 1);

           calculate_combinations(current_inputs, input_weights, biases, recurrent_weights, combinations_1d);

           calculate_activations_derivatives(combinations_1d, hidden_states, activations_derivatives_1d);

           for(Index j = 0; j < neurons_number; j++)
           {
               forward_propagation.combinations_2d(i,j) = combinations_1d(j);
               forward_propagation.activations_2d(i,j) = hidden_states(j);
               forward_propagation.activations_derivatives_2d(i,j) = activations_derivatives_1d(j);
           }
       }
       */
   }

   // Gradient

   Tensor<type, 1> calculate_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&);

   Tensor<type, 1> calculate_input_weights_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&);
   Tensor<type, 1> calculate_recurrent_weights_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&);
   Tensor<type, 1> calculate_biases_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&);

   // Expression methods

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_activation_function_expression() const;

   // Utilities

   Tensor<type, 2> multiply_rows(const Tensor<type,2>&, const Tensor<type,1>&) const;

   

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&) {}

   void write_XML(tinyxml2::XMLPrinter&) const {}

protected:

   Index timesteps = 1;

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's trabsfer function to generate the neuron's output.

   Tensor<type, 2> biases;

   Tensor<type, 2> input_weights;

   /// This matrix containing conection strengths from a recurrent layer inputs to its neurons.

   Tensor<type, 2> recurrent_weights;

   /// Activation function variable.

   ActivationFunction activation_function = HyperbolicTangent;

   Tensor<type, 1> hidden_states;

   /// Display messages to screen.

   bool display;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/recurrent_layer_cuda.h"
#endif

};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
