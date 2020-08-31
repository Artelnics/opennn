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

   // Constructors

   explicit RecurrentLayer();

   explicit RecurrentLayer(const Index&, const Index&);

   // Destructor
   
   virtual ~RecurrentLayer();

   // Get methods

   bool is_empty() const;

   Index get_inputs_number() const;
   Index get_neurons_number() const;

   const Tensor<type, 1>& get_hidden_states() const;

   // Parameters

   Index get_timesteps() const;

   Tensor<type, 2> get_biases() const;
   const Tensor<type, 2>& get_input_weights() const;
   const Tensor<type, 2>& get_recurrent_weights() const;

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

   void calculate_current_combinations(const Tensor<type, 1>& current_inputs,
                               Tensor<type, 1>& current_combinations);


   void calculate_current_activations(const Tensor<type, 1>& current_combinations, Tensor<type, 1>& current_activations) const;

   void calculate_current_activations_derivatives(const Tensor<type, 1>& current_combinations,
                                          Tensor<type, 1>& current_activations,
                                          Tensor<type, 1>& current_activations_derivatives) const;


   // neuron layer outputs

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);

   void forward_propagate(const Tensor<type, 2>& inputs, ForwardPropagation& forward_propagation);

   void calculate_hidden_delta(Layer* next_layer_pointer,
                               const Tensor<type, 2>&,
                               ForwardPropagation& forward_propagation,
                               const Tensor<type, 2>& next_layer_delta,
                               Tensor<type, 2>& hidden_delta) const;


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
   void from_XML(const tinyxml2::XMLDocument&);
   void write_XML(tinyxml2::XMLPrinter&) const;

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

   bool display = true;

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
