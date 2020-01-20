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
#include "functions.h"
#include "metrics.h"
#include "perceptron_layer.h"

namespace OpenNN
{

/// This class represents a layer of neurons.
/// Layers of neurons will be used to construct multilayer neurons.

class RecurrentLayer : public Layer
{

public:

    // Enumerations

    /// Enumeration of the available activation functions for the recurrent layer.

    enum ActivationFunction{Threshold, SymmetricThreshold, Logistic, HyperbolicTangent, Linear, RectifiedLinear, ExponentialLinear, ScaledExponentialLinear, SoftPlus, SoftSign, HardSigmoid};

   // Constructors

   explicit RecurrentLayer();

   explicit RecurrentLayer(const int&, const int&);

   RecurrentLayer(const RecurrentLayer&);

   // Destructor
   
   virtual ~RecurrentLayer();

   // Get methods

   bool is_empty() const;

   vector<int> get_input_variables_dimensions() const;

   int get_inputs_number() const;
   int get_neurons_number() const;

   Tensor<type, 1> get_hidden_states() const;

   // Parameters

   int get_timesteps()const;

   Tensor<type, 1> get_biases() const;
   Tensor<type, 2> get_input_weights() const;
   Tensor<type, 2> get_recurrent_weights() const;

   int get_biases_number() const;
   int get_input_weights_number() const;
   int get_recurrent_weights_number() const;

   int get_parameters_number() const;
   Tensor<type, 1> get_parameters() const;

   Tensor<type, 1> get_biases(const Tensor<type, 1>&) const;
   Tensor<type, 2> get_input_weights(const Tensor<type, 1>&) const;
   Tensor<type, 2> get_recurrent_weights(const Tensor<type, 1>&) const;

   Tensor<type, 2> get_input_weights_transpose() const;
   Tensor<type, 2> get_recurrent_weights_transpose() const;

   // Activation functions

   const RecurrentLayer::ActivationFunction& get_activation_function() const;

   string write_activation_function() const;

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const int&, const int&);
   void set(const RecurrentLayer&);

   void set_default();

   // Architecture

   void set_inputs_number(const int&);
   void set_neurons_number(const int&);
   void set_input_shape(const vector<int>&);

   // Parameters

   void set_timesteps(const int&);

   void set_biases(const Tensor<type, 1>&);

   void set_input_weights(const Tensor<type, 2>&);

   void set_recurrent_weights(const Tensor<type, 2>&);

   void set_parameters(const Tensor<type, 1>&);

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   // Display messages

   void set_display(const bool&);

   // Parameters initialization methods

   void initialize_hidden_states(const double&);

   void initialize_biases(const double&);

   void initialize_input_weights(const double&);
   void initialize_recurrent_weights(const double&);
   void initialize_input_weights_Glorot(const double&, const double&);

   void initialize_parameters(const double&);

   void randomize_parameters_uniform(const double&, const double&);

   void randomize_parameters_normal(const double&, const double&);

   // Parameters norm

   double calculate_parameters_norm() const;

   // neuron layer combinations

   Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&) const;

   Tensor<type, 2> calculate_combinations(const Tensor<type, 2>&);

   void calculate_combinations(const Tensor<type, 2>& inputs, Tensor<type, 2>& combinations)
   {
   }


   Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&) const;

   Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   // neuron layer activations

   Tensor<type, 1> calculate_activations(const Tensor<type, 1>&) const;

   Tensor<type, 2> calculate_activations(const Tensor<type, 2>&) const;

   Tensor<type, 2> calculate_activations_derivatives(const Tensor<type, 2>&) const;

   // neuron layer outputs

   void update_hidden_states(const Tensor<type, 1>&);

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);
   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 1>&);

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 1>&, const Tensor<type, 2>&, const Tensor<type, 2>&);

   Tensor<type, 2> calculate_output_delta(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   Tensor<type, 2> calculate_hidden_delta(Layer*, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   Layer::ForwardPropagation calculate_forward_propagation(const Tensor<type, 2>& inputs);

   void calculate_forward_propagation(const Tensor<type, 2>& inputs, Layer::ForwardPropagation& forward_propagation)
   {
/*
       calculate_combinations(inputs, forward_propagation.combinations);

       //calculate_activations(combinations, forward_propagation.activations);

       //calculate_activations_derivatives(combinations, layers.activations_derivatives);
*/
   }

   // Gradient

   Tensor<type, 1> calculate_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&);

   Tensor<type, 1> calculate_input_weights_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&);
   Tensor<type, 1> calculate_recurrent_weights_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&);
   Tensor<type, 1> calculate_biases_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&);

   // Expression methods

   string write_expression(const vector<string>&, const vector<string>&) const;
   string write_activation_function_expression() const;

   string object_to_string() const;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&) {};

   void write_XML(tinyxml2::XMLPrinter&) const {};

protected:

   int timesteps = 1;

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's trabsfer function to generate the neuron's output.

   Tensor<type, 1> biases;

   Tensor<type, 2> input_weights;

   /// This matrix containing conection strengths from a recurrent layer inputs to its neurons.

   Tensor<type, 2> recurrent_weights;

   /// Activation function variable.

   ActivationFunction activation_function = HyperbolicTangent;

   Tensor<type, 1> hidden_states;

   /// Display messages to screen.

   bool display;
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
