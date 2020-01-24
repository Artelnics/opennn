//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LONGSHORTTERMMEMORYLAYER_H
#define LONGSHORTTERMMEMORYLAYER_H

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

class LongShortTermMemoryLayer : public Layer
{

public:

    // Enumerations

    /// Enumeration of available activation functions for the long-short term memory layer.

    enum ActivationFunction{Threshold, SymmetricThreshold, Logistic, HyperbolicTangent, Linear, RectifiedLinear, ExponentialLinear, ScaledExponentialLinear, SoftPlus, SoftSign, HardSigmoid};

   // Constructors

   explicit LongShortTermMemoryLayer();

   explicit LongShortTermMemoryLayer(const int&, const int&);

   LongShortTermMemoryLayer(const LongShortTermMemoryLayer&);

   // Destructor
   
   virtual ~LongShortTermMemoryLayer();

   // Get methods

   bool is_empty() const;

   int get_inputs_number() const;
   int get_neurons_number() const;

   // Parameters

   Tensor<type, 1> get_input_biases() const;
   Tensor<type, 1> get_forget_biases() const;
   Tensor<type, 1> get_state_biases() const;
   Tensor<type, 1> get_output_biases() const;

   Tensor<type, 2> get_input_weights() const;
   Tensor<type, 2> get_forget_weights() const;
   Tensor<type, 2> get_state_weights() const;
   Tensor<type, 2> get_output_weights() const;

   Tensor<type, 2> get_input_recurrent_weights() const;
   Tensor<type, 2> get_forget_recurrent_weights() const;
   Tensor<type, 2> get_state_recurrent_weights() const;
   Tensor<type, 2> get_output_recurrent_weights() const;

   Tensor<type, 2> get_biases() const;
   Tensor<type, 2> get_weights() const;
   Tensor<type, 2> get_recurrent_weights() const;

   int get_timesteps() const;

   int get_parameters_number() const;
   Tensor<type, 1> get_parameters() const;

   // Activation functions

   const LongShortTermMemoryLayer::ActivationFunction& get_activation_function() const;
   const LongShortTermMemoryLayer::ActivationFunction& get_recurrent_activation_function() const;

   string write_activation_function() const;
   string write_recurrent_activation_function() const;
   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const int&, const int&);
   void set(const LongShortTermMemoryLayer&);

   void set_default();

   // Architecture

   void set_inputs_number(const int&);
   void set_neurons_number(const int&);
   void set_input_shape(const Tensor<int, 1>&);

   // Parameters

   void set_input_biases(const Tensor<type, 1>&);
   void set_forget_biases(const Tensor<type, 1>&);
   void set_state_biases(const Tensor<type, 1>&);
   void set_output_biases(const Tensor<type, 1>&);

   void set_input_weights(const Tensor<type, 2>&);
   void set_forget_weights(const Tensor<type, 2>&);
   void set_state_weights(const Tensor<type, 2>&);
   void set_output_weights(const Tensor<type, 2>&);

   void set_input_recurrent_weights(const Tensor<type, 2>&);
   void set_forget_recurrent_weights(const Tensor<type, 2>&);
   void set_state_recurrent_weights(const Tensor<type, 2>&);
   void set_output_recurrent_weights(const Tensor<type, 2>&);

   void set_parameters(const Tensor<type, 1>&);

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   void set_recurrent_activation_function(const ActivationFunction&);
   void set_recurrent_activation_function(const string&);

   void set_timesteps(const int&);

   // Display messages

   void set_display(const bool&);

   // Parameters initialization methods

   void initialize_biases(const double&); 

   void initialize_forget_biases(const double&);
   void initialize_input_biases(const double&);
   void initialize_state_biases(const double&);
   void initialize_output_biases(const double&);

   void initialize_weights(const double&);

   void initialize_forget_weights(const double&);
   void initialize_input_weights(const double&);
   void initialize_state_weights(const double&);
   void initialize_output_weights(const double&);

   void initialize_recurrent_weights(const double&);

   void initialize_forget_recurrent_weights(const double&);
   void initialize_input_recurrent_weights(const double&);
   void initialize_state_recurrent_weights(const double&);
   void initialize_output_recurrent_weights(const double&);

   void initialize_hidden_states(const double&);
   void initialize_cell_states(const double&);

   void initialize_weights_Glorot(const double&, const double&);

   void initialize_parameters(const double&);

   void set_parameters_random();

   // Parameters norm 

   double calculate_parameters_norm() const;

   // Long short term memory layer combinations

   Tensor<type, 1> calculate_forget_combinations(const Tensor<type, 1>&) const;
   Tensor<type, 1> calculate_input_combinations(const Tensor<type, 1>&) const;
   Tensor<type, 1> calculate_state_combinations(const Tensor<type, 1>&) const;
   Tensor<type, 1> calculate_output_combinations(const Tensor<type, 1>&) const;

   Tensor<type, 2> calculate_activations_states(const Tensor<type, 2>&);

   // Long short term memory layer activations

   Tensor<type, 2> calculate_activations(const Tensor<type, 2>&) const;
   Tensor<type, 1> calculate_activations(const Tensor<type, 1>&) const;
   Tensor<type, 2> calculate_recurrent_activations(const Tensor<type, 2>&) const;
   Tensor<type, 1> calculate_recurrent_activations(const Tensor<type, 1>&) const;

   // Long short term memory layer derivatives

   Tensor<type, 2> calculate_activations_derivatives(const Tensor<type, 2>&) const;
   Tensor<type, 1> calculate_activations_derivatives(const Tensor<type, 1>&) const;
   Tensor<type, 1> calculate_recurrent_activations_derivatives(const Tensor<type, 1>&) const;

   // Long short term memory layer outputs

   void update_cell_states(const Tensor<type, 1>&);
   void update_hidden_states(const Tensor<type, 1>&);

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);
   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&,const Tensor<type, 1>&);
   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&);

   ForwardPropagation calculate_forward_propagation(const Tensor<type, 2>&);

   Tensor<type, 2> calculate_output_delta(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   Tensor<type, 2> calculate_hidden_delta(Layer*, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   Tensor<type, 1> calculate_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&);

   Tensor<type, 1> calculate_forget_weights_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&, const Tensor<type, 2>&);
   Tensor<type, 1> calculate_input_weights_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&, const Tensor<type, 2>&);
   Tensor<type, 1> calculate_state_weights_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&, const Tensor<type, 2>&);
   Tensor<type, 1> calculate_output_weights_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&, const Tensor<type, 2>&);

   Tensor<type, 1> calculate_forget_recurrent_weights_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&, const Tensor<type, 2>&);
   Tensor<type, 1> calculate_input_recurrent_weights_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&, const Tensor<type, 2>&);
   Tensor<type, 1> calculate_state_recurrent_weights_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&, const Tensor<type, 2>&);
   Tensor<type, 1> calculate_output_recurrent_weights_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&, const Tensor<type, 2>&);

   Tensor<type, 1> calculate_forget_biases_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&, const Tensor<type, 2>&);
   Tensor<type, 1> calculate_input_biases_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&, const Tensor<type, 2>&);
   Tensor<type, 1> calculate_state_biases_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&, const Tensor<type, 2>&);
   Tensor<type, 1> calculate_output_biases_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&, const Tensor<type, 2>&);

   // Expression methods

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_recurrent_activation_function_expression() const;
   string write_activation_function_expression() const;

   string object_to_string() const;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&) {};

   void write_XML(tinyxml2::XMLPrinter&) const {};

protected:

   int timesteps = 10;

   Tensor<type, 1> input_biases;
   Tensor<type, 1> forget_biases;
   Tensor<type, 1> state_biases;
   Tensor<type, 1> output_biases;

   Tensor<type, 2> input_weights;
   Tensor<type, 2> forget_weights;
   Tensor<type, 2> state_weights;
   Tensor<type, 2> output_weights;

   Tensor<type, 2> forget_recurrent_weights;
   Tensor<type, 2> input_recurrent_weights;
   Tensor<type, 2> state_recurrent_weights;
   Tensor<type, 2> output_recurrent_weights;

   /// Activation function variable.

   ActivationFunction activation_function = HyperbolicTangent;
   ActivationFunction recurrent_activation_function = HardSigmoid;

   int batch;
   int variables;

   Tensor<type, 1> hidden_states;
   Tensor<type, 1> cell_states;

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
