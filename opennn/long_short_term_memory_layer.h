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

#include "vector.h"
#include "layer.h"
#include "matrix.h"
#include "tensor.h"
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

   explicit LongShortTermMemoryLayer(const size_t&, const size_t&);

   LongShortTermMemoryLayer(const LongShortTermMemoryLayer&);

   // Destructor
   
   virtual ~LongShortTermMemoryLayer();

   // Get methods

   bool is_empty() const;

   size_t get_inputs_number() const;
   size_t get_neurons_number() const;

   // Parameters

   Vector<double> get_input_biases() const;
   Vector<double> get_forget_biases() const;
   Vector<double> get_state_biases() const;
   Vector<double> get_output_biases() const;

   Matrix<double> get_input_weights() const;
   Matrix<double> get_forget_weights() const;
   Matrix<double> get_state_weights() const;
   Matrix<double> get_output_weights() const;

   Matrix<double> get_input_recurrent_weights() const;
   Matrix<double> get_forget_recurrent_weights() const;
   Matrix<double> get_state_recurrent_weights() const;
   Matrix<double> get_output_recurrent_weights() const;

   Matrix<double> get_biases() const;
   Tensor<double> get_weights() const;
   Tensor<double> get_recurrent_weights() const;

   size_t get_timesteps() const;

   size_t get_parameters_number() const;
   Vector<double> get_parameters() const;

   // Activation functions

   const LongShortTermMemoryLayer::ActivationFunction& get_activation_function() const;
   const LongShortTermMemoryLayer::ActivationFunction& get_recurrent_activation_function() const;

   string write_activation_function() const;
   string write_recurrent_activation_function() const;
   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const size_t&, const size_t&);
   void set(const LongShortTermMemoryLayer&);

   void set_default();

   // Architecture

   void set_inputs_number(const size_t&);
   void set_neurons_number(const size_t&);
   void set_input_shape(const Vector<size_t>&);

   // Parameters

   void set_input_biases(const Vector<double>&);
   void set_forget_biases(const Vector<double>&);
   void set_state_biases(const Vector<double>&);
   void set_output_biases(const Vector<double>&);

   void set_input_weights(const Matrix<double>&);
   void set_forget_weights(const Matrix<double>&);
   void set_state_weights(const Matrix<double>&);
   void set_output_weights(const Matrix<double>&);

   void set_input_recurrent_weights(const Matrix<double>&);
   void set_forget_recurrent_weights(const Matrix<double>&);
   void set_state_recurrent_weights(const Matrix<double>&);
   void set_output_recurrent_weights(const Matrix<double>&);

   void set_parameters(const Vector<double>&);

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   void set_recurrent_activation_function(const ActivationFunction&);
   void set_recurrent_activation_function(const string&);

   void set_timesteps(const size_t&);

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

   void randomize_parameters_uniform();
   void randomize_parameters_uniform(const double&, const double&);

   void randomize_parameters_normal(const double& = 0.0, const double& = 1.0);

   // Parameters norm 

   double calculate_parameters_norm() const;

   // Long short term memory layer combinations

   Vector<double> calculate_forget_combinations(const Vector<double>&) const;
   Vector<double> calculate_input_combinations(const Vector<double>&) const;
   Vector<double> calculate_state_combinations(const Vector<double>&) const;
   Vector<double> calculate_output_combinations(const Vector<double>&) const;

   Tensor<double> calculate_activations_states(const Tensor<double>&);

   // Long short term memory layer activations

   Tensor<double> calculate_activations(const Tensor<double>&) const;
   Vector<double> calculate_activations(const Vector<double>&) const;
   Tensor<double> calculate_recurrent_activations(const Tensor<double>&) const;
   Vector<double> calculate_recurrent_activations(const Vector<double>&) const;

   // Long short term memory layer derivatives

   Tensor<double> calculate_activations_derivatives(const Tensor<double>&) const;
   Vector<double> calculate_activations_derivatives(const Vector<double>&) const;
   Vector<double> calculate_recurrent_activations_derivatives(const Vector<double>&) const;

   // Long short term memory layer outputs

   void update_cell_states(const Vector<double>&);
   void update_hidden_states(const Vector<double>&);

   Tensor<double> calculate_outputs(const Tensor<double>&);
   Tensor<double> calculate_outputs(const Tensor<double>&,const Vector<double>& );
   Tensor<double> calculate_outputs(const Tensor<double>&, const Matrix<double>&, const Tensor<double>&, const Tensor<double>&);

   FirstOrderActivations calculate_first_order_activations(const Tensor<double>&);

   Tensor<double> calculate_output_delta(const Tensor<double>&, const Tensor<double>&) const;

   Tensor<double> calculate_hidden_delta(Layer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;

   Vector<double> calculate_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&);

   Vector<double> calculate_forget_weights_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&, const Tensor<double>&);
   Vector<double> calculate_input_weights_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&, const Tensor<double>&);
   Vector<double> calculate_state_weights_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&, const Tensor<double>&);
   Vector<double> calculate_output_weights_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&, const Tensor<double>&);

   Vector<double> calculate_forget_recurrent_weights_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&, const Tensor<double>&);
   Vector<double> calculate_input_recurrent_weights_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&, const Tensor<double>&);
   Vector<double> calculate_state_recurrent_weights_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&, const Tensor<double>&);
   Vector<double> calculate_output_recurrent_weights_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&, const Tensor<double>&);

   Vector<double> calculate_forget_biases_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&, const Tensor<double>&);
   Vector<double> calculate_input_biases_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&, const Tensor<double>&);
   Vector<double> calculate_state_biases_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&, const Tensor<double>&);
   Vector<double> calculate_output_biases_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&, const Tensor<double>&);

   // Expression methods

   string write_expression(const Vector<string>&, const Vector<string>&) const;
   string write_recurrent_activation_function_expression() const;
   string write_activation_function_expression() const;

   string object_to_string() const;

protected:

   size_t timesteps = 10;

   Vector<double> input_biases;
   Vector<double> forget_biases;
   Vector<double> state_biases;
   Vector<double> output_biases;

   Matrix<double> input_weights;
   Matrix<double> forget_weights;
   Matrix<double> state_weights;
   Matrix<double> output_weights;

   Matrix<double> forget_recurrent_weights;
   Matrix<double> input_recurrent_weights;
   Matrix<double> state_recurrent_weights;
   Matrix<double> output_recurrent_weights;

   /// Activation function variable.

   ActivationFunction activation_function = HyperbolicTangent;
   ActivationFunction recurrent_activation_function = HardSigmoid;

   size_t batch;
   size_t variables;

   Vector<double> hidden_states;
   Vector<double> cell_states;

   /// Display messages to screen. 

   bool display;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
