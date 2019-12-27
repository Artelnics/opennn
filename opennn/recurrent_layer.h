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

class RecurrentLayer : public Layer
{

public:

    // Enumerations

    /// Enumeration of the available activation functions for the recurrent layer.

    enum ActivationFunction{Threshold, SymmetricThreshold, Logistic, HyperbolicTangent, Linear, RectifiedLinear, ExponentialLinear, ScaledExponentialLinear, SoftPlus, SoftSign, HardSigmoid};

   // Constructors

   explicit RecurrentLayer();

   explicit RecurrentLayer(const size_t&, const size_t&);

   RecurrentLayer(const RecurrentLayer&);

   // Destructor
   
   virtual ~RecurrentLayer();

   // Get methods

   bool is_empty() const;

   Vector<size_t> get_input_variables_dimensions() const;

   size_t get_inputs_number() const;
   size_t get_neurons_number() const;

   Vector<double> get_hidden_states() const;

   // Parameters

   size_t get_timesteps()const;

   Vector<double> get_biases() const;
   Matrix<double> get_input_weights() const;
   Matrix<double> get_recurrent_weights() const;

   size_t get_biases_number() const;
   size_t get_input_weights_number() const;
   size_t get_recurrent_weights_number() const;

   size_t get_parameters_number() const;
   Vector<double> get_parameters() const;

   Vector<double> get_biases(const Vector<double>&) const;
   Matrix<double> get_input_weights(const Vector<double>&) const;
   Matrix<double> get_recurrent_weights(const Vector<double>&) const;

   Matrix<double> get_input_weights_transpose() const;
   Matrix<double> get_recurrent_weights_transpose() const;

   // Activation functions

   const RecurrentLayer::ActivationFunction& get_activation_function() const;

   string write_activation_function() const;

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const size_t&, const size_t&);
   void set(const RecurrentLayer&);

   void set_default();

   // Architecture

   void set_inputs_number(const size_t&);
   void set_neurons_number(const size_t&);
   void set_input_shape(const Vector<size_t>&);

   // Parameters

   void set_timesteps(const size_t&);

   void set_biases(const Vector<double>&);

   void set_input_weights(const Matrix<double>&);

   void set_recurrent_weights(const Matrix<double>&);

   void set_parameters(const Vector<double>&);

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

   Vector<double> calculate_combinations(const Vector<double>&) const;

   Tensor<double> calculate_combinations(const Tensor<double>&);

   Vector<double> calculate_combinations(const Vector<double>&, const Vector<double>&) const;

   Vector<double> calculate_combinations(const Vector<double>&, const Vector<double>&, const Matrix<double>&, const Matrix<double>&) const;

   // neuron layer activations

   Vector<double> calculate_activations(const Vector<double>&) const;

   Tensor<double> calculate_activations(const Tensor<double>&) const;

   Tensor<double> calculate_activations_derivatives(const Tensor<double>&) const;

   // neuron layer outputs

   void update_hidden_states(const Vector<double>&);

   Tensor<double> calculate_outputs(const Tensor<double>&);
   Tensor<double> calculate_outputs(const Tensor<double>&, const Vector<double>&);

   Tensor<double> calculate_outputs(const Tensor<double>&, const Vector<double>&, const Matrix<double>&, const Matrix<double>&);

   Tensor<double> calculate_output_delta(const Tensor<double>&, const Tensor<double>&) const;

   Tensor<double> calculate_hidden_delta(Layer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;

   Layer::FirstOrderActivations calculate_first_order_activations(const Tensor<double>& inputs);

   // Gradient

   Vector<double> calculate_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&);

   Vector<double> calculate_input_weights_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&);
   Vector<double> calculate_recurrent_weights_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&);
   Vector<double> calculate_biases_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&);

   // Expression methods

   string write_expression(const Vector<string>&, const Vector<string>&) const;
   string write_activation_function_expression() const;

   string object_to_string() const;

protected:

   size_t timesteps = 1;

   Vector<double> biases;

   Matrix<double> input_weights;

   Matrix<double> recurrent_weights;

   /// Activation function variable.

   ActivationFunction activation_function = HyperbolicTangent;

   Vector<double> hidden_states;

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
