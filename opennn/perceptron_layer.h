//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   C L A S S   H E A D E R             
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PERCEPTRONLAYER_H
#define PERCEPTRONLAYER_H

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
#include "probabilistic_layer.h"

namespace OpenNN
{

/// This class represents a layer of perceptrons.

/// PerceptronLayer is a single-layer network with a hard-limit trabsfer function.
/// This network is often trained with the perceptron learning rule.
///
/// Layers of perceptrons will be used to construct multilayer perceptrons, such as an approximation problems .

class PerceptronLayer : public Layer
{

public:

    // Enumerations

    /// Enumeration of available activation functions for the perceptron neuron model.

    enum ActivationFunction{Threshold, SymmetricThreshold, Logistic, HyperbolicTangent, Linear, RectifiedLinear, ExponentialLinear, ScaledExponentialLinear, SoftPlus, SoftSign, HardSigmoid};

   // Constructors

   explicit PerceptronLayer();

   explicit PerceptronLayer(const size_t&, const size_t&, const ActivationFunction& = PerceptronLayer::HyperbolicTangent);

   PerceptronLayer(const PerceptronLayer&);

   // Destructor
   
   virtual ~PerceptronLayer();

   // Get methods

   bool is_empty() const;

   Vector<size_t> get_input_variables_dimensions() const;

   size_t get_inputs_number() const;
   size_t get_neurons_number() const;

   // Parameters

   Vector<double> get_biases() const;
   Matrix<double> get_synaptic_weights() const;

   Vector<double> get_biases(const Vector<double>&) const;
   Matrix<double> get_synaptic_weights(const Vector<double>&) const;

   Matrix<double> get_synaptic_weights_transpose() const;

   size_t get_parameters_number() const;
   Vector<double> get_parameters() const;

   // Activation functions

   const PerceptronLayer::ActivationFunction& get_activation_function() const;

   string write_activation_function() const;

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const size_t&, const size_t&, const PerceptronLayer::ActivationFunction& = PerceptronLayer::HyperbolicTangent);
   void set(const PerceptronLayer&);

   void set_default();

   // Architecture

   void set_inputs_number(const size_t&);
   void set_neurons_number(const size_t&);

   // Parameters

   void set_biases(const Vector<double>&);
   void set_synaptic_weights(const Matrix<double>&);

   void set_parameters(const Vector<double>&);

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   // Display messages

   void set_display(const bool&);

   // Growing and pruning

   void grow_input();
   void grow_perceptron();
   void grow_perceptrons(const size_t&);

   void prune_input(const size_t&);
   void prune_neuron(const size_t&);

   // Parameters initialization methods

   void initialize_biases(const double&); 
   void initialize_synaptic_weights(const double&);
   void initialize_synaptic_weights_glorot_uniform();

   void initialize_parameters(const double&);

   void randomize_parameters_uniform();
   void randomize_parameters_uniform(const double&, const double&);

   void randomize_parameters_normal();
   void randomize_parameters_normal(const double& = 0.0, const double& = 1.0);

   // Parameters norm 

   double calculate_parameters_norm() const;

   // Perceptron layer combinations

   Tensor<double> calculate_combinations(const Tensor<double>&) const;

   Tensor<double> calculate_combinations(const Tensor<double>&, const Vector<double>&) const;

   Tensor<double> calculate_combinations(const Tensor<double>&, const Vector<double>&, const Matrix<double>&) const;

   // Perceptron layer activations

   Tensor<double> calculate_activations(const Tensor<double>&) const;
   Tensor<double> calculate_activations_derivatives(const Tensor<double>&) const;

   void calculate_activations(const Tensor<double>& combinations, Tensor<double>& activations) const
   {
        #ifdef __OPENNN_DEBUG__

        const size_t neurons_number = get_neurons_number();

        const size_t combinations_columns_number = combinations.get_dimension(1);

        if(combinations_columns_number != neurons_number)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: PerceptronLayer class.\n"
                  << "void calculate_activations(const Tensor<double>&, Tensor<double>&) const method.\n"
                  << "Number of combinations columns (" << combinations_columns_number << ") must be equal to number of neurons (" << neurons_number << ").\n";

           throw logic_error(buffer.str());
        }

        #endif

        switch(activation_function)
        {
            case Linear: linear(combinations, activations); break;

            case Logistic: logistic(combinations, activations); break;

            case HyperbolicTangent: hyperbolic_tangent(combinations, activations); break;

            case Threshold: threshold(combinations, activations); break;

            case SymmetricThreshold: symmetric_threshold(combinations, activations); break;

            case RectifiedLinear: rectified_linear(combinations, activations); break;

            case ScaledExponentialLinear: scaled_exponential_linear(combinations, activations); break;

            case SoftPlus: soft_plus(combinations, activations); break;

            case SoftSign: soft_sign(combinations, activations); break;

            case HardSigmoid: hard_sigmoid(combinations, activations); break;

            case ExponentialLinear: exponential_linear(combinations, activations); break;
        }
   }


   void calculate_activations_derivatives(const Tensor<double>& combinations, Tensor<double>& activations_derivatives) const
   {
        #ifdef __OPENNN_DEBUG__

        const size_t neurons_number = get_neurons_number();

        const size_t combinations_columns_number = combinations.get_dimension(1);

        if(combinations_columns_number != neurons_number)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: PerceptronLayer class.\n"
                  << "void calculate_activations_derivatives(const Tensor<double>&, Tensor<double>&) const method.\n"
                  << "Number of combinations columns (" << combinations_columns_number << ") must be equal to number of neurons (" << neurons_number << ").\n";

           throw logic_error(buffer.str());
        }

        #endif

        switch(activation_function)
        {
            case Linear: linear_derivatives(combinations, activations_derivatives); break;

            case Logistic: logistic_derivatives(combinations, activations_derivatives); break;

            case HyperbolicTangent: hyperbolic_tangent_derivatives(combinations, activations_derivatives); break;

            case Threshold: threshold_derivatives(combinations, activations_derivatives); break;

            case SymmetricThreshold: symmetric_threshold_derivatives(combinations, activations_derivatives); break;

            case RectifiedLinear: rectified_linear_derivatives(combinations, activations_derivatives); break;

            case ScaledExponentialLinear: scaled_exponential_linear_derivatives(combinations, activations_derivatives); break;

            case SoftPlus: soft_plus_derivatives(combinations, activations_derivatives); break;

            case SoftSign: soft_sign_derivatives(combinations, activations_derivatives); break;

            case HardSigmoid: hard_sigmoid_derivatives(combinations, activations_derivatives); break;

            case ExponentialLinear: exponential_linear_derivatives(combinations, activations_derivatives); break;
        }
   }

   // Perceptron layer outputs

   Tensor<double> calculate_outputs(const Tensor<double>&);
   Tensor<double> calculate_outputs(const Tensor<double>&, const Vector<double>&);
   Tensor<double> calculate_outputs(const Tensor<double>&, const Vector<double>&, const Matrix<double>&) const;

   FirstOrderActivations calculate_first_order_activations(const Tensor<double>&);

   void calculate_first_order_activations(const Tensor<double>& inputs, FirstOrderActivations& first_order_activations)
   {
       Tensor<double> combinations;

       if(inputs.get_dimensions_number() != 2)
       {
           combinations = calculate_combinations(inputs.to_2d_tensor());
       }
       else
       {
           combinations = calculate_combinations(inputs);
       }

       calculate_activations(combinations, first_order_activations.activations);

       calculate_activations_derivatives(combinations, first_order_activations.activations_derivatives);
   }


   // Delta methods

   Tensor<double> calculate_output_delta(const Tensor<double>&, const Tensor<double>&) const;
   Tensor<double> calculate_hidden_delta(Layer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;

   // Gradient methods

   Vector<double> calculate_error_gradient(const Tensor<double>&, const Layer::FirstOrderActivations&, const Tensor<double>&);

   // Expression methods

   string write_expression(const Vector<string>&, const Vector<string>&) const;
   string write_activation_function_expression() const;

   string object_to_string() const;

protected:

   // MEMBERS

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's trabsfer function to generate the neuron's output.

   Vector<double> biases;

   /// This matrix containing conection strengths from a layer's inputs to its neurons.

   Matrix<double> synaptic_weights;

   /// Activation function variable.

   ActivationFunction activation_function;

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
