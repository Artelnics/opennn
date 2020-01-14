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


#ifdef __OPENNN_CUDA__
    #include "../../artelnics/opennn_cuda/opennn_cuda/kernels.h"
    #include "cuda_runtime_api.h"
#endif

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

   const Vector<double>& get_biases() const;
   const Matrix<double>& get_synaptic_weights() const;

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

   void calculate_combinations(const Tensor<double>& inputs, Tensor<double>& combinations) const
   {
       dot(inputs, synaptic_weights, combinations);

       combinations += biases;
   }

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
            case Linear: linear(combinations, activations); return;

            case Logistic: logistic(combinations, activations); return;

            case HyperbolicTangent: hyperbolic_tangent(combinations, activations); return;

            case Threshold: threshold(combinations, activations); return;

            case SymmetricThreshold: symmetric_threshold(combinations, activations); return;

            case RectifiedLinear: rectified_linear(combinations, activations); return;

            case ScaledExponentialLinear: scaled_exponential_linear(combinations, activations); return;

            case SoftPlus: soft_plus(combinations, activations); return;

            case SoftSign: soft_sign(combinations, activations); return;

            case HardSigmoid: hard_sigmoid(combinations, activations); return;

            case ExponentialLinear: exponential_linear(combinations, activations); return;
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
            case Linear: linear_derivatives(combinations, activations_derivatives); return;

            case Logistic: logistic_derivatives(combinations, activations_derivatives); return;

            case HyperbolicTangent: hyperbolic_tangent_derivatives(combinations, activations_derivatives); return;

            case Threshold: threshold_derivatives(combinations, activations_derivatives); return;

            case SymmetricThreshold: symmetric_threshold_derivatives(combinations, activations_derivatives); return;

            case RectifiedLinear: rectified_linear_derivatives(combinations, activations_derivatives); return;

            case ScaledExponentialLinear: scaled_exponential_linear_derivatives(combinations, activations_derivatives); return;

            case SoftPlus: soft_plus_derivatives(combinations, activations_derivatives); return;

            case SoftSign: soft_sign_derivatives(combinations, activations_derivatives); return;

            case HardSigmoid: hard_sigmoid_derivatives(combinations, activations_derivatives); return;

            case ExponentialLinear: exponential_linear_derivatives(combinations, activations_derivatives); return;
        }
   }

   // Perceptron layer outputs

   Tensor<double> calculate_outputs(const Tensor<double>&);
   Tensor<double> calculate_outputs(const Tensor<double>&, const Vector<double>&);
   Tensor<double> calculate_outputs(const Tensor<double>&, const Vector<double>&, const Matrix<double>&) const;

   ForwardPropagation calculate_forward_propagation(const Tensor<double>&);

   void calculate_forward_propagation(const Tensor<double>& inputs, ForwardPropagation& forward_propagation)
   {
       calculate_combinations(inputs, forward_propagation.combinations);

       calculate_activations(forward_propagation.combinations, forward_propagation.activations);

       calculate_activations_derivatives(forward_propagation.combinations, forward_propagation.activations_derivatives);
   }

   // Delta methods

   Tensor<double> calculate_output_delta(const Tensor<double>&, const Tensor<double>&) const;

   void calculate_output_delta(const Tensor<double>& activations_derivatives, const Tensor<double>& output_gradient, Tensor<double>& output_delta) const
   {
       output_delta = activations_derivatives;

       output_delta *= output_gradient;
   }


   Tensor<double> calculate_hidden_delta(Layer*, const Tensor<double>&, const Tensor<double>&, const Tensor<double>&) const;

   void calculate_hidden_delta(Layer* next_layer_pointer,
                               const Tensor<double>&,
                               const Tensor<double>& activations_derivatives,
                               const Tensor<double>& next_layer_delta,
                               Tensor<double>& hidden_delta) const
   {
       const Type layer_type = next_layer_pointer->get_type();

       if(layer_type == Perceptron)
       {
           const PerceptronLayer* perceptron_layer = dynamic_cast<PerceptronLayer*>(next_layer_pointer);

           //synaptic_weights_transpose = perceptron_layer->get_synaptic_weights_transpose();

           const Matrix<double>& synaptic_weights = perceptron_layer->get_synaptic_weights();

           dot_transpose(next_layer_delta, synaptic_weights, hidden_delta);

           hidden_delta *= activations_derivatives;
       }
       else if(layer_type == Probabilistic)
       {
           const ProbabilisticLayer* probabilistic_layer = dynamic_cast<ProbabilisticLayer*>(next_layer_pointer);
       }
       else
       {
           /// @todo Throw exception.
       }


   }


   // Gradient methods

   Vector<double> calculate_error_gradient(const Tensor<double>&, const Layer::ForwardPropagation&, const Tensor<double>&);

   void calculate_error_gradient(const Tensor<double>& inputs,
                                 const Layer::ForwardPropagation&,
                                 const Tensor<double>& deltas,
                                 Vector<double>& error_gradient)
   {
       //Tensor<double> reshaped_inputs = inputs.to_2d_tensor();

       //Tensor<double> reshaped_deltas = deltas.to_2d_tensor();

       const size_t inputs_number = get_inputs_number();
       const size_t neurons_number = get_neurons_number();

       const size_t parameters_number = get_parameters_number();

       const size_t synaptic_weights_number = neurons_number*inputs_number;

       Vector<double> layer_error_gradient(parameters_number, 0.0);

       // Synaptic weights

       error_gradient.embed(0, dot(inputs.to_matrix().calculate_transpose(), deltas).to_vector());

       // Biases

       error_gradient.embed(synaptic_weights_number, deltas.to_matrix().calculate_columns_sum());
   }


   // Expression methods

   string write_expression(const Vector<string>&, const Vector<string>&) const;
   string write_activation_function_expression() const;

   string object_to_string() const;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&);
   void write_XML(tinyxml2::XMLPrinter&) const;

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

#ifdef __OPENNN_CUDA__
    #include "../../artelnics/opennn_cuda/opennn_cuda/perceptron_layer_cuda.h"
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
