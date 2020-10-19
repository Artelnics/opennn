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

#include "config.h"
#include "layer.h"
#include "probabilistic_layer.h"
#include "opennn_strings.h"

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

    /// Enumeration of available activation functions for the perceptron neuron model.

    enum ActivationFunction{Threshold, SymmetricThreshold, Logistic, HyperbolicTangent, Linear, RectifiedLinear,
                            ExponentialLinear, ScaledExponentialLinear, SoftPlus, SoftSign, HardSigmoid};

    enum PerceptronLayerType{HiddenLayer, OutputLayer};

   // Constructors

   explicit PerceptronLayer();

   explicit PerceptronLayer(const Index&, const Index&, const Index& = 0 , const ActivationFunction& = PerceptronLayer::HyperbolicTangent);

   // Destructor
   
   virtual ~PerceptronLayer();

   // Get methods

   bool is_empty() const;

   Index get_inputs_number() const;
   Index get_neurons_number() const;

   // Parameters

   const Tensor<type, 2>& get_biases() const;
   const Tensor<type, 2>& get_synaptic_weights() const;

   Tensor<type, 2> get_biases(const Tensor<type, 1>&) const;
   Tensor<type, 2> get_synaptic_weights(const Tensor<type, 1>&) const;

   Index get_biases_number() const;
   Index get_synaptic_weights_number() const;
   Index get_parameters_number() const;
   Tensor<type, 1> get_parameters() const;

   // Activation functions

   const PerceptronLayer::ActivationFunction& get_activation_function() const;

   string write_activation_function() const;

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&, const Index&, const PerceptronLayer::ActivationFunction& = PerceptronLayer::HyperbolicTangent);

   void set_default();
   void set_layer_name(const string&);

   // Architecture

   void set_inputs_number(const Index&);
   void set_neurons_number(const Index&);

   // Parameters

   void set_biases(const Tensor<type, 2>&);
   void set_synaptic_weights(const Tensor<type, 2>&);

   void set_parameters(const Tensor<type, 1>&, const Index& index=0);

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   // Display messages

   void set_display(const bool&);

   // Parameters initialization methods
   void set_biases_constant(const type&);
   void set_synaptic_weights_constant(const type&);
   void set_synaptic_weights_glorot();

   void set_parameters_constant(const type&);

   void set_parameters_random();

   // Perceptron layer combinations_2d

   void calculate_combinations(const Tensor<type, 2>& inputs,
                               const Tensor<type, 2>& biases,
                               const Tensor<type, 2>& synaptic_weights,
                               Tensor<type, 2>& combinations_2d) const;

   // Perceptron layer activations_2d

   void calculate_activations(const Tensor<type, 2>& combinations_2d, Tensor<type, 2>& activations_2d) const;

   void calculate_activations_derivatives(const Tensor<type, 2>& combinations_2d,
                                          Tensor<type, 2>& activations,
                                          Tensor<type, 2>& activations_derivatives) const;

   // Perceptron layer outputs

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);


   void forward_propagate(const Tensor<type, 2>& inputs,
                                      ForwardPropagation& forward_propagation) const;


   void forward_propagate(const Tensor<type, 2>& inputs,
                                      Tensor<type, 1> potential_parameters,
                                      ForwardPropagation& forward_propagation) const;

   // Delta methods

   void calculate_output_delta(ForwardPropagation& forward_propagation,
                                  const Tensor<type, 2>& output_gradient,
                                  Tensor<type, 2>& output_delta) const;

   void calculate_hidden_delta(Layer* next_layer_pointer,
                               const Tensor<type, 2>&,
                               ForwardPropagation& forward_propagation,
                               const Tensor<type, 2>& next_layer_delta,
                               Tensor<type, 2>& hidden_delta) const;

   void calculate_hidden_delta_perceptron(Layer* next_layer_pointer,
                                          const Tensor<type, 2>& activations_derivatives,
                                          const Tensor<type, 2>& next_layer_delta,
                                          Tensor<type, 2>& hidden_delta) const;

   void calculate_hidden_delta_probabilistic(Layer* next_layer_pointer,
                                             const Tensor<type, 2>& activations_derivatives,
                                             const Tensor<type, 2>& next_layer_delta,
                                             Tensor<type, 2>& hidden_delta) const;

   // Gradient methods

   void calculate_error_gradient(const Tensor<type, 2>& inputs,
                                 const Layer::ForwardPropagation&,
                                 Layer::BackPropagation& back_propagation) const;

   void insert_gradient(const BackPropagation& back_propagation, const Index& index, Tensor<type, 1>& gradient) const;

   // Expression methods   

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_hidden_layer_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_output_layer_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_activation_function_expression() const;

   string write_expression_c() const;
   string write_combinations_c() const;
   string write_activations_c() const;

   string write_combinations_python() const;
   string write_activations_python() const;
   string write_expression_python() const;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&);
   void write_XML(tinyxml2::XMLPrinter&) const;

protected:

   // MEMBERS

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's transfer function to generate the neuron's output.

   Tensor<type, 2> biases;

   /// This matrix containing conection strengths from a layer's inputs to its neurons.

   Tensor<type, 2> synaptic_weights;

   /// Activation function variable.

   ActivationFunction activation_function;

   /// Layer type variable.

   PerceptronLayerType perceptron_layer_type = OutputLayer;

   /// Display messages to screen. 

   bool display = true;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/perceptron_layer_cuda.h"
#endif

#ifdef OPENNN_MKL
    #include"../../opennn-mkl/opennn_mkl/perceptron_layer_mkl.h"
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
