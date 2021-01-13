//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PROBABILISTICLAYER_H
#define PROBABILISTICLAYER_H

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

#include "opennn_strings.h"

namespace OpenNN
{

/// This class represents a layer of probabilistic neurons.

///
/// The neural network defined in OpenNN includes a probabilistic layer for those problems
/// when the outptus are to be interpreted as probabilities.
/// It does not has Synaptic weights or Biases

class ProbabilisticLayer : public Layer
{

public:

   // Constructors

   explicit ProbabilisticLayer();

   explicit ProbabilisticLayer(const Index&, const Index&);

   // Destructor

   virtual ~ProbabilisticLayer();

   // Enumerations

   /// Enumeration of available methods for interpreting variables as probabilities.

   enum ActivationFunction{Binary, Logistic, Competitive, Softmax};

   // Get methods

   

   Index get_inputs_number() const;
   Index get_neurons_number() const;

   Index get_biases_number() const;
   Index get_synaptic_weights_number() const;

   const type& get_decision_threshold() const;

   const ActivationFunction& get_activation_function() const;
   string write_activation_function() const;
   string write_activation_function_text() const;

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&, const Index&);
   void set(const ProbabilisticLayer&);

   void set_inputs_number(const Index&);
   void set_neurons_number(const Index&);

   void set_biases(const Tensor<type, 2>&);
   void set_synaptic_weights(const Tensor<type, 2>&);

   void set_parameters(const Tensor<type, 1>&, const Index& index=0);
   void set_synaptic_weights_glorot();
   void set_decision_threshold(const type&);

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   virtual void set_default();

   // Parameters

   const Tensor<type, 2>& get_biases() const;
   const Tensor<type, 2>& get_synaptic_weights() const;

   Tensor<type, 2> get_biases(Tensor<type, 1>&) const;
   Tensor<type, 2> get_synaptic_weights(Tensor<type, 1>&) const;

   Index get_parameters_number() const;
   Tensor<type, 1> get_parameters() const;

   // Display messages

   void set_display(const bool&);

   // Parameters initialization methods

   void set_biases_constant(const type&);
   void set_synaptic_weights_constant(const type&);
   void set_synaptic_weights_constant_Glorot();

   void set_parameters_constant(const type&);

   void set_parameters_random();

   void insert_parameters(const Tensor<type, 1>& parameters, const Index& );

   // Combinations

   void calculate_combinations(const Tensor<type, 2>& inputs,
                               const Tensor<type, 2>& biases,
                               const Tensor<type, 2>& synaptic_weights,
                               Tensor<type, 2>& combinations_2d) const;

   // Activations

   void calculate_activations(const Tensor<type, 2>& combinations_2d, Tensor<type, 2>& activations_2d) const;

   void calculate_activations_derivatives(const Tensor<type, 2>& combinations_2d,
                                          Tensor<type, 2>& activations,
                                          Tensor<type, 3>& activations_derivatives) const;

   // Outputs

   Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);


   void forward_propagate(const Tensor<type, 2>& inputs, ForwardPropagation& forward_propagation) const;

   void forward_propagate(const Tensor<type, 2>& inputs, Tensor<type, 1> potential_parameters, ForwardPropagation& forward_propagation) const;

   void calculate_output_delta(ForwardPropagation& forward_propagation,
                               const Tensor<type, 2>& output_gradient,
                               Tensor<type, 2>& output_delta) const;


   // Gradient methods

   void calculate_error_gradient(const Tensor<type, 2>& inputs,
                                 const Layer::ForwardPropagation&,
                                 Layer::BackPropagation& back_propagation) const;

   void insert_gradient(const BackPropagation& back_propagation, const Index& index, Tensor<type, 1>& gradient) const;

   // Expression methods

   string write_binary_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_logistic_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_competitive_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_softmax_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_no_probabilistic_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

//   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;
   string write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const;
   string write_combinations(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const;
   string write_activations(const Tensor<string, 1>& outputs_names) const;

   string write_expression_c() const;
   string write_combinations_c() const;
   string write_activations_c() const;

   string write_expression_python() const;
   string write_combinations_python() const;
   string write_activations_python() const;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

protected:

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's trabsfer function to generate the neuron's output.

   Tensor<type, 2> biases;

   /// This matrix containing conection strengths from a layer's inputs to its neurons.

   Tensor<type, 2> synaptic_weights;

   /// Activation function variable.

   ActivationFunction activation_function = Logistic;

   type decision_threshold;

   /// Display messages to screen.

   bool display = true;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/probabilistic_layer_cuda.h"
#endif

#ifdef OPENNN_MKL
    #include"../../opennn-mkl/opennn_mkl/probabilistic_layer_mkl.h"
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
