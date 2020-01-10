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

#include "vector.h"
#include "matrix.h"
#include "functions.h"
#include "layer.h"
#include "metrics.h"
#include "tinyxml2.h"

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

   explicit ProbabilisticLayer(const size_t&, const size_t&);

   ProbabilisticLayer(const ProbabilisticLayer&);

   // Destructor

   virtual ~ProbabilisticLayer();

   // Enumerations

   /// Enumeration of available methods for interpreting variables as probabilities.

   enum ActivationFunction{Binary, Logistic, Competitive, Softmax};

   // Get methods

   Vector<size_t> get_input_variables_dimensions() const;

   size_t get_inputs_number() const;
   size_t get_neurons_number() const;

   const double& get_decision_threshold() const;

   const ActivationFunction& get_activation_function() const;
   string write_activation_function() const;
   string write_activation_function_text() const;

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const size_t&, const size_t&);
   void set(const ProbabilisticLayer&);

   void set_inputs_number(const size_t&);
   void set_neurons_number(const size_t&);

   void set_biases(const Vector<double>&);
   void set_synaptic_weights(const Matrix<double>&);

   void set_parameters(const Vector<double>&);

   void set_decision_threshold(const double&);

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   virtual void set_default();

   // Parameters

   Vector<double> get_biases() const;
   Matrix<double> get_synaptic_weights() const;

   Vector<double> get_biases(const Vector<double>&) const;
   Matrix<double> get_synaptic_weights(const Vector<double>&) const;

   Matrix<double> get_synaptic_weights_transpose() const;

   size_t get_parameters_number() const;
   Vector<double> get_parameters() const;

//   void randomize_parameters_normal(const double& = 0.0, const double& = 1.0);

   // Display messages

   void set_display(const bool&);

   // Pruning and growing

   void prune_neuron(const size_t&);

   // Parameters initialization methods

   void initialize_biases(const double&);
   void initialize_synaptic_weights(const double&);
   void initialize_synaptic_weights_Glorot(const double&,const double&);

   void initialize_parameters(const double&);

   void randomize_parameters_uniform();
   void randomize_parameters_uniform(const double&, const double&);

   void randomize_parameters_normal();
   void randomize_parameters_normal(const double&, const double&);

   // Combinations

   Tensor<double> calculate_combinations(const Tensor<double>&) const;

   void calculate_combinations(const Tensor<double>& inputs, Tensor<double>& combinations) const
   {
       linear_combinations(inputs, synaptic_weights, biases, combinations);
   }

   // Outputs

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


   // Deltas

   Tensor<double> calculate_output_delta(const Tensor<double>&, const Tensor<double>&) const;

   // Gradient methods

   Vector<double> calculate_error_gradient(const Tensor<double>&, const Layer::ForwardPropagation&, const Tensor<double>&);

   // Activations

   Tensor<double> calculate_activations(const Tensor<double>&) const;

   void calculate_activations(const Tensor<double>& combinations, Tensor<double>& activations) const
   {
        #ifdef __OPENNN_DEBUG__

        const size_t dimensions_number = combinations.get_dimensions_number();

        if(dimensions_number != 2)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                  << "void calculate_activations(const Tensor<double>&, Tensor<double>&) const method.\n"
                  << "Dimensions of combinations (" << dimensions_number << ") must be 2.\n";

           throw logic_error(buffer.str());
        }

        const size_t neurons_number = get_neurons_number();

        const size_t combinations_columns_number = combinations.get_dimension(1);

        if(combinations_columns_number != neurons_number)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                  << "void calculate_activations(const Tensor<double>&, Tensor<double>&) const method.\n"
                  << "Number of combinations columns (" << combinations_columns_number << ") must be equal to number of neurons (" << neurons_number << ").\n";

           throw logic_error(buffer.str());
        }

        #endif

        switch(activation_function)
        {
            case Binary: binary(combinations, activations); return;

            case Logistic: logistic(combinations, activations); return;

            case Competitive: competitive(combinations, activations); return;

            case Softmax: softmax(combinations, activations); return;
        }

        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void calculate_activations(const Tensor<double>&, Tensor<double>&) const method.\n"
               << "Unknown probabilistic method.\n";

        throw logic_error(buffer.str());
   }

   Tensor<double> calculate_activations_derivatives(const Tensor<double>&) const;

   void calculate_activations_derivatives(const Tensor<double>& combinations, Tensor<double>& activations_derivatives) const
   {
        #ifdef __OPENNN_DEBUG__

        const size_t dimensions_number = combinations.get_dimensions_number();

        if(dimensions_number != 2)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                  << "void calculate_activations_derivatives(const Tensor<double>&, Tensor<double>&) const method.\n"
                  << "Dimensions of combinations (" << dimensions_number << ") must be 2.\n";

           throw logic_error(buffer.str());
        }

        const size_t neurons_number = get_neurons_number();

        const size_t combinations_columns_number = combinations.get_dimension(1);

        if(combinations_columns_number != neurons_number)
        {
           ostringstream buffer;

           buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                  << "Tensor<double> calculate_activations_derivatives(const Tensor<double>&) const method.\n"
                  << "Number of combinations columns (" << combinations_columns_number << ") must be equal to number of neurons (" << neurons_number << ").\n";

           throw logic_error(buffer.str());
        }

        #endif

        switch(activation_function)
        {
            case Binary:
            {
                 ostringstream buffer;

                 buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                        << "Binary derivative doesn't exist.\n";

                 throw logic_error(buffer.str());
            }

            case Logistic:
            {
                    logistic_derivatives(combinations, activations_derivatives); return;
            }

            case Competitive:
            {
                 ostringstream buffer;

                 buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                        << "Competitive derivative doesn't exist.\n";

                 throw logic_error(buffer.str());
            }

            case Softmax:
            {
                softmax_derivatives(combinations, activations_derivatives); return;
            }
        }

        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void calculate_activations_derivatives(const Tensor<double>&, Tensor<double>&) const method.\n"
               << "Unknown probabilistic method.\n";

        throw logic_error(buffer.str());

   }

   // Expression methods

   string write_binary_expression(const Vector<string>&, const Vector<string>&) const;
   string write_probability_expression(const Vector<string>&, const Vector<string>&) const;
   string write_competitive_expression(const Vector<string>&, const Vector<string>&) const;
   string write_softmax_expression(const Vector<string>&, const Vector<string>&) const;
   string write_no_probabilistic_expression(const Vector<string>&, const Vector<string>&) const;

   string write_expression(const Vector<string>&, const Vector<string>&) const;

   // Serialization methods

   string object_to_string() const;

   virtual tinyxml2::XMLDocument* to_XML() const;

   virtual void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;
   
protected:

   // MEMBERS

   /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
   /// and passed through the neuron's trabsfer function to generate the neuron's output.

   Vector<double> biases;

   /// This matrix containing conection strengths from a layer's inputs to its neurons.

   Matrix<double> synaptic_weights;

   /// Activation function variable.

   ActivationFunction activation_function = Logistic;

   double decision_threshold;

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
