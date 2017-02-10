/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural MultilayerPerceptrons Library                                                          */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M U L T I L A Y E R   P E R C E P T R O N   C L A S S   H E A D E R                                        */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef _MULTILAYERPERCEPTRON_H__
#define _MULTILAYERPERCEPTRON_H__

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "perceptron.h"
#include "perceptron_layer.h"
#include "inputs.h"
#include "outputs.h"

#include "vector.h"
#include "matrix.h"
#include "numerical_differentiation.h"

// TinyXml includes

#include "../tinyxml2/tinyxml2.h"


namespace OpenNN
{

/// This class represents the concept of multilayer perceptron.
/// A multilayer perceptron is a feed-forward network of layers of perceptrons. 
/// This is the most important class included in the definition of a neural network. 

class MultilayerPerceptron
{

public:

   // DEFAULT CONSTRUCTOR

   explicit MultilayerPerceptron(void);

   // LAYER CONSTRUCTOR

   explicit MultilayerPerceptron(const Vector<PerceptronLayer>&);

   // NETWORK ARCHITECTURE CONSTRUCTOR

    explicit MultilayerPerceptron(const Vector<size_t>&);
    explicit MultilayerPerceptron(const Vector<int>&);

   // ONE LAYER CONSTRUCTOR 

   explicit MultilayerPerceptron(const size_t&, const size_t&);

   // TWO LAYERS CONSTRUCTOR 

   explicit MultilayerPerceptron(const size_t&, const size_t&, const size_t&);

   // COPY CONSTRUCTOR

   MultilayerPerceptron(const MultilayerPerceptron&);

   // DESTRUCTOR

   virtual ~MultilayerPerceptron(void);

   // ASSIGNMENT OPERATOR

   MultilayerPerceptron& operator = (const MultilayerPerceptron&);

   // EQUAL TO OPERATOR

   bool operator == (const MultilayerPerceptron&) const;

   // GET METHODS

   /// Returns a vector with the architecture of the multilayer perceptron.
   /// The elements of this vector are as follows;
   /// <UL>
   /// <LI> Number of inputs.</LI>
   /// <LI> Size of the first hidden layer.</LI>
   /// <LI> ...</LI>
   /// <LI> Number of output perceptrons.</LI>
   /// </UL>

   inline Vector<size_t> arrange_architecture(void) const
   {
      const size_t layers_number = get_layers_number();

      Vector<size_t> architecture;

      if(layers_number != 0)
      {
         const size_t inputs_number = get_inputs_number();
         architecture.resize(1+layers_number);

         architecture[0] = inputs_number;
         
         Vector<size_t> layers_size = arrange_layers_perceptrons_numbers();

         for(size_t i = 0; i < layers_number; i++)
         {
            architecture[i+1] = layers_size[i];
         }
      }

      return(architecture);
   }

   /// Returns a vector with the architecture of the multilayer perceptron.
   /// The elements of this vector are as follows;
   /// <UL>
   /// <LI> Number of inputs.</LI>
   /// <LI> Size of the first hidden layer.</LI>
   /// <LI> ...</LI>
   /// <LI> Number of output perceptrons.</LI>
   /// </UL>

   inline Vector<int> arrange_architecture_int(void) const
   {
      const size_t layers_number = get_layers_number();

      Vector<int> architecture;

      if(layers_number != 0)
      {
         const size_t inputs_number = get_inputs_number();
         architecture.resize(1+layers_number);

         architecture[0] = (int)inputs_number;

         Vector<size_t> layers_size = arrange_layers_perceptrons_numbers();

         for(size_t i = 0; i < layers_number; i++)
         {
            architecture[i+1] = (int)layers_size[i];
         }
      }

      return(architecture);
   }

   /// Returns a vector with the sizes of the hidden layers in the multilayer perceptron.

   inline Vector<size_t> arrange_complexity(void) const
   {
      const size_t layers_number = get_layers_number();

      Vector<size_t> complexity;

      if(layers_number != 1)
      {
         complexity.resize(layers_number-1);

         Vector<size_t> layers_size = arrange_layers_perceptrons_numbers();

         for(size_t i = 0; i < layers_number-1; i++)
         {
            complexity[i] = layers_size[i];
         }
      }

      return(complexity);
   }


   /// Returns the number of inputs to the multilayer perceptron.

   inline size_t get_inputs_number(void) const
   {
      const size_t layers_number = get_layers_number();

      if(layers_number == 0)
      {
         return(0);
      }
      else
      {
         return(layers[0].get_inputs_number());
      }
   }

   /// Returns the number of layers in the multilayer perceptron. 

   inline size_t get_layers_number(void) const
   {
      return(layers.size());
   }

   /// Returns a vector with the number of inputs of each layer. 

   inline Vector<size_t> get_layers_inputs_number(void) const
   {
      const size_t layers_number = get_layers_number();

      Vector<size_t> layers_inputs_number(layers_number);

      for(size_t i = 0; i < layers_number; i++)
      {
	     layers_inputs_number[i] = layers[i].get_inputs_number();
      }

      return(layers_inputs_number);      
   }


   /// Returns a vector with the size of each layer. 

   inline Vector<size_t> arrange_layers_perceptrons_numbers(void) const
   {
      const size_t layers_number = get_layers_number();

      Vector<size_t> layers_perceptrons_number(layers_number);

      for(size_t i = 0; i < layers_number; i++)
      {
         layers_perceptrons_number[i] = layers[i].get_perceptrons_number();
      }

      return(layers_perceptrons_number);
   }


   /// Returns the number of outputs neurons in the multilayer perceptron 

   inline size_t get_outputs_number(void) const
   {
      const size_t layers_number = get_layers_number();

      if(layers_number == 0)
      {
         return(0);
      }
      else
      {
         return(layers[layers_number-1].get_perceptrons_number());
      }
   }

   const Vector<PerceptronLayer>& get_layers(void) const;
   const PerceptronLayer& get_layer(const size_t&) const;
   PerceptronLayer* get_layer_pointer(const size_t&);

   size_t count_perceptrons_number(void) const;
   Vector<size_t> count_cumulative_perceptrons_number(void) const;
 
   // Parameters
   
   Vector<size_t> arrange_layers_parameters_number(void) const;
   Vector<size_t> arrange_layers_cumulative_parameters_number(void) const;

   Vector< Vector<double> > arrange_layers_biases(void) const;
   Vector< Matrix<double> > arrange_layers_synaptic_weights(void) const;

   Vector< Vector<double> > get_layers_parameters(void) const;

   size_t count_parameters_number(void) const;
   Vector<double> arrange_parameters(void) const;   
   
   Vector<double> arrange_parameters_statistics(void) const;

   Vector<size_t> count_layers_parameters_numbers(void) const;

   size_t get_layer_index(const size_t&) const;
   size_t get_perceptron_index(const size_t&, const size_t&) const;

   size_t get_layer_bias_index(const size_t&, const size_t&) const;
   size_t get_layer_synaptic_weight_index(const size_t&, const size_t&, const size_t&) const;
   
   Vector<size_t> arrange_parameter_indices(const size_t&) const;
   Matrix<size_t> arrange_parameters_indices(void) const;

   // Activation functions

   Vector<Perceptron::ActivationFunction> get_layers_activation_function(void) const;
   Vector<std::string> write_layers_activation_function(void) const;

   // Display messages

   const bool& get_display(void) const;

   // SET METHODS

   void set(void);
   void set(const Vector<PerceptronLayer>&);
   void set(const Vector<size_t>&);
   void set(const Vector<int>&);
   void set(const size_t&, const size_t&);
   void set(const size_t&, const size_t&, const size_t&);
   void set(const MultilayerPerceptron&);

   virtual void set_default(void);

   // Architecture

   void set_inputs_number(const size_t&);

   void set_layers_perceptrons_number(const Vector<size_t>&);
   void set_layer_perceptrons_number(const size_t&, const size_t&);

   void set_layers(const Vector<PerceptronLayer>&);
   //void set_layer(const size_t&, const PerceptronLayer&);

   // Parameters

   void set_layers_biases(const Vector< Vector<double> >&);
   void set_layers_synaptic_weights(const Vector< Matrix<double> >&);

   void set_layer_parameters(const size_t, const Vector<double>&);
   void set_layers_parameters(const Vector< Vector<double> >&);

   void set_parameters(const Vector<double>&);

   void initialize_biases(const double&); 
   void initialize_synaptic_weights(const double&);
   void initialize_parameters(const double&);

   void randomize_parameters_uniform(void);
   void randomize_parameters_uniform(const double&, const double&);
   void randomize_parameters_uniform(const Vector<double>&, const Vector<double>&);
   void randomize_parameters_uniform(const Vector< Vector<double> >&);

   void randomize_parameters_normal(void);
   void randomize_parameters_normal(const double&, const double&);
   void randomize_parameters_normal(const Vector<double>&, const Vector<double>&);
   void randomize_parameters_normal(const Vector< Vector<double> >&);
 
   void initialize_parameters(void);

   void perturbate_parameters(const double&);

   double calculate_parameters_norm(void) const;

   // Activation functions

   void set_layers_activation_function(const Vector<Perceptron::ActivationFunction>&);
   void set_layers_activation_function(const Vector<std::string>&);

   void set_layer_activation_function(const size_t&, const Perceptron::ActivationFunction&);


   // Display messages

   void set_display(const bool&);

   // Check methods

   bool is_empty(void) const;

   // Growing and pruning

   void grow_input(void);

   void prune_input(const size_t&);
   void prune_output(const size_t&);

   void grow_layer_perceptron(const size_t&, const size_t& = 1);
   void prune_layer_perceptron(const size_t&, const size_t&);

   // Multilayer perceptron initialization methods

   void initialize_random(void);

   // PerceptronLayer combination-combination

   Vector<double> calculate_layer_combination_combination(const size_t&, const Vector<double>&) const;
   Matrix<double> calculate_layer_combination_combination_Jacobian(const size_t&, const Vector<double>&) const;

   // Layers multilayer_perceptron_pointer outputs delta

   Vector<double> calculate_output_layer_combination(const size_t&, const Vector<double>&) const;

   Vector< Matrix<double> > calculate_output_layers_delta(const Vector< Vector<double> >&) const;
//   Vector< Matrix<double> > calculate_output_layers_delta(const Vector< Vector<double> >&, const Vector<double>&, const Vector<double>&) const;

   // Interlayers multilayer_perceptron_pointer outputs Delta

   Matrix< Vector< Matrix<double> > > calculate_output_interlayers_Delta(const Vector< Vector< Vector<double> > >&, const Matrix< Matrix<double> >&, const Vector< Matrix<double> >&) const;

   // Interlayer combination-combination

   Vector<double> calculate_interlayer_combination_combination(const size_t&, const size_t&, const Vector<double>&) const;
   Matrix<double> calculate_interlayer_combination_combination_Jacobian(const size_t&, const size_t&, const Vector<double>&) const;

   // Forward propagation inputs

   Vector< Vector<double> > calculate_layers_input(const Vector<double>&) const;
   Vector< Vector<double> > arrange_layers_input(const Vector<double>&, const Vector< Vector<double> >&) const;
   Vector< Vector<double> > arrange_layers_input(const Vector<double>&, const Vector< Vector< Vector<double> > >&) const;

   // Forward propagation combination

   Vector< Vector<double> > calculate_layers_combination(const Vector<double>&) const;
   Vector< Matrix<double> > calculate_layers_combination_Jacobian(const Vector<double>&) const;

   Vector< Matrix<double> > calculate_layers_combination_parameters_Jacobian(const Vector< Vector<double> >&) const;

   Vector< Vector< Vector<double> > > calculate_perceptrons_combination_parameters_gradient(const Vector< Vector<double> >&) const;

   // Forward propagation activation

   Vector< Vector<double> > calculate_layers_activation(const Vector<double>&) const;
   Vector< Vector<double> > calculate_layers_activation_derivative(const Vector<double>&) const;
   Vector< Vector<double> > calculate_layers_activation_second_derivative(const Vector<double>&) const;

   // Forward propagation outputs

   //Vector< Vector<double> > calculate_layers_output(const Vector<double>&) const;
   Vector< Matrix<double> > calculate_layers_Jacobian(const Vector<double>&) const;
   Vector< Vector< Matrix<double> > > calculate_layers_Hessian_form(const Vector<double>&) const;

   Matrix <Matrix<double> > calculate_interlayers_combination_combination_Jacobian(const Vector<double>&) const;   
   Matrix <Matrix<double> > calculate_interlayers_combination_combination_Jacobian(const Vector< Vector<double> >&) const;   

   // Forward propagation

   //Vector< Vector< Vector<double> > > calculate_zero_order_forward_propagation(const Vector<double>&) const;
   Vector< Vector< Vector<double> > > calculate_first_order_forward_propagation(const Vector<double>&) const;
   Vector< Vector< Vector<double> > > calculate_second_order_forward_propagation(const Vector<double>&) const;

   // Output 

   Vector<double> calculate_outputs(const Vector<double>&) const;
   Matrix<double> calculate_Jacobian(const Vector<double>&) const;
   Vector< Matrix<double> > calculate_Hessian_form(const Vector<double>&) const;

   Vector<double> calculate_outputs(const Vector<double>&, const Vector<double>&) const;
   Matrix<double> calculate_Jacobian(const Vector<double>&, const Vector<double>&) const;
   Vector< Matrix<double> > calculate_Hessian_form(const Vector<double>&, const Vector<double>&) const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML(void) const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   // void read_XML(   );

   // PMML Methods
   void to_PMML(tinyxml2::XMLElement* ) const;
   void write_PMML(tinyxml2::XMLPrinter&, bool is_softmax_normalization_method = false) const;

   void from_PMML(const tinyxml2::XMLElement*);

   // Information

   Matrix<std::string> write_information(void) const;

   // Expression methods

   std::string write_expression(const Vector<std::string>&, const Vector<std::string>&) const;

   std::string to_string(void) const;


protected:

   // MEMBERS

   /// Multilayer perceptron layers. It is built as a vector of vectors of perceptrons.
   /// The size of this vector is equal to the number of layers. 
   /// The size of each subvector is equal to the number of neurons in the corresponding layer.

   Vector<PerceptronLayer> layers;

   /// Display messages to screen. 

   bool display;
};

}

#endif


// OpenNN: Open Neural MultilayerPerceptrons Library.
// Copyright (c) 2005-2016 Roberto Lopez.
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

