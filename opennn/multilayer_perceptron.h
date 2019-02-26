/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural MultilayerPerceptrons Library                                                          */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M U L T I L A Y E R   P E R C E P T R O N   C L A S S   H E A D E R                                        */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
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

#include "perceptron_layer.h"
#include "inputs.h"
#include "outputs.h"

#include "vector.h"
#include "matrix.h"
#include "numerical_differentiation.h"

// TinyXml includes

#include "tinyxml2.h"


namespace OpenNN
{

/// This class represents the concept of multilayer perceptron.
/// A multilayer perceptron is a feed-forward network of layers of perceptrons. 
/// This is the most important class included in the definition of a neural network. 

class MultilayerPerceptron
{

public:

   // DEFAULT CONSTRUCTOR

   explicit MultilayerPerceptron();

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

   virtual ~MultilayerPerceptron();

   // ASSIGNMENT OPERATOR

   MultilayerPerceptron& operator = (const MultilayerPerceptron&);

   // EQUAL TO OPERATOR

   bool operator == (const MultilayerPerceptron&) const;

   struct LayersParameters
   {
       /// Default constructor.

       LayersParameters(const Vector<size_t> architecture, const Vector<double>& parameters)
       {
           const size_t layers_number = architecture.size()-1;

           biases.set(layers_number);
           synaptic_weights.set(layers_number);

           size_t layer_parameters_number;
           size_t inputs_number;
           size_t perceptrons_number;
           size_t weight_position = 0;
           size_t bias_position = 0;

           for(size_t i = 0; i < layers_number; i++)
           {
               inputs_number = architecture[i];
               perceptrons_number = architecture[i+1];

               layer_parameters_number = perceptrons_number*(inputs_number+1);

               synaptic_weights[i].set(inputs_number, perceptrons_number);
               biases[i].set(perceptrons_number);

               bias_position += perceptrons_number*inputs_number;

               for(size_t k = 0; k < perceptrons_number; k++)
               {
                   for(size_t j = 0; j < inputs_number; j++)
                   {
                       synaptic_weights[i](j,k) = parameters[weight_position];

                       weight_position++;
                   }

                   biases[i][k] = parameters[k + bias_position];
               }

               weight_position += perceptrons_number;
               bias_position += perceptrons_number;
           }
       }

       LayersParameters(const size_t& layers_number)
       {
           biases.set(layers_number);
           synaptic_weights.set(layers_number);
       }

       virtual ~LayersParameters()
       {
       }

       Vector< Vector<double> > biases;
       Vector< Matrix<double> > synaptic_weights;
   };


   struct Pointers
   {
       Pointers()
       {
       }

       virtual ~Pointers();

       Vector<double> get_parameters() const;

       void update_parameters(const Vector<double>&);

       void update_parameters_sgd(const Vector<double*>&, const double&, const bool&, const double&,
                                  const double&, const size_t&, const Vector<double>&);

       Vector<double*> biases_pointers;
       Vector<double*> weights_pointers;

       size_t layers_number;
       Vector<size_t> architecture;
       Vector<string> layer_activations;

       bool CUDA_initialized = false;
   };


   struct FirstOrderForwardPropagation
   {
       /// Default constructor.

       FirstOrderForwardPropagation(const size_t layers_number)
       {
           layers_activations.set(layers_number);
           layers_activation_derivatives.set(layers_number);
       }

       virtual ~FirstOrderForwardPropagation()
       {
       }

       void print() const
       {
           cout << "Layers activations:" << endl;
           cout << layers_activations << endl;
           cout << "Layers activation derivatives:" << endl;
           cout << layers_activation_derivatives << endl;
       }

       Vector< Matrix<double> > layers_activations;
       Vector< Matrix<double> > layers_activation_derivatives;
   };


   // GET METHODS

   /// Returns a vector with the architecture of the multilayer perceptron.
   /// The elements of this vector are as follows;
   /// <UL>
   /// <LI> Number of inputs.</LI>
   /// <LI> Size of the first hidden layer.</LI>
   /// <LI> ...</LI>
   /// <LI> Number of output perceptrons.</LI>
   /// </UL>

   inline Vector<size_t> get_architecture() const
   {
      const size_t layers_number = get_layers_number();

      Vector<size_t> architecture;

      if(layers_number != 0)
      {
         const size_t inputs_number = get_inputs_number();
         architecture.resize(1+layers_number);

         architecture[0] = inputs_number;
         
         Vector<size_t> layers_size = get_layers_perceptrons_numbers();

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

   inline Vector<int> get_architecture_int() const
   {
      const size_t layers_number = get_layers_number();

      Vector<int> architecture;

      if(layers_number != 0)
      {
         const size_t inputs_number = get_inputs_number();
         architecture.resize(1+layers_number);

         architecture[0] = (int)inputs_number;

         Vector<size_t> layers_size = get_layers_perceptrons_numbers();

         for(size_t i = 0; i < layers_number; i++)
         {
            architecture[i+1] = (int)layers_size[i];
         }
      }

      return(architecture);
   }

   /// Returns a vector with the sizes of the hidden layers in the multilayer perceptron.

   inline Vector<size_t> get_complexity() const
   {
      const size_t layers_number = get_layers_number();

      Vector<size_t> complexity;

      if(layers_number != 1)
      {
         complexity.resize(layers_number-1);

         Vector<size_t> layers_size = get_layers_perceptrons_numbers();

         for(size_t i = 0; i < layers_number-1; i++)
         {
            complexity[i] = layers_size[i];
         }
      }

      return(complexity);
   }


   /// Returns the number of inputs to the multilayer perceptron.

   inline size_t get_inputs_number() const
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

   inline size_t get_layers_number() const
   {
      return(layers.size());
   }


   /// Returns a vector with the number of inputs of each layer. 

   inline Vector<size_t> get_layers_inputs_number() const
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

   inline Vector<size_t> get_layers_perceptrons_numbers() const
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

   inline size_t get_outputs_number() const
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

   const Vector<PerceptronLayer>& get_layers() const;
   const PerceptronLayer& get_layer(const size_t&) const;
   PerceptronLayer* get_layer_pointer(const size_t&);

   size_t get_perceptrons_number() const;
   Vector<size_t> count_cumulative_perceptrons_number() const;
 
   // Parameters
   
   Vector<size_t> get_layers_parameters_number() const;
   Vector<size_t> count_layers_cumulative_parameters_number() const;

   Vector< Vector<double> > get_layers_biases() const;
   Vector< Matrix<double> > get_layers_synaptic_weights() const;

   size_t get_parameters_number() const;
   Vector<double> get_parameters() const;   
   
   Vector<double> get_parameters_statistics() const;

   Vector<size_t> get_layers_parameters_numbers() const;

   size_t get_layer_index(const size_t&) const;
   size_t get_perceptron_index(const size_t&, const size_t&) const;

   size_t get_layer_bias_index(const size_t&, const size_t&) const;
   size_t get_layer_synaptic_weight_index(const size_t&, const size_t&, const size_t&) const;
   
   Vector<size_t> get_parameter_indices(const size_t&) const;
   Matrix<size_t> get_parameters_indices() const;

   // Activation functions

   Vector<PerceptronLayer::ActivationFunction> get_layers_activation_function() const;
   Vector<string> write_layers_activation_function() const;

   // Display messages

   const bool& get_display() const;

   // SET METHODS

   void set();
   void set(const Vector<PerceptronLayer>&);
   void set(const Vector<size_t>&);
   void set(const Vector<int>&);
   void set(const size_t&, const size_t&);
   void set(const size_t&, const size_t&, const size_t&);
   void set(const MultilayerPerceptron&);

   virtual void set_default();

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
   void initialize_synaptic_weights_Glorot();
   void initialize_parameters(const double&);

   void randomize_parameters_uniform();
   void randomize_parameters_uniform(const double&, const double&);
   void randomize_parameters_uniform(const Vector<double>&, const Vector<double>&);
   void randomize_parameters_uniform(const Vector< Vector<double> >&);

   void randomize_parameters_normal();
   void randomize_parameters_normal(const double&, const double&);
   void randomize_parameters_normal(const Vector<double>&, const Vector<double>&);
   void randomize_parameters_normal(const Vector< Vector<double> >&);
 
   void initialize_parameters();

   void perturbate_parameters(const double&);

   double calculate_parameters_norm() const;

   // Activation functions

   void set_layers_activation_function(const Vector<PerceptronLayer::ActivationFunction>&);
   void set_layers_activation_function(const Vector<string>&);

   void set_layer_activation_function(const size_t&, const PerceptronLayer::ActivationFunction&);

   // Display messages

   void set_display(const bool&);

   // Check methods

   bool is_empty() const;

   // Growing and pruning

   void grow_input();

   void prune_input(const size_t&);
   void prune_output(const size_t&);

   void grow_layer_perceptron(const size_t&, const size_t& = 1);
   void prune_layer_perceptron(const size_t&, const size_t&);

   // Multilayer perceptron initialization methods

   void initialize_random();

   // Forward propagation combination

   Vector< Matrix<double> > calculate_layers_combinations(const Matrix<double>&) const;

   Vector < Matrix<double> > calculate_layers_combination_Jacobian(const size_t&, const Matrix<double>&) const;

   Vector < Vector< Matrix<double> > > calculate_layers_combination_parameters_Jacobian(const Vector< Matrix<double> >&) const;

   Vector< Vector< Vector<double> > > calculate_perceptrons_combination_parameters_gradient(const Vector< Vector<double> >&) const;

   // Forward propagation activation

   Vector< Matrix<double> > calculate_layers_activations_derivatives(const Matrix<double>&) const;

   // Forward propagation outputs

   FirstOrderForwardPropagation calculate_first_order_forward_propagation(const Matrix<double>&) const;

   // Output 

   Vector< Vector< Matrix<double> > > calculate_layers_Jacobian(const Matrix<double>&) const;
   Vector< Matrix<double> > calculate_Jacobian(const Matrix<double>&) const;

   Matrix<double> calculate_outputs(const Matrix<double>&) const;
   Matrix<double> calculate_outputs(const Matrix<double>&,  const Vector<double>&) const;

   // Serialization methods

   Pointers host_to_device() const;

   tinyxml2::XMLDocument* to_XML() const;
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   // void read_XML(   );

   // PMML Methods
   void to_PMML(tinyxml2::XMLElement* ) const;
   void write_PMML(tinyxml2::XMLPrinter&, bool is_softmax_normalization_method = false) const;

   void from_PMML(const tinyxml2::XMLElement*);

   // Information

   Matrix<string> write_information() const;

   // Expression methods

   string write_expression(const Vector<string>&, const Vector<string>&) const;
   string write_expression_php(const Vector<string>&, const Vector<string>&) const;

   string object_to_string() const;

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
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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

