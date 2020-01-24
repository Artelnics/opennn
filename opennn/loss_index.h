//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   I N D E X   C L A S S   H E A D E R                         
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LOSSINDEX_H
#define LOSSINDEX_H

// System includes

#include <string>
#include <sstream>
#include <fstream>
#include <ostream>
#include <iostream>
#include <cmath>

// OpenNN includes

#include "config.h"
#include "data_set.h"
#include "neural_network.h"
#include "numerical_differentiation.h"
#include "tinyxml2.h"

namespace OpenNN
{

/// This abstrac class represents the concept of loss index composed of an error term and a regularization term.

///
/// The error terms could be:
/// <ul>
/// <li> Cross Entropy Error.
/// <li> Mean Squared Error.
/// <li> Minkowski Error.
/// <li> Normalized Squared Error.
/// <li> Sum Squared Error.
/// <li> Weighted Squared Error.
/// </ul>

class LossIndex
{

public:

   // Constructors

   explicit LossIndex();

   explicit LossIndex(NeuralNetwork*);

   explicit LossIndex(DataSet*);

   explicit LossIndex(NeuralNetwork*, DataSet*);

   explicit LossIndex(const tinyxml2::XMLDocument&);

   LossIndex(const LossIndex&);

   // Destructor

   virtual ~LossIndex();

   // Methods

   /// Enumeration of available regularization methods.

   enum RegularizationMethod{L1, L2, NoRegularization};


   /// A loss index composed of several terms, this structure represent the First Order for this function.

   ///
   /// Set of loss value and gradient vector of the loss index.
   /// A method returning this structure might be implemented more efficiently than the loss and gradient methods separately.

   struct FirstOrderLoss
   {
       /// Default constructor.

       explicit FirstOrderLoss() {}

       explicit FirstOrderLoss(const LossIndex*);

       virtual ~FirstOrderLoss();

       void print()
       {
           cout << "Output gradient:" << endl;
           cout << output_gradient << endl;

           for(Index i = 0; i < layers_delta.size(); i++)
           {
               cout << "Layers delta " << i << ":" << endl;
               cout << layers_delta[i] << endl;
           }

           cout << "Loss:" << endl;
           cout << loss << endl;

           cout << "Error gradient:" << endl;
           cout << error_gradient << endl;

           cout << "Regularization gradient:" << endl;
           cout << regularization_gradient << endl;

           cout << "Gradient:" << endl;
           cout << gradient << endl;
       }

       Tensor<type, 2> output_gradient;

       Tensor<Tensor<type, 2>, 1> layers_delta;
       Tensor<Tensor<type, 1>, 1> layers_error_gradient;

       type loss;

       Tensor<type, 1> error_gradient;
       Tensor<type, 1> regularization_gradient;
       Tensor<type, 1> gradient;
   };


   /// This structure contains second order information about the loss function (loss, gradient and Hessian).

   ///
   /// Set of loss value, gradient vector and <i>Hessian</i> matrix of the loss index.
   /// A method returning this structure might be implemented more efficiently than the loss, gradient and <i>Hessian</i> methods separately.

   struct SecondOrderLoss
   {
       /// Default constructor.

       SecondOrderLoss() {}

       SecondOrderLoss(const Index& parameters_number)
       {
           loss = 0.0;
           gradient = Tensor<type, 1>(parameters_number);
           hessian = Tensor<type, 2>(parameters_number, parameters_number);
       }

       type loss;
       Tensor<type, 1> gradient;
       Tensor<type, 2> hessian;
   };


   /// Returns a pointer to the neural network object associated to the error term.

   inline NeuralNetwork* get_neural_network_pointer() const 
   {
        #ifdef __OPENNN_DEBUG__

        if(!neural_network_pointer)
        {
             ostringstream buffer;

             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "NeuralNetwork* get_neural_network_pointer() const method.\n"
                    << "Neural network pointer is nullptr.\n";

             throw logic_error(buffer.str());
        }

        #endif

      return neural_network_pointer;
   }

   /// Returns a pointer to the data set object associated to the error term.

   inline DataSet* get_data_set_pointer() const 
   {
        #ifdef __OPENNN_DEBUG__

        if(!data_set_pointer)
        {
             ostringstream buffer;

             buffer << "OpenNN Exception: LossIndex class.\n"
                    << "DataSet* get_data_set_pointer() const method.\n"
                    << "DataSet pointer is nullptr.\n";

             throw logic_error(buffer.str());
        }

        #endif

      return data_set_pointer;
   }

   const type& get_regularization_weight() const;

   const bool& get_display() const;

   bool has_neural_network() const;

   bool has_data_set() const;

   // Get methods

   RegularizationMethod get_regularization_method() const;

   // Set methods

   void set();
   void set(NeuralNetwork*);
   void set(DataSet*);
   void set(NeuralNetwork*, DataSet*);

   void set(const LossIndex&);

   void set_neural_network_pointer(NeuralNetwork*);

   void set_data_set_pointer(DataSet*);

   void set_default();

   void set_regularization_method(const RegularizationMethod&);
   void set_regularization_method(const string&);
   void set_regularization_weight(const type&);

   void set_display(const bool&);

   bool has_selection() const;

   // Loss methods

   type calculate_training_loss() const;
   type calculate_training_loss(const Tensor<type, 1>&) const;
   type calculate_training_loss(const Tensor<type, 1>&, const type&) const;

   // Loss gradient methods

   Tensor<type, 1> calculate_training_loss_gradient() const;

   // ERROR METHODS

   virtual type calculate_training_error() const;
   virtual type calculate_training_error(const Tensor<type, 1>&) const;

   virtual type calculate_selection_error() const;

   virtual type calculate_batch_error(const Tensor<Index, 1>&) const = 0;
   virtual type calculate_batch_error(const Tensor<Index, 1>&, const Tensor<type, 1>&) const = 0;

   // GRADIENT METHODS

   virtual Tensor<type, 2> calculate_output_gradient(const Tensor<type, 2>&, const Tensor<type, 2>&) const = 0;

   virtual void calculate_output_gradient(const DataSet::Batch&, const NeuralNetwork::ForwardPropagation&, FirstOrderLoss&) const = 0;

   virtual Tensor<type, 1> calculate_batch_error_gradient(const Tensor<Index, 1>&) const;

   Tensor<type, 1> calculate_training_error_gradient() const;

   Tensor<type, 1> calculate_training_error_gradient_numerical_differentiation() const;

   // ERROR TERMS METHODS

   virtual Tensor<type, 1> calculate_batch_error_terms(const Tensor<Index, 1>&) const {return Tensor<type, 1>();}
   virtual Tensor<type, 2> calculate_batch_error_terms_Jacobian(const Tensor<Index, 1>&) const {return Tensor<type, 2>();}

   virtual FirstOrderLoss calculate_first_order_loss(const DataSet::Batch&) const = 0;

   virtual void calculate_first_order_loss(const ThreadPoolDevice&, const DataSet::Batch&, const NeuralNetwork::ForwardPropagation&, FirstOrderLoss&) const = 0;

   virtual FirstOrderLoss calculate_first_order_loss() const {return FirstOrderLoss();}
   virtual SecondOrderLoss calculate_terms_second_order_loss() const {return SecondOrderLoss();}

   // Regularization methods

   type calculate_regularization() const;
   Tensor<type, 1> calculate_regularization_gradient() const;
   Tensor<type, 2> calculate_regularization_hessian() const;

   type calculate_regularization(const Tensor<type, 1>&) const;
   Tensor<type, 1> calculate_regularization_gradient(const Tensor<type, 1>&) const;
   Tensor<type, 2> calculate_regularization_hessian(const Tensor<type, 1>&) const;

   // Delta methods

   Tensor<Tensor<type, 2>, 1> calculate_layers_delta(const Tensor<Layer::ForwardPropagation, 1>&, const Tensor<type, 2>&) const;

   void calculate_layers_delta(const ThreadPoolDevice& thread_pool_device, const NeuralNetwork::ForwardPropagation& forward_propagation, FirstOrderLoss& first_order_loss) const
   {
        const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

        const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

        if(trainable_layers_number == 0) return;

        // Output layer

        trainable_layers_pointers[trainable_layers_number-1]
        ->calculate_output_delta(thread_pool_device,
                                 forward_propagation.layers[trainable_layers_number-1].activations_derivatives,
                                 first_order_loss.output_gradient,
                                 first_order_loss.layers_delta[trainable_layers_number-1]);

      // Hidden layers

      for(Index i = static_cast<Index>(trainable_layers_number)-2; i >= 0; i--)
      {
          Layer* previous_layer_pointer = trainable_layers_pointers[static_cast<Index>(i+1)];

          trainable_layers_pointers[i]
          ->calculate_hidden_delta(thread_pool_device,
                                   previous_layer_pointer,
                                   forward_propagation.layers[i].activations,
                                   forward_propagation.layers[i].activations_derivatives,
                                   first_order_loss.layers_delta[static_cast<Index>(i+1)],
                                   first_order_loss.layers_delta[i]);
      }
   }

   Tensor<type, 1> calculate_error_gradient(const Tensor<type, 2>&, const Tensor<Layer::ForwardPropagation, 1>&, const Tensor<Tensor<type, 2>, 1>&) const;

   void calculate_error_gradient(const ThreadPoolDevice& thread_pool_device,
                                 const DataSet::Batch& batch,
                                 const NeuralNetwork::ForwardPropagation& forward_propagation,
                                 FirstOrderLoss& first_order_loss) const
   {
       const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

       #ifdef __OPENNN_DEBUG__

       check();

       // Hidden errors size

       const Index layers_delta_size = first_order_loss.layers_delta.size();

       if(layers_delta_size != trainable_layers_number)
       {
          ostringstream buffer;

         buffer << "OpenNN Exception: LossIndex class.\n"
                << "void calculate_error_gradient(const DataSet::Batch&, const NeuralNetwork::ForwardPropagation&, FirstOrderLoss&) method.\n"
                << "Size of layers delta(" << layers_delta_size << ") must be equal to number of layers(" << trainable_layers_number << ").\n";

         throw logic_error(buffer.str());
       }

       #endif

       const Tensor<Index, 1> trainable_layers_parameters_number = neural_network_pointer->get_trainable_layers_parameters_numbers();

       const Tensor<Layer*, 1> trainable_layers_pointers = neural_network_pointer->get_trainable_layers_pointers();

       Index index = 0;

       trainable_layers_pointers[0]->calculate_error_gradient(thread_pool_device,
                                                              batch.inputs_2d,
                                                              forward_propagation.layers[0],
                                                              first_order_loss.layers_delta[0],
                                                              first_order_loss.layers_error_gradient[0]);

       for(Index i = 0; i < trainable_layers_parameters_number[0]; i++)
       {
            first_order_loss.error_gradient[i] = first_order_loss.layers_error_gradient[0](i);
       }

       index += trainable_layers_parameters_number[0];

       for(Index i = 1; i < trainable_layers_number; i++)
       {
           trainable_layers_pointers[i]->calculate_error_gradient(thread_pool_device,
                   forward_propagation.layers[i-1].activations,
                   forward_propagation.layers[i-1],
                   first_order_loss.layers_delta[i],
                   first_order_loss.layers_error_gradient[i]);

           for(Index j = 0; j < trainable_layers_parameters_number[i]; j++)
           {
                first_order_loss.error_gradient[index + j] = first_order_loss.layers_error_gradient[i](j);
           }

           index += trainable_layers_parameters_number[i];
       }
   }

   Tensor<type, 2> calculate_layer_error_terms_Jacobian(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   Tensor<type, 2> calculate_error_terms_Jacobian(const Tensor<type, 2>&, const Tensor<Layer::ForwardPropagation, 1>&, const Tensor<Tensor<type, 2>, 1>&) const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML() const;

   void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;

   void regularization_from_XML(const tinyxml2::XMLDocument&);
   void write_regularization_XML(tinyxml2::XMLPrinter&) const;

   string get_error_type() const;
   virtual string get_error_type_text() const;

   string write_information() const;

   string write_regularization_method() const;

   // Checking methods

   void check() const;

protected:

   /// Pointer to a neural network object.

   NeuralNetwork* neural_network_pointer = nullptr;

   /// Pointer to a data set object.

   DataSet* data_set_pointer = nullptr;

   /// Pointer to a regularization method object.

   RegularizationMethod regularization_method = L2;

   /// Regularization weight value.

   type regularization_weight;

   /// Display messages to screen. 

   bool display = true;
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
