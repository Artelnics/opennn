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

namespace OpenNN
{

/// This abstract class represents the concept of loss index composed of an error term and a regularization term.

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

   explicit LossIndex(NeuralNetwork*, DataSet*);

   // Destructor

   virtual ~LossIndex();

   // Methods

   /// Enumeration of available regularization methods.

   enum RegularizationMethod{L1, L2, NoRegularization};


   /// A loss index composed of several terms, this structure represent the First Order for this function.

   ///
   /// Set of loss value and gradient vector of the loss index.
   /// A method returning this structure might be implemented more efficiently than the loss and gradient methods separately.

   struct BackPropagation
   {
       /// Default constructor.

       explicit BackPropagation() {}

       explicit BackPropagation(const Index& new_batch_samples_number, LossIndex* new_loss_index_pointer)
       {
           if(new_batch_samples_number == 0) return;

           set(new_batch_samples_number, new_loss_index_pointer);
       }

       virtual ~BackPropagation();

       void set(const Index& new_batch_samples_number, LossIndex* new_loss_index_pointer)
       {                      
           batch_samples_number = new_batch_samples_number;

           loss_index_pointer = new_loss_index_pointer;

           // Neural network

           NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

           const Index parameters_number = neural_network_pointer->get_parameters_number();

           const Index outputs_number = neural_network_pointer->get_outputs_number();

           // First order loss

           neural_network.set(batch_samples_number, neural_network_pointer);

           error = 0;

           loss = 0;

           output_gradient.resize(batch_samples_number, outputs_number);

           errors.resize(batch_samples_number, outputs_number);

           gradient.resize(parameters_number);
       }


       void print()
       {
           cout << "Error:" << endl;
           cout << error << endl;

           cout << "Loss:" << endl;
           cout << loss << endl;

           cout << "Output gradient:" << endl;
           cout << output_gradient << endl;

           cout << "Gradient:" << endl;
           cout << gradient << endl; 
       }

       LossIndex* loss_index_pointer = nullptr;

       Index batch_samples_number = 0;

       NeuralNetwork::BackPropagation neural_network;

       type error;

       type loss;

       Tensor<type, 2> output_gradient;

       Tensor<type, 2> errors;

       Tensor<type, 1> gradient;
   };


   /// This structure contains second order information about the loss function (loss, gradient and Hessian).
   /// Set of loss value, gradient vector and <i>Hessian</i> matrix of the loss index.
   /// A method returning this structure might be implemented more efficiently than the loss,
   /// gradient and <i>Hessian</i> methods separately.

   struct SecondOrderLoss
   {
       /// Default constructor.

       SecondOrderLoss() {}

       SecondOrderLoss(const Index& parameters_number, const Index& samples_number)
       {
           loss = 0;           
           gradient.resize(parameters_number);
           error_terms_Jacobian.resize(samples_number, parameters_number);
           hessian.resize(parameters_number, parameters_number);
           error_terms.resize(samples_number);
       }

       void sum_hessian_diagonal(const type& value)
       {
           const Index parameters_number = hessian.dimension(0);

           for(Index i = 0; i < parameters_number; i++)
               hessian(i,i) += value;
       }

       type error;
       type loss;

       Tensor<type, 1> error_terms;
       Tensor<type, 2> error_terms_Jacobian;

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

   void set_threads_number(const int&);

   void set_neural_network_pointer(NeuralNetwork*);

   virtual void set_data_set_pointer(DataSet*);

   void set_default();

   void set_regularization_method(const RegularizationMethod&);
   void set_regularization_method(const string&);
   void set_regularization_weight(const type&);

   void set_display(const bool&);

   bool has_selection() const;

   // GRADIENT METHODS

   virtual void calculate_output_gradient(const DataSet::Batch&,
                                          const NeuralNetwork::ForwardPropagation&,
                                          BackPropagation&) const = 0;

   // Numerical differentiation

   type calculate_eta() const;
   type calculate_h(const type&) const;

   Tensor<type, 1> calculate_error_gradient_numerical_differentiation(LossIndex*) const;

   Tensor<type, 2> calculate_Jacobian_numerical_differentiation(LossIndex*) const;

   // ERROR TERMS METHODS

   virtual Tensor<type, 2> calculate_batch_error_terms_Jacobian(const Tensor<Index, 1>&) const {return Tensor<type, 2>();}

   virtual void calculate_error(const DataSet::Batch&,
                                const NeuralNetwork::ForwardPropagation&,
                                BackPropagation&) const = 0;

   virtual void calculate_error_terms(const DataSet::Batch&,
                                      const NeuralNetwork::ForwardPropagation&,
                                      SecondOrderLoss&) const {return;}

   void back_propagate(const DataSet::Batch& batch,
                       NeuralNetwork::ForwardPropagation& forward_propagation,
                       BackPropagation& back_propagation) const;

   // Second Order loss

   void calculate_terms_second_order_loss(const DataSet::Batch& batch,
                                          NeuralNetwork::ForwardPropagation& forward_propagation,
                                          BackPropagation& back_propagation,
                                          SecondOrderLoss& second_order_loss) const;

   void calculate_error_terms_output_gradient(const DataSet::Batch& batch,
                                              NeuralNetwork::ForwardPropagation& forward_propagation,
                                              BackPropagation& back_propagation,
                                              SecondOrderLoss& second_order_loss) const;

   virtual void calculate_Jacobian_gradient(const DataSet::Batch&, SecondOrderLoss&) const {}

   virtual void calculate_hessian_approximation(const DataSet::Batch&, SecondOrderLoss&) const {}

   // Regularization methods

   type calculate_regularization(const Tensor<type, 1>&) const;
   Tensor<type, 1> calculate_regularization_gradient(const Tensor<type, 1>&) const;
   Tensor<type, 2> calculate_regularization_hessian(const Tensor<type, 1>&) const;

   // Delta methods

   void calculate_layers_delta(NeuralNetwork::ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation) const;

   void calculate_error_gradient(const DataSet::Batch& batch,
                                 const NeuralNetwork::ForwardPropagation& forward_propagation,
                                 BackPropagation& back_propagation) const;

   Tensor<type, 2> calculate_layer_error_terms_Jacobian(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   void calculate_error_terms_Jacobian(const DataSet::Batch&,
                                       const NeuralNetwork::ForwardPropagation&,
                                       const BackPropagation&,
                                       SecondOrderLoss&) const;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&);

   virtual void write_XML(tinyxml2::XMLPrinter&) const;

   void regularization_from_XML(const tinyxml2::XMLDocument&);
   void write_regularization_XML(tinyxml2::XMLPrinter&) const;

   string get_error_type() const;
   virtual string get_error_type_text() const;

   string write_regularization_method() const;

   // Checking methods

   void check() const;

   // Metrics

   Tensor<type, 2> kronecker_product(const Tensor<type, 1>&, const Tensor<type, 1>&) const;

   type l1_norm(const Tensor<type, 1>& parameters) const;
   Tensor<type, 1> l1_norm_gradient(const Tensor<type, 1>& parameters) const;
   Tensor<type, 2> l1_norm_hessian(const Tensor<type, 1>& parameters) const;

   type l2_norm(const Tensor<type, 1>& parameters) const;
   Tensor<type, 1> l2_norm_gradient(const Tensor<type, 1>& parameters) const;
   Tensor<type, 2> l2_norm_hessian(const Tensor<type, 1>& parameters) const;


protected:

   NonBlockingThreadPool* non_blocking_thread_pool = nullptr;
   ThreadPoolDevice* thread_pool_device = nullptr;

   /// Pointer to a neural network object.

   NeuralNetwork* neural_network_pointer = nullptr;

   /// Pointer to a data set object.

   DataSet* data_set_pointer = nullptr;

   /// Pointer to a regularization method object.

   RegularizationMethod regularization_method = L2;

   /// Regularization weight value.

   type regularization_weight = static_cast<type>(0.01);

   /// Display messages to screen. 

   bool display = true;

   const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0)};
   const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

   const Eigen::array<IndexPair<Index>, 2> SSE = {IndexPair<Index>(0, 0), IndexPair<Index>(1, 1)};

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/loss_index_cuda.h"
#endif


#ifdef OPENNN_MKL
    #include "../../opennn-mkl/opennn_mkl/loss_index_mkl.h"
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
