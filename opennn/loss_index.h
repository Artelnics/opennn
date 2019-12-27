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

#include "vector.h"
#include "matrix.h"

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
   /// Set of loss value and gradient vector of the peformance function.
   /// A method returning this structure might be implemented more efficiently than the loss and gradient methods separately.

   struct FirstOrderLoss
   {
       /// Default constructor.

       explicit FirstOrderLoss() {}

       explicit FirstOrderLoss(const size_t& new_parameters_number);

       void set_parameters_number(const size_t& new_parameters_number);

       virtual ~FirstOrderLoss();

       double loss;

       Vector<double> gradient;
   };


   /// This structure represents the Second Order in the loss function.

   ///
   /// Set of loss value, gradient vector and <i>Hessian</i> matrix of the peformance function.
   /// A method returning this structure might be implemented more efficiently than the loss, gradient and <i>Hessian</i> methods separately.

   struct SecondOrderLoss
   {
       /// Default constructor.

       SecondOrderLoss() {}

       SecondOrderLoss(const size_t& parameters_number)
       {
           loss = 0.0;
           gradient.set(parameters_number, 0.0);
           hessian.set(parameters_number, parameters_number, 0.0);
       }

       double loss;
       Vector<double> gradient;
       Matrix<double> hessian;
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

   const double& get_regularization_weight() const;

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
   void set_regularization_weight(const double&);

   void set_display(const bool&);

   bool has_selection() const;

   // Loss methods

   double calculate_training_loss() const;
   double calculate_training_loss(const Vector<double>&) const;
   double calculate_training_loss(const Vector<double>&, const double&) const;

   // Loss gradient methods

   Vector<double> calculate_training_loss_gradient() const;

   // ERROR METHODS

   virtual double calculate_training_error() const;
   virtual double calculate_training_error(const Vector<double>&) const;

   virtual double calculate_selection_error() const;

   virtual double calculate_batch_error(const Vector<size_t>&) const = 0;
   virtual double calculate_batch_error(const Vector<size_t>&, const Vector<double>&) const = 0;

   // GRADIENT METHODS

   virtual Tensor<double> calculate_output_gradient(const Tensor<double>&, const Tensor<double>&) const = 0;

   virtual Vector<double> calculate_batch_error_gradient(const Vector<size_t>&) const;

   Vector<double> calculate_training_error_gradient() const;

   Vector<double> calculate_training_error_gradient_numerical_differentiation() const;

   // ERROR TERMS METHODS

   virtual Vector<double> calculate_batch_error_terms(const Vector<size_t>&) const {return Vector<double>();}
   virtual Matrix<double> calculate_batch_error_terms_Jacobian(const Vector<size_t>&) const {return Matrix<double>();}

   virtual FirstOrderLoss calculate_batch_first_order_loss(const Vector<size_t>&) const {return FirstOrderLoss();}

   virtual FirstOrderLoss calculate_first_order_loss() const {return FirstOrderLoss();}
   virtual SecondOrderLoss calculate_terms_second_order_loss() const {return SecondOrderLoss();}

   // Regularization methods

   double calculate_regularization() const;
   Vector<double> calculate_regularization_gradient() const;
   Matrix<double> calculate_regularization_hessian() const;

   double calculate_regularization(const Vector<double>&) const;
   Vector<double> calculate_regularization_gradient(const Vector<double>&) const;
   Matrix<double> calculate_regularization_hessian(const Vector<double>&) const;

   // Delta methods

   Vector<Tensor<double>> calculate_layers_delta(const Vector<Layer::FirstOrderActivations>&, const Tensor<double>&) const;

   Vector<double> calculate_error_gradient(const Tensor<double>&, const Vector<Layer::FirstOrderActivations>&, const Vector<Tensor<double>>&) const;

   Matrix<double> calculate_layer_error_terms_Jacobian(const Tensor<double>&, const Tensor<double>&) const;

   Matrix<double> calculate_error_terms_Jacobian(const Tensor<double>&, const Vector<Layer::FirstOrderActivations>&, const Vector<Tensor<double>>&) const;

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

   double regularization_weight = 0.01;

   /// Display messages to screen. 

   bool display = true;
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
