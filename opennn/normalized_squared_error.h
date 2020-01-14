//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z E D   S Q U A R E D   E R R O R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NORMALIZEDSQUAREDERROR_H
#define NORMALIZEDSQUAREDERROR_H

// System includes

#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <sstream>

// OpenNN includes

#include "loss_index.h"
#include "data_set.h"
#include "tinyxml2.h"

namespace OpenNN
{

/// This class represents the normalized squared error term. 

///
/// This error term is used in data modeling problems.
/// If it has a value of unity then the neural network is predicting the data "in the mean",
/// A value of zero means perfect prediction of data.

class NormalizedSquaredError : public LossIndex
{

public:

   explicit NormalizedSquaredError(NeuralNetwork*, DataSet*);

   // NEURAL NETWORK CONSTRUCTOR

   explicit NormalizedSquaredError(NeuralNetwork*);

   // DATA SET CONSTRUCTOR

   explicit NormalizedSquaredError(DataSet*);

   // DEFAULT CONSTRUCTOR

   explicit NormalizedSquaredError();

   // XML CONSTRUCTOR

   explicit NormalizedSquaredError(const tinyxml2::XMLDocument&);

   virtual ~NormalizedSquaredError();

   // Get methods

    double get_normalization_coefficient() const;

   // Set methods

    void set_normalization_coefficient();
    void set_normalization_coefficient(const double&);

    void set_selection_normalization_coefficient();
    void set_selection_normalization_coefficient(const double&);

    void set_default();

   // Normalization coefficients 

   double calculate_normalization_coefficient(const Matrix<double>&, const Vector<double>&) const;

   // Error methods

   double calculate_training_error() const;
   double calculate_training_error(const Vector<double>&) const;

   double calculate_selection_error() const;

   double calculate_batch_error(const Vector<size_t>&) const;
   double calculate_batch_error(const Vector<size_t>&, const Vector<double>&) const;

   // Gradient methods

   Tensor<double> calculate_output_gradient(const Tensor<double>&, const Tensor<double>&) const;

   void calculate_output_gradient(const DataSet::Batch& batch,
                                  const NeuralNetwork::ForwardPropagation& forward_propagation,
                                  FirstOrderLoss& first_order_loss) const
   {
        #ifdef __OPENNN_DEBUG__

        check();

        #endif

        const size_t trainable_layers_number = neural_network_pointer->get_trainable_layers_number();

        first_order_loss.output_gradient = forward_propagation.layers[trainable_layers_number-1].activations;

        first_order_loss.output_gradient -= batch.targets;

        first_order_loss.output_gradient *= 2.0 / normalization_coefficient;
   }


   LossIndex::FirstOrderLoss calculate_first_order_loss() const;

   LossIndex::FirstOrderLoss calculate_first_order_loss(const DataSet::Batch&) const;

   void calculate_first_order_loss(const DataSet::Batch& batch,
                                   const NeuralNetwork::ForwardPropagation& forward_propagation,
                                   FirstOrderLoss& first_order_loss) const
   {
    #ifdef __OPENNN_DEBUG__

    check();

    #endif

    // Neural network

    const size_t layers_number = neural_network_pointer->get_trainable_layers_number();

    // Loss index

    first_order_loss.loss = sum_squared_error(forward_propagation.layers[layers_number-1].activations, batch.targets) / normalization_coefficient;

    calculate_output_gradient(batch, forward_propagation, first_order_loss);

    calculate_layers_delta(forward_propagation, first_order_loss);

    calculate_error_gradient(batch, forward_propagation, first_order_loss);

    // Regularization

    if(regularization_method != RegularizationMethod::NoRegularization)
    {
        first_order_loss.loss += regularization_weight*calculate_regularization();
        first_order_loss.regularization_gradient = calculate_regularization_gradient()*regularization_weight;
    }

    first_order_loss.gradient += first_order_loss.error_gradient;
    first_order_loss.gradient += first_order_loss.regularization_gradient;
   }

   // Error terms methods

   Vector<double> calculate_training_error_terms(const Tensor<double>&, const Tensor<double>&) const;
   Vector<double> calculate_training_error_terms(const Vector<double>&) const;

   // Squared errors methods

   Vector<double> calculate_squared_errors() const;

   Vector<size_t> calculate_maximal_errors(const size_t& = 10) const;

   LossIndex::SecondOrderLoss calculate_terms_second_order_loss() const;

   string get_error_type() const;
   string get_error_type_text() const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML() const;   
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

private:

   /// Coefficient of normalization for the calculation of the training error.

   double normalization_coefficient;

   double selection_normalization_coefficient;
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

