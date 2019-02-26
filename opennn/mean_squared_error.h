/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M E A N   S Q U A R E D   E R R O R    C L A S S   H E A D E R                                             */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __MEANSQUAREDERROR_H__
#define __MEANSQUAREDERROR_H__

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <limits>
#include <math.h>

// OpenNN includes

#include "loss_index.h"
#include "data_set.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

/// This class represents the mean squared error term.
/// The mean squared error measures the difference between the outputs from a neural network and the targets in a data set. 
/// This functional is used in data modeling problems, such as function regression, 
/// classification and time series prediction.

class MeanSquaredError : public LossIndex
{

public:

   // DEFAULT CONSTRUCTOR

   explicit MeanSquaredError();

   // NEURAL NETWORK CONSTRUCTOR

   explicit MeanSquaredError(NeuralNetwork*);

   // DATA SET CONSTRUCTOR

   explicit MeanSquaredError(DataSet*);

   // GENERAL CONSTRUCTOR

   explicit MeanSquaredError(NeuralNetwork*, DataSet*);

   // XML CONSTRUCTOR

   explicit MeanSquaredError(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   MeanSquaredError(const MeanSquaredError&);

   // DESTRUCTOR

   virtual ~MeanSquaredError();

   // STRUCTURES


   // METHODS

   // Error methods

   double calculate_training_error() const;

   double calculate_selection_error() const;

   double calculate_training_error(const Vector<double>&) const;

   double calculate_batch_error(const Vector<size_t> &) const;

   double calculate_batch_error_cuda(const Vector<size_t>&, const MultilayerPerceptron::Pointers&) const;

   // Gradient methods

   Vector<double> calculate_training_error_gradient() const;

   FirstOrderLoss calculate_first_order_loss() const;

   Vector<double> calculate_batch_error_gradient(const Vector<size_t>&) const;

   Vector<double> calculate_batch_error_gradient_cuda(const Vector<size_t>&, const MultilayerPerceptron::Pointers&) const;

   FirstOrderLoss calculate_batch_first_order_loss(const Vector<size_t>&) const;

   FirstOrderLoss calculate_batch_first_order_loss_cuda(const Vector<size_t>&, const MultilayerPerceptron::Pointers&) const;

   FirstOrderLoss calculate_batch_first_order_loss_cuda(const Vector<size_t>&,
                                                        const MultilayerPerceptron::Pointers&, const Vector<double*>&) const;
   // Error terms methods

   Vector<double> calculate_error_terms(const Matrix<double>&, const Matrix<double>&) const;
   Vector<double> calculate_error_terms(const Vector<double>&) const;

   string write_error_term_type() const;

   Matrix<double> calculate_output_gradient(const Matrix<double>&, const Matrix<double>&) const;

   LossIndex::SecondOrderLoss calculate_terms_second_order_loss() const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML() const;   

   void write_XML(tinyxml2::XMLPrinter &) const;
   // void read_XML(   );

private:

};

}

#endif


// OpenNN: Open Neural Networks Library.
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
