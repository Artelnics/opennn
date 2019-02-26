/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S U M   S Q U A R E D   E R R O R   C L A S S   H E A D E R                                                */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#pragma once
#ifndef __SUMSQUAREDERROR_H__
#define __SUMSQUAREDERROR_H__

// System includes

#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <string>
#include <limits>

// OpenNN includes

#include "loss_index.h"
#include "data_set.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

/// This class represents the sum squared peformance term functional. 
/// This is used as the error term in data modeling problems, such as function regression, 
/// classification or time series prediction.

class SumSquaredError : public LossIndex
{

public:

   // DEFAULT CONSTRUCTOR

   explicit SumSquaredError();

   // NEURAL NETWORK CONSTRUCTOR

   explicit SumSquaredError(NeuralNetwork*);

   // DATA SET CONSTRUCTOR

   explicit SumSquaredError(DataSet*);

   // GENERAL CONSTRUCTOR

   explicit SumSquaredError(NeuralNetwork*, DataSet*);

   // XML CONSTRUCTOR

   explicit SumSquaredError(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   SumSquaredError(const SumSquaredError&);

   // DESTRUCTOR

   virtual ~SumSquaredError();    

   // METHODS

   // Error methods

   double calculate_training_error() const;

   double calculate_selection_error() const;

   double calculate_training_error(const Vector<double>&) const;

   Vector<double> calculate_training_error_gradient() const;

   double calculate_batch_error(const Vector<size_t>&) const;

   double calculate_batch_error_cuda(const Vector<size_t>&, const MultilayerPerceptron::Pointers&) const;

   double calculate_error(const Matrix<double>&, const Matrix<double>&) const;

   double calculate_error(const Vector<size_t>&, const Vector<double>&) const;

   // Gradient methods

   Vector<double> calculate_batch_error_gradient(const Vector<size_t>&) const;

   Vector<double> calculate_batch_error_gradient_cuda(const Vector<size_t>&, const MultilayerPerceptron::Pointers&) const;

   LossIndex::FirstOrderLoss calculate_first_order_loss() const;
   LossIndex::FirstOrderLoss calculate_batch_first_order_loss(const Vector<size_t>&) const;

   LossIndex::FirstOrderLoss calculate_batch_first_order_loss_cuda(const Vector<size_t>&, const MultilayerPerceptron::Pointers&) const;

   LossIndex::FirstOrderLoss calculate_batch_first_order_loss_cuda(const Vector<size_t>&,
                                                                   const MultilayerPerceptron::Pointers&, const Vector<double*>&) const;

   // Terms methods

   Vector<double> calculate_error_terms(const Vector<double>&) const;
   Vector<double> calculate_error_terms(const Matrix<double>&, const Matrix<double>&) const;

   // Serialization methods

   string write_error_term_type() const;

   tinyxml2::XMLDocument* to_XML() const;   
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

   Matrix<double> calculate_output_gradient(const Matrix<double>&, const Matrix<double>&) const;

   LossIndex::SecondOrderLoss calculate_terms_second_order_loss() const;

private:

   // Squared errors methods

   Vector<double> calculate_squared_errors() const;
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
