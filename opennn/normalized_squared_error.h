/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   N O R M A L I Z E D   S Q U A R E D   E R R O R   C L A S S   H E A D E R                                  */
/*                                                                                                              */

/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __NORMALIZEDSQUAREDERROR_H__
#define __NORMALIZEDSQUAREDERROR_H__

// System includes

#include <string>
#include <sstream>

#include <iostream>
#include <fstream>
#include <limits>
#include <cmath>

// OpenNN includes

#include "loss_index.h"
#include "data_set.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

/// This class represents the normalized squared error term. 
/// This error term is used in data modeling problems.
/// If it has a value of unity then the neural network is predicting the data "in the mean",
/// A value of zero means perfect prediction of data.

class NormalizedSquaredError : public LossIndex
{

public:

   // GENERAL CONSTRUCTOR

   explicit NormalizedSquaredError(NeuralNetwork*, DataSet*);

   // NEURAL NETWORK CONSTRUCTOR

   explicit NormalizedSquaredError(NeuralNetwork*);

   // DATA SET CONSTRUCTOR

   explicit NormalizedSquaredError(DataSet*);

   // DEFAULT CONSTRUCTOR

   explicit NormalizedSquaredError();

   // XML CONSTRUCTOR

   explicit NormalizedSquaredError(const tinyxml2::XMLDocument&);

   // DESTRUCTOR

   virtual ~NormalizedSquaredError();

   // METHODS

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

   double calculate_selection_error() const;

   double calculate_training_error(const Vector<double>&) const;

   double calculate_batch_error(const Vector<size_t> &) const;

   double calculate_error(const Matrix<double>&, const Matrix<double>&) const;
   double calculate_error(const Vector<size_t>&, const Vector<double>&) const;

   // Gradient methods

   Vector<double> calculate_training_error_gradient() const;

   Matrix<double> calculate_output_gradient(const Matrix<double>&, const Matrix<double>&) const;

   LossIndex::FirstOrderLoss calculate_first_order_loss() const;

   Vector<double> calculate_batch_error_gradient(const Vector<size_t>&) const;

   LossIndex::FirstOrderLoss calculate_batch_first_order_loss(const Vector<size_t>&) const;

   LossIndex::FirstOrderLoss calculate_batch_first_order_loss_cuda(const Vector<size_t>&, const MultilayerPerceptron::Pointers&) const;

   LossIndex::FirstOrderLoss calculate_batch_first_order_loss_cuda(const Vector<size_t>&,
                                                                   const MultilayerPerceptron::Pointers&, const Vector<double*>&) const;

   // Error terms methods

   Vector<double> calculate_error_terms(const Matrix<double>&, const Matrix<double>&) const;
   Vector<double> calculate_error_terms(const Vector<double>&) const;

   // Squared errors methods

   Vector<double> calculate_squared_errors() const;

   Vector<size_t> calculate_maximal_errors(const size_t& = 10) const;

   LossIndex::SecondOrderLoss calculate_terms_second_order_loss() const;

   string write_error_term_type() const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML() const;   
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

private:

   // METHODS


   // MEMBERS

   /// Coefficient of normalization for the calculation of the training error.

   double normalization_coefficient;
   double selection_normalization_coefficient;
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

