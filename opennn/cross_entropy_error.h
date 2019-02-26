/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   C R O S S   E N T R O P Y   E R R O R   C L A S S   H E A D E R                                            */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __CROSSENTROPYERROR_H__
#define __CROSSENTROPYERROR_H__

// System includes

#include <iostream>
#include <fstream>
#include <math.h>

// OpenNN includes

#include "loss_index.h"
#include "data_set.h"

// TinyXml includes

#include "tinyxml2.h"


namespace OpenNN
{

/// This class represents the cross entropy error term. 
/// This functional is used in classification problems.

class CrossEntropyError : public LossIndex
{

public:

   // DEFAULT CONSTRUCTOR

   explicit CrossEntropyError();

   // NEURAL NETWORK CONSTRUCTOR

   explicit CrossEntropyError(NeuralNetwork*);

   // DATA SET CONSTRUCTOR

   explicit CrossEntropyError(DataSet*);

   // GENERAL CONSTRUCTOR

   explicit CrossEntropyError(NeuralNetwork*, DataSet*);

   // XML CONSTRUCTOR

   explicit CrossEntropyError(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   CrossEntropyError(const CrossEntropyError&);

   // DESTRUCTOR

   virtual ~CrossEntropyError();

   // ASSIGNMENT OPERATOR

   CrossEntropyError& operator = (const CrossEntropyError&);

   // EQUAL TO OPERATOR

   bool operator == (const CrossEntropyError&) const;

   // METHODS

   // Error methods

   double calculate_training_error() const;

   double calculate_selection_error() const;

   double calculate_training_error(const Vector<double>&) const;

   double calculate_batch_error(const Vector<size_t> &) const;

   double calculate_batch_error_cuda(const Vector<size_t>&, const MultilayerPerceptron::Pointers&) const;

   // Gradient methods

   Vector<double> calculate_training_error_gradient() const;

   FirstOrderLoss calculate_batch_first_order_loss_cuda(const Vector<size_t>&,
                                                        const MultilayerPerceptron::Pointers&, const Vector<double*>&) const;

   Matrix<double> calculate_output_gradient(const Matrix<double>&, const Matrix<double>&) const;

   string write_error_term_type() const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML() const;   
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
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
