/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M I N K O W S K I   E R R O R   C L A S S   H E A D E R                                                    */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __MINKOWSKIERROR_H__
#define __MINKOWSKIERROR_H__

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath>

// OpenNN includes

#include "loss_index.h"
#include "data_set.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

/// This class represents the Minkowski error term. 
/// The Minkowski error measures the difference between the outputs of a neural network and the targets in a data set. 
/// This error term is used in data modeling problems.
/// It can be more useful when the data set presents outliers. 

class MinkowskiError : public LossIndex
{

public:

   // DEFAULT CONSTRUCTOR

   explicit MinkowskiError();

   // NEURAL NETWORK CONSTRUCTOR

   explicit MinkowskiError(NeuralNetwork*);

   // DATA SET CONSTRUCTOR

   explicit MinkowskiError(DataSet*);

   // NEURAL NETWORK AND DATA SET CONSTRUCTOR

   explicit MinkowskiError(NeuralNetwork*, DataSet*);

   // XML CONSTRUCTOR

   explicit MinkowskiError(const tinyxml2::XMLDocument&);

   // DESTRUCTOR

   virtual ~MinkowskiError();


   // METHODS

   // Get methods

   double get_Minkowski_parameter() const;

   // Set methods

   void set_default();

   void set_Minkowski_parameter(const double&);


   // loss methods

   double calculate_training_error() const
   {
       return 0;
   }

   double calculate_selection_error() const
   {
       return 0;
   }

   double calculate_training_error(const Vector<double>&) const
   {
       return 0;
   }

   double calculate_batch_error(const Vector<size_t> &) const
   {
       return 0;
   }

   Vector<double> calculate_training_error_gradient() const
   {
       return Vector<double>();
   }



   double calculate_error(const Matrix<double>&, const Matrix<double>&) const;

   double calculate_error(const Vector<double>&) const;

   string write_error_term_type() const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML() const;   
   void from_XML(const tinyxml2::XMLDocument&);   

   void write_XML(tinyxml2::XMLPrinter&) const;

private:

   Vector<double> calculate_output_gradient(const Vector<size_t>&, const Vector<double>&, const Vector<double>&) const;

   // MEMBERS

   /// Minkowski exponent value.

   double Minkowski_parameter;

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
