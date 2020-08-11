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

#include "config.h"
#include "loss_index.h"
#include "data_set.h"

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

    // Constructors

   explicit NormalizedSquaredError(NeuralNetwork*, DataSet*);

   explicit NormalizedSquaredError();   

    // Destructor

   virtual ~NormalizedSquaredError();

   // Get methods

    type get_normalization_coefficient() const;
    type get_selection_normalization_coefficient() const;

   // Set methods

    void set_normalization_coefficient();
    void set_normalization_coefficient(const type&);

    void set_selection_normalization_coefficient();
    void set_selection_normalization_coefficient(const type&);

    void set_default();

    void set_data_set_pointer(DataSet* new_data_set_pointer);

   // Normalization coefficients 

   type calculate_normalization_coefficient(const Tensor<type, 2>&, const Tensor<type, 1>&) const;

   // Error methods
     
   void calculate_error(const DataSet::Batch& batch,
                        const NeuralNetwork::ForwardPropagation& forward_propagation,
                        LossIndex::BackPropagation& back_propagation) const;

   void calculate_error_terms(const DataSet::Batch&,
                              const NeuralNetwork::ForwardPropagation&,
                              SecondOrderLoss&) const;

   // Gradient methods

   void calculate_output_gradient(const DataSet::Batch& batch,
                                  const NeuralNetwork::ForwardPropagation& forward_propagation,
                                  BackPropagation& back_propagation) const;


   void calculate_Jacobian_gradient(const DataSet::Batch& batch,
                                       LossIndex::SecondOrderLoss& second_order_loss) const;

   // Hessian method

   void calculate_hessian_approximation(const DataSet::Batch&,
                                        LossIndex::SecondOrderLoss&) const;


   // Serialization methods

   string get_error_type() const;
   string get_error_type_text() const;

   
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

private:

   /// Coefficient of normalization for the calculation of the training error.

   type normalization_coefficient;

   type selection_normalization_coefficient;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/normalized_squared_error_cuda.h"
#endif


#ifdef OPENNN_MKL
    #include "../../opennn-mkl/opennn_mkl/normalized_squared_error_mkl.h"
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

