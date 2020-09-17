//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   W E I G H T E D   S Q U A R E D   E R R O R    C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef WEIGHTEDSQUAREDERROR_H
#define WEIGHTEDSQUAREDERROR_H

// System includes

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <limits>
#include <math.h>

// OpenNN includes

#include "config.h"
#include "loss_index.h"
#include "data_set.h"


namespace OpenNN
{

/// This class represents the weighted squared error term.

///
/// The weighted squared error measures the difference between the outputs from a neural network and the targets in a data set.
/// This functional is used in data modeling problems, such as function regression, 
/// classification and time series prediction.

class WeightedSquaredError : public LossIndex
{

public:

   // Constructors

   explicit WeightedSquaredError();

   explicit WeightedSquaredError(NeuralNetwork*, DataSet*); 

   // Destructor

   virtual ~WeightedSquaredError(); 

   // Get methods

   type get_positives_weight() const;
   type get_negatives_weight() const;

   type get_normalizaton_coefficient() const;

   // Set methods

   void set_default();

   void set_positives_weight(const type&);
   void set_negatives_weight(const type&);

   void set_weights(const type&, const type&);

   void set_weights();

   void set_normalization_coefficient();

   void set_data_set_pointer(DataSet* new_data_set_pointer);

   // Error methods

   type weighted_sum_squared_error(const Tensor<type, 2>&, const Tensor<type, 2>& ) const;

   void calculate_error(const DataSet::Batch& batch,
                        const NeuralNetwork::ForwardPropagation& forward_propagation,
                        LossIndex::BackPropagation& back_propagation) const;

   void calculate_error_terms(const DataSet::Batch&,
                              const NeuralNetwork::ForwardPropagation&,
                              SecondOrderLoss&) const;

   string get_error_type() const;
   string get_error_type_text() const;

   // Gradient methods

   void calculate_output_gradient(const DataSet::Batch& batch,
                                  const NeuralNetwork::ForwardPropagation& forward_propagation,
                                  BackPropagation& back_propagation) const;

   void calculate_Jacobian_gradient(const DataSet::Batch&,
                                    LossIndex::SecondOrderLoss&) const;

   // Hessian method

   void calculate_hessian_approximation(const DataSet::Batch&,
                                        LossIndex::SecondOrderLoss&) const;

   // Serialization methods

      
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;   

private:

   /// Weight for the positives for the calculation of the error.

   type positives_weight;

   /// Weight for the negatives for the calculation of the error.

   type negatives_weight;

   /// Coefficient of normalization

   type normalization_coefficient;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/weighted_squared_error_cuda.h"
#endif

#ifdef OPENNN_MKL
    #include "../../opennn-mkl/opennn_mkl/weighted_squared_error_mkl.h"
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
