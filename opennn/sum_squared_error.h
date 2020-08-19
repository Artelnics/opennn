//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S U M   S Q U A R E D   E R R O R   C L A S S   H E A D E R           
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef SUMSQUAREDERROR_H
#define SUMSQUAREDERROR_H

// System includes

#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <string>
#include <limits>

// OpenNN includes

#include "config.h"

#include "loss_index.h"
#include "data_set.h"

namespace OpenNN
{

/// This class represents the sum squared peformance term functional. 

///
/// This is used as the error term in data modeling problems, such as function regression, 
/// classification or time series prediction.

class SumSquaredError : public LossIndex
{

public:

   // DEFAULT CONSTRUCTOR

   explicit SumSquaredError();

   explicit SumSquaredError(NeuralNetwork*, DataSet*);   

   virtual ~SumSquaredError();

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

   void calculate_hessian_approximation(const DataSet::Batch& batch, LossIndex::SecondOrderLoss& second_order_loss) const;

   // Serialization methods

   string get_error_type() const;
   string get_error_type_text() const;

      
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

private:

   // Squared errors methods

   Tensor<type, 1> calculate_squared_errors() const;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/sum_squared_error_cuda.h"
#endif

#ifdef OPENNN_MKL
    #include "../../opennn-mkl/opennn_mkl/sum_squared_error_mkl.h"
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
