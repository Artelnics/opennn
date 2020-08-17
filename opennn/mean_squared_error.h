//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E A N   S Q U A R E D   E R R O R    C L A S S   H E A D E R        
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef MEANSQUAREDERROR_H
#define MEANSQUAREDERROR_H

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

/// This class represents the mean squared error term.

///
/// The mean squared error measures the difference between the outputs from a neural network and the targets in a data set. 
/// This functional is used in data modeling problems, such as function regression, 
/// classification and time series prediction.

class MeanSquaredError : public LossIndex
{

public:

   // DEFAULT CONSTRUCTOR

   explicit MeanSquaredError();
   
   explicit MeanSquaredError(NeuralNetwork*, DataSet*);

   // Destructor

   virtual ~MeanSquaredError(); 

   // Error methods

   void calculate_error(const DataSet::Batch& batch,
                        const NeuralNetwork::ForwardPropagation& forward_propagation,
                        LossIndex::BackPropagation& back_propagation) const;

   void calculate_error_terms(const DataSet::Batch&,
                              const NeuralNetwork::ForwardPropagation&,
                              SecondOrderLoss&) const;

   // Gradient methods

   void calculate_output_gradient(const DataSet::Batch&,
                                  const NeuralNetwork::ForwardPropagation&,
                                  BackPropagation&) const;

   void calculate_Jacobian_gradient(const DataSet::Batch&,
                                    LossIndex::SecondOrderLoss&) const;

   // Hessian method

   void calculate_hessian_approximation(const DataSet::Batch& batch, LossIndex::SecondOrderLoss& second_order_loss) const;

   // Serialization methods

      

   void write_XML(tinyxml2::XMLPrinter &) const;

   string get_error_type() const;
   string get_error_type_text() const;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/mean_squared_error_cuda.h"
#endif

#ifdef OPENNN_MKL
    #include "../../opennn-mkl/opennn_mkl/mean_squared_error_mkl.h"
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
