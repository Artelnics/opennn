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

namespace opennn
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

   // Back propagation

   void calculate_error(const DataSetBatch&,
                        const NeuralNetworkForwardPropagation&,
                        LossIndexBackPropagation&) const override;

   void calculate_output_delta(const DataSetBatch&,
                               NeuralNetworkForwardPropagation&,
                               LossIndexBackPropagation&) const final;

   // Back propagation LM

   void calculate_error_lm(const DataSetBatch&,
                           const NeuralNetworkForwardPropagation&,
                           LossIndexBackPropagationLM&) const final;

   void calculate_output_delta_lm(const DataSetBatch&,
                                  NeuralNetworkForwardPropagation&,
                                  LossIndexBackPropagationLM&) const final;

   void calculate_error_gradient_lm(const DataSetBatch&,
                              LossIndexBackPropagationLM&) const final;

   void calculate_error_hessian_lm(const DataSetBatch&,
                                        LossIndexBackPropagationLM&) const final;

   // Serialization methods

   void write_XML(tinyxml2::XMLPrinter &) const final;

   string get_error_type() const final;
   string get_error_type_text() const final;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/mean_squared_error_cuda.h"
#endif

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
