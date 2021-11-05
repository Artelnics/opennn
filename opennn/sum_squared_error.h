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

   // Back propagation

   void calculate_error(const DataSetBatch&,
                        const NeuralNetworkForwardPropagation&,
                        LossIndexBackPropagation&) const;

   void calculate_output_delta(const DataSetBatch&,
                               NeuralNetworkForwardPropagation&,
                               LossIndexBackPropagation&) const;

   // Back propagation LM

   void calculate_error_lm(const DataSetBatch&,
                           const NeuralNetworkForwardPropagation&,
                           LossIndexBackPropagationLM&) const;

   void calculate_output_delta_lm(const DataSetBatch&,
                               NeuralNetworkForwardPropagation&,
                               LossIndexBackPropagationLM&) const;

   void calculate_error_gradient_lm(const DataSetBatch&,
                              LossIndexBackPropagationLM&) const;

   void calculate_error_hessian_lm(const DataSetBatch&,
                                        LossIndexBackPropagationLM&) const;

   // Serialization methods

   string get_error_type() const;
   string get_error_type_text() const;
      
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

private:

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/sum_squared_error_cuda.h"
#endif

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
