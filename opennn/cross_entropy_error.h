//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CROSSENTROPYERROR_H
#define CROSSENTROPYERROR_H

// System includes

#include <iostream>
#include <fstream>
#include <math.h>

// OpenNN includes

#include "loss_index.h"
#include "data_set.h"
#include "config.h"

namespace OpenNN
{

/// This class represents the cross entropy error term, used for predicting probabilities.

///
/// This functional is used in classification problems.

class CrossEntropyError : public LossIndex
{

public:

   // Constructors

   explicit CrossEntropyError();

   explicit CrossEntropyError(NeuralNetwork*, DataSet*);

   // Destructor

   virtual ~CrossEntropyError();

   // Error methods

   void calculate_error(const DataSet::Batch& batch,
                        const NeuralNetwork::ForwardPropagation& forward_propagation,
                        LossIndex::BackPropagation& back_propagation) const;

   void calculate_binary_error(const DataSet::Batch& batch,
                        const NeuralNetwork::ForwardPropagation& forward_propagation,
                        LossIndex::BackPropagation& back_propagation) const;

   void calculate_multiple_error(const DataSet::Batch& batch,
                        const NeuralNetwork::ForwardPropagation& forward_propagation,
                        LossIndex::BackPropagation& back_propagation) const;

   // Gradient methods

   void calculate_output_gradient(const DataSet::Batch& batch,
                                  const NeuralNetwork::ForwardPropagation& forward_propagation,
                                  BackPropagation& back_propagation) const;

   void calculate_binary_output_gradient(const DataSet::Batch& batch,
                                         const NeuralNetwork::ForwardPropagation& forward_propagation,
                                         BackPropagation& back_propagation) const;

   void calculate_multiple_output_gradient(const DataSet::Batch& batch,
                                           const NeuralNetwork::ForwardPropagation& forward_propagation,
                                           BackPropagation& back_propagation) const;

   string get_error_type() const;
   string get_error_type_text() const;

   // Serialization methods

      
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn_cuda/cross_entropy_error_cuda.h"
#endif

#ifdef OPENNN_MKL
    #include "../../opennn-mkl/opennn_mkl/cross_entropy_error_mkl.h"
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
