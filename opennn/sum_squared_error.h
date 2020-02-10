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

#include "tinyxml2.h"

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

   // NEURAL NETWORK CONSTRUCTOR

   explicit SumSquaredError(NeuralNetwork*);

   // DATA SET CONSTRUCTOR

   explicit SumSquaredError(DataSet*);

   explicit SumSquaredError(NeuralNetwork*, DataSet*);   

   explicit SumSquaredError(const tinyxml2::XMLDocument&);

   // COPY CONSTRUCTOR

   SumSquaredError(const SumSquaredError&);

   virtual ~SumSquaredError();    

   // Error methods

   
   

   // Gradient methods

   Tensor<type, 1> calculate_training_error_terms(const Tensor<type, 1>&) const;
   Tensor<type, 1> calculate_training_error_terms(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   // Serialization methods

   string get_error_type() const;
   string get_error_type_text() const;

   tinyxml2::XMLDocument* to_XML() const;   
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

   void calculate_output_gradient(const NeuralNetwork::ForwardPropagation& forward_propagation,
                                  BackPropagation& back_propagation) const
   {
        #ifdef __OPENNN_DEBUG__

        check();

        #endif

        const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();
/*
        back_propagation.output_gradient = forward_propagation.layers[trainable_layers_number-1].activations;

        back_propagation.output_gradient -= batch.targets_2d;

        back_propagation.output_gradient *= 2.0;
*/
   }

   LossIndex::SecondOrderLoss calculate_terms_second_order_loss() const;

private:

   // Squared errors methods

   Tensor<type, 1> calculate_squared_errors() const;
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
