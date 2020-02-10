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
#include "tinyxml2.h"

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

   explicit WeightedSquaredError(NeuralNetwork*);

   explicit WeightedSquaredError(DataSet*);

   explicit WeightedSquaredError(NeuralNetwork*, DataSet*); 

   explicit WeightedSquaredError(const tinyxml2::XMLDocument&);

   WeightedSquaredError(const WeightedSquaredError&);

   // Destructor

   virtual ~WeightedSquaredError(); 

   // Get methods

   type get_positives_weight() const;
   type get_negatives_weight() const;

   type get_training_normalization_coefficient() const;
   type get_selection_normalization_coefficient() const;

   // Set methods

   // Error methods

   void set_default();

   void set_positives_weight(const type&);
   void set_negatives_weight(const type&);

   void set_training_normalization_coefficient(const type&);
   void set_selection_normalization_coefficient(const type&);

   void set_weights(const type&, const type&);

   void set_weights();

   void set_training_normalization_coefficient();
   void set_selection_normalization_coefficient();

   Tensor<type, 1> calculate_training_error_gradient() const;

   void calculate_output_gradient(const NeuralNetwork::ForwardPropagation& forward_propagation,
                                  BackPropagation& back_propagation) const
   {
        #ifdef __OPENNN_DEBUG__

        check();

        #endif

        const Index trainable_layers_number = neural_network_pointer->get_trainable_layers_number();
/*
        back_propagation.output_gradient = (forward_propagation.layers[trainable_layers_number-1].activations-batch.targets_2d)
                *((batch.targets_2d-static_cast<type>(1.0))*(static_cast<type>(-1.0))*negatives_weight + batch.targets_2d*positives_weight);
*/
   }


   // Error terms methods

   Tensor<type, 1> calculate_training_error_terms(const Tensor<type, 1>&) const;
   Tensor<type, 1> calculate_training_error_terms(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   LossIndex::SecondOrderLoss calculate_terms_second_order_loss() const;

   string get_error_type() const;
   string get_error_type_text() const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML() const;   
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;

   string object_to_string() const;

private:

   /// Weight for the positives for the calculation of the error.

   type positives_weight;

   /// Weight for the negatives for the calculation of the error.

   type negatives_weight;

   /// Coefficient of normalization for the calculation of the training error.

   type training_normalization_coefficient;

   /// Coefficient of normalization for the calculation of the selection error.

   type selection_normalization_coefficient;
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
