//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   N O R M A L I Z A T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef BATCHNORMALIZATIONLAYER_H
#define BATCHNORMALIZATIONLAYER_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
#include "layer.h"
#include "opennn_strings.h"

namespace opennn
{
struct BatchNormalizationLayerForwardPropagation;

class BatchNormalizationLayer : public Layer

{

public:
   // Constructors

   explicit BatchNormalizationLayer();

   explicit BatchNormalizationLayer(const Index&);

   // Get methods

    Index get_inputs_number() const;

   // Parameters

   // Display messages

   // Set methods

   void set(const Index&);

   void set_default();

   // Architecture

   // Parameters

   // Display messages

   void set_parameters_random();

   // Perceptron layer combinations

   void calculate_combinations(const Tensor<type, 2>&,
                               const Tensor<type, 2>&,
                               Tensor<type, 2>&);

    void perform_normalization(const Tensor<type, 2>&, BatchNormalizationLayerForwardPropagation*)const;
   // Perceptron layer outputs

   void forward_propagate(type*, const Tensor<Index, 1>&,
                          LayerForwardPropagation*);

   // Gradient methods

   // Serialization methods

protected:

   // MEMBERS

   /// Inputs

   Tensor<type, 2> inputs;

   /// Fixed parameters

//   Tensor<type, 2> mean;
//   Tensor<type, 2> std;

   /// Outputs

   Tensor<type, 2> outputs;

   /// learneable_parameters

   Tensor<type, 2> normalization_weights;

   /// Display messages to screen. 

   bool display = true;
};


struct BatchNormalizationLayerForwardPropagation : LayerForwardPropagation
{
    // Default constructor

    explicit BatchNormalizationLayerForwardPropagation() : LayerForwardPropagation()
    {
    }

    explicit BatchNormalizationLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }

    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        layer_pointer = new_layer_pointer;

        batch_samples_number = new_batch_samples_number;

        const Index inputs_number = layer_pointer->get_inputs_number();

        // Outputs

        outputs_dimensions.resize(2);
        outputs_dimensions.setValues({batch_samples_number, inputs_number});
        outputs_data = (type*) malloc( static_cast<size_t>(batch_samples_number * inputs_number*sizeof(type)) );

        // fixed parameters

        mean.resize(inputs_number);
        variance.resize(inputs_number);
    }

    Tensor<type,1> mean;
    Tensor<type,1> variance;
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
