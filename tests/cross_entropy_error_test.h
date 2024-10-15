//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   T E S T   C L A S S   H E A D E R      
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CROSSENTROPYERRORTEST_H
#define CROSSENTROPYERRORTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/neural_network_forward_propagation.h"
#include "../opennn/back_propagation.h"
#include "../opennn/cross_entropy_error.h"
#include "../opennn/image_data_set.h"

namespace opennn
{

class CrossEntropyErrorTest : public UnitTesting
{

public:

    explicit CrossEntropyErrorTest();

    void test_constructor();

    void test_back_propagate();

    void run_test_case();

private:

   Index samples_number;
   Index inputs_number;
   Index outputs_number;
   Index neurons_number;

   Tensor<type, 2> data;

   Tensor<Index, 1> training_samples_indices;
   Tensor<Index, 1> input_variables_indices;
   Tensor<Index, 1> target_variables_indices;

   DataSet data_set;
   ImageDataSet image_data_set;

   NeuralNetwork neural_network;

   CrossEntropyError cross_entropy_error;

   Batch batch;

   ForwardPropagation forward_propagation;

   BackPropagation back_propagation;

   Tensor<type, 1> numerical_gradient;
   Tensor<type, 2> numerical_jacobian;

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
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
