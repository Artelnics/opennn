//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   3 D   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CrossEntropyError3DTest_H
#define CrossEntropyError3DTest_H

// Unit testing includes

#include "../opennn/unit_testing.h"
#include "../opennn/neural_network_forward_propagation.h"
#include "../opennn/loss_index_back_propagation.h"

class CrossEntropyError3DTest : public UnitTesting
{

public:

    explicit CrossEntropyError3DTest();

    virtual ~CrossEntropyError3DTest();

    void test_constructor();

    void test_destructor();

    // Back-propagation methods

    void test_back_propagate();

    // Transformer gradient method

    void test_calculate_gradient_transformer();

    // Unit testing methods

    void run_test_case();

private:

    Index samples_number = 0;
    Index inputs_number = 0;
    Index inputs_dim = 0;
    Index outputs_dim = 0;

    Tensor<type, 3> data;

    Tensor<Index, 1> training_samples_indices;
    Tensor<Index, 1> input_variables_indices;
    Tensor<Index, 1> target_variables_indices;

    DataSet data_set;

    NeuralNetwork neural_network;

    CrossEntropyError3D cross_entropy_error_3d;

    DataSetBatch batch;

    ForwardPropagation forward_propagation;

    BackPropagation back_propagation;

    Tensor<type, 1> numerical_gradient;
    Tensor<type, 2> numerical_jacobian;

};

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
