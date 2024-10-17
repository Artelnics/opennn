//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   T E S T   C L A S S   H E A D E R       
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NEURALNETWORKTEST_H
#define NEURALNETWORKTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"
#include "../opennn/data_set.h"
#include "../opennn/batch.h"
#include "../opennn/neural_network.h"

namespace opennn
{

class NeuralNetworkTest : public UnitTesting
{

public:

    explicit NeuralNetworkTest();

    void test_constructor();

    void test_add_layer();

    void test_calculate_outputs();

    void test_calculate_directional_inputs();

    void test_forward_propagate();

    void test_save();
    void test_load();

    void run_test_case();

private:

    Index inputs_number;
    Index outputs_number;
    Index batch_samples_number;

    Tensor<type, 2> data;

    DataSet data_set;

    Batch batch;

    Tensor<Index,1> training_samples_indices;
    Tensor<Index,1> input_variables_indices;
    Tensor<Index,1> target_variables_indices;

    NeuralNetwork neural_network;
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
