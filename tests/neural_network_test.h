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

class NeuralNetworkTest : public UnitTesting
{

public:

    explicit NeuralNetworkTest();

    virtual ~NeuralNetworkTest();

    // Constructor and destructor methods

    void test_constructor();

    void test_destructor();

    // Appending layers

    void test_add_layer();
    void check_layer_type();

    // Get methods

    void test_has_methods();

    // Set methods

    void test_set();

    void test_set_names();
    void test_set_inputs_number();

    void test_set_pointers();

    void test_set_default();

    // Architecture

    void test_set_network();

    void test_set_parameters();

    // Parameters initialization methods

    void test_set_parameters_constant();
    void test_set_parameters_random();

    // Parameters norm / descriptives / histogram

    void test_calculate_parameters_norm();
    void test_calculate_parameters_descriptives();
    void test_calculate_parameters_histogram();

    void test_perturbate_parameters();

    // Output

    void test_calculate_trainable_outputs();
    void test_calculate_outputs();

    void test_calculate_directional_inputs();
    void test_calculate_outputs_histograms();

    // Forward propagation

    void test_forward_propagate();

    // Expression methods

    void test_save_expression();

    // Serialization methods

    void test_to_XML();
    void test_from_XML();

    void test_save();
    void test_load();

    // Unit testing methods

    void run_test_case();

private:

    Index inputs_number;
    Index outputs_number;
    Index batch_samples_number;

    Tensor<type, 2> data;

    DataSet data_set;

    DataSetBatch batch;

    Tensor<Index,1> training_samples_indices;
    Tensor<Index,1> input_variables_indices;
    Tensor<Index,1> target_variables_indices;

    NeuralNetwork neural_network;
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
