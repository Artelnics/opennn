//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PROBABILISTICLAYERTEST_H
#define PROBABILISTICLAYERTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"

class ProbabilisticLayerTest : public UnitTesting
{

public:

    explicit ProbabilisticLayerTest();

    virtual ~ProbabilisticLayerTest();

    // Constructor and destructor methods

    void test_constructor();

    void test_destructor();

    // Set methods

    void test_set();
    void test_set_default();
    void test_set_biases();
    void test_set_synaptic_weights();
    void test_set_parameters();
    void test_set_decision_threshold();

    // Activation function

    void test_set_activation_function();

    void test_calculate_combinations();
    void test_calculate_activations();
    void test_calculate_activations_derivatives();

    // Forward propagate

    void test_forward_propagate();

    // Serialization methods

    void test_to_XML();
    void test_from_XML();

    // Unit testing methods

    void run_test_case();

private:

    Index inputs_number;
    Index outputs_number;
    Index neurons_number;
    Index samples_number;

    ProbabilisticLayer probabilistic_layer;

    ProbabilisticLayerForwardPropagation probabilistic_layer_forward_propagation;

    NumericalDifferentiation numerical_differentiation;
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
