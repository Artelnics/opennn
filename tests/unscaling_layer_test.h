//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R   T E S T   C L A S S   H E A D E R     
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef UNSCALINGLAYERTEST_H
#define UNSCALINGLAYERTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"

class UnscalingLayerTest : public UnitTesting
{

public:

    explicit UnscalingLayerTest();

    virtual ~UnscalingLayerTest();

    // Constructor and destructor methods

    void test_constructor();

    void test_destructor();

    // Set methods

    void test_set();
    void test_set_inputs_number();
    void test_set_neurons_number();
    void test_set_default();

    // Output variables descriptives

    void test_set_descriptives();
    void test_set_item_descriptives();

    // Variables scaling and unscaling

    void test_set_unscaling_method();

    // Check methods

    void test_is_empty();

    // Outputs unscaling

    void test_calculate_minimum_maximum_outputs();
    void test_calculate_mean_standard_deviation_outputs();
    void test_calculate_logarithmic_outputs();

    // Serialization methods

    void test_to_XML();
    void test_from_XML();

    // Unit testing methods

    void run_test_case();

private:

    UnscalingLayer unscaling_layer;

    UnscalingLayerForwardPropagation unscaling_layer_forward_propagation;

    Tensor<Descriptives, 1> descriptives;

    Index samples_number;
    Index inputs_number;
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
