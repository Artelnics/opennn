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
#include "../opennn/probabilistic_layer.h"

namespace opennn
{

class ProbabilisticLayerTest : public UnitTesting
{

public:

    explicit ProbabilisticLayerTest();

    void test_constructor();

    void test_calculate_combinations();
    void test_calculate_activations();

    void test_forward_propagate();

    void run_test_case();

private:

    Index inputs_number = 0;
    Index outputs_number = 0;
    Index neurons_number = 0;
    Index samples_number = 0;

    Tensor<type, 2> inputs;
    Tensor<type, 2> outputs;
    Tensor<type, 2> combinations;
    Tensor<type, 2> activations;
    Tensor<type, 2> activations_derivatives;

    pair<type*, dimensions> input_pairs;

    bool is_training = true;

    ProbabilisticLayer probabilistic_layer;

    ProbabilisticLayerForwardPropagation probabilistic_layer_forward_propagation;
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
