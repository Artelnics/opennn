//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LONGSHORTTERMMEMORYLAYERTEST_H
#define LONGSHORTTERMMEMORYLAYERTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"
#include "../opennn/long_short_term_memory_layer.h"

namespace opennn
{

class LongShortTermMemoryLayerTest : public UnitTesting
{

public:

    explicit LongShortTermMemoryLayerTest();

    void test_constructor();

    // Forward propagate

    void test_forward_propagate();

    // Unit testing

    void run_test_case();

private:

    LongShortTermMemoryLayer long_short_term_memory_layer;

    LongShortTermMemoryLayerForwardPropagation long_short_term_layer_forward_propagation;

    Index inputs_number = 0;
    Index neurons_number = 0;
    Index time_steps = 0;

    Tensor<type, 2> synaptic_weights;
    Tensor<type, 2> recurrent_initializer;
    Tensor<type, 1> biases;

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
