//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   3 D   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PROBABILISTICLAYER3DTEST_H
#define PROBABILISTICLAYER3DTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"
#include "../opennn/probabilistic_layer_3d.h"
#include "../opennn/numerical_differentiation.h"

namespace opennn
{

class ProbabilisticLayer3DTest : public UnitTesting
{

public:

    explicit ProbabilisticLayer3DTest();

    virtual ~ProbabilisticLayer3DTest();

    // Constructor and destructor

    void test_constructor();

    void test_destructor();

    // Forward propagate

    void test_calculate_combinations();
    void test_calculate_activations();
    void test_calculate_activations_derivatives();

    bool check_softmax_derivatives(Tensor<type, 3>&, Tensor<type, 4>&) const;

    void test_forward_propagate();

    // Unit testing

    void run_test_case();

private:

    Index inputs_number = 0;
    Index inputs_depth = 0;
    Index neurons_number = 0;
    Index samples_number = 0;

    ProbabilisticLayer3D probabilistic_layer_3d;

    ProbabilisticLayer3DForwardPropagation probabilistic_layer_3d_forward_propagation;

    NumericalDifferentiation numerical_differentiation;
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
