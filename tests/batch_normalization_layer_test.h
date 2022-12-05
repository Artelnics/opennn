//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   N O R M A L I Z A T I O N   L A Y E R   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef BATCHNORMALIZATIONLAYERTEST_H
#define BATCHNORMALIZATIONLAYERTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"

class BatchNormalizationLayerTest : public UnitTesting
{

public:

    explicit BatchNormalizationLayerTest();

    virtual ~BatchNormalizationLayerTest();

//    // Constructor and destructor methods

    void test_perform_inputs_normalization();

    void run_test_case();

private:

//    Index inputs_number;
//    Index neurons_number;
//    Index samples_number;

//    PerceptronLayer perceptron_layer;

//    PerceptronLayerForwardPropagation forward_propagation;
//    PerceptronLayerBackPropagation back_propagation;

//    NumericalDifferentiation numerical_differentiation;
};


#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2022 Artificial Intelligence Techniques, SL.
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
