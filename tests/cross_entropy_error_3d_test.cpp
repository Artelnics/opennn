//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   E N T R O P Y   E R R O R   3 D   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cross_entropy_error_3d_test.h"

CrossEntropyError3DTest::CrossEntropyError3DTest() : UnitTesting()
{
    cross_entropy_error_3d.set(&neural_network, &data_set);
    cross_entropy_error_3d.set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
}


CrossEntropyError3DTest::~CrossEntropyError3DTest()
{
}


void CrossEntropyError3DTest::test_constructor()
{
    cout << "test_constructor\n";

    CrossEntropyError3D cross_entropy_error_3d;
}


void CrossEntropyError3DTest::test_destructor()
{
    cout << "test_destructor\n";

    CrossEntropyError3D* cross_entropy_error_3d = new CrossEntropyError3D;

    delete cross_entropy_error_3d;
}


void CrossEntropyError3DTest::test_back_propagate()
{
    cout << "test_back_propagate\n";

    // Empty test does not work
    // cross_entropy_error.back_propagate(batch, forward_propagation, back_propagation);

    /// @todo
}


void CrossEntropyError3DTest::test_calculate_gradient_transformer()
{
    cout << "Running calculate gradient transformer test case...\n";

    Tensor<type, 1> numerical_gradient;

    // Test
    {
        data_set.set();
        neural_network.set();

        numerical_gradient = cross_entropy_error_3d.calculate_numerical_gradient();

        assert_true(numerical_gradient.size() == 0, LOG);
    }

    // Test
    {
        samples_number = 1;
        inputs_number = 1;

        inputs_dim = 1;
        outputs_dim = 1;

        Tensor<type, 3> inputs(samples_number, inputs_number, inputs_dim);
        inputs.setConstant(0);

        Tensor<type, 3> targets(samples_number, inputs_number, outputs_dim);
        targets.setConstant(0);

        ProbabilisticLayer3D probabilistic_layer_3d(inputs_number, inputs_dim, outputs_dim);
        neural_network.add_layer(&probabilistic_layer_3d);

        cout << "Before calculate_numerical_gradient()" << endl;
        numerical_gradient = cross_entropy_error_3d.calculate_numerical_gradient(inputs, targets); // calculate_errors() assumes outputs and targets of range 2

        cout << numerical_gradient << endl;
    }

    cout << "End of calculate gradient transformer test case...\n";
}


void CrossEntropyError3DTest::run_test_case()
{
    cout << "Running cross-entropy error test case...\n";

    // Test constructor

    test_constructor();
    test_destructor();

    // Transformer test

    test_calculate_gradient_transformer();

    // Back-propagation methods

    test_back_propagate();

    cout << "End of cross-entropy error test case.\n\n";
}


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
