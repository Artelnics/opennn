//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q U A S I - N E W T O N   M E T H O D   T E S T   C L A S S   H E A D E R      
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef QUASINEWTONMETHODTEST_H
#define QUASINEWTONMETHODTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"
#include "../opennn/sum_squared_error.h"
#include "../opennn/quasi_newton_method.h"

namespace opennn
{

class QuasiNewtonMethodTest : public UnitTesting 
{

public:

    explicit QuasiNewtonMethodTest();

    virtual ~QuasiNewtonMethodTest();

    // Constructor and destructor

    void test_constructor();

    void test_destructor();

    // Training

    void test_calculate_DFP_inverse_hessian_approximation();
    void test_calculate_BFGS_inverse_hessian_approximation();

    void test_calculate_inverse_hessian_approximation();

    void test_perform_training();

    // Unit testing

    void run_test_case();

private:

    Index samples_number;
    Index inputs_number;
    Index outputs_number;
    Index neurons_number;

    DataSet data_set;

    NeuralNetwork neural_network;

    SumSquaredError sum_squared_error;

    QuasiNewtonMethod quasi_newton_method;

    QuasiNewtonMehtodData quasi_newton_method_data;

    TrainingResults training_results;

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
