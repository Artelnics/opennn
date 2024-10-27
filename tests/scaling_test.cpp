//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "scaling_test.h"

#include "../opennn/descriptives.h"
#include "../opennn/tensors.h"

namespace opennn
{


ScalingTest::ScalingTest() : UnitTesting()
{
}


void ScalingTest::test_scale_data_mean_standard_deviation()
{
    cout << "test_scale_data_inputs_mean_standard_deviation\n";

    Tensor<type, 2> matrix(10 + rand()%10, 1);
    Tensor<type, 2> scaled_matrix;

    Tensor<Descriptives, 1> matrix_descriptives;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::MeanStandardDeviation);

    data_set.scale_data();
    scaled_matrix = data_set.get_data();

    matrix_descriptives = data_set.calculate_variables_descriptives();

    assert_true(abs(matrix_descriptives(0).mean) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(matrix_descriptives(0).standard_deviation - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void ScalingTest::test_scale_data_minimum_maximum()
{
    cout << "test_scale_data_minimum_maximum\n";

    Tensor<type, 2> matrix(10 + rand()%10, 1);
    Tensor<type, 2> scaled_matrix;

    Tensor<Descriptives, 1> matrix_descriptives;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::MinimumMaximum);
    data_set.scale_data();

    scaled_matrix = data_set.get_data();
    matrix_descriptives = data_set.calculate_variables_descriptives();

    assert_true(abs(matrix_descriptives(0).minimum + type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(matrix_descriptives(0).maximum - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

}

void ScalingTest::test_scale_data_no_scaling()
{
    cout << "test_scale_data_no_scaling\n";

    Tensor<type, 2> matrix(1 + rand()%10, 1 + rand()%10);
    Tensor<type, 2> scaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::None);
    data_set.scale_data();

    scaled_matrix = data_set.get_data();

    assert_true(are_equal(matrix, scaled_matrix,type(NUMERIC_LIMITS_MIN)), LOG);
}


void ScalingTest::test_scale_data_standard_deviation()
{
    cout << "test_scale_data_standard_deviation\n";

    Tensor<type, 2> matrix(10 + rand()%10, 1);
    Tensor<type, 2> scaled_matrix;

    Tensor<Descriptives, 1> matrix_descriptives;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::StandardDeviation);
    data_set.scale_data();

    scaled_matrix = data_set.get_data();
    matrix_descriptives = data_set.calculate_variables_descriptives();

    assert_true(abs(matrix_descriptives(0).standard_deviation - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void ScalingTest::test_scale_data_logarithmic()
{
    cout << "test_scale_data_logarithmic\n";

    Tensor<type, 2> matrix(10 + rand()%10, 1);
    Tensor<type, 2> scaled_matrix;
    Tensor<type, 2> solution_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::Logarithm);
    data_set.scale_data();

    scaled_matrix = data_set.get_data();

    solution_matrix.resize(matrix.dimension(0),1);
    for(Index i = 0; i < matrix.size() ; i++)
    {
        solution_matrix(i) = log(matrix(i));
    }

    assert_true(are_equal(scaled_matrix, solution_matrix, type(NUMERIC_LIMITS_MIN)), LOG);
}


void ScalingTest::test_unscale_data_mean_standard_deviation()
{
    cout << "test_unscale_data_mean_standard_deviation\n";

    Tensor<type, 2> matrix(1 + rand()%10, 1 + rand()%10);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::MeanStandardDeviation);

    variables_descriptives = data_set.calculate_variables_descriptives();

    data_set.scale_data();
    data_set.unscale_data(variables_descriptives);

    unscaled_matrix = data_set.get_data();

    assert_true(are_equal(matrix, unscaled_matrix,type(NUMERIC_LIMITS_MIN)), LOG);
}


void ScalingTest::test_unscale_data_minimum_maximum()
{
    cout << "test_unscale_data_minimum_maximum\n";

    Tensor<type, 2> matrix(1 + rand()%10, 1 + rand()%10);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::MinimumMaximum);

    variables_descriptives = data_set.calculate_variables_descriptives();
    data_set.scale_data();
    data_set.unscale_data(variables_descriptives);

    unscaled_matrix = data_set.get_data();

    assert_true(are_equal(matrix, unscaled_matrix,type(NUMERIC_LIMITS_MIN)), LOG);
}


void ScalingTest::test_unscale_data_no_scaling()
{
    cout << "test_unscale_data_no_scaling\n";

    Tensor<type, 2> matrix(1 + rand()%10,1 + rand()%10);
    Tensor<type, 2> scaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::None);

    variables_descriptives = data_set.calculate_variables_descriptives();
    data_set.unscale_data(variables_descriptives);

    scaled_matrix = data_set.get_data();

    assert_true(are_equal(matrix, scaled_matrix,type(NUMERIC_LIMITS_MIN)), LOG);
}


void ScalingTest::test_unscale_data_standard_deviation()
{
    cout << "test_unscale_data_standard_deviation\n";

    Tensor<type, 2> matrix(1 + rand()%10, 1 + rand()%10);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::StandardDeviation);

    variables_descriptives = data_set.calculate_variables_descriptives();
    data_set.scale_data();
    data_set.unscale_data(variables_descriptives);

    unscaled_matrix = data_set.get_data();

    assert_true(are_equal(matrix, unscaled_matrix,type(NUMERIC_LIMITS_MIN)), LOG);
}


void ScalingTest::test_unscale_data_logarithmic()
{
    cout << "test_unscale_data_minimum_maximum\n";

    Tensor<type, 2> matrix(1 + rand()%10, 1 + rand()%10);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::Logarithm);

    variables_descriptives = data_set.calculate_variables_descriptives();
    data_set.scale_data();
    data_set.unscale_data(variables_descriptives);

    unscaled_matrix = data_set.get_data();

    assert_true(are_equal(matrix, unscaled_matrix,type(NUMERIC_LIMITS_MIN)), LOG);
}


void ScalingTest::run_test_case()
{
    cout << "Running scaling test case...\n";

    // Scaling

    test_scale_data_mean_standard_deviation();
    test_scale_data_minimum_maximum();
    test_scale_data_no_scaling();
    test_scale_data_standard_deviation();
    test_scale_data_logarithmic();

    // Unscaling

    test_unscale_data_mean_standard_deviation();
    test_unscale_data_minimum_maximum();
    test_unscale_data_no_scaling();
    test_unscale_data_standard_deviation();
    test_unscale_data_logarithmic();

    cout << "End of scaling test case.\n\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library sl free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library sl distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
