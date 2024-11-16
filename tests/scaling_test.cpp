#include "pch.h"

#include "../opennn/descriptives.h"
#include "../opennn/tensors.h"


/*
void ScalingTest::test_scale_data_mean_standard_deviation()
{
    cout << "test_scale_data_inputs_mean_standard_deviation\n";

    Tensor<type, 2> matrix(10 + rand()%10, 1);
    Tensor<type, 2> scaled_matrix;

    vector<Descriptives> matrix_descriptives;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::MeanStandardDeviation);

    data_set.scale_data();
    scaled_matrix = data_set.get_data();

    matrix_descriptives = data_set.calculate_variable_descriptives();

    EXPECT_EQ(abs(matrix_descriptives(0).mean) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(matrix_descriptives(0).standard_deviation - type(1)) < type(NUMERIC_LIMITS_MIN));
}


void ScalingTest::test_scale_data_minimum_maximum()
{
    cout << "test_scale_data_minimum_maximum\n";

    Tensor<type, 2> matrix(10 + rand()%10, 1);
    Tensor<type, 2> scaled_matrix;

    vector<Descriptives> matrix_descriptives;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::MinimumMaximum);
    data_set.scale_data();

    scaled_matrix = data_set.get_data();
    matrix_descriptives = data_set.calculate_variable_descriptives();

    EXPECT_EQ(abs(matrix_descriptives(0).minimum + type(1)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(matrix_descriptives(0).maximum - type(1)) < type(NUMERIC_LIMITS_MIN));

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

    EXPECT_EQ(are_equal(matrix, scaled_matrix,type(NUMERIC_LIMITS_MIN)));
}


void ScalingTest::test_scale_data_standard_deviation()
{
    cout << "test_scale_data_standard_deviation\n";

    Tensor<type, 2> matrix(10 + rand()%10, 1);
    Tensor<type, 2> scaled_matrix;

    vector<Descriptives> matrix_descriptives;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::StandardDeviation);
    data_set.scale_data();

    scaled_matrix = data_set.get_data();
    matrix_descriptives = data_set.calculate_variable_descriptives();

    EXPECT_EQ(abs(matrix_descriptives(0).standard_deviation - type(1)) < type(NUMERIC_LIMITS_MIN));
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

    EXPECT_EQ(are_equal(scaled_matrix, solution_matrix, type(NUMERIC_LIMITS_MIN)));
}


void ScalingTest::test_unscale_data_mean_standard_deviation()
{
    cout << "test_unscale_data_mean_standard_deviation\n";

    Tensor<type, 2> matrix(1 + rand()%10, 1 + rand()%10);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::MeanStandardDeviation);

    variable_descriptives = data_set.calculate_variable_descriptives();

    data_set.scale_data();
    data_set.unscale_data(variable_descriptives);

    unscaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,type(NUMERIC_LIMITS_MIN)));

}


void ScalingTest::test_unscale_data_minimum_maximum()
{
    cout << "test_unscale_data_minimum_maximum\n";

    Tensor<type, 2> matrix(1 + rand()%10, 1 + rand()%10);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::MinimumMaximum);

    variable_descriptives = data_set.calculate_variable_descriptives();
    data_set.scale_data();
    data_set.unscale_data(variable_descriptives);

    unscaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,type(NUMERIC_LIMITS_MIN)));

}


void ScalingTest::test_unscale_data_no_scaling()
{
    cout << "test_unscale_data_no_scaling\n";

    Tensor<type, 2> matrix(1 + rand()%10,1 + rand()%10);
    Tensor<type, 2> scaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::None);

    variable_descriptives = data_set.calculate_variable_descriptives();
    data_set.unscale_data(variable_descriptives);

    scaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, scaled_matrix,type(NUMERIC_LIMITS_MIN)));

}


void ScalingTest::test_unscale_data_standard_deviation()
{
    cout << "test_unscale_data_standard_deviation\n";

    Tensor<type, 2> matrix(1 + rand()%10, 1 + rand()%10);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::StandardDeviation);

    variable_descriptives = data_set.calculate_variable_descriptives();
    data_set.scale_data();
    data_set.unscale_data(variable_descriptives);

    unscaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,type(NUMERIC_LIMITS_MIN)));

}


void ScalingTest::test_unscale_data_logarithmic()
{
    cout << "test_unscale_data_minimum_maximum\n";

    Tensor<type, 2> matrix(1 + rand()%10, 1 + rand()%10);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::Logarithm);

    variable_descriptives = data_set.calculate_variable_descriptives();
    data_set.scale_data();
    data_set.unscale_data(variable_descriptives);

    unscaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,type(NUMERIC_LIMITS_MIN)));
}
*/
