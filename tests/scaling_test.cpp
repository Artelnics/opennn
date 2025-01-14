#include "pch.h"

#include "../opennn/descriptives.h"
#include "../opennn/tensors.h"

using namespace opennn;

TEST(ScalingTest, ScaleDataMeanStandardDeviation)
{
    
    Tensor<type, 2> data(10 + rand() % 10, 1);
    data.setRandom();

    Tensor<type, 2> scaled_data;

/*
    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::MeanStandardDeviation);

    data_set.scale_data();
    scaled_matrix = data_set.get_data();

    vector<Descriptives> matrix_descriptives = data_set.calculate_variable_descriptives();

    EXPECT_NEAR(matrix_descriptives(0).mean, 0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(matrix_descriptives(0).standard_deviation, type(1), NUMERIC_LIMITS_MIN);
*/
}



TEST(ScalingTest, ScaleDataMinimumMaximum)
{
/*
    Tensor<type, 2> matrix(10 + rand()%10, 1);
    Tensor<type, 2> scaled_matrix;

    vector<Descriptives> matrix_descriptives;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::MinimumMaximum);
    data_set.scale_data();

    scaled_matrix = data_set.get_data();
    matrix_descriptives = data_set.calculate_variable_descriptives();

    EXPECT_NEAR(abs(matrix_descriptives(0).minimum + type(1)) < NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(matrix_descriptives(0).maximum - type(1)) < NUMERIC_LIMITS_MIN);
*/
}

TEST(ScalingTest, ScaleDataNoScaling)
{
/*
    Tensor<type, 2> matrix(1 + rand()%10, 1 + rand()%10);
    Tensor<type, 2> scaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::None);
    data_set.scale_data();

    scaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, scaled_matrix,NUMERIC_LIMITS_MIN));
*/
}


TEST(ScalingTest, ScaleDataStandardDeviation)
{
/*
    Tensor<type, 2> matrix(10 + rand()%10, 1);
    Tensor<type, 2> scaled_matrix;

    vector<Descriptives> matrix_descriptives;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::StandardDeviation);
    data_set.scale_data();

    scaled_matrix = data_set.get_data();
    matrix_descriptives = data_set.calculate_variable_descriptives();

    EXPECT_EQ(abs(matrix_descriptives(0).standard_deviation - type(1)) < NUMERIC_LIMITS_MIN);
*/
}


TEST(ScalingTest, ScaleDataLogarithmic)
{
/*
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
        solution_matrix(i) = log(matrix(i));

    EXPECT_EQ(are_equal(scaled_matrix, solution_matrix, NUMERIC_LIMITS_MIN));
*/
}

TEST(ScalingTest, UnscaleDataMeanStandardDeviation)
{
/*
    Tensor<type, 2> matrix(1 + rand() % 10, 1 + rand() % 10);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::MeanStandardDeviation);

    variable_descriptives = data_set.calculate_variable_descriptives();

    data_set.scale_data();
    data_set.unscale_data(variable_descriptives);

    unscaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,NUMERIC_LIMITS_MIN));
*/
}


TEST(ScalingTest, UnscaleDataMinimumMaximum)
{
/*
    Tensor<type, 2> matrix(1 + rand()%10, 1 + rand()%10);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::MinimumMaximum);

    variable_descriptives = data_set.calculate_variable_descriptives();
    data_set.scale_data();
    data_set.unscale_data(variable_descriptives);

    unscaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,NUMERIC_LIMITS_MIN));
*/
}


TEST(ScalingTest, UnscaleDataNoScaling)
{
/*
    Tensor<type, 2> matrix(1 + rand()%10,1 + rand()%10);
    Tensor<type, 2> scaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::None);

    variable_descriptives = data_set.calculate_variable_descriptives();
    data_set.unscale_data(variable_descriptives);

    scaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, scaled_matrix,NUMERIC_LIMITS_MIN));
*/
}


TEST(ScalingTest, UnscaleDataStandardDeviation)
{

    Tensor<type, 2> matrix(1 + rand()%10, 1 + rand()%10);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();
/*
    DataSet data_set;
    data_set.set_raw_variable_scalers(Scaler::StandardDeviation);

    variable_descriptives = data_set.calculate_variable_descriptives();
    data_set.scale_data();
    data_set.unscale_data(variable_descriptives);

    unscaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,NUMERIC_LIMITS_MIN));
*/
}

TEST(ScalingTest, UnscaleDataLogarithmic)
{
/*
    Tensor<type, 2> matrix(1 + rand()%10, 1 + rand()%10);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    data_set.set(matrix);
    data_set.set_raw_variable_scalers(Scaler::Logarithm);

    variable_descriptives = data_set.calculate_variable_descriptives();
    data_set.scale_data();
    data_set.unscale_data(variable_descriptives);

    unscaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,NUMERIC_LIMITS_MIN));
*/
}

