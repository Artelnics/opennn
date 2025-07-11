#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/dataset.h"

using namespace opennn;

TEST(ScalingTest, ScaleDataMeanStandardDeviation)
{
    Index samples_number = 10 + rand() % 10;

    Tensor<type, 2> data(samples_number, 1);
    data.setRandom();

    Dataset dataset(samples_number, { 1 }, { 0 });

    dataset.set_data(data);
    dataset.set_raw_variable_scalers(Scaler::MeanStandardDeviation);

    dataset.scale_data();
   
    vector<Descriptives> matrix_descriptives = dataset.calculate_variable_descriptives();

    EXPECT_NEAR(matrix_descriptives[0].mean, type(0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(matrix_descriptives[0].standard_deviation, type(1), NUMERIC_LIMITS_MIN);
}


TEST(ScalingTest, ScaleDataMinimumMaximum)
{
    Index samples_number = 10 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, 1);
    matrix.setRandom();

    Tensor<type, 2> scaled_matrix;

    Dataset dataset(samples_number, { 1 }, { 0 });

    dataset.set_data(matrix);
    dataset.set_raw_variable_scalers(Scaler::MinimumMaximum);

    dataset.scale_data();

    vector<Descriptives> matrix_descriptives = dataset.calculate_variable_descriptives();

    EXPECT_NEAR(abs(matrix_descriptives[0].minimum), type(0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(matrix_descriptives[0].maximum), type(1), NUMERIC_LIMITS_MIN);
}


TEST(ScalingTest, ScaleDataNoScaling2d)
{   
    Index samples_number = 1 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, samples_number);
    matrix.setRandom();
    
    Tensor<type, 2> scaled_matrix;

    Dataset dataset(samples_number, { samples_number }, { 0 });

    dataset.set_data(matrix);
    dataset.set_raw_variable_scalers(Scaler::None);

    dataset.scale_data();

    scaled_matrix = dataset.get_data();
    
    EXPECT_EQ(are_equal(matrix, scaled_matrix, NUMERIC_LIMITS_MIN),true);
}


TEST(ScalingTest, ScaleDataStandardDeviation)
{
    Index samples_number = 10 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, 1);
    matrix.setRandom();

    Dataset dataset(samples_number, { 1 }, { 0 });

    dataset.set_data(matrix);
    dataset.set_raw_variable_scalers(Scaler::StandardDeviation);
    dataset.scale_data();

    vector<Descriptives> matrix_descriptives = dataset.calculate_variable_descriptives();

    EXPECT_NEAR(abs(matrix_descriptives[0].standard_deviation), type(1), NUMERIC_LIMITS_MIN);
}


TEST(ScalingTest, ScaleDataLogarithmic)
{
    Index samples_number = 10 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, 1);
    Tensor<type, 2> scaled_matrix;
    Tensor<type, 2> solution_matrix;

    matrix.setRandom();

    Dataset dataset(samples_number, { 1 }, { 0 });

    dataset.set_data(matrix);
    dataset.set_raw_variable_scalers(Scaler::Logarithm);
    dataset.scale_data();

    scaled_matrix = dataset.get_data();

    solution_matrix.resize(matrix.dimension(0),1);

    for(Index i = 0; i < matrix.size() ; i++)
        solution_matrix(i) = log(matrix(i));

    EXPECT_EQ(are_equal(scaled_matrix, solution_matrix, NUMERIC_LIMITS_MIN),true);
}

TEST(ScalingTest, UnscaleDataMeanStandardDeviation)
{
    Index samples_number = 1 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, samples_number);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    Dataset dataset(samples_number, { samples_number }, { 0 });

    dataset.set_data(matrix);
    dataset.set_raw_variable_scalers(Scaler::MeanStandardDeviation);

    vector<Descriptives> matrix_descriptives = dataset.calculate_variable_descriptives();

    dataset.scale_data();

    dataset.unscale_variables("Input",matrix_descriptives);

    unscaled_matrix = dataset.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,NUMERIC_LIMITS_MIN),true);
}


TEST(ScalingTest, UnscaleDataMinimumMaximum)
{
    Index samples_number = 1 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, samples_number);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();
    
    Dataset dataset(samples_number, { samples_number }, { 0 });

    dataset.set_data(matrix);
    dataset.set_raw_variable_scalers(Scaler::MinimumMaximum);

    vector<Descriptives> matrix_descriptives = dataset.calculate_variable_descriptives();

    dataset.scale_data();
    dataset.unscale_variables("Input", matrix_descriptives);

    unscaled_matrix = dataset.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,NUMERIC_LIMITS_MIN),true);
}


TEST(ScalingTest, UnscaleDataNoScaling2d)
{
    Index samples_number = 1 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, samples_number);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    Dataset dataset(samples_number, { samples_number }, { 0 });

    dataset.set_data(matrix);
    dataset.set_raw_variable_scalers(Scaler::None);

    vector<Descriptives> matrix_descriptives = dataset.calculate_variable_descriptives();

    dataset.scale_data();
    dataset.unscale_variables("Input", matrix_descriptives);

    unscaled_matrix = dataset.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,NUMERIC_LIMITS_MIN),true);
}


TEST(ScalingTest, UnscaleDataStandardDeviation)
{
    Index samples_number = 1 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, samples_number);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    Dataset dataset(samples_number, { samples_number }, { 0 });
    dataset.set_data(matrix);
    dataset.set_raw_variable_scalers(Scaler::StandardDeviation);

    vector<Descriptives> matrix_descriptives = dataset.calculate_variable_descriptives();

    dataset.scale_data();
    dataset.unscale_variables("Input", matrix_descriptives);

    unscaled_matrix = dataset.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,NUMERIC_LIMITS_MIN),true);
}


TEST(ScalingTest, UnscaleDataLogarithmic)
{
    Index samples_number = 1 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, samples_number);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    Dataset dataset(samples_number, { samples_number }, { 0 });

    dataset.set_data(matrix);
    dataset.set_raw_variable_scalers(Scaler::Logarithm);

    vector<Descriptives> matrix_descriptives = dataset.calculate_variable_descriptives();

    dataset.scale_data();
    dataset.unscale_variables("Input", matrix_descriptives);

    unscaled_matrix = dataset.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,NUMERIC_LIMITS_MIN),true);
}
