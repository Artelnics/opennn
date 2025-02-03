#include "pch.h"

#include "../opennn/descriptives.h"
#include "../opennn/tensors.h"
#include "../opennn/data_set.h"

using namespace opennn;

/*
TEST(ScalingTest, ScaleDataMeanStandardDeviation)
{
    Index samples_number = 10 + rand() % 10;

    Tensor<type, 2> data(samples_number, 1);
    data.setRandom();

    DataSet data_set(samples_number, { 1 }, { 0 });

    data_set.set_data(data);
    data_set.set_raw_variable_scalers(Scaler::MeanStandardDeviation);

    data_set.scale_data();
   
    vector<Descriptives> matrix_descriptives = data_set.calculate_variable_descriptives();

    EXPECT_NEAR(matrix_descriptives[0].mean, 0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(matrix_descriptives[0].standard_deviation, type(1), NUMERIC_LIMITS_MIN);
    
}

TEST(ScalingTest, ScaleDataMinimumMaximum)
{
    Index samples_number = 10 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, 1);
    matrix.setRandom();

    DataSet data_set(samples_number, { 1 }, { 0 });

    data_set.set_data(matrix);
    data_set.set_raw_variable_scalers(Scaler::MinimumMaximum);
    data_set.scale_data();

    vector<Descriptives> matrix_descriptives = data_set.calculate_variable_descriptives();

    EXPECT_NEAR(abs(matrix_descriptives[0].minimum), type(1), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(abs(matrix_descriptives[0].maximum), type(1), NUMERIC_LIMITS_MIN);
}

TEST(ScalingTest, ScaleDataNoScaling)
{   
    Index samples_number = 1 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, samples_number);
    matrix.setRandom();
    
    Tensor<type, 2> scaled_matrix;

    DataSet data_set(samples_number, { samples_number }, { 0 });

    data_set.set_data(matrix);
    data_set.set_raw_variable_scalers(Scaler::None);

    data_set.scale_data();

    scaled_matrix = data_set.get_data();
    
    EXPECT_EQ(are_equal(matrix, scaled_matrix, NUMERIC_LIMITS_MIN),true);
}


TEST(ScalingTest, ScaleDataStandardDeviation)
{
    Index samples_number = 10 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, 1);
    matrix.setRandom();

    DataSet data_set(samples_number, { 1 }, { 0 });

    data_set.set_data(matrix);
    data_set.set_raw_variable_scalers(Scaler::StandardDeviation);
    data_set.scale_data();

    vector<Descriptives> matrix_descriptives = data_set.calculate_variable_descriptives();

    EXPECT_EQ(abs(matrix_descriptives[0].standard_deviation), type(1), NUMERIC_LIMITS_MIN);
}


TEST(ScalingTest, ScaleDataLogarithmic)
{
    Index samples_number = 10 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, 1);
    Tensor<type, 2> scaled_matrix;
    Tensor<type, 2> solution_matrix;

    matrix.setRandom();

    DataSet data_set(samples_number, { 1 }, { 0 });

    data_set.set_data(matrix);
    data_set.set_raw_variable_scalers(Scaler::Logarithm);
    data_set.scale_data();

    scaled_matrix = data_set.get_data();

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

    DataSet data_set(samples_number, { samples_number }, { 0 });

    data_set.set_data(matrix);
    data_set.set_raw_variable_scalers(Scaler::MeanStandardDeviation);

    vector<Descriptives> matrix_descriptives = data_set.calculate_variable_descriptives();

    data_set.scale_data();

    data_set.unscale_variables(DataSet::VariableUse::Input,matrix_descriptives);

    unscaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,NUMERIC_LIMITS_MIN),true);

}


TEST(ScalingTest, UnscaleDataMinimumMaximum)
{
    Index samples_number = 1 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, samples_number);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();
    
    DataSet data_set(samples_number, { samples_number }, { 0 });

    data_set.set_data(matrix);
    data_set.set_raw_variable_scalers(Scaler::MinimumMaximum);

    vector<Descriptives> matrix_descriptives = data_set.calculate_variable_descriptives();

    data_set.scale_data();
    data_set.unscale_variables(DataSet::VariableUse::Input, matrix_descriptives);

    unscaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,NUMERIC_LIMITS_MIN),true);

}


TEST(ScalingTest, UnscaleDataNoScaling)
{
    Index samples_number = 1 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, samples_number);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    DataSet data_set(samples_number, { samples_number }, { 0 });

    data_set.set_data(matrix);
    data_set.set_raw_variable_scalers(Scaler::None);

    vector<Descriptives> matrix_descriptives = data_set.calculate_variable_descriptives();

    data_set.scale_data();
    data_set.unscale_variables(DataSet::VariableUse::Input, matrix_descriptives);

    unscaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,NUMERIC_LIMITS_MIN),true);
}


TEST(ScalingTest, UnscaleDataStandardDeviation)
{
    Index samples_number = 1 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, samples_number);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    DataSet data_set(samples_number, { samples_number }, { 0 });
    data_set.set_data(matrix);
    data_set.set_raw_variable_scalers(Scaler::StandardDeviation);

    vector<Descriptives> matrix_descriptives = data_set.calculate_variable_descriptives();

    data_set.scale_data();
    data_set.unscale_variables(DataSet::VariableUse::Input, matrix_descriptives);

    unscaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,NUMERIC_LIMITS_MIN),true);

}

TEST(ScalingTest, UnscaleDataLogarithmic)
{
    Index samples_number = 1 + rand() % 10;

    Tensor<type, 2> matrix(samples_number, samples_number);
    Tensor<type, 2> unscaled_matrix;

    matrix.setRandom();

    DataSet data_set(samples_number, { samples_number }, { 0 });

    data_set.set_data(matrix);
    data_set.set_raw_variable_scalers(Scaler::Logarithm);

    vector<Descriptives> matrix_descriptives = data_set.calculate_variable_descriptives();

    data_set.scale_data();
    data_set.unscale_variables(DataSet::VariableUse::Input, matrix_descriptives);

    unscaled_matrix = data_set.get_data();

    EXPECT_EQ(are_equal(matrix, unscaled_matrix,NUMERIC_LIMITS_MIN),true);

}
*/