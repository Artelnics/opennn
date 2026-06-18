#include "pch.h"

#include "../opennn/tensor_types.h"
#include "../opennn/dataset.h"
#include "../opennn/tabular_dataset.h"
#include "../opennn/scaling.h"
#include "../opennn/statistics.h"

using namespace opennn;

TEST(ScalingTest, ScaleDataMeanStandardDeviation)
{
    Index samples_number = 10 + rand() % 10;

    MatrixR data(samples_number, 1);
    data.setRandom();

    TabularDataset dataset(samples_number, { 1 }, { 0 });

    dataset.set_data(data);
    dataset.set_variable_scalers("MeanStandardDeviation");

    dataset.scale_data();
   
    vector<Descriptives> matrix_descriptives = dataset.calculate_feature_descriptives();

    EXPECT_NEAR(matrix_descriptives[0].mean, type(0), EPSILON);
    EXPECT_NEAR(matrix_descriptives[0].standard_deviation, type(1), EPSILON);
}


TEST(ScalingTest, ScaleDataMinimumMaximum)
{
    Index samples_number = 10 + rand() % 10;

    MatrixR matrix(samples_number, 1);
    matrix.setRandom();

    TabularDataset dataset(samples_number, { 1 }, { 0 });

    dataset.set_data(matrix);
    dataset.set_variable_scalers("MinimumMaximum");

    dataset.scale_data();

    vector<Descriptives> matrix_descriptives = dataset.calculate_feature_descriptives();

    EXPECT_NEAR(matrix_descriptives[0].minimum, type(-1.0), EPSILON);
    EXPECT_NEAR(matrix_descriptives[0].maximum, type(1.0), EPSILON);
}


TEST(ScalingTest, ScaleDataNoScaling2d)
{   
    Index samples_number = 1 + rand() % 10;

    MatrixR matrix(samples_number, samples_number);
    matrix.setRandom();
    
    MatrixR scaled_matrix;

    TabularDataset dataset(samples_number, { samples_number }, { 0 });

    dataset.set_data(matrix);
    dataset.set_variable_scalers("None");

    dataset.scale_data();

    scaled_matrix = dataset.get_data();
    
    EXPECT_LT((matrix - scaled_matrix).array().abs().maxCoeff(), EPSILON);
}


TEST(ScalingTest, ScaleDataStandardDeviation)
{
    Index samples_number = 10 + rand() % 10;

    MatrixR matrix(samples_number, 1);
    matrix.setRandom();

    TabularDataset dataset(samples_number, { 1 }, { 0 });

    dataset.set_data(matrix);
    dataset.set_variable_scalers("StandardDeviation");
    dataset.scale_data();

    vector<Descriptives> matrix_descriptives = dataset.calculate_feature_descriptives();

    EXPECT_NEAR(abs(matrix_descriptives[0].standard_deviation), type(1), EPSILON);
}


TEST(ScalingTest, ScaleDataLogarithmic)
{
    Index samples_number = 10 + rand() % 10;

    MatrixR matrix(samples_number, 1);
    MatrixR scaled_matrix;
    MatrixR solution_matrix;

    matrix.setRandom();

    matrix.array() = matrix.array().abs() + 1.0f;

    TabularDataset dataset(samples_number, { 1 }, { 0 });

    dataset.set_data(matrix);
    dataset.set_variable_scalers("Logarithm");
    dataset.scale_data();

    scaled_matrix = dataset.get_data();

    solution_matrix.resize(matrix.rows(),1);

    for(Index i = 0; i < matrix.size(); i++)
        solution_matrix(i) = log(matrix(i));

    EXPECT_LT((scaled_matrix - solution_matrix).array().abs().maxCoeff(), type(1e-4));
}


TEST(ScalingTest, UnscaleDataMeanStandardDeviation)
{
    Index samples_number = 1 + rand() % 10;

    MatrixR matrix(samples_number, samples_number);
    MatrixR unscaled_matrix;

    matrix.setRandom();

    TabularDataset dataset(samples_number, { samples_number }, { 0 });

    dataset.set_data(matrix);
    dataset.set_variable_scalers("MeanStandardDeviation");

    vector<Descriptives> matrix_descriptives = dataset.calculate_feature_descriptives();

    dataset.scale_data();

    dataset.unscale_features("Input",matrix_descriptives);

    unscaled_matrix = dataset.get_data();

    EXPECT_LT((matrix - unscaled_matrix).array().abs().maxCoeff(), EPSILON);
}


TEST(ScalingTest, UnscaleDataMinimumMaximum)
{
    Index samples_number = 1 + rand() % 10;

    MatrixR matrix(samples_number, samples_number);
    MatrixR unscaled_matrix;

    matrix.setRandom();
    
    TabularDataset dataset(samples_number, { samples_number }, { 0 });

    dataset.set_data(matrix);
    dataset.set_variable_scalers("MinimumMaximum");

    vector<Descriptives> matrix_descriptives = dataset.calculate_feature_descriptives();

    dataset.scale_data();
    dataset.unscale_features("Input", matrix_descriptives);

    unscaled_matrix = dataset.get_data();

    EXPECT_LT((matrix - unscaled_matrix).array().abs().maxCoeff(), EPSILON);
}


TEST(ScalingTest, UnscaleDataNoScaling2d)
{
    Index samples_number = 1 + rand() % 10;

    MatrixR matrix(samples_number, samples_number);
    MatrixR unscaled_matrix;

    matrix.setRandom();

    TabularDataset dataset(samples_number, { samples_number }, { 0 });

    dataset.set_data(matrix);
    dataset.set_variable_scalers("None");

    vector<Descriptives> matrix_descriptives = dataset.calculate_feature_descriptives();

    dataset.scale_data();
    dataset.unscale_features("Input", matrix_descriptives);

    unscaled_matrix = dataset.get_data();

    EXPECT_LT((matrix - unscaled_matrix).array().abs().maxCoeff(), EPSILON);
}


TEST(ScalingTest, UnscaleDataStandardDeviation)
{
    Index samples_number = 2 + rand() % 10;

    MatrixR matrix(samples_number, samples_number);
    MatrixR unscaled_matrix;

    matrix.setRandom();

    TabularDataset dataset(samples_number, { samples_number }, { 0 });
    dataset.set_data(matrix);
    dataset.set_variable_scalers("StandardDeviation");

    vector<Descriptives> matrix_descriptives = dataset.calculate_feature_descriptives();

    dataset.scale_data();
    dataset.unscale_features("Input", matrix_descriptives);

    unscaled_matrix = dataset.get_data();

    EXPECT_LT((matrix - unscaled_matrix).array().abs().maxCoeff(), type(1e-4));
}


TEST(ScalingTest, UnscaleDataLogarithmic)
{
    Index samples_number = 1 + rand() % 10;

    MatrixR matrix(samples_number, samples_number);
    MatrixR unscaled_matrix;

    matrix.setRandom();

    matrix.array() = matrix.array().abs() + 1.0f;

    TabularDataset dataset(samples_number, { samples_number }, { 0 });

    dataset.set_data(matrix);
    dataset.set_variable_scalers("Logarithm");

    vector<Descriptives> matrix_descriptives = dataset.calculate_feature_descriptives();

    dataset.scale_data();
    dataset.unscale_features("Input", matrix_descriptives);

    unscaled_matrix = dataset.get_data();

    EXPECT_LT((matrix - unscaled_matrix).array().abs().maxCoeff(), type(1e-4));
}


TEST(ScalingTest, ScaleLogarithmicShiftsNonPositiveValues)
{
    MatrixR matrix(3, 1);
    matrix << type(-2), type(0), type(3);

    MatrixMap map(matrix.data(), 3, 1);
    scale_logarithmic(map, 0);

    const float offset = 2.0f + 1.0f + EPSILON;

    EXPECT_NEAR(matrix(0, 0), std::log(-2.0f + offset), 1e-5f);
    EXPECT_NEAR(matrix(1, 0), std::log( 0.0f + offset), 1e-5f);
    EXPECT_NEAR(matrix(2, 0), std::log( 3.0f + offset), 1e-5f);

    EXPECT_TRUE(matrix.array().isFinite().all());
}


TEST(ScalingTest, UnscaleStandardDeviationZeroDeviationIsNoOp)
{
    MatrixR matrix(3, 1);
    matrix << type(1), type(2), type(3);

    Descriptives descriptives;
    descriptives.standard_deviation = type(0);

    MatrixMap map(matrix.data(), 3, 1);
    unscale_standard_deviation(map, 0, descriptives);

    EXPECT_NEAR(matrix(0, 0), type(1), 1e-6);
    EXPECT_NEAR(matrix(1, 0), type(2), 1e-6);
    EXPECT_NEAR(matrix(2, 0), type(3), 1e-6);

    EXPECT_TRUE(matrix.array().isFinite().all());
}
