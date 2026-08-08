#include "pch.h"

#include "opennn/tensor_types.h"
#include "opennn/dataset.h"
#include "opennn/tabular_dataset.h"
#include "opennn/scaling.h"
#include "opennn/statistics.h"
#include "opennn/tensor_operations.h"
#include "opennn/network_differential.h"

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
        solution_matrix(i) = std::log(matrix(i));

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

TEST(ScalingTest, ScaleLogarithmicClampsNonPositiveValues)
{
    MatrixR matrix(3, 1);
    matrix << type(-2), type(0), type(3);

    TabularDataset dataset(3, { 1 }, { 0 });
    dataset.set_data(matrix);
    dataset.set_variable_scalers("Logarithm");

    dataset.scale_data();

    const MatrixR scaled = dataset.get_data();

    EXPECT_NEAR(scaled(0, 0), std::log(EPSILON), 1e-5f);
    EXPECT_NEAR(scaled(1, 0), std::log(EPSILON), 1e-5f);
    EXPECT_NEAR(scaled(2, 0), std::log(3.0f), 1e-5f);

    EXPECT_TRUE(scaled.array().isFinite().all());
}

TEST(ScalingTest, UnscaleStandardDeviationZeroDeviationIsNoOp)
{
    MatrixR matrix(3, 1);
    matrix << type(1), type(1), type(1);

    TabularDataset dataset(3, { 1 }, { 0 });
    dataset.set_data(matrix);
    dataset.set_variable_scalers("StandardDeviation");

    vector<Descriptives> descriptives = dataset.calculate_feature_descriptives();
    ASSERT_NEAR(descriptives[0].standard_deviation, type(0), 1e-6);

    dataset.unscale_features("Input", descriptives);

    const MatrixR unscaled = dataset.get_data();

    EXPECT_NEAR(unscaled(0, 0), type(1), 1e-6);
    EXPECT_NEAR(unscaled(1, 0), type(1), 1e-6);
    EXPECT_NEAR(unscaled(2, 0), type(1), 1e-6);

    EXPECT_TRUE(unscaled.array().isFinite().all());
}

TEST(ScalingTest, ScaleValueMinimumMaximumUsesMinusOneOneRange)
{
    Descriptives descriptives;
    descriptives.minimum = type(2);
    descriptives.maximum = type(6);

    EXPECT_NEAR(scale_value(ScalerMethod::MinimumMaximum, descriptives, type(2)), type(-1), 1e-6);
    EXPECT_NEAR(scale_value(ScalerMethod::MinimumMaximum, descriptives, type(4)), type(0), 1e-6);
    EXPECT_NEAR(scale_value(ScalerMethod::MinimumMaximum, descriptives, type(6)), type(1), 1e-6);

    Descriptives constant;
    constant.minimum = type(3);
    constant.maximum = type(3);

    EXPECT_NEAR(scale_value(ScalerMethod::MinimumMaximum, constant, type(3)), type(0), 1e-6);
}

TEST(ScalingTest, ScaleValueGuardsDegenerateDeviationToZero)
{
    Descriptives descriptives;
    descriptives.mean = type(5);
    descriptives.standard_deviation = type(0);

    EXPECT_NEAR(scale_value(ScalerMethod::MeanStandardDeviation, descriptives, type(7)), type(0), 1e-6);
    EXPECT_NEAR(scale_value(ScalerMethod::StandardDeviation, descriptives, type(7)), type(0), 1e-6);

    descriptives.standard_deviation = type(2);

    EXPECT_NEAR(scale_value(ScalerMethod::MeanStandardDeviation, descriptives, type(7)), type(1), 1e-6);
    EXPECT_NEAR(scale_value(ScalerMethod::StandardDeviation, descriptives, type(7)), type(3.5), 1e-6);
    EXPECT_NEAR(scale_value(ScalerMethod::None, descriptives, type(7)), type(7), 1e-6);
}

TEST(ScalingTest, ScalingAffineMatchesScaleValue)
{
    Descriptives descriptives;
    descriptives.minimum = type(0);
    descriptives.maximum = type(255);
    descriptives.mean = type(100);
    descriptives.standard_deviation = type(50);

    const auto [minmax_scale, minmax_offset] =
        scaling_affine(ScalerMethod::MinimumMaximum, descriptives, type(0), type(1));
    EXPECT_NEAR(minmax_scale, type(1) / type(255), 1e-9);
    EXPECT_NEAR(minmax_offset, type(0), 1e-9);

    const auto [meanstd_scale, meanstd_offset] =
        scaling_affine(ScalerMethod::MeanStandardDeviation, descriptives, type(0), type(1));
    EXPECT_NEAR(meanstd_scale, type(1) / type(50), 1e-9);
    EXPECT_NEAR(meanstd_offset, type(-100) / type(50), 1e-6);

    const auto [image_scale, image_offset] =
        scaling_affine(ScalerMethod::ImageMinMax, descriptives, type(0), type(1));
    EXPECT_NEAR(image_scale, type(1) / type(255), 1e-9);
    EXPECT_NEAR(image_offset, type(0), 1e-9);

    Descriptives constant;
    constant.minimum = type(3);
    constant.maximum = type(3);
    constant.standard_deviation = type(0);

    // A feature with no spread collapses to zero, exactly as scale_value does.
    const auto [flat_scale, flat_offset] =
        scaling_affine(ScalerMethod::MinimumMaximum, constant, type(-1), type(1));
    EXPECT_NEAR(flat_scale, type(0), 1e-9);
    EXPECT_NEAR(flat_offset, type(0), 1e-9);

    const auto [zero_deviation_scale, zero_deviation_offset] =
        scaling_affine(ScalerMethod::StandardDeviation, constant, type(0), type(1));
    EXPECT_NEAR(zero_deviation_scale, type(0), 1e-9);
    EXPECT_NEAR(zero_deviation_offset, type(0), 1e-9);
}

// ---------------------------------------------------------------------------
// Degenerate-range agreement.
//
// Six places in the library divide by a range that can be zero, and they do not
// agree on what a constant feature means. The tests above pin each behaviour on
// its own; these pin the RELATIONSHIP between them, which is what any future
// consolidation has to preserve or deliberately change.
// ---------------------------------------------------------------------------

namespace
{

Descriptives constant_feature()
{
    Descriptives descriptives;
    descriptives.minimum = type(3);
    descriptives.maximum = type(3);
    descriptives.mean = type(3);
    descriptives.standard_deviation = type(0);
    return descriptives;
}

}

// scale_value and the tensor path agree: a constant feature scales to zero.
TEST(ScalerDegenerateAgreement, ScalarAndTensorPathsBothCollapseToZero)
{
    const Descriptives descriptives = constant_feature();

    EXPECT_NEAR(scale_value(ScalerMethod::MinimumMaximum, descriptives, type(3)), type(0), 1e-6);
    EXPECT_NEAR(scale_value(ScalerMethod::MeanStandardDeviation, descriptives, type(3)), type(0), 1e-6);

    MatrixR values = MatrixR::Constant(2, 1, type(3));

    VectorR minimums   = VectorR::Constant(1, descriptives.minimum);
    VectorR maximums   = VectorR::Constant(1, descriptives.maximum);
    VectorR means      = VectorR::Constant(1, descriptives.mean);
    VectorR deviations = VectorR::Constant(1, descriptives.standard_deviation);
    VectorR scalers    = VectorR::Constant(1, float(int(ScalerMethod::MinimumMaximum)));

    MatrixR scaled = values;

    TensorView input(values.data(), Shape{2, 1});
    TensorView output(scaled.data(), Shape{2, 1});
    TensorView minimums_view(minimums.data(), Shape{1});
    TensorView maximums_view(maximums.data(), Shape{1});
    TensorView means_view(means.data(), Shape{1});
    TensorView deviations_view(deviations.data(), Shape{1});
    TensorView scalers_view(scalers.data(), Shape{1});

    scale(input, minimums_view, maximums_view, means_view, deviations_view,
          scalers_view, type(-1), type(1), output);

    EXPECT_NEAR(scaled(0, 0), type(0), 1e-6);
    EXPECT_NEAR(scaled(1, 0), type(0), 1e-6);
}

// All three forward paths now agree on a constant feature. scaling_affine used to
// add EPSILON to the denominator instead of guarding, which turned a constant image
// channel into a ~1.7e7 multiplier -- and nothing downstream checked it, unlike the
// NetworkDifferential surrogate, which at least has a validation gate.
TEST(ScalerDegenerateAgreement, AffinePathAgreesWithScaleValue)
{
    const Descriptives descriptives = constant_feature();

    const auto [affine_scale, affine_offset] =
        scaling_affine(ScalerMethod::MinimumMaximum, descriptives, type(-1), type(1));

    EXPECT_NEAR(affine_scale, type(0), 1e-9);
    EXPECT_NEAR(affine_offset, type(0), 1e-9);

    // Agreement must hold away from the minimum too. The old behaviour only looked
    // correct at x == minimum, where float cancellation hid the divergence.
    for (const float x : {type(3), type(4), type(-2)})
    {
        const float scalar_result =
            scale_value(ScalerMethod::MinimumMaximum, descriptives, x, type(-1), type(1));
        const float affine_result = x * affine_scale + affine_offset;

        EXPECT_NEAR(scalar_result, type(0), 1e-6);
        EXPECT_NEAR(affine_result, scalar_result, 1e-6);
    }
}

// NetworkDifferential floors the divisor at 1e-12 while the forward paths guard
// at EPSILON (~1.19e-7) -- about five orders of magnitude apart. For a constant
// feature the forward output is 0 (derivative 0), but the analytic Jacobian uses
// 1/1e-12. Pinned so the mismatch is visible if the Jacobian is ever checked
// against the forward pass on degenerate inputs.
TEST(ScalerDegenerateAgreement, JacobianGuardFloorDiffersFromForwardGuard)
{
    EXPECT_NEAR(NetworkDifferential::guarded(type(0)), type(1e-12), type(1e-18));

    const float jacobian_slope = type(1) / NetworkDifferential::guarded(type(0));
    EXPECT_GT(jacobian_slope, type(1e11));

    const Descriptives descriptives = constant_feature();
    EXPECT_NEAR(scale_value(ScalerMethod::MinimumMaximum, descriptives, type(4)), type(0), 1e-6);

    EXPECT_LT(EPSILON / type(1e-12), type(1e6));
    EXPECT_GT(EPSILON / type(1e-12), type(1e4));
}
