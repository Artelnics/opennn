#include "tests/pch.h"

#include "opennn/core/tensor_types.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/image_dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/core/variable.h"
#include "opennn/dataset/batch.h"
#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"

using namespace opennn;

TEST(Dataset, FilterDataRespectsUsedFeaturesRolesMissingValuesAndTolerance)
{
    TabularDataset dataset(6, {2}, {1});
    MatrixR observations(6, 3);
    const float missing = numeric_limits<float>::quiet_NaN();
    observations << -0.5f * EPSILON, 1000, 1,
                     0.5f,          1000, 3,
                     missing,       1000, 1,
                     0.75f,         1000, missing,
                    -4,             1000, 1,
                    -2 * EPSILON,   1000, 1;
    dataset.set_data(observations);
    dataset.set_variable_role(1, "None");
    dataset.set_sample_roles("Training");
    dataset.set_sample_roles({1, 3}, "Testing");
    dataset.set_sample_roles({2}, "Validation");
    dataset.set_sample_roles({4}, "None");
    VectorR minimums(2), maximums(2);
    minimums << 0, 0;
    maximums << 1, 2;

    const VectorI removed = dataset.filter_data(minimums, maximums);
    ASSERT_EQ(removed.size(), 2);
    EXPECT_EQ(removed(0), 1);
    EXPECT_EQ(removed(1), 5);
    EXPECT_EQ(dataset.get_sample_indices("None"), (vector<Index>{1, 4, 5}));
    EXPECT_EQ(dataset.get_sample_indices("Training"), (vector<Index>{0}));
    EXPECT_EQ(dataset.get_sample_indices("Validation"), (vector<Index>{2}));
    EXPECT_EQ(dataset.get_sample_indices("Testing"), (vector<Index>{3}));
    EXPECT_EQ(dataset.filter_data(minimums, maximums).size(), 0);
}

TEST(Dataset, FilterDataUsesAvailableBoundsIncludingOneSidedNanLimits)
{
    TabularDataset dataset(4, {1}, {1});
    const float missing = numeric_limits<float>::quiet_NaN();
    MatrixR observations(4, 2);
    observations << 2, 100, missing, 100, 1, 100, 0.5f, 100;
    dataset.set_data(observations);
    dataset.set_sample_roles("Training");
    EXPECT_EQ(dataset.filter_data(VectorR{}, VectorR{}).size(), 0);
    EXPECT_EQ(dataset.get_samples_number("Training"), 4);

    const VectorI removed = dataset.filter_data(VectorR::Constant(1, missing), VectorR::Constant(1, 1));
    ASSERT_EQ(removed.size(), 1);
    EXPECT_EQ(removed(0), 0);
    EXPECT_EQ(dataset.get_sample_indices("Training"), (vector<Index>{1, 2, 3}));
}

TEST(Dataset, SplitRetainsUnusedRowsAndAssignsRemainderToTraining)
{
    TabularDataset dataset(10, {1}, {1});
    dataset.set_sample_roles("Training");
    dataset.set_sample_roles({0, 4, 9}, "None");
    dataset.split_samples_sequential(0.6f, 0.2f, 0.2f);
    EXPECT_EQ(dataset.get_sample_indices("Training"), (vector<Index>{1, 2, 3, 5, 6}));
    EXPECT_EQ(dataset.get_sample_indices("Validation"), (vector<Index>{7}));
    EXPECT_EQ(dataset.get_sample_indices("Testing"), (vector<Index>{8}));
    EXPECT_EQ(dataset.get_sample_indices("None"), (vector<Index>{0, 4, 9}));

    dataset.set_sample_roles("None");
    EXPECT_NO_THROW(dataset.split_samples_sequential(0, 0, 0));
    EXPECT_EQ(dataset.get_used_samples_number(), 0);
}

TEST(Dataset, SplitPreservesTheSeededPermutationIncludingUnusedRows)
{
    TabularDataset dataset(12, {1}, {1});
    dataset.set_sample_roles("Training");
    dataset.set_sample_roles({1, 8}, "None");
    vector<Index> order(12);
    iota(order.begin(), order.end(), 0);
    set_seed(4231);
    shuffle_vector(order);
    erase_if(order, [](Index i) { return i == 1 || i == 8; });
    vector<Index> training(order.begin(), order.begin() + 6);
    vector<Index> validation(order.begin() + 6, order.begin() + 8);
    vector<Index> testing(order.begin() + 8, order.end());
    ranges::sort(training);
    ranges::sort(validation);
    ranges::sort(testing);
    set_seed(4231);
    dataset.split_samples_random(0.6f, 0.2f, 0.2f);
    EXPECT_EQ(dataset.get_sample_indices("Training"), training);
    EXPECT_EQ(dataset.get_sample_indices("Validation"), validation);
    EXPECT_EQ(dataset.get_sample_indices("Testing"), testing);
}

TEST(Dataset, SplitRejectsInvalidRatiosBeforeChangingRoles)
{
    TabularDataset dataset(10, {1}, {1});
    dataset.set_sample_roles("Testing");
    for (float invalid : {-1.0f, numeric_limits<float>::infinity(), numeric_limits<float>::quiet_NaN()})
    {
        EXPECT_THROW(dataset.split_samples_sequential(invalid, 1, 1), runtime_error);
        EXPECT_THROW(dataset.split_samples_sequential(1, invalid, 1), runtime_error);
        EXPECT_THROW(dataset.split_samples_sequential(1, 1, invalid), runtime_error);
        EXPECT_EQ(dataset.get_samples_number("Testing"), 10);
    }
    EXPECT_THROW(dataset.split_samples_sequential(0, 0, 0), runtime_error);
    EXPECT_EQ(dataset.get_samples_number("Testing"), 10);
    dataset.split_samples_sequential(0, numeric_limits<float>::max(), 0);
    EXPECT_EQ(dataset.get_samples_number("Validation"), 10);
    dataset.split_samples_sequential(0, 0, 1);
    EXPECT_EQ(dataset.get_samples_number("Testing"), 10);
}

TEST(Dataset, UnsupportedInputSelectionCapabilitiesThrow)
{
    const ImageDataset dataset;

    EXPECT_THROW(dataset.calculate_input_target_correlation_values(), runtime_error);
    EXPECT_THROW(dataset.calculate_used_feature_scaling(VariableRole::Input), runtime_error);
}

TEST(Dataset, SetDataAndDimensions)
{
    TabularDataset tabular(3, { 2 }, { 1 });
    Dataset& dataset = tabular;

    MatrixR data(3, 3);
    data << type(1), type(2), type(3),
            type(4), type(5), type(6),
            type(7), type(8), type(9);

    dataset.set_data(data);

    const MatrixR& stored = dataset.get_data();

    EXPECT_EQ(stored.rows(), 3);
    EXPECT_EQ(stored.cols(), 3);

    EXPECT_NEAR(stored(0, 0), type(1), EPSILON);
    EXPECT_NEAR(stored(1, 1), type(5), EPSILON);
    EXPECT_NEAR(stored(2, 2), type(9), EPSILON);

    EXPECT_EQ(dataset.get_samples_number(), 3);
    EXPECT_EQ(dataset.get_variables_number(), 3);
}

TEST(Dataset, SetDataConstant)
{
    TabularDataset tabular(2, { 1 }, { 1 });
    Dataset& dataset = tabular;

    dataset.set_data_constant(type(5));

    const MatrixR& stored = dataset.get_data();

    for (Index i = 0; i < stored.rows(); ++i)
        for (Index j = 0; j < stored.cols(); ++j)
            EXPECT_NEAR(stored(i, j), type(5), EPSILON);
}

TEST(Dataset, SampleRoles)
{
    TabularDataset tabular(10, { 2 }, { 1 });
    Dataset& dataset = tabular;

    dataset.set_sample_roles("Training");

    EXPECT_EQ(dataset.get_samples_number(), 10);
    EXPECT_EQ(dataset.get_samples_number("Training"), 10);
    EXPECT_EQ(dataset.get_used_samples_number(), 10);
    EXPECT_EQ(ssize(dataset.get_sample_indices("Training")), 10);
}

TEST(Dataset, GetBatchesIncludesRemainder)
{
    TabularDataset dataset(5, { 1 }, { 1 });
    const vector<Index> sample_indices = { 0, 1, 2, 3, 4 };
    vector<vector<Index>> batches;

    dataset.get_batches(sample_indices, 2, Shuffle::No, batches);

    ASSERT_EQ(batches.size(), 3);
    EXPECT_EQ(batches[0], (vector<Index>{ 0, 1 }));
    EXPECT_EQ(batches[1], (vector<Index>{ 2, 3 }));
    EXPECT_EQ(batches[2], (vector<Index>{ 4 }));
}

TEST(Dataset, SetSampleRoleIndividual)
{
    TabularDataset tabular(10, { 2 }, { 1 });
    Dataset& dataset = tabular;

    dataset.set_sample_roles("Training");

    dataset.set_sample_role(0, "Testing");
    dataset.set_sample_role(1, "Validation");

    EXPECT_EQ(dataset.get_samples_number("Testing"), 1);
    EXPECT_EQ(dataset.get_samples_number("Validation"), 1);
    EXPECT_EQ(dataset.get_samples_number("Training"), 8);

    const vector<Index> testing_indices = dataset.get_sample_indices("Testing");
    ASSERT_EQ(ssize(testing_indices), 1);
    EXPECT_EQ(testing_indices[0], 0);
}

TEST(Dataset, SetVariableIndices)
{
    TabularDataset tabular(5, { 3 }, { 1 });
    Dataset& dataset = tabular;

    const vector<Index> input_indices = { 0, 1, 2 };
    const vector<Index> target_indices = { 3 };

    dataset.set_variable_indices(input_indices, target_indices);

    EXPECT_EQ(dataset.get_variables_number("Input"), 3);
    EXPECT_EQ(dataset.get_variables_number("Target"), 1);

    EXPECT_EQ(dataset.get_variable_indices("Input"), input_indices);
    EXPECT_EQ(dataset.get_variable_indices("Target"), target_indices);

    const auto& variables = dataset.get_variables();
    EXPECT_EQ(variables[0].role, VariableRole::Input);
    EXPECT_EQ(variables[3].role, VariableRole::Target);
}

TEST(Dataset, Shapes)
{
    TabularDataset tabular(3, { 2 }, { 1 });
    Dataset& dataset = tabular;

    EXPECT_EQ(dataset.get_input_shape().get_rank(), 1);
    EXPECT_EQ(dataset.get_input_shape()[0], 2);

    EXPECT_EQ(dataset.get_target_shape().get_rank(), 1);
    EXPECT_EQ(dataset.get_target_shape()[0], 1);

    EXPECT_EQ(dataset.get_shape("Input")[0], 2);
    EXPECT_EQ(dataset.get_shape("Target")[0], 1);
}

TEST(Dataset, SplitSamplesSequential)
{
    TabularDataset tabular(10, { 2 }, { 1 });
    Dataset& dataset = tabular;

    dataset.set_sample_roles("Training");

    dataset.split_samples_sequential(0.6f, 0.2f, 0.2f);

    EXPECT_EQ(dataset.get_samples_number("Training"), 6);
    EXPECT_EQ(dataset.get_samples_number("Validation"), 2);
    EXPECT_EQ(dataset.get_samples_number("Testing"), 2);

    EXPECT_EQ(dataset.get_samples_number("Training")
            + dataset.get_samples_number("Validation")
            + dataset.get_samples_number("Testing"),
              dataset.get_samples_number());
}

TEST(Dataset, BatchFill)
{
    TabularDataset tabular(3, { 2 }, { 1 });
    Dataset& dataset = tabular;

    MatrixR data(3, 3);
    data << type(3),  type(-2), type(0),
            type(-7), type(5),  type(1),
            type(6),  type(9),  type(1);

    dataset.set_data(data);

    dataset.set_sample_roles("Training");

    const Index samples_number = dataset.get_samples_number("Training");
    const vector<Index> training_samples_indices = dataset.get_sample_indices("Training");

    const vector<Index> input_variables_indices = { 0, 1 };
    const vector<Index> target_variables_indices = { 2 };

    Batch batch(samples_number, &dataset, EffectiveConfig{});
    batch.fill(training_samples_indices,
               FeatureSelection{input_variables_indices, {}, target_variables_indices});

    MatrixR input_data(3, 2);
    input_data << type(3),  type(-2),
                  type(-7), type(5),
                  type(6),  type(9);

    MatrixR target_data(3, 1);
    target_data << type(0),
                   type(1),
                   type(1);

    MatrixMap inputs = batch.get_inputs()[0].as_matrix();

    ASSERT_EQ(inputs.rows(), input_data.rows());
    ASSERT_EQ(inputs.cols(), input_data.cols());

    for (Index row = 0; row < inputs.rows(); ++row)
        for (Index column = 0; column < inputs.cols(); ++column)
            EXPECT_NEAR(inputs(row, column), input_data(row, column), 1e-6);

    MatrixMap targets = batch.get_targets().as_matrix();

    ASSERT_EQ(targets.rows(), target_data.rows());
    ASSERT_EQ(targets.cols(), target_data.cols());

    for (Index row = 0; row < targets.rows(); ++row)
        for (Index column = 0; column < targets.cols(); ++column)
            EXPECT_NEAR(targets(row, column), target_data(row, column), 1e-6);
}

TEST(Variable, ConstructorResolvesStrings)
{
    Variable variable("price", "Input", VariableType::Numeric, "MinimumMaximum");

    EXPECT_EQ(variable.name, "price");
    EXPECT_EQ(variable.get_role_type(), VariableRole::Input);
    EXPECT_EQ(variable.type, VariableType::Numeric);
    EXPECT_EQ(variable.get_scaler_type(), ScalerMethod::MinimumMaximum);

    EXPECT_EQ(variable.get_role(), "Input");
    EXPECT_EQ(variable.get_scaler(), "MinimumMaximum");
}

TEST(Variable, SetRoleAndScalerFromString)
{
    Variable variable;

    variable.set_role("Target");
    variable.set_scaler("StandardDeviation");

    EXPECT_EQ(variable.get_role_type(), VariableRole::Target);
    EXPECT_EQ(variable.get_scaler_type(), ScalerMethod::StandardDeviation);
}

TEST(Variable, TypeStringRoundTrip)
{
    const vector<VariableType> types = {
        VariableType::None, VariableType::Numeric, VariableType::Binary,
        VariableType::Integer, VariableType::Categorical, VariableType::DateTime,
        VariableType::Constant
    };

    for (const VariableType variable_type : types)
        EXPECT_EQ(string_to_variable_type(variable_type_to_string(variable_type)), variable_type);
}

TEST(Variable, ScalerStringRoundTrip)
{
    const vector<ScalerMethod> scalers = {
        ScalerMethod::None, ScalerMethod::MinimumMaximum, ScalerMethod::MeanStandardDeviation,
        ScalerMethod::StandardDeviation, ScalerMethod::Logarithm, ScalerMethod::ImageMinMax
    };

    for (const ScalerMethod scaler : scalers)
        EXPECT_EQ(string_to_scaler_method(scaler_method_to_string(scaler)), scaler);
}
