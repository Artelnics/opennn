#include "pch.h"

#include "../opennn/tensor_types.h"
#include "../opennn/dataset.h"
#include "../opennn/tabular_dataset.h"
#include "../opennn/variable.h"
#include "../opennn/batch.h"
#include "../opennn/configuration.h"

using namespace opennn;


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

    EXPECT_TRUE(dataset.is_sample_used(0));
    EXPECT_TRUE(dataset.is_sample_used(9));
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

    EXPECT_EQ(dataset.get_input_shape().rank, 1);
    EXPECT_EQ(dataset.get_input_shape()[0], 2);

    EXPECT_EQ(dataset.get_target_shape().rank, 1);
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
    data << type(1),type(4),type(1),
            type(2),type(-5),type(0),
            type(-3),type(6),type(1);
    dataset.set_data(data);

    dataset.set_sample_roles("Training");

    const Index samples_number = dataset.get_samples_number("Training");
    const vector<Index> training_samples_indices = dataset.get_sample_indices("Training");

    const vector<Index> input_variables_indices = { 0, 1 };
    const vector<Index> target_variables_indices = { 2 };

    Batch batch(samples_number, &dataset, Configuration::Resolved{});
    batch.fill(training_samples_indices, input_variables_indices, {}, target_variables_indices);

    MatrixR input_data(3, 2);
    input_data << type(1),type(4),
                  type(2),type(-5),
                  type(-3),type(6);

    MatrixR target_data(3, 1);
    target_data << type(1),
                   type(0),
                   type(1);

    MatrixMap inputs(batch.get_inputs()[0].as<type>(), 3, 2);

    ASSERT_EQ(inputs.rows(), input_data.rows());
    ASSERT_EQ(inputs.cols(), input_data.cols());

    for(Index i = 0; i < inputs.rows(); ++i)
        for(Index j = 0; j < inputs.cols(); ++j)
            EXPECT_NEAR(inputs(i, j), input_data(i, j), 1e-6);

    MatrixMap targets(batch.get_targets().as<type>(), 3, 1);

    ASSERT_EQ(targets.rows(), target_data.rows());
    ASSERT_EQ(targets.cols(), target_data.cols());

    for(Index i = 0; i < targets.rows(); ++i)
        for(Index j = 0; j < targets.cols(); ++j)
            EXPECT_NEAR(targets(i, j), target_data(i, j), 1e-6);
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
