#include "tests/common/pch.h"

#include "opennn/core/random_utilities.h"
#include "opennn/model_selection/growing_inputs.h"
#include "opennn/training/training.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/network/network.h"

using namespace opennn;

namespace
{

// Column 0 is a uniform ramp on [0, 1), column 1 is irrelevant uniform noise
// on [0, 1] and column 2 (the target) is a bit-exact copy of column 0.
MatrixR ramp_target_with_noise(const Index rows)
{
    MatrixR data(rows, 3);

    for (Index row = 0; row < rows; ++row)
    {
        const type ramp = type(row) / type(rows);

        data(row, 0) = ramp;
        data(row, 1) = random_uniform(type(0), type(1));
        data(row, 2) = ramp;
    }

    return data;
}

}

TEST(GrowingInputsTest, DefaultConstructor)
{
    GrowingInputs growing_inputs;
}

TEST(GrowingInputsTest, GeneralConstructor)
{
    Training training;

    GrowingInputs growing_inputs(&training);
}

TEST(GrowingInputsTest, InputSelection)
{
    TabularDataset dataset(20, {2}, {1});
    dataset.set_data_random();
    dataset.split_samples_random();

    Network network;
    network.add_layer(make_unique<opennn::Dense>(Shape{2}, Shape{1}));

    Training training(&network, &dataset);

    GrowingInputs growing_inputs(&training);
    growing_inputs.set_display(false);

    InputSelectionResult input_selection_results = growing_inputs.perform_input_selection();
    EXPECT_GE(input_selection_results.optimal_input_variables_indices[0], Index(0));
}

TEST(GrowingInputsTest, InputSelectionKnownResult)
{
    const Index samples = 50;

    TabularDataset dataset(samples, {2}, {1});

    dataset.set_data(ramp_target_with_noise(samples));

    dataset.split_samples_random();

    Network network;
    network.add_layer(make_unique<opennn::Dense>(Shape{1}, Shape{1}));

    Training training(&network, &dataset);

    GrowingInputs growing_inputs(&training);
    growing_inputs.set_display(false);

    growing_inputs.set_maximum_inputs_number(1);
    InputSelectionResult results = growing_inputs.perform_input_selection();

    EXPECT_EQ(results.optimal_input_variables_indices.size(), size_t(1));
    if (!results.optimal_input_variables_indices.empty())
        EXPECT_EQ(results.optimal_input_variables_indices.front(), Index(0));
}

TEST(GrowingInputsTest, CrossValidationKeepsPersistentRoles)
{
    const Index samples = 60;

    TabularDataset dataset(samples, {2}, {1});
    dataset.set_data(ramp_target_with_noise(samples));
    dataset.split_samples_random();

    const vector<SampleRole> roles_before = dataset.get_sample_roles();

    Network network;
    network.add_layer(make_unique<opennn::Dense>(Shape{1}, Shape{1}));
    Training training(&network, &dataset);

    GrowingInputs growing_inputs(&training);
    growing_inputs.set_display(false);
    growing_inputs.set_maximum_inputs_number(1);
    growing_inputs.set_folds_number(3);

    InputSelectionResult results = growing_inputs.perform_input_selection();

    EXPECT_EQ(results.optimal_input_variables_indices.size(), size_t(1));
    if (results.optimal_input_variables_indices.size() == 1)
        EXPECT_EQ(results.optimal_input_variables_indices[0], Index(0));

    EXPECT_TRUE(dataset.get_sample_roles() == roles_before);
}
