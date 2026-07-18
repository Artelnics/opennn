#include "pch.h"

#include "opennn/growing_inputs.h"
#include "opennn/training_strategy.h"
#include "opennn/dataset.h"
#include "opennn/tabular_dataset.h"
#include "opennn/dense_layer.h"
#include "opennn/standard_networks.h"


using namespace opennn;


TEST(GrowingInputsTest, DefaultConstructor)
{
    GrowingInputs growing_inputs;

    EXPECT_EQ(growing_inputs.has_training_strategy(), false);
}


TEST(GrowingInputsTest, GeneralConstructor)
{

    TrainingStrategy training_strategy;

    GrowingInputs growing_inputs(&training_strategy);

    EXPECT_EQ(growing_inputs.has_training_strategy(), true);

}

TEST(GrowingInputsTest, InputSelection)
{
    TabularDataset dataset(20, {2}, {1});
    dataset.set_data_random();
    dataset.split_samples_random();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{2}, Shape{1}));

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingInputs growing_inputs(&training_strategy);
    growing_inputs.set_display(false);

    EXPECT_EQ(growing_inputs.has_training_strategy(), true);

    InputsSelectionResult input_selection_results = growing_inputs.perform_input_selection();
    EXPECT_GE(input_selection_results.optimal_input_variables_indices[0], 0);
}


TEST(GrowingInputsTest, InputSelectionKnownResult)
{
    const Index samples = 50;

    TabularDataset dataset(samples, {2}, {1});

    MatrixR data(samples, 3);
    for(Index i = 0; i < samples; i++)
    {
        data(i, 0) = type(i) / samples;           // input1: señal clara
        data(i, 1) = type(rand()) / RAND_MAX;      // input2: ruido puro
        data(i, 2) = data(i, 0);                   // output = input1
    }
    dataset.set_data(data);
    dataset.split_samples_random();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{1}, Shape{1}));

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingInputs growing_inputs(&training_strategy);
    growing_inputs.set_display(false);

    growing_inputs.set_maximum_inputs_number(1);
    InputsSelectionResult results = growing_inputs.perform_input_selection();

    EXPECT_EQ(results.optimal_input_variables_indices.size(), 1);
    EXPECT_EQ(results.optimal_input_variables_indices[0], 0);
}


TEST(GrowingInputsTest, CrossValidationKeepsPersistentRoles)
{
    // folds_number > 1 scores subsets by k-fold CV over Training+Validation through a transient
    // overlay -- it must still select the informative input AND leave the user's roles intact.
    const Index samples = 60;

    TabularDataset dataset(samples, {2}, {1});
    MatrixR data(samples, 3);
    for (Index i = 0; i < samples; i++)
    {
        data(i, 0) = type(i) / samples;             // input1: clear signal
        data(i, 1) = type(rand()) / RAND_MAX;       // input2: pure noise
        data(i, 2) = data(i, 0);                    // output = input1
    }
    dataset.set_data(data);
    dataset.split_samples_random();

    const vector<SampleRole> roles_before = dataset.get_sample_roles();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{1}, Shape{1}));
    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingInputs growing_inputs(&training_strategy);
    growing_inputs.set_display(false);
    growing_inputs.set_maximum_inputs_number(1);
    growing_inputs.set_folds_number(3);   // stratified 3-fold CV scoring

    InputsSelectionResult results = growing_inputs.perform_input_selection();

    EXPECT_EQ(results.optimal_input_variables_indices.size(), 1);
    EXPECT_EQ(results.optimal_input_variables_indices[0], 0);   // still picks the informative input

    // The overlay must NEVER mutate the user's persistent roles.
    EXPECT_TRUE(dataset.get_sample_roles() == roles_before);
}


TEST(GrowingInputsTest, FoldsNumberSurvivesJsonRoundTrip)
{
    // The k-fold setting must persist through the project's JSON so a saved model keeps its CV
    // configuration. Mirrors the Neural Designer flow: a selector attached to a training strategy is
    // serialized and reloaded (from_JSON reconfigures inputs against the dataset). A default selector
    // stays at folds = 1 (legacy single-split behaviour).
    TabularDataset dataset(20, {2}, {1});
    dataset.set_data_random();
    dataset.split_samples_random();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{2}, Shape{1}));
    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingInputs saved(&training_strategy);
    EXPECT_EQ(saved.get_folds_number(), Index(1));   // default = legacy
    saved.set_folds_number(5);

    const filesystem::path file = filesystem::temp_directory_path() / "opennn_growing_inputs_folds.json";
    saved.save(file);

    GrowingInputs loaded(&training_strategy);
    loaded.load(file);
    EXPECT_EQ(loaded.get_folds_number(), Index(5));

    error_code ignored;
    filesystem::remove(file, ignored);
}
