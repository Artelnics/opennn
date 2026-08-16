#include "tests/pch.h"

#include "opennn/dataset/dataset.h"
#include "opennn/dataset/time_series_dataset.h"
#include "opennn/model_selection/cross_validation.h"
#include "opennn/training_strategy/training_strategy.h"
#include "opennn/model_selection/model_selection.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/model_selection/growing_neurons.h"

using namespace opennn;

namespace
{

class InputsSelectionProbe final : public InputsSelection
{
public:
    void configure(NeuralNetwork* neural_network, Dataset* dataset, Index input_features)
    {
        configure_neural_network_inputs(neural_network, dataset, input_features);
    }

    Index get_minimum_inputs_number() const override { return 1; }
    Index get_maximum_inputs_number() const override { return 1; }
    InputsSelectionResult perform_input_selection() override { return {}; }
    void from_JSON(const JsonDocument&) override {}
    void to_JSON(JsonWriter&) const override {}
};

}

TEST(ModelSelectionTest, DefaultConstructor)
{
    ModelSelection model_selection;
}

TEST(ModelSelectionTest, GeneralConstructor)
{
    TrainingStrategy training_strategy;

    ModelSelection model_selection(&training_strategy);
}

TEST(ModelSelectionTest, OrderedDatasetsProduceContiguousFolds)
{
    TimeSeriesDataset dataset(8, {1}, {1});
    dataset.set_sample_roles(SampleRole::Training);

    NeuralNetwork neural_network;
    TrainingStrategy training_strategy(&neural_network, &dataset);

    const vector<vector<Index>> folds =
        build_fold_partition(&training_strategy, 3, 17);

    ASSERT_EQ(folds.size(), 3);
    EXPECT_EQ(folds[0], vector<Index>({0, 1}));
    EXPECT_EQ(folds[1], vector<Index>({2, 3, 4}));
    EXPECT_EQ(folds[2], vector<Index>({5, 6, 7}));
}

TEST(ModelSelectionTest, ConfiguresForecastingInputsThroughDatasetContract)
{
    TimeSeriesDataset dataset(8, {2}, {1});
    dataset.set_variable_names({"temperature", "pressure", "target"});
    dataset.set_past_time_steps(3);

    ForecastingNetwork neural_network({3, 2}, {2}, {1});
    InputsSelectionProbe inputs_selection;
    inputs_selection.configure(&neural_network, &dataset, 2);

    EXPECT_EQ(neural_network.get_input_shape(), (Shape{3, 2}));
    const vector<Variable>& input_variables = neural_network.get_input_variables();
    ASSERT_EQ(input_variables.size(), 6);
    EXPECT_EQ(input_variables.front().name, "temperature_lag0");
    EXPECT_EQ(input_variables.back().name, "pressure_lag2");
}
