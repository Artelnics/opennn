#include "tests/pch.h"

#include "opennn/core/json.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/time_series_dataset.h"
#include "opennn/model_selection/cross_validation.h"
#include "opennn/model_selection/selection_utilities.h"
#include "opennn/training/training.h"
#include "opennn/model_selection/model_selection.h"
#include "opennn/models/models.h"
#include "opennn/model_selection/growing_neurons.h"

using namespace opennn;

namespace
{

class InputSelectionProbe final : public InputSelection
{
public:
    void configure(Network* network, Dataset* dataset, Index input_features)
    {
        configure_network_inputs(network, dataset, input_features);
    }

    Index get_minimum_inputs_number() const override { return 1; }
    Index get_maximum_inputs_number() const override { return 1; }
    InputSelectionResult perform_input_selection() override { return {}; }
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
    Training training;

    ModelSelection model_selection(&training);
}

TEST(ModelSelectionTest, InputSelectionConfigurationRoundTrips)
{
    Training training;
    ModelSelection selection(&training);
    JsonWriter writer;
    selection.to_JSON(writer);
    JsonDocument document;
    document.set_root(Json::parse(writer.c_str()));
    const Json& configuration = document.get_root().at("ModelSelection");
    EXPECT_FALSE(configuration.has("InputsSelection"));
    const Json& input = configuration.at("InputSelection");
    EXPECT_EQ(input.at("InputSelectionMethod").as_string(), "GrowingInputs");
    EXPECT_TRUE(input.has("GrowingInputs"));

    ModelSelection restored(&training);
    restored.from_JSON(document);
    EXPECT_EQ(restored.get_input_selection_name(), "GrowingInputs");
    EXPECT_EQ(restored.get_training(), &training);
    JsonWriter restored_writer;
    restored.to_JSON(restored_writer);
    EXPECT_EQ(Json::parse(restored_writer.c_str()).dump(), document.get_root().dump());
}

TEST(ModelSelectionTest, OrderedDatasetsProduceContiguousFolds)
{
    TimeSeriesDataset dataset(8, {1}, {1});
    dataset.set_sample_roles(SampleRole::Training);

    Network network;
    Training training(&network, &dataset);

    const vector<vector<Index>> folds =
        build_fold_partition(&training, 3, 17);

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

    ForecastingNetwork network({3, 2}, {2}, {1});
    InputSelectionProbe input_selection;
    input_selection.configure(&network, &dataset, 2);

    EXPECT_EQ(network.get_input_shape(), (Shape{3, 2}));
    const vector<Variable>& input_variables = network.get_input_variables();
    ASSERT_EQ(input_variables.size(), 6);
    EXPECT_EQ(input_variables.front().name, "temperature_lag0");
    EXPECT_EQ(input_variables.back().name, "pressure_lag2");
}

TEST(ModelSelectionTest, AppliesInputScalingThroughEndpointContract)
{
    ApproximationNetwork network({2}, {}, {1});
    auto* const endpoint = dynamic_cast<FeatureScalingEndpoint*>(
        network.get_layers().front().get());
    ASSERT_NE(endpoint, nullptr);

    FeatureScaling current = endpoint->get_feature_scaling();
    current.min_range = -2.0f;
    current.max_range = 2.0f;
    endpoint->set_feature_scaling(current);

    FeatureScaling selected;
    selected.descriptives = {
        Descriptives(1.0f, 3.0f, 2.0f, 1.0f),
        Descriptives(2.0f, 6.0f, 4.0f, 2.0f)};
    selected.scalers = {
        ScalerMethod::MeanStandardDeviation,
        ScalerMethod::MinimumMaximum};

    apply_input_scaling(&network, selected);

    const FeatureScaling actual = endpoint->get_feature_scaling();
    EXPECT_EQ(actual.scalers, selected.scalers);
    EXPECT_FLOAT_EQ(actual.descriptives[0].minimum, 1.0f);
    EXPECT_FLOAT_EQ(actual.descriptives[1].maximum, 6.0f);
    EXPECT_FLOAT_EQ(actual.min_range, -2.0f);
    EXPECT_FLOAT_EQ(actual.max_range, 2.0f);
}
