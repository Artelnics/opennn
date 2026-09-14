#include "tests/pch.h"

#include "opennn/core/json.h"
#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/time_series_dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/model_selection/cross_validation.h"
#include "opennn/model_selection/selection_utilities.h"
#include "opennn/training/training.h"
#include "opennn/model_selection/model_selection.h"
#include "opennn/models/models.h"
#include "opennn/model_selection/growing_neurons.h"
#include "opennn/model_selection/growing_inputs.h"
#include "opennn/model_selection/genetic_algorithm.h"
#include "opennn/training/sgd.h"

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

TEST(ModelSelectionTest, SmallStratifiedPartitionsHaveBalancedNonemptyFolds)
{
    TabularDataset dataset(4, {1}, {1});
    MatrixR data(4, 2);
    data << 0, 0, 1, 0, 2, 1, 3, 1;
    dataset.set_data(data);
    dataset.set_sample_roles(SampleRole::Training);
    Network network;
    Training training(&network, &dataset);

    for (const Index count : {Index(2), Index(3), Index(4)})
    {
        const auto folds = build_fold_partition(&training, count, 17);
        EXPECT_EQ(folds, build_fold_partition(&training, count, 17));
        vector<Index> samples;
        size_t smallest = 4;
        size_t largest = 0;
        for (const auto& fold : folds)
        {
            EXPECT_FALSE(fold.empty());
            smallest = min(smallest, fold.size());
            largest = max(largest, fold.size());
            samples.insert(samples.end(), fold.begin(), fold.end());
        }
        EXPECT_LE(largest - smallest, 1u);
        ranges::sort(samples);
        EXPECT_EQ(samples, vector<Index>({0, 1, 2, 3}));
    }
    for (const Index count : {Index(-1), Index(0), Index(1), Index(5)})
        EXPECT_THROW(build_fold_partition(&training, count), runtime_error);
}

TEST(ModelSelectionTest, InvalidPartitionsAreRejectedBeforeTraining)
{
    TabularDataset dataset(5, {1}, {1});
    dataset.set_sample_roles(SampleRole::Training);
    dataset.set_sample_role(4, SampleRole::Testing);
    ApproximationNetwork network({1}, {}, {1});
    const VectorR original_parameters = network.get_parameters_map();
    const auto original_roles = dataset.get_sample_roles();
    Training training(&network, &dataset);
    const vector<vector<vector<Index>>> invalid_partitions = {
        {}, {{0, 1, 2, 3}}, {{0, 1}, {2, 3}, {}},
        {{0, 0, 1}, {2, 3}}, {{0, 1}, {1, 2, 3}},
        {{0, 1}, {2}}, {{0, 1}, {2, 3, 4}},
        {{0, 1}, {2, 3, -1}}, {{0, 1}, {2, 3, 5}}
    };
    for (const auto& partition : invalid_partitions)
    {
        EXPECT_THROW(evaluate_folds(&training, partition), runtime_error);
        EXPECT_EQ(dataset.get_sample_roles(), original_roles);
        EXPECT_TRUE(network.get_parameters_map().isApprox(original_parameters, 0.0f));
    }
}

TEST(ModelSelectionTest, SmallCrossValidationMeasuresEveryFold)
{
    TabularDataset dataset(4, {1}, {1});
    MatrixR data(4, 2);
    data << 0, 0, 1, 0, 2, 1, 3, 1;
    dataset.set_data(data);
    dataset.set_variable_scalers("None");
    dataset.set_sample_roles(SampleRole::Training);
    ApproximationNetwork network({1}, {}, {1});
    Training training(&network, &dataset);
    training.set_optimization_algorithm("SGD");
    auto* optimizer = dynamic_cast<SGD*>(training.get_optimization_algorithm());
    ASSERT_NE(optimizer, nullptr);
    optimizer->set_initial_learning_rate(0.0f);
    optimizer->set_maximum_epochs(1);
    optimizer->set_display(false);
    vector<float> errors;
    optimizer->post_epoch_callback = [&](Index, float, float validation_error, Network*)
    {
        EXPECT_GT(dataset.get_samples_number(SampleRole::Validation), 0);
        EXPECT_GT(dataset.get_samples_number(SampleRole::Training), 0);
        EXPECT_TRUE(isfinite(validation_error));
        errors.push_back(validation_error);
    };

    const auto result = evaluate_folds(&training, build_fold_partition(&training, 3));
    ASSERT_EQ(errors.size(), 3u);
    EXPECT_NEAR(result.validation_error, (errors[0] + errors[1] + errors[2]) / 3.0f, 1e-6f);
    EXPECT_EQ(dataset.get_samples_number(SampleRole::Training), 4);
    EXPECT_EQ(dataset.get_samples_number(SampleRole::Validation), 0);
}

TEST(ModelSelectionTest, MissingValidationCannotWinACandidateTrial)
{
    TabularDataset dataset(4, {1}, {1});
    dataset.set_data_constant(0.0f);
    dataset.set_variable_scalers("None");
    dataset.set_sample_roles(SampleRole::Training);
    ApproximationNetwork network({1}, {}, {1});
    Training training(&network, &dataset);
    training.get_optimization_algorithm()->set_maximum_epochs(1);
    training.get_optimization_algorithm()->set_display(false);
    bool observed = false;
    const auto result = evaluate_candidate(&training, &network, 1, {}, 1, false,
        [&](Index, float, float validation_error, bool improved)
        {
            observed = true;
            EXPECT_TRUE(isnan(validation_error));
            EXPECT_FALSE(improved);
        });
    EXPECT_TRUE(observed);
    EXPECT_EQ(result.validation_error, MAX);
}

namespace
{
void expect_candidate_scores_returned_model(Device device)
{
    Configuration::instance().set(device, Type::FP32);
    const ScopeExit reset_configuration([] { Configuration::instance().set(Device::CPU, Type::FP32); });
    for (bool restore_best : {false, true})
        for (bool legacy_history_minimum : {false, true})
        {
            SCOPED_TRACE(testing::Message() << "restore=" << restore_best
                                           << " legacy_minimum=" << legacy_history_minimum);
            TabularDataset dataset(6, {1}, {1});
            MatrixR values(6, 2);
            values << 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0;
            dataset.set_data(values);
            dataset.set_variable_scalers("None");
            dataset.set_sample_roles(vector<string>{"Training", "Training", "Training",
                                                    "Validation", "Validation", "Validation"});
            ApproximationNetwork network({1}, {}, {1});
            Training training(&network, &dataset);
            training.set_loss("MeanSquaredError");
            training.set_optimization_algorithm("SGD");
            auto* optimizer = dynamic_cast<SGD*>(training.get_optimization_algorithm());
            ASSERT_NE(optimizer, nullptr);
            optimizer->set_initial_learning_rate(0.1f);
            optimizer->set_initial_decay(0.0f);
            optimizer->set_batch_size(2);
            optimizer->set_maximum_epochs(5);
            optimizer->set_validation_period(2);
            optimizer->set_display(false);
            optimizer->set_shuffle(false);
            optimizer->set_restore_best(restore_best);
            optimizer->set_cuda_graph(device == Device::CUDA);
            vector<float> training_errors;
            vector<float> validation_errors;
            optimizer->post_epoch_callback = [&](Index, float error, float validation_error, Network*)
            {
                training_errors.push_back(error);
                validation_errors.push_back(validation_error);
            };
            Index trials = 0;
            const CandidateEvaluation result = evaluate_candidate(
                &training, &network, 1, {}, 1, legacy_history_minimum,
                [&](Index trial, float error, float validation_error, bool improved)
                {
                    ++trials;
                    EXPECT_EQ(trial, 0);
                    EXPECT_TRUE(improved);
                    EXPECT_TRUE(isfinite(error));
                    EXPECT_TRUE(isfinite(validation_error));
                },
                [&](Index) { network.set_parameters(VectorR::Zero(network.get_parameters_buffer_size())); });

            ASSERT_EQ(training_errors.size(), 5u);
            ASSERT_EQ(validation_errors.size(), 5u);
            EXPECT_EQ(trials, 1);
            EXPECT_GT(validation_errors.back(), validation_errors.front());
            const size_t reported_epoch = restore_best ? 0 : 4;
            EXPECT_FLOAT_EQ(result.training_error, training_errors[reported_epoch]);
            EXPECT_FLOAT_EQ(result.validation_error, validation_errors[reported_epoch]);
            const float prediction = network.calculate_outputs(MatrixR::Zero(1, 1))(0, 0);
            EXPECT_NEAR(result.validation_error, 0.5f * prediction * prediction, 1.0e-5f);
            EXPECT_FALSE(optimizer->get_cuda_graph_capture_failed());
        }
}
}

TEST(ModelSelectionTest, CandidateScoringMatchesRestorationWithSparseValidationCPU)
{
    expect_candidate_scores_returned_model(Device::CPU);
}

TEST(ModelSelectionTest, CandidateScoringMatchesRestorationWithSparseValidationCUDA)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";
    expect_candidate_scores_returned_model(Device::CUDA);
}

TEST(ModelSelectionTest, GrowingSelectorsRequireValidationBeforeChangingTheModel)
{
    TabularDataset dataset(4, {1}, {1});
    dataset.set_sample_roles(SampleRole::Training);
    ApproximationNetwork network({1}, {2}, {1});
    const VectorR original_parameters = network.get_parameters_map();
    const auto original_inputs = dataset.get_variable_indices(VariableRole::Input);
    Training training(&network, &dataset);
    GrowingInputs inputs(&training);
    GrowingNeurons neurons(&training);
    EXPECT_THROW(inputs.perform_input_selection(), runtime_error);
    EXPECT_THROW(neurons.perform_neurons_selection(), runtime_error);
    EXPECT_EQ(dataset.get_variable_indices(VariableRole::Input), original_inputs);
    EXPECT_TRUE(network.get_parameters_map().isApprox(original_parameters, 0.0f));
}

TEST(ModelSelectionTest, SelectorsRejectMissingTrainingConfiguration)
{
    Training empty_training;
    for (Training* training : {static_cast<Training*>(nullptr), &empty_training})
    {
        GrowingInputs inputs(training);
        GrowingNeurons neurons(training);
        GeneticAlgorithm genetic(training);
        EXPECT_THROW(inputs.perform_input_selection(), runtime_error);
        EXPECT_THROW(neurons.perform_neurons_selection(), runtime_error);
        EXPECT_THROW(genetic.perform_input_selection(), runtime_error);
    }
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
