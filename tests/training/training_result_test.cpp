#include "tests/pch.h"

#include "opennn/training/training_result.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/training/sgd.h"
#include "opennn/training/training.h"

using namespace opennn;

TEST(TrainingResult, EpochsNumberMatchesRecordedHistory)
{
    const TrainingResult empty_results;
    EXPECT_EQ(empty_results.get_epochs_number(), 0);

    const TrainingResult results(3);
    EXPECT_EQ(results.get_epochs_number(), 3);
    EXPECT_EQ(results.write_override_results()(0, 1), "3");

    testing::internal::CaptureStdout();
    results.print();
    const string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("Epochs number: 3"), string::npos);
}

TEST(TrainingResult, ReportsTheModelReturnedAfterTraining)
{
    for (const bool restore_best : {false, true})
    {
        TabularDataset dataset(6, {1}, {1});
        MatrixR values(6, 2);
        values << 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0;
        dataset.set_data(values);
        dataset.set_variable_scalers("None");
        dataset.set_sample_roles(vector<string>{"Training", "Training", "Training",
                                                "Validation", "Validation", "Validation"});
        ApproximationNetwork network({1}, {}, {1});
        network.get_parameters_map().setZero();
        Training training(&network, &dataset);
        training.set_loss("MeanSquaredError");
        training.set_optimization_algorithm("SGD");
        auto* optimizer = dynamic_cast<SGD*>(training.get_optimization_algorithm());
        ASSERT_NE(optimizer, nullptr);
        optimizer->set_initial_learning_rate(0.1f);
        optimizer->set_initial_decay(0.0f);
        optimizer->set_maximum_epochs(3);
        optimizer->set_display(false);
        optimizer->set_shuffle(false);
        optimizer->set_restore_best(restore_best);

        const TrainingResult results = training.train();
        ASSERT_EQ(results.get_epochs_number(), 3);
        ASSERT_EQ(results.restored_epoch.has_value(), restore_best);
        const Index reported_epoch = restore_best ? 0 : 2;
        if (restore_best) EXPECT_EQ(*results.restored_epoch, reported_epoch);
        const MatrixR input = MatrixR::Zero(1, 1);
        const float prediction = network.calculate_outputs(input)(0, 0);
        const float actual_error = 0.5f * prediction * prediction;
        EXPECT_NEAR(results.get_validation_error(), actual_error, 1e-6f);
        EXPECT_FLOAT_EQ(results.get_training_error(), results.training_error_history(reported_epoch));
        EXPECT_GT(results.validation_error_history(2), results.validation_error_history(0));

        const auto table = results.write_override_results(7);
        EXPECT_NEAR(stof(table(3, 1)), results.get_training_error(), 1e-6f);
        EXPECT_NEAR(stof(table(4, 1)), actual_error, 1e-6f);
    }
}

TEST(TrainingResult, MissingValidationIsUnavailableAndSparseMeasurementsRemainVisible)
{
    TrainingResult results(3);
    results.training_error_history << 3.0f, 2.0f, 1.0f;
    EXPECT_TRUE(isnan(results.get_validation_error()));
    EXPECT_EQ(results.write_override_results()(4, 1), "NA");

    results.validation_error_history(0) = 0.25f;
    EXPECT_FLOAT_EQ(results.get_validation_error(), 0.25f);
    EXPECT_EQ(results.write_override_results()(4, 1), "0.25");

    results.restored_epoch = 1;
    EXPECT_TRUE(isnan(results.get_validation_error()));
    EXPECT_EQ(results.write_override_results()(4, 1), "NA");

    results.restored_epoch.reset();
    results.validation_error_history.setConstant(numeric_limits<float>::infinity());
    EXPECT_TRUE(isnan(results.get_validation_error()));
    results.resize_validation_error_history(0);
    EXPECT_TRUE(isnan(results.get_validation_error()));

    const TrainingResult empty;
    EXPECT_TRUE(isnan(empty.get_training_error()));
    EXPECT_TRUE(isnan(empty.get_validation_error()));
    EXPECT_NO_THROW(empty.print());
}

TEST(TrainingResult, PrintedAndSavedMetricsUseTheRestoredEpoch)
{
    TrainingResult results(2);
    results.training_error_history << 0.5f, 0.25f;
    results.validation_error_history << 0.125f, 0.75f;
    results.restored_epoch = 0;
    testing::internal::CaptureStdout();
    results.print();
    const string output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("Training error: 0.5"), string::npos);
    EXPECT_NE(output.find("Validation error: 0.125"), string::npos);

    const filesystem::path path = filesystem::temp_directory_path() / "opennn_training_result_restored.txt";
    const ScopeExit cleanup([&] { error_code error; filesystem::remove(path, error); });
    results.save(path);
    ifstream saved(path);
    const string contents((istreambuf_iterator<char>(saved)), istreambuf_iterator<char>());
    EXPECT_NE(contents.find("Training error; 0.5"), string::npos);
    EXPECT_NE(contents.find("Validation error; 0.125"), string::npos);
}
