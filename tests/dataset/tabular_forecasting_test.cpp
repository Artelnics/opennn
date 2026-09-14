#include "tests/pch.h"

#include "opennn/dataset/tabular_dataset.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/json.h"
#include "opennn/core/device_backend.h"
#include "opennn/dataset/batch.h"

#include <fstream>

using namespace opennn;

namespace
{

MatrixR forecasting_rows(Index count)
{
    MatrixR rows(count, 4);
    for (Index row = 0; row < count; ++row)
        rows.row(row) << 10.5f + float(row), 100.5f + 2.0f * float(row),
                         1000.5f + 3.0f * float(row), 10000.5f + 4.0f * float(row);
    return rows;
}

}

TEST(TabularForecasting, ConfigurationUsesCompleteChronologicalWindowsAndCanBeCleared)
{
    TabularDataset dataset(30, {2}, {2});
    const MatrixR raw = forecasting_rows(30);
    dataset.set_data(raw);
    EXPECT_FALSE(dataset.is_forecasting());
    EXPECT_FALSE(dataset.sample_order_matters());
    EXPECT_THROW(dataset.get_sequence_data("Training", "Input"), runtime_error);

    dataset.configure_forecasting(3, 2, true);
    EXPECT_EQ(dataset.get_input_shape(), Shape({3, 2}));
    EXPECT_EQ(dataset.get_target_shape(), Shape({4}));
    EXPECT_EQ(dataset.get_samples_number("Training"), 14);
    EXPECT_EQ(dataset.get_sample_indices("Validation"), vector<Index>({18, 19}));
    EXPECT_EQ(dataset.get_sample_indices("Testing"), vector<Index>({24, 25}));
    const Tensor3 sequences = dataset.get_sequence_data("Validation", "Input");
    ASSERT_EQ(sequences.dimension(0), 2);
    EXPECT_FLOAT_EQ(sequences(0, 0, 0), raw(18, 0));
    EXPECT_FLOAT_EQ(sequences(1, 2, 1), raw(21, 1));
    EXPECT_THROW(dataset.get_sequence_data("Training", "Target"), runtime_error);

    const vector<SampleRole> roles = dataset.get_sample_roles();
    dataset.clear_forecasting();
    EXPECT_FALSE(dataset.is_forecasting());
    EXPECT_FALSE(dataset.sample_order_matters());
    EXPECT_EQ(dataset.get_input_shape(), Shape({2}));
    EXPECT_EQ(dataset.get_target_shape(), Shape({2}));
    EXPECT_EQ(dataset.get_sample_roles(), roles);
    EXPECT_TRUE(dataset.get_data().isApprox(raw, 0.0f));
    std::array<float, 2> target;
    dataset.fill_targets({1}, {2, 3}, target.data(), FillMode::Inference);
    EXPECT_FLOAT_EQ(target[0], raw(1, 2));
}

TEST(TabularForecasting, WindowLayoutsAndScaledPaddingAreIndependentOfColumnOrder)
{
    TabularDataset dataset(8, {2}, {2});
    const MatrixR raw = forecasting_rows(8);
    dataset.set_data(raw);
    dataset.configure_forecasting(2, 3, true);
    std::array<float, 8> inputs;
    std::array<float, 12> targets;
    dataset.fill_inputs({0, 7}, {0, 1}, inputs.data(), FillMode::Inference);
    EXPECT_EQ(inputs, (std::array<float, 8>{10.5f, 100.5f, 11.5f, 102.5f,
                                          17.5f, 114.5f, 0.0f, 0.0f}));
    dataset.fill_targets({0, 7}, {2, 3}, targets.data(), FillMode::Inference);
    EXPECT_EQ(targets, (std::array<float, 12>{1006.5f, 1009.5f, 1012.5f,
                                             10008.5f, 10012.5f, 10016.5f,
                                             0, 0, 0, 0, 0, 0}));

    dataset.configure_forecasting(2, 3);
    std::array<float, 2> distant_target;
    dataset.fill_targets({0}, {2, 3}, distant_target.data(), FillMode::Inference);
    EXPECT_EQ(distant_target, (std::array<float, 2>{1012.5f, 10016.5f}));

    dataset.set_sample_roles("Training");
    dataset.set_variable_scalers("MinimumMaximum");
    FeatureScaling scaling;
    scaling.min_range = 0.25f;
    scaling.max_range = 1.25f;
    dataset.prepare_training_scaling(VariableRole::Input, scaling, 2);
    std::array<float, 4> contiguous;
    std::array<float, 4> reordered;
    dataset.fill_inputs({7}, {0, 1}, contiguous.data(), FillMode::Inference);
    dataset.fill_inputs({7}, {1, 0}, reordered.data(), FillMode::Inference);
    EXPECT_EQ(contiguous, (std::array<float, 4>{1.25f, 1.25f, 0, 0}));
    EXPECT_EQ(reordered, contiguous);
}

TEST(TabularForecasting, BinaryWindowsShareStorageScalingAndJsonMetadata)
{
    const filesystem::path csv = filesystem::temp_directory_path() / "opennn_forecasting_storage.csv";
    const filesystem::path cache = filesystem::temp_directory_path() / "opennn_forecasting_storage.bin";
    const ScopeExit cleanup([&]
    {
        error_code error;
        filesystem::remove(csv, error);
        filesystem::remove(cache, error);
    });
    const MatrixR raw = forecasting_rows(8);
    {
        std::ofstream output(csv);
        output << "input_a;input_b;target_a;target_b\n";
        for (Index row = 0; row < raw.rows(); ++row)
        {
            string line = format("{:.1f};{:.1f};{:.1f};{:.1f}\n",
                                 raw(row, 0), raw(row, 1), raw(row, 2), raw(row, 3));
            ranges::replace(line, '.', ',');
            output << line;
        }
    }
    TabularDataset memory(8, {2}, {2});
    memory.set_data(raw);
    TabularDataset binary;
    binary.set_storage_mode(Dataset::StorageMode::BinaryFile);
    binary.set_binary_cache_path(cache);
    binary.set_data_path(csv);
    binary.set_separator(Dataset::Separator::Semicolon);
    binary.set_has_header(true);
    binary.set_number_format({',', '.'});
    binary.set_display(false);
    binary.read_csv();
    binary.set_variable_indices({0, 1}, {2, 3});
    ASSERT_EQ(binary.get_data().size(), 0);

    for (TabularDataset* dataset : {&memory, &binary})
    {
        dataset->configure_forecasting(3, 2, true);
        dataset->set_sample_roles("Training");
        dataset->set_variable_scalers("MinimumMaximum");
        dataset->prepare_training_scaling(VariableRole::Input, FeatureScaling{}, 2);
        dataset->prepare_training_scaling(VariableRole::Target, FeatureScaling{}, 4);
    }
    for (const vector<Index>& columns : {vector<Index>{0, 1}, vector<Index>{1, 0}})
    {
        std::array<float, 18> expected;
        std::array<float, 18> actual;
        memory.fill_inputs({0, 4, 7}, columns, expected.data(), FillMode::Inference);
        binary.fill_inputs({0, 4, 7}, columns, actual.data(), FillMode::Inference);
        for (size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1.0e-6f);
    }
    std::array<float, 12> expected_targets;
    std::array<float, 12> actual_targets;
    memory.fill_targets({0, 4, 7}, {2, 3}, expected_targets.data(), FillMode::Inference);
    binary.fill_targets({0, 4, 7}, {2, 3}, actual_targets.data(), FillMode::Inference);
    for (size_t i = 0; i < actual_targets.size(); ++i)
        EXPECT_NEAR(actual_targets[i], expected_targets[i], 1.0e-6f);

    JsonWriter writer;
    binary.to_JSON(writer);
    JsonDocument document;
    document.set_root(Json::parse(writer.c_str()));
    TabularDataset restored;
    restored.set_binary_cache_path(cache);
    restored.from_JSON(document);
    EXPECT_TRUE(restored.is_forecasting());
    EXPECT_EQ(restored.get_storage_mode(), Dataset::StorageMode::BinaryFile);
    EXPECT_EQ(restored.get_number_format().decimal_separator, ',');
    EXPECT_EQ(restored.get_number_format().group_separator, '.');
    EXPECT_EQ(restored.get_sample_roles(), binary.get_sample_roles());
    restored.prepare_training_scaling(VariableRole::Input, FeatureScaling{}, 2);
    restored.prepare_training_scaling(VariableRole::Target, FeatureScaling{}, 4);
    restored.fill_targets({0, 4, 7}, {2, 3}, actual_targets.data(), FillMode::Inference);
    for (size_t i = 0; i < actual_targets.size(); ++i)
        EXPECT_NEAR(actual_targets[i], expected_targets[i], 1.0e-6f);
}

TEST(TabularForecasting, MissingRowsExcludeEveryAffectedWindow)
{
    const filesystem::path csv = filesystem::temp_directory_path() / "opennn_forecasting_missing.csv";
    const filesystem::path cache = filesystem::temp_directory_path() / "opennn_forecasting_missing.bin";
    const ScopeExit cleanup([&]
    {
        error_code error;
        filesystem::remove(csv, error);
        filesystem::remove(cache, error);
    });
    MatrixR raw = forecasting_rows(10);
    raw(3, 0) = QUIET_NAN;
    raw(8, 3) = QUIET_NAN;
    {
        std::ofstream output(csv);
        output << "input_a,input_b,target_a,target_b\n";
        for (Index row = 0; row < raw.rows(); ++row)
            for (Index column = 0; column < raw.cols(); ++column)
            {
                if (isnan(raw(row, column))) output << "NA";
                else output << raw(row, column);
                output << (column + 1 == raw.cols() ? '\n' : ',');
            }
    }
    for (bool binary : {false, true})
    {
        TabularDataset dataset(10, {2}, {2});
        if (binary)
        {
            dataset.set_storage_mode(Dataset::StorageMode::BinaryFile);
            dataset.set_binary_cache_path(cache);
            dataset.set_data_path(csv);
            dataset.set_separator(Dataset::Separator::Comma);
            dataset.set_has_header(true);
            dataset.set_display(false);
            dataset.read_csv();
            dataset.set_variable_indices({0, 1}, {2, 3});
        }
        else
            dataset.set_data(raw);
        dataset.configure_forecasting(2);
        dataset.set_sample_roles("Training");
        dataset.set_missing_values_method(TabularDataset::MissingValuesMethod::Unuse);
        dataset.scrub_missing_values();
        EXPECT_EQ(dataset.get_sample_indices("Training"), vector<Index>({0, 4, 5}));
    }
}

TEST(TabularForecasting, MissingBinaryCacheThrowsForLargeWindowBatches)
{
    TabularDataset original(8, {2}, {2});
    original.configure_forecasting(64);
    JsonWriter writer;
    original.to_JSON(writer);
    JsonDocument document;
    document.set_root(Json::parse(writer.c_str()));
    document.get_root()["Dataset"]["DataSource"]["StorageMode"] = "BinaryFile";
    const filesystem::path cache = filesystem::temp_directory_path()
                                  / "opennn_missing_forecasting_cache.bin";
    ASSERT_FALSE(filesystem::exists(cache));
    TabularDataset binary;
    binary.set_binary_cache_path(cache);
    binary.from_JSON(document);
    const vector<Index> samples(2048, 0);
    vector<float> inputs(2048 * 64 * 2);
    EXPECT_THROW(binary.fill_inputs(samples, {0, 1}, inputs.data(), FillMode::Inference), runtime_error);
}

TEST(TabularForecasting, TabularBatchesStillSupportAbsentTargetShapes)
{
    TabularDataset dataset(2, {1}, {1});
    dataset.set_data_constant(1.0f);
    dataset.set_variable_role(1, "None");
    dataset.set_shape(VariableRole::Target, {});
    Batch batch(2, &dataset, {Device::CPU, Type::FP32, 0});
    EXPECT_NO_THROW(batch.fill({0, 1}, dataset.get_feature_selection()));
    EXPECT_TRUE(batch.target.shape.empty());
}

TEST(TabularForecasting, CudaWindowsMatchHostForResidentAndBf16Staging)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable.";
    TabularDataset dataset(8, {2}, {2});
    dataset.set_data(forecasting_rows(8));
    dataset.configure_forecasting(3, 2, true);
    dataset.set_sample_roles("Training");
    dataset.set_variable_scalers("MinimumMaximum");
    dataset.prepare_training_scaling(VariableRole::Input, FeatureScaling{}, 2);
    dataset.prepare_training_scaling(VariableRole::Target, FeatureScaling{}, 4);
    dataset.enable_device_residency();
    ASSERT_TRUE(dataset.is_device_resident());

    const vector<Index> samples{0, 4, 7};
    std::array<float, 18> expected_inputs;
    std::array<float, 12> expected_targets;
    dataset.fill_inputs(samples, {0, 1}, expected_inputs.data(), FillMode::Inference);
    dataset.fill_targets(samples, {2, 3}, expected_targets.data(), FillMode::Inference);
    for (const Type precision : {Type::FP32, Type::BF16})
    {
        if (precision == Type::BF16 && device::cuda_compute_capability() < 80) continue;
        SCOPED_TRACE(precision == Type::FP32 ? "FP32" : "BF16");
        const EffectiveConfig config{Device::CUDA, precision, 0};
        Batch staged(3, &dataset, config, true);
        Batch destination(3, &dataset, config);
        staged.fill(samples, dataset.get_feature_selection(), FillMode::Inference);
        EXPECT_EQ(staged.device_gather.has_value(), precision == Type::FP32);
        const DeviceStream stream = device::get_compute_stream();
        staged.upload_to_device_batch_async(destination, stream);
        std::array<float, 18> actual_inputs;
        std::array<float, 12> actual_targets;
        copy_device_to_host_float(destination.input.buffer.data(), precision,
                                  Index(actual_inputs.size()), actual_inputs.data(), stream);
        copy_device_to_host_float(destination.target.buffer.data(), Type::FP32,
                                  Index(actual_targets.size()), actual_targets.data(), stream);
        device::synchronize(stream);
        for (size_t i = 0; i < actual_inputs.size(); ++i)
            EXPECT_NEAR(actual_inputs[i], expected_inputs[i], precision == Type::BF16 ? 0.004f : 1.0e-6f);
        for (size_t i = 0; i < actual_targets.size(); ++i)
            EXPECT_NEAR(actual_targets[i], expected_targets[i], 1.0e-6f);
    }
    dataset.configure_forecasting(2);
    EXPECT_FALSE(dataset.is_device_resident());
}

TEST(TabularForecasting, ModelInputVariablesReflectForecastingWindow)
{
    TabularDataset dataset(4, {2}, {1});
    dataset.set_variable_names({"temperature", "pressure", "target"});
    dataset.configure_forecasting(3);
    dataset.resize_input_shape(2);

    const vector<Variable> model_variables = dataset.get_model_input_variables();

    ASSERT_EQ(model_variables.size(), 6);
    EXPECT_TRUE(ranges::all_of(model_variables, [](const Variable& variable)
    {
        return variable.role == VariableRole::Input;
    }));
    EXPECT_EQ(dataset.get_input_shape(), Shape({3, 2}));
    EXPECT_TRUE(dataset.sample_order_matters());

    MatrixR raw(4, 3);
    raw << 1.0f, 101.0f, 1001.0f,
           2.0f, 102.0f, 1002.0f,
           3.0f, 103.0f, 1003.0f,
           4.0f, 104.0f, 1004.0f;
    dataset.set_data(raw);
    std::array<float, 6> inputs{};
    dataset.fill_inputs({0}, {0, 1}, inputs.data(), FillMode::Inference);
    const vector<pair<string, float>> expected{
        {"temperature_lag0", 1.0f}, {"pressure_lag0", 101.0f},
        {"temperature_lag1", 2.0f}, {"pressure_lag1", 102.0f},
        {"temperature_lag2", 3.0f}, {"pressure_lag2", 103.0f}};
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(model_variables[i].name, expected[i].first);
        EXPECT_FLOAT_EQ(inputs[i], expected[i].second);
    }
}

TEST(TabularForecasting, MultiTargetJsonRoundTripPreservesTargetLayout)
{
    TabularDataset original(12, {1}, {2});
    original.configure_forecasting(2, 2, true);
    JsonWriter writer;
    original.to_JSON(writer);
    JsonDocument document;
    document.set_root(Json::parse(writer.c_str()));

    MatrixR raw(12, 3);
    for (Index row = 0; row < raw.rows(); ++row)
        raw.row(row) << float(row), float(100 + row), float(1000 + row);

    const auto check = [&](TabularDataset& dataset)
    {
        dataset.from_JSON(document);
        ASSERT_TRUE(dataset.get_multi_target());
        ASSERT_EQ(dataset.get_target_shape(), Shape({4}));
        EXPECT_EQ(dataset.get_sample_roles(), original.get_sample_roles());
        dataset.set_data(raw);
        std::array<float, 10> targets;
        targets.fill(-777.0f);
        dataset.fill_targets({0, 1}, {1, 2}, targets.data(), FillMode::Inference);
        const std::array<float, 10> expected{
            102.0f, 103.0f, 1002.0f, 1003.0f,
            103.0f, 104.0f, 1003.0f, 1004.0f, -777.0f, -777.0f};
        EXPECT_EQ(targets, expected);
    };

    TabularDataset fresh;
    check(fresh);
    check(original);
}

TEST(TabularForecasting, LegacyJsonDefaultsToSingleTargetOnReusedDataset)
{
    TabularDataset dataset(12, {1}, {1});
    dataset.configure_forecasting(2, 2, true);
    JsonWriter writer;
    dataset.to_JSON(writer);
    JsonDocument legacy;
    legacy.set_root(Json::parse(writer.c_str()));
    auto& source = legacy.get_root()["Dataset"]["DataSource"].as_object();
    std::erase_if(source, [](const auto& field) { return field.first == "MultiTarget"; });

    dataset.from_JSON(legacy);

    EXPECT_FALSE(dataset.get_multi_target());
    EXPECT_EQ(dataset.get_future_time_steps(), 2);
    EXPECT_EQ(dataset.get_target_shape(), Shape({1}));
}

TEST(TabularForecasting, RejectsInvalidWindowAndStaleBatchShape)
{
    TabularDataset dataset(12, {1}, {1});
    EXPECT_THROW(dataset.configure_forecasting(0), runtime_error);
    EXPECT_THROW(dataset.configure_forecasting(2, -1), runtime_error);
    dataset.configure_forecasting(2);
    dataset.set_data_constant(1.0f);
    Batch batch(2, &dataset, {Device::CPU, Type::FP32, 0});
    dataset.configure_forecasting(2, 2, true);
    EXPECT_THROW(batch.fill({0, 1}, dataset.get_feature_selection()), runtime_error);
    dataset.set_shape(VariableRole::Target, {1});
    std::array<float, 4> targets{};
    EXPECT_THROW(dataset.fill_targets({0, 1}, {1}, targets.data(), FillMode::Inference),
                 runtime_error);
    dataset.clear_forecasting();
    EXPECT_THROW(batch.fill({0, 1}, dataset.get_feature_selection()), runtime_error);
}

TEST(TabularForecasting, RejectsOverflowingForecastingWindowWithoutChangingShape)
{
    TabularDataset dataset(12, {2}, {2});
    dataset.configure_forecasting(2, 1, true);
    EXPECT_THROW(dataset.configure_forecasting(2, numeric_limits<Index>::max() / 2 + 1, true),
                 runtime_error);
    EXPECT_EQ(dataset.get_future_time_steps(), 1);
    EXPECT_EQ(dataset.get_target_shape(), Shape({2}));
    EXPECT_THROW(dataset.configure_forecasting(numeric_limits<Index>::max()), runtime_error);
    EXPECT_EQ(dataset.get_past_time_steps(), 2);
    EXPECT_THROW(dataset.configure_forecasting(numeric_limits<Index>::max() / 2 + 1), runtime_error);
    EXPECT_EQ(dataset.get_input_shape(), Shape({2, 2}));
}

TEST(TabularForecasting, InterpolationRetainsTabularAndSequencePolicies)
{
    MatrixR raw(5, 3);
    raw << 0, QUIET_NAN, 100,
           QUIET_NAN, 5, QUIET_NAN,
           999, 999, 999,
           999, 999, 999,
           40, QUIET_NAN, 140;
    for (bool forecasting : {false, true})
    {
        TabularDataset dataset(5, {2}, {1});
        dataset.set_data(raw);
        if (forecasting) dataset.configure_forecasting(2);
        dataset.set_sample_roles(SampleRole::None);
        dataset.set_sample_roles({0, 1, 4}, SampleRole::Training);
        dataset.impute_missing_values_interpolate();
        EXPECT_FLOAT_EQ(dataset.get_data()(1, 0), forecasting ? 20.0f : 10.0f);
        EXPECT_FLOAT_EQ(dataset.get_data()(0, 1), 5.0f);
        EXPECT_FLOAT_EQ(dataset.get_data()(4, 1), 5.0f);
        if (forecasting)
        {
            EXPECT_FLOAT_EQ(dataset.get_data()(1, 2), 120.0f);
            EXPECT_EQ(dataset.get_sample_roles()[1], SampleRole::Training);
        }
        else
        {
            EXPECT_TRUE(isnan(dataset.get_data()(1, 2)));
            EXPECT_EQ(dataset.get_sample_roles()[1], SampleRole::None);
        }
    }
}

TEST(TabularForecasting, TrainingScalingExpandsMultiStepTargetsWithoutMutatingData)
{
    TabularDataset dataset(6, {1}, {1});
    MatrixR raw(6, 2);
    raw << 0.0f, 10.0f,
           1.0f, 20.0f,
           2.0f, 30.0f,
           3.0f, 40.0f,
           4.0f, 50.0f,
           5.0f, 60.0f;
    dataset.set_data(raw);
    dataset.set_variable_scalers("MinimumMaximum");
    dataset.configure_forecasting(2, 2, true);

    FeatureScaling requested;
    requested.descriptives.resize(2);
    requested.scalers.resize(2, ScalerMethod::None);

    const FeatureScaling effective = dataset.prepare_training_scaling(
        VariableRole::Target, requested, 2);

    ASSERT_EQ(effective.descriptives.size(), 2);
    ASSERT_EQ(effective.scalers.size(), 2);
    EXPECT_EQ(effective.scalers[0], ScalerMethod::MinimumMaximum);
    EXPECT_EQ(effective.scalers[1], ScalerMethod::MinimumMaximum);
    EXPECT_FLOAT_EQ(effective.descriptives[0].minimum,
                    effective.descriptives[1].minimum);
    EXPECT_FLOAT_EQ(effective.descriptives[0].maximum,
                    effective.descriptives[1].maximum);
    EXPECT_TRUE(dataset.Dataset::get_data().isApprox(raw, 0.0f));
}
