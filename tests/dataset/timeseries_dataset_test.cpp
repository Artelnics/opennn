#include "tests/pch.h"

#include "opennn/dataset/time_series_dataset.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/json.h"

using namespace opennn;

TEST(TimeSeriesDataset, DefaultConstructor)
{
    TimeSeriesDataset time_series_data_set;

    EXPECT_EQ(time_series_data_set.get_variables_number(), 0);
    EXPECT_EQ(time_series_data_set.get_samples_number(), 0);
}

TEST(TimeSeriesDataset, GeneralConstructor)
{
    Shape input_shape = { 1 };
    Shape target_shape = { 1 };

    TimeSeriesDataset time_series_data_set_3(1, input_shape, target_shape);

    EXPECT_EQ(time_series_data_set_3.get_variables_number(), 2);
    EXPECT_EQ(time_series_data_set_3.get_samples_number(), 1);
}

TEST(TimeSeriesDataset, Autocorrelations)
{
    TimeSeriesDataset dataset;
    dataset.set_display(false);

    MatrixR autocorrelations;

    Index samples_number = 1;
    Shape inputs_number = { 1 };
    Shape targets_number ={ 1 };

    Index lags_number = 1;
    Index steps_ahead_number = 1;

    dataset.set(samples_number, inputs_number, targets_number);

    dataset.set_past_time_steps(lags_number);
    dataset.set_future_time_steps(steps_ahead_number);

    autocorrelations = dataset.calculate_autocorrelations(lags_number);

    EXPECT_EQ(autocorrelations.rows(), 2);
    EXPECT_EQ(autocorrelations.cols(), 1);
}

TEST(TimeSeriesDataset, CrossCorrelations)
{

    Shape input_shape = { 2 };
    Shape target_shape = { 1 };

    TimeSeriesDataset dataset(6, input_shape, target_shape);
    dataset.set_display(false);

    Index lags_number;

    Tensor3 cross_correlations;

    MatrixR data;

    lags_number = 6;

    data.resize(6, 3);

    data << type(5),type(2),type(8),
            type(7),type(8),type(7),
            type(3),type(6),type(4),
            type(8),type(1),type(6),
            type(5),type(8),type(6),
            type(6),type(3),type(4);

    dataset.set_data(data);
    dataset.set_past_time_steps(lags_number);
    dataset.set_future_time_steps(1);

    cross_correlations = dataset.calculate_cross_correlations(lags_number);

    EXPECT_EQ(cross_correlations.dimension(0), 3);

}

TEST(TimeSeriesDataset, test_transform_time_series)
{
    Shape input_shape = { 1 };
    Shape target_shape = { 1 };

    TimeSeriesDataset dataset(9, input_shape, target_shape);

    MatrixR data;

    data.resize(9, 2);

    data << 1, 10,
            2, 20,
            3, 30,
            4, 40,
            5, 50,
            6, 60,
            7, 70,
            8, 80,
            9, 90;

    dataset.set_data(data);

    vector<string> variable_names = { "x", "y" };

    dataset.set_variable_names(variable_names);

    dataset.set_past_time_steps(2);
    dataset.set_future_time_steps(1);

    EXPECT_EQ(dataset.get_variables_number(), 2);
    EXPECT_EQ(dataset.get_variables_number(), 2);
    EXPECT_EQ(dataset.get_samples_number(), 9);

    EXPECT_EQ(dataset.get_variables_number("Input"), 1);
    EXPECT_EQ(dataset.get_variables_number("Target"), 1);
    EXPECT_EQ(dataset.get_variables_number("Target"), 1);
    EXPECT_EQ(dataset.get_variables_number("None"), 0);

    vector<string> input_variable_names = dataset.get_variable_names("Input");
    vector<string> target_variable_names = dataset.get_variable_names("Target");

    EXPECT_EQ(input_variable_names[0], "x");
    EXPECT_EQ(target_variable_names[0], "y");

}

TEST(TimeSeriesDataset, test_set_steps_ahead_number)
{

    Shape input_shape = { 1 };
    Shape target_shape = { 1 };

    TimeSeriesDataset dataset(4, input_shape, target_shape);

    MatrixR data;
    
    data.resize(4, 2);
    data << type(0),type(0),
            type(1),type(10),
            type(2),type(20),
            type(3),type(30);

    dataset.set_data(data);
    dataset.set_past_time_steps(2);
    dataset.set_future_time_steps(2);

    EXPECT_EQ(dataset.get_past_time_steps(), 2);

}

TEST(TimeSeriesDataset, test_set_lags_number)
{

    Shape input_shape = { 1 };
    Shape target_shape = { 1 };

    TimeSeriesDataset dataset(4, input_shape, target_shape);

    MatrixR data;

    data.resize(4, 2);
    data << type(0),type(0),
            type(1),type(10),
            type(2),type(20),
            type(3),type(30);

    dataset.set_data(data);
    dataset.set_past_time_steps(2);
    dataset.set_future_time_steps(2);

    EXPECT_EQ(dataset.get_future_time_steps(), 2);

}

TEST(TimeSeriesDataset, ModelInputVariablesReflectForecastingWindow)
{
    TimeSeriesDataset dataset(4, {2}, {1});
    dataset.set_variable_names({"temperature", "pressure", "target"});
    dataset.set_past_time_steps(3);
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

TEST(TimeSeriesDataset, MultiTargetJsonRoundTripPreservesTargetLayout)
{
    TimeSeriesDataset original(12, {1}, {2});
    original.set_past_time_steps(2);
    original.set_future_time_steps(2);
    original.set_multi_target(true);
    JsonWriter writer;
    original.to_JSON(writer);
    JsonDocument document;
    document.set_root(Json::parse(writer.c_str()));

    MatrixR raw(12, 3);
    for (Index row = 0; row < raw.rows(); ++row)
        raw.row(row) << float(row), float(100 + row), float(1000 + row);

    const auto check = [&](TimeSeriesDataset& dataset)
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

    TimeSeriesDataset fresh;
    check(fresh);
    check(original);
}

TEST(TimeSeriesDataset, LegacyJsonDefaultsToSingleTargetOnReusedDataset)
{
    TimeSeriesDataset dataset(12, {1}, {1});
    dataset.set_future_time_steps(2);
    dataset.set_multi_target(true);
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

TEST(TimeSeriesDataset, RejectsInvalidWindowAndStaleBatchShape)
{
    TimeSeriesDataset dataset(12, {1}, {1});
    EXPECT_THROW(dataset.set_past_time_steps(0), runtime_error);
    EXPECT_THROW(dataset.set_future_time_steps(-1), runtime_error);
    dataset.set_past_time_steps(2);
    dataset.set_data_constant(1.0f);
    Batch batch(2, &dataset, {Device::CPU, Type::FP32, 0});
    dataset.set_future_time_steps(2);
    dataset.set_multi_target(true);
    EXPECT_THROW(batch.fill({0, 1}, dataset.get_feature_selection()), runtime_error);
    dataset.set_shape(VariableRole::Target, {1});
    std::array<float, 4> targets{};
    EXPECT_THROW(dataset.fill_targets({0, 1}, {1}, targets.data(), FillMode::Inference),
                 runtime_error);
}

TEST(TimeSeriesDataset, RejectsOverflowingForecastingWindowWithoutChangingShape)
{
    TimeSeriesDataset dataset(12, {1}, {2});
    dataset.set_multi_target(true);
    EXPECT_THROW(dataset.set_future_time_steps(numeric_limits<Index>::max() / 2 + 1),
                 runtime_error);
    EXPECT_EQ(dataset.get_future_time_steps(), 1);
    EXPECT_EQ(dataset.get_target_shape(), Shape({2}));
    EXPECT_THROW(dataset.set_past_time_steps(numeric_limits<Index>::max()), runtime_error);
    EXPECT_EQ(dataset.get_past_time_steps(), 2);
}

TEST(TimeSeriesDataset, TrainingScalingExpandsMultiStepTargetsWithoutMutatingData)
{
    TimeSeriesDataset dataset(6, {1}, {1});
    MatrixR raw(6, 2);
    raw << 0.0f, 10.0f,
           1.0f, 20.0f,
           2.0f, 30.0f,
           3.0f, 40.0f,
           4.0f, 50.0f,
           5.0f, 60.0f;
    dataset.set_data(raw);
    dataset.set_variable_scalers("MinimumMaximum");
    dataset.set_future_time_steps(2);
    dataset.set_multi_target(true);

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

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
