#include "pch.h"

#include "../opennn/time_series_dataset.h"
#include "../opennn/tensor_utilities.h"

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
    //EXPECT_EQ(time_series_data_set_3.get_target_variables_number(), 1);
    //EXPECT_EQ(time_series_data_set_3.get_input_variables_number(), 1);
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
    Shape target_shape = { 2 };

    TimeSeriesDataset dataset(6, input_shape, target_shape);
    dataset.set_display(false);

    Index lags_number;

    Tensor3 cross_correlations;

    MatrixR data;

    // Test

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
    Shape target_shape = { 2 };

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

    std::vector<string> variable_names = { "x", "y" };

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

    std::vector<string> input_variable_names = dataset.get_variable_names("Input");
    std::vector<string> target_variable_names = dataset.get_variable_names("Target");

    EXPECT_EQ(input_variable_names[0], "x");
    EXPECT_EQ(target_variable_names[0], "y");

}


TEST(TimeSeriesDataset, test_set_steps_ahead_number)
{

    Shape input_shape = { 1 };
    Shape target_shape = { 2 };

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
    Shape target_shape = { 2 };

    TimeSeriesDataset dataset(4, input_shape, target_shape);

    MatrixR data;
    // Test

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


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.
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
