#include "pch.h"

#include "../opennn/time_series_data_set.h"

TEST(TimeSeriesDataSet, DefaultConstructor)
{

    TimeSeriesDataSet time_series_data_set;

    EXPECT_EQ(time_series_data_set.get_variables_number(), 0);
    EXPECT_EQ(time_series_data_set.get_samples_number(), 0);
}


TEST(TimeSeriesDataSet, GeneralConstructor)
{
/*
    TimeSeriesDataSet time_series_data_set_3(1, 1, 1);

    EXPECT_EQ(time_series_data_set.get_variables_number(), 2);
    EXPECT_EQ(time_series_data_set.get_samples_number(), 1);
    EXPECT_EQ(time_series_data_set.get_target_variables_number(), 1);
    EXPECT_EQ(time_series_data_set.get_input_variables_number(), 1);
*/
}


TEST(TimeSeriesDataSet, Autocorrelations)
{

    Tensor<type, 2> autocorrelations;

    Index samples_number = 1;
    Index inputs_number = 1;
    Index targets_number = 1;

    Index lags_number = 1;
    Index steps_ahead_number = 1;
/*
    data_set.set(samples_number, inputs_number, targets_number);

    data_set.set_lags_number(lags_number);
    data_set.set_steps_ahead_number(steps_ahead_number);

    data_set.transform_time_series();

    autocorrelations = data_set.calculate_autocorrelations(lags_number);

    EXPECT_EQ(autocorrelations.dimension(0), 2);
    EXPECT_EQ(autocorrelations.dimension(1), 1);
*/
}


TEST(TimeSeriesDataSet, CrossCorrelations)
{
/*
    Index lags_number;

    Tensor<type, 3> cross_correlations;

    // Test

    lags_number = 6;

    data.resize(6, 3);

    data.setValues({{type(5),type(2),type(8)},
                    {type(7),type(8),type(7)},
                    {type(3),type(6),type(4)},
                    {type(8),type(1),type(6)},
                    {type(5),type(8),type(6)},
                    {type(6),type(3),type(4)}});

    data_set.set_data(data);
    data_set.set_lags_number(lags_number);
    data_set.set_steps_ahead_number(1);

    data_set.transform_time_series();

    cross_correlations = data_set.calculate_cross_correlations(lags_number);

    EXPECT_EQ(cross_correlations.dimension(0), 3);
*/
}

/*
void TimeSeriesDataSet::test_transform_time_series()
{
    data.resize(9, 2);

    data.setValues({{1,10},
                    {2, 20},
                    {3, 30},
                    {4, 40},
                    {5, 50},
                    {6, 60},
                    {7, 70},
                    {8, 80},
                    {9, 90}});

    data_set.set_data(data);
/*
    data_set.set_variable_name(0, "x");
    data_set.set_variable_name(1, "y");

    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(1);

    data_set.transform_time_series();

    EXPECT_EQ(data_set.get_raw_variables_number() == 6);
    EXPECT_EQ(data_set.get_variables_number() == 6);
    EXPECT_EQ(data_set.get_samples_number() == 7);

    EXPECT_EQ(data_set.get_variables_number(DataSet::VariableUse::Input) == 4);
    EXPECT_EQ(data_set.get_variables_number(DataSet::VariableUse::Target) == 1);
    EXPECT_EQ(data_set.get_raw_variables_number(DataSet::VariableUse::Target) == 1);
    EXPECT_EQ(data_set.get_variables_number(DataSet::VariableUse::None) == 1);

    EXPECT_EQ(data_set.get_variable_name(0) == "x_lag_1");
    EXPECT_EQ(data_set.get_variable_name(1) == "y_lag_1");
    EXPECT_EQ(data_set.get_variable_name(2) == "x_lag_0");
    EXPECT_EQ(data_set.get_variable_name(3) == "y_lag_0");

}


void TimeSeriesDataSet::test_set_steps_ahead_number()
{
    data.resize(4,2);
    data.setValues({{type(0),type(0)},
                    {type(1),type(10)},
                    {type(2),type(20)},
                    {type(3),type(30)}});

    data_set.set_data(data);
    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(2);
    data_set.transform_time_series();

    EXPECT_EQ(data_set.get_lags_number(), 2);
}


void TimeSeriesDataSet::test_set_lags_number()
{
    // Test

    data.resize(4,2);
    data.setValues({{type(0),type(0)},
                    {type(1),type(10)},
                    {type(2),type(20)},
                    {type(3),type(30)}});

    data_set.set_data(data);
    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(2);
    data_set.transform_time_series();

    EXPECT_EQ(data_set.get_steps_ahead(), 2);
}
*/

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
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
