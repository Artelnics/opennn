#include "pch.h"

#include "../opennn/time_series_data_set.h"
#include "../opennn/tensors.h"

TEST(TimeSeriesDataSet, DefaultConstructor)
{

    TimeSeriesDataSet time_series_data_set;

    EXPECT_EQ(time_series_data_set.get_variables_number(), 0);
    EXPECT_EQ(time_series_data_set.get_samples_number(), 0);
}


TEST(TimeSeriesDataSet, GeneralConstructor)
{
    dimensions input_dimensions = { 1 };
    dimensions target_dimensions = { 1 }; 

    TimeSeriesDataSet time_series_data_set_3(1, input_dimensions, target_dimensions);

    EXPECT_EQ(time_series_data_set_3.get_variables_number(), 2);
    EXPECT_EQ(time_series_data_set_3.get_samples_number(), 1);
    //EXPECT_EQ(time_series_data_set_3.get_target_variables_number(), 1);
    //EXPECT_EQ(time_series_data_set_3.get_input_variables_number(), 1); 

}


TEST(TimeSeriesDataSet, Autocorrelations)
{
    TimeSeriesDataSet data_set;

    Tensor<type, 2> autocorrelations;

    Index samples_number = 1;
    dimensions inputs_number = { 1 };
    dimensions targets_number ={ 1 };

    Index lags_number = 1;
    Index steps_ahead_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);

    data_set.set_lags_number(lags_number);
    data_set.set_steps_ahead_number(steps_ahead_number);

    //data_set.transform_time_series();

    //autocorrelations = data_set.calculate_autocorrelations(lags_number);

    //EXPECT_EQ(autocorrelations.dimension(0), 2);
    //EXPECT_EQ(autocorrelations.dimension(1), 1);

}


TEST(TimeSeriesDataSet, CrossCorrelations)
{
    dimensions input_dimensions = { 2 };
    dimensions target_dimensions = { 2 };

    TimeSeriesDataSet data_set(6, input_dimensions, target_dimensions);
    
    Index lags_number;

    Tensor<type, 3> cross_correlations;

    Tensor<type, 2> data;

    // Test

    lags_number = 6;

    data.resize(6, 3);

    data.setValues({ {type(5),type(2),type(8)},
                    {type(7),type(8),type(7)},
                    {type(3),type(6),type(4)},
                    {type(8),type(1),type(6)},
                    {type(5),type(8),type(6)},
                    {type(6),type(3),type(4)} });

    data_set.set_data(data);
    data_set.set_lags_number(lags_number);
    data_set.set_steps_ahead_number(1);

    //data_set.transform_time_series();

    //cross_correlations = data_set.calculate_cross_correlations(lags_number);

    //EXPECT_EQ(cross_correlations.dimension(0), 3);

}

TEST(TimeSeriesDataSet, test_transform_time_series) {

    dimensions input_dimensions = { 1 };
    dimensions target_dimensions = { 2 };

    TimeSeriesDataSet data_set(9, input_dimensions, target_dimensions);

    Tensor<type, 2> data;

    data.resize(9, 2);

    data.setValues({ {1,10},
                    {2, 20},
                    {3, 30},
                    {4, 40},
                    {5, 50},
                    {6, 60},
                    {7, 70},
                    {8, 80},
                    {9, 90} });

    data_set.set_data(data);

    std::vector<std::string> variable_names = { "x", "y" };

    data_set.set_variable_names(variable_names);

    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(1);

    //data_set.transform_time_series();

    EXPECT_EQ(data_set.get_raw_variables_number(), 2);
    EXPECT_EQ(data_set.get_variables_number(), 2);
    EXPECT_EQ(data_set.get_samples_number(), 9);

    EXPECT_EQ(data_set.get_variables_number(DataSet::VariableUse::Input), 1);
    EXPECT_EQ(data_set.get_variables_number(DataSet::VariableUse::Target), 1);
    EXPECT_EQ(data_set.get_raw_variables_number(DataSet::VariableUse::Target), 1);
    EXPECT_EQ(data_set.get_variables_number(DataSet::VariableUse::None), 0);

    std::vector<std::string> input_variable_names = data_set.get_variable_names(DataSet::VariableUse::Input);

    //EXPECT_EQ(input_variable_names[0], "x_lag_1");
    //EXPECT_EQ(input_variable_names[1], "y_lag_1");
    //EXPECT_EQ(input_variable_names[2], "x_lag_0");
    //EXPECT_EQ(input_variable_names[3], "y_lag_0");
    
}

TEST(TimeSeriesDataSet, test_set_steps_ahead_number)
{
    dimensions input_dimensions = { 1 };
    dimensions target_dimensions = { 2 };

    TimeSeriesDataSet data_set(4, input_dimensions, target_dimensions);

    Tensor<type, 2> data;
    
    data.resize(4, 2);
    data.setValues({ {type(0),type(0)},
                    {type(1),type(10)},
                    {type(2),type(20)},
                    {type(3),type(30)} });

    data_set.set_data(data);
    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(2);
    //data_set.transform_time_series();

    EXPECT_EQ(data_set.get_lags_number(), 2);
}

TEST(TimeSeriesDataSet, test_set_lags_number) {
    dimensions input_dimensions = { 1 };
    dimensions target_dimensions = { 2 };

    TimeSeriesDataSet data_set(4, input_dimensions, target_dimensions);

    Tensor<type, 2> data;
    // Test

    data.resize(4, 2);
    data.setValues({ {type(0),type(0)},
                    {type(1),type(10)},
                    {type(2),type(20)},
                    {type(3),type(30)} });

    data_set.set_data(data);
    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(2);
    //data_set.transform_time_series();

    EXPECT_EQ(data_set.get_steps_ahead(), 2);
}


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
