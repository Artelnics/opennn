//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T I M E   S E R I E S   D A T A   S E T   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "time_series_data_set_test.h"

#include "../opennn/time_series_data_set.h"

namespace opennn
{

TimeSeriesDataSetTest::TimeSeriesDataSetTest() : UnitTesting()
{
    data_set.set_display(false);
}


TimeSeriesDataSetTest::~TimeSeriesDataSetTest()
{
}


void TimeSeriesDataSetTest::test_constructor()
{
    cout << "test_constructor\n";

    // Default constructor

    TimeSeriesDataSet data_set_1;

    assert_true(data_set_1.get_variables_number() == 0, LOG);
    assert_true(data_set_1.get_samples_number() == 0, LOG);

    /*
    // Samples and variables number constructor

    TimeSeriesDataSet data_set_2(1, 2);

    assert_true(data_set_2.get_samples_number() == 1, LOG);
    assert_true(data_set_2.get_variables_number() == 2, LOG);

    // Inputs, targets and samples numbers constructor

    TimeSeriesDataSet data_set_3(1, 1, 1);

    assert_true(data_set_3.get_variables_number() == 2, LOG);
    assert_true(data_set_3.get_samples_number() == 1, LOG);
    assert_true(data_set_3.get_target_variables_number() == 1,LOG);
    assert_true(data_set_3.get_input_variables_number() == 1,LOG);
*/
}


void TimeSeriesDataSetTest::test_destructor()
{
    cout << "test_destructor\n";

    TimeSeriesDataSet* data_set = new TimeSeriesDataSet();
    delete data_set;
}


void TimeSeriesDataSetTest::test_calculate_autocorrelations()
{
    cout << "test_calculate_autocorrelations\n";

    Tensor<type, 2> autocorrelations;

    Index samples_number;
    Index inputs_number;
    Index targets_number;

    Index lags_number;
    Index steps_ahead_number;

    // Test

    samples_number = 1;
    inputs_number = 1;
    targets_number = 1;

    lags_number = 1;
    steps_ahead_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);

    data_set.set_lags_number(lags_number);
    data_set.set_steps_ahead_number(steps_ahead_number);

    data_set.transform_time_series();
/*
    autocorrelations = data_set.calculate_autocorrelations(lags_number);
*/
    assert_true(autocorrelations.dimension(0) == 2, LOG);
    assert_true(autocorrelations.dimension(1) == 1, LOG);

}


void TimeSeriesDataSetTest::test_calculate_cross_correlations()
{
    cout << "test_calculate_cross_correlations\n";

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
/*
    cross_correlations = data_set.calculate_cross_correlations(lags_number);
*/
    assert_true(cross_correlations.dimension(0) == 3, LOG);

}


void TimeSeriesDataSetTest::test_transform_time_series()
{
    cout << "test_transform_time_series\n";

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

    data_set.set_variable_name(0, "x");
    data_set.set_variable_name(1, "y");

    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(1);

    data_set.transform_time_series();

    assert_true(data_set.get_raw_variables_number() == 6, LOG);
    assert_true(data_set.get_variables_number() == 6, LOG);
    assert_true(data_set.get_samples_number() == 7, LOG);

    assert_true(data_set.get_input_variables_number() == 4, LOG);
    assert_true(data_set.get_target_variables_number() == 1, LOG);
    assert_true(data_set.get_target_raw_variables_number() == 1, LOG);
    assert_true(data_set.get_unused_variables_number() == 1, LOG);

    assert_true(data_set.get_variable_name(0) == "x_lag_1", LOG);
    assert_true(data_set.get_variable_name(1) == "y_lag_1", LOG);
    assert_true(data_set.get_variable_name(2) == "x_lag_0", LOG);
    assert_true(data_set.get_variable_name(3) == "y_lag_0", LOG);
}


void TimeSeriesDataSetTest::test_set_time_series_data()
{
    cout << "test_set_time_series_data\n";

    data.resize(4,2);

    data.setValues({{type(0),type(0)},
                    {type(1),type(10)},
                    {type(2),type(20)},
                    {type(3),type(30)}});

    data_set.set_data(data);

    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(2);

    data_set.transform_time_series();

    data.resize(5,3);
    data.setValues({{type(15),type(14),type(13)},
                    {type(12),type(11),type(10)},
                    {type(9),type(8),type(7)},
                    {type(6),type(5),type(4)},
                    {type(3),type(2),type(1)}});

    data_set.set_time_series_data(data);

    /*assert_true(data_set.get_time_series_data()(0) - type(15) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(data_set.get_time_series_data()(1) - type(12) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(data_set.get_time_series_data()(2) - type(9) < type(NUMERIC_LIMITS_MIN), LOG);*/
}


void TimeSeriesDataSetTest::test_save_time_series_data_binary()
{
    cout << "test_save_time_series_data_binary\n";

    const string data_path = "../data/test";

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
    /*
    data_set.set_data_source_path(data_path);
    */
    data_set.save_time_series_data_binary(data_path);
    data_set.load_time_series_data_binary(data_path);
    /*
    assert_true(data_set.get_time_series_data()(0) - type(0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(data_set.get_time_series_data()(1) - type(1) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(data_set.get_time_series_data()(2) - type(2) < type(NUMERIC_LIMITS_MIN), LOG);
    */
}


void TimeSeriesDataSetTest::test_set_steps_ahead_number()
{
    cout << "test_set_steps_ahead_nuber\n";

    data.resize(4,2);
    data.setValues({{type(0),type(0)},
                    {type(1),type(10)},
                    {type(2),type(20)},
                    {type(3),type(30)}});

    data_set.set_data(data);
    data_set.set_lags_number(2);
    data_set.set_steps_ahead_number(2);
    data_set.transform_time_series();

    assert_true(data_set.get_lags_number() == 2, LOG);
}


void TimeSeriesDataSetTest::test_set_lags_number()
{
    cout << "test_set_lags_number\n";

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

    assert_true(data_set.get_steps_ahead() == 2, LOG);
}


void TimeSeriesDataSetTest::run_test_case()
{
    cout << "Running time series data set test case...\n";

    // Constructor and destructor

    test_constructor();
    test_destructor();

    // Correlations

    test_calculate_autocorrelations();
    test_calculate_cross_correlations();

    // Transform

    test_transform_time_series();

    // Set series

    test_set_time_series_data();
    test_set_steps_ahead_number();
    test_set_lags_number();

    // Saving

    test_save_time_series_data_binary();
    //test_has_time_raw_variables();

    cout << "End of time series data set test case.\n\n";
}

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
