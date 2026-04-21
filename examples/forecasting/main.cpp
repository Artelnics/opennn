//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R E C A S T I N G   E X A M P L E   (M A D R I D   N O 2)
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "../../opennn/time_series_dataset.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/normalized_squared_error.h"
#include "../../opennn/training_strategy.h"
#include "adaptive_moment_estimation.h"
#include "quasi_newton_method.h"
#include "stochastic_gradient_descent.h"
#include "testing_analysis.h"
#include "recurrent_layer.h"

using namespace opennn;


// Configure dataset roles: ignore Date column, set chosen variables as
// InputTarget (so they are part of both the input window and the target).
static void configure_madrid_dataset(TimeSeriesDataset& dataset,
                                     const vector<string>& target_variable_names)
{
    const vector<string> all_variable_names = dataset.get_feature_names();

    for(const string& name : all_variable_names)
        if(name == "Date" || name.find("date") != string::npos)
            dataset.set_variable_role(name, "None");

    for(const string& name : all_variable_names)
    {
        if(name == "Date") continue;

        bool is_target = false;
        for(const string& t_name : target_variable_names)
            if(name == t_name) { is_target = true; break; }

        if(is_target)
            dataset.set_variable_role(name, "InputTarget");
        else
            dataset.set_variable_role(name, "Input");
    }
}


static void print_table_header(const vector<string>& column_headers,
                               const vector<Index>& widths)
{
    for(size_t i = 0; i < column_headers.size(); ++i)
        cout << left << setw(widths[i]) << column_headers[i];
    cout << endl;

    Index total_width = 0;
    for(Index w : widths) total_width += w;
    cout << string(total_width, '-') << endl;
}


static void run_scenario(const string& scenario_title,
                         const vector<string>& target_variable_names,
                         const Index past_time_steps,
                         const Index future_time_steps,
                         const bool multi_target,
                         const Index neurons_number = 30,
                         const Index epochs = 300,
                         const Index check_samples = 20)
{
    cout << "\n\n===========================================================\n"
         << " " << scenario_title << "\n"
         << "===========================================================" << endl;

    TimeSeriesDataset dataset("../data/madridNO2forecasting.csv",
                              ",", true, false);

    configure_madrid_dataset(dataset, target_variable_names);

    dataset.impute_missing_values_interpolate();

    dataset.set_multi_target(multi_target);
    dataset.set_past_time_steps(past_time_steps);
    dataset.set_future_time_steps(future_time_steps);

    cout << "\nConfiguration:" << endl;
    cout << "  Target variables   : ";
    for(const string& n : target_variable_names) cout << n << " ";
    cout << endl;
    cout << "  Past  time steps   : " << past_time_steps << endl;
    cout << "  Future time steps  : " << future_time_steps << endl;
    cout << "  multi_target       : " << boolalpha << multi_target << endl;
    cout << "  Input  shape       : [" << dataset.get_input_shape()[0]
         << "," << dataset.get_input_shape()[1] << "]" << endl;
    cout << "  Target shape       : [" << dataset.get_target_shape()[0] << "]" << endl;
    cout << "  #Input features    : " << dataset.get_features_number("Input") << endl;
    cout << "  #Target features   : " << dataset.get_features_number("Target") << endl;
    cout << "  Total samples      : " << dataset.get_samples_number() << endl;

    // Build the neural network
    ForecastingNetwork neural_network(dataset.get_input_shape(),
                                      {neurons_number},
                                      dataset.get_target_shape());
    neural_network.set_parameters_glorot();

    // Train
    TrainingStrategy training_strategy(&neural_network, &dataset);

    AdaptiveMomentEstimation* adam = static_cast<AdaptiveMomentEstimation*>(
        training_strategy.get_optimization_algorithm());
    adam->set_learning_rate(type(0.001));
    adam->set_batch_size(64);
    adam->set_maximum_epochs(epochs);
    adam->set_display_period(500);
    adam->set_display(false);

    cout << "\nTraining (" << epochs << " epochs)..." << endl;

    const TrainingResults results = training_strategy.train();

    cout << "Training finished. Final training error = "
         << fixed << setprecision(6)
         << results.get_training_error() << endl;

    // Compare real vs predicted on the first `check_samples` positions.
    // raw_data is the dataset matrix AFTER training-phase scaling (Optimizer
    // scales the data, then unscales it in set_unscaling at the end of train()).
    const MatrixR raw_data = dataset.Dataset::get_data();

    // Map target variable name -> column index in the data matrix
    vector<Index> target_columns;
    const vector<string> feature_names = dataset.get_feature_names();
    for(const string& t_name : target_variable_names)
    {
        Index idx = -1;
        for(Index k = 0; k < (Index)feature_names.size(); ++k)
            if(feature_names[k] == t_name) { idx = k; break; }
        target_columns.push_back(idx);
    }

    // Build the input window using all Input features
    const vector<Index> input_feature_indices = dataset.get_feature_indices("Input");
    const Index num_input_features = input_feature_indices.size();

    cout << "\nReal vs predicted values:" << endl;

    if(multi_target)
    {
        // multi_target=true: output layout (step-major) =
        //   [v0_t1, v1_t1, ..., vN_t1,
        //    v0_t2, v1_t2, ..., vN_t2,
        //    ...,
        //    v0_tF, v1_tF, ..., vN_tF]
        const Index n_vars = (Index)target_variable_names.size();
        print_table_header({"sample", "step", "variable", "real", "predicted", "abs_err"},
                           {10, 8, 12, 14, 14, 12});

        type total_abs_err = type(0);
        Index total_points = 0;

        // "present" timestep of the i-th checked sample.
        //   input  rows:  [present - past, ..., present - 1]
        //   target rows:  [present + 1, ..., present + future_time_steps]
        for(Index i = 0; i < check_samples; ++i)
        {
            const Index present = past_time_steps + i;

            Tensor3 input_window(1, past_time_steps, num_input_features);
            for(Index l = 0; l < past_time_steps; ++l)
                for(Index f = 0; f < num_input_features; ++f)
                    input_window(0, l, f) =
                        raw_data(present - past_time_steps + l, input_feature_indices[f]);

            const MatrixR prediction = neural_network.calculate_outputs(input_window);

            for(Index s = 0; s < future_time_steps; ++s)
            {
                for(Index v = 0; v < n_vars; ++v)
                {
                    const Index column = s * n_vars + v;
                    const type real_val = raw_data(present + s + 1, target_columns[v]);
                    const type pred_val = prediction(0, column);
                    const type abs_err  = std::abs(real_val - pred_val);
                    total_abs_err += abs_err;
                    total_points++;

                    cout << left << setw(10) << i
                         << setw(8)  << ("t+" + to_string(s + 1))
                         << setw(36) << target_variable_names[v]
                         << setw(14) << real_val
                         << setw(14) << pred_val
                         << setw(12) << abs_err << endl;
                }
            }
        }
        cout << "\nMean absolute error over " << total_points
             << " points = " << (total_abs_err / type(total_points)) << endl;
    }
    else
    {
        // multi_target=false: target is at row (present + future_time_steps),
        // output columns are the target variables in the same order.
        print_table_header({"sample", "step", "variable", "real", "predicted", "abs_err"},
                           {10, 8, 12, 14, 14, 12});

        type total_abs_err = type(0);
        Index total_points = 0;

        for(Index i = 0; i < check_samples; ++i)
        {
            const Index present = past_time_steps + i;

            Tensor3 input_window(1, past_time_steps, num_input_features);
            for(Index l = 0; l < past_time_steps; ++l)
                for(Index f = 0; f < num_input_features; ++f)
                    input_window(0, l, f) =
                        raw_data(present - past_time_steps + l, input_feature_indices[f]);

            const MatrixR prediction = neural_network.calculate_outputs(input_window);

            const Index target_row = present + future_time_steps;

            for(size_t v = 0; v < target_variable_names.size(); ++v)
            {
                const type real_val = raw_data(target_row, target_columns[v]);
                const type pred_val = prediction(0, (Index)v);
                const type abs_err  = std::abs(real_val - pred_val);
                total_abs_err += abs_err;
                total_points++;

                cout << left << setw(10) << i
                     << setw(8)  << ("t+" + to_string(future_time_steps))
                     << setw(12) << target_variable_names[v]
                     << setw(14) << real_val
                     << setw(14) << pred_val
                     << setw(12) << abs_err << endl;
            }
        }
        cout << "\nMean absolute error over " << total_points
             << " points = " << (total_abs_err / type(total_points)) << endl;
    }
}


int main()
{
    try
    {
        cout << "OpenNN. Madrid NO2 Forecasting Example." << endl;
        cout << fixed << setprecision(4);

        // --- Scenario 1: single target variable (NO2), one step ahead ----------
        run_scenario("Scenario 1 | single target = Average temperature | future = 1 | multi_target = false", { "Average temperature" }, 10, 1, false, 14, 3000);
        run_scenario("Scenario 2 | single target = NO2 | future = 1 | multi_target = false", { "NO2" }, 10, 1, false, 14, 1500);
        run_scenario("Scenario 3 | single target = O3 | future = 1 | multi_target = false", { "O3" }, 10, 1, false, 14, 1500);

        // --- Scenario 2: single target variable (NO2), multiple steps ahead ----
        //     multi_target=true with 1 variable -> outputs t+1..t+future for NO2
        run_scenario("Scenario 4 | single target = NO2 | future = 5 | multi_target = true", { "NO2" }, 20, 5, true, 18, 3000);
        run_scenario("Scenario 5 | single target = Average temperature | future = 5 | multi_target = true", { "Average temperature" }, 20, 5, true, 18, 3000);

        // --- Scenario 3: multiple target variables (NO2, O3), one step ahead ---
        run_scenario("Scenario 6 | targets = {NO2, O3} | future = 1 | multi_target = false", { "NO2", "O3" }, 20, 1, false, 30, 5000);
        run_scenario("Scenario 7 | targets = {NO2, O3, Average temperature} | future = 1 | multi_target = false", {"Average temperature","NO2", "O3", }, 20, 1, false, 30, 5000);

        // --- Scenario 4: multiple target variables AND multiple steps ahead ----
        //     multi_target=true with N variables and F future steps ->
        //     the same network outputs N*F values at once (step-major layout):
        //     [NO2_t1, O3_t1, NO2_t2, O3_t2, NO2_t3, O3_t3]
        run_scenario("Scenario 8 | targets = {NO2, O3} | future = 4 | multi_target = true", { "NO2", "O3" }, 30, 4, true, 50, 8000);
        run_scenario("Scenario 9 | targets = {NO2, O3, Average temperature} | future = 4 | multi_target = true", {"Average temperature", "NO2", "O3", }, 30, 4, true, 60, 10000);

        cout << "\nGood bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
