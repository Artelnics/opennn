//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N2O FORECAST EXAMPLE
//
//   Time-series forecasting of N2O emissions in a wastewater treatment plant.
//   Dataset: "Time Series Dataset for Modeling and Forecasting of N2O in
//   Wastewater Treatment" (2-min resampled, 521k samples, 24 columns).
//
//   Column suffix convention:
//     *_control          -> the committed past command at time t
//                           (fixed at Past during response optimization)
//     *_control_shifted  -> forward-shifted copy (= the command at t+1),
//                           FREE during response optimization -> this is what
//                           the optimizer tunes: the next-step action
//                           (action-conditioned forecasting / MPC setup)
//     *_state            -> measured disturbances / process variables
//     *_output           -> tank1_n2o_output, the quantity to forecast & minimize
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>
#include <string>
#include <filesystem>

#include "../../opennn/dataset.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/bounding_layer.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/model_selection.h"
#include "../../opennn/optimizer.h"
#include "../../opennn/variable.h"
#include "../../opennn/response_optimization.h"
#include "adaptive_moment_estimation.h"
#include "recurrent_layer.h"
#include "time_series_dataset.h"
#include "dense_layer.h"
#include "growing_neurons.h"
#include "normalized_squared_error.h"
#include "registry.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. N2O Forecast Example." << endl;

        // --- 1. DATASET ---

        TimeSeriesDataset time_series_dataset("../data/n2o_forecast.csv", ",", true, false);

        cout << "Dataset loaded." << endl;

        time_series_dataset.set_missing_values_method("Unuse");

        const Index num_lags    = 30;     // 30 x 2 min = 60 min history (one SBR phase)
        const Index max_epochs  = 100;

        time_series_dataset.set_past_time_steps(num_lags);
        time_series_dataset.set_future_time_steps(1);

        // Drop any lag-window that touches a NaN row (12.8% of samples are
        // clustered outages of inlet_q / tank airflows / tank2 valve).
        // Must be called AFTER set_past_time_steps so the full 30-step window is checked.
        time_series_dataset.impute_missing_values_unuse();

        time_series_dataset.set_variable_roles("None");

        // --- CONTROLS (6) -- directly manipulable actuators / setpoints ---
        time_series_dataset.set_variable_role("blower_airflow_control",     "Input");
        time_series_dataset.set_variable_role("tank1_valve_pct_control",    "Input");
        time_series_dataset.set_variable_role("tank1_o2_setpoint_control",  "Input");
        time_series_dataset.set_variable_role("tank2_valve_pct_control",    "Input");
        time_series_dataset.set_variable_role("tank2_o2_setpoint_control",  "Input");
        time_series_dataset.set_variable_role("phasecode_setpoint_control", "Input");

        // --- SHIFTED CONTROLS (6) -- forward-shifted (t+1); FREE in response opt ---
        time_series_dataset.set_variable_role("blower_airflow_control_shifted",     "Input");
        time_series_dataset.set_variable_role("tank1_valve_pct_control_shifted",    "Input");
        time_series_dataset.set_variable_role("tank1_o2_setpoint_control_shifted",  "Input");
        time_series_dataset.set_variable_role("tank2_valve_pct_control_shifted",    "Input");
        time_series_dataset.set_variable_role("tank2_o2_setpoint_control_shifted",  "Input");
        time_series_dataset.set_variable_role("phasecode_setpoint_control_shifted", "Input");

        // --- STATES (17) -- measured disturbances / process variables ---
        time_series_dataset.set_variable_role("tank1_nh4_state",          "Input");
        time_series_dataset.set_variable_role("tank1_no3_state",          "Input");
        time_series_dataset.set_variable_role("tank1_o2_state",           "Input");
        time_series_dataset.set_variable_role("tank1_processphase_state", "Input");
        time_series_dataset.set_variable_role("tank1_airflow_state",      "Input");
        time_series_dataset.set_variable_role("tank1_ss_state",           "Input");
        time_series_dataset.set_variable_role("tank1_temperature_state",  "Input");
        time_series_dataset.set_variable_role("tank2_o2_state",           "Input");
        time_series_dataset.set_variable_role("tank2_processphase_state", "Input");
        time_series_dataset.set_variable_role("tank2_airflow_state",      "Input");
        time_series_dataset.set_variable_role("tank2_ss_state",           "Input");
        time_series_dataset.set_variable_role("tank2_temperature_state",  "Input");
        time_series_dataset.set_variable_role("inlet_tank_phase_state",   "Input");
        time_series_dataset.set_variable_role("outlet_tank_phase_state",  "Input");
        time_series_dataset.set_variable_role("tank1_po4_state",          "Input");
        time_series_dataset.set_variable_role("inlet_q_state",            "Input");
        time_series_dataset.set_variable_role("swm_inlet_flow_state",     "Input");

        // --- OUTPUT -- lagged N2O also feeds back as input (InputTarget) ---
        time_series_dataset.set_variable_role("tank1_n2o_output", "InputTarget");

        const Index inputs_count  = time_series_dataset.get_features_number("Input");
        const Index targets_count = time_series_dataset.get_features_number("Target");

        time_series_dataset.set_shape("Input",  {num_lags, inputs_count});
        time_series_dataset.set_shape("Target", {targets_count});

        time_series_dataset.set_default_variable_scalers();
        time_series_dataset.split_samples_sequential(0.7, 0.15, 0.15);

        cout << "Inputs: "  << inputs_count
             << ", Targets: " << targets_count << endl;
        cout << "Used samples  -- Training: "
             << time_series_dataset.get_samples_number("Training")
             << "  Validation: "
             << time_series_dataset.get_samples_number("Validation")
             << "  Testing: "
             << time_series_dataset.get_samples_number("Testing") << endl;

        // --- 2. NEURAL NETWORK (growing neurons or load) ---

        const string neurons_result_file = "n2o_growing_neurons_result.xml";
        const string best_model_file     = "n2o_best_model.xml";

        Index best_neurons = 0;

        ForecastingNetwork* forecasting_network = nullptr;

        if (filesystem::exists(best_model_file))
        {
            cout << "Found saved model: " << best_model_file << ". Loading..." << endl;

            if (filesystem::exists(neurons_result_file))
            {
                ifstream nf(neurons_result_file);
                string line;
                while (getline(nf, line))
                {
                    if (line.find("<OptimalNeuronsNumber>") != string::npos)
                    {
                        size_t start = line.find(">") + 1;
                        size_t end   = line.find("</");
                        best_neurons = stoi(line.substr(start, end - start));
                        break;
                    }
                }
                nf.close();
            }

            if (best_neurons == 0) best_neurons = 35;

            forecasting_network = new ForecastingNetwork(time_series_dataset.get_input_shape(),
                                                          {best_neurons},
                                                          time_series_dataset.get_target_shape());

            forecasting_network->load(best_model_file);

            cout << "Model loaded with " << best_neurons << " neurons." << endl;
        }
        else
        {
            cout << "No saved model found. Running growing neurons (35-80, step 5, 2 trials)..." << endl;

            const Index initial_neurons = 35;

            forecasting_network = new ForecastingNetwork(time_series_dataset.get_input_shape(),
                                                          {initial_neurons},
                                                          time_series_dataset.get_target_shape());

            forecasting_network->set_input_names(time_series_dataset.get_feature_names("Input"));

            TrainingStrategy training_strategy(forecasting_network, &time_series_dataset);

            Registry<Loss>::instance().register_component("NormalizedSquaredError",
                [](){ return make_unique<NormalizedSquaredError>(); });
            training_strategy.set_loss("NormalizedSquaredError");

            AdaptiveMomentEstimation* adam = static_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
            adam->set_batch_size(16);
            adam->set_maximum_epochs(max_epochs);
            adam->set_display_period(32);
            adam->set_scaling();

            ModelSelection model_selection(&training_strategy);
            model_selection.set_neurons_selection("GrowingNeurons");

            GrowingNeurons* growing_neurons = static_cast<GrowingNeurons*>(model_selection.get_neurons_selection());
            growing_neurons->set_minimum_neurons(35);
            growing_neurons->set_maximum_neurons(80);
            growing_neurons->set_neurons_increment(5);
            growing_neurons->set_trials_number(2);
            growing_neurons->set_maximum_time(1e12);

            NeuronsSelectionResults neuron_results = model_selection.perform_neurons_selection();

            best_neurons = neuron_results.optimal_neurons_number;

            cout << "\n--- Growing Neurons Results ---" << endl;
            cout << "Best neurons: "          << best_neurons << endl;
            cout << "Best validation error: " << neuron_results.optimum_validation_error << endl;
            cout << "Stopped by: "            << neuron_results.write_stopping_condition() << endl;
            cout << "Elapsed: "               << neuron_results.elapsed_time << endl;

            ofstream nf(neurons_result_file);
            nf << "<GrowingNeuronsResult>\n";
            nf << "<OptimalNeuronsNumber>" << best_neurons << "</OptimalNeuronsNumber>\n";
            nf << "<OptimumValidationError>" << neuron_results.optimum_validation_error << "</OptimumValidationError>\n";
            nf << "</GrowingNeuronsResult>\n";
            nf.close();

            forecasting_network->save(best_model_file);

            cout << "Model saved to: "         << best_model_file     << endl;
            cout << "Neurons result saved to: " << neurons_result_file << endl;
        }

        // --- 3. TESTING ---

        cout << "\n--- Testing Analysis ---" << endl;
        cout.flush();

        try
        {
            TestingAnalysis testing_analysis(forecasting_network, &time_series_dataset);

            cout << "Parameters count: " << forecasting_network->get_parameters().size() << endl;
            cout << "Parameters norm: "  << forecasting_network->get_parameters().norm() << endl;
            cout.flush();

            auto [targets, outputs] = testing_analysis.get_targets_and_outputs("Testing");

            cout << "Targets: " << targets.rows() << "x" << targets.cols() << endl;
            cout << "Outputs: " << outputs.rows() << "x" << outputs.cols() << endl;
            cout.flush();

            VectorR errors = testing_analysis.calculate_errors("Testing");
            cout << "Testing MSE: " << errors[1] << endl;
            cout << "Testing NMSE: " << errors[3] << endl;
        }
        catch (exception& e)
        {
            cerr << "Testing failed: " << e.what() << endl;
        }

        // --- 4. RESPONSE OPTIMIZATION: MINIMIZE N2O ---

        cout << "\n--- Response Optimization (minimize N2O) ---" << endl;
        cout.flush();

        try
        {
            ResponseOptimization resp_opt(forecasting_network);

            const Index lags     = time_series_dataset.get_past_time_steps();
            const Index features = forecasting_network->get_input_shape().back();

            const Tensor3 latest_history = time_series_dataset.get_data("Testing", "Input")
                                                .slice(array_3(0, 0, 0), array_3(1, lags, features)).eval();

            resp_opt.set_fixed_history(latest_history);

            // Hold all 17 _state variables at their observed past trajectory
            resp_opt.set_condition("tank1_nh4_state",          ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank1_no3_state",          ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank1_o2_state",           ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank1_processphase_state", ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank1_airflow_state",      ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank1_ss_state",           ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank1_temperature_state",  ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank2_o2_state",           ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank2_processphase_state", ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank2_airflow_state",      ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank2_ss_state",           ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank2_temperature_state",  ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("inlet_tank_phase_state",   ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("outlet_tank_phase_state",  ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank1_po4_state",          ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("inlet_q_state",            ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("swm_inlet_flow_state",     ResponseOptimization::ConditionType::Past);

            // Hold the 6 raw controls (committed past trajectory at time t) at Past
            resp_opt.set_condition("blower_airflow_control",     ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank1_valve_pct_control",    ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank1_o2_setpoint_control",  ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank2_valve_pct_control",    ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("tank2_o2_setpoint_control",  ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("phasecode_setpoint_control", ResponseOptimization::ConditionType::Past);

            // The 6 *_control_shifted variables (= commands at t+1) are left
            // free (default) -- these are the next-step actions the optimizer
            // tunes to minimize tank1_n2o_output.

            // Minimize future N2O
            resp_opt.set_condition("tank1_n2o_output", ResponseOptimization::ConditionType::Minimize);

            resp_opt.set_iterations(15);
            resp_opt.set_evaluations_number(1000);
            resp_opt.set_zoom_factor(0.7);

            MatrixR results = resp_opt.perform_response_optimization();

            if (results.rows() > 0)
            {
                cout << "Optimal solution found:" << endl;
                auto var_names = time_series_dataset.get_variable_names();
                for (Index i = 0; i < results.cols(); ++i)
                    cout << var_names[i] << ": " << results(0, i) << endl;
            }
        }
        catch (exception& e)
        {
            cerr << "Response optimization failed: " << e.what() << endl;
        }

        delete forecasting_network;

        cout << "\nGood bye!" << endl;

        return 0;
    }
    catch (exception& e)
    {
        cerr << "Error: " << e.what() << endl;
    }
    return 0;
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2026 Artificial Intelligence Techniques SL
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
