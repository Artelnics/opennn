//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   WWTP OPTIMIZATION EXAMPLE
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
        cout << "OpenNN. Forecasting WWTP Optimization Example (Run 2)." << endl;

        // --- 1. DATASET ---

        TimeSeriesDataset time_series_dataset("../data/WWTP_PO4_NH4_removal.csv", ",", true, false);

        cout << "Dataset loaded." << endl;

        const Index num_lags = 2;
        const Index max_epochs = 100;

        // Configure Time Series parameters
        time_series_dataset.set_past_time_steps(num_lags);
        time_series_dataset.set_future_time_steps(1);

        // Define Variable Roles
        time_series_dataset.set_variable_roles("None");

        // EXTERNAL STATES (Inputs we cannot control)
        time_series_dataset.set_variable_role("FLOW_state", "Input");
        time_series_dataset.set_variable_role("TEMPERATURE_state", "Input");

        // CONTROLLABLE INPUTS (The knobs we can turn)
        time_series_dataset.set_variable_role("IRON_Input", "Input");
        time_series_dataset.set_variable_role("OXYGEN_input", "Input");
        time_series_dataset.set_variable_role("POLYALUMINUM_Input", "Input");
        time_series_dataset.set_variable_role("INLET_valve", "Input");
        time_series_dataset.set_variable_role("OUTLET_valve", "Input");

        // DECAY / DURATION COLUMNS (historical state)
        time_series_dataset.set_variable_role("IRON_Input_decay", "Input");
        time_series_dataset.set_variable_role("OXYGEN_input_decay", "Input");
        time_series_dataset.set_variable_role("POLYALUMINUM_Input_decay", "Input");
        time_series_dataset.set_variable_role("INLET_valve_duration", "Input");
        time_series_dataset.set_variable_role("OUTLET_valve_duration", "Input");

        // SHIFTED HISTORY (past values)
        time_series_dataset.set_variable_role("FLOW_shifted", "Input");
        time_series_dataset.set_variable_role("TEMPERATURE_shifted", "Input");
        time_series_dataset.set_variable_role("IRON_shifted", "Input");
        time_series_dataset.set_variable_role("OXYGEN_shifted", "Input");
        time_series_dataset.set_variable_role("POLYALUMINUM_shifted", "Input");
        time_series_dataset.set_variable_role("INLET_shifted", "Input");
        time_series_dataset.set_variable_role("OUTLETshifted", "Input");

        // TARGETS
        time_series_dataset.set_variable_role("OXYGEN_DEMAND_Output", "InputTarget");
        time_series_dataset.set_variable_role("NH4_Output", "InputTarget");
        time_series_dataset.set_variable_role("PO4_Output", "InputTarget");

        const Index inputs_count = time_series_dataset.get_features_number("Input");
        const Index targets_count = time_series_dataset.get_features_number("Target");

        time_series_dataset.set_shape("Input", {num_lags, inputs_count});
        time_series_dataset.set_shape("Target", {targets_count});

        time_series_dataset.set_default_variable_scalers();
        time_series_dataset.split_samples_sequential(0.7, 0.15, 0.15);

        cout << "Inputs: " << inputs_count << ", Targets: " << targets_count << endl;

        // --- 2. NEURAL NETWORK (growing neurons or load) ---

        const string neurons_result_file = "wwtp_growing_neurons_result_run2.xml";
        const string best_model_file = "wwtp_best_model_run2.xml";

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
                        size_t end = line.find("</");
                        best_neurons = stoi(line.substr(start, end - start));
                        break;
                    }
                }
                nf.close();
            }

            if (best_neurons == 0) best_neurons = 10;

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
            cout << "Best neurons: " << best_neurons << endl;
            cout << "Best validation error: " << neuron_results.optimum_validation_error << endl;
            cout << "Stopped by: " << neuron_results.write_stopping_condition() << endl;
            cout << "Elapsed: " << neuron_results.elapsed_time << endl;

            ofstream nf(neurons_result_file);
            nf << "<GrowingNeuronsResult>\n";
            nf << "<OptimalNeuronsNumber>" << best_neurons << "</OptimalNeuronsNumber>\n";
            nf << "<OptimumValidationError>" << neuron_results.optimum_validation_error << "</OptimumValidationError>\n";
            nf << "</GrowingNeuronsResult>\n";
            nf.close();

            forecasting_network->save(best_model_file);

            cout << "Model saved to: " << best_model_file << endl;
            cout << "Neurons result saved to: " << neurons_result_file << endl;
        }

        // --- 3. TESTING (always runs) ---

        cout << "\n--- Testing Analysis ---" << endl;
        cout.flush();

        try
        {
            TestingAnalysis testing_analysis(forecasting_network, &time_series_dataset);

            cout << "Parameters count: " << forecasting_network->get_parameters().size() << endl;
            cout << "Parameters norm: " << forecasting_network->get_parameters().norm() << endl;
            cout.flush();

            // Test with a small batch first
            const Tensor3 small_input = time_series_dataset.get_data("Testing", "Input")
                                            .slice(array_3(0, 0, 0), array_3(10, num_lags, inputs_count)).eval();
            cout << "Small input shape: " << small_input.dimension(0) << "x"
                 << small_input.dimension(1) << "x" << small_input.dimension(2) << endl;
            cout.flush();

            MatrixR small_output = forecasting_network->calculate_outputs(small_input);
            cout << "Small output OK: " << small_output.rows() << "x" << small_output.cols() << endl;
            cout.flush();

            // Test increasing batch sizes to find the limit
            for (Index test_batch : {100, 1000, 5000, 10000, 20000, 40000, 79000})
            {
                const Index actual_batch = min(test_batch, time_series_dataset.get_samples_number("Testing"));
                const Tensor3 test_input = time_series_dataset.get_data("Testing", "Input")
                                               .slice(array_3(0, 0, 0), array_3(actual_batch, num_lags, inputs_count)).eval();
                cout << "Testing batch " << actual_batch << "... " << flush;
                MatrixR test_output = forecasting_network->calculate_outputs(test_input);
                cout << "OK (" << test_output.rows() << "x" << test_output.cols() << ")" << endl;
                cout.flush();
            }

            cout << "Calling get_targets_and_outputs..." << endl;
            cout.flush();

            auto [targets, outputs] = testing_analysis.get_targets_and_outputs("Testing");

            cout << "Targets: " << targets.rows() << "x" << targets.cols() << endl;
            cout << "Outputs: " << outputs.rows() << "x" << outputs.cols() << endl;
            cout.flush();

            cout << "Calling calculate_errors..." << endl;
            cout.flush();

            VectorR errors = testing_analysis.calculate_errors("Testing");
            cout << "Testing MSE: " << errors[1] << endl;
            cout << "Testing NMSE: " << errors[3] << endl;
        }
        catch (exception& e)
        {
            cerr << "Testing failed: " << e.what() << endl;
        }

        // --- 4. RESPONSE OPTIMIZATION (always runs) ---

        cout << "\n--- Response Optimization ---" << endl;
        cout.flush();

        try
        {
            ResponseOptimization resp_opt(forecasting_network);

            const Index lags = time_series_dataset.get_past_time_steps();
            const Index features = forecasting_network->get_input_shape().back();

            const Tensor3 latest_history = time_series_dataset.get_data("Testing", "Input")
                                               .slice(array_3(0, 0, 0), array_3(1, lags, features)).eval();

            resp_opt.set_fixed_history(latest_history);

            // Past conditions (states + shifted + decays/durations)
            resp_opt.set_condition("FLOW_state", ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("TEMPERATURE_state", ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("IRON_Input_decay", ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("OXYGEN_input_decay", ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("POLYALUMINUM_Input_decay", ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("INLET_valve_duration", ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("OUTLET_valve_duration", ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("FLOW_shifted", ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("TEMPERATURE_shifted", ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("IRON_shifted", ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("OXYGEN_shifted", ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("POLYALUMINUM_shifted", ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("INLET_shifted", ResponseOptimization::ConditionType::Past);
            resp_opt.set_condition("OUTLETshifted", ResponseOptimization::ConditionType::Past);

            // Scenario: Minimize NH4
            cout << "\n[Scenario] Minimizing NH4 Output..." << endl;
            resp_opt.set_condition("NH4_Output", ResponseOptimization::ConditionType::Minimize);

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
