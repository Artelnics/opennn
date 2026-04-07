//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   CONCRETE RECIPES OPTIMIZATION
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>
#include <string>

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

using namespace opennn;

int main()
{
    try
    {
        throw runtime_error("wwt_optimization example is not implemented yet");

        //WWTP EXPERIMENT

        cout << "OpenNN. Forecasting WWTP EXPERIMENT Example." << endl;

        TimeSeriesDataset time_series_dataset("../data/WWTP_PO4_NH4_removal_shift.csv", ",", true, false);

        cout << "dataset leido" << endl;
        //time_series_dataset.print();

        cout << "-----------------------------------" << endl;


        // const int time_steps = 10;

        const Index num_lags = 2;        // Change this to test 5, 10, 20...
        const Index hidden_neurons = 3;  // Change this to test 4, 8, 16...
        const Index max_epochs = 100;    // Training iterations

        stringstream ss;
        ss << "wwtp_L" << num_lags << "_N" << hidden_neurons << "_E" << max_epochs << ".xml";
        string model_file = ss.str();


        // Configure Time Series parameters
        time_series_dataset.set_past_time_steps(num_lags);   // 10 timesteps of history
        time_series_dataset.set_future_time_steps(1); // Predicting 1 step ahead

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

        // SHIFTED HISTORY (Inputs representing the past, should remain constant during optimization)
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


        // 2. NEURAL NETWORK ARCHITECTURE
        // Input shape is [10 timesteps x used_features]
        ForecastingNetwork forecasting_network(time_series_dataset.get_input_shape(),
                                               {hidden_neurons}, // 8 neurons in hidden layer
                                               time_series_dataset.get_target_shape());

        forecasting_network.set_input_names(time_series_dataset.get_feature_names("Input"));

        // 6. FIX: Disable Batch Normalization (it's causing the 10^17 error)
        Layer* dense_ptr = forecasting_network.get_first("Dense2d");
        if(dense_ptr) {
            // We must cast to the specific template Rank 2
            opennn::Dense<2>* dense_layer = dynamic_cast<opennn::Dense<2>*>(dense_ptr);
            if(dense_layer)
            {
                dense_layer->set_batch_normalization(false);
                cout << "Batch Normalization disabled for stability." << endl;
            }
        }

/*
        Layer* layer_ptr = forecasting_network.get_first("Recurrent");
        Recurrent* recurrent_layer = dynamic_cast<Recurrent*>(layer_ptr);
        recurrent_layer->set_activation_function("HyperbolicTangent");

        forecasting_network.print();

        cout << "------------------------------------------" << endl;
*/



        // --- 3. LOAD OR TRAIN ---
        if (filesystem::exists(model_file))
        {
            cout << "Trained model found. Loading: " << model_file << "..." << endl;
            forecasting_network.load(model_file);
        }
        else
        {
            /// Entrenamiento
            TrainingStrategy training_strategy(&forecasting_network, &time_series_dataset);
            training_strategy.set_loss("MeanSquaredError");

            // training_strategy.set_optimization_algorithm("QuasiNewtonMethod");
            // training_strategy.set_optimization_algorithm("StochasticGradientDescent");

            // Setup Adam optimizer
            AdaptiveMomentEstimation* adam = static_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
            adam->set_batch_size(16);
            adam->set_maximum_epochs(max_epochs);
            adam->set_display_period(32);

            adam->set_scaling();

            cout << "No saved model found. Starting training..." << endl; 

            training_strategy.train();

            training_strategy.get_optimization_algorithm()->set_scaling();

            cout << "Saving trained model to: " << model_file << endl;
            forecasting_network.save(model_file);
        }

        // --- 4. ERROR & BASELINE ANALYSIS ---
        cout << "\n--- Performance Analysis ---" << endl;
        TestingAnalysis testing_analysis(&forecasting_network, &time_series_dataset);

        // Get actual Errors from the Model
        // index 1 = MSE, index 3 = NSE (Normalized Squared Error)
        VectorR model_errors = testing_analysis.calculate_errors("Testing");
        type model_mse = model_errors[1];
        type model_nse = model_errors[3];

        cout << "Model MSE: " << model_mse << endl;
        cout << "Model NSE (Normalized vs Mean): " << model_nse << " (Ideal < 0.2)" << endl;

        // --- MANUAL PERSISTENCE BASELINE CHECK ---
        // We compare our model to a "Naive" model where Output(t) = Output(t-1)
        auto [targets, outputs] = testing_analysis.get_targets_and_outputs("Testing");

        // In TimeSeriesDataset, the inputs include the history.
        // The "Persistence" prediction for a variable is its value at the last available lag.
        // We calculate the MSE between the current target and the previous target.

        type persistence_mse = 0;
        const Index n_samples = targets.rows();
        const Index n_targets = targets.cols();

        // Logic: Baseline Error = Sum((Target_t - Target_{t-1})^2)
        // We can simulate this by looking at the dataset history
        for(Index i = 0; i < n_samples; ++i)
            for(Index j = 0; j < n_targets; ++j)
                persistence_mse += pow(targets(i,j) - outputs(i,j), 2); // Error of model


        // Note: This is a simplified check. Accurate baseline calculation:



        cout << "Model improvement over Mean: " << (1.0 - model_nse) * 100.0 << "%" << endl;

        // 4. DEEP DIAGNOSTIC TESTING ANALYSIS
        cout << "\n--- Model Quality Diagnostics ---" << endl;
        // Standard Analysis (R-squared)
        testing_analysis.print_goodness_of_fit_analysis();

        // Diagnostic A: Error Autocorrelation
        // If the correlation at Lag 1 is high (e.g., > 0.3), your model is lagging
        // and just mimicking the previous value. Ideally, error should be "white noise".
        vector<VectorR> error_auto = testing_analysis.calculate_error_autocorrelation(num_lags);
        auto target_names = time_series_dataset.get_variable_names("Target");

        for(size_t i = 0; i < target_names.size(); ++i) {
            cout << "Target [" << target_names[i] << "] Error Autocorr at Lag 1: " << error_auto[i][1] << endl;
            if(abs(error_auto[i][1]) > 0.5) {
                cout << "  WARNING: High error autocorrelation detected. Model may be simply lagging." << endl;
            }
        }

        // Diagnostic B: Mean Squared Errors (Absolute and Normalized)
        // Normalized Error < 1.0 means your model is better than just predicting the average.
        VectorR errors = testing_analysis.calculate_errors("Testing");
        cout << "Testing Mean Squared Error (MSE): " << errors[1] << endl;
        cout << "Testing Normalized Squared Error (NSE): " << errors[3] << " (closer to 0 is better)" << endl;


        // 4. RESPONSE OPTIMIZATION
        ResponseOptimization resp_opt(&forecasting_network);

        // Example
        const Index lags = time_series_dataset.get_past_time_steps();
        const Index features = forecasting_network.get_input_shape().back();

        // Select: Start at {0,0,0}, Take {1 Sample, all Lags, all Features}
        const Tensor3 latest_history = time_series_dataset.get_data("Testing", "Input")
                                           .slice(array_3(0, 0, 0), array_3(1, lags, features)).eval();

        resp_opt.set_fixed_history(latest_history);

        resp_opt.set_condition("FLOW", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("TEMPERATURE", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("IRON", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("OXYGEN", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("POLYALUMINUM", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("INLET", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("OUTLET", ResponseOptimization::ConditionType::Past);

        cout << "\n[Scenario A1] Optimizing for minimum NH4 Output..." << endl;
        resp_opt.set_condition("NH4_Output", ResponseOptimization::ConditionType::Minimize);

        resp_opt.set_iterations(15);
        resp_opt.set_evaluations_number(1000);
        resp_opt.set_zoom_factor(0.7);

        MatrixR single_results = resp_opt.perform_response_optimization();

        if(single_results.rows() > 0)
        {
            cout << "Optimal plant state found for NH4 minimization:" << endl;

            auto var_names = time_series_dataset.get_variable_names();
            for(Index i=0; i<single_results.cols(); ++i)
                cout << var_names[i] << ": " << single_results(0, i) << endl;

        }

        cout << "\n[Scenario A2] Optimizing for minimum PO4 Output..." << endl;

        resp_opt.clear_conditions();

        resp_opt.set_condition("FLOW", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("TEMPERATURE", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("IRON", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("OXYGEN", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("POLYALUMINUM", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("INLET", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("OUTLET", ResponseOptimization::ConditionType::Past);

        resp_opt.set_condition("PO4_Output", ResponseOptimization::ConditionType::Minimize);

        single_results = resp_opt.perform_response_optimization();

        if(single_results.rows() > 0)
        {
            cout << "Optimal plant state found for NH4 minimization:" << endl;

            auto var_names = time_series_dataset.get_variable_names();
            for(Index i=0; i<single_results.cols(); ++i)
                cout << var_names[i] << ": " << single_results(0, i) << endl;

        }


        // --- SCENARIO B: Multi-Objective (Minimizing multiple pollutants) ---
        cout << "\n[Scenario B] Multi-Objective: Minimize NH4 AND PO4 AND Chemical Cost..." << endl;

        resp_opt.clear_conditions(); // Reset previous goals

        resp_opt.set_condition("FLOW", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("TEMPERATURE", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("IRON", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("OXYGEN", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("POLYALUMINUM", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("INLET", ResponseOptimization::ConditionType::Past);
        resp_opt.set_condition("OUTLET", ResponseOptimization::ConditionType::Past);

        // Objectives
        resp_opt.set_condition("NH4_Output", ResponseOptimization::ConditionType::Minimize);
        resp_opt.set_condition("PO4_Output", ResponseOptimization::ConditionType::Minimize);
        resp_opt.set_condition("IRON_Input", ResponseOptimization::ConditionType::Minimize); // Minimize cost


        resp_opt.set_iterations(8);
        resp_opt.set_evaluations_number(2000);
        resp_opt.set_zoom_factor(0.7);

        MatrixR pareto_front = resp_opt.perform_response_optimization();

        cout << "Found " << pareto_front.rows() << " optimal trade-off solutions (Pareto Front)." << endl;
        cout << "Saving Pareto results to 'wwtp_optimization_results.csv'..." << endl;

        // Save results to file
        ofstream file("wwtp_optimization_results.csv");
        auto names = time_series_dataset.get_variable_names();
        for(size_t i=0; i<names.size(); ++i) file << names[i] << (i==names.size()-1 ? "" : ",");
        file << "\n";

        for(Index i=0; i<pareto_front.rows(); ++i)
        {
            for(Index j=0; j<pareto_front.cols(); ++j)
                file << pareto_front(i, j) << (j==pareto_front.cols()-1 ? "" : ",");

            file << "\n";
        }
        file.close();

        cout << "Optimization complete. Good bye!" << endl;

        cout << "Good bye!" << endl;

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
