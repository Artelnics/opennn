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
#include "../../opennn/testing_analysis.h"
#include "../../opennn/optimizer.h"
#include "../../opennn/variable.h"
#include "../../opennn/response_optimization.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN Response Optimization Example: Concrete " << endl;

        Dataset dataset("../data/concrete.csv", ",", true, false);

        dataset.set_variable_types(VariableType::Numeric);

        dataset.impute_missing_values_mean();

        //dataset.set_variable_role("SLUMP(cm)", "None");
        //dataset.set_variable_role("FLOW(cm)", "None");
        dataset.set_variable_role("Compressive Strength (28-day)(Mpa)", "Target");

        dataset.split_samples_random(type(0.6), type(0.2), type(0.2));

        // Neural Network

        const Index neurons_number = 3;
        const type regularization_weight = 0.0001;

        ApproximationNetwork approximation_network(dataset.get_input_shape(), {neurons_number}, dataset.get_target_shape());

        Bounding* bounding_layer = (Bounding*)approximation_network.get_first("Bounding");

        if(bounding_layer)
            bounding_layer->set_bounding_method("NoBounding");

        TrainingStrategy training_strategy(&approximation_network, &dataset);

        //training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        //training_strategy.get_loss_index()->set_regularization_method("L1");
        training_strategy.get_loss_index()->set_regularization_weight(regularization_weight);

        TrainingResults training_results = training_strategy.train();

        TestingAnalysis testing_analysis(&approximation_network, &dataset);
        testing_analysis.print_goodness_of_fit_analysis();

        // 4. RESPONSE OPTIMIZATION
        ResponseOptimization optimizer(&approximation_network, &dataset);

        optimizer.set_zoom_factor(0.8);

        //optimizer.set_threads_number(1);

        cout << "\n[Single Objective Experiment] Maximizing Compressive Strength..." << endl;

        vector<ResponseOptimization::Condition> single_conds(dataset.get_used_variables_number());

        single_conds[dataset.get_variable_index("Compressive Strength (28-day)(Mpa)")] = {ResponseOptimization::ConditionType::Maximize};

        optimizer.set_condition("Compressive Strength (28-day)(Mpa)", ResponseOptimization::ConditionType::Maximize);

        MatrixR single_results = optimizer.perform_response_optimization();

            cout << "Optimal Recipe for Max Strength:" << endl;

            auto names_variables = dataset.get_variable_names();
            for(auto& n : names_variables)
                cout << setw(15) << n;

            cout << endl;

            for(Index i=0; i<single_results.cols(); ++i)
                cout << setw(15) << single_results(0, i);
            cout << endl;


        // --- EXPERIMENT B: Multi-Objective (The Trade-off) ---
        // Goals: Maximize Slump, Flow, and Strength while MINIMIZING Cement.
        cout << "\n[Multi-Objective Experiment] Max Strength & Min Cement..." << endl;
            // Define a simple structure to hold experiment parameters
            struct ExperimentConfig
            {
                type zoom;
                int evals;
                int iterations;
            };

            // ... inside your try block in main() ...

            // 1. Setup the basic optimization goals (these stay the same)
            optimizer.set_condition("Compressive Strength (28-day)(Mpa)", ResponseOptimization::ConditionType::Maximize);
            optimizer.set_condition("Cement", ResponseOptimization::ConditionType::Minimize);

            // 2. Define the list of experiments you want to run
            // You can compare aggressive vs gentle zoom, or high vs low sampling
            vector<ExperimentConfig> experiments =
            {
                {0.9,  5000,  10}, // Gentle zoom, standard sampling
                {0.8,  10000, 10}, // Medium zoom, high sampling
                {0.7,  20000, 15}, // Aggressive zoom, very high sampling
                {0.95, 5000,  20}  // Very slow convergence
            };

            auto variable_names = dataset.get_variable_names();

            for (const auto& config : experiments)
            {
                cout << "\n==========================================================" << endl;
                cout << "RUNNING EXPERIMENT: Zoom=" << config.zoom
                     << ", Evals=" << config.evals
                     << ", Iterations=" << config.iterations << endl;
                cout << "==========================================================" << endl;

                // Apply configuration
                optimizer.set_zoom_factor(config.zoom);
                optimizer.set_evaluations_number(config.evals);
                optimizer.set_iterations(config.iterations);

                // Perform optimization
                MatrixR pareto_results = optimizer.perform_response_optimization();

                cout << "Pareto Front Size: " << pareto_results.rows() << " points." << endl;

                // Generate a clear, descriptive filename
                // Format: pareto_Z0.70_E20000_I10.csv
                stringstream ss;
                ss << "pareto_Z" << fixed << setprecision(2) << config.zoom
                   << "_E" << config.evals
                   << "_I" << config.iterations << ".csv";
                string filename = ss.str();

                // Save results to the specific file
                ofstream file(filename);

                // Write Header
                for(int i = 0; i < variable_names.size(); ++i)
                    file << variable_names[i] << (i == variable_names.size()-1 ? "" : ",");
                file << "\n";

                // Write Data
                for(Index i=0; i < pareto_results.rows(); ++i)
                {
                    for(Index j=0; j < pareto_results.cols(); ++j)
                        file << pareto_results(i, j) << (j == pareto_results.cols()-1 ? "" : ",");
                    file << "\n";
                }
                file.close();

                cout << "Results saved to: " << filename << endl;
            }

            cout << "\nAll experiments complete." << endl;

    }

    catch (exception& e)
    {
        cerr << "Error: " << e.what() << endl;
    }
    return 0;
}
// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques SL
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
