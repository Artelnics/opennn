//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "../opennn/pch.h"
#include "../opennn/dataset.h"
#include "../opennn/adaptive_moment_estimation.h"
#include "../opennn/neural_network.h"
#include "../opennn/standard_networks.h"
#include "../opennn/training_strategy.h"
#include "../opennn/testing_analysis.h"
#include "../opennn/normalized_squared_error.h"
#include "../opennn/optimization_algorithm.h"
#include "../opennn/genetic_algorithm.h"
#include "../opennn/growing_inputs.h"
#include "../opennn/weighted_squared_error.h"
#include "../opennn/cross_entropy_error.h"
#include "../opennn/image_dataset.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "Blank Testing OpenNN" << endl;

        // Data set

        Dataset dataset("/mnt/c/Users/davidgonzalez/Documents/5_years_mortality.csv", ";", true, true);

        const Index inputs_number = dataset.get_variables_number("Input");
        const Index targets_number = dataset.get_variables_number("Target");

        dataset.scrub_missing_values();

        // Neural network

        const Index neurons_number = 6;

        ClassificationNetwork classification_network({ inputs_number }, { neurons_number }, { targets_number });

        // Training Strategy

        TrainingStrategy training_strategy(&classification_network, &dataset);

        training_strategy.set_loss_index("CrossEntropyError2d");
        training_strategy.set_optimization_algorithm("QuasiNewtonMethod");

        // Growing Inputs

        GrowingInputs growing_inputs(&training_strategy);
        growing_inputs.set_display(true); // Mostrar información detallada durante la ejecución.
        growing_inputs.set_maximum_epochs_number(500); // máximo de generaciones.
        growing_inputs.set_maximum_selection_failures(100); // máximo de generaciones.
        growing_inputs.set_maximum_time(360); // Límite de tiempo en segundos
        growing_inputs.set_minimum_inputs_number(10);
        growing_inputs.set_maximum_inputs_number(12);
        growing_inputs.set_trials_number(3);

        // Genetic Algorithm

        GeneticAlgorithm genetic_algorithm(&training_strategy);
        genetic_algorithm.set_display(true); // Mostrar información detallada durante la ejecución.
        genetic_algorithm.set_maximum_epochs_number(100); // máximo de generaciones.
        genetic_algorithm.set_maximum_time(3600); // Límite de tiempo en segundos
        genetic_algorithm.set_minimum_inputs_number(1);
        genetic_algorithm.set_maximum_inputs_number(50);
        genetic_algorithm.set_individuals_number(40); // 40 soluciones candidatas por generación.
        genetic_algorithm.set_elitism_size(10); // Los 4 mejores individuos pasan sin cambios a la siguiente generación.
        genetic_algorithm.set_mutation_rate(0.00);

        genetic_algorithm.set_initialization_method("Correlations");
        //genetic_algorithm.set_initialization_method(Random");

        cout << "\nStarting input selection..." << endl;
        InputsSelectionResults ga_results = genetic_algorithm.perform_input_selection();
        //InputsSelectionResults gi_results = growing_inputs.perform_input_selection();

        cout << "\nInput selection completed." << endl;

        ga_results.print();
        //gi_results.print();

        //cout << "\n--- Process Summary ---" << endl;
        //cout << "Elapsed time: " << gi_results.elapsed_time;
        //cout << "Stopping condition: " << gi_results.write_stopping_condition() << endl;
        //cout << "Total generations performed: " << gi_results.get_epochs_number() << endl;

        cout << "\n--- Process Summary ---" << endl;
        cout << "Elapsed time: " << ga_results.elapsed_time;
        cout << "Stopping condition: " << ga_results.write_stopping_condition() << endl;
        cout << "Total generations performed: " << ga_results.get_epochs_number() << endl;

        cout << "\n--- Final Model State ---" << endl;
        cout << "The neural network is now configured with the optimal inputs." << endl;
        cout << "Final number of inputs in the neural network: " << classification_network.get_input_dimensions()[0] << endl;

        cout << "Completed." << endl;

        return 0;
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
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
