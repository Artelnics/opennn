//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A I R F O I L   S E L F - N O I S E   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// This is an approximation application.

// System includes

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main()
{
    try
    {
        cout << "OpenNN. Airfoil Self-Noise Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("../data/airfoil_self_noise.csv", ';', true);
/*
        data_set.print_data();

        const Tensor<Descriptives, 1> variables_descriptives = data_set.scale_data();
        data_set.unscale_data(variables_descriptives);

        data_set.print_data();
*/

        const Tensor<string, 1> inputs_names = data_set.get_input_variables_names();
        const Tensor<string, 1> targets_names = data_set.get_target_variables_names();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        const Tensor<Scaler, 1> input_variables_scalers =  data_set.get_input_variables_scalers();
        const Tensor<Scaler, 1> target_variables_scalers = data_set.get_target_variables_scalers();

        const Tensor<Descriptives, 1> input_variables_descriptives =  data_set.calculate_input_variables_descriptives();
        const Tensor<Descriptives, 1> target_variables_descriptives = data_set.calculate_target_variables_descriptives();

        const Tensor<Descriptives, 1> variables_descriptives = data_set.scale_data();

        // Neural network

        const Index hidden_neurons_number = 12;

        NeuralNetwork neural_network(NeuralNetwork::Approximation, {input_variables_number, hidden_neurons_number, target_variables_number});

        neural_network.set_inputs_names(inputs_names);
        neural_network.set_outputs_names(targets_names);

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
        scaling_layer_pointer->set(input_variables_descriptives, input_variables_scalers);

        UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();
        unscaling_layer_pointer->set(target_variables_descriptives, target_variables_scalers);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::NORMALIZED_SQUARED_ERROR);

        training_strategy.set_optimization_method(TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM);

        const TrainingResults training_results = training_strategy.perform_training();
/*
        // Model selection

        ModelSelection model_selection(&training_strategy);

        GrowingNeurons* growing_neurons_pointer = model_selection.get_growing_neurons_pointer();
        growing_neurons_pointer->set_neurons_increment(5);
        growing_neurons_pointer->set_maximum_neurons_number(50);

//        model_selection.perform_neurons_selection();

        model_selection.set_inputs_selection_method(ModelSelection::PRUNING_INPUTS);

        GrowingInputs* growing_inputs_pointer = model_selection.get_growing_inputs_pointer();
        growing_inputs_pointer->set_maximum_iterations_number(20);
        growing_inputs_pointer->set_maximum_inputs_number(20);

        GeneticAlgorithm* genetic_algorithm_pointer = model_selection.get_genetic_algorithm_pointer();
        genetic_algorithm_pointer->set_elitism_size(0);
        genetic_algorithm_pointer->set_selective_pressure(1);
        genetic_algorithm_pointer->set_maximum_iterations_number(3);

        model_selection.perform_inputs_selection();

        // Testing analysis

        data_set.unscale_data(variables_descriptives);

        data_set.print_data();

//        data_set.unscale_input_variables(input_variables_descriptives);
//        data_set.unscale_target_variables(target_variables_descriptives);

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        const TestingAnalysis::LinearRegressionAnalysis linear_regression_analysis = testing_analysis.perform_linear_regression_analysis()[0];

        linear_regression_analysis.print();

        // Save results

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression_python("../data/neural_network.py");
*/
        cout << "End Airfoil Self-Noise Example" << endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

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
