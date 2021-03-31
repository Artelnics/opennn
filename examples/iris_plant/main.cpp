//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I R I S   P L A N T   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is a classical pattern recognition problem.

// System includes

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace Eigen;

int main()
{
    try
    {
        cout << "OpenNN. Iris Plant Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("../data/iris_plant_original.csv", ';', true);

        const Tensor<string, 1> inputs_names = data_set.get_input_variables_names();
        const Tensor<string, 1> targets_names = data_set.get_target_variables_names();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        const Tensor<Descriptives, 1> input_variables_descriptives = data_set.scale_input_variables_minimum_maximum();

        // Neural network

        const Index hidden_neurons_number = 3;

        NeuralNetwork neural_network(NeuralNetwork::Classification, {input_variables_number, hidden_neurons_number, target_variables_number});

        neural_network.set_inputs_names(inputs_names);
        neural_network.set_outputs_names(targets_names);

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
        scaling_layer_pointer->set_descriptives(input_variables_descriptives);
        scaling_layer_pointer->set_scaling_methods(ScalingLayer::MinimumMaximum);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::SUM_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::QUASI_NEWTON_METHOD);
/*
        AdaptiveMomentEstimation* adam = training_strategy.get_adaptive_moment_estimation_pointer();

        adam->set_loss_goal(1.0e-3);
        adam->set_maximum_epochs_number(10000);
        adam->set_display_period(1000);
*/
        training_strategy.perform_training();

        // Testing analysis

        Tensor<type, 2> inputs(3, 4);

        inputs.setValues({{5.1,3.5,1.4,0.2},
                          {6.4,3.2,4.5,1.5},
                          {6.3,2.7,4.9,1.8}});

        cout << "Inputs: " << endl;
        cout << inputs << endl;

        cout << "Outputs: " << endl;
        cout << neural_network.calculate_outputs(inputs) << endl;

        data_set.unscale_input_variables_minimum_maximum(input_variables_descriptives);

        const TestingAnalysis testing_analysis(&neural_network, &data_set);

        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

        cout << "Confusion: " << endl;
        cout << confusion << endl;

        // Save results

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression_c("../data/neural_network.c");
        neural_network.save_expression_python("../data/neural_network.py");

        return 0;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }
}  


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques SL
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
