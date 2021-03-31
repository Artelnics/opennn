//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U R I N A R Y   I N F L A M M A T I O N S   D I A G N O S I S   A P P L I C A T I O N
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

int main()
{
    try
    {
        cout << "OpenNN. Urinary Inflammations Diagnosis Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("../data/urinary_inflammations_diagnosis.csv", ';', true);

        // Variables      

        Tensor<string, 1> uses(8);
        uses.setValues({"Input","Input","Input","Input","Input","Input","Unused","Target"});
        data_set.set_columns_uses(uses);

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        const Tensor<string, 1> inputs_names = data_set.get_input_variables_names();
        const Tensor<string, 1> targets_names = data_set.get_target_variables_names();

        const Tensor<Descriptives, 1> input_variables_descriptives = data_set.scale_input_variables_minimum_maximum();

        // Neural network

        const Index hidden_neurons_number = 6;

        NeuralNetwork neural_network(NeuralNetwork::Classification, {input_variables_number, hidden_neurons_number, target_variables_number});

        neural_network.set_inputs_names(inputs_names);
        neural_network.set_outputs_names(targets_names);

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

        scaling_layer_pointer->set_descriptives(input_variables_descriptives);
        scaling_layer_pointer->set_scaling_methods(ScalingLayer::MinimumMaximum);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_optimization_method(TrainingStrategy::QUASI_NEWTON_METHOD);

        training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::L2);
        training_strategy.get_loss_index_pointer()->set_regularization_weight(0.001);

        training_strategy.get_normalized_squared_error_pointer()->set_normalization_coefficient();

        ConjugateGradient* conjugate_gradient_pointer = training_strategy.get_conjugate_gradient_pointer();
        conjugate_gradient_pointer->set_minimum_loss_decrease(1.0e-6);
        conjugate_gradient_pointer->set_loss_goal(1.0e-3);

        const TrainingResults training_results = training_strategy.perform_training();

        // Testing analysis

        data_set.unscale_input_variables_minimum_maximum(input_variables_descriptives);

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

        const Tensor<type, 1> binary_classification_tests = testing_analysis.calculate_binary_classification_tests();

        cout << "Confusion:"<< endl
             << confusion << endl;

        cout << "Binary classification tests" << endl;
        cout << "Classification accuracy : " << binary_classification_tests[0] << endl;
        cout << "Error rate              : " << binary_classification_tests[1] << endl;
        cout << "Sensitivity             : " << binary_classification_tests[2] << endl;
        cout << "Specificity             : " << binary_classification_tests[3] << endl;

        // Save results

        neural_network.save("../data/neural_network.xml");

        neural_network.save_expression_python("neural_network.py");

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
