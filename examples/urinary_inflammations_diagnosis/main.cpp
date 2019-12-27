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

int main(void)
{
    try
    {
        cout << "OpenNN. Urinary Inflammations Diagnosis Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("../data/urinary_inflammations_diagnosis.csv", ';', true);

        data_set.print_data_preview();

        // Variables      

        data_set.set_columns_uses({"Input","Input","Input","Input","Input","Input","UnusedVariable","Target"});

        const Vector<string> inputs_names = data_set.get_input_variables_names();
        const Vector<string> targets_names = data_set.get_target_variables_names();


        // Instances
        
        data_set.split_instances_random();

        const Vector<Descriptives> inputs_descriptives = data_set.scale_inputs_minimum_maximum();

        // Neural network

        NeuralNetwork neural_network(NeuralNetwork::Classification, {6, 6, 1});

        neural_network.set_inputs_names(inputs_names);

        neural_network.set_outputs_names(targets_names);

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

        scaling_layer_pointer->set_descriptives(inputs_descriptives);

        scaling_layer_pointer->set_scaling_methods(ScalingLayer::NoScaling);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();

        quasi_Newton_method_pointer->set_minimum_loss_decrease(1.0e-6);

        quasi_Newton_method_pointer->set_loss_goal(1.0e-3);

        training_strategy.set_display(true);

        training_strategy.perform_training();

        // Model selection

//        ModelSelection model_selection(&training_strategy);

//        model_selection.set_inputs_selection_method(ModelSelection::GENETIC_ALGORITHM);

//        GeneticAlgorithm* genetic_algorithm_pointer = model_selection.get_genetic_algorithm_pointer();

//        genetic_algorithm_pointer->set_approximation(false);

//        genetic_algorithm_pointer->set_inicialization_method(GeneticAlgorithm::Random);

//        genetic_algorithm_pointer->set_display(true);

//        const ModelSelection::Results model_selection_results = model_selection.perform_inputs_selection();

        // Testing analysis

        const TestingAnalysis testing_analysis(&neural_network, &data_set);

        const Matrix<size_t> confusion = testing_analysis.calculate_confusion();

        cout << "Confusion: " << endl;
        cout << confusion << endl;

        // Save results

        data_set.save("../data/data_set.xml");

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression("../data/expression.txt");

        training_strategy.save("../data/training_strategy.xml");

//        model_selection.save("../data/model_selection.xml");
//        model_selection_results.save("../data/model_selection_results.dat");

        confusion.save_csv("../data/confusion.csv");

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
