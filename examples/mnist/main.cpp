//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M N I S T    A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is an approximation application.

// System includes

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdexcept>

// OpenNN includes

#include "../../opennn/opennn.h"
#include "../../opennn/opennn_strings.h"

using namespace std;
using namespace OpenNN;


int main()
{
    try
    {
        cout << "OpenNN. National Institute of Standards and Techonology (MNIST) Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("../data/mnist_train.csv", ',', true);

        data_set.set_input();

        data_set.set_column_use(0, DataSet::VariableUse::Target);

        const Tensor<string, 1> unused_variables = data_set.unuse_constant_columns();

        const Index input_variables_number = data_set.get_input_variables_number();

        const Index target_variables_number = data_set.get_target_variables_number();

        const Tensor<Descriptives, 1> input_variables_descriptives = data_set.scale_input_variables_minimum_maximum();
/*
        const Tensor<DataSet::Column, 1> columns = data_set.get_columns();

        for(Index i = 0; i < columns.size(); i++)
        {
            cout << "Column " << i << ": " << endl;
            cout << "   Name: " << columns(i).name << endl;

            if(columns(i).column_use == OpenNN::DataSet::Input) cout << "   Use: input" << endl;
            else if(columns(i).column_use == OpenNN::DataSet::Target) cout << "   Use: target" << endl;
            else if(columns(i).column_use == OpenNN::DataSet::UnusedVariable) cout << "   Use: unused" << endl;

            if(columns(i).type == OpenNN::DataSet::ColumnType::Categorical) cout << "   Categories: " << columns(i).categories << endl;

            cout << endl;
        }
*/
        cout << "Input variables number: " << data_set.get_target_variables_number() << endl;
        cout << "Target variables number: " << data_set.get_target_variables_number() << endl;

        // Neural network

        cout << "input_variables_number" << input_variables_number <<endl;
        cout << "target_variables_number" << target_variables_number <<endl;

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, {input_variables_number, 50, target_variables_number});
/*
        PerceptronLayer* perceptron_layer_pointer = neural_network.get_first_perceptron_layer_pointer();
        perceptron_layer_pointer->set_activation_function("RectifiedLinear");

        Tensor<Layer*, 1> layers_pointers = neural_network.get_trainable_layers_pointers();

        for(Index i = 0; i < layers_pointers.size(); i++)
        {
            cout << "Layer " << i << ": " << endl;
            cout << "   Type: " << layers_pointers(i)->get_type_string() << endl;

            if(layers_pointers(i)->get_type_string() == "Perceptron") cout << "   Activation: " << static_cast<PerceptronLayer*>(layers_pointers(i))->write_activation_function() << endl;
            if(layers_pointers(i)->get_type_string() == "Probabilistic") cout << "   Activation: " << static_cast<ProbabilisticLayer*>(layers_pointers(i))->write_activation_function() << endl;
        }
*/
        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::NORMALIZED_SQUARED_ERROR);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        training_strategy.set_display_period(10);

        training_strategy.set_maximum_epochs_number(1000);

        training_strategy.perform_training();

        // Testing analysis

        const TestingAnalysis testing_analysis(&neural_network, &data_set);

        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
        const Tensor<type, 1> multiple_classification_tests = testing_analysis.calculate_multiple_classification_tests();

        cout << "Confusion matrix: " << endl;
        cout << confusion << endl;

        cout << "Accuracy: " << multiple_classification_tests(0)*100 << "%" << endl;
        cout << "Error: " << multiple_classification_tests(1)*100 << "%" << endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}



// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2020 Artificial Intelligence Techniques SL
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
