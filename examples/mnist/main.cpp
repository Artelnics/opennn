//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S I M P L E   F U N C T I O N   R E G R E S S I O N   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is an approximation application.

// System includes

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdexcept>

#include <omp.h>

// OpenNN includes

#include "../../opennn/opennn.h"
#include "../../opennn/opennn_strings.h"

using namespace std;
using namespace OpenNN;


string transform_number_to_category(const string& line)
{
    Tensor<string,1> line_tokens = get_tokens(line, ',');
    const Index tokens_number = line_tokens.size();

    if(line_tokens(0) == "0") line_tokens(0) = "zero";
    else if(line_tokens(0) == "1") line_tokens(0) = "one";
    else if(line_tokens(0) == "2") line_tokens(0) = "two";
    else if(line_tokens(0) == "3") line_tokens(0) = "three";
    else if(line_tokens(0) == "4") line_tokens(0) = "four";
    else if(line_tokens(0) == "5") line_tokens(0) = "five";
    else if(line_tokens(0) == "6") line_tokens(0) = "six";
    else if(line_tokens(0) == "7") line_tokens(0) = "seven";
    else if(line_tokens(0) == "8") line_tokens(0) = "eight";
    else if(line_tokens(0) == "9") line_tokens(0) = "nine";

    string new_line = "";

    for(Index i = 0; i < tokens_number; i++)
    {
        new_line += line_tokens(i);

        if(i != tokens_number-1) new_line += ",";
    }

    return new_line;
}

int main(void)
{
    try
    {
        cout << "OpenNN. National Institute of Standards and Techonology (MNIST) Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Preprocess data

        bool reload = true;

        ifstream training_file("../data/mnist_train.csv");

        if(!training_file.is_open() || reload)
        {
            cout << "Creating training data set..." << endl;

            ofstream complete_file("../data/mnist_train.csv");

            ifstream file_1("../data/mnist_train_1.csv");

            if(!file_1.is_open())
            {
                ostringstream buffer;

                buffer << "OpenNN Exception: MNIST example.\n"
                       << "Cannot open data file: mnist_train_1.csv \n";

                throw logic_error(buffer.str());
            }

            Index lines_count = 0;
            string line;

            while(file_1.good())
            {
                getline(file_1, line);

                trim(line);

                erase(line, '"');

                if(line.empty()) continue;

                complete_file << transform_number_to_category(line);
                complete_file << endl;

                lines_count++;
            }

            file_1.close();

            ifstream file_2("../data/mnist_train_2.csv");

            if(!file_2.is_open())
            {
                ostringstream buffer;

                buffer << "OpenNN Exception: MNIST example.\n"
                       << "Cannot open data file: mnist_train_2.csv \n";

                throw logic_error(buffer.str());
            }

            lines_count = 0;

            while(file_2.good())
            {
                getline(file_2, line);

                trim(line);

                erase(line, '"');

                if(line.empty()) continue;

                complete_file << transform_number_to_category(line);
                complete_file << endl;

                lines_count++;
            }

            file_2.close();

            ifstream file_test("../data/mnist_test.csv");

            if(!file_test.is_open())
            {
                ostringstream buffer;

                buffer << "OpenNN Exception: MNIST example.\n"
                       << "Cannot open data file: mnist_test.csv \n";

                throw logic_error(buffer.str());
            }

            lines_count = 0;

            while(file_test.good())
            {
                getline(file_test, line);

                trim(line);

                erase(line, '"');

                if(line.empty()) continue;

                complete_file << transform_number_to_category(line);
                complete_file << endl;

                lines_count++;
            }

            file_test.close();

            complete_file.close();
        }
        else
        {
            training_file.close();
        }

        // Data set

        bool display_data_set = false;

        DataSet data_set("../data/mnist_train.csv",',',true);

        data_set.set_input();
        data_set.set_column_use(0, DataSet::VariableUse::Target);

        const Tensor<string, 1> unused_variables = data_set.unuse_constant_columns();

        const Index input_variables_number = data_set.get_input_variables_number();

        const Index target_variables_number = data_set.get_target_variables_number();

        Tensor<string, 1> scaling_methods(input_variables_number);
        scaling_methods.setConstant("MinimumMaximum");

        const Tensor<Descriptives, 1> inputs_descriptives = data_set.scale_input_variables(scaling_methods);


        data_set.unuse_constant_columns();

        if(display_data_set)
        {
            Tensor<DataSet::Column, 1> columns = data_set.get_columns();

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

            cout << "Input variables number: " << data_set.get_target_variables_number() << endl;
            cout << "Target variables number: " << data_set.get_target_variables_number() << endl;
        }




        // Neural network

        bool display_neural_network = false;

        cout << "input_variables_number" << input_variables_number <<endl;
        cout << "target_variables_number" << target_variables_number <<endl;

        Tensor<Index, 1> architecture(3);
        architecture[0] = input_variables_number;
        architecture[1] = 50;
        architecture[2] = target_variables_number;

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, architecture);
        PerceptronLayer* perceptron_layer_pointer = neural_network.get_first_perceptron_layer_pointer();
        perceptron_layer_pointer->set_activation_function("RectifiedLinear");

        if(display_neural_network)
        {
            Tensor<Layer*, 1> layers_pointers = neural_network.get_trainable_layers_pointers();

            for(Index i = 0; i < layers_pointers.size(); i++)
            {
                cout << "Layer " << i << ": " << endl;
                cout << "   Type: " << layers_pointers(i)->get_type_string() << endl;

                if(layers_pointers(i)->get_type_string() == "Perceptron") cout << "   Activation: " << static_cast<PerceptronLayer*>(layers_pointers(i))->write_activation_function() << endl;
                if(layers_pointers(i)->get_type_string() == "Probabilistic") cout << "   Activation: " << static_cast<ProbabilisticLayer*>(layers_pointers(i))->write_activation_function() << endl;
            }
        }



        ///TEST
//        Index samples_number = data_set.get_samples_number();
//        DataSet::Batch batch(samples_number, &data_set);

//        Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();
//        const Tensor<Index, 1> input_indices = data_set.get_input_variables_indices();
//        const Tensor<Index, 1> target_indices = data_set.get_target_variables_indices();

//        batch.fill(samples_indices, input_indices, target_indices);

//        NormalizedSquaredError nse(&neural_network, &data_set);

//        NeuralNetwork::ForwardPropagation forward_propagation(samples_number, &neural_network);
//        LossIndex::BackPropagation training_back_propagation(samples_number, &nse);

//        neural_network.forward_propagate(batch, forward_propagation);

////        forward_propagation.print();

//        nse.back_propagate(batch, forward_propagation, training_back_propagation);



////        training_back_propagation.print();


        // Training strategy



        TrainingStrategy training_strategy(&neural_network, &data_set);


        training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);

        NormalizedSquaredError* normalized_squared_error_pointer = training_strategy.get_normalized_squared_error_pointer();
        normalized_squared_error_pointer->set_normalization_coefficient();

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);

        training_strategy.set_display_period(5);

        training_strategy.set_maximum_epochs_number(20);

        neural_network.set_parameters_random();
        neural_network.get_first_perceptron_layer_pointer()->set_synaptic_weights_constant_glorot_uniform();
        neural_network.get_first_perceptron_layer_pointer()->set_biases_constant(0);
        neural_network.get_probabilistic_layer_pointer()->set_synaptic_weights_constant_glorot_uniform();

        training_strategy.perform_training();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
        Tensor<type, 1> multiple_classification_tests = testing_analysis.calculate_multiple_classification_tests();

        cout << "Confusion matrix: " << endl;
        cout << confusion << endl;

        cout << "Accuracy: " << multiple_classification_tests(0)*100 << endl;
        cout << "Error: " << multiple_classification_tests(1)*100 << endl;

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
