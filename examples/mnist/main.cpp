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

        bool reload = false;

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

        // Device

        const int n = 4;
        NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
        ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);


        // Data set

        bool display_data_set = false;

        DataSet data_set("../data/mnist_train.csv",',',true);
        data_set.set_thread_pool_device(thread_pool_device);

        data_set.set_input();
        data_set.set_column_use(0, DataSet::VariableUse::Target);

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        Tensor<string, 1> scaling_methods(input_variables_number);
        scaling_methods.setConstant("MinimumMaximum");

        const Tensor<Descriptives, 1> inputs_descriptives = data_set.calculate_input_variables_descriptives();
        data_set.scale_inputs(scaling_methods, inputs_descriptives);

        if(display_data_set)
        {
            Tensor<DataSet::Column, 1> columns = data_set.get_columns();

            for(Index i = 0; i < columns.size(); i++)
            {
                cout << "Column " << i << ": " << endl;
                cout << "   Name: " << columns(i).name << endl;

                if(columns(i).column_use == OpenNN::DataSet::Input) cout << "   Use: input" << endl;
                else if(columns(i).column_use == OpenNN::DataSet::Target) cout << "   Use: target" << endl;


                if(columns(i).type == OpenNN::DataSet::ColumnType::Categorical) cout << "   Categories: " << columns(i).categories << endl;

                cout << endl;
            }

            cout << "Input variables number: " << data_set.get_target_variables_number() << endl;
            cout << "Target variables number: " << data_set.get_target_variables_number() << endl;
        }


        // Neural network

        bool display_neural_network = true;

        Tensor<Index, 1> architecture(3);
        architecture[0] = input_variables_number;
        architecture[1] = 100;
        architecture[2] = target_variables_number;

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, architecture);
        neural_network.set_thread_pool_device(thread_pool_device);

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

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);
        training_strategy.set_thread_pool_device(thread_pool_device);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);

        NormalizedSquaredError* normalized_squared_error_pointer = training_strategy.get_normalized_squared_error_pointer();
        normalized_squared_error_pointer->set_normalization_coefficient();

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);

        training_strategy.perform_training();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);
        testing_analysis.set_thread_pool_device(thread_pool_device);

        cout << "Confusion matrix: " << endl;
        cout << testing_analysis.calculate_confusion() << endl;


        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

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
