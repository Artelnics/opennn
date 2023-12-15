//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <chrono>

// OpenNN includes

#include "../opennn/opennn.h"
#include <iostream>

using namespace std;
using namespace opennn;


int main()
{
   try
   {
        cout << "Blank\n";

        srand(static_cast<unsigned>(time(nullptr)));


// Train neural network

        DataSet data_set("/home/alvaromartin/Downloads/LowAlloySteels.csv", ',', true);
/*
        data_set.scrub_missing_values();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        Tensor<DataSet::Column, 1> columns = data_set.get_columns();

        Tensor<Index, 1> input_column_indices(columns.size() - 48);
        Tensor<Index, 1> target_column_indices(48);

        for(Index i = 0; i < columns.size(); i++)
        {
            if (i < 47){ target_column_indices(i) = i;}
            else { input_column_indices(i - 48) = i;}
        }

        data_set.set_input_target_columns(input_column_indices, target_column_indices);

        columns = data_set.get_columns();

        for(Index i = 0; i < columns.size(); i++)
        {
            columns(i).print();
        }

        cout << "==================================================================" << endl;
        cout << "==================================================================" << endl;
        std::cout << "input_variables_number: " << input_variables_number << std::endl;
        std::cout << "target_variables_number: " << target_variables_number << std::endl;
        cout << "Data set preview " << endl;
        data_set.print_data_preview();
        cout << "Data set dimensions: " << data_set.get_data().dimensions() << endl;
        cout << "==================================================================" << endl;

        // if (data_set.get_samples_number() <= 1)
        // {
        //     std::string error_message = "Not enough samples of model " + std::string(scope_ids[i]) +  " to train. ";

        //     std::cerr << error_message << std::endl;

        //     PQfinish(conn);
        //     return grpc::Status(grpc::StatusCode::INTERNAL, error_message);
        // }        

        // Neural network
        
        // Index hidden_layers_number = 3;
        
        // const Index total_layers_number = hidden_layers_number + 2;

        // Tensor<Index, 1> architecture(total_layers_number); // hidden_layers_number + input_layer + output_layer
        // architecture(0) = input_variables_number;
        // architecture(total_layers_number - 1) = target_variables_number;

        // Index user_layer_count = 0;

        // for (Index i = 1; i < total_layers_number - 1; i++)
        // {
        //     architecture(i) = neurons_per_layer[user_layer_count];
        //     user_layer_count++;
        //     if(user_layer_count == hidden_layers_number) break;
        // }

        // NeuralNetwork neural_network(NeuralNetwork::ProjectType::Approximation, architecture);

        // cout << "==================================================================" << endl;
        // neural_network.print();
        // cout << "==================================================================" << endl;

        // Training Strategy

        // TrainingStrategy training_strategy(&neural_network, &data_set);
        // training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
        // training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);

        // training_strategy.get_adaptive_moment_estimation_pointer()->set_batch_samples_number(20);

        // cout << "==================================================================" << endl;
        // training_strategy.perform_training();
        // cout << "==================================================================" << endl;

        // string model_file_name = "ñiñi_model.xml";

        // neural_network.save(model_file_name);
*/

        cout << "Bye!" << endl;

        return 0;
   }
   catch (const exception& e)
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
