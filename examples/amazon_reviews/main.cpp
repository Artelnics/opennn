//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A M A Z O N   R E V I E W S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace opennn;
using namespace std;
using namespace Eigen;

int main()
{
    try
    {
        cout << "OpenNN. Amazon reviews example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set;

        data_set.set_data_file_name("../data/amazon_cells_labelled.txt");

        data_set.set_text_separator(DataSet::Separator::Semicolon);

        data_set.read_txt();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        // Neural network

        const Index hidden_neurons_number = 1;

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, {input_variables_number, hidden_neurons_number, target_variables_number});

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.perform_training();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        testing_analysis.print_binary_classification_tests();

        // Calculate outputs

        string review_1 = "Highly recommend for any one who has a blue tooth phone.";
        string review_2 = "You have to hold the phone at a particular angle for the other party to hear you clearly.";

        Tensor<type,1> processed_review_1 = data_set.sentence_to_data(review_1);
        Tensor<type,1> processed_review_2 = data_set.sentence_to_data(review_2);

        Tensor<type,2> input_data(2, input_variables_number);
        for(Index i = 0; i < input_variables_number; i++)
        {
            input_data(0,i) = processed_review_1(i);
            input_data(1,i) = processed_review_2(i);
        }

        Tensor<type,2> outputs = neural_network.calculate_outputs(input_data);

        cout << "\n\n" << review_1 << endl << "\nBad:" << outputs(0,0) << "%\tGood:" << (1 - outputs(0,0)) << "%" << endl;
        cout << "\n" << review_2 << endl << "\nBad:" << outputs(1,0) << "%\tGood:" << (1 - outputs(1,0)) << "%\n" << endl;

        // Save results

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression_python("../data/neural_network.py");

        cout << "Good bye!" << endl;

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
