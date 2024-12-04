//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A M A Z O N   R E V I E W S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

#include "../../opennn/opennn.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Amazon reviews example." << endl;

        srand(unsigned(time(nullptr)));

        // DataSet

        TextDataSet text_data_set;
        
        text_data_set.set_data_path("../data/amazon_cells_reduced.txt");
        text_data_set.set_separator(DataSet::Separator::Tab);

        text_data_set.read_txt();
        
        text_data_set.split_samples_random();

        const vector<string> input_words = text_data_set.get_raw_variable_names(DataSet::VariableUse::Input);
        const vector<string> targets_names = text_data_set.get_variable_names(DataSet::VariableUse::Input);

        const Index words_number = text_data_set.get_variables_number(DataSet::VariableUse::Input);
        const Index target_variables_number = text_data_set.get_variables_number(DataSet::VariableUse::Target);

        // Neural Network

        const Index hidden_neurons_number = 6;

        NeuralNetwork neural_network(NeuralNetwork::ModelType::TextClassification,
            { words_number }, { hidden_neurons_number }, { target_variables_number });

        neural_network.print();

        // Training Strategy

        TrainingStrategy training_strategy(&neural_network, &text_data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        training_strategy.perform_training();

        // Testing Analysis
        TestingAnalysis testing_analysis(&neural_network, &text_data_set);

        testing_analysis.print_binary_classification_tests();

        // Model deployment

        string review_1 = "Highly recommend for any one who has a bluetooth phone.";
        Tensor<type, 1> processed_review_1 = text_data_set.sentence_to_data(review_1);

        string review_2 = "You have to hold the phone at a particular angle for the other party to hear you clearly.";
        Tensor<type, 1> processed_review_2 = text_data_set.sentence_to_data(review_2);

        Tensor<type,2> input_data(2, words_number);
        Tensor<Index, 1> input_dimensions = get_dimensions(input_data);

        Tensor<type, 2> output_data;

        for(Index i = 0; i < words_number; i++)
        {
          input_data(0,i) = processed_review_1(i);
          input_data(1,i) = processed_review_2(i);
        }

        output_data = neural_network.calculate_outputs(input_data);

        cout << "\n\n" << review_1 << "\nBad:" << output_data(0,0) << "%\tGood:" << (1 - output_data(0,0)) << "%" << endl;
        cout << "\n" << review_2 << "\nBad:" << output_data(1,0) << "%\tGood:" << (1 - output_data(1,0)) << "%\n" << endl;

        // Save results

        neural_network.save_expression(NeuralNetwork::ProgrammingLanguage::Python, "../data/amazon_reviews.py");
        
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
