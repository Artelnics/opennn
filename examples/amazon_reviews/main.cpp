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

#include "../../opennn/language_data_set.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Amazon reviews example." << endl;

        LanguageDataset language_dataset("../data/amazon_cells_labelled.txt");

        const Index vocabulary_input_size = language_dataset.get_input_vocabulary_size();
        const Index sequence_length = language_dataset.get_input_length();
        const Index embedding_dimension = 32;
        const Index neurons_number = 64;

        const Index targets_number = language_dataset.get_target_length();
        // const Index vocabulary_target_size = language_dataset.get_target_vocabulary_size();

        dimensions input_dimensions      = {vocabulary_input_size, sequence_length, embedding_dimension};
        dimensions complexity_dimensions = {neurons_number};
        dimensions output_dimensions     = {targets_number};

        NeuralNetwork neural_network(
            NeuralNetwork::ModelType::TextClassification,
            input_dimensions,
            complexity_dimensions,
            output_dimensions
            );

        neural_network.print();




        // dimensions Addition3d::get_input_dimensions() const
        // {
        //     return { sequence_length, embedding_dimension };
        // }


        // const vector<string> input_words = language_dataset.get_raw_variable_names(Dataset::VariableUse::Input);
        // const vector<string> targets_names = language_dataset.get_variable_names(Dataset::VariableUse::Target);

        // const Index words_number = language_dataset.get_variables_number(Dataset::VariableUse::Input);
        // const Index target_variables_number = language_dataset.get_variables_number(Dataset::VariableUse::Target);

        // cout<<words_number<<endl;
        // cout<<target_variables_number<<endl;

        // @todo get input and target dimensions

        // Neural Network

//         NeuralNetwork neural_network(NeuralNetwork::ModelType::TextClassification,
//                                      { 1885, 40, 32 }, { }, { 1 });

// //        neural_network.print(); Improve to show something nice

//         // Training Strategy

//         TrainingStrategy training_strategy(&neural_network, &language_dataset);

//         training_strategy.print();

        //training_strategy.perform_training();

        // language_dataset.set(Dataset::SampleUse::Testing);
/*
        const TestingAnalysis testing_analysis(&neural_network, &language_dataset);

        TestingAnalysis::RocAnalysis roc_analysis = testing_analysis.perform_roc_analysis();

        cout << "TESTING ANALYSIS:" << endl;
        cout << "Roc curve: " << roc_analysis.area_under_curve << endl;

        // pair<type, type> transformer_error_accuracy = testing_analysis.test_transformer();

        // cout << "TESTING ANALYSIS:" << endl;
        // cout << "Testing error: " << transformer_error_accuracy.first << endl;
        // cout << "Testing accuracy: " << transformer_error_accuracy.second << endl;

        // transformer.save("/home/artelnics/Escritorio/andres_alonso/ViT/ENtoES_model.xml");
*/
        cout << "Good bye!" << endl;

        return 0;
    }
        catch(const exception& e)
        {
            cout << e.what() << endl;

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
// License along with this library; if not, write to the Free Software Foundation.
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
