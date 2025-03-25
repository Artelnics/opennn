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

        // Data set

        LanguageDataSet text_data_set;

        // text_data_set.set_data_path("../data/amazon_cells_reduced.txt");
        // text_data_set.set_data_path("/Users/artelnics/Documents/opennn/examples/amazon_reviews/data/amazon_cells_reduced.txt");
        text_data_set.set_data_path("/Users/artelnics/Documents/opennn/examples/amazon_reviews/data/amazon_cells_labelled.txt");

        text_data_set.set_separator(DataSet::Separator::Tab);

        text_data_set.read_csv();

        // const vector<string> input_words = text_data_set.get_raw_variable_names(DataSet::VariableUse::Input);
        // const vector<string> targets_names = text_data_set.get_variable_names(DataSet::VariableUse::Target);

        // const Index words_number = text_data_set.get_variables_number(DataSet::VariableUse::Input);
        // const Index target_variables_number = text_data_set.get_variables_number(DataSet::VariableUse::Target);

        // cout<<words_number<<endl;
        // cout<<target_variables_number<<endl;

        // // Neural Network

        // const Index hidden_neurons_number = 6;

        // NeuralNetwork neural_network(NeuralNetwork::ModelType::TextClassification,
        //                              { words_number }, { hidden_neurons_number }, { target_variables_number });

        // neural_network.print();

        // Neural Network

        const Index maximum_sequence_length = text_data_set.get_input_length();
        const Index vocabulary_size = text_data_set.get_input_vocabulary_size();
        const Index embedding_dimension = 32;
        const Index heads_number = 4;

        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Embedding>(vocabulary_size, maximum_sequence_length, embedding_dimension, "Embedding"));
        neural_network.add_layer(make_unique<MultiHeadAttention>(maximum_sequence_length, maximum_sequence_length, embedding_dimension, heads_number, false, "Multihead_attention"));
        neural_network.set_layer_inputs_indices("Multihead_attention",{"Embedding", "Embedding"});
        // neural_network.add_layer(make_unique<Perceptron3d>(maximum_sequence_length, embedding_dimension, 64));
        neural_network.add_layer(make_unique<Flatten3D>(neural_network.get_output_dimensions()));
        neural_network.add_layer(make_unique<ProbabilisticLayer>(neural_network.get_output_dimensions(), (dimensions){ 1 }));

        // Training Strategy

        TrainingStrategy training_strategy(&neural_network, &text_data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);

        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        // training_strategy.get_adaptive_moment_estimation()->set_custom_learning_rate(depth);

        training_strategy.get_adaptive_moment_estimation()->set_loss_goal(0.4);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(3000);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_time(244800);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(64);

        training_strategy.get_adaptive_moment_estimation()->set_display(true);
        training_strategy.get_adaptive_moment_estimation()->set_display_period(1);

        TrainingResults training_results = training_strategy.perform_training();

        const TestingAnalysis testing_analysis(&neural_network, &text_data_set);

        TestingAnalysis::RocAnalysis roc_analysis = testing_analysis.perform_roc_analysis();

        cout << "TESTING ANALYSIS:" << endl;
        cout << "Roc curve: " << roc_analysis.area_under_curve << endl;


        // pair<type, type> transformer_error_accuracy = testing_analysis.test_transformer();

        // cout << "TESTING ANALYSIS:" << endl;
        // cout << "Testing error: " << transformer_error_accuracy.first << endl;
        // cout << "Testing accuracy: " << transformer_error_accuracy.second << endl;

        // transformer.save("/home/artelnics/Escritorio/andres_alonso/ViT/ENtoES_model.xml");


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
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
