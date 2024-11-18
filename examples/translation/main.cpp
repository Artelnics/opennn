//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I R I S   P L A N T   A P P L I C A T I O N
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

using namespace std;
using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Translation Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        /*
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        */

        // Data set

        LanguageDataSet language_data_set;

        language_data_set.set_data_source_path("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/ENtoES_dataset50000.txt");
        //language_data_set.set_data_source_path("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/language_dataset_debug.txt");

        language_data_set.set_separator(DataSet::Separator::Tab);

        language_data_set.read_txt_language_model();

        language_data_set.set_raw_variable_scalers(Scaler::None);

        Tensor<string, 1> completion_vocabulary = language_data_set.get_completion_vocabulary();
        Tensor<string, 1> context_vocabulary = language_data_set.get_context_vocabulary();

        // cout<<completion_vocabulary.dimensions()<<endl;
        // cout<<completion_vocabulary<<endl;
        // cout<<context_vocabulary.dimensions()<<endl;

        // Neural network

        Index input_length = language_data_set.get_completion_length();
        Index context_length = language_data_set.get_context_length();
        Index inputs_dimension = language_data_set.get_completion_vocabulary_size();
        Index context_dimension = language_data_set.get_context_vocabulary_size();

        Index number_of_layers = 1;
        Index depth = 64;
        Index perceptron_depth = 128;
        Index heads_number = 4;



        Transformer transformer({ input_length, context_length, inputs_dimension, context_dimension,
                                 depth, perceptron_depth, heads_number, number_of_layers });

        transformer.set_model_type_string("TextClassification");
        transformer.set_dropout_rate(0);

        cout << "Total number of parameters: " << transformer.get_parameters_number() << endl;

        transformer.set_input_vocabulary(completion_vocabulary);
        transformer.set_context_vocabulary(context_vocabulary);

        // Training strategy

        TrainingStrategy training_strategy(&transformer, &language_data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR_3D);

        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        training_strategy.get_adaptive_moment_estimation()->set_custom_learning_rate(depth);

        training_strategy.get_adaptive_moment_estimation()->set_loss_goal(0.99);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(4000);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_time(237600);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(64);

        training_strategy.get_adaptive_moment_estimation()->set_display(true);
        training_strategy.get_adaptive_moment_estimation()->set_display_period(1);

        TrainingResults training_results = training_strategy.perform_training();

        // Save results-

        transformer.save("/home/artelnics/Escritorio/andres_alonso/ViT/ENtoES_model.xml");

        // // Testing analysis

        // transformer.load("/home/artelnics/Escritorio/andres_alonso/ViT/ENtoES_model.xml");

        // const TestingAnalysis testing_analysis(&transformer, &language_data_set);

        // pair<type, type> transformer_error_accuracy = testing_analysis.test_transformer();

        // cout << "TESTING ANALYSIS:" << endl;
        // cout << "Testing error: " << transformer_error_accuracy.first << endl;
        // cout << "Testing accuracy: " << transformer_error_accuracy.second << endl;

        cout << "Bye!" << endl;
        return 0;
    }
    catch(const exception& e)
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
