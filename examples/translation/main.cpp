//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//  T R A N S L A T I O N   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

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

        LanguageDataSet language_data_set("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/amazon_cells_labelled.txt");

        //language_data_set.set_data_source_path("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/ENtoES_dataset.txt");
        // language_data_set.set_data_source_path("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/ENtoES_dataset50000.txt");
        // language_data_set.set_data_source_path("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/test50000-60000.txt");
        // language_data_set.set_data_source_path("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/dataset_ingles_espanol.txt");
        // language_data_set.set_data_source_path("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/amazon_cells_reduced.txt");
        // language_data_set.set_data_source_path("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/amazon_cells_labelled.txt");

        language_data_set.read_txt();
        // cout<<language_data_set.get_data().dimensions()<<endl;
        language_data_set.set_raw_variable_scalers(Scaler::None);

        vector<string> completion_vocabulary = language_data_set.get_completion_vocabulary();
        // const vector<string> completion_vocabulary = { "good" ,  "bad" };
        vector<string> context_vocabulary = language_data_set.get_context_vocabulary();

        const Index embedding_depth = 64;
        const Index perceptron_depth = 128;
        const Index heads_number = 4;
        const Index number_of_layers = 1;
        const vector <Index> complexity = {embedding_depth, perceptron_depth, heads_number, number_of_layers};

        const dimensions input_dimensions = {language_data_set.get_completion_length(), language_data_set.get_completion_vocabulary_size()};
        const dimensions context_dimensions = {language_data_set.get_context_length(), language_data_set.get_context_vocabulary_size()};


        language_data_set.save_vocabulary("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/completion_vocabulary.txt",completion_vocabulary);
        language_data_set.save_vocabulary("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/context_vocabulary.txt",context_vocabulary);
        language_data_set.save_lengths("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/lengths.txt", input_dimensions[0], context_dimensions[0]);

        // Neural network

        // Transformer transformer({ input_length, context_length, inputs_dimension, context_dimension,
        //                          embedding_depth, perceptron_depth, heads_number, number_of_layers });

        Transformer transformer(input_dimensions, context_dimensions, complexity);

        //transformer.set_context_vocabulary();
        //transformer.set_context_vocabulary();

        transformer.set_model_type_string("TextClassification");
        transformer.set_dropout_rate(0);

        cout << "Total number of parameters: " << transformer.get_parameters_number() << endl;

        // Training strategy

        TrainingStrategy training_strategy(&transformer, &language_data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR_3D);

        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        training_strategy.get_adaptive_moment_estimation()->set_custom_learning_rate(embedding_depth);

        training_strategy.get_adaptive_moment_estimation()->set_loss_goal(0.01);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(100000);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_time(59400);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(64);

        training_strategy.get_adaptive_moment_estimation()->set_display(true);
        training_strategy.get_adaptive_moment_estimation()->set_display_period(1);

        TrainingResults training_results = training_strategy.perform_training();

        const TestingAnalysis testing_analysis(&transformer, &language_data_set);

        pair<type, type> transformer_error_accuracy = testing_analysis.test_transformer();

        cout << "TESTING ANALYSIS:" << endl;
        cout << "Testing error: " << transformer_error_accuracy.first << endl;
        cout << "Testing accuracy: " << transformer_error_accuracy.second << endl;

        transformer.save("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/sentimental_analysis.xml");

        cout << "Calculating confusion...." << endl;
        const Tensor<Index, 2> confusion = testing_analysis.calculate_sentimental_analysis_transformer_confusion();
        cout << "\nConfusion matrix:\n" << confusion << endl;

        // string prediction = testing_analysis.test_transformer({"I only hear garbage for audio."},false);
        // cout<<prediction<<endl;

        // string translation = testing_analysis.test_transformer({"I like dogs."},false);
        // // cout<<translation<<endl;





        // // Testing analysis

        // LanguageDataSet language_data_set("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/amazon_cells_labelled.txt");

        // vector<string> completion_vocabulary;
        // vector<string> context_vocabulary;

        // // language_data_set.import_vocabulary("/home/artelnics/Escritorio/andres_alonso/ViT/completion_vocabulary.txt",completion_vocabulary);
        // // language_data_set.import_vocabulary("/home/artelnics/Escritorio/andres_alonso/ViT/context_vocabulary.txt",context_vocabulary);

        // language_data_set.import_vocabulary("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/testing/completion_vocabulary.txt",completion_vocabulary);
        // language_data_set.import_vocabulary("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/testing/context_vocabulary.txt",context_vocabulary);

        // language_data_set.set_completion_vocabulary(completion_vocabulary);
        // language_data_set.set_context_vocabulary(context_vocabulary);

        // Index input_length;
        // Index context_length;

        // // language_data_set.import_lengths("/home/artelnics/Escritorio/andres_alonso/ViT/lengths.txt", input_length, context_length);
        // language_data_set.import_lengths("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/testing/lengths.txt", input_length, context_length);

        // const dimensions input_dimensions = {input_length, language_data_set.get_completion_vocabulary_size()};
        // const dimensions context_dimensions = {context_length, language_data_set.get_context_vocabulary_size()};

        // const Index embedding_depth = 64;
        // const Index perceptron_depth = 128;
        // const Index heads_number = 4;
        // const Index number_of_layers = 1;
        // const vector <Index> complexity = {embedding_depth, perceptron_depth, heads_number, number_of_layers};

        // Transformer transformer(input_dimensions, context_dimensions, complexity);

        // transformer.set_dropout_rate(0);

        // cout << "Total number of parameters: " << transformer.get_parameters_number() << endl;

        // transformer.set_input_vocabulary(completion_vocabulary);
        // transformer.set_context_vocabulary(context_vocabulary);

        // // transformer.load_transformer("/home/artelnics/Escritorio/andres_alonso/ViT/EnToEs.xml");
        // transformer.load_transformer("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/testing/sentimental_analysis.xml");

        // transformer.set_model_type_string("TextClassification");

        // const TestingAnalysis testing_analysis(&transformer, &language_data_set);

        // // cout << "Calculating confusion...." << endl;
        // // const Tensor<Index, 2> confusion = testing_analysis.calculate_transformer_confusion();
        // // cout << "\nConfusion matrix:\n" << confusion << endl;

        // string prediction = testing_analysis.test_transformer({"I only hear garbage for audio."},false);
        // cout<<prediction<<endl;
        // cout<<"Target: bad"<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"Mic Doesn't work."},false);
        // cout<<prediction<<endl;
        // cout<<"Target: bad"<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"I love this phone , It is very handy and has a lot of features ."},false);
        // cout<<prediction<<endl;
        // cout<<"Target: good"<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"Buyer Beware, you could flush money right down the toilet."},false);
        // cout<<prediction<<endl;
        // cout<<"Target: bad"<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"Best I've found so far .... I've tried 2 other bluetooths and this one has the best quality (for both me and the listener) as well as ease of using."},false);
        // cout<<prediction<<endl;
        // cout<<"Target: good"<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"Arrived quickly and much less expensive than others being sold."},false);
        // cout<<prediction<<endl;
        // cout<<"Target: good"<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"I can't use this case because the smell is disgusting."},false);
        // cout<<prediction<<endl;
        // cout<<"Target: bad"<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"Excellent sound, battery life and inconspicuous to boot!."},false);
        // cout<<prediction<<endl;
        // cout<<"Target: good"<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"I do not like the product. Very bad quality."},false);
        // cout<<prediction<<endl;
        // cout<<"Target: bad"<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"Incredible product. The sound is just excellent."},false);
        // cout<<prediction<<endl;
        // cout<<"Target: good"<<endl;
        // cout<<endl;


        // //only good reviews:

        // string prediction = testing_analysis.test_transformer({"I have to use the smallest earpieces provided, but it stays on pretty well."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"I have always used corded headsets and the freedom from the wireless is very helpful."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"This BlueAnt Supertooth hands-free phone speaker is AWESOME."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"I bought this battery with a coupon from Amazon and I'm very happy with my purchase."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"you can even take self portraits with the outside (exterior) display, very cool."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"Also its slim enough to fit into my alarm clock docking station without removing the case."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"Best of all is the rotating feature, very helpful."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"I would highly recommend this."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"I had absolutely no problem with this headset linking to my 8530 Blackberry Curve!"},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"The keyboard is a nice compromise between a full QWERTY and the basic cell phone number keypad."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // string translation = testing_analysis.test_transformer({"I like dogs."},true);
        // cout<<translation<<endl;

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
// Copyright (C) 2005-2024 Artificial Intelligence Techniques SL
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
