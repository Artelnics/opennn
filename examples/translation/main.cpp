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

        LanguageDataSet language_data_set("C:/translation.csv");
/*
        // cout<<language_data_set.get_context_length()<<endl;
        // cout<<language_data_set.get_completion_length()<<endl;
        cout << language_data_set.get_data().dimensions() << endl;
        
        /*
        const Index embedding_dimension = 64;
        const Index perceptron_depth = 128;
        const Index heads_number = 4;
        const Index number_of_layers = 1;

        const vector <Index> complexity = {embedding_dimension, perceptron_depth, heads_number, number_of_layers};


        // Neural network
        const dimensions target_dimensions = language_data_set.get_completion_dimensions();
        const dimensions input_dimensions = language_data_set.get_context_dimensions();
        
        Transformer transformer(target_dimensions, input_dimensions, complexity);
        transformer.set_input_vocabulary(language_data_set.get_completion_vocabulary());
        transformer.set_context_vocabulary(language_data_set.get_context_vocabulary());
        transformer.set_model_type_string("TextClassification");
        transformer.set_dropout_rate(0);

        cout << "Total number of parameters: " << transformer.get_parameters_number() << endl;

        const filesystem::path& file_name = "/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/language_data_set.xml";

        ofstream file(file_name);

        if (!file.is_open())
            throw runtime_error("file not found");

        XMLPrinter printer;
        language_data_set.to_XML(printer);
        file << printer.CStr();


        // Training strategy

        TrainingStrategy training_strategy(&transformer, &language_data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR_3D);

        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        training_strategy.get_adaptive_moment_estimation()->set_custom_learning_rate(complexity[0]);

        training_strategy.get_adaptive_moment_estimation()->set_loss_goal(0.5);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(10000);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_time(59400);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(64);

        training_strategy.get_adaptive_moment_estimation()->set_display(true);
        training_strategy.get_adaptive_moment_estimation()->set_display_period(1);

        TrainingResults training_results = training_strategy.perform_training();

        transformer.save("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/sentimental_analysis.xml");

        //Testing

        const TestingAnalysis testing_analysis(&transformer, &language_data_set);
        pair<type, type> transformer_error_accuracy = testing_analysis.test_transformer();

        cout << "TESTING ANALYSIS:" << endl;
        cout << "Testing error: " << transformer_error_accuracy.first << endl;
        cout << "Testing accuracy: " << transformer_error_accuracy.second << endl;


        string prediction = testing_analysis.test_transformer({"Good case, Excellent value."},false);
        cout<<prediction<<endl;
        cout<<"Target: good"<<endl;
        cout<<endl;

        prediction = testing_analysis.test_transformer({"So there is no way for me to plug it in here in the US unless I go by a converter."},false);
        cout<<prediction<<endl;
        cout<<"Target: bad"<<endl;
        cout<<endl;

        prediction = testing_analysis.test_transformer({"Great for the jawbone."},false);
        cout<<prediction<<endl;
        cout<<"Target: good"<<endl;
        cout<<endl;

        prediction = testing_analysis.test_transformer({"Tied to charger for conversations lasting more than 45 minutes.MAJOR PROBLEMS!!"},false);
        cout<<prediction<<endl;
        cout<<"Target: bad"<<endl;
        cout<<endl;

        prediction = testing_analysis.test_transformer({"The mic is great."},false);
        cout<<prediction<<endl;
        cout<<"Target: good"<<endl;
        cout<<endl;

        prediction = testing_analysis.test_transformer({"I have to jiggle the plug to get it to line up right to get decent volume."},false);
        cout<<prediction<<endl;
        cout<<"Target: bad"<<endl;
        cout<<endl;

        prediction = testing_analysis.test_transformer({"If you have several dozen or several hundred contacts, then imagine the fun of sending each of them one by one."},false);
        cout<<prediction<<endl;
        cout<<"Target: bad"<<endl;
        cout<<endl;

        prediction = testing_analysis.test_transformer({"If you are Razr owner...you must have this!"},false);
        cout<<prediction<<endl;
        cout<<"Target: good"<<endl;
        cout<<endl;

        prediction = testing_analysis.test_transformer({"Needless to say, I wasted my money."},false);
        cout<<prediction<<endl;
        cout<<"Target: bad"<<endl;
        cout<<endl;
        */

//----------------------------------------------------------------------------------------------------------------------------------------------//

        // Data Set


/*
        LanguageDataSet language_data_set({0},{0});

        language_data_set.load("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/language_data_set.xml");


        const vector<string>& completion_vocabulary = language_data_set.get_completion_vocabulary();
        const vector<string>& context_vocabulary = language_data_set.get_context_vocabulary();

        const Index embedding_dimension = 64;
        const Index perceptron_depth = 128;
        const Index heads_number = 4;
        const Index number_of_layers = 1;

        const vector <Index> complexity = {embedding_dimension, perceptron_depth, heads_number, number_of_layers};

        const dimensions target_dimensions = {language_data_set.get_completion_length(), language_data_set.get_completion_vocabulary_size()};

        const dimensions input_dimensions = {language_data_set.get_context_length(), language_data_set.get_context_vocabulary_size()};

        Transformer transformer(target_dimensions, input_dimensions, complexity);
        transformer.load_transformer("/home/artelnics/Escritorio/andres_alonso/ViT/dataset/amazon_reviews/sentimental_analysis.xml");
        transformer.set_model_type_string("TextClassification");

        transformer.set_input_vocabulary(completion_vocabulary);
        transformer.set_context_vocabulary(context_vocabulary);

        const TestingAnalysis testing_analysis(&transformer, &language_data_set);

        string prediction = testing_analysis.test_transformer({"Mic Doesn't work."},false);
        cout<<prediction<<endl;
        cout<<"Target: bad"<<endl;
        cout<<endl;


        // cout << "Calculating confusion...." << endl;
        // const Tensor<Index, 2> confusion = testing_analysis.calculate_transformer_confusion();
        // cout << "\nConfusion matrix:\n" << confusion << endl;

        prediction = testing_analysis.test_transformer({"I love this phone , It is very handy and has a lot of features ."},false);
        cout<<prediction<<endl;
        cout<<"Target: good"<<endl;
        cout<<endl;

        prediction = testing_analysis.test_transformer({"Buyer Beware, you could flush money right down the toilet."},false);
        cout<<prediction<<endl;
        cout<<"Target: bad"<<endl;
        cout<<endl;

        prediction = testing_analysis.test_transformer({"Best I've found so far .... I've tried 2 other bluetooths and this one has the best quality (for both me and the listener) as well as ease of using."},false);
        cout<<prediction<<endl;
        cout<<"Target: good"<<endl;
        cout<<endl;

        prediction = testing_analysis.test_transformer({"Arrived quickly and much less expensive than others being sold."},false);
        cout<<prediction<<endl;
        cout<<"Target: good"<<endl;
        cout<<endl;

        prediction = testing_analysis.test_transformer({"I can't use this case because the smell is disgusting."},false);
        cout<<prediction<<endl;
        cout<<"Target: bad"<<endl;
        cout<<endl;

        prediction = testing_analysis.test_transformer({"Excellent sound, battery life and inconspicuous to boot!."},false);
        cout<<prediction<<endl;
        cout<<"Target: good"<<endl;
        cout<<endl;

        prediction = testing_analysis.test_transformer({"I do not like the product. Very bad quality."},false);
        cout<<prediction<<endl;
        cout<<"Target: bad"<<endl;
        cout<<endl;

        prediction = testing_analysis.test_transformer({"Incredible product. The sound is just excellent."},false);
        cout<<prediction<<endl;
        cout<<"Target: good"<<endl;
        cout<<endl;




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





        // //only bad reviews:
        // string prediction = testing_analysis.test_transformer({"Tied to charger for conversations lasting more than 45 minutes.MAJOR PROBLEMS!!"},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"I have to jiggle the plug to get it to line up right to get decent volume."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"Not a good bargain."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"The construction of the headsets is poor."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"Could not get strong enough signal."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"it did not work in my cell phone plug i am very up set with the charger!."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"Basically the service was very bad."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"The majority of the Logitech earbud headsets failed."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"very disappointed."},false);
        // cout<<prediction<<endl;
        // cout<<endl;

        // prediction = testing_analysis.test_transformer({"This is essentially a communications tool that does not communicate."},false);
        // cout<<prediction<<endl;
        // cout<<endl;


*/
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
