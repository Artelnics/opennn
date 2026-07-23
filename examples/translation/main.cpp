//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//  T R A N S L A T I O N   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// System includes

#include <iostream>
#include <string>
#include <cstring>


#include "opennn/training_strategy.h"
#include "opennn/text_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/adaptive_moment_estimation.h"
#include "opennn/random_utilities.h"

using namespace std;
using namespace opennn;

int main(int argc, char* argv[])
{
    try
    {
        cout << "OpenNN. Translation Example." << endl;

        Configuration::instance().set(Device::Auto, Type::FP32);


        unique_ptr<TextDataset> language_dataset =
            TextDataset::from_sequence_to_sequence("../data/translation/ES-EN-small.txt");

        const Index input_vocabulary_size  = ssize(language_dataset->get_input_vocabulary());
        const Index output_vocabulary_size = ssize(language_dataset->get_target_vocabulary());

        const Index input_sequence_length   = language_dataset->get_shape("Input")[0];
        const Index decoder_sequence_length = language_dataset->get_shape("Decoder")[0];
        const Index target_sequence_length  = language_dataset->get_shape("Target")[0];

        if(decoder_sequence_length != target_sequence_length)
            throw runtime_error("Decoder and target sequence lengths must match.");


        const Index embedding_dimension = 256;
        const Index heads_number = 8;
        const Index feed_forward_dimension = 1024;
        const Index layers_number = 1;

        Transformer transformer(input_sequence_length,
                                decoder_sequence_length,
                                input_vocabulary_size,
                                output_vocabulary_size,
                                embedding_dimension,
                                heads_number,
                                feed_forward_dimension,
                                layers_number);


        TrainingStrategy training_strategy(&transformer, language_dataset.get());

        training_strategy.set_loss("CrossEntropyError3d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());

        if(!adam)
            throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

        adam->set_batch_size(16);
        adam->set_learning_rate(0.0005);
        const Index maximum_epochs = argc > 1 ? Index(stol(argv[1])) : 50;
        adam->set_maximum_epochs(maximum_epochs);
        adam->set_display_period(5);

        cout << "\nTraining on "
             << (Configuration::instance().is_gpu() ? "GPU" : "CPU")
             << "..." << endl;
        training_strategy.train();


        cout << "\n================ TRANSFORMER PREDICTIONS ================\n";

        const vector<string> test_sources =
            {
                "yo tengo hambre",
                "tu estas feliz",
                "el esta cansado",
                "yo veo el gato"
            };

        transformer.set_input_vocabulary(language_dataset->get_input_vocabulary());
        transformer.set_target_vocabulary(language_dataset->get_target_vocabulary());

        if(!Configuration::instance().is_gpu())
        {
            cout << "Autoregressive decoding is GPU-only; skipping prediction samples in this CPU build.\n";
            cout << "=========================================================\n";
            cout << "Bye!" << endl;
            return 0;
        }

        for(Index i = 0; i < static_cast<Index>(test_sources.size()); i++)
        {
            const string prediction = transformer.decode(test_sources[i]);

            cout << "Sample " << i << endl;
            cout << "  Source:    " << test_sources[i] << endl;
            cout << "  Predicted: " << prediction << endl;
            cout << endl;
        }

        cout << "=========================================================\n";
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
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
