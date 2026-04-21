//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A   —   Translation benchmark (refactor)
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <chrono>

#include "../opennn/language_dataset.h"
#include "../opennn/standard_networks.h"
#include "../opennn/training_strategy.h"
#include "../opennn/adaptive_moment_estimation.h"
#include "../opennn/random_utilities.h"

using namespace opennn;
using namespace std::chrono;

int main()
{
    try
    {
        cout << "OpenNN. Translation benchmark (refactor)." << endl;

#ifdef OPENNN_WITH_CUDA

        LanguageDataset dataset("/home/artelnics/Documents/openNN/opennn/temp/translation_en_es.txt");
        dataset.split_samples_random(0.8, 0.1, 0.1);

        const Index input_vocab  = dataset.get_input_vocabulary_size();
        const Index output_vocab = dataset.get_target_vocabulary_size();
        const Index input_seq    = dataset.get_shape("Input")[0];
        const Index decoder_seq  = dataset.get_shape("Decoder")[0];
        const Index target_seq   = dataset.get_shape("Target")[0];

        if(decoder_seq != target_seq)
            throw runtime_error("Decoder and target sequence lengths must match.");

        const Index embedding_dimension     = 128;
        const Index heads_number            = 4;
        const Index feed_forward_dimension  = 256;
        const Index layers_number           = 3;

        Transformer transformer(input_seq,
                                decoder_seq,
                                input_vocab,
                                output_vocab,
                                embedding_dimension,
                                heads_number,
                                feed_forward_dimension,
                                layers_number);

        transformer.set_input_vocabulary(dataset.get_input_vocabulary());
        transformer.set_output_vocabulary(dataset.get_target_vocabulary());
        transformer.set_dropout_rate(type(0));

        TrainingStrategy training_strategy(&transformer, &dataset);

        training_strategy.set_loss("CrossEntropyError3d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        if(!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

        adam->set_batch_size(64);
        adam->set_learning_rate(type(5e-4));
        adam->set_maximum_epochs(2);
        adam->set_display_period(1);

        Device::instance().set(DeviceType::Gpu);

        cout << "[PARITY] train="  << dataset.get_samples_number("Training")
             << " val="            << dataset.get_samples_number("Validation")
             << " test="           << dataset.get_samples_number("Testing")
             << " input_vocab="    << input_vocab
             << " target_vocab="   << output_vocab
             << " input_seq="      << input_seq
             << " decoder_seq="    << decoder_seq
             << " params="         << transformer.get_parameters_number()
             << " (buffer="        << transformer.get_parameters_size() << ")" << endl;

        const auto t0 = steady_clock::now();
        training_strategy.train();
        const auto t1 = steady_clock::now();

        const double training_seconds = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;
        cout << "\nTotal training time (refactor): " << training_seconds << " s" << endl;

        // Predictions

        cout << "\n================ TRANSFORMER PREDICTIONS ================\n";

        const vector<string> test_sources =
        {
            "the cat eats an apple",
            "a boy sings happily",
            "my friend writes a book",
            "the teacher speaks loudly"
        };

        for(Index i = 0; i < Index(test_sources.size()); ++i)
        {
            const string prediction = transformer.calculate_outputs(test_sources[i]);

            cout << "Sample " << i << endl;
            cout << "  Source:    " << test_sources[i] << endl;
            cout << "  Predicted: " << prediction << endl;
            cout << endl;
        }

        cout << "=========================================================\n";

#endif

        cout << "Bye!" << endl;
        return 0;
    }
    catch(const exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
