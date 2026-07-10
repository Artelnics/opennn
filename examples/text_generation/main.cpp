//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//  T E X T   G E N E R A T I O N   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// System includes

#include <iostream>
#include <string>

// OpenNN includes

#include "opennn/training_strategy.h"
#include "opennn/text_generation_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/adaptive_moment_estimation.h"

using namespace std;
using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Text Generation Example." << endl;

        Configuration::instance().set(Device::Auto, Type::FP32);

        // Dataset

        const Index sequence_length = 64;

        TextGenerationDataset dataset("../data/shakespeare.txt", sequence_length);

        const Index vocabulary_size = dataset.get_vocabulary_size();

        cout << "Vocabulary size: " << vocabulary_size << endl;
        cout << "Samples number: " << dataset.get_samples_number() << endl;

        // Neural network

        const Index embedding_dimension = 256;
        const Index heads_number = 4;
        const Index feed_forward_dimension = 512;
        const Index layers_number = 4;

        TextGenerationNetwork neural_network(sequence_length,
                                             vocabulary_size,
                                             embedding_dimension,
                                             heads_number,
                                             feed_forward_dimension,
                                             layers_number);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &dataset);

        training_strategy.set_loss("CrossEntropyError3d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());

        if(!adam)
            throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

        adam->set_batch_size(32);
        adam->set_learning_rate(0.0005);
        adam->set_maximum_epochs(100);
        adam->set_display_period(5);

        training_strategy.train();

        // Generation

        cout << "\n================ GENERATED TEXT ================\n";

        // The vocabulary travels with the network; inference needs no dataset.
        neural_network.set_vocabulary(dataset.get_vocabulary());

        // Inference requires GPU (generate is GPU-only).
        SamplingConfig sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 40;
        sampling.maximum_tokens = 40;

        const vector<string> prompts =
            {
                "first citizen :",
                "to be or not",
                "the king shall"
            };

        for(const string& prompt : prompts)
        {
            cout << "Prompt:    " << prompt << endl;
            cout << "Generated: " << neural_network.generate(prompt, sampling) << endl;
            cout << endl;
        }

        cout << "================================================\n";
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
