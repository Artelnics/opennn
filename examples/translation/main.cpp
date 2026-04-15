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

// OpenNN includes

#include "../../opennn/training_strategy.h"
#include "../../opennn/language_dataset.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/adaptive_moment_estimation.h"
#include "../../opennn/random_utilities.h"

using namespace std;
using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Translation Example." << endl;

        set_seed(42);

        // Dataset
        
        LanguageDataset language_dataset("../data/ES-EN-small.txt");

        const Index input_vocabulary_size  = language_dataset.get_input_vocabulary_size();
        const Index output_vocabulary_size = language_dataset.get_target_vocabulary_size();

        const Index input_sequence_length   = language_dataset.get_shape("Input")[0];
        const Index decoder_sequence_length = language_dataset.get_shape("Decoder")[0];
        const Index target_sequence_length  = language_dataset.get_shape("Target")[0];

        if(decoder_sequence_length != target_sequence_length)
            throw runtime_error("Decoder and target sequence lengths must match.");

        // Transformer

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

        transformer.set_input_vocabulary(language_dataset.get_input_vocabulary());
        transformer.set_output_vocabulary(language_dataset.get_target_vocabulary());

        // Training strategy

        TrainingStrategy training_strategy(&transformer, &language_dataset);

        training_strategy.set_loss("CrossEntropyError3d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());

        if(!adam)
            throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

        adam->set_batch_size(16);
        adam->set_learning_rate(0.0005);
        adam->set_maximum_epochs(50);
        adam->set_display_period(5);

#ifdef OPENNN_WITH_CUDA
        Device::instance().set(DeviceType::Gpu);
        cout << "\nTraining on GPU..." << endl;
#else
        cout << "\nTraining on CPU..." << endl;
#endif
        training_strategy.train();

        // Predictions
/*
        cout << "\n================ TRANSFORMER PREDICTIONS ================\n";

        const vector<string> test_sources =
            {
                "yo tengo hambre",
                "tu estas feliz",
                "el esta cansado",
                "el perro es grande",
                "yo veo el gato"
            };

        for(Index i = 0; i < static_cast<Index>(test_sources.size()); i++)
        {
            const string prediction = transformer.calculate_outputs(test_sources[i]);

            cout << "Sample " << i << endl;
            cout << "  Source:    " << test_sources[i] << endl;
            cout << "  Predicted: " << prediction << endl;
            cout << endl;
        }

        cout << "=========================================================\n";
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
// Copyright (C) 2005-2026 Artificial Intelligence Techniques SL
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA