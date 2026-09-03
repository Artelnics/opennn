//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//  T R A N S L A T I O N   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>
#include <string>

#include "opennn/core/configuration.h"
#include "opennn/dataset/language_dataset.h"
#include "opennn/models/models.h"
#include "opennn/neural_network/chat.h"
#include "opennn/training_strategy/training_strategy.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Translation Example." << endl;

        // Autoregressive generation currently requires CUDA.
        Configuration::instance().set(Device::CUDA, Type::FP32);

        LanguageDataset dataset("../data/translation/ES-EN-small.txt");

        Transformer transformer(dataset.get_input_shape()[0],
                                dataset.get_shape(VariableRole::Decoder)[0],
                                dataset.get_input_vocabulary_size(),
                                dataset.get_target_vocabulary_size(),
                                256, 8, 1024, 1);

        TrainingStrategy training_strategy(&transformer, &dataset);
        auto& optimizer = *training_strategy.get_optimization_algorithm();
        optimizer.set_batch_size(16);
        optimizer.set_maximum_epochs(50);

        training_strategy.train();

        ChatSession session(transformer);
        const string source = "yo tengo hambre";

        cout << "Translation of '" << source << "': "
             << session.send(source).content << endl;

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
