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

#include "../../opennn/training_strategy.h"
#include "../../opennn/language_dataset.h"
#include "embedding_layer.h"
#include "flatten_layer_3d.h"
#include "multihead_attention_layer.h"
#include "normalization_layer_3d.h"
#include "perceptron_layer.h"
#include "transformer.h"

using namespace std;
using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Translation Example." << endl;

        // Data set

        LanguageDataset language_dataset("/Users/artelnics/Documents/opennn/examples/translation/data/ENtoES_dataset_reduced_6.txt");

        // Sentiment analysis case

        const Index sequence_length = 10;
        const Index vocabulary_size = 50;
        const Index embedding_dimension = 32;
        const Index heads_number = 4;
        const dimensions outputs_number = { 1 };

        Transformer transformer;

        TrainingStrategy training_strategy(&transformer, &language_dataset);

        training_strategy.perform_training();

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
// Copyright (C) 2005-2025 Artificial Intelligence Techniques SL
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
