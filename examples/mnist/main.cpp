//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M N I S T    A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>

#include "opennn/dataset/image_dataset.h"
#include "opennn/models/models.h"
#include "opennn/training_strategy/training_strategy.h"
#include "opennn/testing_analysis/testing_analysis.h"
#include "opennn/core/random_utilities.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. MNIST example." << endl;

        set_seed(42);

        ImageDataset dataset("../data/mnist");

        ImageClassificationNetwork network(dataset.get_input_shape(),
                                           {4},
                                           dataset.get_target_shape());

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.get_optimization_algorithm()->set_maximum_epochs(20);

        training_strategy.train();

        const TestingAnalysis testing_analysis(&network, &dataset);

        testing_analysis.print_multiple_classification_tests();

        return 0;
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
