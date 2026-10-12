// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include <iostream>

#include "opennn/dataset/image_dataset.h"
#include "opennn/models/models.h"
#include "opennn/training/training.h"
#include "opennn/evaluation/evaluation.h"
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

        Training training(&network, &dataset);
        training.get_optimization_algorithm()->set_maximum_epochs(20);

        training.train();

        const Evaluation evaluation(&network, &dataset);

        evaluation.print_multiple_classification_tests();

        return 0;
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
