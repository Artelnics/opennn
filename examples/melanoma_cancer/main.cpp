// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, S.L. and Roberto Lopez

//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E L A N O M A   C A N C E R   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

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
        cout << "OpenNN. Melanoma cancer example." << endl;

        set_seed(0);

        // ISIC 2016 images and labels, dedicated to the public domain under CC0.
        ImageDataset dataset("../data/melanoma_cancer/isic2016");

        ImageClassificationNetwork network(dataset.get_input_shape(),
                                           {32, 64, 16},
                                           dataset.get_target_shape());

        Training training(&network, &dataset);

        training.train();

        Evaluation evaluation(&network, &dataset);

        evaluation.print_binary_classification_tests();

        cout << "Bye!" << endl;

        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
