//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B R E A S T   C A N C E R   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>

#include "opennn/core/configuration.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/training/training.h"
#include "opennn/evaluation/evaluation.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Breast Cancer Example." << endl;

        Configuration::instance().set(Device::CPU, Type::FP32);

        TabularDataset dataset("../data/breast_cancer/breast_cancer.csv", ";", true, false);

        ClassificationNetwork network(dataset.get_input_shape(), {3}, dataset.get_target_shape());

        Training training(&network, &dataset);

        training.train();

        Evaluation evaluation(&network, &dataset);

        evaluation.print_binary_classification_tests();

        cout << "Good bye!" << endl;

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
