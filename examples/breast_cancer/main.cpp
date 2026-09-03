//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B R E A S T   C A N C E R   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>

#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/training_strategy/training_strategy.h"
#include "opennn/testing_analysis/testing_analysis.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Breast Cancer Example." << endl;

        TabularDataset dataset("../data/breast_cancer/breast_cancer.csv", ";", true, false);

        ClassificationNetwork network(dataset.get_input_shape(), {3}, dataset.get_target_shape());

        TrainingStrategy training_strategy(&network, &dataset);

        training_strategy.train();

        TestingAnalysis testing_analysis(&network, &dataset);

        testing_analysis.print_binary_classification_tests();

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
