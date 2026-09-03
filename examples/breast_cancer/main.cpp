//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B R E A S T   C A N C E R   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>

#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/models/models.h"
#include "opennn/training_strategy/training_strategy.h"
#include "opennn/testing_analysis/testing_analysis.h"
#include "opennn/training_strategy/optimizer.h"
#include "opennn/training_strategy/stochastic_gradient_descent.h"
#include "opennn/core/random_utilities.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Breast Cancer Example." << endl;

        TabularDataset dataset("../data/breast_cancer/breast_cancer.csv", ";", true, false);

        const Index neurons_number = 3;

        ClassificationNetwork classification_network(dataset.get_input_shape(), { neurons_number}, dataset.get_target_shape());

        TrainingStrategy training_strategy(&classification_network, &dataset);

        training_strategy.train();

        TestingAnalysis testing_analysis(&classification_network, &dataset);

        testing_analysis.print_binary_classification_tests();

        TestingAnalysis::RocAnalysis roc = testing_analysis.perform_roc_analysis();

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
