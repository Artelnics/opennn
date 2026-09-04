//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A I R F O I L   S E L F   N O I S E   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>

#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/testing_analysis/testing_analysis.h"
#include "opennn/training_strategy/training_strategy.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Airfoil self noise Example." << endl;

        set_seed(0);

        TabularDataset dataset("../data/airfoil_self_noise/airfoil_self_noise.csv", ";", true, false);

        ApproximationNetwork network(dataset.get_input_shape(),
                                     {12},
                                     dataset.get_target_shape());

        TrainingStrategy training_strategy(&network, &dataset);

        training_strategy.train();

        TestingAnalysis testing_analysis(&network, &dataset);

        testing_analysis.print_goodness_of_fit_analysis();

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
