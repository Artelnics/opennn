//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A I R F O I L   S E L F   N O I S E   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <string>

#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/training_strategy/training_strategy.h"
#include "opennn/testing_analysis/testing_analysis.h"
#include "opennn/model_selection/model_selection.h"
#include "opennn/core/random_utilities.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Airfoil self noise Example." << endl;

        set_seed(0);

        const Index neurons_number = 12;

        TabularDataset dataset("../data/airfoil_self_noise/airfoil_self_noise.csv", ";", true, false);

        ApproximationNetwork approximation_network(dataset.get_input_shape(), 
                                                   {neurons_number}, 
                                                   dataset.get_target_shape());

        TrainingStrategy training_strategy(&approximation_network, &dataset);

        TrainingResult training_results = training_strategy.train();

        TestingAnalysis testing_analysis(&approximation_network, &dataset);

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
