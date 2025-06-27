//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A I R F O I L   S E L F   N O I S E   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <string>
#include <time.h>

#include "../../opennn/registry.h"
#include "../../opennn/dataset.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/perceptron_layer.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/model_selection.h"
#include "../../opennn/testing_analysis.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "Airfoil self noise " << endl;

        const Index neurons_number = 3;

        Dataset dataset("../data/airfoil_self_noise.csv", ";", true, false);

        ApproximationNetwork approximation_network(dataset.get_input_dimensions(), {neurons_number}, dataset.get_target_dimensions());

        //ApproximationNetwork approximation_network({1}, {1}, {1});

        approximation_network.print();

        approximation_network.save("../data/approximation_network.xml");

        approximation_network.load("../data/approximation_network.xml");
/*
        for (const auto& name : Registry<LossIndex>::instance().registered_names())
            std::cout << "Registered loss: " << name << "\n";

        for (const auto& name : Registry<OptimizationAlgorithm>::instance().registered_names())
            std::cout << "Registered optimizer: " << name << "\n";


        MeanSquaredError mean_squared_error(&aproximation_network, &dataset);

        cout << mean_squared_error.calculate_numerical_error() << endl;

/*
        AdaptiveMomentEstimation adaptive_moment_estimation(&mean_squared_error);

        adaptive_moment_estimation.perform_training();

//        TrainingStrategy training_strategy(&neural_network, &dataset);

//        training_strategy.perform_training();
/*
        TestingAnalysis testing_analysis(&neural_network, &dataset);

        testing_analysis.print_goodness_of_fit_analysis();
*/
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
// Copyright (C) Artificial Intelligence Techniques SL.
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
