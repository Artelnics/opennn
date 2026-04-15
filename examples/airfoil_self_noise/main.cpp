//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A I R F O I L   S E L F   N O I S E   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <string>

#include "../../opennn/dataset.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/bounding_layer.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/model_selection.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/optimizer.h"
#include "../../opennn/stochastic_gradient_descent.h"
#include "../../opennn/random_utilities.h"

using namespace opennn;

int main()
{
    try
    {

        set_seed(42);

        cout << "Airfoil self noise" << endl;

        const Index neurons_number = 12;
        const type regularization_weight = type(0.001);

        // DataSet

        Dataset dataset("../data/airfoil_self_noise.csv", ";", true, false);

        dataset.split_samples_random(type(0.8), type(0.0), type(0.2));

        // Neural Network

        ApproximationNetwork approximation_network(dataset.get_input_shape(), {neurons_number}, dataset.get_target_shape());

        Bounding* bounding_layer = (Bounding*)approximation_network.get_first("Bounding");

        if(bounding_layer)
            bounding_layer->set_bounding_method("NoBounding");

        // Training strategy

        TrainingStrategy training_strategy(&approximation_network, &dataset);

        training_strategy.set_optimization_algorithm("StochasticGradientDescent");

        training_strategy.get_loss()->set_regularization("L2");
        training_strategy.get_loss()->set_regularization_weight(regularization_weight);
        training_strategy.get_optimization_algorithm()->set_display_period(50);

#ifdef OPENNN_WITH_CUDA
        Device::instance().set(DeviceType::Gpu);
#endif
        TrainingResults training_results = training_strategy.train();

        // Testing analysis

        //TestingAnalysis testing_analysis(&approximation_network, &dataset);
        //testing_analysis.print_goodness_of_fit_analysis();

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
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
