//   OpenNN: Open Neural Networks Library
//   www.opennn.org
//
//   V E H I C L E   I N S U R A N C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <limits.h>
#include <regex>

// Systems Complementaries

#include <cmath>
#include <cstdlib>
#include <ostream>


// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace chrono;

int main(void)
{
    try
    {
        cout << "OpenNN. Vehicle insurances Example." << endl;

        // Dataset

        DataSet data_set("../data/vehicle_insurances.csv", ';', true);

        data_set.split_samples_random();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        // Neural network

        const Index hidden_neurons_number = 3;

        Tensor<Index, 1> architecture(3);
        architecture.setValues({input_variables_number, hidden_neurons_number, target_variables_number});

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, architecture);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        AdaptiveMomentEstimation* adam = training_strategy.get_adaptive_moment_estimation_pointer();

        adam->set_loss_goal(type(1.0e-3));
        adam->set_maximum_epochs_number(10000);
        adam->set_display_period(1000);

        training_strategy.perform_training();

        // Testing analysis

        Tensor<type, 2> inputs(1,8);

        inputs.setValues({{type(1),type(1),type(33),type(0),type(2),type(1),type(23412),type(120)}});

        cout << "Inputs: " << endl;
        cout << inputs << endl;

        cout << "Outputs: " << endl;
        cout << neural_network.calculate_outputs(inputs) << endl;

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

        cout << "Confusion: " << endl;
        cout << confusion << endl;

        // Save results

        data_set.save("../data/data_set.xml");
        neural_network.save("../data/neural_network.xml");
        training_strategy.save("../data/training_strategy.xml");

        return 0;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques SL
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
