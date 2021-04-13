//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R O S E N B R O C K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace Eigen;

int main()
{
    try
    {
        cout << "OpenNN. Blank application." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data Set

        DataSet data_set("../data/airfoil_self_noise.csv", ';', true);

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        //data_set.set_default_columns_scalers();

        const Tensor<Descriptives, 1> input_descriptives = data_set.scale_input_variables();

        const Tensor<Descriptives, 1> target_descriptives = data_set.scale_target_variables();

        // Neural network

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Approximation, {input_variables_number, 10, target_variables_number});

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
        scaling_layer_pointer->set_descriptives(input_descriptives);
        //scaling_layer_pointer->set_scalers(scaling_methods);

        UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();
        unscaling_layer_pointer->set_descriptives(target_descriptives);
        //unscaling_layer_pointer->set_scalers(unscaling_methods);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM);

//        training_strategy.set_display_period(1);

//        training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        training_strategy.perform_training();

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
