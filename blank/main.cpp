//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

#include "../opennn/opennn.h"

using namespace std;
using namespace opennn;
using namespace std::chrono;
using namespace Eigen;

int main()
{
    try
    {
        cout << "OpenNN. Blank." << endl;

        // DataSet data_set;
        // NeuralNetwork neural_network;
        // data_set.save("data_set_empty.xml");
        // neural_network.save("neural_network_empty.xml");

        // TrainingStrategy training_strategy(&neural_network, &data_set);

        // training_strategy.save("training_strategy_empty.xml");

        // ModelSelection model_selection(&training_strategy);

        // model_selection.save("model_selection_empty.xml");

        // @todo Create an example to test if the numerical hessian works properly.
        // Also test if the outputs deltas are modified or it's the jacobian itself to test if the NAN error problem lays there

        NeuralNetwork neural_network;
        DataSet data_set(7, {1}, {1});
        Tensor<type, 2> data(7,2);
        for(Index i = 0; i < 7; i++)
        {
            data(i,0) = i;
            data(i,1) = sin(i);
        }
        data_set.set_data(data);

        neural_network.add_layer(make_unique<PerceptronLayer>((dimensions){1}, (dimensions){1}, PerceptronLayer::ActivationFunction::RectifiedLinear));

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM);

        cerr << data << endl;
        training_strategy.perform_training();


        cout << "Bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques SL
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
