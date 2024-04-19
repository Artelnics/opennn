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

#include "../../opennn/opennn.h"

using namespace opennn;

int main()
{          
    try
    {
        cout << "OpenNN. Rosenbrock Example." << endl;
        
//        srand(static_cast<unsigned>(time(nullptr)));
     
        // Data Set

        const Index samples_number = 1000000;
        const Index inputs_number = 1000;
        const Index outputs_number = 1;
        const Index hidden_neurons_number = 1000;
        
        DataSet data_set;// ("C:/R_100000_samples_11_variables.csv", ',', true);
        
        data_set.generate_Rosenbrock_data(samples_number, inputs_number + outputs_number);

        data_set.set_training();

        // Neural network

        NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {inputs_number, hidden_neurons_number, outputs_number});

        neural_network.get_first_perceptron_layer()->set_activation_function(PerceptronLayer::ActivationFunction::HyperbolicTangent);

        PerceptronLayer* pl = static_cast<PerceptronLayer*>(neural_network.get_layers()(2));

        pl->set_activation_function(PerceptronLayer::ActivationFunction::Linear);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        //training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::L2);
        //training_strategy.get_loss_index()->set_regularization_weight(0.01);

        training_strategy.set_maximum_epochs_number(10000);
        training_strategy.set_display_period(1);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(1000);
        training_strategy.set_maximum_time(86400);

        training_strategy.perform_training();
        
        cout << "End Rosenbrock" << endl;

        return 0;
    }
    catch(const exception& e)
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
