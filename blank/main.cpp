//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
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
#include "../opennn/layer.h"

using namespace opennn;
using namespace std;
using namespace Eigen;

#include "data_set.h"

int main()
{
    try
    {
        cout<<"Blank script! "<<endl;

        srand(static_cast<unsigned>(time(nullptr)));

        DataSet data_set("/home/artelnics2020/Documents/NeuralDesignerProjects/NO2forecasting/madridNO2forecasting.csv",',',true);

        data_set.set_lags_number(2);
        data_set.set_steps_ahead_number(1);

        data_set.transform_time_series();

        data_set.print_data();

        data_set.set_missing_values_method(DataSet::MissingValuesMethod::Mean);
        data_set.scrub_missing_values();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

//        data_set.print_data();

        // Neural network

        const Index hidden_neurons_number = 3;

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Forecasting, {input_variables_number, hidden_neurons_number, target_variables_number});

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

        training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::L2);

        training_strategy.set_display(true);

        training_strategy.perform_training();


        // Dataset
        /*
        DataSet data_set("/home/artelnics2020/Escritorio/datasets/SUMAS2.csv",',',true);

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        data_set.scrub_missing_values();

        data_set.print();

        // Neural network

        const Index hidden_neurons_number = 3;

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Approximation, {input_variables_number, hidden_neurons_number, target_variables_number});

        neural_network.get_first_perceptron_layer_pointer()->set_activation_function(PerceptronLayer::ActivationFunction::Linear);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

//        training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::L2);

        training_strategy.set_display(false);

//        training_strategy.perform_training();

        // Model Selection

        ModelSelection model_selection;

        model_selection.set(&training_strategy);

        model_selection.set_inputs_selection_method(ModelSelection::InputsSelectionMethod::GENETIC_ALGORITHM);

        model_selection.get_genetic_algorithm_pointer()->set_elitism_size(2);

        model_selection.get_genetic_algorithm_pointer()->set_individuals_number(8);

        model_selection.get_genetic_algorithm_pointer()->set_maximum_epochs_number(100);

        model_selection.get_genetic_algorithm_pointer()->set_display(false);

        model_selection.get_genetic_algorithm_pointer()->set_mutation_rate(0.01);

        InputsSelectionResults inputs_selection_results = model_selection.perform_inputs_selection();
        */
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
