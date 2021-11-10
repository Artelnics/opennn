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

using namespace OpenNN;
using namespace std;
using namespace Eigen;

int main()
{
    try
    {
        cout << "OpenNN. Blank application." << endl;

        Index rows_number = 3;
        Index columns_number = 4;

        Tensor<type,2> diagonal_matrix(columns_number,columns_number);
        Tensor<type,1> row;

        Tensor<type,2> activations(rows_number,columns_number);
        activations.setValues({{1,1,1,1},{2,2,2,2},{3,3,3,3}});

        Tensor<type,3> activations_derivatives(columns_number,rows_number,columns_number);

        diagonal_matrix.setZero();
        sum_diagonal(diagonal_matrix,type(1));

        Tensor<type,2> temp_matrix(rows_number,columns_number);

        cout << "activations:\n" << activations << endl;
        for(Index row_index = 0; row_index < rows_number; row_index++)
        {
            row = activations.chip(row_index,0);
            cout << "row:\n" << row << endl;

            temp_matrix =
                     row.reshape(Eigen::array<Index,2>({columns_number,1})).broadcast(Eigen::array<Index,2>({1,columns_number}))
                     *(diagonal_matrix - row.reshape(Eigen::array<Index,2>({1,columns_number})).broadcast(Eigen::array<Index,2>({columns_number,1})));

            cout << "temp_matrix:\n" << temp_matrix << endl;

            memcpy(activations_derivatives.data()+temp_matrix.size()*row_index, temp_matrix.data(), static_cast<size_t>(temp_matrix.size())*sizeof(type));
        }

        cout << "activations_derivatives:\n" << activations_derivatives << endl;

        system("pause");

        // -----------------------

        DataSet data_set("C:/Program Files/Neural Designer/examples/activityrecognition/activityrecognition.csv",';',true);

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        // Neural network

        const Index hidden_neurons_number = 10;

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, {input_variables_number, hidden_neurons_number, target_variables_number});

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_maximum_epochs_number(200);

        training_strategy.perform_training();

        /*
        srand(static_cast<unsigned>(time(nullptr)));

        DataSet ds("C:/Users/Usuario/Documents/Waste_monthly.csv", ',', true);

        ds.set_lags_number(2);
        ds.set_steps_ahead_number(1);

        ds.transform_time_series();

        ds.split_samples_sequential();

        const Index columns_number = ds.get_columns_number();

        ds.set_column_use(columns_number-1, DataSet::VariableUse::UnusedVariable);
        ds.set_column_use(columns_number-2, DataSet::VariableUse::UnusedVariable);
        ds.set_column_use(columns_number-3, DataSet::VariableUse::UnusedVariable);

        const Index inputs_number = ds.get_input_variables_number();
        const Index targets_number = ds.get_target_variables_number();

        const Index neurons_number = 3;

        ScalingLayer sl(inputs_number);

        LongShortTermMemoryLayer lstm(inputs_number, neurons_number);
        lstm.set_activation_function(LongShortTermMemoryLayer::ActivationFunction::Linear);
        lstm.set_recurrent_activation_function(LongShortTermMemoryLayer::ActivationFunction::HyperbolicTangent);
        lstm.set_timesteps(2);

        PerceptronLayer pl(neurons_number, targets_number);
        pl.set_activation_function(PerceptronLayer::ActivationFunction::Linear);

        UnscalingLayer ul(inputs_number);

        NeuralNetwork nn;

        nn.add_layer(&sl);
        nn.add_layer(&lstm);
        nn.add_layer(&pl);
        nn.add_layer(&ul);

        TrainingStrategy ts(&nn, &ds);

        ts.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        ts.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

        ts.perform_training();

        cout << "outputs: " << endl << nn.calculate_outputs(ds.get_input_data()) << endl;

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
