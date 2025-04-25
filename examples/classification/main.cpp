//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C L A S S I F I C A T I O N   P R O J E C T
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
        cout << "OpenNN. Classification Example." << endl;

        // Data set

        DataSet data_set; //("../data/1000_classification_problem.csv", ',', false);

        const Index samples_number = 100000;
        const Index variables_number = 100;
        const Index classes_number = 100;
        const Index hidden_neurons_number = 1000;

        //data_set.set_data_classification();

        //      const Index input_variables_number = data_set.get_input_variables_number();
        //      const Index target_variables_number = data_set.get_target_variables_number();

        // Set input and target indices

        vector<DataSet::RawVariable> columns = data_set.get_raw_variables();
        Tensor<Index, 1> input_columns_indices(variables_number);
        Tensor<Index, 1> target_columns_indices(classes_number);

        for(Index i = 0; i < columns.size(); i ++)
        {
            if(i < variables_number)
                input_columns_indices(i) = i;
            else
                target_columns_indices(i-variables_number) = i;
        }

        //data_set.set_input_target_raw_variable_indices(input_columns_indices, target_columns_indices);
        data_set.set(DataSet::SampleUse::Training);

        // Neural network

        NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification,
                                     {variables_number}, {hidden_neurons_number}, {classes_number});

        cout << "Number of parameters: " << neural_network.get_parameters_number() << endl;

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        training_strategy.set_maximum_epochs_number(10);
        training_strategy.set_display_period(1);
        training_strategy.set_maximum_time(86400);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(1000);

        training_strategy.perform_training();

        cout << "End Classification" << endl;

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
