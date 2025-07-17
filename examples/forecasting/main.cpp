//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R E C A S T I N G   P R O J E C T
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

#include "../../opennn/time_series_dataset.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/training_strategy.h"
#include "adaptive_moment_estimation.h"
#include "testing_analysis.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Forecasting Example." << endl;

        //loss inddex calculate numerica gragient y gradient igual a 0
        // Data set

        //TimeSeriesDataset time_series_dataset("../data/madridNO2forecasting.csv", ",", true, false);
        //TimeSeriesDataset time_series_dataset("../data/Pendulum.csv", ",", false, false);
        //TimeSeriesDataset time_series_dataset("../data/twopendulum.csv", ";", false, false);
        TimeSeriesDataset time_series_dataset("../data/funcion_seno.csv", ";", false, false);
        //TimeSeriesDataset time_series_dataset("../data/funcion_seno_inputTarget.csv", ",", false, false);

        time_series_dataset.split_samples_sequential(type(0.7), type(0.15), type(0.15));

        time_series_dataset.print();

        const Index neurons_number = 1;
        const Index time_steps =4;
        const Index input_size = time_series_dataset.get_raw_variables_number("Input");
        const Index batch_size = 1000;
        const Index epoch = 1000;

        const vector<Index> input_variable_indices = time_series_dataset.get_variable_indices("Input");
        const vector<Index> target_variable_indices = time_series_dataset.get_variable_indices("Target");
        const vector<Index> decoder_variable_indices = time_series_dataset.get_variable_indices("Decoder");
        const vector<Index> all_variable_indices = time_series_dataset.get_used_variable_indices();

        const Index training_batches_number = time_series_dataset.get_samples_number() / batch_size;

        const vector<Index> training_samples_indices = time_series_dataset.get_sample_indices("Training");

        // NeuralNetwork neural_network;

        ForecastingNetwork neural_network({time_series_dataset.get_variables_number("Input")},
                                          {neurons_number},
                                          {time_series_dataset.get_variables_number("Target")});

        Layer* layer_ptr = neural_network.get_first("Recurrent");
        Recurrent* recurrent_layer = dynamic_cast<Recurrent*>(layer_ptr);
        recurrent_layer->set_activation_function("HyperbolicTangent");
        recurrent_layer->set_timesteps(1);

        neural_network.print();

        TrainingStrategy training_strategy(&neural_network, &time_series_dataset);
        training_strategy.set_loss_index("MeanSquaredError");
        training_strategy.get_loss_index()->set_regularization_method("NoRegularization");

        training_strategy.perform_training();



        cout << "Good bye!" << endl;

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
