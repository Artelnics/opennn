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
#include "../../opennn/normalized_squared_error.h"
#include "../../opennn/training_strategy.h"
#include "adaptive_moment_estimation.h"
#include "testing_analysis.h"
#include "recurrent_layer.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Forecasting Example." << endl;

        const Index batch_size = 3;

        TimeSeriesDataset time_series_dataset("../data/funcion_seno_inputTarget.csv", ",", false, false);

        time_series_dataset.split_samples_sequential(type(0.01), type(0.5), type(0.49));

        time_series_dataset.set_raw_variable_use(0, "InputTarget");

//        print_vector(time_series_dataset.get_dimensions("Input"));
//        print_vector(time_series_dataset.get_dimensions("Target"));

//        time_series_dataset.print();

        Batch batch(batch_size, &time_series_dataset);

        const vector<Index> sample_indices = {0, 1, 2};

        const vector<Index> input_indices = {0};

        const vector<Index> target_indices =  {0};

        batch.fill(sample_indices, input_indices, target_indices);

        batch.print();
/*
        // cout << "#############" << endl;

        // Tensor<type, 3> inputs(3, 1, 2);
        // inputs.setValues({{{0.0,0.099833417} },
        //                  {{0.099833417, 0.198669331} },
        //                  {{0.198669331, 0.295520207}}});

        // Tensor<type, 2> targets(3, 1);
        // targets.setValues({{0.198669331},
        //                    {0.295520207},
        //                    {0.389418342}});
        // cout << "inputs" << endl;
        // cout << inputs << endl;
        // cout << "target" << endl;
        // cout << targets << endl;

        cout << "------------------------------------------" << endl;

        // ForecastingNetwork forecasting_network({time_series_dataset.get_variables_number("Input")},
        //                                   {},
        //                                   {time_series_dataset.get_variables_number("Target")});

        // Layer* layer_ptr = forecasting_network.get_first("Recurrent");
        // Recurrent* recurrent_layer = dynamic_cast<Recurrent*>(layer_ptr);
        // recurrent_layer->set_activation_function("HyperbolicTangent");
        // recurrent_layer->set_timesteps(1);

        ForecastingNetwork forecasting_network({1},
                                          {},
                                          {1});

        forecasting_network.print();

        cout << "------------------------------------------" << endl;
        // for(Index i = 0; i < 10; i++)
        // {
        //     Tensor<type, 3> inputs(1,1,2);
        //     inputs.setValues({{{0}, {0.0998334166468282}}});
        //     cout << "Inputs" << endl;
        //     cout << inputs << endl;

        //     const Tensor<type, 2> outputs = forecasting_network.calculate_outputs<3,2>(inputs);

        //     cout << outputs << endl;
        // }


        NormalizedSquaredError normalized_squared_error(&forecasting_network, &time_series_dataset);

        // for(Index i = 0; i < 10; i++)
        cout << normalized_squared_error.calculate_numerical_error() << endl;


        // const Tensor<type, 1> gradient = normalized_squared_error.calculate_gradient();
        // const Tensor<type, 1> numerical_gradient = normalized_squared_error.calculate_numerical_gradient();

        // cout << "Gradient" << endl;
        // cout << gradient << endl;
        // cout << "Numerical Gradient" << endl;
        // cout << numerical_gradient << endl;
        // cout << "diferencia" << endl;
        // cout << gradient.abs() - numerical_gradient.abs() << endl;

        // cout << "Error" << endl;
        // cout << normalized_squared_error.calculate_numerical_error() << endl;

        // TrainingStrategy training_strategy(&forecasting_network, &time_series_dataset);
        // training_strategy.set_loss_index("MeanSquaredError");
        // training_strategy.get_loss_index()->set_regularization_method("None");

        // training_strategy.train();

        // cout << "Error" << endl;
        // cout << normalized_squared_error.calculate_numerical_error() << endl;
*/

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
