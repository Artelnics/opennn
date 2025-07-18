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

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Forecasting Example." << endl;


        //TimeSeriesDataset time_series_dataset("../data/madridNO2forecasting.csv", ",", true, false);
        //TimeSeriesDataset time_series_dataset("../data/Pendulum.csv", ",", false, false);
        //TimeSeriesDataset time_series_dataset("../data/twopendulum.csv", ";", false, false);
        //TimeSeriesDataset time_series_dataset("../data/funcion_seno.csv", ";", false, false);
        TimeSeriesDataset time_series_dataset("../data/funcion_seno_inputTarget.csv", ",", false, false);

        //time_series_dataset.print();


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

        cout << "======================================" << endl;
        NormalizedSquaredError normalized_squared_error(&forecasting_network, &time_series_dataset);

        const Tensor<type, 1> gradient = normalized_squared_error.calculate_gradient();
        const Tensor<type, 1> numerical_gradient = normalized_squared_error.calculate_numerical_gradient();

        cout << "Gradient" << endl;
        cout << gradient << endl;
        cout << "Numerical Gradient" << endl;
        cout << numerical_gradient << endl;
        cout << "diferencia" << endl;
        cout << gradient.abs() - numerical_gradient.abs() << endl;

        // cout << "Error" << endl;
        // cout << normalized_squared_error.calculate_numerical_error() << endl;

        // TrainingStrategy training_strategy(&forecasting_network, &time_series_dataset);
        // training_strategy.set_loss_index("MeanSquaredError");
        // training_strategy.get_loss_index()->set_regularization_method("NoRegularization");

        // training_strategy.perform_training();


        // cout << "Error" << endl;
        // cout << normalized_squared_error.calculate_numerical_error() << endl;

        // Tensor<type, 2> inputs(1,2);
        // inputs.setValues({{0}, {0.0998334166468282}});
        // cout << "Inputs" << endl;
        // cout << inputs << endl;

        // const Tensor<type, 2> outputs = forecasting_network.calculate_outputs<2,2>(inputs);

        // cout << "Outputs" << endl;
        // cout << outputs << endl;
        //0.198669330795061 - 0.29552020666134 - 0.38941834230865

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
