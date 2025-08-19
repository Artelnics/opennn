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
#include "../../opennn/adaptive_moment_estimation.h"
#include "../../opennn/quasi_newton_method.h"
#include "../../opennn/stochastic_gradient_descent.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/recurrent_layer.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Forecasting Example." << endl;

        TimeSeriesDataset time_series_dataset("../data/funcion_seno_inputTarget.csv", ",", false, false);
        // TimeSeriesDataset time_series_dataset("../data/madridNO2forecasting_copy.csv", ",", true, false);
        // TimeSeriesDataset time_series_dataset("../data/madridNO2forecasting.csv", ",", true, false);
        // TimeSeriesDataset time_series_dataset("../data/Pendulum.csv", ",", false, false);
        // TimeSeriesDataset time_series_dataset("../data/twopendulum.csv", ";", false, false);


        cout << "dataset leido" << endl;
        // time_series_dataset.split_samples_sequential(type(0.8), type(0.2), type(0));

        time_series_dataset.print();

        //time_series_dataset.scale_data();
        cout << "-----------------------------------" << endl;

        if(time_series_dataset.has_nan())
            time_series_dataset.impute_missing_values_interpolate();

        if(time_series_dataset.has_nan())
            cout << "sigue habiendo nans" << endl;
        else
            cout << "nans arreglados" << endl;

        cout << "-----------------------------------" << endl;

        // const vector<Index> sample_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        // const vector<Index> input_indices = {0};
        // const vector<Index> target_indices =  {0};

        // const vector<Index> samples_indices = time_series_dataset.get_sample_indices("Training");
        // const Index samples_number = time_series_dataset.get_samples_number("Training");
        // const vector<Index> input_variable_indices = time_series_dataset.get_variable_indices("Input");
        // const vector<Index> target_variable_indices = time_series_dataset.get_variable_indices("Target");

        // const Index batch_size = 20;

        // const Index batch_samples_number = min(samples_number, batch_size);
        // const Index batches_number = std::ceil(static_cast<double>(samples_number) / batch_size);

        // vector<vector<Index>> batches(batches_number);

        // Batch batch(batch_samples_number, &time_series_dataset);

        // batches = time_series_dataset.get_batches(samples_indices, batch_samples_number, false);

        // for (int i = 0; i < batches_number; ++i) {
        //     cout << "iteracion: " << i << endl;
        //     for (const auto& element : batches[i])
        //         std::cout << element << " ";  // Imprimir el elemento de cada fila
        //     cout << endl;
        //     batch.fill(batches[i], input_variable_indices, target_variable_indices);
        //     batch.print();
        // }

        // batch.fill(batches[0], input_variable_indices, target_variable_indices);
        // batch.print();
        // cout << "------------------------------------------" << endl;

        ForecastingNetwork forecasting_network({time_series_dataset.get_input_dimensions()},
                                          {4},
                                          {time_series_dataset.get_target_dimensions()});

        // Layer* layer_ptr_scaling = forecasting_network.get_first("Scaling3d");
        // Scaling3d* scaling_layer = dynamic_cast<Scaling3d*>(layer_ptr_scaling);
        // scaling_layer->set_scalers("None");

        Layer* layer_ptr = forecasting_network.get_first("Recurrent");
        Recurrent* recurrent_layer = dynamic_cast<Recurrent*>(layer_ptr);
        recurrent_layer->set_activation_function("HyperbolicTangent");
        // recurrent_layer->set_timesteps(1);

        forecasting_network.print();

        cout << "------------------------------------------" << endl;

        /// Calcular gradiente
        // NormalizedSquaredError normalized_squared_error(&forecasting_network, &time_series_dataset);

        // const Tensor<type, 1> gradient = normalized_squared_error.calculate_gradient();
        // const Tensor<type, 1> numerical_gradient = normalized_squared_error.calculate_numerical_gradient();

        // cout << "Gradient" << endl;
        // cout << gradient << endl;
        // cout << "Numerical Gradient" << endl;
        // cout << numerical_gradient << endl;
        // cout << "diferencia" << endl;
        // cout << gradient.abs() - numerical_gradient.abs() << endl;

        // cout << "------------------------------------------" << endl;

        /// Entrenamiento
        TrainingStrategy training_strategy(&forecasting_network, &time_series_dataset);
        training_strategy.set_loss_index("MeanSquaredError");
        // training_strategy.set_optimization_algorithm("QuasiNewtonMethod");
        // training_strategy.set_optimization_algorithm("StochasticGradientDescent");

        AdaptiveMomentEstimation* adam = static_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_batch_size(1000);
        adam->set_maximum_epochs_number(10000);

        // QuasiNewtonMethod* quasi = static_cast<QuasiNewtonMethod*>(training_strategy.get_optimization_algorithm());
        // quasi->set_loss_goal(0.001);

        // StochasticGradientDescent* stochastic = static_cast<StochasticGradientDescent*>(training_strategy.get_optimization_algorithm());
        // stochastic->set_batch_size(10000);

        training_strategy.train();

        // cout << "Error: " << normalized_squared_error.calculate_numerical_error() << endl;

        /// Testing analysis
        // TestingAnalysis testing_analysis(&forecasting_network, &time_series_dataset);
        // cout << "Goodness of fit analysis: " << endl;
        // testing_analysis.print_goodness_of_fit_analysis();

        /// Pruebas output
        Tensor<type, 3> inputs(1,2,2);
        inputs.setValues({
            {
                {1.76624F, 1.41520F},
                {2.11640F, 1.80730F},
               // {1.2405F,  2.1018F}
            }
        });
        cout << "Inputs: \n" << inputs << endl;
        const Tensor<type, 2> outputs = forecasting_network.calculate_outputs<3,2>(inputs);
        cout << "outputs: " << outputs << endl;

        /// Pruebas output funcion seno
        // const std::vector<std::pair<type, type>> input_views = {
        //     {0.0,          0.0998334166},
        //     {0.0998334166, 0.1986693308},
        //     {0.198669331,  0.295520207},
        //     {0.295520207,  0.389418342},
        //     {0.389418342,  0.479425539},
        //     {0.479425539,  0.564642473},
        //     {0.564642473,  0.644217687},
        //     {0.644217687,  0.717356091},
        //     {0.717356091,  0.78332691},
        //     {0.78332691,   0.841470985},
        //     {0.841470985,  0.89120736}
        // };

        // for (size_t i = 0; i < input_views.size() - 1; ++i)
        // {
        //     const auto& current_pair = input_views[i];
        //     const type input_val_1 = current_pair.first;
        //     const type input_val_2 = current_pair.second;

        //     cout << "\n--- Prueba " << i + 1 << " ---" << endl;
        //     cout << "Inputs: [ " << input_val_1 << ", " << input_val_2 << " ]" << endl;

        //     Tensor<type, 3> inputs(1, 2, 1); //batch, time, input
        //     inputs.setValues({{{input_val_1}, {input_val_2}}});
        //     const Tensor<type, 2> outputs = forecasting_network.calculate_outputs<3,2>(inputs);

        //     cout << "Output: " << outputs << endl;
        //     cout << "Target: " << input_views[i+1].second << endl;
        // }

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
