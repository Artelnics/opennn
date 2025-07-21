//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A M A Z O N   R E V I E W S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

#include "../opennn/language_dataset.h"
#include "../opennn/embedding_layer.h"
#include "../opennn/multihead_attention_layer.h"
#include "../opennn/probabilistic_layer_3d.h"
#include "../opennn/training_strategy.h"
#include "../opennn/perceptron_layer.h"
#include "../opennn/recurrent_layer.h"
#include "../opennn/testing_analysis.h"
#include "../opennn/time_series_dataset.h"
#include "scaling_layer_2d.h"
#include "unscaling_layer.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "Recurrent layer OpenNN" << endl;

        TimeSeriesDataset time_series_data_set("C:/Users/Artelnics/Downloads/load_transformer_measures.csv", ",", true, false);

        auto dataset = time_series_data_set.get_data();
        const Index rows_number = dataset.dimension(0);
        Tensor<type,2> data(rows_number, 1);
        for(Index i = 0; i < rows_number; ++i)
        {
            data(i, 0) = dataset(i, 1);
        }

        time_series_data_set.set_lags_number(2);
        time_series_data_set.set_steps_ahead_number(1);

        const Index N = data.dimension(0);
        Index lags_number=time_series_data_set.get_lags_number();
        Index steps_ahead=time_series_data_set.get_steps_ahead();
        const Index new_samples = N - lags_number - steps_ahead + 1;
        Tensor<type,2> new_data(new_samples, lags_number+steps_ahead);

        for(Index i = 1; i < new_samples; ++i)
        {
            for(Index j = 0; j < lags_number; ++j)
            {
                new_data(i, j) = data(i + j, 0);
            }
            for(Index j = 0; j < steps_ahead; ++j)
            {
                new_data(i, lags_number + j) = data(i + lags_number + j, 0);
            }
        }
        time_series_data_set.set(new_samples, {lags_number,1}, {steps_ahead,1});
        time_series_data_set.set_data(new_data);

        time_series_data_set.print();
        //time_series_data_set.print_data();

        const Index input_variables_number = time_series_data_set.get_variables_number("Input");
        const Index target_variables_number = time_series_data_set.get_variables_number("Target");
        const vector<string>& variable_names = time_series_data_set.get_variable_names();
        const Index time_steps=time_series_data_set.get_lags_number();

        NeuralNetwork neural_network;

        neural_network.add_layer(make_unique<Scaling2d>(dimensions{ input_variables_number*time_steps }));

        neural_network.add_layer(make_unique<Recurrent>(dimensions{ input_variables_number,time_steps  },
                                                        dimensions{ target_variables_number  }));

        neural_network.add_layer(make_unique<Dense2d>(dimensions{ input_variables_number  },
                                                      dimensions{ target_variables_number  }));

        neural_network.add_layer(make_unique<Unscaling>(dimensions{ input_variables_number*time_steps} ));

        TrainingStrategy training_strategy(&neural_network, &time_series_data_set);

        training_strategy.perform_training();

        TestingAnalysis testing_analysis(&neural_network, &time_series_data_set);
        TestingAnalysis::GoodnessOfFitAnalysis perform_goodness_of_fit_analysis{};
        testing_analysis.print_goodness_of_fit_analysis();

        TestingAnalysis::RocAnalysis roc_curve = testing_analysis.perform_roc_analysis();
        roc_curve.print();

        cout << "Completed." << endl;

        return 0;
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
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
