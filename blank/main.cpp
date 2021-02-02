//   OpenNN: Open Neural Networks Library
//   www.opennn.org
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <limits.h>
#include <statistics.h>
#include <regex>

// Systems Complementaries

#include <cmath>
#include <cstdlib>
#include <ostream>


// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace chrono;



int main(void)
{
    try
    {
        cout << "OpenNN. Blank Application." << endl;

        // Data set

        DataSet ds("C:/Users/Usuario/Documents/AirPassengers.csv", ';', true);
//        ds.set_min_max_range(0,1);

        ds.set_lags_number(1);
        ds.set_steps_ahead_number(1);

        ds.transform_time_series();

        Tensor<DataSet::Column, 1> columns = ds.get_columns();

        for(Index i = 0; i < columns.dimension(0); i++)
        {
            cout << "Column " << i << ": " << columns(i).name;

            if(columns(i).column_use == DataSet::Input) cout << " input" << endl;
            if(columns(i).column_use == DataSet::Target) cout << " target" << endl;
        }

        const Index input_variables_number = ds.get_input_variables_number();
        const Index target_variables_number = ds.get_target_variables_number();

        Tensor<string, 1> input_scaling_methods(input_variables_number);
        input_scaling_methods.setConstant("MinimumMaximum");

        Tensor<string, 1> target_scaling_methods(target_variables_number);
        target_scaling_methods.setConstant("MinimumMaximum");

        const Tensor<Descriptives, 1> input_descriptives = ds.scale_input_variables(input_scaling_methods);
        const Tensor<Descriptives, 1> target_descriptives = ds.scale_target_variables(target_scaling_methods);

        // Neural network

        Tensor<Index, 1> architecture(3);
        architecture(0) = input_variables_number;
        architecture(1) = 5;
        architecture(2) = target_variables_number;

        NeuralNetwork nn;

        ScalingLayer sl(input_variables_number);
        sl.set_descriptives(input_descriptives);
        sl.set_scaling_methods(ScalingLayer::ScalingMethod::MinimumMaximum);

        RecurrentLayer rl(input_variables_number, 5);

        PerceptronLayer pl(5, target_variables_number);

        UnscalingLayer ul(input_variables_number);
        ul.set_descriptives(input_descriptives);
        ul.set_unscaling_methods(UnscalingLayer::UnscalingMethod::MinimumMaximum);

        nn.add_layer(&sl);
        nn.add_layer(&rl);
        nn.add_layer(&pl);
        nn.add_layer(&ul);

        cout << "nn created" << endl;

        Tensor<Layer*, 1> layers = nn.get_layers_pointers();

        for(Index i = 0; i < layers.dimension(0); i++)
        {
            cout << "Layer " << i << ": " << layers(i)->get_type_string() << endl;
            cout << "   Inputs: " << layers(i)->get_inputs_number() << "; neurons: " << layers(i)->get_neurons_number() << endl;
        }


/*
        NeuralNetwork nn(NeuralNetwork::ProjectType::Forecasting, architecture);

        Tensor<Layer*, 1> layers = nn.get_layers_pointers();

        for(Index i = 0; i < layers.dimension(0); i++)
        {
            cout << "Layer " << i << ": " << layers(i)->get_type_string() << endl;
            cout << "   Inputs: " << layers(i)->get_inputs_number() << "; neurons: " << layers(i)->get_neurons_number() << endl;
        }

        ScalingLayer* sl = nn.get_scaling_layer_pointer();
        sl->set_descriptives(input_descriptives);
        sl->set_scaling_methods(ScalingLayer::ScalingMethod::MinimumMaximum);
//        sl->set_min_max_range(0,1);

        LongShortTermMemoryLayer* lstm = nn.get_long_short_term_memory_layer_pointer();
        lstm->set_timesteps(20);

        UnscalingLayer* ul = nn.get_unscaling_layer_pointer();
        ul->set_descriptives(target_descriptives);
        ul->set_unscaling_methods(UnscalingLayer::UnscalingMethod::MinimumMaximum);
//        ul->set_min_max_range(0,1);
*/
        // Training strategy

        TrainingStrategy training_strategy(&nn, &ds);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
//        training_strategy.set_maximum_epochs_number(10);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);

//        training_strategy.perform_training();

        // Outputs

//        ds.unscale_input_variables(input_scaling_methods, input_descriptives);
//        ds.unscale_target_variables(target_scaling_methods, target_descriptives);

//        const Tensor<type, 2> testing_input_data = ds.get_testing_input_data();
//        const Tensor<type, 2> testing_target_data = ds.get_testing_input_data();

//        const Tensor<type, 2> testing_outputs = nn.calculate_outputs(testing_input_data);


//        for(Index i = 0; i < testing_input_data.dimension(0); i++)
//        {
//            cout << testing_outputs(i,0) << endl;
//        }

        cout << "End." << endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
