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

using namespace OpenNN;
using namespace std;
using namespace Eigen;

int main()
{
    try
    {
        cout << "OpenNN. Blank application." << endl;

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
