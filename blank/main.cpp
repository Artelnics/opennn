//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "../opennn/dataset.h"
#include "../opennn/standard_networks.h"
#include "../opennn/mean_squared_error.h"
#include "../opennn/adaptive_moment_estimation.h"

using namespace opennn;

int main()
{
    try
    {
        // Dataset
        // 1000 random samples: 8 input features, 1 regression target

        Dataset dataset(1000, {8}, {1});

        dataset.set_data_random();
        dataset.split_samples_random(0.8, 0.1, 0.1);

        const Index inputs_number = dataset.get_features_number("Input");
        const Index targets_number = dataset.get_features_number("Target");

        cout << "Samples: " << dataset.get_samples_number() << endl;
        cout << "Inputs:  " << inputs_number << endl;
        cout << "Targets: " << targets_number << endl;

        // Neural network
        // One hidden layer with 32 neurons, linear output (regression)

        ApproximationNetwork network({inputs_number}, {32}, {targets_number});

        network.set_parameters_glorot();

        cout << "Parameters: " << network.get_parameters_number() << endl;

        // Loss function

        MeanSquaredError loss(&network, &dataset);

        // Adam optimizer

        AdaptiveMomentEstimation optimizer(&loss);

        optimizer.set_learning_rate(type(0.001));
        optimizer.set_beta_1(type(0.9));
        optimizer.set_beta_2(type(0.999));
        optimizer.set_batch_size(64);
        optimizer.set_maximum_epochs(200);
        optimizer.set_loss_goal(type(1e-4));
        optimizer.set_display(true);

        // Train

        const TrainingResults results = optimizer.train();

        // Results

        cout << "\nStopping condition: " << results.write_stopping_condition() << endl;
        cout << "Training loss:      " << results.get_training_error() << endl;
        cout << "Validation loss:    " << results.get_validation_error() << endl;
        cout << "Epochs:             " << results.get_epochs_number() << endl;
        cout << "Elapsed time:       " << results.elapsed_time << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
