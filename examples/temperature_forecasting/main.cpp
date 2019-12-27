//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E M P E R A T U R E   F O R E C A S T I N G   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques, S.L. (Artelnics)
//   artelnics@artelnics.com

// System includes

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
    try
    {
        cout << "OpenNN. Temperature Forecasting Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        //Data Set

        DataSet data_set("../data/temperature.csv", ',', true);

        Vector<Descriptives> columns_statistics = data_set.calculate_columns_descriptives();

        Vector<Histogram> columns_histograms = data_set.calculate_columns_histograms();

        cout << "Converting to time series" << endl;

        data_set.set_lags_number(12);
        data_set.set_steps_ahead_number(1);
        data_set.transform_time_series();

        cout << "Correlations" << endl;

        // Missing values

        data_set.impute_missing_values_mean();

        // Instances

        data_set.split_instances_sequential();

        const Vector<Descriptives> inputs_descriptives = data_set.scale_inputs_minimum_maximum();
        const Vector<Descriptives> targets_descriptives = data_set.scale_targets_minimum_maximum();

        // Neural network

        cout << "Neural network" << endl;

        const size_t inputs_number = data_set.get_input_variables_number();
        const size_t hidden_perceptrons_number = 6;
        const size_t outputs_number = data_set.get_target_variables_number();

        NeuralNetwork neural_network(NeuralNetwork::Forecasting, {inputs_number, hidden_perceptrons_number, outputs_number});

//        neural_network.set(NeuralNetwork::Forecasting, {inputs_number, hidden_perceptrons_number, outputs_number});

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
        scaling_layer_pointer->set_descriptives(inputs_descriptives);
        scaling_layer_pointer->set_scaling_methods(ScalingLayer::NoScaling);


        UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();
        unscaling_layer_pointer->set_descriptives(targets_descriptives);
        unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::NoUnscaling);

        // Training strategy object

        TrainingStrategy training_strategy;
        QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();
        quasi_Newton_method_pointer->set_maximum_epochs_number(10000);
        quasi_Newton_method_pointer->set_maximum_time(250);
        quasi_Newton_method_pointer->set_display_period(10);
        quasi_Newton_method_pointer->set_minimum_loss_decrease(0.0);
        quasi_Newton_method_pointer->set_reserve_training_error_history(true);
        quasi_Newton_method_pointer->set_reserve_selection_error_history(true);

        const OptimizationAlgorithm::Results training_strategy_results = training_strategy.perform_training();

        // Testing analysis

        cout << "Testing" << endl;

        TestingAnalysis testing_analysis(&neural_network, &data_set);
        TestingAnalysis::LinearRegressionAnalysis linear_regression_analysis = testing_analysis.perform_linear_regression_analysis()[0];
        Vector< Vector<double> > error_autocorrelation = testing_analysis.calculate_error_autocorrelation();
        Vector< Vector<double> > error_crosscorrelation = testing_analysis.calculate_inputs_errors_cross_correlation();
        Vector< Matrix<double> > error_data = testing_analysis.calculate_error_data();
        Vector< Vector<Descriptives> > error_data_statistics = testing_analysis.calculate_error_data_statistics();

        // Save results

        data_set.save("../data/data_set.xml");

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression("../data/expression.txt");

        training_strategy.save("../data/training_strategy.xml");

        training_strategy.save("../data/training_strategy.xml");
        training_strategy_results.save("../data/training_strategy_results.dat");

        linear_regression_analysis.save("../data/linear_regression_analysis.dat");
        error_autocorrelation.save("../data/error_autocorrelation.dat");
        error_crosscorrelation.save("../data/error_crosscorrelation.dat");
        error_data.save("../data/error_data.dat");
//        error_data_statistics.save("../data/error_data_statistics.dat");

        // Deployment

        scaling_layer_pointer->set_scaling_methods(ScalingLayer::MinimumMaximum);
        unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::MinimumMaximum);

        scaling_layer_pointer->set_scaling_methods(ScalingLayer::MinimumMaximum);
        unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::MinimumMaximum);

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques SL
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
