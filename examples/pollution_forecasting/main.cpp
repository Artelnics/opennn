//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O L L U T I O N   F O R E C A S T I N G   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is a forecasting application.

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
        cout << "OpenNN. Pollution forecasting example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        //  Load data set

        DataSet data_set;

        data_set.set_data_file_name("../../../datasets/pollution.csv");
        data_set.set_separator(',');
        data_set.set_has_columns_names(true);
        size_t lags_number = 2;
        size_t steps_ahead = 1;

        data_set.set_lags_number(lags_number);
        data_set.set_steps_ahead_number(steps_ahead);

        data_set.set_time_index(0);

        data_set.set_missing_values_method("Mean");

        data_set.read_csv();

        cout<<"dta"<< data_set.get_data()<<endl;

        // Autocorrelations

        const Matrix<double> autocorrelations = data_set.calculate_autocorrelations();

        const Matrix<Vector<double>> cross_correlations = data_set.calculate_cross_correlations();

        const Vector<Descriptives> inputs_descriptives = data_set.scale_inputs_minimum_maximum();

        const Vector<Descriptives> targets_descriptives = data_set.scale_targets_minimum_maximum();


        // Neural network

        const size_t inputs_number = data_set.get_input_variables_number();

        const size_t outputs_number = data_set.get_target_variables_number();

        NeuralNetwork neural_network(NeuralNetwork::Approximation, {inputs_number, outputs_number});

        neural_network.print_summary();

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_optimization_method(TrainingStrategy::GRADIENT_DESCENT);

        GradientDescent* quasi_Newton_method_pointer = training_strategy.get_gradient_descent_pointer();

        quasi_Newton_method_pointer->get_learning_rate_algorithm_pointer()->set_learning_rate_method(LearningRateAlgorithm::Fixed);

        quasi_Newton_method_pointer->set_maximum_epochs_number(2);

        quasi_Newton_method_pointer->set_display_period(1);

        const OptimizationAlgorithm::Results training_strategy_results = training_strategy.perform_training();

        training_strategy.print();

        // Testing analysis

        data_set.unscale_inputs_minimum_maximum(inputs_descriptives);

        data_set.unscale_targets_minimum_maximum(targets_descriptives);

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        const TestingAnalysis::LinearRegressionAnalysis linear_regression_results = testing_analysis.perform_linear_regression_analysis()[0];

        // Save results

        data_set.save("../data/data_set.xml");

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression("../data/expression.txt");

        training_strategy.save("../data/training_strategy.xml");
        training_strategy_results.save("../data/training_strategy_results.dat");

        linear_regression_results.save("../data/linear_regression_analysis_results.dat");

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
