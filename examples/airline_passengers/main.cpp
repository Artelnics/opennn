//   OpenNN: Open Neural Networks Library                                                                       
//   www.opennn.net                                                                                   
//                                                                                                              
//   A I R L I N E   P A S S E N G E R S   A P P L I C A T I O N                                                
//                                                                                                              
//   Artificial Intelligence Techniques SL                                                                      
//   artelnics@artelnics.com                                                                                    

// This is a forecasting problem.

// System includes

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
    try
    {
        cout << "OdsfadfenNN. Airline Passengers Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Device

        Device device(Device::EigenSimpleThreadPool);

        // Data set

        DataSet data_set("../data/airline_passengers_simple.csv", ',', true);
        data_set.set_device_pointer(&device);

        data_set.print_columns_types();

        data_set.set_lags_number(3);
        data_set.set_steps_ahead_number(1);

        data_set.transform_time_series();

//        data_set.set_batch_instances_number(1);

        const Tensor<string, 1> inputs_names = data_set.get_input_variables_names();
        const Tensor<string, 1> targets_names = data_set.get_target_variables_names();

        // Instances

        data_set.split_instances_sequential();

        const Tensor<Descriptives, 1> inputs_descriptives = data_set.scale_inputs_minimum_maximum();
        const Tensor<Descriptives, 1> targets_descriptives = data_set.scale_targets_minimum_maximum();

        // Neural network

        const Index inputs_number = data_set.get_input_variables_number();
        const Index hidden_neurons_number = 4;
        const Index outputs_number = data_set.get_target_variables_number();

        Tensor<Index, 1> neural_network_architecture(3);
        neural_network_architecture.setValues({inputs_number, hidden_neurons_number, outputs_number});

        NeuralNetwork neural_network(NeuralNetwork::Forecasting, neural_network_architecture);
        neural_network.set_device_pointer(&device);

        neural_network.set_inputs_names(inputs_names);
        neural_network.set_outputs_names(targets_names);

        neural_network.print_summary();

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
        scaling_layer_pointer->set_descriptives(inputs_descriptives);
        scaling_layer_pointer->set_scaling_methods(ScalingLayer::NoScaling);

        LongShortTermMemoryLayer* long_short_term_memory_layer_pointer = neural_network.get_long_short_term_memory_layer_pointer();
        long_short_term_memory_layer_pointer->set_timesteps(1);

        UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();
        unscaling_layer_pointer->set_descriptives(targets_descriptives);
        unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::NoUnscaling);

        // Training strategy object

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method("MEAN_SQUARED_ERROR");
        training_strategy.get_loss_index_pointer()->set_regularization_method("NO_REGULARIZATION");

        training_strategy.set_optimization_method("ADAPTIVE_MOMENT_ESTIMATION");

        AdaptiveMomentEstimation* adaptative_moment_estimation_pointer = training_strategy.get_adaptive_moment_estimation_pointer();
        adaptative_moment_estimation_pointer->set_maximum_epochs_number(1000);
        adaptative_moment_estimation_pointer->set_display_period(10);

        const OptimizationAlgorithm::Results training_strategy_results = training_strategy.perform_training();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        TestingAnalysis::LinearRegressionAnalysis linear_regression_results = testing_analysis.perform_linear_regression_analysis()[0];

        // Save results

        data_set.save("../data/data_set.xml");

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression("../data/expression.txt");

        training_strategy.save("../data/training_strategy.xml");
        training_strategy_results.save("../data/training_strategy_results.dat");

        linear_regression_results.save("../data/linear_regression_analysis_results.dat");


        cout << "Bye" << endl;

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
