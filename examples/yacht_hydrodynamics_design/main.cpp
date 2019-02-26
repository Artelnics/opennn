/****************************************************************************************************************/
/*                                                                                                              */ 
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   Y A C H T   R E S I S T A N C E   D E S I G N   A P P L I C A T I O N                                      */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL (Artelnics)                                                          */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */  
/****************************************************************************************************************/

// This is a function regression problem. 

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
        cout << "OpenNN. Yacht Resistance Design Application." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set;

        data_set.set_data_file_name("../data/yachtresistance.dat");

        data_set.load_data();

        // Variables

        Variables* variables_pointer = data_set.get_variables_pointer();

        variables_pointer->set_name(0, "longitudinal_center_buoyancy");
        variables_pointer->set_name(1, "prismatic_coefficient");
        variables_pointer->set_name(2, "length_displacement_ratio");
        variables_pointer->set_name(3, "beam_draught_ratio");
        variables_pointer->set_name(4, "length_beam_ratio");
        variables_pointer->set_name(5, "froude_number");
        variables_pointer->set_name(6, "residuary_resistance");

        const Matrix<string> inputs_information = variables_pointer->get_inputs_information();
        const Matrix<string> targets_information = variables_pointer->get_targets_information();

        // Instances

        Instances* instances_pointer = data_set.get_instances_pointer();

        instances_pointer->split_random_indices();

        const Vector< Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();
        const Vector< Statistics<double> > targets_statistics = data_set.scale_targets_minimum_maximum();

        // Neural network

        const size_t inputs_number = data_set.get_variables().get_inputs_number();
        const size_t hidden_neurons_number = 30;
        const size_t outputs_number = data_set.get_variables().get_targets_number();

        NeuralNetwork neural_network(inputs_number, hidden_neurons_number, outputs_number);

        Inputs* inputs = neural_network.get_inputs_pointer();

        inputs->set_information(inputs_information);

        Outputs* outputs = neural_network.get_outputs_pointer();

        outputs->set_information(targets_information);

        neural_network.construct_scaling_layer();

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

        scaling_layer_pointer->set_statistics(inputs_statistics);

        scaling_layer_pointer->set_scaling_methods(ScalingLayer::NoScaling);

        neural_network.construct_unscaling_layer();

        UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();

        unscaling_layer_pointer->set_statistics(targets_statistics);

        unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::NoUnscaling);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();

        quasi_Newton_method_pointer->set_maximum_epochs_number(1000);

//        quasi_Newton_method_pointer->set_reserve_loss_history(true);

        quasi_Newton_method_pointer->set_display_period(100);

        const TrainingStrategy::Results training_strategy_results = training_strategy.perform_training();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

//        TestingAnalysis::LinearRegressionResults linear_regression_results = testing_analysis.perform_linear_regression_analysis();

        // Save results

        data_set.save("../data/data_set.xml");

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression("../data/expression.txt");

        training_strategy.save("../data/training_strategy.xml");
        training_strategy_results.save("../data/training_strategy_results.dat");

//        linear_regression_results.save("../data/linear_regression_analysis_results.dat");

        return(0);
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return(1);
    }
}  


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques SL
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
