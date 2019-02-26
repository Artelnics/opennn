/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   L E U K E M I A   A P P L I C A T I O N                                                                    */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// This is a classical pattern recognition problem.

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
//    try
//    {
//        std::cout << "OpenNN. Leukemia Application." << std::endl;

//        srand((unsigned)time(NULL));

//        // Data set

//        DataSet data_set;

//        data_set.set_data_file_name("../data/leukemia.dat");

//        data_set.set_separator("Space");

//        data_set.load_data();

//        data_set.balance_binary_targets_distribution();

//        // Variables

//        Variables* variables_pointer = data_set.get_variables_pointer();

//        const Matrix<std::string> inputs_information = variables_pointer->get_inputs_information();
//        const Matrix<std::string> targets_information = variables_pointer->get_targets_information();

//        const size_t inputs_number = variables_pointer->get_inputs_number();
//        const size_t targets_number = variables_pointer->get_targets_number();

//        // Instances

//        Instances* instances_pointer = data_set.get_instances_pointer();

//        instances_pointer->split_random_indices();

//        const Vector< Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();

//        // Neural network

//        NeuralNetwork neural_network(inputs_number, targets_number);

//        neural_network.get_multilayer_perceptron_pointer()->set_layer_activation_function(0,Perceptron::Logistic);

//        Inputs* inputs_pointer = neural_network.get_inputs_pointer();

//        inputs_pointer->set_information(inputs_information);

//        Outputs* outputs_pointer = neural_network.get_outputs_pointer();

//        outputs_pointer->set_information(targets_information);

//        neural_network.construct_scaling_layer();

//        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

//        scaling_layer_pointer->set_statistics(inputs_statistics);

//        scaling_layer_pointer->set_scaling_methods(ScalingLayer::NoScaling);

//        neural_network.construct_probabilistic_layer();

//        ProbabilisticLayer* probabilistic_layer_pointer = neural_network.get_probabilistic_layer_pointer();

//        probabilistic_layer_pointer->set_probabilistic_method(ProbabilisticLayer::Probability);

//        // Loss index

//        LossIndex loss_index(&neural_network, &data_set);

//        loss_index.set_error_type(LossIndex::WEIGHTED_SQUARED_ERROR);

//        WeightedSquaredError* weighted_squared_error_pointer = loss_index.get_weighted_squared_error_pointer();

//        weighted_squared_error_pointer->set_weights();

//        weighted_squared_error_pointer->set_normalization_coefficient();

//        // Training strategy

//        TrainingStrategy training_strategy(&loss_index);

//        training_strategy.set_main_type(TrainingStrategy::QUASI_NEWTON_METHOD);

//        QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();

//        quasi_Newton_method_pointer->set_minimum_loss_decrease(1.0e-6);

//        training_strategy.set_display(false);

//        // Model selection

//        ModelSelection model_selection(&training_strategy);

//        model_selection.set_inputs_selection_type(ModelSelection::GROWING_INPUTS);

//        GrowingInputs* growing_inputs_pointer = model_selection.get_growing_inputs_pointer();

//        growing_inputs_pointer->set_approximation(false);

//        growing_inputs_pointer->set_maximum_selection_failures(3);

//        ModelSelection::ModelSelectionResults model_selection_results = model_selection.perform_inputs_selection();

//        instances_pointer->set_training();

//        training_strategy.perform_training();

//        instances_pointer->set_testing();

//        // Testing analysis

//        TestingAnalysis testing_analysis(&neural_network, &data_set);

//        const Matrix<size_t> confusion = testing_analysis.calculate_confusion();

//        // Save results

//        data_set.save("../data/data_set.xml");

//        neural_network.save("../data/neural_network.xml");

//        training_strategy.save("../data/training_strategy.xml");

//        model_selection.save("../data/model_selection.xml");
////        model_selection_results.save("../data/model_selection_results.dat");

//        confusion.save("../data/confusion.dat");

//        return(0);
//    }
//    catch(std::exception& e)
//    {
//        std::cout << e.what() << std::endl;

//        return(1);
//    }
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
