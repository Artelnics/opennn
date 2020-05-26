//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B R E A S T   C A N C E R   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is a pattern recognition problem.

// System includes

#include <iostream>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
    try
    {
        cout << "OpenNN. Breast Cancer Application." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Device

        const int n = omp_get_max_threads();
        NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
        ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

        // Data set

        DataSet data_set("../data/breast_cancer.csv",';',true);
        data_set.set_thread_pool_device(thread_pool_device);

        data_set.split_instances_random();

        const Tensor<string, 1> inputs_names = data_set.get_input_variables_names();
        const Tensor<string, 1> targets_names = data_set.get_target_variables_names();

        const Tensor<Descriptives, 1> inputs_descriptives = data_set.scale_inputs_minimum_maximum();

        // Neural network

        Tensor<Index, 1> neural_netowrk_architecture(3);
        neural_netowrk_architecture.setValues({9, 3, 1});

        NeuralNetwork neural_network(NeuralNetwork::Approximation, neural_netowrk_architecture);
        neural_network.set_thread_pool_device(thread_pool_device);

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

        scaling_layer_pointer->set_descriptives(inputs_descriptives);

        scaling_layer_pointer->set_scaling_methods(ScalingLayer::MinimumMaximum);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);
        training_strategy.set_thread_pool_device(thread_pool_device);

        training_strategy.set_loss_method(TrainingStrategy::WEIGHTED_SQUARED_ERROR);

        training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::L2);
        training_strategy.get_loss_index_pointer()->set_regularization_weight(0.001);

        QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();

        quasi_Newton_method_pointer->set_loss_goal(1.0e-3);

        quasi_Newton_method_pointer->set_display(true);

        training_strategy.set_display(true);

        training_strategy.perform_training();

        // Model selection

        ModelSelection model_selection(&training_strategy);

        model_selection.perform_neurons_selection();

        // Testing analysis

        data_set.unscale_inputs_minimum_maximum(inputs_descriptives);

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

        cout << "Confusion: " << endl;
        cout << confusion << endl;

        Tensor<type, 1> binary_classification_tests = testing_analysis.calculate_binary_classification_tests();

        cout << "Binary classification tests: " << endl;
        cout << "Classification accuracy         : " << binary_classification_tests[0] << endl;
        cout << "Error rate                      : " << binary_classification_tests[1] << endl;
        cout << "Sensitivity                     : " << binary_classification_tests[2] << endl;
        cout << "Specificity                     : " << binary_classification_tests[3] << endl;
        cout << "Precision                       : " << binary_classification_tests[4] << endl;
        cout << "Positive likelihood             : " << binary_classification_tests[5] << endl;
        cout << "Negative likelihood             : " << binary_classification_tests[6] << endl;
        cout << "F1 score                        : " << binary_classification_tests[7] << endl;
        cout << "False positive rate             : " << binary_classification_tests[8] << endl;
        cout << "False discovery rate            : " << binary_classification_tests[9] << endl;
        cout << "False negative rate             : " << binary_classification_tests[10] << endl;
        cout << "Negative predictive value       : " << binary_classification_tests[11] << endl;
        cout << "Matthews correlation coefficient: " << binary_classification_tests[12] << endl;
        cout << "Informedness                    : " << binary_classification_tests[13] << endl;
        cout << "Markedness                      : " << binary_classification_tests[14] << endl;

        cout << "End" << endl;

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
