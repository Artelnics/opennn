//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S I M P L E   P A T T E R N   R E C O G N I T I O N   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is a pattern recognition problem. 

// System includes

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdexcept>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
    try
    {
        cout << "OpenNN. Simple classification example." << endl;

        // Device
        
        const int n = omp_get_max_threads();
        NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
        ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

        // Data set

        DataSet data_set("../data/simple_pattern_recognition.csv", ';', true);
        data_set.set_thread_pool_device(thread_pool_device);

        // Variables

        data_set.set_training();

        const Tensor<string, 1> inputs_names = data_set.get_input_variables_names();
        const Tensor<string, 1> targets_names = data_set.get_target_variables_names();

        const Tensor<Descriptives, 1> inputs_descriptives = data_set.scale_inputs_minimum_maximum();

        // Neural network

        Tensor<Index, 1> neural_network_architecture(3);
        neural_network_architecture.setValues({2, 2, 1});

        NeuralNetwork neural_network(NeuralNetwork::Classification, neural_network_architecture);
        neural_network.set_thread_pool_device(thread_pool_device);

        neural_network.set_inputs_names(inputs_names);

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

        scaling_layer_pointer->set_descriptives(inputs_descriptives);

        neural_network.set_outputs_names(targets_names);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

//        QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();

        const OptimizationAlgorithm::Results training_strategy_results = training_strategy.perform_training();

        // Testing analysis

        data_set.set_testing();

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        Tensor<type, 1> binary_classification_tests = testing_analysis.calculate_binary_classification_tests();

        Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

        // Save results

        data_set.save("../data/data_set.xml");

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression("../data/expression.txt");

        training_strategy.save("../data/training_strategy.xml");
//        training_strategy_results.save("../data/training_strategy_results.dat");

//        binary_classification_tests.save("../data/binary_classification_tests.dat");
//        confusion.save("../data/confusion.dat");

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
