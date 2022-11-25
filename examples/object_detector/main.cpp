//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O B J E C T    D E T E C T O R    A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#ifndef _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#endif

// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <ctime>
#include <exception>
#include <random>
#include <regex>
#include <map>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <list>
#include <vector>

// OpenNN includes

#include "../../opennn/opennn.h"
#include "../../opennn/opennn_strings.h"

using namespace std;
using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Region Based Object Detector Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set;

        data_set.set_data_file_name("../data/neuralLabelerAnnotationFile.xml");

        data_set.read_ground_truth();

//        data_set.scale_input_variables();

//        data_set.print_data();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        cout << "Input variables number: " << input_variables_number << endl;
        cout << "Number of categories: " << target_variables_number << endl;

        data_set.set_training();

        const Tensor<Index, 1> samples_indices = data_set.get_training_samples_indices();

        const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
        const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();

        const Tensor<Index, 1> input_variables_dimensions = data_set.get_input_variables_dimensions();

        Tensor<Index, 1> input_dataset_batch_dimenison(4);

        NeuralNetwork neural_network;
        Tensor<Index, 1> filters_dimensions(4);
        filters_dimensions.setValues({3,3,1,3});

//        NeuralNetwork neural_network(input_variables_dimensions, 0, filters_dimensions, target_variables_number);

        ScalingLayer scaling_layer(input_variables_dimensions);
        neural_network.add_layer(&scaling_layer);

        cout << "Scaling layer inputs dimensions: " << scaling_layer.get_inputs_number() << endl;

        Tensor<Index, 1> scaling_outputs_dimensions = scaling_layer.get_outputs_dimensions();

        cout << "Scaling layer outputs dimensions: " << scaling_outputs_dimensions << endl;

        FlattenLayer flatten_layer(scaling_outputs_dimensions);
        neural_network.add_layer(&flatten_layer);

        cout << "flatten input variables dimensions: " << flatten_layer.get_input_variables_dimensions() << endl;

        cout << "flatten output variables dimensions: " << flatten_layer.get_outputs_dimensions() << endl;

        ProbabilisticLayer probabilistic_layer(input_variables_number, target_variables_number);
        neural_network.add_layer(&probabilistic_layer);

        cout << "Probabilistic layer inputs number: " << probabilistic_layer.get_inputs_number() << endl;

        cout << "Neural network architecture: " << neural_network.get_architecture() << endl;

        system("pause");

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.perform_training();

        // Testing analysis
/*
        Tensor<type, 4> inputs_4d;

        const TestingAnalysis testing_analysis(&neural_network, &data_set);

        Tensor<unsigned char,1> zero = data_set.read_bmp_image("../data/images/zero/0_1.bmp");
        Tensor<unsigned char,1> one = data_set.read_bmp_image("../data/images/one/1_1.bmp");

        vector<type> zero_int(zero.size()); ;
        vector<type> one_int(one.size());

        for(Index i = 0 ; i < zero.size() ; i++ )
        {
            zero_int[i]=(type)zero[i];
            one_int[i]=(type)one[i];
        }

        Tensor<type, 2> inputs(2, zero.size());
        Tensor<type, 2> outputs(2, neural_network.get_outputs_number());

        Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);
        Tensor<Index, 1> outputs_dimensions = get_dimensions(outputs);

        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

        neural_network.calculate_outputs(inputs.data(), inputs_dimensions, outputs.data(), outputs_dimensions);

        cout << "\nInputs:\n" << inputs << endl;

        cout << "\nOutputs:\n" << outputs << endl;

        cout << "\nConfusion matrix:\n" << confusion << endl;
*/
        cout << "Bye!" << endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques SL
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
