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

int main(int argc, char *argv[])
{
    try
    {
        cout << "OpenNN. Region Based Object Detector Example." << endl;


        srand(time(NULL));

        DataSet data_set;

        data_set.set_data_file_name("Z:/Images/DatasetRedDots-bmp/ground_truth.xml");

        data_set.read_ground_truth();

        data_set.set_training();

        const Index target_variables_number = data_set.get_target_variables_number();
        const Index input_variables_number = data_set.get_input_variables_number();

        cout << "input_variables_number: "  << input_variables_number<< endl;

        Tensor<Index, 1> input_variables_dimensions = data_set.get_input_variables_dimensions();

        cout << "input_variables_dimensions: " << input_variables_dimensions << endl;

        NeuralNetwork neural_network;
/*
        RegionProposalLayer region_proposal_layer;
        neural_network.add_layer(&region_proposal_layer);
*/
        FlattenLayer flatten_layer(input_variables_dimensions);
        neural_network.add_layer(&flatten_layer);

        const Index flatten_output_numbers = flatten_layer.get_outputs_number();

        cout << "flatten_output_numbers: " << flatten_output_numbers << endl;

        ProbabilisticLayer probabilistic_layer(input_variables_number, target_variables_number);
        neural_network.add_layer(&probabilistic_layer);


/*
        NonMaxSupressionLayer non_max_supression_layer;
        neural_network.add_layer(&non_max_supression_layer);
*/
        Tensor<type, 4> inputs(6,6,3,1);
        inputs.setRandom();

        Tensor<Index, 1> inputs_dimensions(4);
        inputs_dimensions(0) = 6;
        inputs_dimensions(1) = 6;
        inputs_dimensions(2) = 3;
        inputs_dimensions(3) = 1;

//        Tensor<type, 2> outputs(1, 108);

//        Tensor<Index, 1> outputs_dimensions(2);
//        outputs_dimensions(0) = 1;
//        outputs_dimensions(1) = 108;

//        flatten_layer.calculate_outputs(inputs.data(), inputs_dimensions, outputs.data(), outputs_dimensions);

//        Tensor<type, 2> outputs = neural_network.calculate_outputs(inputs.data(), inputs_dimensions);

//        cout << "inputs: " << endl;
//        cout << inputs << endl;
//        cout << endl << endl << endl << endl << endl;
//        cout << "outputs: " << endl;
//        cout << outputs << endl;
//        cout << "outputs_dimension: " << endl;
//        cout << outputs.dimensions() << endl;


        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);


        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

        training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_adaptive_moment_estimation_pointer()->set_batch_samples_number(128);
        training_strategy.get_adaptive_moment_estimation_pointer()->set_maximum_epochs_number(10000);
        training_strategy.perform_training();

//        training_strategy.get_mean_squared_error_pointer()->calculate_regularization();


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
