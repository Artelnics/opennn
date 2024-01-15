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
/*
        // UNDER DEVELOPMENT

        srand(time(NULL));

        DataSet data_set;

        //        data_set.set_intersection_over_union_threshold(0.3);
        //        data_set.set_regions_number(100);
        //        data_set.set_region_rows(12);
        //        data_set.set_region_columns(12);

        data_set.set_data_file_name("../img/ground_truth.xml");


        data_set.read_ground_truth();

        DataSetBatch training_batch(1, &data_set);

        Tensor<Index, 2> training_batches = data_set.get_batches(training_samples_indices, training_batch_samples_number, shuffle);

        training_batch.fill(training_batches.chip(0, 0), input_variables_indices, target_variables_indices);

        Tensor<Index, 1> inputs_dimensions(3);
        inputs_dimensions(0) = 28;
        inputs_dimensions(1) = 28;
        inputs_dimensions(2) = 3;

        RegionProposalLayer region_proposal_layer(inputs_dimensions);
        region_proposal_layer.set_regions_number(2);
        region_proposal_layer.set_region_rows(1);
        region_proposal_layer.set_region_columns(1);
        region_proposal_layer.set_channels_number(1);

        const Tensor<Index, 1> region_proposal_layer_outputs_dimensions = region_proposal_layer.get_outputs_dimensions();

        cout << region_proposal_layer_outputs_dimensions << endl;

        FlattenLayer flatten_layer(region_proposal_layer_outputs_dimensions);

        const Tensor<Index, 1> flatten_layer_outputs_dimensions = flatten_layer.get_outputs_dimensions();

        cout << flatten_layer_outputs_dimensions << endl;

        ProbabilisticLayer probabilistic_layer(1, 1);

        NonMaxSuppressionLayer non_max_suppression_layer;

        NeuralNetwork neural_network;

        neural_network.add_layer(&region_proposal_layer); // 0
        neural_network.add_layer(&flatten_layer); // 1
        neural_network.add_layer(&probabilistic_layer); // 2
        neural_network.add_layer(&non_max_suppression_layer); // 3

        Tensor<Index, 1> non_max_suppression_layer_inputs_indices(2);
        non_max_suppression_layer_inputs_indices(0) = type(0);
        non_max_suppression_layer_inputs_indices(1) = 2;

        neural_network.set_layer_inputs_indices(3, non_max_suppression_layer_inputs_indices);

        neural_network.print_layers_inputs_indices();

        NeuralNetworkForwardPropagation neural_network_forward_propagation(1, &neural_network);
        neural_network_forward_propagation.print();

        neural_network.forward_propagate( neural_network_forward_propagation, false);

        const string filename = "Z:/Images/DatasetRedDots-bmp/9.bmp";
        const Tensor<Tensor<type, 1>, 1> input_image = read_bmp_image(filename);

        Tensor<type, 2> image(1, input_image(0).dimensions()[0] + input_image(1).dimensions()[0]);

        Index pixel_valule_index = 0;
        Index dimensions_index = 0;

        for(Index i = 0; i < image.dimension(1); i++)
        {
            if(i < input_image(0).dimensions()[0])
            {
                image(0,i) = input_image(0)(pixel_valule_index);
                pixel_valule_index++;
            }
            else
            {
                image(0,i) = input_image(1)(dimensions_index);
                dimensions_index++;
            }
        }

        Tensor<Index, 1> inputs_dimensions = get_dimensions(image);

        const Index regions_number = 2000;
        const Index columns_number = 22;
        const Index rows_number = 22;
        const Index channels_number = input_image(1)(2);

        Tensor<type,2> output(regions_number, rows_number * columns_number * channels_number);
        Tensor<Index, 1> output_dimensions = get_dimensions(output);

        region_proposal_layer.calculate_outputs(image.data(), inputs_dimensions, output.data(), output_dimensions);

        cout << "output: " << output << endl;

        FlattenLayer flatten_layer(input_variables_dimensions);
        neural_network.add_layer(&flatten_layer);

        const Index flatten_output_numbers = flatten_layer.get_outputs_number();

        cout << "flatten_output_numbers: " << flatten_output_numbers << endl;

        ProbabilisticLayer probabilistic_layer(input_variables_number, target_variables_number);
        neural_network.add_layer(&probabilistic_layer);

        NonMaxSuppressionLayer non_max_supression_layer;
        neural_network.add_layer(&non_max_supression_layer);

        Tensor<Index, 1> inputs_dimensions(4);
        inputs_dimensions(0) = 6;
        inputs_dimensions(1) = 6;
        inputs_dimensions(2) = 3;
        inputs_dimensions(3) = 1;

//        Tensor<type, 2> outputs(1, 108);

//        Tensor<Index, 1> outputs_dimensions(2);
//        outputs_dimensions[0] = 1;
//        outputs_dimensions(1) = 108;

//        flatten_layer.calculate_outputs(inputs.data(), inputs_dimensions, outputs.data(), outputs_dimensions);

//        Tensor<type, 2> outputs = neural_network.calculate_outputs(inputs);

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
// Copyright (C) 2005-2024 Artificial Intelligence Techniques SL
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
