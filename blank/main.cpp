//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../opennn/opennn.h"
#include "../opennn/text_analytics.h"

using namespace opennn;
using namespace std;
using namespace Eigen;

#include "data_set.h"

int main()
{
    try
    {
        cout << "Blank script! " << endl;


        stof("NA");

        getchar();
        DataSet data_set;

        data_set.set_data_file_name("/home/artelnics2020/Escritorio/datasets/amazon_cells_labelled.txt");
//        data_set.set_data_file_name("/home/artelnics2020/Escritorio/datasets/PRIMERASEGUNDA.txt");

        data_set.read_txt();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        const Tensor<type,2> data = data_set.get_data();
        const Tensor<string,1> columns_names = data_set.get_columns_names();

        std::ofstream file("/home/artelnics2020/Escritorio/datasets/text_data.csv");

        for(Index i = 0; i < columns_names.size() - 1 ; i++)
        {
            file << columns_names(i) << ",";
        }

        file << "TARGET" << endl;

        for (Index i = 0; i < data.dimension(0); i++)
        {
            for(Index j = 0; j < data.dimension(1) - 1; j++)
            {
                file << data(i,j) << ",";
            }

            file << data(i,data.dimension(1) - 1) << endl;
        }

        // Neural Network

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, {input_variables_number, 10, target_variables_number});

        neural_network.get_scaling_layer_pointer()->set_display(false);

        ////

        string prediction = "I don't like this product, it doesn't work propperly";

        string prediction_2 = "I love this product! I will recommend it to all my friends, can i?";

        const Tensor<type,1> prediction_vector = data_set.sentence_to_data(prediction);
        const Tensor<type,1> prediction_vector_2 = data_set.sentence_to_data(prediction_2);

        Tensor<type, 2> inputs(2,prediction_vector.dimension(0));

        for(Index i = 0; i < prediction_vector.dimension(0); i++)
        {
            inputs(0,i) = prediction_vector(i);
            inputs(1,i) = prediction_vector_2(i);
        }

        // Training Strategy

        TrainingStrategy training_strategy(&neural_network,&data_set);

//        training_strategy.set_loss_method(TrainingStrategy::LossMethod::WEIGHTED_SQUARED_ERROR);
//        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        getchar();

        training_strategy.perform_training();

        // Testing Analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        TestingAnalysis::RocAnalysisResults roc_results = testing_analysis.perform_roc_analysis();

        cout << "Area ander curve: " << roc_results.area_under_curve << endl;
        cout << "Confidence limit: " << roc_results.confidence_limit << endl;
        cout << "Optimal threshold: " << roc_results.optimal_threshold << endl;

        // Calculate output

        cout <<  prediction << " -> " << neural_network.calculate_outputs(inputs)(0) << endl;
        cout <<  prediction_2 << " -> " << neural_network.calculate_outputs(inputs)(1) << endl;

        cout << "Goodbye!" << endl;

        return 0;
    }
    catch(const exception& e)
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
