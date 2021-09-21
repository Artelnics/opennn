//   OpenNN: Open Neural Networks Library
//   www.opennn.org
//
//   T E X T   G E N E R A T I O N   E X A M P L E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <limits.h>
#include <statistics.h>
#include <regex>

// Systems Complementaries

#include <cmath>
#include <cstdlib>
#include <ostream>


// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace chrono;

string read_text(const string& filename)
{
    string text;

    // Reads a CSV file into a vector of vector<int>
    ifstream myFile(filename);

    // Make sure the file is open
    if(!myFile.is_open()) throw runtime_error("Could not open file");

    string line;

    // Read data, line by line
    while(getline(myFile, line))
    {
        // Erase semicolons ";"
        erase(line,';');
        text.append(line);
    }

    myFile.close();
    return text;
}

string create_character_list(const string& text)
{
    string character_list;
    for (auto i : text)
    {
        if( character_list.find(i) == string::npos)
        {
            character_list.append(&i);
        }
    }
    return character_list;
}

Tensor<type,2> text_to_one_hot(const string& text, const string& character_list)
{
    Tensor<type,2> one_hot(text.length(),character_list.length());

    int hot_index;
    for(Index i=0; i<int(text.length());i++)
    {
        for(Index j=0; j<int(character_list.length());j++)
        {
           one_hot(i,j)=0;
        }
        hot_index = character_list.find(text[i]);
        one_hot(i,hot_index)=1;
    }
    cout << "Transformation text-onehot completed" <<endl;
    return one_hot;
}

string one_hot_to_text(Tensor<type,2>& &one_hot, const string& character_list)
{
    string text;

    const Tensor<type, 2>::Dimensions& dim = one_hot.dimensions(); // dim[0] = text length, dim[1] = character list length

    for(Index i=0; i<dim[0] ; i++)
    {
        for(Index j=0; j<dim[1]; j++)
        {
            if(one_hot(i,j)==1)
            {
                text.push_back(character_list[j]);
            }
        }
    }
    cout << "Transformation onehot-text completed" <<endl;
    return text;
}



int main(void)
{
    try
    {

        cout << "OpenNN. Text Generation Example." << endl;

        // Dataset

        string text = read_text("../data/text_generation.csv");

        string character_list = create_character_list(text);
        cout << "Character list length: " << character_list.length() <<endl;

        Tensor<type, 2> one_hot = text_to_one_hot(text,character_list);

        DataSet data_set(one_hot);

        int lags_number = 1;
        data_set.set_lags_number(lags_number);
        data_set.set_steps_ahead_number(1);

        data_set.transform_time_series();
        data_set.print();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        cout << "Input variables number: " << input_variables_number << endl;

        cout << "Target variables number: " << target_variables_number << endl;

        // Neural network

        const Index hidden_neurons_number = 1;

        Tensor<Index, 1> architecture(3);
        architecture.setValues({input_variables_number, hidden_neurons_number, target_variables_number});

        //NeuralNetwork neural_network(NeuralNetwork::Forecasting, architecture);

        NeuralNetwork neural_network;

        LongShortTermMemoryLayer lstm_layer(input_variables_number,hidden_neurons_number);
        ProbabilisticLayer probabilistic_layer(hidden_neurons_number,target_variables_number);

        neural_network.add_layer(&lstm_layer);
        neural_network.add_layer(&probabilistic_layer);

        neural_network.print();

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::ADAPTIVE_MOMENT_ESTIMATION);

        AdaptiveMomentEstimation* adam = training_strategy.get_adaptive_moment_estimation_pointer();

        adam->set_loss_goal(1.0e-3);
        adam->set_maximum_epochs_number(5);
        adam->set_display_period(1);

        training_strategy.perform_training();

//        // Testing analysis

//        Tensor<type, 2> inputs(3,6);

//        inputs.setValues({{1,0,0,0,0,0},
//                          {1,1,1,0.5,0.5,1},
//                          {0,1,0,1,0,1}});

//        cout << "inputs: " << endl;
//        cout << inputs << endl;

//        cout << "outputs: " << endl;
//        cout << neural_network.calculate_outputs(inputs) << endl;

//        data_set.unscale_input_variables(scaling_inputs_methods, inputs_descriptives);

//        TestingAnalysis testing_analysis(&neural_network, &data_set);

//        Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

//        cout << "Confusion: " << endl;
//        cout << confusion << endl;

//        // Save results

//        data_set.save("../data/data_set.xml");
//        neural_network.save("../data/neural_network.xml");
//        training_strategy.save("../data/training_strategy.xml");

        cout << "End Text Generation Example" << endl;

        return 0;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;

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
