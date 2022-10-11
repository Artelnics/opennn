//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <vector>
#include <string>
#include <time.h>
#include <regex>
#include <list>
// OpenNN includes

#include "../opennn/opennn.h"
using namespace opennn;


using namespace std;



int main(int argc, char* argv[])
{
    try
    {
        //=======================================================//
        //                                                       //
        //         1) prepare input names in a tensor            //
        //         2) prepare output names in a tensor           //
        //         3) set the model, and start it                //
        //         4) set input names in the model               //
        //         5) set output names in the model              //
        //         6) call network.write_expression_api          //
        //                                                       //
        //=======================================================//



//        ///THREE INPUTS AND TWO OUTPUTS TEST
//        //                                          ({inputs_number, hidden_neurons_number, outputs_number}).
//        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, {4,4,1});
//        Eigen::Tensor<string, 1> t2(4);
//        t2(0) = "input1";
//        t2(1) = "input2";
//        t2(2) = "input3";
//        t2(3) = "input4";
//
//        Eigen::Tensor<string, 1> t(1);
//        t(0) = "output1";
//        neural_network.set_inputs_names(t2);
//        neural_network.set_outputs_names(t);


        ///THREE INPUTS AND TWO OUTPUTS TEST
        //                                          ({inputs_number, hidden_neurons_number, outputs_number}).
        //Error in Aproximation, forecasting and
        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Forecasting, {3,7,4,2});

        Eigen::Tensor<string, 1> t2(3);
        t2(0) = "input1";
        t2(1) = "input2";
        t2(2) = "input3";

        Eigen::Tensor<string, 1> t(2);
        t(0) = "output1";
        t(1) = "output2";
        neural_network.set_inputs_names(t2);
        neural_network.set_outputs_names(t);

        //LongShortTermMemoryLayer lstm;
        //const string s = "layer0";
        //lstm.set_name(s);
        //lstm.set_inputs_number(20);
        //lstm.set_activation_function("Logistic");
        //lstm.set_recurrent_activation_function("RectifiedLinear");
        //neural_network.add_layer(lstm);

        string expression_api = neural_network.write_expression_api();
        cout << expression_api << endl;
    }
    catch (const exception& e)
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

