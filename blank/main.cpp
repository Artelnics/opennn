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

/*
void replace_all(std::string& s, std::string const& toReplace, std::string const& replaceWith) {
    std::string buf;
    std::size_t pos = 0;
    std::size_t prevPos;

    // Reserves rough estimate of final size of string.
    buf.reserve(s.size());

    while (true) {
        prevPos = pos;
        pos = s.find(toReplace, pos);
        if (pos == std::string::npos)
            break;
        buf.append(s, prevPos, pos - prevPos);
        buf += replaceWith;
        pos += toReplace.size();
    }

    buf.append(s, prevPos, s.size() - prevPos);
    s.swap(buf);
}
*/

/**
string get_output(string& input)
{
    std::string output = input;
    string phpVAR = "$";
    std::vector<std::string> found_tokens;

    std::vector<std::string> tokens;
    std::vector<std::string> tokens_output;
    std::string token;
    std::stringstream ss(input);

    while (getline(ss, token, '\n')) {
        tokens.push_back(token);
    }
    for (auto& s : tokens) {
        string word = "";
        for (char& c : s) {
            if (c != ' ') {
                word += c;
            }
            else {
                break;
            }
        }
        found_tokens.push_back(word);

    }

    //AT THIS POINT, WE KNOW WHICH THE INPUTS ARE, WHICH THE LEFT SIDE VARIABLES ARE    
    //AND WE HAVE A VECTOR WHERE EACH INDEX IS A LINE 

    for (auto& key_word : found_tokens) {
        string new_word = "";
        new_word = phpVAR + key_word;
        replace_all(output, key_word, new_word);
    }

    //std::cout << output << std::endl;

    return output;
}
*/

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
        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, {3,4,2});
        Eigen::Tensor<string, 1> t2(3);
        t2(0) = "input1";
        t2(1) = "input2";
        t2(2) = "input3";

        Eigen::Tensor<string, 1> t(2);
        t(0) = "jadnlk";
        t(1) = "aksdasdl";
        neural_network.set_inputs_names(t2);
        neural_network.set_outputs_names(t);


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

