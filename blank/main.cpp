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
        cout << "Hello OpenNN!" << endl;
        /*
        string input = "scaled_glcm = (glcm-127.2750015)/10.34020042;\nscaled_green = (green - 208.875) / 73.70839691;\nscaled_red = (red - 109.0800018) / 67.42389679;\nscaled_nir = (nir - 449.0880127) / 151.2619934;\nscaled_pan_band = (pan_band - 20.60160065) / 6.621739864;\nperceptron_layer_1_output_0 = tanh(-1.0725 + (scaled_glcm * 0.375452) + (scaled_green * 3.50946) + (scaled_red * 0.578346) + (scaled_nir * -0.695273) + (scaled_pan_band * -0.170702));\nperceptron_layer_1_output_1 = tanh(-0.224889 + (scaled_glcm * 0.0303833) + (scaled_green * -2.72553) + (scaled_red * 1.48712) + (scaled_nir * 0.458527) + (scaled_pan_band * -0.0243582));\nperceptron_layer_1_output_2 = tanh(-0.472289 + (scaled_glcm * -0.126885) + (scaled_green * 1.35568) + (scaled_red * -4.50396) + (scaled_nir * 0.533323) + (scaled_pan_band * -0.0159912));\nprobabilistic_layer_combinations_0 = -2.61513 - 3.37299 * perceptron_layer_1_output_0 + 3.29145 * perceptron_layer_1_output_1 - 5.03247 * perceptron_layer_1_output_2;\nwilt = 1.0 / (1.0 + exp(-probabilistic_layer_combinations_0);\n";
        string output = get_output(input);

        cout << output << endl;
*/
        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Approximation, {3,4,5});

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

