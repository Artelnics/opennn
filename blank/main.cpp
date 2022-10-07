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
#include <string>
#include <time.h>

// OpenNN includes

//#include "../opennn/opennn.h"
//using namespace opennn;
using namespace std;


string get_output(const string& input)
{
    string output;

    return output;
}


int main(int argc, char *argv[])
{
    try
    {
        cout << "Hello OpenNN!" << endl;

       
        string input = R"(scaled_glcm = (glcm-127.2750015)/10.34020042;
            scaled_green = (green - 208.875) / 73.70839691;
            scaled_red = (red - 109.0800018) / 67.42389679;
            scaled_nir = (nir - 449.0880127) / 151.2619934;
            scaled_pan_band = (pan_band - 20.60160065) / 6.621739864;
            perceptron_layer_1_output_0 = tanh(-1.0725 + (scaled_glcm * 0.375452) + (scaled_green * 3.50946) + (scaled_red * 0.578346) + (scaled_nir * -0.695273) + (scaled_pan_band * -0.170702));
            perceptron_layer_1_output_1 = tanh(-0.224889 + (scaled_glcm * 0.0303833) + (scaled_green * -2.72553) + (scaled_red * 1.48712) + (scaled_nir * 0.458527) + (scaled_pan_band * -0.0243582));
            perceptron_layer_1_output_2 = tanh(-0.472289 + (scaled_glcm * -0.126885) + (scaled_green * 1.35568) + (scaled_red * -4.50396) + (scaled_nir * 0.533323) + (scaled_pan_band * -0.0159912));
            probabilistic_layer_combinations_0 = -2.61513 - 3.37299 * perceptron_layer_1_output_0 + 3.29145 * perceptron_layer_1_output_1 - 5.03247 * perceptron_layer_1_output_2;
            wilt = 1.0 / (1.0 + exp(-probabilistic_layer_combinations_0);\n)";

        cout << input << endl;

        //string output = get_output(input);


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

