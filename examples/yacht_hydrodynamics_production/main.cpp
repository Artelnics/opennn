//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Y A C H T   R E S I S T A N C E   P R O D U C T I O N   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is an example of a neural network working on the production phase. 

// System includes

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <omp.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
    try
    {

        cout << "OpenNN. Yacht Resistance Production Example." << endl;

        // Neural network

        const string neural_network_file_name = "../data/neural_network.xml";

        NeuralNetwork neural_network(neural_network_file_name);

        type longitudinal_position_center_buoyancy;
        type prismatic_coefficient;
        type length_displacement_ratio;
        type beam_draught_ratio;
        type lenght_beam_ratio;
        type Froude_number;

        cout << "Enter longitudinal position of the center of buoyancy (-5-0):" << endl;
        cin >> longitudinal_position_center_buoyancy;

        cout << "Enter prismatic coeficient (0.53-0.6):" << endl;
        cin >> prismatic_coefficient;

        cout << "Enter length-displacement ratio (4.34-5.14):" << endl;
        cin >> length_displacement_ratio;

        cout << "Enter beam-draught ratio (2.81-5.35):" << endl;
        cin >> beam_draught_ratio;

        cout << "Enter length-beam ratio (2.73-3.64):" << endl;
        cin >> lenght_beam_ratio;

        cout << "Enter Froude number (0.125-0.45):" << endl;
        cin >> Froude_number;

        Tensor<type, 2> inputs(1, 6);
        inputs.setValues({{longitudinal_position_center_buoyancy, prismatic_coefficient,length_displacement_ratio, beam_draught_ratio, lenght_beam_ratio, Froude_number}});

        Tensor<type, 2> outputs = neural_network.calculate_outputs(inputs);

        double residuary_resistance = outputs(0, 0);

        cout << "Residuary resistance per unit weight of displacement:\n"
                  << residuary_resistance << endl;

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

