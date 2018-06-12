/****************************************************************************************************************/
/*                                                                                                              */ 
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   Y A C H T   R E S I S T A N C E   P R O D U C T I O N   A P P L I C A T I O N                              */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL (Artelnics)                                                          */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */  
/****************************************************************************************************************/

// This is an example of a neural network working on the production phase. 

// System includes

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
    try
    {

        cout << "OpenNN. Yacht Resistance Production Application." << endl;

        // Neural network

        const string neural_network_file_name = "../data/neural_network.xml";

        NeuralNetwork neural_network(neural_network_file_name);

        double longitudinal_position_center_buoyancy;
        double prismatic_coefficient;
        double length_displacement_ratio;
        double beam_draught_ratio;
        double lenght_beam_ratio;
        double Froude_number;

        cout << "Enter longitudinal position of the center of buoyancy (-5-0):" << endl;
        std::cin >> longitudinal_position_center_buoyancy;

        cout << "Enter prismatic coeficient (0.53-0.6):" << endl;
        std::cin >> prismatic_coefficient;

        cout << "Enter length-displacement ratio (4.34-5.14):" << endl;
        std::cin >> length_displacement_ratio;

        cout << "Enter beam-draught ratio (2.81-5.35):" << endl;
        std::cin >> beam_draught_ratio;

        cout << "Enter length-beam ratio (2.73-3.64):" << endl;
        std::cin >> lenght_beam_ratio;

        cout << "Enter Froude number (0.125-0.45):" << endl;
        std::cin >> Froude_number;

        Vector<double> inputs(6);
        inputs[0] = longitudinal_position_center_buoyancy;
        inputs[1] = prismatic_coefficient;
        inputs[2] = length_displacement_ratio;
        inputs[3] = beam_draught_ratio;
        inputs[4] = lenght_beam_ratio;
        inputs[5] = Froude_number;

        Vector<double> outputs = neural_network.calculate_outputs(inputs.to_row_matrix());

        double residuary_resistance = outputs[0];

        cout << "Residuary resistance per unit weight of displacement:\n"
                  << residuary_resistance << endl;

        return(0);
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return(1);
    }
}  

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques SL
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

