/****************************************************************************************************************/
/*                                                                                                              */ 
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   Y A C H T   R E S I S T A N C E   P R O D U C T I O N   A P P L I C A T I O N                              */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
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
      std::cout << "OpenNN. Yacht Resistance Production Application." << std::endl;	
  
      // Neural network

      const std::string neural_network_file_name = "../data/neural_network.xml";

      NeuralNetwork neural_network(neural_network_file_name);

      double longitudinal_position_center_buoyancy;
      double prismatic_coefficient;
      double length_displacement_ratio;
      double beam_draught_ratio;
      double lenght_beam_ratio;
      double Froude_number;

      std::cout << "Enter longitudinal position of the center of buoyancy (-5-0):" << std::endl;
      std::cin >> longitudinal_position_center_buoyancy;

      std::cout << "Enter prismatic coeficient (0.53-0.6):" << std::endl;
      std::cin >> prismatic_coefficient;

      std::cout << "Enter length-displacement ratio (4.34-5.14):" << std::endl;
      std::cin >> length_displacement_ratio;

      std::cout << "Enter beam-draught ratio (2.81-5.35):" << std::endl;
      std::cin >> beam_draught_ratio;

      std::cout << "Enter length-beam ratio (2.73-3.64):" << std::endl;
      std::cin >> lenght_beam_ratio;

      std::cout << "Enter Froude number (0.125-0.45):" << std::endl;
      std::cin >> Froude_number;

      Vector<double> inputs(6);
      inputs[0] = longitudinal_position_center_buoyancy;
      inputs[1] = prismatic_coefficient;
      inputs[2] = length_displacement_ratio;
      inputs[3] = beam_draught_ratio;
      inputs[4] = lenght_beam_ratio;
      inputs[5] = Froude_number;

      Vector<double> outputs = neural_network.calculate_outputs(inputs);

      double residuary_resistance = outputs[0];

      std::cout << "Residuary resistance per unit weight of displacement:\n"
                << residuary_resistance << std::endl;

      return(0);
   }
   catch(std::exception& e)
   {
      std::cerr << e.what() << std::endl;

      return(1);
   }
}  

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2015 Roberto Lopez
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

