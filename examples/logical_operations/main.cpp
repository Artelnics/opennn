/****************************************************************************************************************/
/*                                                                                                              */ 
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   L O G I C A L   O P E R A T O R S   A P P L I C A T I O N                                                  */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL (Artelnics)                                                          */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */  
/****************************************************************************************************************/

// This is a classical learning problem.

// System includes

#include <iostream>
#include <math.h>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
    try
    {
        cout << "OpenNN. Logical Operations Application." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set;

        data_set.set_data_file_name("../data/logical_operations.dat");

        data_set.load_data();

        Variables* variables_pointer = data_set.get_variables_pointer();

        variables_pointer->set(2, 6);

        variables_pointer->set_name(0, "X");
        variables_pointer->set_name(1, "Y");
        variables_pointer->set_name(2, "AND");
        variables_pointer->set_name(3, "OR");
        variables_pointer->set_name(4, "NAND");
        variables_pointer->set_name(5, "NOR");
        variables_pointer->set_name(6, "XNOR");
        variables_pointer->set_name(7, "XNOR");

        const Matrix<string> inputs_information = variables_pointer->get_inputs_information();
        const Matrix<string> targets_information = variables_pointer->get_targets_information();

        // Neural network

        NeuralNetwork neural_network(2, 6, 6);

        Inputs* inputs_pointer = neural_network.get_inputs_pointer();

        inputs_pointer->set_information(inputs_information);

        Outputs* outputs_pointer = neural_network.get_outputs_pointer();

        outputs_pointer->set_information(targets_information);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.perform_training();

        // Save results

        data_set.save("../data/data_set.xml");

        neural_network.save("../data/neural_network.xml");

        training_strategy.save("../data/training_strategy.xml");

        // Print results to screen

        Vector<double> inputs(2, 0.0);
        Vector<double> outputs(6, 0.0);

        cout << "X Y AND OR NAND NOR XOR XNOR" << endl;

        inputs[0] = 1.0;
        inputs[1] = 1.0;

        outputs = neural_network.get_multilayer_perceptron_pointer()->calculate_outputs(inputs.to_row_matrix());

        cout << inputs.calculate_binary() << " " << outputs.calculate_binary() << endl;

        inputs[0] = 1.0;
        inputs[1] = 0.0;

        outputs = neural_network.get_multilayer_perceptron_pointer()->calculate_outputs(inputs.to_row_matrix());

        cout << inputs.calculate_binary() << " " << outputs.calculate_binary() << endl;

        inputs[0] = 0.0;
        inputs[1] = 1.0;

        outputs = neural_network.get_multilayer_perceptron_pointer()->calculate_outputs(inputs.to_row_matrix());

        cout << inputs.calculate_binary() << " " << outputs.calculate_binary() << endl;

        inputs[0] = 0.0;
        inputs[1] = 0.0;

        outputs = neural_network.get_multilayer_perceptron_pointer()->calculate_outputs(inputs.to_row_matrix());

        cout << inputs.calculate_binary() << " " << outputs.calculate_binary() << endl;

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
