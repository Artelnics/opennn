//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S I M P L E   C L A S S I F I C A T I O N    A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is a pattern recognition problem. 

// System includes

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdexcept>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main()
{
    try
    {
        cout << "OpenNN. Simple classification example." << endl;

        // Data set

        DataSet data_set("../data/simple_pattern_recognition.csv", ';', true);

        // Neural network

        NeuralNetwork neural_network(NeuralNetwork::Classification, {2, 10, 1});

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        const TrainingResults training_results = training_strategy.perform_training();

        // Testing analysis

        const TestingAnalysis testing_analysis(&neural_network, &data_set);

        const Tensor<type, 1> binary_classification_tests = testing_analysis.calculate_binary_classification_tests();

        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

        // Save results

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression_python("../data/expression.py");

        cout << "Bye" << endl;

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
