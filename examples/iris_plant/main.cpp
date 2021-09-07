//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I R I S   P L A N T   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is a classical pattern recognition problem.

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
using namespace std;
using namespace Eigen;

int main()
{
    try
    {
        cout << "OpenNN. Iris Plant Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        DataSet data_set("D:/opennn/examples/iris_plant/data/iris_plant_original.csv", ';', true);

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, {4, 3, 3});

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.perform_training();

        const TestingAnalysis testing_analysis(&neural_network, &data_set);

        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

        neural_network.save_expression_python("D:/opennn/examples/iris_plant/data/neural_network.py");

        system("pause");

        return 0;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;

        system("pause");

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
