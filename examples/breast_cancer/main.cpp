//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B R E A S T   C A N C E R   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is a pattern recognition problem.

// System includes

#include <iostream>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main()
{
    try
    {
        cout << "OpenNN. Breast Cancer Application." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        DataSet data_set("../data/breast_cancer.csv", ';', true);

        NeuralNetwork neural_network(NeuralNetwork::Classification, {9, 6, 1});

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::SUM_SQUARED_ERROR);

        training_strategy.set_optimization_method(TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM);

        training_strategy.perform_training();

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        testing_analysis.print_binary_classification_tests();

        neural_network.save_expression_python("../data/breast_cancer.py");

        cout << "End breast cancer application" << endl;

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
