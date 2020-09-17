//   OpenNN: Open Neural Networks Library
//   www.opennn.org
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <limits.h>
#include <statistics.h>
#include <regex>

// Systems Complementaries

#include <cmath>
#include <cstdlib>
#include <ostream>


// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace chrono;


int main(void)
{
    try
    {
        cout << "OpenNN. Blank Application." << endl;

        DataSet data_set("C:\\Users\\Usuario\\Documents\\breastcancer.csv", ';', true);

        data_set.impute_missing_values_mean();

        const Index inputs_number = data_set.get_input_variables_number();
        const Index targets_number = data_set.get_target_variables_number();

        Tensor<Index, 1> architecture(3);
        architecture[0] = inputs_number;
        architecture[1] = 7;
        architecture[2] = targets_number;

        cout << "Architecture: " << architecture << endl;

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, architecture);

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);
        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);

        training_strategy.perform_training();

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

        const Tensor<Index, 0> confusion_sum = confusion.sum();

        cout << "Confusion: " << endl << confusion << endl;

        cout << "Accuracy: " << (static_cast<type>(confusion(0,0))+static_cast<type>(confusion(1,1)))/static_cast<type>(confusion_sum()) << endl;

        cout << "End" << endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
