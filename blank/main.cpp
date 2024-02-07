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
#include <chrono>

// OpenNN includes

#include "../opennn/opennn.h"
#include <iostream>

using namespace std;
using namespace opennn;


int main()
{
   try
   {
        cout << "OpenNN. Simple Function Regression Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data Set

        DataSet data_set("/home/artelnics/Escritorio/gyd_copia.csv", ',', true);

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();
        const Index hidden_neurons_number = 3;

        Tensor<Correlation, 2> result = data_set.calculate_input_target_columns_correlations();

        // calculate_input_columns_correlations(const bool& calculate_pearson_correlations, const bool& calculate_spearman_correlations)

        cout << "==============================" << endl;
        cout << "result.size() :: " << result.size() << endl;
        cout << "==============================" << endl;
        result(0).print();
        cout << "==============================" << endl;

        // Neural Network

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Approximation,
                                     {input_variables_number, hidden_neurons_number, target_variables_number});


        // Training Strategy

        //TrainingStrategy training_strategy(&neural_network, &data_set);
        //training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);
        //training_strategy.set_display_period(1000);
        //training_strategy.perform_training();

        //Model Selection
        //GrowingNeurons gn(&training_strategy);
        //gn.perform_neurons_selection();

        // Save results
        //neural_network.save_expression_python("simple_function_regresion.py");

        cout << "Bye Simple Function Regression" << endl;//

        return 0;
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
