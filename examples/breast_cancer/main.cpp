//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B R E A S T   C A N C E R   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is a pattern recognition problem.



#include <iostream>
#include <time.h>



#include "../../opennn/opennn.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Breast Cancer Application." << endl;

        srand(unsigned(time(nullptr)));

        // Data set

        DataSet data_set("../data/breast_cancer.csv", ";", true);

        data_set.save("../data/data_set.xml");
        data_set.load("../data/data_set.xml");
/*
        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        // Neural network

        const Index neurons_number = 6;

        NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification,
                                     {input_variables_number, neurons_number, target_variables_number});

        neural_network.print();

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        //training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
        //training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);


        // OKR
        training_strategy.set_loss_method(TrainingStrategy::LossMethod::SUM_SQUARED_ERROR);
        //training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
        //training_strategy.set_loss_method(TrainingStrategy::LossMethod::MINKOWSKI_ERROR);
        //training_strategy.set_loss_method(TrainingStrategy::LossMethod::WEIGHTED_SQUARED_ERROR);

        // cross entropy

        // OKR
        //training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::GRADIENT_DESCENT);
        //training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::CONJUGATE_GRADIENT);
        //training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM); //Fail
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);
        //training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        training_strategy.perform_training();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        testing_analysis.print_binary_classification_tests();

        // Save results

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression_python("../data/breast_cancer.py");

        cout << "End breast cancer application" << endl;

        // OKR
        cout << " \n write_loss_method \n" << training_strategy.write_loss_method_text();
        cout << " \n write_opt_method \n" << training_strategy.write_optimization_method_text();
*/
        return 0;
    }
    catch(const exception& e)
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
