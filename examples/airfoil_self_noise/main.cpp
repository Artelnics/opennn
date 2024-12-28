//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A I R F O I L   S E L F   N O I S E   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <string>
#include <time.h>

#include "../../opennn/data_set.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/model_selection.h"
#include "../../opennn/testing_analysis.h"

int main()
{
    try
    {
        cout << "Airfoil self noise" << endl;

        // Data set
        
        DataSet data_set("C:/airfoil_self_noise.csv", ";", true);
        
        const Index input_variables_number = data_set.get_variables_number(DataSet::VariableUse::Input);
        const Index target_variables_number = data_set.get_variables_number(DataSet::VariableUse::Target);

        data_set.set(DataSet::SampleUse::Training);
        
        //data_set.print_input_target_raw_variables_correlations();

        //data_set.save("../opennn/examples/airfoil_self_noise/data/neural_network.xml");
        //data_set.load("../opennn/examples/airfoil_self_noise/data/neural_network.xml");

        // Neural network

        const Index neurons_number = 10;

        NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation,
                                     {input_variables_number}, {neurons_number}, {target_variables_number});

        // neural_network.save("../opennn/examples/airfoil_self_noise/data/neural_network.xml");
        // neural_network.load("../opennn/examples/airfoil_self_noise/data/neural_network.xml");

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

//        training_strategy.set_display(false);

        //training_strategy.print();

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
       //training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
        //training_strategy.set_loss_method(TrainingStrategy::LossMethod::SUM_SQUARED_ERROR);
        // training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
        //training_strategy.set_loss_method(TrainingStrategy::LossMethod::MINKOWSKI_ERROR); // @todo gives 0.56

        //training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);
        //training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::CONJUGATE_GRADIENT);
        //training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM); //Fail-Mean Squared error / Doesnt work with MINKOWSKI_ERROR / is not implemented yet with weighted squared error
        //training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(1);

        //training_strategy.set_maximum_epochs_number(10000);

        //training_strategy.save("../data/training_strategy.xml");
        //training_strategy.load("../data/training_strategy.xml");

        training_strategy.perform_training();
/*
        ModelSelection model_selection(&training_strategy);

        model_selection.perform_inputs_selection();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

//        testing_analysis.print_goodness_of_fit_analysis();

        // Save results
        
//        neural_network.save("../opennn/examples/airfoil_self_noise/data/neural_network.xml");
//        neural_network.save_expression_c("../opennn/examples/airfoil_self_noise/data/airfoil_self_noise.c");

        // // Deploy

        // NeuralNetwork new_neural_network("../opennn/examples/airfoil_self_noise/data/neural_network.xml");

        // Tensor<type, 2> inputs(1, input_variables_number);
        // inputs.setRandom();

        // inputs.setValues({{type(800), type(0), type(0.3048), type(71.3), type(0.00266337)}});

        // cout << inputs << endl;

        // const Tensor<type, 2> outputs = new_neural_network.calculate_outputs(inputs);

//        cout << outputs << endl;


        Tensor<type, 2> inputs(1, 1);
        inputs(0,0) = 0.1;

        cout << inputs << endl;

        const Tensor<type, 2> outputs = neural_network.calculate_outputs(inputs);

        cout << outputs << endl;

        neural_network.print();
*/

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(exception& e)
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
