//   OpenNN: An Open Source Neural Networks C++ Library                    
//   www.opennn.net
//
//   M O D E L   S E L E C T I O N   T E S T   C L A S S   H E A D E R     
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#ifndef RESPONSEOPTIMIZATIONTEST_H
#define RESPONSEOPTIMIZATIONTEST_H

#include "../opennn/unit_testing.h"

class ResponseOptimizationTest : public UnitTesting
{

public:

    explicit ResponseOptimizationTest();

    virtual ~ResponseOptimizationTest();

    // Constructor and destructor methods

    void test_constructor();

    void test_destructor();

    // Set methods

    void test_set();

    void test_set_evaluations_number();

    void test_set_input_condition();
    void test_set_output_condition();
    void test_set_inputs_outputs_conditions();

    //

    void test_get_conditions();

    void test_get_values_conditions();

    // Performance methods

    void test_calculate_inputs();

    void test_perform_optimization();

    // Unit testing methods

    void run_test_case();

private:

    Tensor<type,1> conditions_values;

    DataSet data_set;

    NeuralNetwork neural_network;

    NeuralNetwork neural_network_2;

    TrainingStrategy training_strategy;

    ResponseOptimization response_optimization;

    void generate_neural_networks(){

        // Simple outputs

        Index input_variables_number = 2;
        Index target_variables_number = 1;
        Index hidden_neurons_number = 2;

        Tensor<type,2> data(15,3);
        data.setRandom();

        for(Index i = 0; i < data.dimension(0); i++)
        {
            data(i,2) = data(i,0) * data(i,0) + data(i,1) * data(i,1) + 1;
        }

        data_set.set(data);

        Tensor<string,1> names(3);
        names.setValues({"x","y","z"});
        data_set.set_variables_names(names);
        data_set.set_training();

        neural_network.set(NeuralNetwork::ModelType::Approximation,
                                     { input_variables_number, hidden_neurons_number, target_variables_number});

        training_strategy.set(&neural_network, &data_set);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);
        training_strategy.set_display(false);
        training_strategy.perform_training();

        // Multiple outputs

        data.resize(15, 4);
        data.setRandom();

        for(Index i = 0; i < data.dimension(0); i++)
        {
            data(i,2) = data(i,0) * data(i,0) + data(i,1) * data(i,1) + 1;
            data(i,3) = data(i,0) * data(i,0) + data(i,1) * data(i,1) - 1;
        }

        data_set.set(data);

        Tensor<string,1> names_2(4);
        names_2.setValues({"x","y","z","t"});
        data_set.set_variables_names(names_2);
        data_set.set_training();

        Tensor<Index,1> inputs_index(2);
        Tensor<Index,1> outputs_index(2);

        inputs_index.setValues({0,1});
        outputs_index.setValues({2,3});

        data_set.set_input_target_columns(inputs_index,outputs_index);

        neural_network_2.set(NeuralNetwork::ModelType::Approximation,
                                     { data_set.get_input_numeric_variables_number(), 2, data_set.get_target_numeric_variables_number()});

        training_strategy.set(&neural_network_2, &data_set);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);
        training_strategy.set_display(false);
        training_strategy.perform_training();

    };
};

#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
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
