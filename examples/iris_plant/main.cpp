/****************************************************************************************************************/
/*                                                                                                              */ 
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   I R I S   P L A N T   A P P L I C A T I O N                                                                */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL (Artelnics)                                                          */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */  
/****************************************************************************************************************/

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

int main(void)
{
    try
    {
        cout << "OpenNN. Iris Plant Application." << endl;

        srand((unsigned)time(NULL));

        // Data set

        DataSet data_set;

        data_set.set_data_file_name("../data/iris_plant.dat");

        data_set.set_separator("Space");

        data_set.load_data();

        // Variables

        Variables* variables_pointer = data_set.get_variables_pointer();

        variables_pointer->set_name(0, "sepal_length");
        variables_pointer->set_units(0, "centimeters");
        variables_pointer->set_use(0, Variables::Input);

        variables_pointer->set_name(1, "sepal_width");
        variables_pointer->set_units(1, "centimeters");
        variables_pointer->set_use(1, Variables::Input);

        variables_pointer->set_name(2, "petal_length");
        variables_pointer->set_units(2, "centimeters");
        variables_pointer->set_use(2, Variables::Input);

        variables_pointer->set_name(3, "petal_width");
        variables_pointer->set_units(3, "centimeters");
        variables_pointer->set_use(3, Variables::Input);

        variables_pointer->set_name(4, "iris_setosa");
        variables_pointer->set_use(4, Variables::Target);

        variables_pointer->set_name(5, "iris_versicolour");
        variables_pointer->set_use(5, Variables::Target);

        variables_pointer->set_name(6, "iris_virginica");
        variables_pointer->set_use(6, Variables::Target);

        const Matrix<string> inputs_information = variables_pointer->get_inputs_information();
        const Matrix<string> targets_information = variables_pointer->get_targets_information();

        // Instances

        Instances* instances_pointer = data_set.get_instances_pointer();

        instances_pointer->split_random_indices();

        const Vector< Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();

        // Neural network

        NeuralNetwork neural_network(4, 6, 3);

        neural_network.get_multilayer_perceptron_pointer()->get_layer_pointer(0)->set_activation_function(PerceptronLayer::ActivationFunction::Logistic);
        neural_network.get_multilayer_perceptron_pointer()->get_layer_pointer(1)->set_activation_function(PerceptronLayer::ActivationFunction::Logistic);

        Inputs* inputs_pointer = neural_network.get_inputs_pointer();

        inputs_pointer->set_information(inputs_information);

        Outputs* outputs_pointer = neural_network.get_outputs_pointer();

        outputs_pointer->set_information(targets_information);

        neural_network.construct_scaling_layer();

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

        scaling_layer_pointer->set_statistics(inputs_statistics);

        scaling_layer_pointer->set_scaling_methods(ScalingLayer::NoScaling);

        neural_network.construct_probabilistic_layer();

        ProbabilisticLayer* probabilistic_layer_pointer = neural_network.get_probabilistic_layer_pointer();

        probabilistic_layer_pointer->set_probabilistic_method(ProbabilisticLayer::Softmax);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_training_method(TrainingStrategy::QUASI_NEWTON_METHOD);

        QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();

        //quasi_Newton_method_pointer->get_training_rate_algorithm_pointer()->set_training_rate_method(TrainingRateAlgorithm::TrainingRateMethod::GoldenSection);

        quasi_Newton_method_pointer->set_minimum_loss_decrease(1.0e-6);

        quasi_Newton_method_pointer->perform_training();

        training_strategy.set_display(false);

        // Model selection

//        ModelSelection model_selection(&training_strategy);

//        model_selection.set_order_selection_method(ModelSelection::GOLDEN_SECTION);

//        GoldenSectionOrder* golden_section_order_pointer = model_selection.get_golden_section_order_pointer();

//        golden_section_order_pointer->set_tolerance(1.0e-7);

//        ModelSelection::Results model_selection_results = model_selection.perform_order_selection();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        const Matrix<size_t> confusion = testing_analysis.calculate_confusion();

        // Save results

        data_set.save("../data/data_set.xml");

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression("../data/expression.txt");

        training_strategy.save("../data/training_strategy.xml");

//        model_selection.save("./data/model_selection.xml");
//        model_selection_results.save("./data/model_selection_results.dat");

        confusion.save("../data/confusion.dat");

        return(0);
    }
    catch(exception& e)
    {
        cout << e.what() << endl;

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
