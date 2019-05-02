/****************************************************************************************************************/
/*                                                                                                              */ 
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   B R E A S T   C A N C E R   A P P L I C A T I O N                                                          */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL (Artelnics)                                                          */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */  
/****************************************************************************************************************/

// This is a pattern recognition problem.

// System includes

#include <iostream>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
    try
    {
        cout << "OpenNN. Breast Cancer Application." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set;

        data_set.set_data_file_name("../data/breast_cancer.dat");

        data_set.load_data();

        data_set.set_file_type("dat");

        // Variables

        Variables* variables_pointer = data_set.get_variables_pointer();

        variables_pointer->set_name(0, "clump_thickness");
        variables_pointer->set_name(1, "cell_size_uniformity");
        variables_pointer->set_name(2, "cell_shape_uniformity");
        variables_pointer->set_name(3, "marginal_adhesion");
        variables_pointer->set_name(4, "single_epithelial_cell_size");
        variables_pointer->set_name(5, "bare_nuclei");
        variables_pointer->set_name(6, "bland_chromatin");
        variables_pointer->set_name(7, "normal_nucleoli");
        variables_pointer->set_name(8, "mitoses");
        variables_pointer->set_name(9, "diagnose");

        // Instances

        Instances* instances_pointer = data_set.get_instances_pointer();

        instances_pointer->split_random_indices();

        const Matrix<string> inputs_information = variables_pointer->get_inputs_information();
        const Matrix<string> targets_information = variables_pointer->get_targets_information();

        const Vector< Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();

        // Neural network

        NeuralNetwork neural_network(9, 6, 1);

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

        probabilistic_layer_pointer->set_probabilistic_method(ProbabilisticLayer::Probability);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();

        quasi_Newton_method_pointer->set_minimum_loss_decrease(1.0e-6);

        quasi_Newton_method_pointer->set_display(false);

        training_strategy.set_display(false);

        // Model selection

        ModelSelection model_selection(&training_strategy);

        model_selection.set_order_selection_method(ModelSelection::SIMULATED_ANNEALING);

        SimulatedAnnealingOrder* simulated_annealing_order_pointer = model_selection.get_simulated_annealing_order_pointer();

        simulated_annealing_order_pointer->set_cooling_rate(0.9);

        simulated_annealing_order_pointer->set_maximum_iterations_number(15);

        const ModelSelection::Results model_selection_results = model_selection.perform_order_selection();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        Matrix<size_t> confusion = testing_analysis.calculate_confusion();

        Vector<double> binary_classification_tests = testing_analysis.calculate_binary_classification_tests();

        // Save results

        data_set.save("../data/data_set.xml");

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression("../data/expression.txt");

        training_strategy.save("../data/training_strategy.xml");

        model_selection.save("../data/model_selection.xml");
        model_selection_results.save("../data/model_selection_results.dat");

        confusion.save("../data/confusion.dat");
        binary_classification_tests.save("../data/binary_classification_tests.dat");

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
