/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   U R I N A R Y   I N F L A M M A T I O N S   D I A G N O S I S   A P P L I C A T I O N                      */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
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
        int rank = 0;

#ifdef __OPENNN_MPI__

        int size = 1;

        MPI_Init(NULL,NULL);

        MPI_Comm_size(MPI_COMM_WORLD, &size);

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#endif

        if(rank == 0)
        {
            std::cout << "OpenNN. Urinary Inflammations Diagnosis Application." << std::endl;
        }
        srand((unsigned)time(NULL));

        // Global variables

        DataSet data_set;

        NeuralNetwork neural_network;

        LossIndex loss_index;

        TrainingStrategy training_strategy;

        ModelSelection model_selection;

        // Local variables

        DataSet local_data_set;

        NeuralNetwork local_neural_network;

        LossIndex local_loss_index;

        TrainingStrategy local_training_strategy;

        ModelSelection local_model_selection;

        if(rank == 0)
        {
            // Data set

            data_set.set_data_file_name("../data/urinary_inflammations_diagnosis.dat");

            data_set.set_separator("Space");

            data_set.load_data();

            // Variables

            Variables* variables_pointer = data_set.get_variables_pointer();

            variables_pointer->set_name(0, "temperature");
            variables_pointer->set_units(0, "degree_celsius");
            variables_pointer->set_use(0, Variables::Input);

            variables_pointer->set_name(1, "nausea");
            variables_pointer->set_units(1, "boolean");
            variables_pointer->set_use(1, Variables::Input);

            variables_pointer->set_name(2, "lumbar_pain");
            variables_pointer->set_units(2, "boolean");
            variables_pointer->set_use(2, Variables::Input);

            variables_pointer->set_name(3, "urine_pushing");
            variables_pointer->set_units(3, "boolean");
            variables_pointer->set_use(3, Variables::Input);

            variables_pointer->set_name(4, "micturition_pain");
            variables_pointer->set_units(4, "boolean");
            variables_pointer->set_use(4, Variables::Input);

            variables_pointer->set_name(5, "urethra_pain");
            variables_pointer->set_units(5, "boolean");
            variables_pointer->set_use(5, Variables::Input);

            variables_pointer->set_name(6, "urinary_inflammation");
            variables_pointer->set_units(6, "boolean");
            variables_pointer->set_use(6, Variables::Unused);

            variables_pointer->set_name(7, "nephritis_renal_pelvis_origin");
            variables_pointer->set_units(7, "boolean");
            variables_pointer->set_use(7, Variables::Target);

            const Matrix<std::string> inputs_information = variables_pointer->arrange_inputs_information();
            const Matrix<std::string> targets_information = variables_pointer->arrange_targets_information();

            // Instances

            Instances* instances_pointer = data_set.get_instances_pointer();

            instances_pointer->split_random_indices();

            const Vector< Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();

            // Neural network

            neural_network.set(6, 6, 1);

            Inputs* inputs_pointer = neural_network.get_inputs_pointer();

            inputs_pointer->set_information(inputs_information);

            Outputs* outputs_pointer = neural_network.get_outputs_pointer();

            outputs_pointer->set_information(targets_information);

            neural_network.construct_scaling_layer();

            ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

            scaling_layer_pointer->set_statistics(inputs_statistics);

            scaling_layer_pointer->set_scaling_method(ScalingLayer::NoScaling);

            // Loss index

            loss_index.set_data_set_pointer(&data_set);
            loss_index.set_neural_network_pointer(&neural_network);

            // Training strategy

            training_strategy.set(&loss_index);

            training_strategy.set_main_type(TrainingStrategy::QUASI_NEWTON_METHOD);

            QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();

            quasi_Newton_method_pointer->set_minimum_loss_increase(1.0e-6);

            training_strategy.set_display(false);

            // Model selection

            model_selection.set_training_strategy_pointer(&training_strategy);

            model_selection.set_inputs_selection_type(ModelSelection::GENETIC_ALGORITHM);

            GeneticAlgorithm* genetic_algorithm_pointer = model_selection.get_genetic_algorithm_pointer();

            genetic_algorithm_pointer->set_approximation(false);

            genetic_algorithm_pointer->set_inicialization_method(GeneticAlgorithm::Random);

            genetic_algorithm_pointer->set_display(true);
        }

#ifdef __OPENNN_MPI__
        MPI_Barrier(MPI_COMM_WORLD);

        local_data_set.set_MPI(&data_set);

        local_neural_network.set_MPI(&neural_network);

        local_loss_index.set_MPI(&local_data_set,&local_neural_network,&loss_index);

        local_training_strategy.set_MPI(&local_loss_index,&training_strategy);

        local_model_selection.set_MPI(&local_training_strategy, &model_selection);

        MPI_Barrier(MPI_COMM_WORLD);

        local_training_strategy.set_display(false);

        local_model_selection.set_approximation(false);

        ModelSelection::ModelSelectionResults model_selection_results = local_model_selection.perform_inputs_selection();
#else
        ModelSelection::ModelSelectionResults model_selection_results = model_selection.perform_inputs_selection();
#endif

        if(rank == 0)
        {
#ifdef __OPENNN_MPI__
            neural_network.set_multilayer_perceptron_pointer(local_neural_network.get_multilayer_perceptron_pointer());
#endif
            // Testing analysis

            TestingAnalysis testing_analysis(&neural_network, &data_set);

            const Matrix<size_t> confusion = testing_analysis.calculate_confusion();

            // Save results

            ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

            scaling_layer_pointer->set_scaling_method(ScalingLayer::MinimumMaximum);

            data_set.save("../data/data_set.xml");

            neural_network.save("../data/neural_network.xml");
            neural_network.save_expression("../data/expression.txt");

            training_strategy.save("../data/training_strategy.xml");

            model_selection.save("../data/model_selection.xml");
            //        model_selection_results.save("../data/model_selection_results.dat");

            confusion.save("../data/confusion.dat");
        }

#ifdef __OPENNN_MPI__

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Finalize();
#endif
        return(0);
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;

        return(1);
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2015 Roberto Lopez
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
