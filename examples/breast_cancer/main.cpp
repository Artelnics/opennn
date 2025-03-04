//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B R E A S T   C A N C E R   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>
#include <time.h>

#include "../../opennn/opennn.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Breast Cancer Application." << endl;

        // Data set

<<<<<<< HEAD
        DataSet data_set("data/breast_cancer.csv", ";", true);
        //DataSet data_set("/Users/artelnics/Documents/opennn/examples/breast_cancer/data/breast_cancer.csv", ";", true);
=======
        // DataSet data_set("data/breast_cancer.csv", ";", true);
        DataSet data_set("/Users/artelnics/Documents/opennn/examples/breast_cancer/data/breast_cancer.csv", ";", true, false);
>>>>>>> 575d26b6acd6f039338141494c32b9bf6ba190f9

        // Example downloaded dataset

        // DataSet data_set("/Users/artelnics/Desktop/breast-cancer-modified.csv", ",", true);

        // 5 years mortality dataset

        // DataSet data_set("/Users/artelnics/Desktop/5_years_mortality_modified.csv", ";", true, false);
        /*
        const Index input_variables_number = data_set.get_variables_number(DataSet::VariableUse::Input);
        const Index target_variables_number = data_set.get_variables_number(DataSet::VariableUse::Target);
<<<<<<< HEAD
       
=======


>>>>>>> 575d26b6acd6f039338141494c32b9bf6ba190f9
        // Neural network
        /*
        const Index neurons_number = 30;
        
        NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification,
            { input_variables_number }, { neurons_number }, { target_variables_number });
<<<<<<< HEAD
        /*
=======

        // data_set.print();

>>>>>>> 575d26b6acd6f039338141494c32b9bf6ba190f9
        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
        // training_strategy.set_loss_method(TrainingStrategy::LossMethod::MINKOWSKI_ERROR);
        // training_strategy.set_loss_method(TrainingStrategy::LossMethod::WEIGHTED_SQUARED_ERROR);
        // training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);


        // training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::CONJUGATE_GRADIENT);
        // training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM); //The probabilistic layer hasn't got implemented the lm back propagation
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);
        // training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        // data_set.set(DataSet::SampleUse::Training);

        training_strategy.perform_training();

        //data_set.set(DataSet::SampleUse::Testing);

        GeneticAlgorithm genetic_algorithm(&training_strategy);

        genetic_algorithm.perform_input_selection();

        // GrowingInputs growing_inputs(&training_strategy);

        // growing_inputs.perform_input_selection();

        // ModelSelection model_selection(&training_strategy);

        // model_selection.set_inputs_selection_method(ModelSelection::InputsSelectionMethod::GROWING_INPUTS);

        // model_selection.perform_input_selection();

        // model_selection.perform_neurons_selection();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        // data_set.print();

        testing_analysis.print_binary_classification_tests();
        TestingAnalysis::RocAnalysis roc_analysis = testing_analysis.perform_roc_analysis();

        cout << "Area under the curve: " << roc_analysis.area_under_curve << endl << "Roc curve:\n" << roc_analysis.roc_curve << endl;

        cout << "Confidence limit: " << roc_analysis.confidence_limit << endl << "Optimal threshold: " << roc_analysis.optimal_threshold << endl;

<<<<<<< HEAD
        */
=======

>>>>>>> 575d26b6acd6f039338141494c32b9bf6ba190f9
        cout << "Good bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques SL
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
