//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
<<<<<<< HEAD
//   BLANK PROJECT
=======
//   B L A N K   A P P L I C A T I O N
>>>>>>> ef5b0587e160bd6ea29d14f5226356124d1ff546
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

<<<<<<< HEAD
=======
#include <cstring>
>>>>>>> ef5b0587e160bd6ea29d14f5226356124d1ff546
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
<<<<<<< HEAD
#include <cstring>
#include <time.h>
#include <stdio.h>
#include <vector>

// OpenNN includes

#include "../../opennn/opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace Eigen;

int main()
{
    //DataSet class

    DataSet data_set;

    data_set.set_data_file_name("../data/iris_plant.csv");
    data_set.set_separator("Space");
    data_set.read_csv();

    data_set.set_column_name(0, "sepal_length");
    data_set.set_column_name(1, "sepal_width");
    data_set.set_column_name(2, "petal_length");
    data_set.set_column_name(3, "petal_width");
    data_set.set_column_name(4, "iris_type");

    //data_set.split_samples_random(type(0.7), type(0.1), type(0.2));
    data_set.split_samples_sequential(type(0.7), type(0.1), type(0.2));

    const Tensor<Descriptives, 1> inputs_descriptives = data_set.scale_input_variables();
    //cout << inputs_descriptives << endl;

    const Tensor<Correlation, 2> input_target_columns_correlations = data_set.calculate_input_target_columns_correlations();
    data_set.print_input_target_columns_correlations();

    //NeuralNetwork class

    NeuralNetwork neural_network(NeuralNetwork::Classification, {4, 6, 3});

    const Tensor<string, 1> inputs_names = data_set.get_input_variables_names();
    neural_network.set_inputs_names(inputs_names);
    const Tensor<string, 1> targets_names = data_set.get_target_variables_names();
    neural_network.set_outputs_names(targets_names);

    Index inputs_number = data_set.get_input_variables_number();
    neural_network.set_inputs_number(inputs_number);

    ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
    scaling_layer_pointer->set_descriptives(inputs_descriptives);
    scaling_layer_pointer->set_scalers(NoScaling);

    ProbabilisticLayer* probabilistic_layer_pointer = neural_network.get_probabilistic_layer_pointer();
    probabilistic_layer_pointer->set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);

    //TrainingStrategy class

    TrainingStrategy training_strategy(&neural_network,&data_set);

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);
    training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);

    QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();
    quasi_Newton_method_pointer->set_minimum_loss_decrease(type(1.0e-6));
    quasi_Newton_method_pointer->set_loss_goal(type(1.0e-3));

    NormalizedSquaredError* normalized_squared_error_pointer = training_strategy.get_normalized_squared_error_pointer();
    normalized_squared_error_pointer->set_normalization_coefficient();

    TrainingResults results = training_strategy.perform_training();

    //ModelSelection class

    //ModelSelection model_selection(&training_strategy);

    //model_selection.set_inputs_selection_method(ModelSelection::GENETIC_ALGORITHM);
    //model_selection.set_neurons_selection_method(ModelSelection::GROWING_NEURONS);
    //GeneticAlgorithm* genetic_algorithm = model_selection.get_genetic_algorithm_pointer();
    //genetic_algorithm->set_individuals_number(100);
    //GrowingNeurons* growing_neurons = model_selection.get_growing_neurons_pointer();
    //growing_neurons ->set_maximum_selection_failures(3);

    //model_selection.perform_inputs_selection();
    //model_selection.perform_neurons_selection();

    //model_selection.save("model_selection.xml");

    //TestingAnalysis class

    TestingAnalysis testing_analysis(&neural_network, &data_set);

    Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

    Tensor<type, 1> confusion_matrix = testing_analysis.calculate_testing_errors();

    return 0;
=======
#include <time.h>

// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace Eigen;

int main()
{
    try
    {
        cout << "OpenNN. Blank application." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        DataSet ds("C:/Users/Usuario/Documents/Waste_monthly.csv", ',', true);

        ds.set_lags_number(2);
        ds.set_steps_ahead_number(1);

        ds.transform_time_series();

        ds.split_samples_sequential();

        const Index columns_number = ds.get_columns_number();

        ds.set_column_use(columns_number-1, DataSet::VariableUse::UnusedVariable);
        ds.set_column_use(columns_number-2, DataSet::VariableUse::UnusedVariable);
        ds.set_column_use(columns_number-3, DataSet::VariableUse::UnusedVariable);

        const Index inputs_number = ds.get_input_variables_number();
        const Index targets_number = ds.get_target_variables_number();

        const Index neurons_number = 3;

        ScalingLayer sl(inputs_number);

        LongShortTermMemoryLayer lstm(inputs_number, neurons_number);
        lstm.set_activation_function(LongShortTermMemoryLayer::ActivationFunction::Linear);
        lstm.set_recurrent_activation_function(LongShortTermMemoryLayer::ActivationFunction::HyperbolicTangent);
        lstm.set_timesteps(2);

        PerceptronLayer pl(neurons_number, targets_number);
        pl.set_activation_function(PerceptronLayer::ActivationFunction::Linear);

        UnscalingLayer ul(inputs_number);

        NeuralNetwork nn;

        nn.add_layer(&sl);
        nn.add_layer(&lstm);
        nn.add_layer(&pl);
        nn.add_layer(&ul);

        TrainingStrategy ts(&nn, &ds);

        ts.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        ts.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

        ts.perform_training();

        cout << "outputs: " << endl << nn.calculate_outputs(ds.get_input_data()) << endl;



        cout << "Good bye!" << endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;
>>>>>>> ef5b0587e160bd6ea29d14f5226356124d1ff546

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
