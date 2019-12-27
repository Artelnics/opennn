//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M N I S T   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

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
        cout << "OpenNN. MNIST Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("../data/mnist_train.csv", ',', false);

        data_set.set_input();
        data_set.set_column_use(0, OpenNN::DataSet::VariableUse::Target);
        data_set.numeric_to_categorical(0);
        data_set.set_batch_instances_number(5);

        const Vector<size_t> inputs_dimensions({1, 28, 28});
        const Vector<size_t> targets_dimensions({10});
        data_set.set_input_variables_dimensions(inputs_dimensions);
        data_set.set_target_variables_dimensions(targets_dimensions);

        const size_t total_instances = 100;
        data_set.set_instances_uses((Vector<string>(total_instances, "Training").assemble(Vector<string>(60000 - total_instances, "Unused"))));
        data_set.split_instances_random(0.75, 0, 0.25);

        // Neural network

        const size_t outputs_number = 10;

        NeuralNetwork neural_network;

        // Scaling layer

        ScalingLayer* scaling_layer = new ScalingLayer(inputs_dimensions);
        neural_network.add_layer(scaling_layer);

        const Vector<size_t> scaling_layer_outputs_dimensions = scaling_layer->get_outputs_dimensions();

        // Convolutional layer 1

        ConvolutionalLayer* convolutional_layer_1 = new ConvolutionalLayer(scaling_layer_outputs_dimensions, {8, 5, 5});
        neural_network.add_layer(convolutional_layer_1);

        const Vector<size_t> convolutional_layer_1_outputs_dimensions = convolutional_layer_1->get_outputs_dimensions();

        // Pooling layer 1

        PoolingLayer* pooling_layer_1 = new PoolingLayer(convolutional_layer_1_outputs_dimensions);
        neural_network.add_layer(pooling_layer_1);

        const Vector<size_t> pooling_layer_1_outputs_dimensions = pooling_layer_1->get_outputs_dimensions();

        // Convolutional layer 2

        ConvolutionalLayer* convolutional_layer_2 = new ConvolutionalLayer(pooling_layer_1_outputs_dimensions, {4, 3, 3});
        neural_network.add_layer(convolutional_layer_2);

        const Vector<size_t> convolutional_layer_2_outputs_dimensions = convolutional_layer_2->get_outputs_dimensions();

        // Pooling layer 2

        PoolingLayer* pooling_layer_2 = new PoolingLayer(convolutional_layer_2_outputs_dimensions);
        neural_network.add_layer(pooling_layer_2);

        const Vector<size_t> pooling_layer_2_outputs_dimensions = pooling_layer_2->get_outputs_dimensions();

        // Convolutional layer 3

        ConvolutionalLayer* convolutional_layer_3 = new ConvolutionalLayer(pooling_layer_2_outputs_dimensions, {2, 3, 3});
        neural_network.add_layer(convolutional_layer_3);

        const Vector<size_t> convolutional_layer_3_outputs_dimensions = convolutional_layer_3->get_outputs_dimensions();

        // Pooling layer 3

        PoolingLayer* pooling_layer_3 = new PoolingLayer(convolutional_layer_3_outputs_dimensions);
        neural_network.add_layer(pooling_layer_3);

        const Vector<size_t> pooling_layer_3_outputs_dimensions = pooling_layer_3->get_outputs_dimensions();

        // Perceptron layer

        PerceptronLayer* perceptron_layer = new PerceptronLayer(pooling_layer_3_outputs_dimensions.calculate_product(), 18);
        neural_network.add_layer(perceptron_layer);

        const size_t perceptron_layer_outputs = perceptron_layer->get_neurons_number();

        // Probabilistic layer

        ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer_outputs, outputs_number);
        neural_network.add_layer(probabilistic_layer);

        neural_network.print_summary();

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);
        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
        training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        StochasticGradientDescent* sgd_pointer = training_strategy.get_stochastic_gradient_descent_pointer();

        sgd_pointer->set_minimum_loss_increase(1.0e-6);
        sgd_pointer->set_maximum_epochs_number(12);
        sgd_pointer->set_display_period(1);
        sgd_pointer->set_maximum_time(1800);

        const OptimizationAlgorithm::Results training_strategy_results = training_strategy.perform_training();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        Matrix<size_t> confusion = testing_analysis.calculate_confusion();

        cout << "\n\nConfusion matrix: \n" << endl << confusion << endl;
        cout << "\nAccuracy: " << (confusion.calculate_trace()/confusion.calculate_sum())*100 << " %" << endl << endl;

        // Save results
/*
        data_set.save("../data/data_set.xml");

        neural_network.save("../data/neural_network.xml");
//        neural_network.save_expression("../data/expression.txt");

        training_strategy.save("../data/training_strategy.xml");
        training_strategy_results.save("../data/training_strategy_results.dat");
*/       
        return 0;
    }
    catch(exception& e)
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
