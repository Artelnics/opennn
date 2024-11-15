#include "pch.h"

#include <iostream>
#include <string>
#include <exception>

#include "../opennn/opennn.h"

using namespace opennn;

TEST(PerformanceTest, Rosenbrock)
{

/*
    const Index samples_number = 1000000;
    const Index inputs_number = 1000;
    const Index outputs_number = 1;
    const Index hidden_neurons_number = 1000;

    DataSet data_set;

    data_set.generate_Rosenbrock_data(samples_number, inputs_number + outputs_number);

    data_set.set(DataSet::SampleUse::Training);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation,
        { inputs_number }, { hidden_neurons_number }, { outputs_number });

    TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

    training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

    training_strategy.set_maximum_epochs_number(10);
    training_strategy.set_display_period(1);
    training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(1000);
    training_strategy.set_maximum_time(86400);

    training_strategy.perform_training();
*/
}


TEST(PerformanceTest, ImageClassification)
{
    const Index samples_number = 6;
    const Index image_height = 4;
    const Index image_width = 4;
    const Index channels = 3;
    const Index targets = 2;
    
    ImageDataSet image_data_set(samples_number, { image_height, image_width, channels }, { targets });
    image_data_set.set_image_data_random();
    image_data_set.set(DataSet::SampleUse::Training);

    //ImageDataSet image_data_set;
    //image_data_set.set_data_source_path("C:/binary_mnist");
    //image_data_set.read_bmp();
    
    const dimensions complexity_dimensions = { 8 };

    NeuralNetwork neural_network(NeuralNetwork::ModelType::ImageClassification,
        image_data_set.get_input_dimensions(),
        complexity_dimensions,
        image_data_set.get_target_dimensions());

    const Index batch_samples_number = 6;
    ForwardPropagation training_forward_propagation(batch_samples_number, &neural_network);

    //TrainingStrategy training_strategy(&neural_network, &image_data_set);
   
    //training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
    //training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
    //training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
    //training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(512);
    //training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(2);
    //training_strategy.set_display_period(1);
    
    //training_strategy.perform_training(); 
    
    EXPECT_EQ(1, 1);
}

