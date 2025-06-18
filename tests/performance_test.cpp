#include "pch.h"


//using namespace opennn;

TEST(PerformanceTest, Rosenbrock)
{
    /*
    const Index samples_number = 100;
    const Index inputs_number = 10;
    const Index outputs_number = 1;
    const Index hidden_neurons_number = 10;

    Dataset dataset(samples_number, { inputs_number }, {outputs_number});

    dataset.set_data_rosenbrock();

    dataset.set(Dataset::SampleUse::Training);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation,
        { inputs_number }, { hidden_neurons_number }, { outputs_number });

    TrainingStrategy training_strategy(&neural_network, &dataset);
    
    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

    training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

    training_strategy.set_maximum_epochs_number(10);
    training_strategy.set_display_period(1);
    training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(1000);
    training_strategy.set_maximum_time(86400);

//    training_strategy.perform_training();

}


TEST(PerformanceTest, ImageClassification)
{
    
    const Index samples_number = 6;
    const Index image_height = 4;
    const Index image_width = 4;
    const Index channels = 3;
    const Index targets = 2;
    
    ImageDataSet image_data_set(samples_number, { image_height, image_width, channels }, { targets });
    
    //image_data_set.set_data_random();
    /*
    image_data_set.set(Dataset::SampleUse::Training);

    const dimensions complexity_dimensions = { 8 };

    NeuralNetwork neural_network(NeuralNetwork::ModelType::ImageClassification,
        image_data_set.get_input_dimensions(),
        complexity_dimensions,
        image_data_set.get_target_dimensions());

    TrainingStrategy training_strategy(&neural_network, &image_data_set);
   
    training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
    training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
    training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(512);
    training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(2);
    training_strategy.set_display_period(1);
    
    training_strategy.perform_training();
    */
}
