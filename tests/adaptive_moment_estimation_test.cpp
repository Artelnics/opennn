#include "pch.h"

#include "../opennn/training_strategy.h"
#include "../opennn/adaptive_moment_estimation.h"
#include "../opennn/mean_squared_error.h"

#include "../opennn/transformer.h"
#include "../opennn/language_data_set.h"
#include "../opennn/cross_entropy_error_3d.h"


TEST(AdaptiveMomentEstimationTest, DefaultConstructor)
{
    AdaptiveMomentEstimation adaptive_moment_estimation;

    EXPECT_EQ(adaptive_moment_estimation.has_loss_index(), false);
}


TEST(AdaptiveMomentEstimationTest, GeneralConstructor)
{
    MeanSquaredError mean_squared_error;
    AdaptiveMomentEstimation adaptive_moment_estimation(&mean_squared_error);

    EXPECT_TRUE(adaptive_moment_estimation.has_loss_index());
}


TEST(AdaptiveMomentEstimationTest, TrainEmpty)
{
    AdaptiveMomentEstimation adaptive_moment_estimation;

    const TrainingResults training_results = adaptive_moment_estimation.perform_training();

    //EXPECT_TRUE(adaptive_moment_estimation.has_loss_index());
}


TEST(AdaptiveMomentEstimationTest, TrainApproximation)
{
    DataSet data_set(1, {1}, {1});
    data_set.set_data_constant(type(1));
    
    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {1}, {}, {1});
    neural_network.set_parameters_constant(type(1));

    TrainingStrategy training_strategy(&neural_network, &data_set);
  
    AdaptiveMomentEstimation* adaptive_moment_estimation = training_strategy.get_adaptive_moment_estimation();

    adaptive_moment_estimation->set_maximum_epochs_number(1);
    adaptive_moment_estimation->set_display(false);

    TrainingResults training_results = adaptive_moment_estimation->perform_training();

    EXPECT_LE(training_results.get_epochs_number(), 1);
}


TEST(AdaptiveMomentEstimationTest, TrainTransformer)
{
    const Index batch_samples_number = 1;

    const Index input_length = 2;
    const Index context_length = 3;
    const Index input_dimensions = 5;
    const Index context_dimension = 6;

    const Index depth = 4;
    const Index perceptron_depth = 6;
    const Index heads_number = 4;
    const Index layers_number = 1;
/*
    LanguageDataSet language_data_set;

    language_data_set.set_data_random_language_model(batch_samples_number,
        input_length,
        context_length,
        input_dimensions,
        context_dimension);

    language_data_set.set(DataSet::SampleUse::Training);

    Transformer transformer({ input_length,
                             context_length,
                             input_dimensions,
                             context_dimension,
                             depth,
                             perceptron_depth,
                             heads_number,
                             layers_number });

    type training_loss_goal = type(0.05);

    CrossEntropyError3D cross_entropy_error_3d(&transformer, &language_data_set);

    AdaptiveMomentEstimation adaptive_moment_estimation(&cross_entropy_error_3d);

    adaptive_moment_estimation.set_display(true);
    adaptive_moment_estimation.set_display_period(100);
    adaptive_moment_estimation.set_loss_goal(training_loss_goal);
    adaptive_moment_estimation.set_maximum_epochs_number(1000);
    adaptive_moment_estimation.set_maximum_time(1000.0);
//    const TrainingResults training_results = adaptive_moment_estimation.perform_training();

//    EXPECT_EQ(training_results.get_training_error() <= training_loss_goal);
*/
}

/*
void AdaptiveMomentEstimationTest::test_perform_training()
{
    type old_error = numeric_limits<float>::max();
    type error;

    // Test
    {
        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_random();

        neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {outputs_number});
        neural_network.set_parameters_constant(-1);

        adaptive_moment_estimation.set_maximum_epochs_number(1);
        
        training_results = adaptive_moment_estimation.perform_training();

        error = training_results.get_training_error();

        EXPECT_EQ(error < old_error);
        
        old_error = error;

        adaptive_moment_estimation.set_maximum_epochs_number(50);
        neural_network.set_parameters_constant(-1);
        
        training_results = adaptive_moment_estimation.perform_training();

        error = training_results.get_training_error();

        EXPECT_EQ(error - old_error < type(1e-2));
        
    }

    // Loss goal
    {

        samples_number = 1;
        inputs_number = 1;
        outputs_number = 1;

        data_set.set(samples_number, inputs_number, outputs_number);
        data_set.set_data_random();
        //data_set.set(DataSet::SampleUse::Testing);

        neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {outputs_number});
        neural_network.set_parameters_random();

        type training_loss_goal = type(0.05);

        //adaptive_moment_estimation.set_display(true);
        adaptive_moment_estimation.set_loss_goal(training_loss_goal);
        adaptive_moment_estimation.set_maximum_epochs_number(10000);
        adaptive_moment_estimation.set_maximum_time(1000.0);

        //for(Index i = 0; i < 100; i++)
        //{
            //data_set.set_data_random();
            //neural_network.set_parameters_random();

        training_results = adaptive_moment_estimation.perform_training();

        //EXPECT_EQ(training_results.get_training_error() <= training_loss_goal);
    //}

        EXPECT_EQ(training_results.get_training_error() <= training_loss_goal);
        
    }
}
*/
