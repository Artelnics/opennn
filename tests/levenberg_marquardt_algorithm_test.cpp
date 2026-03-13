#include "pch.h"

#include "../opennn/dense_layer.h"
#include "../opennn/mean_squared_error.h"
#include "../opennn/levenberg_marquardt_algorithm.h"

using namespace opennn;

TEST(LevenbergMarquardtAlgorithmTest, DefaultConstructor)
{
    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm;

    EXPECT_EQ(levenberg_marquardt_algorithm.has_loss(), false);
}


TEST(LevenbergMarquardtAlgorithmTest, GeneralConstructor)
{
    MeanSquaredError mean_squared_error;

    LevenbergMarquardtAlgorithm levenberg_marquardt_algorithm(&mean_squared_error);

    EXPECT_EQ(levenberg_marquardt_algorithm.has_loss(), true);
}


TEST(LevenbergMarquardtAlgorithmTest, Train)
{
    Dataset dataset(10, {1}, {1});
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense<2>>(Shape{1}, Shape{1}, "Linear"));

    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    LevenbergMarquardtAlgorithm lm(&mean_squared_error);
    lm.set_display(false);

    // Test 1: max 1 epoch
    lm.set_maximum_epochs(1);
    TrainingResults results = lm.train();
    EXPECT_LE(results.get_epochs_number(), 1);

    // Test 2: More Epoch -> Less error
    lm.set_maximum_epochs(100);
    neural_network.set_parameters_random();
    TrainingResults results1 = lm.train();

    neural_network.set_parameters_random();
    lm.set_maximum_epochs(200);
    TrainingResults results2 = lm.train();

    EXPECT_LE(results2.get_training_error(), results1.get_training_error() + type(1e-3));

    // Test 3: loss goal
    neural_network.set_parameters_random();
    const type loss_goal = type(0.5);
    lm.set_loss_goal(loss_goal);
    lm.set_maximum_epochs(1000);
    TrainingResults results3 = lm.train();
    EXPECT_TRUE(results3.get_training_error() <= loss_goal
                || results3.get_epochs_number() == 1000);

<<<<<<< Updated upstream
    EXPECT_EQ(error < old_error);

    // Test

    old_error = error;

    levenberg_marquardt_algorithm.set_maximum_epochs(2);
    neural_network.set_parameters_constant(-1);

    training_results = levenberg_marquardt_algorithm.train();
    error = training_results.get_training_error();

    EXPECT_EQ(error <= old_error);

    // Loss goal

    neural_network.set_parameters_constant(type(-1));

    type training_loss_goal = type(0.1);

    levenberg_marquardt_algorithm.set_loss_goal(training_loss_goal);
    levenberg_marquardt_algorithm.set_minimum_loss_decrease(0.0);
    levenberg_marquardt_algorithm.set_maximum_epochs(1000);
    levenberg_marquardt_algorithm.set_maximum_time(1000.0);

    training_results = levenberg_marquardt_algorithm.train();

    EXPECT_EQ(training_results.get_training_error() <= training_loss_goal);

    // Minimum loss decrease

    neural_network.set_parameters_constant(type(-1));

    type minimum_loss_decrease = type(0.1);

    levenberg_marquardt_algorithm.set_loss_goal(type(0));
    levenberg_marquardt_algorithm.set_minimum_loss_decrease(minimum_loss_decrease);
    levenberg_marquardt_algorithm.set_maximum_epochs(1000);
    levenberg_marquardt_algorithm.set_maximum_time(1000.0);

    training_results = levenberg_marquardt_algorithm.train();

    EXPECT_EQ(levenberg_marquardt_algorithm.get_minimum_loss_decrease() <= minimum_loss_decrease);

    EXPECT_EQ(levenberg_marquardt_algorithm.has_loss(), true);
*/
}
=======
    // Test 4: has_loss_index
    EXPECT_EQ(lm.has_loss_index(), true);
}
>>>>>>> Stashed changes
