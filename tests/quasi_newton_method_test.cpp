#include "pch.h"
#include "../opennn/standard_networks.h"
#include "../opennn/mean_squared_error.h"
#include "../opennn/quasi_newton_method.h"

using namespace opennn;

TEST(QuasiNewtonMethodTest, DefaultConstructor)
{
    QuasiNewtonMethod quasi_newton_method;

    EXPECT_EQ(quasi_newton_method.has_loss_index(), false);
}

TEST(QuasiNewtonMethodTest, GeneralConstructor)
{

    MeanSquaredError mean_squared_error;

    QuasiNewtonMethod quasi_newton_method(&mean_squared_error);

    EXPECT_EQ(quasi_newton_method.has_loss_index(), true);
}


TEST(QuasiNewtonMethodTest, DFP)
{
    /*
    const Index samples_number = 1;
    const Index inputs_number = 1;
    const Index outputs_number = 1;

    Dataset dataset(samples_number, { inputs_number }, { outputs_number });
    dataset.set_data_random();

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, { inputs_number }, {}, { outputs_number });
    neural_network.set_parameters_constant(type(1));

    MeanSquaredError mean_squared_error(&neural_network, &dataset);

    QuasiNewtonMethod quasi_newton_method(&mean_squared_error);

    QuasiNewtonMethodData quasi_newton_method_data(&quasi_newton_method);

    quasi_newton_method.calculate_DFP_inverse_hessian(quasi_newton_method_data);

    const Tensor<type, 2> inverse_hessian = mean_squared_error.calculate_inverse_hessian();

    EXPECT_EQ(are_equal(quasi_newton_method_data.inverse_hessian, inverse_hessian, type(1e-4)), true);
*/
}


TEST(QuasiNewtonMethodTest, BGFS)
{
    /*
    const Index samples_number = 1;
    const Index inputs_number = 1;
    const Index outputs_number = 1;

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {outputs_number});
    neural_network.set_parameters_constant(1);

    Dataset dataset(samples_number, {inputs_number}, {outputs_number});
    dataset.set_data_random();

    MeanSquaredError mean_squared_error(&neural_network, &dataset);
    mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::L2);

    QuasiNewtonMethod quasi_newton_method(&mean_squared_error);

    QuasiNewtonMethodData quasi_newton_method_data(&quasi_newton_method);

    quasi_newton_method.calculate_BFGS_inverse_hessian(quasi_newton_method_data);

    const Tensor<type, 2> inverse_hessian = mean_squared_error.calculate_inverse_hessian();

    EXPECT_EQ(are_equal(quasi_newton_method_data.inverse_hessian, inverse_hessian, type(1e-4)), true);
    */
}


TEST(QuasiNewtonMethodTest, Train)
{
    const Index samples_number = 1;
    const Index inputs_number = 1;
    const Index outputs_number = 1;

    //Test

    Dataset dataset(samples_number, {inputs_number}, {outputs_number});
    dataset.set_data_constant(1);

    ApproximationNetwork neural_network({inputs_number}, {}, {outputs_number});
    neural_network.set_parameters_random();

    QuasiNewtonMethod quasi_newton_method;
    quasi_newton_method.set_maximum_epochs_number(1);
    quasi_newton_method.set_display(false);

    TrainingResults training_results = quasi_newton_method.perform_training();

    EXPECT_LE(training_results.get_epochs_number(), 1);

    // Test

    type old_error = numeric_limits<float>::max();
    type error;

    dataset.set_data_random();

    neural_network.set_parameters_random();

    training_results = quasi_newton_method.perform_training();
    error = training_results.get_training_error();

    EXPECT_LT(error, old_error);

    // Test

    old_error = error;

    quasi_newton_method.set_maximum_epochs_number(2);

    training_results = quasi_newton_method.perform_training();
    error = training_results.get_training_error();

    EXPECT_LE(error, old_error);

    //Loss goal

    neural_network.set_parameters_random();

    type training_loss_goal = type(0.1);

    quasi_newton_method.set_loss_goal(training_loss_goal);
    quasi_newton_method.set_minimum_loss_decrease(0.0);
    quasi_newton_method.set_maximum_epochs_number(1000);
    quasi_newton_method.set_maximum_time(1000.0);

    training_results = quasi_newton_method.perform_training();
/*
    EXPECT_LE(training_results.get_loss(), training_loss_goal);

    // Minimum loss decrease

    neural_network.set_parameters_constant(type(-1));

    type minimum_loss_decrease = type(0.1);

    quasi_newton_method.set_loss_goal(type(0));
    quasi_newton_method.set_minimum_loss_decrease(minimum_loss_decrease);
    quasi_newton_method.set_maximum_epochs_number(1000);
    quasi_newton_method.set_maximum_time(1000.0);

    training_results = quasi_newton_method.perform_training();

    EXPECT_LE(training_results.get_loss_decrease(), minimum_loss_decrease);
*/
}

