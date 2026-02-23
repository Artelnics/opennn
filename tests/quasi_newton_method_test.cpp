#include "pch.h"
#include "../opennn/dataset.h"
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


TEST(QuasiNewtonMethodTest, BFGS_Update)
{
    const Index inputs_number = 1;
    const Index outputs_number = 1;
    const Index samples_number = 10;

    Dataset dataset(samples_number, { inputs_number }, { outputs_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork neural_network({ inputs_number }, {}, { outputs_number });

    MeanSquaredError mean_squared_error(&neural_network, &dataset);
    QuasiNewtonMethod quasi_newton_method(&mean_squared_error);
    quasi_newton_method.set_scaling();
    QuasiNewtonMethodData optimization_data(&quasi_newton_method);

    neural_network.set_parameters_random();

    Tensor1 parameters_k = neural_network.get_parameters();
    Tensor1 gradient_k = mean_squared_error.calculate_gradient();

    Tensor1 parameters_next = parameters_k;
    for(Index i=0; i < parameters_next.size(); ++i) parameters_next(i) += 0.01;

    neural_network.set_parameters(parameters_next);
    Tensor1 gradient_next = mean_squared_error.calculate_gradient();

    optimization_data.parameter_differences = parameters_next - parameters_k;
    optimization_data.gradient_difference = gradient_next - gradient_k;

    set_identity(optimization_data.old_inverse_hessian);

    quasi_newton_method.calculate_inverse_hessian(optimization_data);

    Tensor2 numerical_inverse_hessian = mean_squared_error.calculate_inverse_hessian();

    EXPECT_EQ(optimization_data.inverse_hessian.dimension(0), numerical_inverse_hessian.dimension(0));

    for(Index i = 0; i < optimization_data.inverse_hessian.size(); ++i)
        EXPECT_FALSE(isnan(optimization_data.inverse_hessian(i)));
}


TEST(QuasiNewtonMethodTest, Train)
{
    const Index samples_number = 1;
    const Index inputs_number = 1;
    const Index outputs_number = 1;

    Dataset dataset(samples_number, {inputs_number}, {outputs_number});
    dataset.set_data_constant(1);

    ApproximationNetwork neural_network({inputs_number}, {}, {outputs_number});
    neural_network.set_parameters_random();

    QuasiNewtonMethod quasi_newton_method;

    quasi_newton_method.set_loss_index(new MeanSquaredError(&neural_network, &dataset));

    quasi_newton_method.set_scaling();

    quasi_newton_method.set_maximum_epochs(1);
    quasi_newton_method.set_display(false);

    TrainingResults training_results = quasi_newton_method.train();

    EXPECT_LE(training_results.get_epochs_number(), 1);

    type old_error = numeric_limits<float>::max();
    type error;

    dataset.set_data_random();

    quasi_newton_method.set_scaling();

    neural_network.set_parameters_random();

    training_results = quasi_newton_method.train();
    error = training_results.get_training_error();

    EXPECT_LT(error, old_error);

    old_error = error;

    quasi_newton_method.set_maximum_epochs(2);

    training_results = quasi_newton_method.train();
    error = training_results.get_training_error();

    EXPECT_LE(error, old_error);

    neural_network.set_parameters_random();

    type training_loss_goal = type(0.1);

    quasi_newton_method.set_loss_goal(training_loss_goal);
    quasi_newton_method.set_minimum_loss_decrease(0.0);
    quasi_newton_method.set_maximum_epochs(1000);
    quasi_newton_method.set_maximum_time(1000.0);

    training_results = quasi_newton_method.train();

    EXPECT_LE(training_results.loss, training_loss_goal);

    type minimum_loss_decrease = type(0.1);

    quasi_newton_method.set_loss_goal(type(0));
    quasi_newton_method.set_minimum_loss_decrease(minimum_loss_decrease);
    quasi_newton_method.set_maximum_epochs(1000);
    quasi_newton_method.set_maximum_time(1000.0);

    training_results = quasi_newton_method.train();

    EXPECT_LE(training_results.loss_decrease, minimum_loss_decrease);
}
