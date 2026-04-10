#include "pch.h"
#include "../opennn/dataset.h"
#include "../opennn/standard_networks.h"
#include "../opennn/loss.h"
#include "../opennn/quasi_newton_method.h"

using namespace opennn;

TEST(QuasiNewtonMethodTest, DefaultConstructor)
{
    QuasiNewtonMethod quasi_newton_method;

    EXPECT_EQ(quasi_newton_method.get_loss() == nullptr, true);
}

TEST(QuasiNewtonMethodTest, GeneralConstructor)
{

    Loss loss;

    QuasiNewtonMethod quasi_newton_method(&loss);

    EXPECT_EQ(quasi_newton_method.get_loss() != nullptr, true);
}


TEST(QuasiNewtonMethodTest, BFGS_Update)
{
    // calculate_inverse_hessian is private in QuasiNewtonMethod, so this test
    // cannot directly call it. We test the overall training instead.

    const Index inputs_number = 1;
    const Index outputs_number = 1;
    const Index samples_number = 10;

    Dataset dataset(samples_number, { inputs_number }, { outputs_number });
    dataset.set_data_random();
    dataset.set_sample_roles("Training");

    ApproximationNetwork neural_network({ inputs_number }, {}, { outputs_number });

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::MeanSquaredError);
    QuasiNewtonMethod quasi_newton_method(&loss);
    quasi_newton_method.set_scaling();

    neural_network.set_parameters_random();

    VectorR gradient_k = loss.calculate_gradient();

    // Just verify gradient is computable and has correct size
    EXPECT_EQ(gradient_k.size(), neural_network.get_parameters().size());

    MatrixR numerical_inverse_hessian = loss.calculate_inverse_hessian();

    EXPECT_EQ(numerical_inverse_hessian.rows(), numerical_inverse_hessian.cols());

    for(Index i = 0; i < numerical_inverse_hessian.size(); ++i)
        EXPECT_FALSE(isnan(numerical_inverse_hessian(i)));
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

    Loss* loss_ptr = new Loss(&neural_network, &dataset);
    loss_ptr->set_error(Loss::Error::MeanSquaredError);
    quasi_newton_method.set_loss(loss_ptr);

    quasi_newton_method.set_scaling();

    quasi_newton_method.set_maximum_epochs(1);
    quasi_newton_method.set_display(false);

    TrainingResults training_results = quasi_newton_method.train();

    EXPECT_LE(training_results.get_epochs_number(), 1);

    type old_error = numeric_limits<type>::max();
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

    delete loss_ptr;
}
