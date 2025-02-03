#include "pch.h"

#include "../opennn/mean_squared_error.h"
#include "../opennn/quasi_newton_method.h"

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
    const Index neurons_number = 1;

    DataSet data_set(samples_number, { inputs_number }, { outputs_number });
    data_set.set_data_random();

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, { inputs_number }, {}, { outputs_number });
    neural_network.set_parameters_constant(type(1));

    MeanSquaredError mean_squared_error(&neural_network, &data_set);

    QuasiNewtonMethod quasi_newton_method(&mean_squared_error);

    QuasiNewtonMethodData quasi_newton_method_data(&quasi_newton_method);

    quasi_newton_method.calculate_DFP_inverse_hessian(quasi_newton_method_data);

    EXPECT_EQ(are_equal(quasi_newton_method_data.inverse_hessian, inverse_hessian, type(1e-4)));
*/
}


TEST(QuasiNewtonMethodTest, BGFS)
{
    /*
    const Index samples_number = 1;
    const Index inputs_number = 1;
    const Index outputs_number = 1;
    const Index neurons_number = 1;

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, 
                                { inputs_number }, {}, { outputs_number });
    
    neural_network.set_parameters_constant(type(1));

    MeanSquaredError mean_squared_error(&neural_network);

    mean_squared_error.set_regularization_method(LossIndex::RegularizationMethod::L2);

    QuasiNewtonMethod quasi_newton_method(&mean_squared_error);

    QuasiNewtonMethodData quasi_newton_method_data(&quasi_newton_method);

    quasi_newton_method.calculate_BFGS_inverse_hessian(quasi_newton_method_data);

    EXPECT_EQ(are_equal(BFGS_inverse_hessian, inverse_hessian, type(1e-4)));
*/
}

TEST(QuasiNewtonMethodTest, Train)
{
/*
    type old_error = numeric_limits<float>::max();

    type error;

    // Test

    samples_number = 1;
    inputs_number = 1;
    outputs_number = 1;

    data_set.set(1, 1, 1);
    data_set.set_data_constant(type(1));

    neural_network.set(NeuralNetwork::ModelType::Approximation, { inputs_number }, {}, { outputs_number });
    neural_network.set_parameters_constant(type(1));

    quasi_newton_method.set_maximum_epochs_number(1);
    quasi_newton_method.set_display(false);
    training_results = quasi_newton_method.perform_training();

    EXPECT_EQ(training_results.get_epochs_number() <= 1);

    // Test

    data_set.set(1, 1, 1);
    data_set.set_data_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, { inputs_number }, {}, { outputs_number });
    neural_network.set_parameters_constant(-1);

    quasi_newton_method.set_maximum_epochs_number(1);

    training_results = quasi_newton_method.perform_training();
    error = training_results.get_training_error();

    EXPECT_EQ(error < old_error);

    // Test

    old_error = error;

    quasi_newton_method.set_maximum_epochs_number(2);
    neural_network.set_parameters_constant(-1);

    training_results = quasi_newton_method.perform_training();
    error = training_results.get_training_error();

    EXPECT_EQ(error <= old_error);

    // Loss goal

    neural_network.set_parameters_constant(type(-1));

    type training_loss_goal = type(0.1);

    quasi_newton_method.set_loss_goal(training_loss_goal);
    quasi_newton_method.set_minimum_loss_decrease(0.0);
    quasi_newton_method.set_maximum_epochs_number(1000);
    quasi_newton_method.set_maximum_time(1000.0);

    training_results = quasi_newton_method.perform_training();

    //EXPECT_EQ(training_results.get_loss() <= training_loss_goal);

    // Minimum loss decrease

    neural_network.set_parameters_constant(type(-1));

    type minimum_loss_decrease = type(0.1);

    quasi_newton_method.set_loss_goal(type(0));
    quasi_newton_method.set_minimum_loss_decrease(minimum_loss_decrease);
    quasi_newton_method.set_maximum_epochs_number(1000);
    quasi_newton_method.set_maximum_time(1000.0);

    training_results = quasi_newton_method.perform_training();

    //EXPECT_EQ(training_results.get_loss_decrease() <= minimum_loss_decrease);
*/
}

/*


void QuasiNewtonMethodTest::test_calculate_inverse_hessian_approximation()
{
    Tensor<type, 1> old_parameters;
    Tensor<type, 1> old_gradient;
    Tensor<type, 2> old_inverse_hessian;

    Tensor<type, 1> parameters;
    Tensor<type, 1> gradient;
    Tensor<type, 2> inverse_hessian;

    Tensor<type, 2> inverse_hessian_approximation;

    // Test

    samples_number = 1;
    inputs_number = 1;
    outputs_number = 1;

    data_set.set(samples_number, inputs_number, outputs_number);
    data_set.set_data_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {}, {outputs_number});

    quasi_newton_method.set_inverse_hessian_approximation_method(QuasiNewtonMethod::InverseHessianApproximationMethod::DFP);

    neural_network.set_parameters_constant(type(1));

    quasi_newton_method.calculate_inverse_hessian_approximation(quasi_newton_method_data);

    EXPECT_EQ(inverse_hessian_approximation == inverse_hessian);

    quasi_newton_method.set_inverse_hessian_approximation_method(QuasiNewtonMethod::InverseHessianApproximationMethod::DFP);

    neural_network.set_parameters_constant(type(1));

    neural_network.set_parameters_constant(type(-0.5));

    quasi_newton_method.calculate_inverse_hessian_approximation(quasi_newton_method_data);

    EXPECT_EQ(inverse_hessian_approximation == inverse_hessian);

    // Test

    quasi_newton_method.calculate_inverse_hessian_approximation(quasi_newton_method_data);
}
*/
