//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Q U A S I - N E W T O N   M E T H O D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "loss.h"
#include "quasi_newton_method.h"
#include "batch.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

QuasiNewtonMethod::QuasiNewtonMethod(Loss* new_loss)
    : Optimizer(new_loss)
{
    set_default();
}

void QuasiNewtonMethod::set_default()
{
    name = "QuasiNewtonMethod";

    // Stopping criteria

    minimum_loss_decrease = 0.0f;
    training_loss_goal = 0.0f;
    maximum_validation_failures = 1000;

    maximum_epochs = 1000;
    maximum_time = 3600.0f;

    // UTILITIES

    display = true;
    display_period = 10;
}

void QuasiNewtonMethod::calculate_inverse_hessian(OptimizerData& optimization_data) const
{
    const Index parameters_number = optimization_data.views[ParameterDifferences].size();

    VectorMap parameter_differences = optimization_data.views[ParameterDifferences].as_vector();
    VectorMap gradient_difference = optimization_data.views[GradientDifference].as_vector();

    VectorMap old_inverse_hessian_dot_gradient_difference(
        optimization_data.views[OldInverseHessianDotGradientDifference].as<float>(), parameters_number);

    MatrixMap old_inverse_hessian = optimization_data.views[OldInverseHessian].as_matrix();
    MatrixMap inverse_hessian     = optimization_data.views[InverseHessian].as_matrix();

    VectorMap bfgs = optimization_data.views[BFGS].as_vector();

    const float parameters_difference_dot_gradient_difference = parameter_differences.dot(gradient_difference);

    old_inverse_hessian_dot_gradient_difference.noalias() = old_inverse_hessian * gradient_difference;

    const float gradient_dot_hessian_dot_gradient = gradient_difference.dot(old_inverse_hessian_dot_gradient_difference);

    bfgs = (parameter_differences / parameters_difference_dot_gradient_difference)
           - (old_inverse_hessian_dot_gradient_difference / gradient_dot_hessian_dot_gradient);

    inverse_hessian = old_inverse_hessian;

    inverse_hessian.selfadjointView<Lower>().rankUpdate(
        parameter_differences, 1.0f / parameters_difference_dot_gradient_difference);

    inverse_hessian.selfadjointView<Lower>().rankUpdate(
        old_inverse_hessian_dot_gradient_difference, -1.0f / gradient_dot_hessian_dot_gradient);

    inverse_hessian.selfadjointView<Lower>().rankUpdate(
        bfgs, gradient_dot_hessian_dot_gradient);

    inverse_hessian.triangularView<Upper>() = inverse_hessian.triangularView<Lower>().transpose();
}

void QuasiNewtonMethod::update_parameters(const Batch& batch,
                                          ForwardPropagation& forward_propagation,
                                          BackPropagation& back_propagation,
                                          OptimizerData& optimization_data)
{
    NeuralNetwork* neural_network = forward_propagation.neural_network;

    VectorMap parameters(neural_network->get_parameters_data(),
                         neural_network->get_parameters_size());
    VectorMap gradient(back_propagation.gradient.as<float>(),
                       back_propagation.gradient.size_in_floats());

    VectorMap old_parameters = optimization_data.views[OldParameters].as_vector();
    VectorMap parameter_differences = optimization_data.views[ParameterDifferences].as_vector();
    VectorMap parameter_updates = optimization_data.views[ParameterUpdates].as_vector();

    VectorMap old_gradient = optimization_data.views[OldGradient].as_vector();
    VectorMap gradient_difference = optimization_data.views[GradientDifference].as_vector();

    VectorR& training_direction = optimization_data.training_direction;
    MatrixMap inverse_hessian = optimization_data.views[InverseHessian].as_matrix();

    parameter_differences = parameters - old_parameters;
    gradient_difference = gradient - old_gradient;

    old_parameters = parameters;

    if (parameter_differences.isZero() || gradient_difference.isZero())
        inverse_hessian.setIdentity();
    else
        calculate_inverse_hessian(optimization_data);

    training_direction.noalias() = -(inverse_hessian.selfadjointView<Lower>() * gradient);

    training_slope = gradient.dot(training_direction);

    if (training_slope >= 0.0f)
        training_direction = -gradient;

    optimization_data.initial_learning_rate = (old_learning_rate > 0.0f)
        ? old_learning_rate
        : first_learning_rate;

    tie(learning_rate, back_propagation.loss) = calculate_directional_point(
        batch,
        forward_propagation,
        back_propagation,
        optimization_data,
        back_propagation.loss);

    if (abs(learning_rate) > 0.0f)
    {
        parameter_updates = training_direction * learning_rate;
    }
    else
    {
        parameter_updates = (gradient.array().abs() >= EPSILON)
                                .select(-gradient.array().sign() * EPSILON, 0.0f);
        learning_rate = optimization_data.initial_learning_rate;
    }
    parameters += parameter_updates;

    old_gradient = gradient;
    swap(optimization_data.views[InverseHessian], optimization_data.views[OldInverseHessian]);
    old_learning_rate = learning_rate;
}

TrainingResults QuasiNewtonMethod::train()
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    if (neural_network->is_gpu())
        throw runtime_error("QuasiNewtonMethod does not support GPU training: "
                            "its update path maps device pointers as host memory. "
                            "Use AdaptiveMomentEstimation or StochasticGradientDescent on GPU.");

    TrainingResults results(maximum_epochs + 1);

    if (display) cout << "Training with quasi-Newton method..." << "\n";

    // Dataset

    const Dataset* dataset = loss->get_dataset();

    const bool has_validation = dataset->has_validation();

    const Index training_samples_number = dataset->get_samples_number("Training");

    const Index validation_samples_number = dataset->get_samples_number("Validation");

    const vector<Index> training_sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> validation_sample_indices = dataset->get_sample_indices("Validation");

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    // Neural network

    ForwardPropagation training_forward_propagation(training_samples_number, neural_network);

    unique_ptr<ForwardPropagation> validation_forward_propagation;

    if (has_validation && validation_samples_number != training_samples_number)
        validation_forward_propagation = make_unique<ForwardPropagation>(
            validation_samples_number, neural_network);

    ForwardPropagation* validation_fp = has_validation
        ? (validation_forward_propagation ? validation_forward_propagation.get() : &training_forward_propagation)
        : nullptr;

    set_names();

    set_scaling();
    Batch training_batch(training_samples_number, dataset, neural_network->get_config());
    training_batch.fill(training_sample_indices, input_feature_indices, {}, target_feature_indices, true);

    Batch validation_batch(validation_samples_number, dataset, neural_network->get_config());
    validation_batch.fill(validation_sample_indices, input_feature_indices, {}, target_feature_indices, /*is_training=*/false);

    // Loss

    loss->set_normalization_coefficient();

    BackPropagation training_back_propagation(training_samples_number, loss);

    BackPropagation validation_back_propagation(validation_samples_number, loss);

    // Optimization algorithm

    bool stop_training = false;

    Index validation_failures = 0;

    float old_loss = 0.0f;
    float loss_decrease = MAX;

    time_t beginning_time;
    time(&beginning_time);
    float elapsed_time;

    const Index parameters_number = neural_network->get_parameters_size();

    OptimizerData optimization_data;
    optimization_data.set({
        Shape{parameters_number},                 // OldParameters
        Shape{parameters_number},                 // ParameterDifferences
        Shape{parameters_number},                 // ParameterUpdates
        Shape{parameters_number},                 // OldGradient
        Shape{parameters_number},                 // GradientDifference
        Shape{parameters_number},                 // OldInverseHessianDotGradientDifference
        Shape{parameters_number},                 // BFGS
        Shape{parameters_number, parameters_number},  // InverseHessian
        Shape{parameters_number, parameters_number}   // OldInverseHessian
    });

    optimization_data.potential_parameters.resize(parameters_number);
    optimization_data.training_direction.resize(parameters_number);

    // Initialize OldParameters <- current parameters
    optimization_data.views[OldParameters].as_vector() =
        VectorMap(neural_network->get_parameters_data(), parameters_number);

    // Initialize InverseHessian and OldInverseHessian to identity
    optimization_data.views[InverseHessian].as_matrix().setIdentity();
    optimization_data.views[OldInverseHessian].as_matrix().setIdentity();

    // Main loop

    for (Index epoch = 0; epoch <= maximum_epochs; ++epoch)
    {
        if (should_display(epoch)) cout << "Epoch: " << epoch << "\n";

        neural_network->forward_propagate(training_batch.get_inputs(),
                                          training_forward_propagation,
                                          true);

        // Loss

        loss->back_propagate(training_batch,
                             training_forward_propagation,
                             training_back_propagation);

        results.training_error_history(epoch) = training_back_propagation.error;

        // Update parameters

        update_parameters(training_batch,
                          training_forward_propagation,
                          training_back_propagation,
                          optimization_data);

        // Validation error

        if (has_validation)
        {
            neural_network->forward_propagate(validation_batch.get_inputs(),
                                              *validation_fp,
                                              false);

            // Loss

            const Loss::EvaluationResult eval = loss->calculate_error(validation_batch,
                                                                       *validation_fp);
            validation_back_propagation.error = eval.error;
            validation_back_propagation.accuracy = eval.accuracy;
            validation_back_propagation.active_tokens_count = eval.active_tokens_count;

            results.validation_error_history(epoch) = validation_back_propagation.error;

            if (epoch != 0
                && validation_back_propagation.error > results.validation_error_history(epoch-1))
                ++validation_failures;
        }

        elapsed_time = get_elapsed_time(beginning_time);

        if (should_display(epoch))
        {
            cout << "Training error: " << training_back_propagation.error << "\n";
            if (has_validation) cout << "Validation error: " << validation_back_propagation.error << "\n";
            cout << "Learning rate: " << learning_rate << "\n";
            cout << "Elapsed time: " << get_time(elapsed_time) << "\n";
        }

        if (epoch != 0) loss_decrease = old_loss - training_back_propagation.loss;

        old_loss = training_back_propagation.loss;

        if (loss_decrease < minimum_loss_decrease)
        {
            if (display) cout << "Epoch " << epoch << "\nMinimum loss decrease reached: " << loss_decrease << "\n";
            results.stopping_condition = StoppingCondition::MinimumLossDecrease;
            stop_training = true;
        }
        else
        {
            stop_training = check_stopping_condition(results, epoch, elapsed_time,
                                                      results.training_error_history(epoch),
                                                      validation_failures);
        }

        if (stop_training)
        {
            results.loss = training_back_propagation.loss;
            results.loss_decrease = loss_decrease;
            results.validation_failures = validation_failures;
            results.resize_training_error_history(epoch+1);
            results.resize_validation_error_history(has_validation ? epoch + 1 : 0);
            results.elapsed_time = get_time(elapsed_time);

            break;
        }

    }

    set_unscaling();

    if (display) results.print();

    return results;
}

void QuasiNewtonMethod::to_JSON(JsonWriter& printer) const
{
    printer.open_element("QuasiNewtonMethod");

    add_json_field(printer, "MinimumLossDecrease", to_string(minimum_loss_decrease));
    write_common_json(printer);

    printer.close_element();
}

void QuasiNewtonMethod::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "QuasiNewtonMethod");

    set_minimum_loss_decrease(read_json_type(root_element, "MinimumLossDecrease"));
    read_common_json(root_element);
}

pair<float, float> QuasiNewtonMethod::calculate_directional_point(
    const Batch& batch,
    ForwardPropagation& forward_propagation,
    BackPropagation& back_propagation,
    OptimizerData& optimization_data,
    float current_loss)
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    float alpha = 1.0f;
    const float rho = 0.5f;
    const float armijo_constant = 1e-4f;
    const float previous_error = back_propagation.error;
    const float previous_regularization = back_propagation.regularization;

    Map<const VectorR, AlignedMax> parameters(neural_network->get_parameters_data(),
                                               neural_network->get_parameters_size());
    const VectorR& training_direction = optimization_data.training_direction;
    VectorR& potential_parameters = optimization_data.potential_parameters;

    for (int i = 0; i < 20; ++i)
    {
        potential_parameters = parameters + training_direction * alpha;

        neural_network->forward_propagate(batch.get_inputs(), potential_parameters, forward_propagation);
        const Loss::EvaluationResult eval = loss->calculate_error(batch, forward_propagation);
        const float candidate_regularization = loss->calculate_regularization(potential_parameters);
        const float new_loss = eval.error + candidate_regularization;

        if (new_loss <= current_loss + armijo_constant * alpha * training_slope)
        {
            back_propagation.error = eval.error;
            back_propagation.regularization = candidate_regularization;
            return {alpha, new_loss};
        }

        alpha *= rho;
    }

    back_propagation.error = previous_error;
    back_propagation.regularization = previous_regularization;

    return {0.0f, current_loss};
}

REGISTER(Optimizer, QuasiNewtonMethod, "QuasiNewtonMethod");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
