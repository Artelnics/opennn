//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/training_strategy/quasi_newton_method.h"

#include "opennn/dataset/batch.h"
#include "opennn/dataset/dataset.h"
#include "opennn/neural_network/back_propagation.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/training_strategy/loss.h"

namespace opennn
{

QuasiNewtonMethod::QuasiNewtonMethod(Loss* new_loss)
    : Optimizer(new_loss)
{
    name = "QuasiNewtonMethod";
    maximum_validation_failures = 1000;
    maximum_epochs = 1000;
    maximum_time = 3600.0f;
}

void QuasiNewtonMethod::calculate_inverse_hessian(OptimizerData& optimization_data) const
{
    VectorMap parameter_differences = optimization_data.views[ParameterDifferences].as_vector();
    VectorMap gradient_difference = optimization_data.views[GradientDifference].as_vector();

    VectorMap old_inverse_hessian_dot_gradient_difference =
        optimization_data.views[OldInverseHessianDotGradientDifference].as_vector();

    MatrixMap old_inverse_hessian = optimization_data.views[OldInverseHessian].as_matrix();
    MatrixMap inverse_hessian     = optimization_data.views[InverseHessian].as_matrix();

    VectorMap bfgs = optimization_data.views[BFGS].as_vector();

    const float parameters_difference_dot_gradient_difference = parameter_differences.dot(gradient_difference);

    old_inverse_hessian_dot_gradient_difference.noalias() = old_inverse_hessian * gradient_difference;

    const float gradient_dot_hessian_dot_gradient = gradient_difference.dot(old_inverse_hessian_dot_gradient_difference);

    if (parameters_difference_dot_gradient_difference <= EPSILON
        || gradient_dot_hessian_dot_gradient <= EPSILON)
    {
        inverse_hessian = old_inverse_hessian;
        return;
    }

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

void QuasiNewtonMethod::update_full_batch_parameters(const Batch& batch,
                                                     ForwardPropagation& forward_propagation,
                                                     BackPropagation& back_propagation,
                                                     OptimizerData& optimization_data)
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    VectorMap parameters = neural_network->get_parameters_map();
    VectorMap gradient = back_propagation.gradient.as_vector();

    VectorMap old_parameters = optimization_data.views[OldParameters].as_vector();
    VectorMap parameter_differences = optimization_data.views[ParameterDifferences].as_vector();
    VectorMap parameter_updates = optimization_data.views[ParameterUpdates].as_vector();

    VectorMap old_gradient = optimization_data.views[OldGradient].as_vector();
    VectorMap gradient_difference = optimization_data.views[GradientDifference].as_vector();

    VectorR& training_direction = line_search.direction;
    MatrixMap inverse_hessian = optimization_data.views[InverseHessian].as_matrix();

    parameter_differences = parameters - old_parameters;
    gradient_difference = gradient - old_gradient;

    old_parameters = parameters;

    if (parameter_differences.isZero() || gradient_difference.isZero())
        inverse_hessian.setIdentity();
    else
        calculate_inverse_hessian(optimization_data);

    training_direction.noalias() = -(inverse_hessian.selfadjointView<Lower>() * gradient);

    line_search.slope = gradient.dot(training_direction);

    bool is_gradient_direction = false;

    if (line_search.slope >= 0.0f)
    {
        training_direction = -gradient;
        line_search.slope = gradient.dot(training_direction);
        is_gradient_direction = true;
    }

    line_search.initial = is_gradient_direction
        ? ((line_search.old_learning_rate > 0.0f)
            ? line_search.old_learning_rate : first_learning_rate)
        : 1.0f;

    tie(line_search.learning_rate, back_propagation.metrics.loss_value) = calculate_directional_point(
        batch,
        forward_propagation,
        back_propagation,
        back_propagation.metrics.loss_value);

    if (line_search.learning_rate == 0.0f && !is_gradient_direction)
    {
        inverse_hessian.setIdentity();
        optimization_data.views[OldInverseHessian].as_matrix().setIdentity();

        training_direction = -gradient;
        line_search.slope = gradient.dot(training_direction);

        line_search.initial = (line_search.old_learning_rate > 0.0f)
            ? line_search.old_learning_rate
            : first_learning_rate;

        tie(line_search.learning_rate, back_propagation.metrics.loss_value) = calculate_directional_point(
            batch,
            forward_propagation,
            back_propagation,
            back_propagation.metrics.loss_value);
    }

    if (abs(line_search.learning_rate) > 0.0f)
    {
        parameter_updates = training_direction * line_search.learning_rate;
    }
    else
    {
        parameter_updates = (gradient.array().abs() >= EPSILON)
                                .select(-gradient.array().sign() * EPSILON, 0.0f);
    }
    parameters += parameter_updates;

    old_gradient = gradient;
    swap(optimization_data.views[InverseHessian], optimization_data.views[OldInverseHessian]);

    if (line_search.learning_rate > 0.0f)
        line_search.old_learning_rate = line_search.learning_rate;
}

TrainingResult QuasiNewtonMethod::train()
{
    NeuralNetwork* neural_network = loss->get_neural_network();
    neural_network->warn_if_stale_configuration();

    throw_if(neural_network->is_gpu(),
             "QuasiNewtonMethod does not support GPU training: "
             "its update path maps device pointers as host memory. "
             "Use AdaptiveMomentEstimation or StochasticGradientDescent on GPU.");

    FullBatchContext context;
    prepare_full_batch_training(context, "Training with quasi-Newton method...");

    BackPropagation training_back_propagation(context.training_samples_number, *loss);

    const Index parameters_number = neural_network->get_parameters_buffer_size();

    OptimizerData optimization_data;

    FullBatchHooks hooks;
    hooks.minimum_loss_decrease = minimum_loss_decrease;

    hooks.setup_state = [&]
    {
        optimization_data.set({
            Shape{parameters_number},
            Shape{parameters_number},
            Shape{parameters_number},
            Shape{parameters_number},
            Shape{parameters_number},
            Shape{parameters_number},
            Shape{parameters_number},
            Shape{parameters_number, parameters_number},
            Shape{parameters_number, parameters_number}
        });

        line_search.reset(parameters_number);

        optimization_data.views[OldParameters].as_vector() = neural_network->get_parameters_map();

        optimization_data.views[InverseHessian].as_matrix().setIdentity();
        optimization_data.views[OldInverseHessian].as_matrix().setIdentity();
    };

    hooks.train_step = [&]() -> FullBatchStep
    {
        loss->back_propagate(*context.training_batch,
                             *context.training_forward_propagation,
                             training_back_propagation);

        NeuralNetwork* const network = loss->get_neural_network();

        const TensorView parameters(network->get_parameters_data(),
                                    {network->get_parameters_buffer_size()},
                                    Type::FP32,
                                    network->get_device());

        training_back_propagation.metrics.regularization =
            loss->calculate_regularization(parameters);

        training_back_propagation.metrics.loss_value +=
            training_back_propagation.metrics.regularization;

        const float training_error = training_back_propagation.metrics.error;

        update_full_batch_parameters(*context.training_batch,
                                     *context.training_forward_propagation,
                                     training_back_propagation,
                                     optimization_data);

        return {training_error, training_back_propagation.metrics.error, training_back_propagation.metrics.loss_value};
    };

    hooks.validation_error = [&]
    {
        return loss->calculate_error(*context.validation_batch,
                                     *context.validation_forward_propagation).error;
    };

    hooks.display_extra = [&]{ cout << "Learning rate: " << line_search.learning_rate << "\n"; };

    return train_full_batch(context, hooks);
}

void QuasiNewtonMethod::to_JSON(JsonWriter& printer) const
{
    printer.open_element("QuasiNewtonMethod");

    add_json_field(printer, "MinimumLossDecrease", minimum_loss_decrease);
    write_common_json(printer);

    printer.close_element();
}

void QuasiNewtonMethod::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "QuasiNewtonMethod");

    set_minimum_loss_decrease(read_json_float(root_element, "MinimumLossDecrease"));
    read_common_json(root_element);
}

pair<float, float> QuasiNewtonMethod::calculate_directional_point(
    const Batch& batch,
    ForwardPropagation& forward_propagation,
    BackPropagation& back_propagation,
    float current_loss)
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    float alpha = (line_search.initial > 0.0f)
        ? line_search.initial
        : 1.0f;
    const float rho = 0.5f;
    const float armijo_constant = 1e-4f;
    const float previous_error = back_propagation.metrics.error;
    const float previous_regularization = back_propagation.metrics.regularization;

    const VectorMap parameters = neural_network->get_parameters_map();
    const VectorR& training_direction = line_search.direction;
    VectorR& potential_parameters = line_search.potential;

    for (int i = 0; i < 20; ++i)
    {
        potential_parameters = parameters + training_direction * alpha;

        neural_network->forward_propagate(batch.get_inputs(), potential_parameters, forward_propagation);
        const Loss::EvaluationResult evaluation_result = loss->calculate_error(batch, forward_propagation);
        const float candidate_regularization = loss->calculate_regularization(potential_parameters);
        const float new_loss = evaluation_result.error + candidate_regularization;

        if (new_loss <= current_loss + armijo_constant * alpha * line_search.slope)
        {
            back_propagation.metrics.error = evaluation_result.error;
            back_propagation.metrics.regularization = candidate_regularization;
            return {alpha, new_loss};
        }

        alpha *= rho;
    }

    back_propagation.metrics.error = previous_error;
    back_propagation.metrics.regularization = previous_regularization;

    return {0.0f, current_loss};
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
