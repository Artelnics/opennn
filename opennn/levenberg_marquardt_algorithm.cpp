//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_types.h"
#include "tensor_operations.h"
#include "dataset.h"
#include "loss.h"
#include "batch.h"
#include "dense_layer.h"
#include "levenberg_marquardt_algorithm.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

LevenbergMarquardtAlgorithm::LevenbergMarquardtAlgorithm(Loss* new_loss)
    : Optimizer(new_loss)
{
    set_default();
}

void LevenbergMarquardtAlgorithm::set_default()
{
    name = "LevenbergMarquardt";

    minimum_loss_decrease = 0.0f;
    training_loss_goal = 0.0f;
    maximum_validation_failures = 1000;

    maximum_epochs = 1000;
    maximum_time = 3600.0f;

    display_period = 10;

    initial_damping_parameter = 1.0e-3f;
    damping_parameter = initial_damping_parameter;

    damping_parameter_factor = 10.0f;

    minimum_damping_parameter = 1.0e-6f;
    maximum_damping_parameter = 1.0e6f;
}

void LevenbergMarquardtAlgorithm::set_damping_parameter_factor(const float new_damping_parameter_factor)
{
    damping_parameter_factor = new_damping_parameter_factor;
}

void LevenbergMarquardtAlgorithm::set_minimum_loss_decrease(const float new_minimum_loss_decrease)
{
    minimum_loss_decrease = new_minimum_loss_decrease;
}

void LevenbergMarquardtAlgorithm::back_propagate(const Batch& batch,
                                                  const ForwardPropagation& forward_propagation,
                                                  BackPropagationLM& back_propagation_lm)
{
    if (batch.is_empty()) return;

    calculate_errors(batch, forward_propagation, back_propagation_lm);

    calculate_squared_errors(batch, forward_propagation, back_propagation_lm);

    calculate_error(batch, forward_propagation, back_propagation_lm);

    compute_jacobian(batch, forward_propagation, back_propagation_lm);

    const MatrixR& J = back_propagation_lm.squared_errors_jacobian;
    const VectorR& errors_vector = back_propagation_lm.errors;
    const float factor = 2.0f / float(errors_vector.size());

    back_propagation_lm.gradient.noalias() = factor * J.transpose() * errors_vector;
    back_propagation_lm.hessian.noalias() = factor * J.transpose() * J;

    const TensorView parameters(loss->get_neural_network()->get_parameters_data(),
                                {loss->get_neural_network()->get_parameters_size()},
                                Type::FP32,
                                loss->get_neural_network()->get_device());
    back_propagation_lm.regularization = loss->calculate_regularization(parameters);

    loss->add_regularization_gradient(TensorView(back_propagation_lm.gradient.data(),
                                                 { back_propagation_lm.gradient.size() },
                                                 Type::FP32,
                                                 parameters.device));

    back_propagation_lm.loss = back_propagation_lm.error + back_propagation_lm.regularization;
}

void LevenbergMarquardtAlgorithm::calculate_errors(const Batch& batch,
                                                   const ForwardPropagation& forward_propagation,
                                                   BackPropagationLM& back_propagation_lm) const
{
    const VectorMap output = forward_propagation.get_last_trainable_layer_outputs().as_vector();
    const VectorMap target = batch.get_targets().as_vector();

    throw_if(output.size() != target.size() || output.size() != back_propagation_lm.errors.size(),
             "LevenbergMarquardtAlgorithm: outputs ({}), targets ({}) and errors ({}) sizes do not match. The dataset target count does not match the network outputs.", output.size(), target.size(), back_propagation_lm.errors.size());

    back_propagation_lm.errors.noalias() = output - target;
}

void LevenbergMarquardtAlgorithm::calculate_squared_errors(const Batch&,
                                                           const ForwardPropagation&,
                                                           BackPropagationLM& back_propagation_lm) const
{
    back_propagation_lm.squared_errors = back_propagation_lm.errors.array().square();
}

void LevenbergMarquardtAlgorithm::calculate_error(const Batch&,
                                                   const ForwardPropagation&,
                                                   BackPropagationLM& back_propagation_lm) const
{
    back_propagation_lm.error = back_propagation_lm.squared_errors.sum()
                              / float(back_propagation_lm.squared_errors.size());
}

static void lm_activation_derivative(ActivationFunction activation_function, const MatrixMap& outputs, MatrixR& result)
{
    throw_if(activation_function == ActivationFunction::Softmax,
             "LevenbergMarquardtAlgorithm: Softmax activation is not supported "
             "(non-diagonal Jacobian). Use AdaptiveMomentEstimation, SGD, or QuasiNewtonMethod.");

    result = outputs.unaryExpr([activation_function](float value)
             { return activation_derivative_from_output_value(activation_function, value); });
}

void LevenbergMarquardtAlgorithm::compute_jacobian(const Batch&  ,
                                                   const ForwardPropagation& forward_propagation,
                                                   BackPropagationLM& back_propagation_lm)
{
    NeuralNetwork* neural_network = loss->get_neural_network();
    const auto& layers = neural_network->get_layers();
    const auto& source_layers = neural_network->get_source_layers();

    MatrixR& jacobian = back_propagation_lm.squared_errors_jacobian;
    jacobian.setZero();

    vector<Index>& dense_indices = back_propagation_lm.dense_indices;
    vector<Index>& parameter_offsets = back_propagation_lm.parameter_offsets;
    dense_indices.clear();
    parameter_offsets.clear();

    Index offset = 0;
    for (size_t i = 0; i < layers.size(); ++i)
    {
        if (!layers[i]->get_is_trainable() || layers[i]->get_parameters_number() == 0) continue;

        const auto* dense = dynamic_cast<const Dense*>(layers[i].get());
        throw_if(!dense,
                 "LevenbergMarquardtAlgorithm: only Dense trainable layers are supported. "
                 "Use AdaptiveMomentEstimation, SGD, or QuasiNewtonMethod instead.");

        dense_indices.push_back(Index(i));
        parameter_offsets.push_back(offset);

        offset += get_aligned_size(dense->get_outputs_number())
                + get_aligned_size(dense->get_input_shape()[0] * dense->get_outputs_number());
    }

    if (dense_indices.empty()) return;

    throw_if(offset != neural_network->get_parameters_size(),
             "LevenbergMarquardtAlgorithm: unsupported parameter layout (only plain Dense "
             "layers without batch normalization are supported). Use AdaptiveMomentEstimation, "
             "SGD, or QuasiNewtonMethod instead.");

    for (size_t n = 1; n < dense_indices.size(); ++n)
        throw_if(source_layers[dense_indices[n]].size() != 1
                 || source_layers[dense_indices[n]][0] != dense_indices[n - 1],
                 "LevenbergMarquardtAlgorithm: trainable Dense layers must form a sequential "
                 "chain. Use AdaptiveMomentEstimation, SGD, or QuasiNewtonMethod instead.");

    const Index batch_size = forward_propagation.batch_size;
    const Index last_layer = dense_indices.back();
    const Index outputs_number = static_cast<const Dense*>(layers[last_layer].get())->get_outputs_number();
    const Index rows = batch_size * outputs_number;

    const size_t layers_count = dense_indices.size();

    vector<MatrixR>& deltas = back_propagation_lm.deltas;
    vector<MatrixR>& activation_derivatives = back_propagation_lm.activation_derivatives;
    deltas.resize(layers_count);
    activation_derivatives.resize(layers_count);

    for (size_t n = 0; n < layers_count; ++n)
    {
        const Index neurons = static_cast<const Dense*>(layers[dense_indices[n]].get())->get_outputs_number();
        deltas[n].resize(rows, neurons);
        activation_derivatives[n].resize(batch_size, neurons);
    }

    {
        const size_t output_slot = forward_propagation.forward_slots[last_layer].size() - 1;
        const MatrixMap outputs = forward_propagation.forward_slots[last_layer][output_slot].as_matrix();

        MatrixR& act_deriv = activation_derivatives[layers_count - 1];
        lm_activation_derivative(
            static_cast<const Dense*>(layers[last_layer].get())->get_activation_function(), outputs, act_deriv);

        MatrixR& delta = deltas[layers_count - 1];
        delta.setZero();

        #pragma omp parallel for
        for (Index sample = 0; sample < batch_size; ++sample)
            for (Index j = 0; j < outputs_number; ++j)
                delta(sample * outputs_number + j, j) = act_deriv(sample, j);
    }

    for (Index n = Index(layers_count) - 1; n >= 0; --n)
    {
        const Index layer_index = dense_indices[n];
        const auto* dense = static_cast<const Dense*>(layers[layer_index].get());

        const Index neurons = dense->get_outputs_number();
        const Index inputs_number = dense->get_input_shape()[0];
        const MatrixMap inputs = forward_propagation.input_views[layer_index][0].as_matrix();

        const Index bias_offset = parameter_offsets[n];
        const Index weight_offset = bias_offset + get_aligned_size(neurons);

        const MatrixR& delta = deltas[n];

        jacobian.block(0, bias_offset, rows, neurons) = delta;

        #pragma omp parallel for
        for (Index sample = 0; sample < batch_size; ++sample)
            for (Index j = 0; j < outputs_number; ++j)
            {
                const Index row = sample * outputs_number + j;

                for (Index i = 0; i < inputs_number; ++i)
                    for (Index k = 0; k < neurons; ++k)
                        jacobian(row, weight_offset + i * neurons + k) = inputs(sample, i) * delta(row, k);
            }

        if (n == 0) break;

        const MatrixMap weights(neural_network->get_parameters_data() + weight_offset,
                                inputs_number, neurons);

        const auto* previous_dense = static_cast<const Dense*>(layers[dense_indices[n - 1]].get());
        MatrixR& previous_act_deriv = activation_derivatives[n - 1];
        lm_activation_derivative(previous_dense->get_activation_function(), inputs, previous_act_deriv);

        MatrixR& previous_delta = deltas[n - 1];
        previous_delta.noalias() = delta * weights.transpose();

        #pragma omp parallel for
        for (Index sample = 0; sample < batch_size; ++sample)
            for (Index j = 0; j < outputs_number; ++j)
            {
                const Index row = sample * outputs_number + j;

                for (Index k = 0; k < inputs_number; ++k)
                    previous_delta(row, k) *= previous_act_deriv(sample, k);
            }
    }
}

TrainingResult LevenbergMarquardtAlgorithm::train()
{
    NeuralNetwork* neural_network = loss->get_neural_network();
    neural_network->warn_if_stale_configuration();

    throw_if(neural_network->is_gpu(),
             "LevenbergMarquardtAlgorithm does not support GPU training: "
             "its Jacobian and gradient computation map device pointers as host memory. "
             "Use AdaptiveMomentEstimation or StochasticGradientDescent on GPU.");

    const string loss_name = loss->get_name();
    throw_if(loss_name == "MinkowskiError",
             "Levenberg-Marquardt algorithm cannot work with Minkowski error.");
    throw_if(loss_name == "CrossEntropy",
             "Levenberg-Marquardt algorithm cannot work with cross-entropy error.");
    throw_if(loss_name == "WeightedSquaredError",
             "Levenberg-Marquardt algorithm is not implemented with weighted squared error.");

    damping_parameter = initial_damping_parameter;

    FullBatchContext context;
    prepare_full_batch_training(context, "Training with Levenberg-Marquardt algorithm...");

    BackPropagationLM training_back_propagation_lm(context.training_samples_number, loss);
    BackPropagationLM validation_back_propagation_lm(context.validation_samples_number, loss);

    const Index parameters_number = neural_network->get_parameters_size();

    OptimizerData optimization_data;

    FullBatchHooks hooks;
    hooks.minimum_loss_decrease = minimum_loss_decrease;

    hooks.setup_state = [&]
    {
        optimization_data.set({Shape{parameters_number}});
        optimization_data.potential_parameters.resize(parameters_number);
    };

    hooks.train_step = [&]() -> FullBatchStep
    {
        back_propagate(*context.training_batch,
                       *context.training_forward_propagation,
                       training_back_propagation_lm);

        return {training_back_propagation_lm.error,
                training_back_propagation_lm.error,
                training_back_propagation_lm.loss};
    };

    hooks.validation_error = [&]
    {
        calculate_errors(*context.validation_batch, *context.validation_fp, validation_back_propagation_lm);
        calculate_squared_errors(*context.validation_batch, *context.validation_fp, validation_back_propagation_lm);
        calculate_error(*context.validation_batch, *context.validation_fp, validation_back_propagation_lm);

        return validation_back_propagation_lm.error;
    };

    hooks.display_extra = [&]{ cout << "Damping parameter: " << damping_parameter << "\n"; };

    hooks.post_step = [&]
    {
        update_parameters(*context.training_batch,
                          *context.training_forward_propagation,
                          training_back_propagation_lm,
                          optimization_data);
    };

    return train_full_batch(context, hooks);
}

void LevenbergMarquardtAlgorithm::update_parameters(const Batch& batch,
                                                    ForwardPropagation& forward_propagation,
                                                    BackPropagationLM& back_propagation_lm,
                                                    OptimizerData& optimization_data)
{
    NeuralNetwork* neural_network = loss->get_neural_network();

    VectorMap parameters(neural_network->get_parameters_data(),
                         neural_network->get_parameters_size());

    float& error = back_propagation_lm.error;
    float& regularization = back_propagation_lm.regularization;
    float& current_loss = back_propagation_lm.loss;
    const float previous_error = error;
    const float previous_regularization = regularization;

    const VectorR& gradient = back_propagation_lm.gradient;
    MatrixR& hessian = back_propagation_lm.hessian;

    VectorR& potential_parameters = optimization_data.potential_parameters;
    VectorMap parameter_updates = optimization_data.views[ParameterUpdate].as_vector();

    bool success = false;

    const VectorR neg_gradient = -gradient;

    do
    {
        hessian.diagonal().array() += damping_parameter;

        parameter_updates = perform_Householder_QR_decomposition(hessian, neg_gradient);

        potential_parameters = parameters + parameter_updates;

        neural_network->forward_propagate(batch.get_inputs(),
                                          potential_parameters,
                                          forward_propagation);

        calculate_errors(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors(batch, forward_propagation, back_propagation_lm);

        calculate_error(batch, forward_propagation, back_propagation_lm);

        const float candidate_regularization = loss->calculate_regularization(potential_parameters);
        float new_loss = error + candidate_regularization;

        if (!isfinite(new_loss))
            new_loss = current_loss;

        if (new_loss < current_loss)
        {
            damping_parameter = clamp(damping_parameter / damping_parameter_factor,
                                      minimum_damping_parameter,
                                      maximum_damping_parameter);

            parameters = potential_parameters;

            regularization = candidate_regularization;
            current_loss = new_loss;

            success = true;

            break;
        }

        hessian.diagonal().array() -= damping_parameter;

        damping_parameter = clamp(damping_parameter * damping_parameter_factor,
                                  minimum_damping_parameter,
                                  maximum_damping_parameter);

    } while (damping_parameter < maximum_damping_parameter);

    if (!success)
    {
        error = previous_error;
        regularization = previous_regularization;

        parameter_updates = (gradient.array().abs() >= EPSILON)
                                .select(-gradient.array().sign() * EPSILON, 0.0f);
        parameters += parameter_updates;
    }

    neural_network->set_parameters(parameters);
}

void LevenbergMarquardtAlgorithm::to_JSON(JsonWriter& printer) const
{
    printer.open_element("LevenbergMarquardt");

    write_json(printer, {
        {"DampingParameterFactor", damping_parameter_factor},
        {"MinimumLossDecrease", minimum_loss_decrease}
    });
    write_common_json(printer);

    printer.close_element();
}

void LevenbergMarquardtAlgorithm::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "LevenbergMarquardt");

    set_damping_parameter_factor(read_json_float(root_element, "DampingParameterFactor"));
    set_minimum_loss_decrease(read_json_float(root_element, "MinimumLossDecrease"));
    read_common_json(root_element);
}

REGISTER(Optimizer, LevenbergMarquardtAlgorithm, "LevenbergMarquardt");

BackPropagationLM::BackPropagationLM(const Index new_samples_number, Loss* new_loss)
{
    set(new_samples_number, new_loss);
}

void BackPropagationLM::set(const Index new_samples_number, Loss* new_loss)
{
    loss_pointer = new_loss;
    samples_number = new_samples_number;
    error = 0.0f;
    regularization = 0.0f;
    loss = 0.0f;

    if (!new_loss || !new_loss->get_neural_network() || new_samples_number == 0) return;

    const NeuralNetwork* neural_network = new_loss->get_neural_network();

    const Index outputs_number = neural_network->get_outputs_number();
    const Index parameters_number = neural_network->get_parameters_size();
    const Index total_error_terms = new_samples_number * outputs_number;

    errors                  = VectorR::Zero(total_error_terms);
    squared_errors          = VectorR::Zero(total_error_terms);
    squared_errors_jacobian = MatrixR::Zero(total_error_terms, parameters_number);
    gradient                = VectorR::Zero(parameters_number);
    hessian                 = MatrixR::Zero(parameters_number, parameters_number);
}
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
