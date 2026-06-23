//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_types.h"
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

void LevenbergMarquardtAlgorithm::set_damping_parameter(const float new_damping_parameter)
{
    initial_damping_parameter = clamp(new_damping_parameter, minimum_damping_parameter, maximum_damping_parameter);
    damping_parameter = initial_damping_parameter;
}

void LevenbergMarquardtAlgorithm::set_damping_parameter_factor(const float new_damping_parameter_factor)
{
    damping_parameter_factor = new_damping_parameter_factor;
}

void LevenbergMarquardtAlgorithm::set_minimum_damping_parameter(const float new_minimum_damping_parameter)
{
    minimum_damping_parameter = new_minimum_damping_parameter;
}

void LevenbergMarquardtAlgorithm::set_maximum_damping_parameter(const float new_maximum_damping_parameter)
{
    maximum_damping_parameter = new_maximum_damping_parameter;
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
             "LevenbergMarquardtAlgorithm: outputs (" + to_string(output.size())
             + "), targets (" + to_string(target.size())
             + ") and errors (" + to_string(back_propagation_lm.errors.size())
             + ") sizes do not match. The dataset target count does not match the network outputs.");

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

static MatrixR activation_derivative(ActivationOp::Function activation_function, const MatrixMap& outputs)
{
    using enum ActivationOp::Function;
    switch (activation_function)
    {
    case Identity:
        return MatrixR::Ones(outputs.rows(), outputs.cols());
    case Softmax:
        throw runtime_error("LevenbergMarquardtAlgorithm: Softmax activation is not supported "
                            "(non-diagonal Jacobian). Use AdaptiveMomentEstimation, SGD, or QuasiNewtonMethod.");
    case Sigmoid:
        return outputs.array() * (1.0f - outputs.array());
    case Tanh:
        return 1.0f - outputs.array().square();
    case ReLU:
        return (outputs.array() > 0.0f).cast<float>();
    case LeakyReLU: return outputs.unaryExpr([](float y) { return y >= 0.0f ? 1.0f : LEAKY_RELU_SLOPE; });
    }

    return MatrixR::Ones(outputs.rows(), outputs.cols());
}

void LevenbergMarquardtAlgorithm::compute_jacobian(const Batch& /*batch*/,
                                                   const ForwardPropagation& forward_propagation,
                                                   BackPropagationLM& back_propagation_lm)
{
    NeuralNetwork* neural_network = loss->get_neural_network();
    const auto& layers = neural_network->get_layers();
    const auto& source_layers = neural_network->get_source_layers();

    MatrixR& jacobian = back_propagation_lm.squared_errors_jacobian;
    jacobian.setZero();

    vector<Index> dense_indices;
    vector<Index> parameter_offsets;

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

    // delta(sample * outputs_number + j, k) = d output_j(sample) / d combination_k(sample)
    // of the layer being processed; rows match the errors vector layout.
    MatrixR delta = MatrixR::Zero(rows, outputs_number);
    {
        const size_t output_slot = forward_propagation.forward_slots[last_layer].size() - 1;
        const MatrixMap outputs = forward_propagation.forward_slots[last_layer][output_slot].as_matrix();
        const MatrixR act_deriv = activation_derivative(
            static_cast<const Dense*>(layers[last_layer].get())->get_activation_function(), outputs);

        for (Index j = 0; j < outputs_number; ++j)
            for (Index sample = 0; sample < batch_size; ++sample)
                delta(sample * outputs_number + j, j) = act_deriv(sample, j);
    }

    for (Index n = Index(dense_indices.size()) - 1; n >= 0; --n)
    {
        const Index layer_index = dense_indices[n];
        const auto* dense = static_cast<const Dense*>(layers[layer_index].get());

        const Index neurons = dense->get_outputs_number();
        const Index inputs_number = dense->get_input_shape()[0];
        const MatrixMap inputs = forward_propagation.input_views[layer_index][0].as_matrix();

        const Index bias_offset = parameter_offsets[n];
        const Index weight_offset = bias_offset + get_aligned_size(neurons);

        jacobian.block(0, bias_offset, rows, neurons) = delta.leftCols(neurons);

        for (Index i = 0; i < inputs_number; ++i)
            for (Index k = 0; k < neurons; ++k)
            {
                const Index column = weight_offset + i * neurons + k;
                for (Index j = 0; j < outputs_number; ++j)
                    for (Index sample = 0; sample < batch_size; ++sample)
                        jacobian(sample * outputs_number + j, column) =
                            inputs(sample, i) * delta(sample * outputs_number + j, k);
            }

        if (n == 0) break;

        const MatrixMap weights(neural_network->get_parameters_data() + weight_offset,
                                inputs_number, neurons);

        const auto* previous_dense = static_cast<const Dense*>(layers[dense_indices[n - 1]].get());
        const MatrixR previous_act_deriv =
            activation_derivative(previous_dense->get_activation_function(), inputs);

        MatrixR previous_delta(rows, inputs_number);
        previous_delta.noalias() = delta * weights.transpose();

        for (Index k = 0; k < inputs_number; ++k)
            for (Index j = 0; j < outputs_number; ++j)
                for (Index sample = 0; sample < batch_size; ++sample)
                    previous_delta(sample * outputs_number + j, k) *= previous_act_deriv(sample, k);

        delta = std::move(previous_delta);
    }
}

TrainingResult LevenbergMarquardtAlgorithm::train()
{
    NeuralNetwork* neural_network = loss->get_neural_network();

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


    if (display) cout << "Training with Levenberg-Marquardt algorithm..." << "\n";

    TrainingResult results(maximum_epochs+1);


    Dataset* dataset = loss->get_dataset();

    const bool has_validation = dataset->has_validation();

    const Index training_samples_number = dataset->get_samples_number("Training");
    const Index validation_samples_number = dataset->get_samples_number("Validation");

    const vector<Index> training_sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> validation_sample_indices = dataset->get_sample_indices("Validation");

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");


    set_names();

    set_scaling();

    Batch training_batch(training_samples_number, dataset, neural_network->get_config());
    training_batch.fill(training_sample_indices, input_feature_indices, {}, target_feature_indices, true);

    Batch validation_batch(validation_samples_number, dataset, neural_network->get_config());
    validation_batch.fill(validation_sample_indices, input_feature_indices, {}, target_feature_indices, /*is_training=*/false);

    ForwardPropagation training_forward_propagation(training_samples_number, neural_network);

    const unique_ptr<ForwardPropagation> validation_forward_propagation =
        (has_validation && validation_samples_number != training_samples_number)
            ? make_unique<ForwardPropagation>(validation_samples_number, neural_network)
            : nullptr;

    ForwardPropagation* validation_fp = has_validation
        ? (validation_forward_propagation ? validation_forward_propagation.get() : &training_forward_propagation)
        : nullptr;


    loss->set_normalization_coefficient();

    float old_loss = 0.0f;
    float loss_decrease = MAX;

    Index validation_failures = 0;
    reset_best_parameters();

    BackPropagationLM training_back_propagation_lm(training_samples_number, loss);
    BackPropagationLM validation_back_propagation_lm(validation_samples_number, loss);

    time_t beginning_time;
    time(&beginning_time);
    float elapsed_time = 0.0f;

    const Index parameters_number = neural_network->get_parameters_size();

    OptimizerData optimization_data;
    optimization_data.set({Shape{parameters_number}});
    optimization_data.potential_parameters.resize(parameters_number);


    for (Index epoch = 0; epoch <= maximum_epochs; ++epoch)
    {
        if (should_display(epoch)) cout << "Epoch: " << epoch << "\n";

        neural_network->forward_propagate(training_batch.get_inputs(),
                                          training_forward_propagation,
                                          true);

        back_propagate(training_batch,
                       training_forward_propagation,
                       training_back_propagation_lm);

        results.training_error_history(epoch) = training_back_propagation_lm.error;

        if (has_validation)
        {
            neural_network->forward_propagate(validation_batch.get_inputs(),
                                              *validation_fp,
                                              false);

            calculate_errors(validation_batch, *validation_fp, validation_back_propagation_lm);
            calculate_squared_errors(validation_batch, *validation_fp, validation_back_propagation_lm);
            calculate_error(validation_batch, *validation_fp, validation_back_propagation_lm);

            results.validation_error_history(epoch) = validation_back_propagation_lm.error;

            update_best_parameters(neural_network, validation_back_propagation_lm.error,
                                   epoch, validation_failures);
        }

        elapsed_time = get_elapsed_time(beginning_time);

        if (epoch != 0) loss_decrease = old_loss - training_back_propagation_lm.loss;

        old_loss = training_back_propagation_lm.loss;

        if (should_display(epoch))
        {
            cout << "Training error: " << results.training_error_history(epoch) << "\n";
            if (has_validation) cout << "Validation error: " << results.validation_error_history(epoch) << "\n";
            cout << "Damping parameter: " << damping_parameter << "\n";
            cout << "Elapsed time: " << get_time(elapsed_time) << "\n";
        }

        if (loss_decrease < minimum_loss_decrease)
        {
            if (display) cout << "Epoch " << epoch << "\nMinimum loss decrease reached: " << loss_decrease << "\n";
            results.stopping_condition = StoppingCondition::MinimumLossDecrease;
        }

        if (check_stopping_condition(results, epoch, elapsed_time,
                                     results.training_error_history(epoch),
                                     validation_failures,
                                     training_back_propagation_lm.loss,
                                     has_validation))
        {
            results.loss_decrease = loss_decrease;
            break;
        }

        update_parameters(training_batch,
                          training_forward_propagation,
                          training_back_propagation_lm,
                          optimization_data);
    }

    restore_best_parameters(neural_network, results);

    set_unscaling();

    if (display) results.print();

    return results;
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
        {"DampingParameterFactor", to_string(damping_parameter_factor)},
        {"MinimumLossDecrease", to_string(minimum_loss_decrease)}
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

    errors.resize(total_error_terms);
    errors.setZero();

    squared_errors.resize(total_error_terms);
    squared_errors.setZero();

    squared_errors_jacobian.resize(total_error_terms, parameters_number);
    squared_errors_jacobian.setZero();

    gradient.resize(parameters_number);
    gradient.setZero();

    hessian.resize(parameters_number, parameters_number);
    hessian.setZero();
}
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
