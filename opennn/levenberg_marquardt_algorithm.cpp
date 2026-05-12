//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E V E N B E R G - M A R Q U A R D T   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
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

    // Stopping criteria

    minimum_loss_decrease = 0.0f;
    training_loss_goal = 0.0f;
    maximum_validation_failures = 1000;

    maximum_epochs = 1000;
    maximum_time = 3600.0f;

    // UTILITIES

    display_period = 10;

    // Training parameters

    damping_parameter = 1.0e-3f;

    damping_parameter_factor = 10.0f;

    minimum_damping_parameter = 1.0e-6f;
    maximum_damping_parameter = 1.0e6f;
}

void LevenbergMarquardtAlgorithm::set_damping_parameter(const float new_damping_parameter)
{
    damping_parameter = clamp(new_damping_parameter, minimum_damping_parameter, maximum_damping_parameter);
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

    back_propagation_lm.loss_value = back_propagation_lm.error;
}

void LevenbergMarquardtAlgorithm::calculate_errors(const Batch& batch,
                                                   const ForwardPropagation& forward_propagation,
                                                   BackPropagationLM& back_propagation_lm) const
{
    const VectorMap output = forward_propagation.get_last_trainable_layer_outputs().as_vector();
    const VectorMap target = batch.get_targets().as_vector();

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

void LevenbergMarquardtAlgorithm::compute_jacobian(const Batch& batch,
                                                   const ForwardPropagation& forward_propagation,
                                                   BackPropagationLM& back_propagation_lm)
{
    NeuralNetwork* neural_network = loss->get_neural_network();
    const auto& layers = neural_network->get_layers();

    back_propagation_lm.squared_errors_jacobian.setZero();

    Index last_trainable_dense = -1;
    for (size_t i = 0; i < layers.size(); ++i)
        if (layers[i]->get_is_trainable() && dynamic_cast<Dense*>(layers[i].get()))
            last_trainable_dense = i;

    if (last_trainable_dense < 0) return;

    // Compute parameter offset up to the last trainable Dense layer
    Index parameter_offset = 0;
    for (size_t i = 0; i < static_cast<size_t>(last_trainable_dense); ++i)
        if (layers[i]->get_is_trainable())
            parameter_offset += layers[i]->get_parameters_number();

    auto* dense = dynamic_cast<Dense*>(layers[last_trainable_dense].get());
    insert_dense_jacobian(dense, forward_propagation, last_trainable_dense, parameter_offset, back_propagation_lm.squared_errors_jacobian);
}

static MatrixR activation_derivative(ActivationOp::Function activation_function, const MatrixMap& outputs)
{
    switch (activation_function)
    {
    case ActivationOp::Function::Sigmoid:
        return outputs.array() * (1.0f - outputs.array());
    case ActivationOp::Function::Tanh:
        return 1.0f - outputs.array().square();
    case ActivationOp::Function::ReLU:
        return (outputs.array() > 0.0f).cast<float>();
    default:                                                  // Identity / unrecognized -> identity
        return MatrixR::Ones(outputs.rows(), outputs.cols());
    }
}

void LevenbergMarquardtAlgorithm::insert_dense_jacobian(const Dense* layer,
                                                        const ForwardPropagation& forward_propagation,
                                                        Index layer_index,
                                                        Index parameter_offset,
                                                        MatrixR& jacobian)
{
    const Index batch_size  = forward_propagation.batch_size;
    const Index num_neurons = layer->get_outputs_number();
    const Index num_inputs  = layer->get_input_shape()[0];

    const MatrixMap inputs  = forward_propagation.views[layer_index][0][0].as_matrix();

    // Output is in the last slot of views for this layer (slot 0 = input, last slot = output).
    const size_t output_slot = forward_propagation.views[layer_index].size() - 1;
    const MatrixMap outputs  = forward_propagation.views[layer_index][output_slot][0].as_matrix();

    const MatrixR act_deriv = activation_derivative(layer->get_activation_function(), outputs);

    // Parameter layout (aligned): [bias(aligned) | weights(aligned) | gamma | beta]
    const Index bias_aligned  = get_aligned_size(num_neurons);
    const Index weight_offset = parameter_offset + bias_aligned;

    // Bias derivatives: dE/db_j = act_deriv(s,j)
    for (Index j = 0; j < num_neurons; ++j)
        for (Index sample = 0; sample < batch_size; ++sample)
            jacobian(sample * num_neurons + j, parameter_offset + j) = act_deriv(sample, j);

    // Weight derivatives: dE/dw_{k,j} = input_k * act_deriv(s,j)
    for (Index k = 0; k < num_inputs; ++k)
        for (Index j = 0; j < num_neurons; ++j)
        {
            const Index col = weight_offset + k * num_neurons + j;
            for (Index sample = 0; sample < batch_size; ++sample)
                jacobian(sample * num_neurons + j, col) = inputs(sample, k) * act_deriv(sample, j);
        }
}

TrainingResults LevenbergMarquardtAlgorithm::train()
{
    const string loss_name = loss->get_name();
    if (loss_name == "MinkowskiError")
        throw runtime_error("Levenberg-Marquard algorithm cannot work with Minkowski error.");
    if (loss_name == "CrossEntropy")
        throw runtime_error("Levenberg-Marquard algorithm cannot work with cross-entropy error.");
    if (loss_name == "WeightedSquaredError")
        throw runtime_error("Levenberg-Marquard algorithm is not implemented with weighted squared error.");

    // Start training

    if (display) cout << "Training with Levenberg-Marquardt algorithm..." << "\n";

    TrainingResults results(maximum_epochs+1);

    // Dataset

    Dataset* dataset = loss->get_dataset();

    const bool has_validation = dataset->has_validation();

    const Index training_samples_number = dataset->get_samples_number("Training");
    const Index validation_samples_number = dataset->get_samples_number("Validation");

    const vector<Index> training_sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> validation_sample_indices = dataset->get_sample_indices("Validation");

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    // Neural network

    NeuralNetwork* neural_network = loss->get_neural_network();

    set_names();

    set_scaling();

    Batch training_batch(training_samples_number, dataset);
    training_batch.fill(training_sample_indices, input_feature_indices, {}, target_feature_indices, true);

    Batch validation_batch(validation_samples_number, dataset);
    validation_batch.fill(validation_sample_indices, input_feature_indices, {}, target_feature_indices);

    ForwardPropagation training_forward_propagation(training_samples_number, neural_network);

    // Reuse the training FP for validation iff sample counts match exactly.
    // LM is full-batch so splits typically differ — separate FP is the common
    // case, but the alias activates for symmetric splits (e.g. 50/50).
    unique_ptr<ForwardPropagation> validation_forward_propagation;

    if (has_validation && validation_samples_number != training_samples_number)
        validation_forward_propagation = make_unique<ForwardPropagation>(
            validation_samples_number, neural_network);

    ForwardPropagation* validation_fp = has_validation
        ? (validation_forward_propagation ? validation_forward_propagation.get() : &training_forward_propagation)
        : nullptr;

    // Loss index

    loss->set_normalization_coefficient();

    float old_loss = 0.0f;
    float loss_decrease = MAX;

    Index validation_failures = 0;

    BackPropagationLM training_back_propagation_lm(training_samples_number, loss);
    BackPropagationLM validation_back_propagation_lm(validation_samples_number, loss);

    time_t beginning_time;
    time(&beginning_time);
    float elapsed_time = 0.0f;

    const Index parameters_number = neural_network->get_parameters_size();

    OptimizerData optimization_data;
    optimization_data.set({Shape{parameters_number}});
    optimization_data.potential_parameters.resize(parameters_number);

    // Main loop

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

            if (epoch != 0 && validation_back_propagation_lm.error > results.validation_error_history(epoch-1))
                ++validation_failures;
        }

        elapsed_time = get_elapsed_time(beginning_time);

        if (epoch != 0) loss_decrease = old_loss - training_back_propagation_lm.loss_value;

        old_loss = training_back_propagation_lm.loss_value;

        if (should_display(epoch))
        {
            cout << "Training error: " << results.training_error_history(epoch) << "\n";
            if (has_validation) cout << "Validation error: " << results.validation_error_history(epoch) << "\n";
            cout << "Damping parameter: " << damping_parameter << "\n";
            cout << "Elapsed time: " << get_time(elapsed_time) << "\n";
        }

        bool stop = false;

        if (loss_decrease < minimum_loss_decrease)
        {
            if (display) cout << "Epoch " << epoch << "\nMinimum loss decrease reached: " << loss_decrease << "\n";
            results.stopping_condition = StoppingCondition::MinimumLossDecrease;
            stop = true;
        }
        else
        {
            stop = check_stopping_condition(results, epoch, elapsed_time,
                                            results.training_error_history(epoch),
                                            validation_failures);
        }

        if (stop)
        {
            results.loss = training_back_propagation_lm.loss_value;
            results.loss_decrease = loss_decrease;
            results.validation_failures = validation_failures;
            results.resize_training_error_history(epoch+1);
            results.resize_validation_error_history(has_validation ? epoch + 1 : 0);
            results.elapsed_time = get_time(elapsed_time);
            break;
        }

        update_parameters(training_batch,
                          training_forward_propagation,
                          training_back_propagation_lm,
                          optimization_data);
    }

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
    float& loss_value = back_propagation_lm.loss_value;

    const VectorR& gradient = back_propagation_lm.gradient;
    MatrixR& hessian = back_propagation_lm.hessian;

    VectorR& potential_parameters = optimization_data.potential_parameters;
    VectorMap parameter_updates(optimization_data.views[ParameterUpdate].as<float>(),
                                optimization_data.views[ParameterUpdate].size());

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

        float new_loss_value = error + loss->calculate_regularization(potential_parameters);

        if (!isfinite(new_loss_value))
            new_loss_value = loss_value;

        if (new_loss_value < loss_value)
        {
            set_damping_parameter(damping_parameter/damping_parameter_factor);

            parameters = potential_parameters;

            loss_value = new_loss_value;

            success = true;

            break;
        }
        else
        {
            hessian.diagonal().array() -= damping_parameter;

            set_damping_parameter(damping_parameter*damping_parameter_factor);
        }

    } while (damping_parameter < maximum_damping_parameter);

    if (!success)
    {
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
    write_common_xml(printer);

    printer.close_element();
}

void LevenbergMarquardtAlgorithm::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "LevenbergMarquardt");

    set_damping_parameter_factor(read_json_type(root_element, "DampingParameterFactor"));
    set_minimum_loss_decrease(read_json_type(root_element, "MinimumLossDecrease"));
    read_common_xml(root_element);
}

REGISTER(Optimizer, LevenbergMarquardtAlgorithm, "LevenbergMarquardt");

BackPropagationLM::BackPropagationLM(const Index new_samples_number, Loss* new_loss)
{
    set(new_samples_number, new_loss);
}

void BackPropagationLM::set(const Index new_samples_number, Loss* new_loss)
{
    loss = new_loss;
    samples_number = new_samples_number;

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
