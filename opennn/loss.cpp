//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   I N D E X   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensor_utilities.h"
#include "batch.h"
#include "dataset.h"
#include "loss.h"
#include "error_utilities.h"
#include "profiler.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#include <Eigen/LU>

namespace opennn
{

Loss::Loss(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{
    set(new_neural_network, new_dataset);
}

void Loss::set(NeuralNetwork* new_neural_network, Dataset* new_dataset)
{
    neural_network = new_neural_network;
    dataset = new_dataset;

    regularization_method = Regularization::L2;
    set_error(Error::MeanSquaredError);

}

void Loss::set_normalization_coefficient()
{
    normalization_coefficient = 1.0f;
    positives_weight = 1.0f;
    negatives_weight = 1.0f;

    if (!dataset || dataset->get_samples_number() == 0)
        return;

    if (error == Error::WeightedSquaredError)
    {
        const Index targets_number = dataset->get_features_number("Target");
        if (targets_number != 1) return;

        const VectorI distribution = dataset->calculate_target_distribution();
        const Index negatives = distribution(0);
        const Index positives = distribution(1);

        if (positives == 0 || negatives == 0) return;

        const float total = float(positives + negatives);
        positives_weight = total / (2.0f * float(positives));
        negatives_weight = total / (2.0f * float(negatives));
        normalization_coefficient = 1.0f;
    }
}

void Loss::back_propagate(const Batch& batch,
                          ForwardPropagation& forward_propagation,
                          BackPropagation& back_propagation) const
{
    if (batch.is_empty()) return;

    calculate_error(batch, forward_propagation, back_propagation);

    calculate_layers_error_gradient(batch, forward_propagation, back_propagation);

    back_propagation.loss_value = back_propagation.error;

    // Regularization

    add_regularization(back_propagation);

    add_regularization_gradient(back_propagation);
}

float Loss::get_weighted_coefficient(const Batch& batch) const
{
    const Index total = dataset ? dataset->get_samples_number() : batch.get_samples_number();
    const Index samples = batch.get_samples_number();
    return float(total) / (float(samples) * (normalization_coefficient + EPSILON));
}

void Loss::calculate_error(const Batch& batch, const ForwardPropagation& forward_propagation, BackPropagation& back_propagation) const
{
    const TensorView input = forward_propagation.get_last_trainable_layer_outputs();
    const TensorView target = batch.get_targets();

#ifdef OPENNN_WITH_CUDA
    float* workspace_device = Configuration::instance().is_gpu() ? back_propagation.errors_device.as<float>() : nullptr;
#else
    float* workspace_device = nullptr;
#endif

    switch (error)
    {
    case Error::MeanSquaredError:
        mean_squared_error(input, target, back_propagation.error, workspace_device);
        break;
    case Error::NormalizedSquaredError:
        normalized_squared_error(input, target, normalization_coefficient, back_propagation.error, workspace_device);
        break;
    case Error::WeightedSquaredError:
        weighted_squared_error(input, target, positives_weight, negatives_weight, back_propagation.error, workspace_device);
        back_propagation.error *= get_weighted_coefficient(batch);
        break;
    case Error::CrossEntropy:
        if (input.shape.back() == 1)
            binary_cross_entropy(input, target, back_propagation.error, workspace_device);
        else
            categorical_cross_entropy(input, target, back_propagation.error, workspace_device);
        break;
    case Error::CrossEntropy3d:
    {
        Index correct_tokens = 0;
        cross_entropy_3d(input, target, back_propagation.error, back_propagation.active_tokens_count, correct_tokens, workspace_device);
        const Index active = back_propagation.active_tokens_count;
        back_propagation.accuracy.setValues(
            {active > 0 ? float(correct_tokens) / float(active) : 0.0f});
        break;
    }
    case Error::MinkowskiError:
        minkowski_error(input, target, minkowski_parameter, back_propagation.error, workspace_device);
        break;
    }
}

void Loss::calculate_output_deltas(const Batch& batch, const ForwardPropagation& forward_propagation, BackPropagation& back_propagation) const
{
    const TensorView input = forward_propagation.get_last_trainable_layer_outputs();
    const TensorView target = batch.get_targets();
    TensorView input_delta = back_propagation.get_output_deltas();

    switch (error)
    {
    case Error::MeanSquaredError:
        mean_squared_error_gradient(input, target, input_delta);
        break;
    case Error::NormalizedSquaredError:
        normalized_squared_error_gradient(input, target, normalization_coefficient, input_delta);
        break;
    case Error::WeightedSquaredError:
        weighted_squared_error_gradient(input, target, positives_weight, negatives_weight, get_weighted_coefficient(batch), input_delta);
        break;
    case Error::CrossEntropy:
        cross_entropy_gradient(input, target, input_delta);
        break;
    case Error::CrossEntropy3d:
        cross_entropy_3d_gradient(input, target, input_delta, back_propagation.active_tokens_count);
        break;
    case Error::MinkowskiError:
        minkowski_error_gradient(input, target, minkowski_parameter, input_delta);
        break;
    }
}

void Loss::add_regularization(BackPropagation& back_propagation) const
{
    if (regularization_method == Regularization::NoRegularization) return;

    check_neural_network();

#ifdef OPENNN_WITH_CUDA
    if (Configuration::instance().is_gpu()) {
        return;
    }
#endif
    Map<const VectorR, AlignedMax> params_vec(neural_network->get_parameters_data(),
                                               neural_network->get_parameters_size());
    back_propagation.loss_value += calculate_regularization(params_vec);
}

float Loss::calculate_regularization(const VectorR& parameters_vec) const
{
    if (regularization_method == Regularization::NoRegularization || regularization_weight == 0.0f) return 0.0f;

    const TensorView parameters(const_cast<float*>(parameters_vec.data()), { ssize(parameters_vec) });
    float penalty = 0.0f;

    if (regularization_method == Regularization::L1)
        l1_regularization(parameters, regularization_weight, penalty);
    else if (regularization_method == Regularization::L2)
        l2_regularization(parameters, regularization_weight, penalty);

    return penalty;
}

void Loss::calculate_layers_error_gradient(const Batch& batch,
                                           ForwardPropagation& forward_propagation,
                                           BackPropagation& back_propagation) const
{
    check_neural_network();

    const vector<unique_ptr<Layer>>& layers = neural_network->get_layers();
    const size_t layers_number = neural_network->get_layers_number();

    if (layers_number == 0) return;

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    {
        PROFILE_SCOPE("loss:calculate_output_deltas");
        calculate_output_deltas(batch, forward_propagation, back_propagation);
    }

    for (Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
    {
        if (i != last_trainable_layer_index)
        {
            PROFILE_SCOPE("bwd:accumulate_output_deltas");
            back_propagation.accumulate_output_deltas(static_cast<size_t>(i));
        }
        const std::string key = "bwd:" + layers[i]->get_name();
        PROFILE_SCOPE(key);
        layers[i]->back_propagate(forward_propagation, back_propagation, i);
    }
}

static const vector<pair<Loss::Error, string>> error_map = {
    {Loss::Error::MeanSquaredError,      "MeanSquaredError"},
    {Loss::Error::NormalizedSquaredError, "NormalizedSquaredError"},
    {Loss::Error::WeightedSquaredError,   "WeightedSquaredError"},
    {Loss::Error::CrossEntropy,           "CrossEntropy"},
    {Loss::Error::CrossEntropy3d,        "CrossEntropyError3d"},
    {Loss::Error::MinkowskiError,         "MinkowskiError"}
};

void Loss::set_error(const Error& new_error)
{
    error = new_error;

    for (const auto& [error_value, error_name] : error_map)
        if (error_value == error) { name = error_name; return; }
}

void Loss::set_error(const string& new_name)
{
    for (const auto& [error_value, error_name] : error_map)
        if (error_name == new_name) { set_error(error_value); return; }

    throw runtime_error("Unknown loss method: " + new_name);
}

void Loss::add_regularization_gradient(BackPropagation& back_propagation) const
{
    if (regularization_method == Regularization::NoRegularization || regularization_weight == 0.0f) return;

    check_neural_network();

    const Index parameters_number = neural_network->get_parameters_size();

    const TensorView parameters(neural_network->get_parameters_data(), { parameters_number });
    TensorView gradient(back_propagation.gradient.as<float>(), { parameters_number });

    if (regularization_method == Regularization::L1)
        l1_regularization_gradient(parameters, regularization_weight, gradient);
    else if (regularization_method == Regularization::L2)
        l2_regularization_gradient(parameters, regularization_weight, gradient);
}

void Loss::regularization_from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "Regularization");

    set_regularization(read_json_string(root_element, "Type"));

    if (root_element->has("RegularizationWeight"))
        set_regularization_weight(float(read_json_type(root_element, "RegularizationWeight")));
}

void Loss::regularization_to_JSON(JsonWriter& file_stream) const
{
    file_stream.open_element("Regularization");
    write_json(file_stream, {
        {"Type", regularization_to_string(regularization_method)},
        {"RegularizationWeight", to_string(regularization_weight)}
    });
    file_stream.close_element();
}

float Loss::calculate_h(const float x)
{
    static const float sqrt_eta = 1e-3f;

    return sqrt_eta * (1.0f + abs(x));
}

void Loss::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Loss");
    write_json(printer, {
        {"Method", get_name()},
        {"Regularization", regularization_to_string(regularization_method)},
        {"RegularizationWeight", to_string(regularization_weight)}
    });

    if (error == Error::NormalizedSquaredError)
        add_json_field(printer, "NormalizationCoefficient", to_string(normalization_coefficient));

    if (error == Error::WeightedSquaredError)
        write_json(printer, {
            {"PositivesWeight", to_string(positives_weight)},
            {"NegativesWeight", to_string(negatives_weight)}
        });

    if (error == Error::MinkowskiError)
        add_json_field(printer, "MinkowskiParameter", to_string(minkowski_parameter));

    printer.close_element();
}

void Loss::from_JSON(const JsonDocument& document)
{
    const Json* root = document.first_child("Loss");
    if (!root) throw runtime_error("Loss::from_JSON error: missing Loss element.");

    set_error(read_json_string(root, "Method"));

    set_regularization(read_json_string(root, "Regularization"));
    regularization_weight = read_json_type(root, "RegularizationWeight");

    if (root->first_child("NormalizationCoefficient"))
        normalization_coefficient = read_json_type(root, "NormalizationCoefficient");

    if (root->first_child("PositivesWeight")) {
        positives_weight = read_json_type(root, "PositivesWeight");
        negatives_weight = read_json_type(root, "NegativesWeight");
    }

    if (root->first_child("MinkowskiParameter"))
        minkowski_parameter = read_json_type(root, "MinkowskiParameter");
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
