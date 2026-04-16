//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   I N D E X   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensor_utilities.h"
#include "math_utilities.h"
#include "batch.h"
#include "dataset.h"
#include "loss.h"
#include "error_utilities.h"
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
    // Defaults — overwritten below for losses that need data-derived values.
    normalization_coefficient = type(1);
    positives_weight = type(1);
    negatives_weight = type(1);

    if(!dataset || dataset->get_samples_number() == 0)
        return;

    if(error == Error::WeightedSquaredError)
    {
        const Index targets_number = dataset->get_features_number("Target");
        if(targets_number != 1) return;  // only for single-target binary

        const VectorI distribution = dataset->calculate_target_distribution();
        const Index negatives = distribution(0);
        const Index positives = distribution(1);

        if(positives == 0 || negatives == 0) return;

        negatives_weight = type(1);
        positives_weight = type(negatives) / type(positives);
        normalization_coefficient = type(negatives) * negatives_weight * type(0.5);
    }
}

void Loss::back_propagate(const Batch& batch,
                          ForwardPropagation& forward_propagation,
                          BackPropagation& back_propagation) const
{
    if(batch.is_empty()) return;

    calculate_error(batch, forward_propagation, back_propagation);

    calculate_layers_error_gradient(batch, forward_propagation, back_propagation);

    back_propagation.loss_value = back_propagation.error;

    // Regularization

    add_regularization(back_propagation);

    add_regularization_gradient(back_propagation);
}

void Loss::calculate_error(const Batch& batch, const ForwardPropagation& forward_propagation, BackPropagation& back_propagation) const
{
    const TensorView input = forward_propagation.get_last_trainable_layer_outputs();

#ifdef OPENNN_WITH_CUDA
    const TensorView target = Device::instance().is_gpu()
                                  ? batch.get_targets_device()
                                  : batch.get_targets();
    float* workspace_device = Device::instance().is_gpu()
                                  ? back_propagation.errors_device
                                  : nullptr;
#else
    const TensorView target = batch.get_targets();
    float* workspace_device = nullptr;
#endif

    switch(error)
    {
    case Error::MeanSquaredError:
        mean_squared_error(input, target, back_propagation.error, workspace_device);
        break;
    case Error::NormalizedSquaredError:
        normalized_squared_error(input, target, normalization_coefficient, back_propagation.error, workspace_device);
        break;
    case Error::WeightedSquaredError:
    {
        weighted_squared_error(input, target, positives_weight, negatives_weight, back_propagation.error, workspace_device);
        const Index total = dataset ? dataset->get_samples_number() : batch.get_samples_number();
        const Index samples = batch.get_samples_number();
        const type coefficient = type(total) / (type(samples) * (normalization_coefficient + EPSILON));
        back_propagation.error *= coefficient;
        break;
    }
    case Error::CrossEntropy:
        if (input.shape.back() == 1)
            binary_cross_entropy(input, target, back_propagation.error, workspace_device);
        else
            categorical_cross_entropy(input, target, back_propagation.error, workspace_device);
        break;
    case Error::CrossEntropy3d:
        cross_entropy_3d(input, target, back_propagation.error, back_propagation.active_tokens_count, workspace_device);
        break;
    case Error::MinkowskiError:
        minkowski_error(input, target, minkowski_parameter, back_propagation.error, workspace_device);
        break;
    }
}

void Loss::calculate_output_gradients(const Batch& batch, const ForwardPropagation& forward_propagation, BackPropagation& back_propagation) const
{
    const TensorView input = forward_propagation.get_last_trainable_layer_outputs();

#ifdef OPENNN_WITH_CUDA
    const TensorView target = Device::instance().is_gpu()
                                  ? batch.get_targets_device()
                                  : batch.get_targets();
    TensorView input_gradient = Device::instance().is_gpu()
                                    ? back_propagation.get_output_gradients_device()
                                    : back_propagation.get_output_gradients();
#else
    const TensorView target = batch.get_targets();
    TensorView input_gradient = back_propagation.get_output_gradients();
#endif

    switch(error)
    {
    case Error::MeanSquaredError:
        mean_squared_error_gradient(input, target, input_gradient);
        break;

    case Error::NormalizedSquaredError:
        normalized_squared_error_gradient(input, target, normalization_coefficient, input_gradient);
        break;
    case Error::WeightedSquaredError:
    {
        const Index total = dataset ? dataset->get_samples_number() : batch.get_samples_number();
        const Index samples = batch.get_samples_number();
        const type coefficient = type(total) / (type(samples) * (normalization_coefficient + EPSILON));
        weighted_squared_error_gradient(input, target, positives_weight, negatives_weight, coefficient, input_gradient);
        break;
    }
    case Error::CrossEntropy:
        cross_entropy_gradient(input, target, input_gradient);
        break;
    case Error::CrossEntropy3d:
        cross_entropy_3d_gradient(input, target, input_gradient, back_propagation.active_tokens_count);
        break;
    case Error::MinkowskiError:
        minkowski_error_gradient(input, target, minkowski_parameter, input_gradient);
        break;
    }
}

void Loss::add_regularization(BackPropagation& back_propagation) const
{
    if(regularization_method == Regularization::NoRegularization) return;

    check_neural_network();

#ifdef OPENNN_WITH_CUDA
    if (Device::instance().is_gpu()) {
        // In the master, CUDA regularization value is not computed on GPU either
        // Just skip the loss_value update (gradient is the important part)
        return;
    }
#endif
    const VectorR& params_vec = neural_network->get_parameters();
    back_propagation.loss_value += calculate_regularization(params_vec);
}

type Loss::calculate_regularization(const VectorR& parameters_vec) const
{
    if(regularization_method == Regularization::NoRegularization || regularization_weight == 0.0f) return 0.0f;

    const TensorView parameters(const_cast<type*>(parameters_vec.data()), { static_cast<Index>(parameters_vec.size()) });
    type penalty = 0.0f;

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

    if(layers_number == 0) return;

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    calculate_output_gradients(batch, forward_propagation, back_propagation);

    for (Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
    {
        if(i != last_trainable_layer_index)
            back_propagation.accumulate_output_gradients(static_cast<size_t>(i));
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

    for(const auto& [e, n] : error_map)
        if (e == error) { name = n; return; }
}

void Loss::set_error(const string& new_name)
{
    for(const auto& [e, n] : error_map)
        if (n == new_name) { set_error(e); return; }

    throw runtime_error("Unknown loss method: " + new_name);
}

void Loss::add_regularization_gradient(BackPropagation& back_propagation) const
{
    if(regularization_method == Regularization::NoRegularization || regularization_weight == 0.0f) return;

    check_neural_network();

    const Index n = neural_network->get_parameters_size();

#ifdef OPENNN_WITH_CUDA
    const TensorView parameters = Device::instance().is_gpu()
        ? TensorView(neural_network->get_parameters_device(), { n })
        : TensorView(const_cast<type*>(neural_network->get_parameters().data()), { n });
    TensorView gradient = Device::instance().is_gpu()
        ? TensorView(back_propagation.gradient.device(), { n })
        : TensorView(back_propagation.gradient.data(), { n });
#else
    const TensorView parameters(const_cast<type*>(neural_network->get_parameters().data()), { n });
    TensorView gradient(back_propagation.gradient.data(), { n });
#endif

    if (regularization_method == Regularization::L1)
        l1_regularization_gradient(parameters, regularization_weight, gradient);
    else if (regularization_method == Regularization::L2)
        l2_regularization_gradient(parameters, regularization_weight, gradient);
}

void Loss::regularization_from_XML(const XmlDocument& document)
{
    const XmlElement* root_element = get_xml_root(document, "Regularization");

    const string new_regularization_method = root_element->attribute("Type");

    set_regularization(new_regularization_method);

    const XmlElement* element = root_element->first_child_element("RegularizationWeight");

    if(element)
    {
        const type new_regularization_weight = type(atof(element->get_text()));

        set_regularization_weight(new_regularization_weight);
    }
}

void Loss::regularization_to_XML(XmlPrinter& file_stream) const
{
    file_stream.open_element("Regularization");

    file_stream.push_attribute("Type", regularization_to_string(regularization_method).c_str());

    // Regularization weight

    file_stream.open_element("RegularizationWeight");
    file_stream.push_text(to_string(regularization_weight).c_str());
    file_stream.close_element();

    // Close regularization

    file_stream.close_element();
}

type Loss::calculate_numerical_error() const
{
    check_neural_network();
    check_dataset();

    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> training_indices = dataset->get_sample_indices("Training");

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> decoder_feature_indices = dataset->get_feature_indices("Decoder");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);

    batch.fill(training_indices, input_feature_indices, decoder_feature_indices, target_feature_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    neural_network->forward_propagate(batch.get_inputs(), forward_propagation);

    BackPropagation back_propagation(samples_number, const_cast<Loss*>(this));

    calculate_error(batch, forward_propagation, back_propagation);

    return back_propagation.error;
}

VectorR Loss::calculate_gradient()
{
    check_neural_network();
    check_dataset();

    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> training_indices = dataset->get_sample_indices("Training");

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> decoder_feature_indices = dataset->get_feature_indices("Decoder");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);
    batch.fill(training_indices, input_feature_indices, decoder_feature_indices, target_feature_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagation back_propagation(samples_number, this);

    const VectorR& parameters = neural_network->get_parameters();

    neural_network->forward_propagate(batch.get_inputs(),
                                      parameters,
                                      forward_propagation);

    back_propagate(batch, forward_propagation, back_propagation);

    return back_propagation.gradient.vector;
}

VectorR Loss::calculate_numerical_gradient()
{
    check_neural_network();
    check_dataset();

    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> training_indices = dataset->get_sample_indices("Training");

    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> decoder_feature_indices = dataset->get_feature_indices("Decoder");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);
    batch.fill(training_indices, input_feature_indices, decoder_feature_indices, target_feature_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagation back_propagation(samples_number, this);

    VectorR& parameters = neural_network->get_parameters();

    const Index parameters_number = parameters.size();

    type h = 0;

    VectorR perturbed = parameters;

    type error_forward = 0;
    type error_backward = 0;

    VectorR numerical_gradient(parameters_number);
    numerical_gradient.setZero();

    for(Index i = 0; i < parameters_number; i++)
    {
        h = calculate_h(parameters(i));

        perturbed(i) += h;

        neural_network->forward_propagate(batch.get_inputs(),
                                          perturbed,
                                          forward_propagation);

        calculate_error(batch, forward_propagation, back_propagation);

        error_forward = back_propagation.error;

        perturbed(i) -= type(2) * h;

        neural_network->forward_propagate(batch.get_inputs(),
                                          perturbed,
                                          forward_propagation);

        calculate_error(batch, forward_propagation, back_propagation);

        error_backward = back_propagation.error;

        perturbed(i) += h;

        numerical_gradient(i) = (error_forward - error_backward)/type(2*h);
    }

    return numerical_gradient;
}

VectorR Loss::calculate_numerical_input_gradients()
{
    check_neural_network();
    check_dataset();

    const Index samples_number = dataset->get_samples_number("Training");

    const Index values_number = neural_network->get_inputs_number()*samples_number;

    const vector<Index> sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);
    batch.fill(sample_indices, input_feature_indices, {}, target_feature_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagation back_propagation(samples_number, this);

    type h;

    type error_forward;
    type error_backward;

    VectorR numerical_inputs_gradients(values_number);
    numerical_inputs_gradients.setZero();

    const vector<TensorView>& input_views = batch.get_inputs();

    TensorMap4 inputs_vector = tensor_map<4>(input_views[0]);

    for(Index i = 0; i < values_number; i++)
    {
        h = calculate_h(inputs_vector(i));

        input_views[0].data[i] += h;

        neural_network->forward_propagate(input_views, forward_propagation);

        calculate_error(batch, forward_propagation, back_propagation);
        error_forward = back_propagation.error;

        input_views[0].data[i] -= 2*h;

        neural_network->forward_propagate(input_views, forward_propagation);

        calculate_error(batch, forward_propagation, back_propagation);
        error_backward = back_propagation.error;

        input_views[0].data[i] += h;

        numerical_inputs_gradients(i) = (error_forward - error_backward) / type(2 * h);
    }

    return numerical_inputs_gradients;
}

MatrixR Loss::calculate_inverse_hessian()
{
    MatrixR numerical_hessian = calculate_numerical_hessian();
    const Index parameters_number = numerical_hessian.rows();

    using MatrixType = Matrix<type, Dynamic, Dynamic, ColMajor>;
    Map<MatrixType> hessian_map(numerical_hessian.data(), parameters_number, parameters_number);

    FullPivLU<MatrixType> const hessian_decomposition(hessian_map);

    if(!hessian_decomposition.isInvertible())
    {
        MatrixType hessian_damped = hessian_map + MatrixType::Identity(parameters_number, parameters_number) * 1e-4;

        FullPivLU<MatrixType> const hessian_decomposition_damped(hessian_damped);

        const MatrixType hessian_map_inverse = hessian_decomposition_damped.inverse();

        MatrixR hessian_inverse(parameters_number, parameters_number);
        Map<MatrixType>(hessian_inverse.data(), parameters_number, parameters_number) = hessian_map_inverse;

        return hessian_inverse;
    }

    const MatrixType hessian_map_inverse = hessian_decomposition.inverse();
    MatrixR hessian_inverse(parameters_number, parameters_number);
    Map<MatrixType>(hessian_inverse.data(), parameters_number, parameters_number) = hessian_map_inverse;

    return hessian_inverse;
}

type Loss::calculate_h(const type x)
{
    static const type sqrt_eta = type(1e-3); // sqrt(1e-6)

    return sqrt_eta * (type(1) + abs(x));
}

void Loss::to_XML(XmlPrinter& printer) const
{
    printer.open_element("Loss");
    write_xml_properties(printer, {
        {"Method", get_name()},
        {"Regularization", regularization_to_string(regularization_method)},
        {"RegularizationWeight", to_string(regularization_weight)}
    });

    if (error == Error::NormalizedSquaredError)
        add_xml_element(printer, "NormalizationCoefficient", to_string(normalization_coefficient));

    if (error == Error::WeightedSquaredError)
        write_xml_properties(printer, {
            {"PositivesWeight", to_string(positives_weight)},
            {"NegativesWeight", to_string(negatives_weight)}
        });

    if (error == Error::MinkowskiError)
        add_xml_element(printer, "MinkowskiParameter", to_string(minkowski_parameter));

    printer.close_element();
}

void Loss::from_XML(const XmlDocument& document)
{
    const XmlElement* root = document.first_child_element("Loss");
    if(!root) throw runtime_error("Loss::from_XML error: missing Loss element.");

    set_error(read_xml_string(root, "Method"));

    set_regularization(read_xml_string(root, "Regularization"));
    regularization_weight = read_xml_type(root, "RegularizationWeight");

    if (root->first_child_element("NormalizationCoefficient"))
        normalization_coefficient = read_xml_type(root, "NormalizationCoefficient");

    if (root->first_child_element("PositivesWeight")) {
        positives_weight = read_xml_type(root, "PositivesWeight");
        negatives_weight = read_xml_type(root, "NegativesWeight");
    }

    if (root->first_child_element("MinkowskiParameter"))
        minkowski_parameter = read_xml_type(root, "MinkowskiParameter");
}

MatrixR Loss::calculate_numerical_hessian()
{
    // @todo Stub - not yet refactored
    const VectorR gradient = calculate_numerical_gradient();
    const Index n = gradient.size();
    return MatrixR::Zero(n, n);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
