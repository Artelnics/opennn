//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   I N D E X   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensor_utilities.h"
#include "dataset.h"
#include "loss.h"
#include "error_utilities.h"
#include "../eigen/Eigen/LU"

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

    regularization_method = "L2";
    set_error(Error::MeanSquaredError);

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

    add_regularization_gradient(back_propagation.gradient);
}

void Loss::calculate_error(const Batch& batch, const ForwardPropagation& forward_propagation, BackPropagation& back_propagation) const
{
    const TensorView input = forward_propagation.get_last_trainable_layer_outputs();
    const TensorView target = batch.get_targets();

    // workspace_device is used by CUDA to store intermediate diffs or CE values
#ifdef CUDA
    float* workspace_device = back_propagation.errors;
#else
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
        weighted_squared_error(input, target, positives_weight, negatives_weight, back_propagation.error, workspace_device);
        break;
    case Error::CrossEntropy:
        // Math utility handles logic based on num_classes (input.shape.back())
        if (input.shape.back() == 1)
            binary_cross_entropy(input, target, back_propagation.error, workspace_device);
        else
            categorical_cross_entropy(input, target, back_propagation.error, workspace_device);
        break;
    case Error::MinkowskiError:
        minkowski_error(input, target, minkowski_parameter, back_propagation.error, workspace_device);
        break;
    }
}

void Loss::calculate_output_gradients(const Batch& batch, const ForwardPropagation& forward_propagation, BackPropagation& back_propagation) const
{
    const TensorView input = forward_propagation.get_last_trainable_layer_outputs();
    const TensorView target = batch.get_targets();
    TensorView input_gradient = back_propagation.get_output_gradients();

    switch(error)
    {
    case Error::MeanSquaredError:
        mean_squared_error_gradient(input, target, input_gradient);
        break;

    case Error::NormalizedSquaredError:
        normalized_squared_error_gradient(input, target, normalization_coefficient, input_gradient);
        break;
    case Error::WeightedSquaredError:
        // Passing 1.0/N as coefficient to match MSE style if required
        weighted_squared_error_gradient(input, target, positives_weight, negatives_weight, 1.0f, input_gradient);
        break;
    case Error::CrossEntropy:
        cross_entropy_gradient(input, target, input_gradient);
        break;
    case Error::MinkowskiError:
        minkowski_error_gradient(input, target, minkowski_parameter, input_gradient);
        break;
    }
}

void Loss::add_regularization(BackPropagation& back_propagation) const
{
    if(regularization_method == "None") return;

    const VectorR& params_vec = neural_network->get_parameters();
    back_propagation.loss_value += calculate_regularization(params_vec);
}

type Loss::calculate_regularization(const VectorR& parameters_vec) const
{
    if(regularization_method == "None" || regularization_weight == 0.0f) return 0.0f;

    const TensorView parameters(const_cast<type*>(parameters_vec.data()), { static_cast<Index>(parameters_vec.size()) });
    type penalty = 0.0f;

    if (regularization_method == "L1")
        l1_regularization(parameters, regularization_weight, penalty);
    else if (regularization_method == "L2")
        l2_regularization(parameters, regularization_weight, penalty);

    return penalty;
}

void Loss::calculate_layers_error_gradient(const Batch& batch,
                                           ForwardPropagation& forward_propagation,
                                           BackPropagation& back_propagation) const
{
    const vector<unique_ptr<Layer>>& layers = neural_network->get_layers();
    const Index layers_number = neural_network->get_layers_number();

    if(layers_number == 0) return;

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    calculate_output_gradients(batch, forward_propagation, back_propagation);

    for (Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
        layers[i]->back_propagate(forward_propagation, back_propagation, i);
}

static const vector<pair<Loss::Error, string>> error_map = {
    {Loss::Error::MeanSquaredError,      "MeanSquaredError"},
    {Loss::Error::NormalizedSquaredError, "NormalizedSquaredError"},
    {Loss::Error::WeightedSquaredError,   "WeightedSquaredError"},
    {Loss::Error::CrossEntropy,           "CrossEntropy"},
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

void Loss::add_regularization_gradient(VectorR& gradient_vec) const
{
    if(regularization_method == "None" || regularization_weight == 0.0f) return;

    const VectorR& params_vec = neural_network->get_parameters();

    // Wrap vectors in views for hardware-agnostic utilities
    const TensorView parameters(const_cast<type*>(params_vec.data()), { static_cast<Index>(params_vec.size()) });
    TensorView gradient(reinterpret_cast<type*>(gradient_vec.data()), { static_cast<Index>(gradient_vec.size()) });

    if (regularization_method == "L1")
        l1_regularization_gradient(parameters, regularization_weight, gradient);
    else if (regularization_method == "L2")
        l2_regularization_gradient(parameters, regularization_weight, gradient);
}

void Loss::regularization_from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = get_xml_root(document, "Regularization");

    const string new_regularization_method = root_element->Attribute("Type");

    set_regularization(new_regularization_method);

    const XMLElement* element = root_element->FirstChildElement("RegularizationWeight");

    if(element)
    {
        const type new_regularization_weight = type(atof(element->GetText()));

        try
        {
            set_regularization_weight(new_regularization_weight);
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }
}

void Loss::write_regularization_XML(XMLPrinter& file_stream) const
{
    file_stream.OpenElement("Regularization");

    file_stream.PushAttribute("Type", regularization_method.c_str());

    // Regularization weight

    file_stream.OpenElement("RegularizationWeight");
    file_stream.PushText(to_string(regularization_weight).c_str());
    file_stream.CloseElement();

    // Close regularization

    file_stream.CloseElement();
}

BackPropagation::BackPropagation(const Index new_batch_size, Loss* new_loss)
{
    set(new_batch_size, new_loss);
}

void BackPropagation::set(const Index new_batch_size, Loss* new_loss)
{
    batch_size = new_batch_size;
    loss = new_loss;

    if(!loss) return;

    const NeuralNetwork* neural_network = loss->get_neural_network();
    if(!neural_network) return;

    const Index layers_number = neural_network->get_layers_number();

    const vector<vector<Shape>> parameter_shapes = neural_network->get_parameter_shapes();

    Index total_parameters_size = 0;

    for(const auto& layer_shapes : parameter_shapes)
        for(const Shape& s : layer_shapes)
            total_parameters_size += get_aligned_size(s.size());

    gradient.resize(total_parameters_size);
    gradient.setZero();

    gradient_views.resize(layers_number);
    type* g_ptr = (total_parameters_size > 0) ? gradient.data() : nullptr;

    for(Index i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& layer_param_shapes = parameter_shapes[i];
        gradient_views[i].resize(layer_param_shapes.size());

        for(size_t j = 0; j < layer_param_shapes.size(); ++j)
        {
            const Shape& s = layer_param_shapes[j];
            if(s.size() > 0 && g_ptr)
            {
                gradient_views[i][j] = TensorView(g_ptr, s);
                g_ptr += get_aligned_size(s.size());
            }
        }
    }

    const vector<vector<Shape>> backward_shapes = neural_network->get_backward_shapes(batch_size);

    Index total_backward_size = 0;

    for(const auto& layer_shapes : backward_shapes)
        for(const Shape& s : layer_shapes)
            total_backward_size += get_aligned_size(s.size());

    backward.resize(total_backward_size);
    backward.setZero();

    backward_views.resize(layers_number);
    type* b_ptr = (total_backward_size > 0) ? backward.data() : nullptr;

    for(Index i = 0; i < layers_number; ++i)
    {
        const vector<Shape>& shapes = backward_shapes[i];
        const size_t slots = shapes.size();

        // Slot 0: Reserved for OutputGradients (wired from downstream)
        // Slot 1..N: Internal gradients/InputGradients allocated from backward_shapes
        backward_views[i].resize(slots + 1);
        backward_views[i][0].resize(1); // One OutputGradient view per layer

        for(size_t j = 0; j < slots; ++j)
        {
            const Shape& s = shapes[j];
            backward_views[i][j + 1].resize(1);

            if(s.size() > 0 && b_ptr)
            {
                backward_views[i][j + 1][0] = TensorView(b_ptr, s);
                b_ptr += get_aligned_size(s.size());
            }
        }
    }

    const Shape output_shape = neural_network->get_output_shape();
    const Index outputs_number = output_shape[0];

    loss_value = type(0);
    error = type(0);
    built_mask = false;
    accuracy.setZero();

    errors.resize(batch_size, outputs_number);

    output_gradient_dimensions = Shape({batch_size}).append(output_shape);

    const Index total_output_elements = output_shape.size() * batch_size;
    output_gradients.resize(total_output_elements);
    output_gradients.setZero();

    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();
    const auto& layer_output_indices = neural_network->get_layer_output_indices();
    const auto& layer_input_indices = neural_network->get_layer_input_indices();

    for(Index i = 0; i < layers_number; ++i)
    {
        if(backward_views[i].empty()) continue;

        if(i == last_trainable_layer_index)
        {
            // The very last trainable layer gets gradients from the loss's main output_gradients buffer
            backward_views[i][0][0] = TensorView(output_gradients.data(), output_gradient_dimensions);
        }
        else
        {
            // Internal layers connect to their consumers
            for(const Index consumer_idx : layer_output_indices[i])
            {
                if(consumer_idx >= 0 && consumer_idx < layers_number)
                {
                    // Find which input port of the consumer layer connects to layer i
                    const auto& consumer_inputs = layer_input_indices[consumer_idx];

                    Index port = 0;

                    for(Index p = 0; p < static_cast<Index>(consumer_inputs.size()); ++p)
                        if(consumer_inputs[p] == i)
                        {
                            port = p;
                            break;
                        }

                    // By convention, Layer InputGradients are stored in slot 1 of backward_views
                    if(backward_views[consumer_idx].size() > 1 && !backward_views[consumer_idx][1].empty())
                        backward_views[i][0][0] = backward_views[consumer_idx][1][0];
                }
                break; // Standard implementation assumes single-path gradient flow for simplicity here
            }
        }
    }
}

vector<vector<TensorView>> BackPropagation::get_layer_gradients() const
{
    const NeuralNetwork* neural_network_ptr = loss->get_neural_network();

    const Index layers_number = neural_network_ptr->get_layers_number();

    const vector<vector<Index>>& layer_input_indices = neural_network_ptr->get_layer_input_indices();
    const vector<vector<Index>> layer_output_indices = neural_network_ptr->get_layer_output_indices();

    vector<TensorView> const input_gradient_views;

    vector<vector<TensorView>> layer_gradient_views(layers_number);
/*
    const Index first_trainable_layer_index = neural_network_ptr->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network_ptr->get_last_trainable_layer_index();

    for(Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
    {
        if (i == last_trainable_layer_index)
        {
            layer_gradient_views[i].push_back(get_output_gradients());

            continue;
        }

        for(Index j = 0; j < Index(layer_output_indices[i].size()); j++)
        {
            const Index output_index = layer_output_indices[i][j];
            const Index input_index = neural_network_ptr->find_input_index(layer_input_indices[output_index], i);

            input_gradient_views = layer_back_propagations[output_index]->get_input_gradients();

            layer_gradient_views[i].push_back(input_gradient_views[input_index]);
        }
    }
*/
    return layer_gradient_views;

}

TensorView BackPropagation::get_output_gradients() const
{
    return {const_cast<type*>(output_gradients.data()), output_gradient_dimensions};
}

void BackPropagation::print() const
{
    cout << "Back-propagation" << endl
         << "Errors:" << endl
         << errors << endl
         << "Error:" << endl
         << error << endl
         << "Loss:" << endl
         << loss << endl;
    //<< "Gradient:" << endl
    //<< gradient << endl;
}

type Loss::calculate_numerical_error() const
{  
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

    return back_propagation.gradient;
}

VectorR Loss::calculate_numerical_gradient()
{
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

    VectorR parameters_forward = parameters;
    VectorR parameters_backward = parameters;

    type error_forward = 0;
    type error_backward = 0;

    VectorR numerical_gradient(parameters_number);
    numerical_gradient.setConstant(type(0));

    for(Index i = 0; i < parameters_number; i++)
    {
        h = calculate_h(parameters(i));

        parameters_forward(i) += h;

        neural_network->forward_propagate(batch.get_inputs(),
                                          parameters_forward,
                                          forward_propagation);

        calculate_error(batch, forward_propagation, back_propagation);

        error_forward = back_propagation.error;

        parameters_forward(i) -= h;
        parameters_backward(i) -= h;

        neural_network->forward_propagate(batch.get_inputs(),
                                          parameters_backward,
                                          forward_propagation);

        calculate_error(batch, forward_propagation, back_propagation);

        error_backward = back_propagation.error;

        parameters_backward(i) += h;

        numerical_gradient(i) = (error_forward - error_backward)/type(2*h);
    }

    return numerical_gradient;
}

VectorR Loss::calculate_numerical_input_gradients()
{
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
    numerical_inputs_gradients.setConstant(type(0));

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
    const Index precision_digits = 6;

    const type eta = pow(type(10.0), type(-1.0 * precision_digits));

    return sqrt(eta)*(type(1) + abs(x));
}

void Loss::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Loss");
    write_xml_properties(printer, {
        {"Method", get_name()},
        {"Regularization", regularization_method},
        {"RegularizationWeight", to_string(regularization_weight)}
    });

    if (error == Error::NormalizedSquaredError)
        add_xml_element(printer, "NormalizationCoefficient", to_string(normalization_coefficient));

    if (error == Error::WeightedSquaredError) {
        add_xml_element(printer, "PositivesWeight", to_string(positives_weight));
        add_xml_element(printer, "NegativesWeight", to_string(negatives_weight));
    }

    if (error == Error::MinkowskiError)
        add_xml_element(printer, "MinkowskiParameter", to_string(minkowski_parameter));

    printer.CloseElement();
}

void Loss::from_XML(const XMLDocument& document)
{
    const XMLElement* root = document.FirstChildElement("Loss");
    if(!root) return;

    set_error(read_xml_string(root, "Method"));

    regularization_method = read_xml_string(root, "Regularization");
    regularization_weight = read_xml_type(root, "RegularizationWeight");

    if (root->FirstChildElement("NormalizationCoefficient"))
        normalization_coefficient = read_xml_type(root, "NormalizationCoefficient");

    if (root->FirstChildElement("PositivesWeight")) {
        positives_weight = read_xml_type(root, "PositivesWeight");
        negatives_weight = read_xml_type(root, "NegativesWeight");
    }

    if (root->FirstChildElement("MinkowskiParameter"))
        minkowski_parameter = read_xml_type(root, "MinkowskiParameter");
}

} 

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
