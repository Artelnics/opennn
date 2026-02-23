//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   I N D E X   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "dataset.h"
#include "loss_index.h"
#include "cross_entropy_error_3d.h"
#include "../eigen/Eigen/LU"

namespace opennn
{

Loss::Loss(const NeuralNetwork* new_neural_network, const Dataset* new_dataset)
{
    set(new_neural_network, new_dataset);
}


type Loss::get_regularization_weight() const
{
    return regularization_weight;
}


bool Loss::get_display() const
{
    return display;
}


bool Loss::has_neural_network() const
{
    return neural_network;
}


bool Loss::has_dataset() const
{
    return dataset;
}


string Loss::get_regularization_method() const
{
    return regularization_method;
}


void Loss::set(const NeuralNetwork* new_neural_network, const Dataset* new_dataset)
{
    neural_network = const_cast<NeuralNetwork*>(new_neural_network);
    dataset = const_cast<Dataset*>(new_dataset);

    regularization_method = "L2";
}


void Loss::set_neural_network(const NeuralNetwork* new_neural_network)
{
    neural_network = const_cast<NeuralNetwork*>(new_neural_network);
}


void Loss::set_dataset(const Dataset* new_dataset)
{
    dataset = const_cast<Dataset*>(new_dataset);
}


void Loss::set_regularization_method(const string& new_regularization_method)
{
    regularization_method = new_regularization_method;
}


void Loss::set_regularization_weight(const type new_regularization_weight)
{
    regularization_weight = new_regularization_weight;
}


void Loss::set_display(bool new_display)
{
    display = new_display;
}


void Loss::calculate_errors_lm(const Batch& batch,
                                    const ForwardPropagation & forward_propagation,
                                    BackPropagationLM & back_propagation) const
{
    const TensorView outputs_view
        = forward_propagation.get_last_trainable_layer_outputs();

    const MatrixMap outputs = matrix_map(outputs_view);

    const TensorView targets_view = batch.get_targets();

    const MatrixMap targets = matrix_map(targets_view);

    back_propagation.errors = outputs - targets;
}


void Loss::calculate_squared_errors_lm(const Batch&,
                                       const ForwardPropagation&,
                                       BackPropagationLM& back_propagation_lm) const
{
    const MatrixR& errors = back_propagation_lm.errors;

    VectorR& squared_errors = back_propagation_lm.squared_errors;

    squared_errors = errors.rowwise().norm();
}


void Loss::back_propagate(const Batch& batch,
                               ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation) const
{
    if(batch.is_empty()) return;

    calculate_error(batch, forward_propagation, back_propagation);

    calculate_layers_error_gradient(batch, forward_propagation, back_propagation);

    back_propagation.loss = back_propagation.error;

    // Regularization

    add_regularization(back_propagation);

    add_regularization_to_gradients(back_propagation);
}


void Loss::add_regularization(BackPropagation& back_propagation) const
{
    if(regularization_method == "None" || regularization_weight == 0)
        return;

    const VectorR& parameters = neural_network->get_parameters();

    back_propagation.loss += calculate_regularization(parameters);
}


void Loss::add_regularization_lm(BackPropagationLM& back_propagation_lm) const
{
    if(regularization_method == "None")
        return;

    const VectorR& parameters = neural_network->get_parameters();

    type& loss = back_propagation_lm.loss;

    VectorR& gradient = back_propagation_lm.gradient;

    MatrixR& hessian = back_propagation_lm.hessian;

    if(regularization_method == "L1")
    {
        loss += regularization_weight * parameters.lpNorm<1>();

        gradient.array() += regularization_weight * parameters.array().unaryExpr([](type v) {
            return (v > 0) ? 1.0f : ((v < 0) ? -1.0f : 0.0f);
        });
    }
    else if(regularization_method == "L2")
    {        
        loss += static_cast<type>(0.5) * regularization_weight * parameters.squaredNorm();

        gradient += parameters * regularization_weight;

        hessian.diagonal().array() += regularization_weight;
    }
    else
        throw runtime_error("Unknown regularization method: " + regularization_method);
}


void Loss::back_propagate_lm(const Batch& batch,
                                  ForwardPropagation& forward_propagation,
                                  BackPropagationLM& back_propagation_lm) const
{
    calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

    calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

    calculate_error_lm(batch, forward_propagation, back_propagation_lm);

    calculate_layers_squared_errors_jacobian_lm(batch, forward_propagation, back_propagation_lm);

    calculate_error_gradient_lm(batch, back_propagation_lm);

    calculate_error_hessian_lm(batch, back_propagation_lm);

    back_propagation_lm.loss = back_propagation_lm.error;

    add_regularization_lm(back_propagation_lm);
}


void Loss::calculate_layers_squared_errors_jacobian_lm(const Batch& batch,
                                                            ForwardPropagation& forward_propagation,
                                                            BackPropagationLM& back_propagation_lm) const
{
    const Index layers_number = neural_network->get_layers_number();

    if (layers_number == 0) return;

    back_propagation_lm.squared_errors_jacobian.setZero();

    const vector<unique_ptr<Layer>>& layers = neural_network->get_layers();

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const vector<vector<TensorView>> layer_input_views
        = forward_propagation.get_layer_input_views(batch.get_inputs(), true);

    const vector<vector<TensorView>> layer_delta_views
        = back_propagation_lm.get_layer_gradients();

    calculate_output_gradients_lm(batch, forward_propagation, back_propagation_lm);

    for(Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
        layers[i]->back_propagate_lm(layer_input_views[i],
                                     layer_delta_views[i],
                                     forward_propagation.layers[i],
                                     back_propagation_lm.neural_network.layers[i]);

    const vector<Index> layer_parameter_numbers = neural_network->get_layer_parameter_numbers();

    const Index alignment_elements = EIGEN_MAX_ALIGN_BYTES / sizeof(type);
    const Index mask_elements = ~(alignment_elements - 1);

    Index index = 0;
    for(Index i = 0; i < layers_number; i++)
    {
        layers[i]->insert_squared_errors_Jacobian_lm(back_propagation_lm.neural_network.layers[i], index, back_propagation_lm.squared_errors_jacobian);

        for(const TensorView* tensor_view : layers[i]->get_parameter_views())
        {
            const Index view_size = tensor_view->size();
            if(view_size > 0)
                index += (view_size + alignment_elements - 1) & mask_elements;
        }
    }
}


void Loss::calculate_error_gradient_lm(const Batch&,
                                       BackPropagationLM& back_propagation_lm) const
{
    const VectorR& squared_errors = back_propagation_lm.squared_errors;
    const MatrixR& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    VectorR& gradient = back_propagation_lm.gradient;

    gradient.noalias() = squared_errors_jacobian.transpose() * squared_errors;
}


string Loss::get_name() const
{
    return name;
}


type Loss::calculate_regularization(const VectorR& parameters) const
{
    if(regularization_method == "None")
        return type(0);
    else if(regularization_method == "L1")
        return regularization_weight * parameters.lpNorm<1>();
    else if(regularization_method == "L2")
        return 0.5f * regularization_weight * parameters.squaredNorm();
    else
        throw runtime_error("Unknown regularization method: " + regularization_method);
}


void Loss::calculate_layers_error_gradient(const Batch& batch,
                                                ForwardPropagation& forward_propagation,
                                                BackPropagation& back_propagation) const
{
    const vector<unique_ptr<Layer>>& layers = neural_network->get_layers();
    const Index layers_number = layers.size();

    if(layers_number == 0) return;

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const vector<vector<TensorView>> layer_input_views
        = forward_propagation.get_layer_input_views(batch.get_inputs(), true);

    const vector<vector<TensorView>> layer_delta_views
        = back_propagation.get_layer_gradients();

    calculate_output_gradients(batch, forward_propagation, back_propagation);

    for (Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
        layers[i]->back_propagate(layer_input_views[i],
            layer_delta_views[i],
            forward_propagation.layers[i],
            back_propagation.neural_network.layers[i]);
}


void Loss::add_regularization_gradient(VectorR& gradient) const
{
    if (regularization_method == "None" || regularization_weight == 0)
        return;

    const VectorR& parameters = neural_network->get_parameters();

    if (regularization_method == "L1")
    {
        VectorR l1_gradient = parameters.unaryExpr([](type w) {
            if (w > 0) return type(1);
            if (w < 0) return type(-1);
            return type(0);
        });

        gradient += l1_gradient * regularization_weight;
    }
    else if (regularization_method == "L2")
    {
        gradient += parameters * regularization_weight;
    }
    else
    {
        throw runtime_error("Unknown regularization method: " + regularization_method);
    }
}


void Loss::add_regularization_to_gradients(BackPropagation& back_propagation) const
{
    if(regularization_method == "None" || regularization_weight == 0)
        return;

    NeuralNetwork* neural_network = back_propagation.loss_index->get_neural_network();

    const VectorR& parameters = neural_network->get_parameters();

    VectorR& gradient = back_propagation.neural_network.gradient;

    if(regularization_method == "L1")
        gradient.array() += regularization_weight * parameters.array().sign();
    else if(regularization_method == "L2")
        gradient += parameters*regularization_weight;
    else
        throw runtime_error("Unknown regularization method: " + regularization_method);
}


void Loss::to_XML(XMLPrinter& file_stream) const
{
    file_stream.OpenElement("Loss");

    file_stream.CloseElement();
}


void Loss::regularization_from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("Regularization");

    if(!root_element)
        throw runtime_error("Regularization tag not found.\n");

    const string new_regularization_method = root_element->Attribute("Type");

    set_regularization_method(new_regularization_method);

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


void Loss::from_XML(const XMLDocument& document)
{
    const XMLElement* root_element = document.FirstChildElement("MeanSquaredError");

    if(!root_element)
        throw runtime_error("Mean squared element is nullptr.\n");

    // Regularization

    XMLDocument regularization_document;
    const XMLElement* regularization_element = root_element->FirstChildElement("Regularization");
    regularization_document.InsertFirstChild(regularization_element->DeepClone(&regularization_document));
    regularization_from_XML(regularization_document);
}


BackPropagation::BackPropagation(const Index new_batch_size, const Loss* new_loss)
{
    set(new_batch_size, new_loss);
}


void BackPropagation::set(const Index new_samples_number, const Loss* new_loss)
{
    samples_number = new_samples_number;

    loss_index = const_cast<Loss*>(new_loss);

    if(!loss_index) return;

    // Neural network

    NeuralNetwork* neural_network_ptr = loss_index->get_neural_network();

    const Shape output_shape = neural_network_ptr->get_output_shape();

    const Index outputs_number = output_shape[0];

    // First order loss

    neural_network.set(samples_number, neural_network_ptr);

    loss = type(0);

    errors.resize(samples_number, outputs_number);

    errors_weights.resize(samples_number, outputs_number);

    output_gradient_dimensions = { samples_number };
    output_gradient_dimensions.insert(output_gradient_dimensions.end(), output_shape.begin(), output_shape.end());

    const Index size = accumulate(output_shape.begin(), output_shape.end(), samples_number, multiplies<>());

    output_gradients.resize(size);

    if(is_instance_of<CrossEntropyError3d>(loss_index))
    {
        predictions.resize(samples_number, outputs_number);
        matches.resize(samples_number, outputs_number);
        mask.resize(samples_number, outputs_number);
    }
}


vector<vector<TensorView>> BackPropagation::get_layer_gradients() const
{
    NeuralNetwork* neural_network_ptr = loss_index->get_neural_network();

    const Index layers_number = neural_network_ptr->get_layers_number();

    const vector<vector<Index>>& layer_input_indices = neural_network_ptr->get_layer_input_indices();
    const vector<vector<Index>> layer_output_indices = neural_network_ptr->get_layer_output_indices();
    const vector<unique_ptr<LayerBackPropagation>>& layer_back_propagations = neural_network.get_layers();

    vector<TensorView> input_gradient_views;

    vector<vector<TensorView>> layer_delta_views(layers_number);

    const Index first_trainable_layer_index = neural_network_ptr->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network_ptr->get_last_trainable_layer_index();

    for(Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
    {
        if (i == last_trainable_layer_index)
        {
            layer_delta_views[i].push_back(get_output_gradients());

            continue;
        }

        for(Index j = 0; j < Index(layer_output_indices[i].size()); j++)
        {
            const Index output_index = layer_output_indices[i][j];
            const Index input_index = neural_network_ptr->find_input_index(layer_input_indices[output_index], i);

            input_gradient_views = layer_back_propagations[output_index]->get_input_gradients();

            layer_delta_views[i].push_back(input_gradient_views[input_index]);
        }
    }

    return layer_delta_views;
}


TensorView BackPropagation::get_output_gradients() const
{
    return {(type*)output_gradients.data(), output_gradient_dimensions};
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

    neural_network.print();
}


type Loss::calculate_numerical_error() const
{
    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> training_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_indices = dataset->get_feature_indices("Input");
    // const vector<Index> decoder_feature_indices = dataset->get_feature_indices("Decoder");
    const vector<Index> target_indices = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);

    batch.fill(training_indices, input_indices, target_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    neural_network->forward_propagate(batch.get_inputs(),
                                      forward_propagation);

    BackPropagation back_propagation(samples_number, this);

    calculate_error(batch, forward_propagation, back_propagation);

    return back_propagation.error;
}


VectorR Loss::calculate_gradient()
{
    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");

    // const vector<Index> decoder_feature_indices = dataset->get_feature_indices("Decoder");

    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);
    // batch.fill(sample_indices, input_feature_indices, decoder_feature_indices, target_feature_indices);
    batch.fill(sample_indices, input_feature_indices, target_feature_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagation back_propagation(samples_number, this);

    const VectorR& parameters = neural_network->get_parameters();

    neural_network->forward_propagate(batch.get_inputs(),
                                      parameters,
                                      forward_propagation);

    back_propagate(batch, forward_propagation, back_propagation);

    return back_propagation.neural_network.gradient;
}


VectorR Loss::calculate_numerical_gradient()
{
    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");
    //const vector<Index> decoder_feature_indices = dataset->get_feature_indices("Decoder");

    Batch batch(samples_number, dataset);
    batch.fill(sample_indices, input_feature_indices, target_feature_indices);

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


VectorR Loss::calculate_numerical_gradient_lm()
{
    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);
    batch.fill(sample_indices, input_feature_indices, target_feature_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagationLM back_propagation_lm(samples_number, this);

    VectorR& parameters = neural_network->get_parameters();

    const Index parameters_number = parameters.size();

    type h = 0;

    VectorR parameters_forward = parameters;
    VectorR parameters_backward = parameters;

    type error_forward = 0;
    type error_backward = 0;

    VectorR numerical_gradient_lm(parameters_number);
    numerical_gradient_lm.setConstant(type(0));

    for(Index i = 0; i < parameters_number; i++)
    {
        h = calculate_h(parameters(i));

        parameters_forward(i) += h;

        neural_network->forward_propagate(batch.get_inputs(),
                                          parameters_forward,
                                          forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_error_lm(batch, forward_propagation, back_propagation_lm);

        error_forward = back_propagation_lm.error;

        parameters_forward(i) -= h;

        parameters_backward(i) -= h;

        neural_network->forward_propagate(batch.get_inputs(),
                                          parameters_backward,
                                          forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_error_lm(batch, forward_propagation, back_propagation_lm);

        error_backward = back_propagation_lm.error;

        parameters_backward(i) += h;

        numerical_gradient_lm(i) = (error_forward - error_backward)/type(2*h);
    }

    return numerical_gradient_lm;
}


VectorR Loss::calculate_numerical_input_gradients()
{
    const Index samples_number = dataset->get_samples_number("Training");
    const Shape input_shape = dataset->get_shape("Input");

    const Index values_number = neural_network->get_inputs_number()*samples_number;

    const vector<Index> sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);
    // batch.fill(sample_indices, input_feature_indices, {}, target_feature_indices);
    batch.fill(sample_indices, input_feature_indices, target_feature_indices);

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


MatrixR Loss::calculate_numerical_jacobian()
{
    const Index samples_number = dataset->get_samples_number("Training");
    const vector<Index> sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);
    batch.fill(sample_indices, input_feature_indices, target_feature_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);
    BackPropagationLM back_propagation_lm(samples_number, this);

    VectorR parameters = neural_network->get_parameters();
    const Index parameters_number = parameters.size();

    const Index total_error_terms = back_propagation_lm.squared_errors.size();

    type perturbation;

    VectorR parameters_forward(parameters);
    VectorR parameters_backward(parameters);

    VectorR error_terms_forward(total_error_terms);
    VectorR error_terms_backward(total_error_terms);

    MatrixR jacobian(total_error_terms, parameters_number);

    for(Index j = 0; j < parameters_number; j++)
    {
        perturbation = calculate_h(parameters(j));

        parameters_backward(j) -= perturbation;
        neural_network->forward_propagate(batch.get_inputs(), parameters_backward, forward_propagation);
        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);
        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);
        error_terms_backward = back_propagation_lm.squared_errors;
        parameters_backward(j) += perturbation;

        parameters_forward(j) += perturbation;
        neural_network->forward_propagate(batch.get_inputs(), parameters_forward, forward_propagation);
        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);
        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);
        error_terms_forward = back_propagation_lm.squared_errors;
        parameters_forward(j) -= perturbation;

        for(Index i = 0; i < total_error_terms; i++)
            jacobian(i, j) = (error_terms_forward(i) - error_terms_backward(i)) / (type(2.0) * perturbation);
    }

    return jacobian;
}


MatrixR Loss::calculate_numerical_hessian()
{
    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_feature_indices = dataset->get_feature_indices("Input");
    const vector<Index> target_feature_indices = dataset->get_feature_indices("Target");

    Batch batch(samples_number, dataset);
    // batch.fill(sample_indices, input_feature_indices, {}, target_feature_indices);
    batch.fill(sample_indices, input_feature_indices, target_feature_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagationLM back_propagation_lm(samples_number, this);

    VectorR& parameters = neural_network->get_parameters();

    const Index parameters_number = parameters.size();

    neural_network->forward_propagate(batch.get_inputs(),
                                      parameters,
                                      forward_propagation);

    calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

    calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

    calculate_error_lm(batch, forward_propagation, back_propagation_lm);

    const type y = back_propagation_lm.error;

    MatrixR H(parameters_number, parameters_number);
    H.setZero();

    type h_i;
    type h_j;

    VectorR x_backward_2i = parameters;
    VectorR x_backward_i = parameters;

    VectorR x_forward_i = parameters;
    VectorR x_forward_2i = parameters;

    VectorR x_backward_ij = parameters;
    VectorR x_forward_ij = parameters;

    VectorR x_backward_i_forward_j = parameters;
    VectorR x_forward_i_backward_j = parameters;

    type y_backward_2i;
    type y_backward_i;

    type y_forward_i;
    type y_forward_2i;

    type y_backward_ij;
    type y_forward_ij;

    type y_backward_i_forward_j;
    type y_forward_i_backward_j;

    for(Index i = 0; i < parameters_number; i++)
    {
        h_i = calculate_h(parameters(i));

        x_backward_2i(i) -= static_cast<type>(2.0) * h_i;

        neural_network->forward_propagate(batch.get_inputs(),
                                          x_backward_2i,
                                          forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_error_lm(batch, forward_propagation, back_propagation_lm);

        y_backward_2i = back_propagation_lm.error;

        x_backward_2i(i) += static_cast<type>(2.0) * h_i;

        x_backward_i(i) -= h_i;

        neural_network->forward_propagate(batch.get_inputs(),
                                          x_backward_i,
                                          forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_error_lm(batch, forward_propagation, back_propagation_lm);

        y_backward_i = back_propagation_lm.error;

        x_backward_i(i) += h_i;

        x_forward_i(i) += h_i;

        neural_network->forward_propagate(batch.get_inputs(),
                                          x_forward_i,
                                          forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_error_lm(batch, forward_propagation, back_propagation_lm);

        y_forward_i = back_propagation_lm.error;

        x_forward_i(i) -= h_i;

        x_forward_2i(i) += static_cast<type>(2.0) * h_i;

        neural_network->forward_propagate(batch.get_inputs(),
                                          x_forward_2i,
                                          forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_error_lm(batch, forward_propagation, back_propagation_lm);

        y_forward_2i = back_propagation_lm.error;

        x_forward_2i(i) -= static_cast<type>(2.0) * h_i;

        H(i, i) = (-y_forward_2i + type(16.0) * y_forward_i - type(30.0) * y + type(16.0) * y_backward_i - y_backward_2i) / (type(12.0) * pow(h_i, type(2)));

        for(Index j = i; j < parameters_number; j++)
        {
            // if(j == i)
            // continue;

            h_j = calculate_h(parameters(j));

            x_backward_ij(i) -= h_i;
            x_backward_ij(j) -= h_j;

            neural_network->forward_propagate(batch.get_inputs(),
                                              x_backward_ij,
                                              forward_propagation);

            calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_error_lm(batch, forward_propagation, back_propagation_lm);

            y_backward_ij = back_propagation_lm.error;

            x_backward_ij(i) += h_i;
            x_backward_ij(j) += h_j;

            x_forward_ij(i) += h_i;
            x_forward_ij(j) += h_j;

            neural_network->forward_propagate(batch.get_inputs(),
                                              x_forward_ij,
                                              forward_propagation);

            calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_error_lm(batch, forward_propagation, back_propagation_lm);

            y_forward_ij = back_propagation_lm.error;

            x_forward_ij(i) -= h_i;
            x_forward_ij(j) -= h_j;

            x_backward_i_forward_j(i) -= h_i;
            x_backward_i_forward_j(j) += h_j;

            neural_network->forward_propagate(batch.get_inputs(),
                                              x_backward_i_forward_j,
                                              forward_propagation);

            calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_error_lm(batch, forward_propagation, back_propagation_lm);

            y_backward_i_forward_j = back_propagation_lm.error;

            x_backward_i_forward_j(i) += h_i;
            x_backward_i_forward_j(j) -= h_j;

            x_forward_i_backward_j(i) += h_i;
            x_forward_i_backward_j(j) -= h_j;

            neural_network->forward_propagate(batch.get_inputs(),
                                              x_forward_i_backward_j,
                                              forward_propagation);

            calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_error_lm(batch, forward_propagation, back_propagation_lm);

            y_forward_i_backward_j = back_propagation_lm.error;

            x_forward_i_backward_j(i) -= h_i;
            x_forward_i_backward_j(j) += h_j;

            H(i, j) = (y_forward_ij - y_forward_i_backward_j - y_backward_i_forward_j + y_backward_ij) / (type(4.0) * h_i * h_j);
        }
    }

    for(Index i = 0; i < parameters_number; i++)
        for(Index j = 0; j < i; j++)
            H(i, j) = H(j, i);

    return H;
}


MatrixR Loss::calculate_inverse_hessian()
{
    MatrixR numerical_hessian = calculate_numerical_hessian();
    const Index parameters_number = numerical_hessian.rows();

    using MatrixType = Matrix<type, Dynamic, Dynamic, ColMajor>;
    Map<MatrixType> hessian_map(numerical_hessian.data(), parameters_number, parameters_number);

    FullPivLU<MatrixType> hessian_decomposition(hessian_map);

    if(!hessian_decomposition.isInvertible())
    {
        MatrixType hessian_damped = hessian_map + MatrixType::Identity(parameters_number, parameters_number) * 1e-4;

        FullPivLU<MatrixType> hessian_decomposition_damped(hessian_damped);

        MatrixType hessian_map_inverse = hessian_decomposition_damped.inverse();

        MatrixR hessian_inverse(parameters_number, parameters_number);
        Map<MatrixType>(hessian_inverse.data(), parameters_number, parameters_number) = hessian_map_inverse;

        return hessian_inverse;
    }

    MatrixType hessian_map_inverse = hessian_decomposition.inverse();
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


void BackPropagationLM::print() const
{
    cout << "Loss index back-propagation LM" << endl
         << "Errors:" << endl
         << errors << endl
         << "Squared errors:" << endl
         << squared_errors << endl
         << "Squared errors Jacobian:" << endl
         << squared_errors_jacobian << endl
         << "Error:" << endl
         << error << endl
         << "Loss:" << endl
         << loss << endl
         << "Gradient:" << endl
         << gradient << endl
         << "Hessian:" << endl
         << hessian << endl;
}


vector<vector<TensorView>> BackPropagationLM::get_layer_gradients() const
{
    NeuralNetwork* neural_network_ptr = loss_index->get_neural_network();

    const Index layers_number = neural_network_ptr->get_layers_number();

    const vector<vector<Index>>& layer_input_indices = neural_network_ptr->get_layer_input_indices();
    const vector<vector<Index>> layer_output_indices = neural_network_ptr->get_layer_output_indices();

    const vector<unique_ptr<LayerBackPropagationLM>>& layer_back_propagations = neural_network.get_layers();

    vector<TensorView> input_gradient_views;

    vector<vector<TensorView>> layer_delta_views(layers_number);

    const Index first_trainable_layer_index = neural_network_ptr->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network_ptr->get_last_trainable_layer_index();

    for(Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
    {
        if(i == last_trainable_layer_index)
        {
            layer_delta_views[i].push_back(get_output_gradients());

            continue;
        }

        for(Index j = 0; j < Index(layer_output_indices[i].size()); j++)
        {
            const Index output_layer_index = layer_output_indices[i][j];

            if(output_layer_index == -1) continue;

            if(!layer_back_propagations[output_layer_index]) continue;

            const Index input_port_index = neural_network_ptr->find_input_index(layer_input_indices[output_layer_index], i);

            if(input_port_index == -1)
                throw runtime_error("Layer indices consistency error.");

            vector<TensorView> input_derivative_pairs = layer_back_propagations[output_layer_index]->get_input_gradients();

            layer_delta_views[i].push_back(input_derivative_pairs[input_port_index]);
        }
    }

    return layer_delta_views;
}


TensorView BackPropagationLM::get_output_gradients() const
{
    return {(type*)output_gradients.data(), output_gradient_dimensions};
}


void BackPropagationLM::set(const Index new_samples_number,
                            Loss *new_loss)
{
    loss_index = new_loss;
    samples_number = new_samples_number;

    if(!loss_index) return;

    NeuralNetwork* neural_network_ptr = loss_index->get_neural_network();

    if(!neural_network_ptr) return;

    const Index alignment_elements = EIGEN_MAX_ALIGN_BYTES / sizeof(type);
    const Index mask_elements = ~(alignment_elements - 1);
    Index padded_parameters_number = 0;

    const Index layers_number = neural_network_ptr->get_layers_number();

    for(Index i = 0; i < layers_number; i++)
    {
        const vector<TensorView*> parameter_views = neural_network_ptr->get_layer(i)->get_parameter_views();

        for(const TensorView* view : parameter_views)
        {
            const Index view_size = view->size();

            if(view_size > 0)
                padded_parameters_number += (view_size + alignment_elements - 1) & mask_elements;
        }
    }

    neural_network.set(samples_number, neural_network_ptr);

    loss = type(0);

    gradient.resize(padded_parameters_number);
    gradient.setZero();

    regularization_gradient.resize(padded_parameters_number);
    regularization_gradient.setZero();

    const Index outputs_number = neural_network_ptr->get_outputs_number();
    const Index total_error_terms = samples_number * outputs_number;

    squared_errors_jacobian.resize(total_error_terms, padded_parameters_number);
    squared_errors_jacobian.setZero();

    hessian.resize(padded_parameters_number, padded_parameters_number);
    hessian.setZero();

    regularization_hessian.resize(padded_parameters_number, padded_parameters_number);
    regularization_hessian.setZero();

    errors.resize(samples_number, outputs_number);
    squared_errors.resize(total_error_terms);

    const Shape output_shape = neural_network_ptr->get_output_shape();
    output_gradient_dimensions = { samples_number };
    output_gradient_dimensions.insert(output_gradient_dimensions.end(), output_shape.begin(), output_shape.end());

    const Index total_output_size = accumulate(output_shape.begin(), output_shape.end(), samples_number, multiplies<Index>());
    output_gradients.resize(total_output_size);
    output_gradients.setZero();
}


BackPropagationLM::BackPropagationLM(const Index new_batch_size, Loss *new_loss)
{
    set(new_batch_size, new_loss);
}


#ifdef OPENNN_CUDA

void Loss::back_propagate(const BatchCuda& batch,
                                    ForwardPropagationCuda& forward_propagation,
                                    BackPropagationCuda& back_propagation)
{
    if (batch.is_empty()) return;

    // Loss index

    calculate_error(batch, forward_propagation, back_propagation);

    calculate_layers_error_gradient_cuda(batch, forward_propagation, back_propagation);

    // Loss

    back_propagation.loss = back_propagation.error;

    // Regularization

    add_regularization_cuda(back_propagation);
}


void Loss::calculate_layers_error_gradient_cuda(const BatchCuda& batch,
                                                     ForwardPropagationCuda& forward_propagation,
                                                     BackPropagationCuda& back_propagation) const
{
    const vector<unique_ptr<Layer>>& layers = neural_network->get_layers();
    const Index layers_number = layers.size();

    if (layers_number == 0) return;

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const vector<vector<TensorViewCuda>> layer_input_views
        = forward_propagation.get_layer_input_views_device(batch.get_inputs_device(), true);

    const vector<vector<TensorViewCuda>> layer_delta_views
        = back_propagation.get_layer_delta_views_device();

    calculate_output_gradients(batch, forward_propagation, back_propagation);

    for (Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
        layers[i]->back_propagate(layer_input_views[i],
                                       layer_delta_views[i],
                                       forward_propagation.layers[i],
                                       back_propagation.neural_network.layers[i]);
}


void Loss::add_regularization_cuda(BackPropagationCuda& back_propagation) const
{
    /*
    if (regularization_method == "None" || regularization_weight == 0.0f)
    {
        back_propagation.regularization = 0.0f;
        return;
    }

    NeuralNetwork* neural_network = back_propagation.loss_index->get_neural_network();

    const Index layers_number = neural_network->get_layers_number();

    type total_sum_squares = 0.0f;
    type total_l1_norm = 0.0f;

    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);

    for(Index layer_index = 0; layer_index < layers_number; ++layer_index)
    {
        Layer* layer = neural_network->get_layer(layer_index).get();

        if(!layer->get_is_trainable())
            continue;

        LayerBackPropagationCuda* layer_back_prop_cuda = back_propagation.neural_network.layers[layer_index].get();

        const vector<TensorView>& parameter_device_pairs = layer->get_parameter_views_device();
        const vector<TensorView>& delta_device_pairs = layer_back_prop_cuda->get_gradient_views_device();

        for(Index param_index = 0; param_index < Index(parameter_device_pairs.size()); ++param_index)
        {
            type* param_device_ptr = parameter_device_pairs[param_index].data;
            const Index param_size = parameter_device_pairs[param_index].size;
            type* delta_device_ptr = delta_device_pairs[param_index].data;

            if (param_size == 0) continue;

            if (regularization_method == "L1")
            {
                type l1_norm = 0.0f;
                cublasSasum(cublas_handle, param_size, param_device_ptr, 1, &l1_norm);
                total_l1_norm += l1_norm;

                apply_l1_gradient_cuda(param_size, delta_device_ptr, param_device_ptr, regularization_weight);
            }
            else if (regularization_method == "L2")
            {
                type sum_squares = 0.0f;

                cublasSdot(cublas_handle, param_size, param_device_ptr, 1, param_device_ptr, 1, &sum_squares);

                total_sum_squares += sum_squares;

                const type alpha = regularization_weight;

                cublasSaxpy(cublas_handle, param_size, &alpha, param_device_ptr, 1, delta_device_ptr, 1);
            }
            else if (regularization_method == "ElasticNet")
            {
                const type mix_factor = 0.5f;
                type l1_sum_abs = 0.0f;
                type l2_sum_sq = 0.0f;

                cublasSasum(cublas_handle, param_size, param_device_ptr, 1, &l1_sum_abs);

                cublasSdot(cublas_handle, param_size, param_device_ptr, 1, param_device_ptr, 1, &l2_sum_sq);

                const type term_l1 = mix_factor * l1_sum_abs;
                const type term_l2 = (1.0f - mix_factor) * 0.5f * l2_sum_sq;

                apply_elastic_net_gradient_cuda(param_size, delta_device_ptr, param_device_ptr, regularization_weight, mix_factor);
            }
        }
    }

    if (regularization_method == "L2")
    {
        const type regularization_term = 0.5f * regularization_weight * total_sum_squares;

        back_propagation.regularization = regularization_term;
        back_propagation.loss += regularization_term;
    }
    else if (regularization_method == "L1")
    {
        const type regularization_term = regularization_weight * total_l1_norm;

        back_propagation.regularization = regularization_term;
        back_propagation.loss += regularization_term;
    }
    */
}


// CUDA structs

BackPropagationCuda::BackPropagationCuda(const Index new_samples_number, Loss* new_loss)
{
    set(new_samples_number, new_loss);
}


void BackPropagationCuda::set(const Index new_samples_number, Loss* new_loss)
{
    samples_number = new_samples_number;
    loss_index = new_loss;

    if(!loss_index) return;

    // Neural network

    NeuralNetwork* neural_network_ptr = loss_index->get_neural_network();

    const Shape output_shape = neural_network_ptr->get_output_shape();

    const Index outputs_number = output_shape[0];

    // First order loss

    neural_network.set(samples_number, neural_network_ptr);

    loss = type(0);
    error = type(0);
    regularization = type(0);

    CHECK_CUDA(cudaMalloc(&errors, samples_number * outputs_number * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&error_device, sizeof(float)));

    // Outputs_delta

    Shape output_gradient_dimensions = { samples_number };
    output_gradient_dimensions.insert(output_gradient_dimensions.end(), output_shape.begin(), output_shape.end());

    const Index size = accumulate(output_shape.begin(), output_shape.end(), samples_number, multiplies<>());

    output_gradients.resize(output_gradient_dimensions);

    // Reduce

    cudnnCreateReduceTensorDescriptor(&reduce_tensor_descriptor);

    cudnnSetReduceTensorDescriptor(reduce_tensor_descriptor,
                                   CUDNN_REDUCE_TENSOR_ADD,
                                   CUDNN_DATA_FLOAT,
                                   CUDNN_PROPAGATE_NAN,
                                   CUDNN_REDUCE_TENSOR_NO_INDICES,
                                   CUDNN_32BIT_INDICES);

    cudnnCreateTensorDescriptor(&output_reduce_tensor_descriptor);

    cudnnSetTensor4dDescriptor(output_reduce_tensor_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               1,
                               1,
                               1,
                               1);

    cudnnGetReductionWorkspaceSize(get_cudnn_handle(),
                                   reduce_tensor_descriptor,
                                   output_gradients.get_descriptor(),
                                   output_reduce_tensor_descriptor,
                                   &workspace_size);

    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

    // Sum

    //if (is_instance_of<CrossEntropyError3d>(loss_index))
    //{
    /* @todo CudaMalloc transformers GPU
        predictions (batch_size, outputs_number);
        matches (batch_size, outputs_number);
        mask (batch_size, outputs_number);
        */
    //}
}


vector<vector<TensorViewCuda>> BackPropagationCuda::get_layer_delta_views_device() const
{
    NeuralNetwork* neural_network_ptr = loss_index->get_neural_network();

    const Index layers_number = neural_network_ptr->get_layers_number();

    const vector<vector<Index>>& layer_input_indices = neural_network_ptr->get_layer_input_indices();
    const vector<vector<Index>> layer_output_indices = neural_network_ptr->get_layer_output_indices();
    const vector<unique_ptr<LayerBackPropagationCuda>>& layer_back_propagations = neural_network.get_layers();

    vector<TensorViewCuda> input_gradient_views;

    vector<vector<TensorViewCuda>> layer_delta_views(layers_number);

    const Index first_trainable_layer_index = neural_network_ptr->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network_ptr->get_last_trainable_layer_index();

    for(Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
    {
        if (i == last_trainable_layer_index)
        {
            layer_delta_views[i].push_back(get_output_gradient_views_device());

            continue;
        }

        for(Index j = 0; j < Index(layer_output_indices[i].size()); j++)
        {
            const Index output_index = layer_output_indices[i][j];
            const Index input_index = neural_network_ptr->find_input_index(layer_input_indices[output_index], i);

            input_gradient_views = layer_back_propagations[output_index]->get_input_gradient_views();

            layer_delta_views[i].push_back(input_gradient_views[input_index]);
        }
    }

    return layer_delta_views;
}


TensorViewCuda BackPropagationCuda::get_output_gradient_views_device() const
{
    return output_gradients.view();
}


void BackPropagationCuda::print() const
{

}


void BackPropagationCuda::free()
{
    cudaFree(error_device);
    error_device = nullptr;

    cudaFree(errors);
    errors = nullptr;

    cudaFree(workspace);
    workspace = nullptr;

    cudnnDestroyReduceTensorDescriptor(reduce_tensor_descriptor);
    cudnnDestroyTensorDescriptor(output_reduce_tensor_descriptor);
}

#endif

} 

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
