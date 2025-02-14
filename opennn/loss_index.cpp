//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   I N D E X   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "forward_propagation.h"
#include "tensors.h"
#include "loss_index.h"
#include "back_propagation.h"
#include "cross_entropy_error_3d.h"

namespace opennn
{

LossIndex::LossIndex(NeuralNetwork* new_neural_network, DataSet* new_data_set)
{
    set(new_neural_network, new_data_set);
}


const type& LossIndex::get_regularization_weight() const
{
    return regularization_weight;
}


const bool& LossIndex::get_display() const
{
    return display;
}


bool LossIndex::has_neural_network() const
{
    return neural_network;
}


bool LossIndex::has_data_set() const
{
    return data_set;
}


LossIndex::RegularizationMethod LossIndex::get_regularization_method() const
{
    return regularization_method;
}


void LossIndex::set(NeuralNetwork* new_neural_network, DataSet* new_data_set)
{
    neural_network = new_neural_network;

    data_set = new_data_set;

    const unsigned int threads_number = thread::hardware_concurrency();

    thread_pool = make_unique<ThreadPool>(threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);

    regularization_method = RegularizationMethod::L2;
}


void LossIndex::set_threads_number(const int& new_threads_number)
{
    thread_pool = make_unique<ThreadPool>(new_threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), new_threads_number);
}


void LossIndex::set_neural_network(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;
}


void LossIndex::set_data_set(DataSet* new_data_set)
{
    data_set = new_data_set;
}


void LossIndex::set_regularization_method(const string& new_regularization_method)
{
    if(new_regularization_method == "L1_NORM")
        set_regularization_method(RegularizationMethod::L1);
    else if(new_regularization_method == "L2_NORM")
        set_regularization_method(RegularizationMethod::L2);
    else if(new_regularization_method == "NO_REGULARIZATION")
        set_regularization_method(RegularizationMethod::NoRegularization);
    else
        throw runtime_error("Unknown regularization method: " + new_regularization_method + ".");
}


void LossIndex::set_regularization_method(const LossIndex::RegularizationMethod& new_regularization_method)
{
    regularization_method = new_regularization_method;
}


void LossIndex::set_regularization_weight(const type& new_regularization_weight)
{
    regularization_weight = new_regularization_weight;
}


void LossIndex::set_display(const bool& new_display)
{
    display = new_display;
}


//void LossIndex::check() const
//{
//    if(!neural_network)
//        throw runtime_error("Pointer to neural network is nullptr.\n");

//    if(!data_set)
//        throw runtime_error("Pointer to data set is nullptr.\n");
//}


void LossIndex::calculate_errors_lm(const Batch& batch,
                                    const ForwardPropagation & forward_propagation,
                                    BackPropagationLM & back_propagation) const
{
    const pair<type*, dimensions> outputs_pair
        = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map_2(outputs_pair);

    const pair<type*, dimensions> targets_pair = batch.get_target_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map_2(targets_pair);

    back_propagation.errors.device(*thread_pool_device) = outputs - targets;
}


void LossIndex::calculate_squared_errors_lm(const Batch&,
                                            const ForwardPropagation&,
                                            BackPropagationLM& back_propagation_lm) const
{
    const Tensor<type, 2>& errors = back_propagation_lm.errors;

    Tensor<type, 1>& squared_errors = back_propagation_lm.squared_errors;

    squared_errors.device(*thread_pool_device) = errors.square().sum(rows_sum).sqrt();
}


void LossIndex::back_propagate(const Batch& batch,
                               ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation) const
{
    if(batch.is_empty()) return;

    // Loss index

    calculate_error(batch, forward_propagation, back_propagation);

    calculate_layers_error_gradient(batch, forward_propagation, back_propagation);

    assemble_layers_error_gradient(back_propagation);

    // Loss

    back_propagation.loss = back_propagation.error();

    // Regularization

    add_regularization(back_propagation);

}


void LossIndex::add_regularization(BackPropagation& back_propagation) const
{
    if(regularization_method == RegularizationMethod::NoRegularization)
        return;

    type& regularization = back_propagation.regularization;
    type& loss = back_propagation.loss;

    const Tensor<type, 1>& parameters = back_propagation.parameters;
    Tensor<type, 1>& regularization_gradient = back_propagation.regularization_gradient;
    Tensor<type, 1>& gradient = back_propagation.gradient;

    regularization = calculate_regularization(parameters);

    loss += regularization_weight * regularization;

    calculate_regularization_gradient(parameters, regularization_gradient);

    gradient.device(*thread_pool_device) += regularization_weight * regularization_gradient;
}


void LossIndex::add_regularization_lm(BackPropagationLM& back_propagation_lm) const
{
    if(regularization_method == RegularizationMethod::NoRegularization)
        return;

    const type regularization = calculate_regularization(back_propagation_lm.parameters);

    back_propagation_lm.loss += regularization_weight*regularization;

    calculate_regularization_gradient(back_propagation_lm.parameters, back_propagation_lm.regularization_gradient);

    back_propagation_lm.gradient.device(*thread_pool_device) += regularization_weight*back_propagation_lm.regularization_gradient;

    calculate_regularization_hessian(back_propagation_lm.parameters, back_propagation_lm.regularization_hessian);

    back_propagation_lm.hessian += regularization_weight*back_propagation_lm.regularization_hessian;
}


void LossIndex::back_propagate_lm(const Batch& batch,
                                  ForwardPropagation& forward_propagation,
                                  BackPropagationLM& back_propagation_lm) const
{
    calculate_errors_lm(batch, forward_propagation, back_propagation_lm);
    
    calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

    // cout << "OK 2" << endl;

    calculate_error_lm(batch, forward_propagation, back_propagation_lm);
 
    calculate_layers_squared_errors_jacobian_lm(batch, forward_propagation, back_propagation_lm);

    calculate_error_gradient_lm(batch, back_propagation_lm);

    calculate_error_hessian_lm(batch, back_propagation_lm);

    // Loss

    back_propagation_lm.loss = back_propagation_lm.error();

    // Regularization
    
    add_regularization_lm(back_propagation_lm);

}


void LossIndex::calculate_layers_squared_errors_jacobian_lm(const Batch& batch,
                                                            ForwardPropagation& forward_propagation,
                                                            BackPropagationLM& back_propagation_lm) const
{
    const Index layers_number = neural_network->get_layers_number();

    if (layers_number == 0) return;

    const vector<unique_ptr<Layer>>& layers = neural_network->get_layers();

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const vector<vector<pair<type*, dimensions>>> layer_input_pairs
        = forward_propagation.get_layer_input_pairs(batch.get_input_pairs(), true);

    const vector<vector<pair<type*, dimensions>>> layer_delta_pairs 
        = back_propagation_lm.get_layer_delta_pairs();

    calculate_output_delta_lm(batch, forward_propagation, back_propagation_lm);
    
    for(Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
        layers[i]->back_propagate_lm(layer_input_pairs[i],
                                     layer_delta_pairs[i],
                                     forward_propagation.layers[i],
                                     back_propagation_lm.neural_network.layers[i]);
    
    const vector<Index> layer_parameter_numbers = neural_network->get_layer_parameter_numbers();

    const Index batch_samples_number = batch.get_samples_number();

    Index index = 0;
    
    for(Index i = 0; i < layers_number; i++)
    {     
        layers[i]->insert_squared_errors_Jacobian_lm(back_propagation_lm.neural_network.layers[i],
                                                     index,
                                                     back_propagation_lm.squared_errors_jacobian);

        index += layer_parameter_numbers[i] * batch_samples_number;
    }

}


void LossIndex::calculate_error_gradient_lm(const Batch&,
                                            BackPropagationLM& back_propagation_lm) const
{
    const Tensor<type, 1>& squared_errors = back_propagation_lm.squared_errors;
    const Tensor<type, 2>& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    Tensor<type, 1>& gradient = back_propagation_lm.gradient;

    gradient.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors, AT_B);
}


string LossIndex::get_loss_method() const
{
    return "USER_ERROR_TERM";
}


string LossIndex::get_error_type_text() const
{
    return "USER_ERROR_TERM";
}


string LossIndex::write_regularization_method() const
{
    switch(regularization_method)
    {
    case RegularizationMethod::NoRegularization:
        return "NO_REGULARIZATION";

    case RegularizationMethod::L1:
        return "L1_NORM";

    case RegularizationMethod::L2:
        return "L2_NORM";

    default: return string();
    }
}


type LossIndex::calculate_regularization(const Tensor<type, 1>& parameters) const
{   
    switch(regularization_method)
    {
        case RegularizationMethod::NoRegularization: 
            return type(0);

        case RegularizationMethod::L1: 
            return l1_norm(thread_pool_device.get(), parameters);

        case RegularizationMethod::L2: 
            return l2_norm(thread_pool_device.get(), parameters);

        default: 
            return type(0);
    }

    return type(0);
}


void LossIndex::calculate_regularization_gradient(const Tensor<type, 1>& parameters, Tensor<type, 1>& regularization_gradient) const
{
    switch(regularization_method)
    {
    case RegularizationMethod::NoRegularization:
        regularization_gradient.setZero(); return;

    case RegularizationMethod::L1:
        l1_norm_gradient(thread_pool_device.get(), parameters, regularization_gradient); return;

    case RegularizationMethod::L2:
        l2_norm_gradient(thread_pool_device.get(), parameters, regularization_gradient); return;

    default:
        return;
    }
}


void LossIndex::calculate_regularization_hessian(Tensor<type, 1>& parameters, Tensor<type, 2>& regularization_hessian) const
{
    switch(regularization_method)
    {
    case RegularizationMethod::L1:
        l1_norm_hessian(thread_pool_device.get(), parameters, regularization_hessian);
        return;

    case RegularizationMethod::L2:
        l2_norm_hessian(thread_pool_device.get(), parameters, regularization_hessian);
        return;

    default:
        return;
    }
}


void LossIndex::calculate_layers_error_gradient(const Batch& batch,
                                                ForwardPropagation& forward_propagation,
                                                BackPropagation& back_propagation) const
{
    const vector<unique_ptr<Layer>>& layers = neural_network->get_layers();

    const Index layers_number = layers.size();

    if(layers_number == 0) return;

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const vector<vector<pair<type*, dimensions>>> layer_input_pairs
        = forward_propagation.get_layer_input_pairs(batch.get_input_pairs(), true);

    const vector<vector<pair<type*, dimensions>>> layer_delta_pairs 
        = back_propagation.get_layer_delta_pairs();

    calculate_output_delta(batch, forward_propagation, back_propagation);

    for (Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
        layers[i]->back_propagate(layer_input_pairs[i],
                                  layer_delta_pairs[i],
                                  forward_propagation.layers[i],
                                  back_propagation.neural_network.layers[i]);

}


void LossIndex::assemble_layers_error_gradient(BackPropagation& back_propagation) const
{
    const vector<unique_ptr<Layer>>& layers = neural_network->get_layers();

    const Index layers_number = neural_network->get_layers_number();

    const vector<Index> layer_parameter_numbers = neural_network->get_layer_parameter_numbers();

    Index index = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        layers[i]->insert_gradient(back_propagation.neural_network.layers[i],
                                   index,
                                   back_propagation.gradient);

        index += layer_parameter_numbers[i];
    }


}


void LossIndex::to_XML(XMLPrinter& file_stream) const
{
    file_stream.OpenElement("LossIndex");

    file_stream.CloseElement();
}


void LossIndex::regularization_from_XML(const XMLDocument& document)
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


void LossIndex::write_regularization_XML(XMLPrinter& file_stream) const
{
    file_stream.OpenElement("Regularization");
    
    switch(regularization_method)
    {
    case RegularizationMethod::L1:
        file_stream.PushAttribute("Type", "L1_NORM");
    break;

    case RegularizationMethod::L2:
        file_stream.PushAttribute("Type", "L2_NORM");
    break;

    case RegularizationMethod::NoRegularization:
        file_stream.PushAttribute("Type", "NO_REGULARIZATION");
    break;

    default: 
    break;
    }

    // Regularization weight

    file_stream.OpenElement("RegularizationWeight");
    file_stream.PushText(to_string(regularization_weight).c_str());
    file_stream.CloseElement();

    // Close regularization

    file_stream.CloseElement();
}


void LossIndex::from_XML(const XMLDocument& document)
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


BackPropagation::BackPropagation(const Index& new_batch_samples_number, LossIndex* new_loss_index)
{
    set(new_batch_samples_number, new_loss_index);
}


void BackPropagation::set(const Index& new_batch_samples_number, LossIndex* new_loss_index)
{
    batch_samples_number = new_batch_samples_number;

    loss_index = new_loss_index;

    if(!loss_index) return;

    // Neural network

    NeuralNetwork* neural_network_ptr = loss_index->get_neural_network();

    const Index parameters_number = neural_network_ptr->get_parameters_number();

    const dimensions output_dimensions = neural_network_ptr->get_output_dimensions();

    const Index outputs_number = output_dimensions[0];

    // First order loss

    neural_network.set(batch_samples_number, neural_network_ptr);

    loss = type(0);

    errors.resize(batch_samples_number, outputs_number);

    parameters = neural_network_ptr->get_parameters();

    gradient.resize(parameters_number);

    regularization_gradient.resize(parameters_number);

    output_deltas_dimensions = { batch_samples_number };
    output_deltas_dimensions.insert(output_deltas_dimensions.end(), output_dimensions.begin(), output_dimensions.end());

    const Index size = accumulate(output_dimensions.begin(), output_dimensions.end(), batch_samples_number, multiplies<>());

    output_deltas.resize(size);

//    output_deltas_dimensions.resize(output_dimensions.size() + 1);
//    output_deltas_dimensions[0] = batch_samples_number;

//    Index size = batch_samples_number;

//    for(size_t i = 0; i < output_dimensions.size(); i++)
//    {
//        output_deltas_dimensions[i + 1] = output_dimensions[i];

//        size *= output_dimensions[i];
//    }


    if(is_instance_of<CrossEntropyError3D>(loss_index))
    {
        predictions.resize(batch_samples_number, outputs_number);
        matches.resize(batch_samples_number, outputs_number);
        mask.resize(batch_samples_number, outputs_number);
    }
}


vector<vector<pair<type*, dimensions>>> BackPropagation::get_layer_delta_pairs() const
{
    NeuralNetwork* neural_network_ptr = loss_index->get_neural_network();

    const Index layers_number = neural_network_ptr->get_layers_number();

    const vector<vector<Index>>& layer_input_indices = neural_network_ptr->get_layer_input_indices();
    const vector<vector<Index>> layer_output_indices = neural_network_ptr->get_layer_output_indices();
    const vector<unique_ptr<LayerBackPropagation>>& layer_back_propagations = neural_network.get_layers();

    vector<pair<type*, dimensions>> input_derivative_pairs;

    vector<vector<pair<type*, dimensions>>> layer_delta_pairs(layers_number);

    const Index first_trainable_layer_index = neural_network_ptr->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network_ptr->get_last_trainable_layer_index();

    for (Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
    {
        if (i == last_trainable_layer_index)
        {
            layer_delta_pairs[i].push_back(get_output_deltas_pair());

            continue;
        }

        for (Index j = 0; j < Index(layer_output_indices[i].size()); j++)
        {
            const Index output_index = layer_output_indices[i][j];
            const Index input_index = neural_network_ptr->find_input_index(layer_input_indices[output_index], i);

            input_derivative_pairs = layer_back_propagations[output_index]->get_input_derivative_pairs();

            layer_delta_pairs[i].push_back(input_derivative_pairs[input_index]);
        }
    }
    return layer_delta_pairs;
}


pair<type*, dimensions> BackPropagation::get_output_deltas_pair() const
{
    return {(type*)output_deltas.data(), output_deltas_dimensions};
}


void BackPropagation::print() const
{
    cout << "Back-propagation" << endl
         << "Errors:" << endl
         << errors << endl
         << "Error:" << endl
         << error << endl
         << "Regularization:" << endl
         << regularization << endl
         << "Loss:" << endl
         << loss << endl
         << "Gradient:" << endl
         << gradient << endl;

    neural_network.print();
}


Tensor<type, 1> LossIndex::calculate_numerical_gradient() 
{
    const Index samples_number = data_set->get_samples_number(DataSet::SampleUse::Training);

    const vector<Index> sample_indices = data_set->get_sample_indices(DataSet::SampleUse::Training);
    const vector<Index> input_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Input);
    const vector<Index> target_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Target);

    Batch batch(samples_number, data_set);
    batch.fill(sample_indices, input_variable_indices, {}, target_variable_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);
    BackPropagation back_propagation(samples_number, this);

    const Tensor<type, 1> parameters = neural_network->get_parameters();

    const Index parameters_number = parameters.size();

    type h = 0;

    Tensor<type, 1> parameters_forward = parameters;
    Tensor<type, 1> parameters_backward = parameters;

    type error_forward = 0;
    type error_backward = 0;

    Tensor<type, 1> numerical_gradient(parameters_number);
    numerical_gradient.setConstant(type(0));

    for(Index i = 0; i < parameters_number; i++)
    {
        h = calculate_h(parameters(i));

       parameters_forward(i) += h;

       neural_network->forward_propagate(batch.get_input_pairs(),
                                         parameters_forward,
                                         forward_propagation);

       calculate_error(batch, forward_propagation, back_propagation);

       error_forward = back_propagation.error();

       parameters_forward(i) -= h;

       parameters_backward(i) -= h;

       neural_network->forward_propagate(batch.get_input_pairs(),
                                         parameters_backward,
                                         forward_propagation);

       calculate_error(batch, forward_propagation, back_propagation);

       error_backward = back_propagation.error();

       parameters_backward(i) += h;

       numerical_gradient(i) = (error_forward - error_backward)/type(2*h);
    }

    return numerical_gradient;
}


Tensor<type, 1> LossIndex::calculate_numerical_inputs_derivatives()
{

    const Index samples_number = data_set->get_samples_number(DataSet::SampleUse::Training);
    const dimensions inputs_dimensions = data_set->get_dimensions(DataSet::VariableUse::Input);

    const Index values_number = neural_network->get_inputs_number()*samples_number;

    const vector<Index> sample_indices = data_set->get_sample_indices(DataSet::SampleUse::Training);
    const vector<Index> input_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Input);
    const vector<Index> target_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Target);

    Batch batch(samples_number, data_set);
    batch.fill(sample_indices, input_variable_indices, {}, target_variable_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagation back_propagation(samples_number, this);

    type h;

    type error_forward;
    type error_backward;

    Tensor<type, 1> numerical_inputs_derivatives(values_number);
    numerical_inputs_derivatives.setConstant(type(0));

    const vector<pair<type*, dimensions>>& input_pairs = batch.get_input_pairs();

    TensorMap<Tensor<type, 4>> inputs_vector = tensor_map_4(input_pairs[0]);

    for (Index i = 0; i < values_number; i++)
    {
        h = calculate_h(inputs_vector(i));

        input_pairs[0].first[i] += h;

        neural_network->forward_propagate(input_pairs, forward_propagation);

        calculate_error(batch, forward_propagation, back_propagation);
        error_forward = back_propagation.error();

        input_pairs[0].first[i] -= 2*h;

        neural_network->forward_propagate(input_pairs, forward_propagation);

        calculate_error(batch, forward_propagation, back_propagation);
        error_backward = back_propagation.error();

        input_pairs[0].first[i] += h;

        numerical_inputs_derivatives(i) = (error_forward - error_backward) / type(2 * h);
    }

    return numerical_inputs_derivatives;
}


Tensor<type, 2> LossIndex::calculate_numerical_jacobian()
{
    const Index samples_number = data_set->get_samples_number(DataSet::SampleUse::Training);
    const vector<Index> sample_indices = data_set->get_sample_indices(DataSet::SampleUse::Training);

    const vector<Index> input_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Input);
    const vector<Index> target_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Target);

    Batch batch(samples_number, data_set);
    batch.fill(sample_indices, input_variable_indices, {}, target_variable_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagationLM back_propagation_lm(samples_number, this);

    BackPropagation back_propagation(samples_number, this);

    Tensor<type, 1> parameters = neural_network->get_parameters();

    const Index parameters_number = parameters.size();
    
    neural_network->forward_propagate(batch.get_input_pairs(),
                                      parameters,
                                      forward_propagation);

    calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

    calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

    type h;

    Tensor<type, 1> parameters_forward(parameters);
    Tensor<type, 1> parameters_backward(parameters);

    Tensor<type, 1> error_terms_forward(parameters_number);
    Tensor<type, 1> error_terms_backward(parameters_number);

    Tensor<type, 2> jacobian(samples_number,parameters_number);

    for(Index j = 0; j < parameters_number; j++)
    {
        h = calculate_h(parameters(j));

        parameters_backward(j) -= h;
        neural_network->forward_propagate(batch.get_input_pairs(),
                                          parameters_backward,
                                          forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        error_terms_backward = back_propagation_lm.squared_errors;

        parameters_backward(j) += h;

        parameters_forward(j) += h;

        neural_network->forward_propagate(batch.get_input_pairs(),
                                          parameters_forward,
                                          forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        error_terms_forward = back_propagation_lm.squared_errors;

        parameters_forward(j) -= h;

        for(Index i = 0; i < samples_number; i++)
            jacobian(i, j) = (error_terms_forward(i) - error_terms_backward(i))/(type(2.0)*h);
    }

    return jacobian;
}


// @todo

Tensor<type, 2> LossIndex::calculate_numerical_hessian()
{
    const Index samples_number = data_set->get_samples_number(DataSet::SampleUse::Training);

    const vector<Index> sample_indices = data_set->get_sample_indices(DataSet::SampleUse::Training);
    const vector<Index> input_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Input);
    const vector<Index> target_variable_indices = data_set->get_variable_indices(DataSet::VariableUse::Target);

    Batch batch(samples_number, data_set);
    batch.fill(sample_indices, input_variable_indices, {}, target_variable_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagationLM back_propagation_lm(samples_number, this);

    const Tensor<type, 1> parameters = neural_network->get_parameters();

    const Index parameters_number = parameters.size();

    neural_network->forward_propagate(batch.get_input_pairs(),
                                      parameters,
                                      forward_propagation);

    calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

    calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

    calculate_error_lm(batch, forward_propagation, back_propagation_lm);

    const type y = back_propagation_lm.error();

    Tensor<type, 2> H(parameters_number, parameters_number);
    H.setZero();

    type h_i;
    type h_j;

    Tensor<type, 1> x_backward_2i = parameters;
    Tensor<type, 1> x_backward_i = parameters;

    Tensor<type, 1> x_forward_i = parameters;
    Tensor<type, 1> x_forward_2i = parameters;

    Tensor<type, 1> x_backward_ij = parameters;
    Tensor<type, 1> x_forward_ij = parameters;

    Tensor<type, 1> x_backward_i_forward_j = parameters;
    Tensor<type, 1> x_forward_i_backward_j = parameters;

    type y_backward_2i;
    type y_backward_i;

    type y_forward_i;
    type y_forward_2i;

    type y_backward_ij;
    type y_forward_ij;

    type y_backward_i_forward_j;
    type y_forward_i_backward_j;

    for (Index i = 0; i < parameters_number; i++)
    {
        h_i = calculate_h(parameters(i));

        x_backward_2i(i) -= static_cast<type>(2.0) * h_i;

        neural_network->forward_propagate(batch.get_input_pairs(),
                                          x_backward_2i,
                                          forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_error_lm(batch, forward_propagation, back_propagation_lm);

        y_backward_2i = back_propagation_lm.error();

        x_backward_2i(i) += static_cast<type>(2.0) * h_i;

        x_backward_i(i) -= h_i;

        neural_network->forward_propagate(batch.get_input_pairs(),
                                          x_backward_i,
                                          forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_error_lm(batch, forward_propagation, back_propagation_lm);

        y_backward_i = back_propagation_lm.error();

        x_backward_i(i) += h_i;

        x_forward_i(i) += h_i;

        neural_network->forward_propagate(batch.get_input_pairs(),
                                          x_forward_i,
                                          forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_error_lm(batch, forward_propagation, back_propagation_lm);

        y_forward_i = back_propagation_lm.error();

        x_forward_i(i) -= h_i;

        x_forward_2i(i) += static_cast<type>(2.0) * h_i;

        neural_network->forward_propagate(batch.get_input_pairs(),
                                          x_forward_2i,
                                          forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_error_lm(batch, forward_propagation, back_propagation_lm);

        y_forward_2i = back_propagation_lm.error();

        x_forward_2i(i) -= static_cast<type>(2.0) * h_i;

        H(i, i) = (-y_forward_2i + type(16.0) * y_forward_i - type(30.0) * y + type(16.0) * y_backward_i - y_backward_2i) / (type(12.0) * pow(h_i, type(2)));

        for (Index j = i; j < parameters_number; j++)
        {
            if(j == i)
                continue;

            h_j = calculate_h(parameters(j));

            x_backward_ij(i) -= h_i;
            x_backward_ij(j) -= h_j;

            neural_network->forward_propagate(batch.get_input_pairs(),
                                              x_backward_ij,
                                              forward_propagation);

            calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_error_lm(batch, forward_propagation, back_propagation_lm);

            y_backward_ij = back_propagation_lm.error();

            x_backward_ij(i) += h_i;
            x_backward_ij(j) += h_j;

            x_forward_ij(i) += h_i;
            x_forward_ij(j) += h_j;

            neural_network->forward_propagate(batch.get_input_pairs(),
                                              x_forward_ij,
                                              forward_propagation);

            calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_error_lm(batch, forward_propagation, back_propagation_lm);

            y_forward_ij = back_propagation_lm.error();

            x_forward_ij(i) -= h_i;
            x_forward_ij(j) -= h_j;

            x_backward_i_forward_j(i) -= h_i;
            x_backward_i_forward_j(j) += h_j;

            neural_network->forward_propagate(batch.get_input_pairs(),
                                              x_backward_i_forward_j,
                                              forward_propagation);

            calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_error_lm(batch, forward_propagation, back_propagation_lm);

            y_backward_i_forward_j = back_propagation_lm.error();

            x_backward_i_forward_j(i) += h_i;
            x_backward_i_forward_j(j) -= h_j;

            x_forward_i_backward_j(i) += h_i;
            x_forward_i_backward_j(j) -= h_j;

            neural_network->forward_propagate(batch.get_input_pairs(),
                                              x_forward_i_backward_j,
                                              forward_propagation);

            calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

            calculate_error_lm(batch, forward_propagation, back_propagation_lm);

            y_forward_i_backward_j = back_propagation_lm.error();

            x_forward_i_backward_j(i) -= h_i;
            x_forward_i_backward_j(j) += h_j;

            H(i, j) = (y_forward_ij - y_forward_i_backward_j - y_backward_i_forward_j + y_backward_ij) / (type(4.0) * h_i * h_j);
        }
    }

    for (Index i = 0; i < parameters_number; i++)
        for (Index j = 0; j < i; j++)
            H(i, j) = H(j, i);

    return H;
}


type LossIndex::calculate_h(const type& x) 
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


vector<vector<pair<type*, dimensions>>> BackPropagationLM::get_layer_delta_pairs() const
{
    NeuralNetwork* neural_network_ptr = loss_index->get_neural_network();

    const Index layers_number = neural_network_ptr->get_layers_number();

    const vector<vector<Index>>& layer_input_indices = neural_network_ptr->get_layer_input_indices();
    const vector<vector<Index>> layer_output_indices = neural_network_ptr->get_layer_output_indices();

    const vector<unique_ptr<LayerBackPropagationLM>>& layer_back_propagations = neural_network.get_layers();

    vector<pair<type*, dimensions>> input_derivative_pairs;

    vector<vector<pair<type*, dimensions>>> layer_delta_pairs(layers_number);

    const Index first_trainable_layer_index = neural_network_ptr->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network_ptr->get_last_trainable_layer_index();

    for (Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
    {
        if (i == last_trainable_layer_index)
        {
            layer_delta_pairs[i].push_back(get_output_deltas_pair());

            continue;
        }

        for (Index j = 0; j < Index(layer_input_indices[i].size()); j++)
        {
            const Index output_index = layer_output_indices[i][j];
            const Index input_index = neural_network_ptr->find_input_index(layer_input_indices[output_index], i);

            input_derivative_pairs = layer_back_propagations[output_index]->get_input_derivative_pairs();

            layer_delta_pairs[i].push_back(input_derivative_pairs[input_index]);
        }
    }

    return layer_delta_pairs;
}


pair<type*, dimensions> BackPropagationLM::get_output_deltas_pair() const
{
    return {(type*)output_deltas.data(), output_deltas_dimensions};
}


void BackPropagationLM::set(const Index &new_batch_samples_number,
                            LossIndex *new_loss_index)
{
    loss_index = new_loss_index;
    
    batch_samples_number = new_batch_samples_number;

    if(!loss_index)
        return;

    NeuralNetwork* neural_network_ptr = loss_index->get_neural_network();

    if(!neural_network_ptr)
        return;

    if(!neural_network_ptr)
        throw runtime_error("Neural network is null.");

    const Index parameters_number =
        neural_network_ptr->get_parameters_number();
    
    const Index outputs_number = neural_network_ptr->get_outputs_number();

    const dimensions output_dimensions = neural_network_ptr->get_output_dimensions();
        
    neural_network.set(batch_samples_number, neural_network_ptr);
    
    parameters = neural_network_ptr->get_parameters();
        
    loss = type(0);
    
    gradient.resize(parameters_number);
    
    regularization_gradient.resize(parameters_number);
    regularization_gradient.setZero();
    
    squared_errors_jacobian.resize(batch_samples_number, parameters_number);

    hessian.resize(parameters_number, parameters_number);
    
    regularization_hessian.resize(parameters_number, parameters_number);
    regularization_hessian.setZero();   

    errors.resize(batch_samples_number, outputs_number);
    
    squared_errors.resize(batch_samples_number);
    
    output_deltas_dimensions.resize(output_dimensions.size() + 1);
    output_deltas_dimensions[0] = batch_samples_number;
    
    Index size = batch_samples_number;

    for(Index i = 0; i < Index(output_dimensions.size()); i++)
    {
        output_deltas_dimensions[i + 1] = output_dimensions[i];

        size *= output_dimensions[i];
    }

    output_deltas.resize(size);
}


BackPropagationLM::BackPropagationLM(const Index &new_batch_samples_number, LossIndex *new_loss_index) 
{
    set(new_batch_samples_number, new_loss_index);
}

} // namespace opennn

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
