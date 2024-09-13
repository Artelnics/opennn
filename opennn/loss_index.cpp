//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O S S   I N D E X   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "neural_network_forward_propagation.h"
#include "tensors.h"
#include "loss_index.h"
#include "back_propagation.h"
#include "cross_entropy_error_3d.h"

namespace opennn
{

LossIndex::LossIndex()
{
    set_default();
}


LossIndex::LossIndex(NeuralNetwork* new_neural_network, DataSet* new_data_set)
    : neural_network(new_neural_network),
      data_set(new_data_set)
{
    set_default();
}


LossIndex::~LossIndex()
{
    delete thread_pool;
    delete thread_pool_device;
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
    if(neural_network)
    {
        return true;
    }
    else
    {
        return false;
    }
}


bool LossIndex::has_data_set() const
{
    if(data_set)
    {
        return true;
    }
    else
    {
        return false;
    }
}


LossIndex::RegularizationMethod LossIndex::get_regularization_method() const
{
    return regularization_method;
}


void LossIndex::set()
{
    neural_network = nullptr;
    data_set = nullptr;

    set_default();
}


void LossIndex::set(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;
    data_set = nullptr;

    set_default();
}


void LossIndex::set(DataSet* new_data_set)
{
    neural_network = nullptr;
    data_set = new_data_set;

    set_default();
}


void LossIndex::set(NeuralNetwork* new_neural_network, DataSet* new_data_set)
{
    neural_network = new_neural_network;

    data_set = new_data_set;

    set_default();
}


void LossIndex::set(const LossIndex& other_error_term)
{
    neural_network = other_error_term.neural_network;

    data_set = other_error_term.data_set;

    regularization_method = other_error_term.regularization_method;

    display = other_error_term.display;
}


void LossIndex::set_threads_number(const int& new_threads_number)
{
    if(thread_pool != nullptr) delete thread_pool;
    if(thread_pool_device != nullptr) delete thread_pool_device;

    thread_pool = new ThreadPool(new_threads_number);
    thread_pool_device = new ThreadPoolDevice(thread_pool, new_threads_number);
}


void LossIndex::set_neural_network(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;
}


void LossIndex::set_data_set(DataSet* new_data_set)
{
    data_set = new_data_set;
}


void LossIndex::set_default()
{
    delete thread_pool;
    delete thread_pool_device;

    const int n = omp_get_max_threads();

    thread_pool = new ThreadPool(n);
    thread_pool_device = new ThreadPoolDevice(thread_pool, n);

    regularization_method = RegularizationMethod::L2;
}


void LossIndex::set_regularization_method(const string& new_regularization_method)
{
    if(new_regularization_method == "L1_NORM")
    {
        set_regularization_method(RegularizationMethod::L1);
    }
    else if(new_regularization_method == "L2_NORM")
    {
        set_regularization_method(RegularizationMethod::L2);
    }
    else if(new_regularization_method == "NO_REGULARIZATION")
    {
        set_regularization_method(RegularizationMethod::NoRegularization);
    }
    else
    {
        throw runtime_error("Unknown regularization method: " + new_regularization_method + ".");
    }
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


bool LossIndex::has_selection() const
{
    if(data_set->get_selection_samples_number() != 0)
        return true;
    else
        return false;
}


Index LossIndex::find_input_index(const Tensor<Index, 1>& layer_inputs_indices, const Index layer_index) const
{
    for(Index i = 0; i < layer_inputs_indices.size(); i++)
    {
        if(layer_inputs_indices(i) == layer_index)  return i;
    }
    return -1;
}


void LossIndex::check() const
{
    if(!neural_network)
        throw runtime_error("Pointer to neural network is nullptr.\n");

    // Data set

    if(!data_set)
        throw runtime_error("Pointer to data set is nullptr.\n");
}


void LossIndex::calculate_errors_lm(const Batch& batch,
                                    const ForwardPropagation & neural_network_forward_propagation,
                                    BackPropagationLM & back_propagation) const
{
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();
    
    const pair<type*, dimensions> outputs_pair = neural_network_forward_propagation.layers(last_trainable_layer_index)->get_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs(outputs_pair.first, outputs_pair.second[0], outputs_pair.second[1]);

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0], targets_pair.second[1]);

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
    // Loss index

    calculate_error(batch, forward_propagation, back_propagation);

    calculate_layers_error_gradient(batch, forward_propagation, back_propagation);

    assemble_layers_error_gradient(back_propagation);

    // Loss

    back_propagation.loss = back_propagation.error;

    // Regularization
    
    add_regularization(back_propagation);
    
}


void LossIndex::add_regularization(BackPropagation& back_propagation) const
{
    if(regularization_method == RegularizationMethod::NoRegularization) return;

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


void LossIndex::back_propagate_lm(const Batch& batch,
                                  ForwardPropagation& forward_propagation,
                                  BackPropagationLM& back_propagation_lm) const
{
    calculate_errors_lm(batch, forward_propagation, back_propagation_lm);
    
    calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);
    
    calculate_error_lm(batch, forward_propagation, back_propagation_lm);
    
    calculate_layers_squared_errors_jacobian_lm(batch, forward_propagation, back_propagation_lm);
    
    calculate_error_gradient_lm(batch, back_propagation_lm);

    calculate_error_hessian_lm(batch, back_propagation_lm);

    // Loss

    back_propagation_lm.loss = back_propagation_lm.error;

    // Regularization
    
    if(regularization_method != RegularizationMethod::NoRegularization)
    {
        const type regularization = calculate_regularization(back_propagation_lm.parameters);

        back_propagation_lm.loss += regularization_weight*regularization;

        calculate_regularization_gradient(back_propagation_lm.parameters, back_propagation_lm.regularization_gradient);

        back_propagation_lm.gradient.device(*thread_pool_device) += regularization_weight*back_propagation_lm.regularization_gradient;

        calculate_regularization_hessian(back_propagation_lm.parameters, back_propagation_lm.regularization_hessian);

        back_propagation_lm.hessian += regularization_weight*back_propagation_lm.regularization_hessian;
    }
}


void LossIndex::calculate_layers_squared_errors_jacobian_lm(const Batch& batch,
                                                            ForwardPropagation& forward_propagation,
                                                            BackPropagationLM& back_propagation_lm) const
{  
    const Tensor<Layer*, 1> layers = neural_network->get_layers();

    const Index layers_number = layers.size();

    if(layers_number == 0) return;

    const Index batch_samples_number = batch.get_batch_samples_number();

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const Tensor<Tensor<Index, 1>, 1>& layers_inputs_indices = neural_network->get_layers_input_indices();
    const Tensor<Tensor<Index, 1>, 1>& layers_outputs_indices = back_propagation_lm.layers_outputs_indices;

    const Tensor<Index, 1> trainable_layers_parameters_number = neural_network->get_trainable_layers_parameters_numbers();

    Layer* layer = nullptr;

    LayerForwardPropagation* layer_forward_propagation = nullptr;
    LayerBackPropagationLM* layer_back_propagation = nullptr;

    Tensor<pair<type*, dimensions>, 1> layer_inputs;
    Tensor<pair<type*, dimensions>, 1> layer_deltas;
    Index input_index;

    // Hidden layers
    
    calculate_output_delta_lm(batch, forward_propagation, back_propagation_lm);
    
    for(Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
    {        
        layer = layers(i);

        layer_forward_propagation = forward_propagation.layers(i);
        layer_back_propagation = back_propagation_lm.neural_network.layers(i - first_trainable_layer_index);
        
        if(i == last_trainable_layer_index)
        {
            layer_deltas.resize(1);

            layer_deltas(0) = back_propagation_lm.get_output_deltas_pair();
        }
        else
        {
            layer_deltas.resize(layers_outputs_indices(i).size());

            for(Index j = 0; j < layers_outputs_indices(i).size(); j++)
            {
                input_index = find_input_index(layers_inputs_indices(layers_outputs_indices(i)(j)), i);

                layer_deltas(j)
                    = back_propagation_lm.neural_network.layers(layers_outputs_indices(i)(j) - first_trainable_layer_index)->get_inputs_derivatives_pair()(input_index);
            }
        }
        
        if(i == first_trainable_layer_index || neural_network->is_input_layer(layers_inputs_indices(i)))
        {
            layer_inputs.resize(1);

            layer_inputs(0) = batch.get_inputs_pair()(0);

            layer_back_propagation->is_first_layer = true;
        }
        else
        {
            layer_inputs.resize(layers_inputs_indices(i).size());

            for(Index j = 0; j < layers_inputs_indices(i).size(); j++)
            {
                layer_inputs(j) = forward_propagation.layers(layers_inputs_indices(i)(j))->get_outputs_pair();
            }
        }
        
        layer->back_propagate_lm(layer_inputs, layer_deltas, layer_forward_propagation, layer_back_propagation);        
    }
    
    Index memory_index = 0;

    for(Index i = 0; i < last_trainable_layer_index - first_trainable_layer_index; i++)
    {
        layer_back_propagation = back_propagation_lm.neural_network.layers(i);
        
        layer->insert_squared_errors_Jacobian_lm(layer_back_propagation,
                                                 memory_index,
                                                 back_propagation_lm.squared_errors_jacobian);
        
        memory_index += trainable_layers_parameters_number(i) * batch_samples_number;
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


string LossIndex::get_error_type() const
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
        case RegularizationMethod::NoRegularization: return type(0);

        case RegularizationMethod::L1: return l1_norm(thread_pool_device, parameters);

        case RegularizationMethod::L2: return l2_norm(thread_pool_device, parameters);

        default: return type(0);
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
        l1_norm_gradient(thread_pool_device, parameters, regularization_gradient); return;

    case RegularizationMethod::L2:
        l2_norm_gradient(thread_pool_device, parameters, regularization_gradient); return;

    default:
        return;
    }
}


void LossIndex::calculate_regularization_hessian(Tensor<type, 1>& parameters, Tensor<type, 2>& regularization_hessian) const
{
    switch(regularization_method)
    {
    case RegularizationMethod::L1:
        l1_norm_hessian(thread_pool_device, parameters, regularization_hessian);

        return;

    case RegularizationMethod::L2:
        l2_norm_hessian(thread_pool_device, parameters, regularization_hessian);

        return;

    default:
        
        return;
    }
}


void LossIndex::calculate_layers_error_gradient(const Batch& batch,
                                                ForwardPropagation& forward_propagation,
                                                BackPropagation& back_propagation) const
{
    const Tensor<Layer*, 1> layers = neural_network->get_layers();

    const Index layers_number = layers.size();

    if(layers_number == 0) return;

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const Tensor<Tensor<Index, 1>, 1>& layers_inputs_indices = neural_network->get_layers_input_indices();
    const Tensor<Tensor<Index, 1>, 1>& layers_outputs_indices = back_propagation.layers_outputs_indices;

    Layer* layer = nullptr;

    LayerForwardPropagation* layer_forward_propagation = nullptr;
    LayerBackPropagation* layer_back_propagation = nullptr;

    Tensor<pair<type*, dimensions>, 1> layer_inputs;
    Tensor<pair<type*, dimensions>, 1> layer_deltas;
    Index input_index;

    // Hidden layers

    calculate_output_delta(batch, forward_propagation, back_propagation);

    for(Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
    {
        layer = layers(i);

        layer_forward_propagation = forward_propagation.layers(i);
        layer_back_propagation = back_propagation.neural_network.layers(i);

        if(i == last_trainable_layer_index)
        {
            layer_deltas.resize(1);

            layer_deltas(0) = back_propagation.get_output_deltas_pair();
        }
        else
        {
            layer_deltas.resize(layers_outputs_indices(i).size());

            for(Index j = 0; j < layers_outputs_indices(i).size(); j++)
            {
                input_index = find_input_index(layers_inputs_indices(layers_outputs_indices(i)(j)), i);

                layer_deltas(j) = back_propagation.neural_network.layers(layers_outputs_indices(i)(j))->get_inputs_derivatives_pair()(input_index); 
            }
        }

        if(i == first_trainable_layer_index || neural_network->is_input_layer(layers_inputs_indices(i)))
        {
            layer_inputs.resize(1);

            layer_inputs(0) = batch.get_inputs_pair()(0);

            layer_back_propagation->is_first_layer = true;
        }
        else if(neural_network->is_context_layer(layers_inputs_indices(i)))
        {
            layer_inputs.resize(1);

            layer_inputs(0) = batch.get_inputs_pair()(1);

            layer_back_propagation->is_first_layer = true;
        }
        else
        {
            layer_inputs.resize(layers_inputs_indices(i).size());

            for(Index j = 0; j < layers_inputs_indices(i).size(); j++)
            {
                layer_inputs(j) = forward_propagation.layers(layers_inputs_indices(i)(j))->get_outputs_pair();
            }
        }

        layer->back_propagate(layer_inputs, layer_deltas, layer_forward_propagation, layer_back_propagation);
    }
}


void LossIndex::assemble_layers_error_gradient(BackPropagation& back_propagation) const
{
    const Tensor<Layer*, 1> layers = neural_network->get_layers();

    const Index layers_number = neural_network->get_layers_number();

    const Tensor<Index, 1> layers_parameters_number = neural_network->get_layers_parameters_numbers();

    Index index = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        layers(i)->insert_gradient(back_propagation.neural_network.layers(i),
                                   index,
                                   back_propagation.gradient);

        index += layers_parameters_number(i);
    }
}


void LossIndex::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("LossIndex");

    file_stream.CloseElement();
}


void LossIndex::regularization_from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("Regularization");

    if(!root_element)
        throw runtime_error("Regularization tag not found.\n");

    const string new_regularization_method = root_element->Attribute("Type");

    set_regularization_method(new_regularization_method);

    const tinyxml2::XMLElement* element = root_element->FirstChildElement("RegularizationWeight");

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


void LossIndex::write_regularization_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("Regularization");

    // Regularization method

    
    switch(regularization_method)
    {
    case RegularizationMethod::L1:
    {
        file_stream.PushAttribute("Type", "L1_NORM");
    }
    break;

    case RegularizationMethod::L2:
    {
        file_stream.PushAttribute("Type", "L2_NORM");
    }
    break;

    case RegularizationMethod::NoRegularization:
    {
        file_stream.PushAttribute("Type", "NO_REGULARIZATION");
    }
    break;

    default: break;
    }

    // Regularization weight

    file_stream.OpenElement("RegularizationWeight");
    file_stream.PushText(to_string(regularization_weight).c_str());
    file_stream.CloseElement();

    // Close regularization

    file_stream.CloseElement();
}


void LossIndex::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("MeanSquaredError");

    if(!root_element)
        throw runtime_error("Mean squared element is nullptr.\n");

    // Regularization

    tinyxml2::XMLDocument regularization_document;

    const tinyxml2::XMLElement* regularization_element = root_element->FirstChildElement("Regularization");

    tinyxml2::XMLNode* element_clone = regularization_element->DeepClone(&regularization_document);

    regularization_document.InsertFirstChild(element_clone);

    regularization_from_XML(regularization_document);
}


BackPropagation::~BackPropagation()
{
}


void BackPropagation::set(const Index& new_batch_samples_number, LossIndex* new_loss_index)
{
    loss_index = new_loss_index;

    batch_samples_number = new_batch_samples_number;

    // Neural network

    NeuralNetwork* neural_network_p = loss_index->get_neural_network();

    const Index parameters_number = neural_network_p->get_parameters_number();

    const dimensions output_dimensions = neural_network_p->get_output_dimensions();

    const Index outputs_number = output_dimensions[0];

    set_layers_outputs_indices(neural_network_p->get_layers_input_indices());

    // First order loss

    neural_network.set(batch_samples_number, neural_network_p);

    error = type(0);

    loss = type(0);

    errors.resize(batch_samples_number, outputs_number);

    parameters = neural_network_p->get_parameters();

    gradient.resize(parameters_number);

    regularization_gradient.resize(parameters_number);

    output_deltas_dimensions.resize(output_dimensions.size() + 1);
    output_deltas_dimensions[0] = batch_samples_number;

    Index size = batch_samples_number;

    for(Index i = 0; i < Index(output_dimensions.size()); i++)
    {
        output_deltas_dimensions[i + 1] = output_dimensions[i];

        size *= output_dimensions[i];
    }

    output_deltas.resize(size);

    if(is_instance_of<CrossEntropyError3D>(loss_index))
    {
        predictions.resize(batch_samples_number, outputs_number);
        matches.resize(batch_samples_number, outputs_number);
        mask.resize(batch_samples_number, outputs_number);
    }
}


void BackPropagation::set_layers_outputs_indices(const Tensor<Tensor<Index, 1>, 1>& layer_inputs_indices)
{
    Index layers_number = layer_inputs_indices.size();

    layers_outputs_indices.resize(layers_number);

    Index layer_count = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        for(Index j = 0; j < layers_number; j++)
        {
            for(Index k = 0; k < layer_inputs_indices(j).size(); k++)    if(layer_inputs_indices(j)(k) == i)    layer_count++;
        }

        layers_outputs_indices(i).resize(layer_count);
        layer_count = 0;

        for(Index j = 0; j < layers_number; j++)
        {
            for(Index k = 0; k < layer_inputs_indices(j).size(); k++)
            {
                if(layer_inputs_indices(j)(k) == i)
                {
                    layers_outputs_indices(i)(layer_count) = j;
                    layer_count++;
                }
            }
        }

        layer_count = 0;
    }
}


pair<type*, dimensions> BackPropagation::get_output_deltas_pair() const
{
    return pair<type*, dimensions>((type*)output_deltas.data(), output_deltas_dimensions);
}


Tensor<type, 1> LossIndex::calculate_numerical_gradient()
{
    const Index samples_number = data_set->get_training_samples_number();

    const Tensor<Index, 1> samples_indices = data_set->get_training_samples_indices();
    const Tensor<Index, 1> input_variables_indices = data_set->get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = data_set->get_target_variables_indices();

    Batch batch(samples_number, data_set);
    batch.fill(samples_indices, input_variables_indices, target_variables_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagation back_propagation(samples_number, this);

    const Tensor<type, 1> parameters = neural_network->get_parameters();

    const Index parameters_number = parameters.size();

    type h;
    Tensor<type, 1> parameters_forward = parameters;
    Tensor<type, 1> parameters_backward = parameters;

    type error_forward;
    type error_backward;

    Tensor<type, 1> numerical_gradient(parameters_number);
    numerical_gradient.setConstant(type(0));

    const Tensor<pair<type*, dimensions>, 1> inputs_pair = batch.get_inputs_pair();

    for(Index i = 0; i < parameters_number; i++)
    {
        h = calculate_h(parameters(i));

       parameters_forward(i) += h;
       
       neural_network->forward_propagate(inputs_pair,
                                         parameters_forward,
                                         forward_propagation);

       calculate_error(batch, forward_propagation, back_propagation);

       error_forward = back_propagation.error;

       parameters_forward(i) -= h;

       parameters_backward(i) -= h;
       
       neural_network->forward_propagate(inputs_pair,
                                         parameters_backward,
                                         forward_propagation);

       calculate_error(batch, forward_propagation, back_propagation);

       error_backward = back_propagation.error;

       parameters_backward(i) += h;

       numerical_gradient(i) = (error_forward - error_backward)/type(2*h);
    }

    return numerical_gradient;
}


Tensor<type, 1> LossIndex::calculate_numerical_inputs_derivatives()
{
    const Index samples_number = data_set->get_training_samples_number();
    const dimensions inputs_dimensions = data_set->get_input_dimensions();

    Index inputs_number = 1;

    for(Index i = 0; i < inputs_dimensions.size(); i++)
        inputs_number *= inputs_dimensions[i];

    inputs_number = samples_number * inputs_number;

    const Tensor<Index, 1> samples_indices = data_set->get_training_samples_indices();
    const Tensor<Index, 1> input_variables_indices = data_set->get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = data_set->get_target_variables_indices();

    Batch batch(samples_number, data_set);
    batch.fill(samples_indices, input_variables_indices, target_variables_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagation back_propagation(samples_number, this);

    type h;

    type error_forward;
    type error_backward;

    Tensor<type, 1> numerical_inputs_derivatives(inputs_number);
    numerical_inputs_derivatives.setConstant(type(0));

    const Tensor<pair<type*, dimensions>, 1> inputs_pair = batch.get_inputs_pair();

    TensorMap<Tensor<type, 1>> inputs_vector(inputs_pair(0).first, inputs_number);

    for (Index i = 0; i < inputs_number; i++)
    {
        h = calculate_h(inputs_vector(i));

        inputs_pair(0).first[i] += h;

        neural_network->forward_propagate(inputs_pair, forward_propagation);

        calculate_error(batch, forward_propagation, back_propagation);
        error_forward = back_propagation.error;

        inputs_pair(0).first[i] -= h;

        inputs_pair(0).first[i] -= h;

        neural_network->forward_propagate(inputs_pair, forward_propagation);

        calculate_error(batch, forward_propagation, back_propagation);
        error_backward = back_propagation.error;

        inputs_pair(0).first[i] += h;

        numerical_inputs_derivatives(i) = (error_forward - error_backward) / type(2 * h);
    }

    return numerical_inputs_derivatives;
}


Tensor<type, 2> LossIndex::calculate_numerical_jacobian()
{
    BackPropagationLM back_propagation_lm;

    const Index samples_number = data_set->get_training_samples_number();

    Batch batch(samples_number, data_set);

    const Tensor<Index, 1> samples_indices = data_set->get_training_samples_indices();

    const Tensor<Index, 1> input_variables_indices = data_set->get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = data_set->get_target_variables_indices();

    batch.fill(samples_indices, input_variables_indices, target_variables_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagation back_propagation(samples_number, this);

    Tensor<type, 1> parameters = neural_network->get_parameters();

    const Index parameters_number = parameters.size();

    back_propagation_lm.set(samples_number, this);
    
    neural_network->forward_propagate(batch.get_inputs_pair(),
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
        neural_network->forward_propagate(batch.get_inputs_pair(),
                                                  parameters_backward,
                                                  forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);
        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);
        error_terms_backward = back_propagation_lm.squared_errors;
        parameters_backward(j) += h;

        parameters_forward(j) += h;
        neural_network->forward_propagate(batch.get_inputs_pair(),
                                                  parameters_forward,
                                                  forward_propagation);
        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);
        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);
        error_terms_forward = back_propagation_lm.squared_errors;
        parameters_forward(j) -= h;

        for(Index i = 0; i < samples_number; i++)
        {
            jacobian(i,j) = (error_terms_forward(i) - error_terms_backward(i))/(type(2.0)*h);
        }
    }

    return jacobian;
}


type LossIndex::calculate_eta() const
{
    const Index precision_digits = 6;

    return pow(type(10.0), type(-1.0*precision_digits));
}


type LossIndex::calculate_h(const type& x) const
{
    const type eta = calculate_eta();

    return sqrt(eta)*(type(1) + abs(x));
}


void BackPropagationLM::print() const {
    cout << "Loss index back-propagation LM" << endl;
    
    cout << "Errors:" << endl;
    cout << errors << endl;
    
    cout << "Squared errors:" << endl;
    cout << squared_errors << endl;
    
    cout << "Squared errors Jacobian:" << endl;
    cout << squared_errors_jacobian << endl;
    
    cout << "Error:" << endl;
    cout << error << endl;
    
    cout << "Loss:" << endl;
    cout << loss << endl;
    
    cout << "Gradient:" << endl;
    cout << gradient << endl;
    
    cout << "Hessian:" << endl;
    cout << hessian << endl;
}

void BackPropagationLM::set_layers_outputs_indices(const Tensor<Tensor<Index, 1>, 1>& layer_inputs_indices)
{
    Index layers_number = layer_inputs_indices.size();

    layers_outputs_indices.resize(layers_number);

    Index layer_count = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        for(Index j = 0; j < layers_number; j++)
        {
            for(Index k = 0; k < layer_inputs_indices(j).size(); k++)    if(layer_inputs_indices(j)(k) == i)    layer_count++;
        }

        layers_outputs_indices(i).resize(layer_count);
        layer_count = 0;

        for(Index j = 0; j < layers_number; j++)
        {
            for(Index k = 0; k < layer_inputs_indices(j).size(); k++)
            {
                if(layer_inputs_indices(j)(k) == i)
                {
                    layers_outputs_indices(i)(layer_count) = j;
                    layer_count++;
                }
            }
        }

        layer_count = 0;
    }
}


pair<type*, dimensions> BackPropagationLM::get_output_deltas_pair() const
{
    return pair<type*, dimensions>((type*)output_deltas.data(), output_deltas_dimensions);
}


void BackPropagationLM::set(const Index &new_batch_samples_number,
                                     LossIndex *new_loss_index) 
{
    loss_index = new_loss_index;
    
    batch_samples_number = new_batch_samples_number;
    
    NeuralNetwork *neural_network_p = loss_index->get_neural_network();
    
    const Index parameters_number =
        neural_network_p->get_parameters_number();
    
    const Index outputs_number = neural_network_p->get_outputs_number();

    const dimensions output_dimensions = neural_network_p->get_output_dimensions();
    
    set_layers_outputs_indices(neural_network_p->get_layers_input_indices());
    
    neural_network.set(batch_samples_number, neural_network_p);
    
    parameters = neural_network_p->get_parameters();
    
    error = type(0);
    
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


BackPropagationLM::BackPropagationLM() 
{
}

}
 // namespace opennn
//  // namespace opennnOpenNN: Open Neural // namespace  // namespace opennnopennn Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
