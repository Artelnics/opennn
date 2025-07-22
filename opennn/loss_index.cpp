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

namespace opennn
{

LossIndex::LossIndex(const NeuralNetwork* new_neural_network, const Dataset* new_dataset)
{
    set(new_neural_network, new_dataset);
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


bool LossIndex::has_dataset() const
{
    return dataset;
}


string LossIndex::get_regularization_method() const
{
    return regularization_method;
}


void LossIndex::set(const NeuralNetwork* new_neural_network, const Dataset* new_dataset)
{
    neural_network = const_cast<NeuralNetwork*>(new_neural_network);
    dataset = const_cast<Dataset*>(new_dataset);

    thread_pool.reset();
    thread_pool_device.reset();

    const unsigned int threads_number = thread::hardware_concurrency();
    thread_pool = make_unique<ThreadPool>(threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);

    regularization_method = "L2";
}


void LossIndex::set_threads_number(const int& new_threads_number)
{
    if(thread_pool != nullptr)
        thread_pool.reset();
    if(thread_pool_device != nullptr)
        thread_pool_device.reset();

    thread_pool = make_unique<ThreadPool>(new_threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), new_threads_number);
}


void LossIndex::set_neural_network(const NeuralNetwork* new_neural_network)
{
    neural_network = const_cast<NeuralNetwork*>(new_neural_network);
}


void LossIndex::set_dataset(const Dataset* new_dataset)
{
    dataset = const_cast<Dataset*>(new_dataset);
}


void LossIndex::set_regularization_method(const string& new_regularization_method)
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


void LossIndex::calculate_errors_lm(const Batch& batch,
                                    const ForwardPropagation & forward_propagation,
                                    BackPropagationLM & back_propagation) const
{
    const pair<type*, dimensions> outputs_pair
        = forward_propagation.get_last_trainable_layer_outputs_pair();

    const TensorMap<Tensor<type, 2>> outputs = tensor_map<2>(outputs_pair);

    const pair<type*, dimensions> targets_pair = batch.get_target_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map<2>(targets_pair);

    back_propagation.errors.device(*thread_pool_device) = outputs - targets;
}


void LossIndex::calculate_squared_errors_lm(const Batch&,
                                            const ForwardPropagation&,
                                            BackPropagationLM& back_propagation_lm) const
{
    const Tensor<type, 2>& errors = back_propagation_lm.errors;

    Tensor<type, 1>& squared_errors = back_propagation_lm.squared_errors;

    squared_errors.device(*thread_pool_device) = errors.square().sum(array<int, 1>({1})).sqrt();
}


void LossIndex::back_propagate(const Batch& batch,
                               ForwardPropagation& forward_propagation,
                               BackPropagation& back_propagation) const
{
    if(batch.is_empty()) return;

    calculate_error(batch, forward_propagation, back_propagation);

    calculate_layers_error_gradient(batch, forward_propagation, back_propagation);

    back_propagation.loss = back_propagation.error();

    add_regularization(back_propagation);
}


void LossIndex::add_regularization(BackPropagation& back_propagation) const
{
    if(regularization_method == "None")
        return;

    NeuralNetwork* neural_network = back_propagation.loss_index->get_neural_network();

    const Index layers_number = neural_network->get_layers_number();

//    type regularization_value = 0;

    //#pragma omp parallel for schedule(dynamic) // @todo check this pragma vs thread_pool

    for (Index layer_index = 0; layer_index < layers_number; layer_index++)
    {
        Layer* layer = neural_network->get_layer(layer_index).get();

        if(!layer->get_is_trainable())
            continue;

        LayerBackPropagation* layer_back_propagation = back_propagation.neural_network.layers[layer_index].get();

        const vector<pair<type*, Index>>& parameter_pairs = layer->get_parameter_pairs();
        const vector<pair<type*, Index>>& delta_pairs = layer_back_propagation->get_parameter_delta_pairs();

        for (Index parameter_index = 0; parameter_index < Index(parameter_pairs.size()); parameter_index++)
        {
            type* parameter_data = parameter_pairs[parameter_index].first;
            const Index parameter_size = parameter_pairs[parameter_index].second;
            TensorMap<Tensor<type, 1>> parameters_map(parameter_data, parameter_size);

            type* delta_data = delta_pairs[parameter_index].first;
            TensorMap<Tensor<type, 1>> delta_map(delta_data, parameter_size);

            if(regularization_method == "L1")
            {
                const Tensor<type, 0> norm = parameters_map.abs().sum();

                back_propagation.loss += regularization_weight*norm(0);

                delta_map += regularization_weight*parameters_map.sign();

            }
            else if(regularization_method == "L2")
            {
                Tensor<type, 0> norm = parameters_map.square().sum().sqrt();

                if(norm(0) >= NUMERIC_LIMITS_MIN)
                {
                    back_propagation.loss += regularization_weight*norm(0);

                    delta_map += parameters_map*(regularization_weight/norm(0));
                }
            }
            else
                throw runtime_error("Unknown regularization method: " + regularization_method);
        }
    }

    //back_propagation.regularization = regularization_value;
    //back_propagation.loss += regularization_weight * regularization_value;
}



void LossIndex::add_regularization_lm(BackPropagationLM& back_propagation_lm) const
{
    if(regularization_method == "None")
        return;

    Tensor<type, 1>& parameters = back_propagation_lm.parameters;

    type& loss = back_propagation_lm.loss;

    Tensor<type, 1>& gradient = back_propagation_lm.gradient;

    Tensor<type, 2>& hessian = back_propagation_lm.hessian;

    if(regularization_method == "L1")
    {
        const Tensor<type, 0> norm = parameters.abs().sum();

        loss += regularization_weight*norm(0);
        gradient += regularization_weight*parameters.sign();

    }
    else if(regularization_method == "L2")
    {
        Tensor<type, 0> norm = parameters.square().sum().sqrt();

        if(norm(0) < NUMERIC_LIMITS_MIN) return;

        loss += regularization_weight*norm(0);
        gradient += parameters*(regularization_weight/norm(0));
        hessian += self_kronecker_product(thread_pool_device.get(), parameters)/(norm(0)*norm(0)*norm(0));
    }
    else
        throw runtime_error("Unknown regularization method: " + regularization_method);
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

    const Index samples_number = batch.get_samples_number();

    Index index = 0;

    for(Index i = 0; i < layers_number; i++)
    {
        layers[i]->insert_squared_errors_Jacobian_lm(back_propagation_lm.neural_network.layers[i],
                                                     index,
                                                     back_propagation_lm.squared_errors_jacobian);

        index += layer_parameter_numbers[i] * samples_number;
    }

}


void LossIndex::calculate_error_gradient_lm(const Batch&,
                                            BackPropagationLM& back_propagation_lm) const
{
    const Tensor<type, 1>& squared_errors = back_propagation_lm.squared_errors;
    const Tensor<type, 2>& squared_errors_jacobian = back_propagation_lm.squared_errors_jacobian;

    Tensor<type, 1>& gradient = back_propagation_lm.gradient;

    gradient.device(*thread_pool_device) = squared_errors_jacobian.contract(squared_errors, axes(1,0));
}


string LossIndex::get_name() const
{
    return string();
}



// @todo parallelize

type LossIndex::calculate_regularization(const Tensor<type, 1>& parameters) const
{
    if(regularization_method == "None")
    {
        return type(0);
    }
    else if(regularization_method == "L1")
    {
        const Tensor<type, 0> norm = parameters.abs().sum();

        return regularization_weight*norm(0);
    }
    else if(regularization_method == "L2")
    {
        const Tensor<type, 0> norm = parameters.square().sum().sqrt();

        if(norm(0) < NUMERIC_LIMITS_MIN) return 0;

        return regularization_weight*norm(0);
    }
    else
        throw runtime_error("Unknown regularization method: " + regularization_method);
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


void LossIndex::assemble_layers_error_gradient(const BackPropagation& back_propagation, Tensor<type, 1>& gradient) const
{
    Index index = 0;

    const Index first_trainable_layer = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer = neural_network->get_last_trainable_layer_index();

    for(Index i = first_trainable_layer; i <= last_trainable_layer; i++)
    {
        if (!back_propagation.neural_network.layers[i].get()) continue;

        const vector<pair<type*, Index>> layer_gradient = back_propagation.neural_network.layers[i]->get_parameter_delta_pairs();

        for(Index j = 0; j < Index(layer_gradient.size()); j++)
        {
            memcpy(gradient.data() + index, layer_gradient[j].first, layer_gradient[j].second * sizeof(type));

            index += layer_gradient[j].second;
        }
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

    file_stream.PushAttribute("Type", regularization_method.c_str());

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


BackPropagation::BackPropagation(const Index& new_batch_size, const LossIndex* new_loss_index)
{
    set(new_batch_size, new_loss_index);
}


void BackPropagation::set(const Index& new_samples_number, const LossIndex* new_loss_index)
{
    samples_number = new_samples_number;

    loss_index = const_cast<LossIndex*>(new_loss_index);

    if(!loss_index) return;

    // Neural network

    NeuralNetwork* neural_network_ptr = loss_index->get_neural_network();

    const dimensions output_dimensions = neural_network_ptr->get_output_dimensions();

    const Index outputs_number = output_dimensions[0];

    // First order loss

    neural_network.set(samples_number, neural_network_ptr);

    loss = type(0);

    errors.resize(samples_number, outputs_number);

    output_deltas_dimensions = { samples_number };
    output_deltas_dimensions.insert(output_deltas_dimensions.end(), output_dimensions.begin(), output_dimensions.end());

    const Index size = accumulate(output_dimensions.begin(), output_dimensions.end(), samples_number, multiplies<>());

    output_deltas.resize(size);

    if(is_instance_of<CrossEntropyError3d>(loss_index))
    {
        predictions.resize(samples_number, outputs_number);
        matches.resize(samples_number, outputs_number);
        mask.resize(samples_number, outputs_number);
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
         << "Loss:" << endl
         << loss << endl;
    //<< "Gradient:" << endl
    //<< gradient << endl;

    neural_network.print();
}


type LossIndex::calculate_numerical_error() const
{
    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> training_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_indices = dataset->get_variable_indices("Input");
    // const vector<Index> decoder_variable_indices = dataset->get_variable_indices("Decoder");
    const vector<Index> target_indices = dataset->get_variable_indices("Target");

    Batch batch(samples_number, dataset);

    // batch.fill(sample_indices, input_variable_indices, decoder_variable_indices, target_variable_indices);
    batch.fill(training_indices, input_indices, target_indices);

    cout << "info del batch" << endl;
    batch.print();

    ForwardPropagation forward_propagation(samples_number, neural_network);

    neural_network->forward_propagate(batch.get_input_pairs(),
                                      forward_propagation);

    BackPropagation back_propagation(samples_number, this);

    calculate_error(batch, forward_propagation, back_propagation);

    return back_propagation.error();

    //return 0;
}


Tensor<type, 1> LossIndex::calculate_gradient()
{
    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_variable_indices = dataset->get_variable_indices("Input");

    // const vector<Index> decoder_variable_indices = dataset->get_variable_indices("Decoder");

    const vector<Index> target_variable_indices = dataset->get_variable_indices("Target");

    Batch batch(samples_number, dataset);

    // batch.fill(sample_indices, input_variable_indices, decoder_variable_indices, target_variable_indices);
    batch.fill(sample_indices, input_variable_indices, target_variable_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);
    BackPropagation back_propagation(samples_number, this);

    Tensor<type, 1> parameters;
    neural_network->get_parameters(parameters);

    neural_network->forward_propagate(batch.get_input_pairs(),
                                      parameters,
                                      forward_propagation);

    back_propagate(batch, forward_propagation, back_propagation);

    Tensor<type, 1> gradient(parameters.size());

    assemble_layers_error_gradient(back_propagation, gradient);

    return gradient;
}


Tensor<type, 1> LossIndex::calculate_numerical_gradient()
{
    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_variable_indices = dataset->get_variable_indices("Input");
    const vector<Index> target_variable_indices = dataset->get_variable_indices("Target");
    //const vector<Index> decoder_variable_indices = dataset->get_variable_indices("Decoder");

    // @todo decoder variables

    Batch batch(samples_number, dataset);    
    batch.fill(sample_indices, input_variable_indices, target_variable_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);
    BackPropagation back_propagation(samples_number, this);

    Tensor<type, 1> parameters;
    neural_network->get_parameters(parameters);

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

    // return Tensor<type, 1>();
    return numerical_gradient;
}


Tensor<type, 1> LossIndex::calculate_numerical_gradient_lm()
{
    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_variable_indices = dataset->get_variable_indices("Input");
    const vector<Index> target_variable_indices = dataset->get_variable_indices("Target");

    Batch batch(samples_number, dataset);
    batch.fill(sample_indices, input_variable_indices, target_variable_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);
    BackPropagationLM back_propagation_lm(samples_number, this);

    Tensor<type, 1> parameters;
    neural_network->get_parameters(parameters);

    const Index parameters_number = parameters.size();

    type h = 0;

    Tensor<type, 1> parameters_forward = parameters;
    Tensor<type, 1> parameters_backward = parameters;

    type error_forward = 0;
    type error_backward = 0;

    Tensor<type, 1> numerical_gradient_lm(parameters_number);
    numerical_gradient_lm.setConstant(type(0));

    for(Index i = 0; i < parameters_number; i++)
    {
        h = calculate_h(parameters(i));

        parameters_forward(i) += h;

        neural_network->forward_propagate(batch.get_input_pairs(),
                                          parameters_forward,
                                          forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_error_lm(batch, forward_propagation, back_propagation_lm);

        error_forward = back_propagation_lm.error();

        parameters_forward(i) -= h;

        parameters_backward(i) -= h;

        neural_network->forward_propagate(batch.get_input_pairs(),
                                          parameters_backward,
                                          forward_propagation);

        calculate_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_squared_errors_lm(batch, forward_propagation, back_propagation_lm);

        calculate_error_lm(batch, forward_propagation, back_propagation_lm);

        error_backward = back_propagation_lm.error();

        parameters_backward(i) += h;

        numerical_gradient_lm(i) = (error_forward - error_backward)/type(2*h);
    }

    return numerical_gradient_lm;
}


Tensor<type, 1> LossIndex::calculate_numerical_input_deltas()
{

    const Index samples_number = dataset->get_samples_number("Training");
    const dimensions inputs_dimensions = dataset->get_dimensions("Input");

    const Index values_number = neural_network->get_inputs_number()*samples_number;

    const vector<Index> sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_variable_indices = dataset->get_variable_indices("Input");
    const vector<Index> target_variable_indices = dataset->get_variable_indices("Target");

    Batch batch(samples_number, dataset);
    // batch.fill(sample_indices, input_variable_indices, {}, target_variable_indices);
    batch.fill(sample_indices, input_variable_indices, target_variable_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagation back_propagation(samples_number, this);

    type h;

    type error_forward;
    type error_backward;

    Tensor<type, 1> numerical_inputs_derivatives(values_number);
    numerical_inputs_derivatives.setConstant(type(0));

    const vector<pair<type*, dimensions>>& input_pairs = batch.get_input_pairs();

    TensorMap<Tensor<type, 4>> inputs_vector = tensor_map<4>(input_pairs[0]);

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
    const Index samples_number = dataset->get_samples_number("Training");
    const vector<Index> sample_indices = dataset->get_sample_indices("Training");

    const vector<Index> input_variable_indices = dataset->get_variable_indices("Input");
    const vector<Index> target_variable_indices = dataset->get_variable_indices("Target");

    Batch batch(samples_number, dataset);
    // batch.fill(sample_indices, input_variable_indices, {}, target_variable_indices);
    batch.fill(sample_indices, input_variable_indices, target_variable_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagationLM back_propagation_lm(samples_number, this);

    BackPropagation back_propagation(samples_number, this);

    Tensor<type, 1> parameters;
    neural_network->get_parameters(parameters);

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


Tensor<type, 2> LossIndex::calculate_numerical_hessian()
{
    const Index samples_number = dataset->get_samples_number("Training");

    const vector<Index> sample_indices = dataset->get_sample_indices("Training");
    const vector<Index> input_variable_indices = dataset->get_variable_indices("Input");
    const vector<Index> target_variable_indices = dataset->get_variable_indices("Target");

    Batch batch(samples_number, dataset);
    // batch.fill(sample_indices, input_variable_indices, {}, target_variable_indices);
    batch.fill(sample_indices, input_variable_indices, target_variable_indices);

    ForwardPropagation forward_propagation(samples_number, neural_network);

    BackPropagationLM back_propagation_lm(samples_number, this);

    Tensor<type, 1> parameters;
    neural_network->get_parameters(parameters);

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
            // if(j == i)
            // continue;

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


Tensor<type, 2> LossIndex::calculate_inverse_hessian()
{
    Tensor<type, 2> H = calculate_numerical_hessian();

    const Index parameters_number = H.dimension(0);

    type determinant = 1;

    for(Index i = 0; i < parameters_number; i++)
        determinant *= H(i, i);

    if(abs(determinant) < NUMERIC_LIMITS_MIN)
        throw runtime_error("Hessian is not invertible.");

    Tensor<type, 2> H_inv(parameters_number, parameters_number);
    H_inv.setZero();

    for(Index i = 0; i < parameters_number; i++)
    {
        for(Index j = 0; j < parameters_number; j++)
        {
            const type cofactor = ((i + j) % 2 == 0 ? 1 : -1) * H(j, i);
            H_inv(i, j) = cofactor / determinant;
        }
    }

    return H_inv;
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


void BackPropagationLM::set(const Index& new_samples_number,
                            LossIndex *new_loss_index)
{
    loss_index = new_loss_index;

    samples_number = new_samples_number;

    if(!loss_index)
        return;

    NeuralNetwork* neural_network_ptr = loss_index->get_neural_network();

    if(!neural_network_ptr)
        return;

    const Index parameters_number =
        neural_network_ptr->get_parameters_number();

    const Index outputs_number = neural_network_ptr->get_outputs_number();

    const dimensions output_dimensions = neural_network_ptr->get_output_dimensions();

    neural_network.set(samples_number, neural_network_ptr);

    parameters.resize(parameters_number);

    neural_network_ptr->get_parameters(parameters);

    loss = type(0);

    gradient.resize(parameters_number);

    regularization_gradient.resize(parameters_number);
    regularization_gradient.setZero();

    squared_errors_jacobian.resize(samples_number, parameters_number);

    hessian.resize(parameters_number, parameters_number);

    regularization_hessian.resize(parameters_number, parameters_number);
    regularization_hessian.setZero();

    errors.resize(samples_number, outputs_number);

    squared_errors.resize(samples_number);

    output_deltas_dimensions.resize(output_dimensions.size() + 1);
    output_deltas_dimensions[0] = samples_number;

    Index size = samples_number;

    for(Index i = 0; i < Index(output_dimensions.size()); i++)
    {
        output_deltas_dimensions[i + 1] = output_dimensions[i];

        size *= output_dimensions[i];
    }

    output_deltas.resize(size);
}


BackPropagationLM::BackPropagationLM(const Index& new_batch_size, LossIndex *new_loss_index)
{
    set(new_batch_size, new_loss_index);
}


#ifdef OPENNN_CUDA

void LossIndex::back_propagate_cuda(const BatchCuda& batch_cuda,
                                    ForwardPropagationCuda& forward_propagation_cuda,
                                    BackPropagationCuda& back_propagation_cuda)
{
    if (batch_cuda.is_empty()) return;

    // Loss index

    calculate_error_cuda(batch_cuda, forward_propagation_cuda, back_propagation_cuda);

    calculate_layers_error_gradient_cuda(batch_cuda, forward_propagation_cuda, back_propagation_cuda);

    // Loss

    back_propagation_cuda.loss = back_propagation_cuda.error();

    // Regularization

    add_regularization_cuda(back_propagation_cuda);
}


void LossIndex::calculate_layers_error_gradient_cuda(const BatchCuda& batch_cuda,
                                                     ForwardPropagationCuda& forward_propagation_cuda,
                                                     BackPropagationCuda& back_propagation_cuda) const
{
    const vector<unique_ptr<Layer>>& layers = neural_network->get_layers();

    const Index layers_number = layers.size();

    if (layers_number == 0) return;

    const Index first_trainable_layer_index = neural_network->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    const vector<vector<float*>> layer_input_pairs
        = forward_propagation_cuda.get_layer_inputs_device(batch_cuda.get_input_device(), true);

    const vector<vector<float*>> layer_delta_pairs
        = back_propagation_cuda.get_layer_deltas_device();

    calculate_output_delta_cuda(batch_cuda, forward_propagation_cuda, back_propagation_cuda);

    for (Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
        layers[i]->back_propagate_cuda(layer_input_pairs[i],
                                       layer_delta_pairs[i],
                                       forward_propagation_cuda.layers[i],
                                       back_propagation_cuda.neural_network.layers[i]);
}


void LossIndex::add_regularization_cuda(BackPropagationCuda& back_propagation_cuda) const
{
    if (regularization_method == "None")
    {
        back_propagation_cuda.regularization = 0.0f;
        return;
    }

    NeuralNetwork* neural_network = back_propagation_cuda.loss_index->get_neural_network();

    const Index layers_number = neural_network->get_layers_number();

    type total_regularization_value = 0.0f;

    for (Index layer_index = 0; layer_index < layers_number; ++layer_index)
    {
        Layer* layer = neural_network->get_layer(layer_index).get();

        if (!layer->get_is_trainable())
            continue;

        LayerBackPropagationCuda* layer_back_prop_cuda = back_propagation_cuda.neural_network.layers[layer_index].get();

        const vector<pair<type*, Index>>& parameter_device_pairs = layer->get_parameter_pair_device();
        const vector<pair<type*, Index>>& delta_device_pairs = layer_back_prop_cuda->get_parameter_delta_pair_device();

        for (Index param_index = 0; param_index < parameter_device_pairs.size(); ++param_index)
        {
            type* param_device_ptr = parameter_device_pairs[param_index].first;
            const Index param_size = parameter_device_pairs[param_index].second;

            if (param_size == 0) continue;

            // Regularization

            type current_param_regularization = 0.0f;

            cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);

            if (regularization_method == "L1")
            {
                cublasSasum(cublas_handle, param_size, param_device_ptr, 1, &current_param_regularization);
                total_regularization_value += current_param_regularization;
            }
            else if (regularization_method == "L2")
            {
                cublasSnrm2(cublas_handle, param_size, param_device_ptr, 1, &current_param_regularization);
                total_regularization_value += 0.5f * current_param_regularization * current_param_regularization;
            }
            else if (regularization_method == "ElasticNet")
            {
                const type mix_factor = 0.5;
                type l1_norm = 0.0f;
                type l2_norm = 0.0f;

                cublasSasum(cublas_handle, param_size, param_device_ptr, 1, &l1_norm);
                cublasSnrm2(cublas_handle, param_size, param_device_ptr, 1, &l2_norm);

                current_param_regularization = mix_factor * l1_norm + (1.0f - mix_factor) * 0.5f * (l2_norm * l2_norm);
                total_regularization_value += current_param_regularization;
            }

            // Regularization Gradient

            type* delta_device_ptr = delta_device_pairs[param_index].first;

            if (regularization_method == "L1")
            {
                apply_l1_gradient_cuda(param_size, delta_device_ptr, param_device_ptr, regularization_weight);
            }
            else if (regularization_method == "L2")
            {
                cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
                cublasSaxpy(cublas_handle, param_size, &regularization_weight, param_device_ptr, 1, delta_device_ptr, 1);
                cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
            }
            else if (regularization_method == "ElasticNet")
            {
                const type mix_factor = 0.5;
                apply_elastic_net_gradient_cuda(param_size, delta_device_ptr, param_device_ptr, regularization_weight, mix_factor);
            }
        }
    }

    back_propagation_cuda.regularization = total_regularization_value;
    back_propagation_cuda.loss += regularization_weight * total_regularization_value;
}


void LossIndex::create_cuda()
{
    cublasCreate(&cublas_handle);
    cudnnCreate(&cudnn_handle);
}


void LossIndex::destroy_cuda()
{
    cublasDestroy(cublas_handle);
    cudnnDestroy(cudnn_handle);
}

cudnnHandle_t LossIndex::get_cudnn_handle()
{
    return cudnn_handle;
}


// CUDA structs

BackPropagationCuda::BackPropagationCuda(const Index& new_samples_number, LossIndex* new_loss_index)
{
    set(new_samples_number, new_loss_index);
}


void BackPropagationCuda::set(const Index& new_samples_number, LossIndex* new_loss_index)
{
    samples_number = new_samples_number;

    loss_index = new_loss_index;

    if (!loss_index) return;

    // Neural network

    NeuralNetwork* neural_network_ptr = loss_index->get_neural_network();

    const Index parameters_number = neural_network_ptr->get_parameters_number();

    const dimensions output_dimensions = neural_network_ptr->get_output_dimensions();

    const Index outputs_number = output_dimensions[0];

    // First order loss

    neural_network.set(samples_number, neural_network_ptr);

    cout << "BackPropagationCuda set:" << endl;

    loss = type(0);
    error(0) = type(0);
    regularization = type(0);

    //CHECK_CUDA(cudaMalloc(&errors, samples_number * outputs_number * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(errors, samples_number * outputs_number * sizeof(float));
    //CHECK_CUDA(cudaMalloc(&error_device, sizeof(float)));
    CUDA_MALLOC_AND_REPORT(error_device, sizeof(float));

    // Outputs_delta

    output_deltas_dimensions = { samples_number };
    output_deltas_dimensions.insert(output_deltas_dimensions.end(), output_dimensions.begin(), output_dimensions.end());

    const Index size = accumulate(output_dimensions.begin(), output_dimensions.end(), samples_number, multiplies<>());

    //CHECK_CUDA(cudaMalloc(&output_deltas, size * sizeof(float)));
    CUDA_MALLOC_AND_REPORT(output_deltas, size * sizeof(float));

    // Reduce

    cudnnCreateReduceTensorDescriptor(&reduce_tensor_descriptor);

    cudnnSetReduceTensorDescriptor(reduce_tensor_descriptor,
                                   CUDNN_REDUCE_TENSOR_ADD,
                                   CUDNN_DATA_FLOAT,
                                   CUDNN_PROPAGATE_NAN,
                                   CUDNN_REDUCE_TENSOR_NO_INDICES,
                                   CUDNN_32BIT_INDICES);

    cudnnCreateTensorDescriptor(&output_tensor_descriptor);

    cudnnSetTensor4dDescriptor(output_tensor_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               samples_number,
                               outputs_number,
                               1,
                               1);

    cudnnCreateTensorDescriptor(&output_reduce_tensor_descriptor);

    cudnnSetTensor4dDescriptor(output_reduce_tensor_descriptor,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               1,
                               1,
                               1,
                               1);

    cudnnGetReductionWorkspaceSize(loss_index->get_cudnn_handle(), reduce_tensor_descriptor, output_tensor_descriptor, output_reduce_tensor_descriptor, &workspaceSize);

    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    // Sum

    cudnnCreateOpTensorDescriptor(&operator_sum_descriptor);

    cudnnSetOpTensorDescriptor(operator_sum_descriptor,
                               CUDNN_OP_TENSOR_ADD,
                               CUDNN_DATA_FLOAT,
                               CUDNN_NOT_PROPAGATE_NAN);

    //if (is_instance_of<CrossEntropyError3d>(loss_index))
    //{
    /* @todo CudaMalloc transformers GPU
        predictions (batch_size, outputs_number);
        matches (batch_size, outputs_number);
        mask (batch_size, outputs_number);
        */
    //}
}


vector<vector<float*>> BackPropagationCuda::get_layer_deltas_device() const
{
    NeuralNetwork* neural_network_ptr = loss_index->get_neural_network();

    const Index layers_number = neural_network_ptr->get_layers_number();

    const vector<vector<Index>>& layer_input_indices = neural_network_ptr->get_layer_input_indices();
    const vector<vector<Index>> layer_output_indices = neural_network_ptr->get_layer_output_indices();
    const vector<unique_ptr<LayerBackPropagationCuda>>& layer_back_propagations = neural_network.get_layers();

    vector<float*> input_deltas;

    vector<vector<float*>> layer_deltas(layers_number);

    const Index first_trainable_layer_index = neural_network_ptr->get_first_trainable_layer_index();
    const Index last_trainable_layer_index = neural_network_ptr->get_last_trainable_layer_index();

    for (Index i = last_trainable_layer_index; i >= first_trainable_layer_index; i--)
    {
        if (i == last_trainable_layer_index)
        {
            layer_deltas[i].push_back(get_output_deltas_device());

            continue;
        }

        for (Index j = 0; j < Index(layer_output_indices[i].size()); j++)
        {
            const Index output_index = layer_output_indices[i][j];
            const Index input_index = neural_network_ptr->find_input_index(layer_input_indices[output_index], i);

            input_deltas = layer_back_propagations[output_index]->get_input_derivatives_device();

            layer_deltas[i].push_back(input_deltas[input_index]);
        }
    }
    return layer_deltas;
}


float* BackPropagationCuda::get_output_deltas_device() const
{
    return output_deltas;
}


void BackPropagationCuda::print()
{

}


void BackPropagationCuda::free()
{
    cudaFree(error_device);
    cudaFree(errors);
    cudaFree(output_deltas);
    cudaFree(workspace);
    //cudaFree(predictions);
    //cudaFree(matches);
    //cudaFree(mask);

    cudnnDestroyReduceTensorDescriptor(reduce_tensor_descriptor);
    cudnnDestroyOpTensorDescriptor(operator_sum_descriptor);
    cudnnDestroyTensorDescriptor(output_tensor_descriptor);
    cudnnDestroyTensorDescriptor(output_reduce_tensor_descriptor);
}

#endif

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
