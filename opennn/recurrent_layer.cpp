//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <sstream>

#include "strings_utilities.h"
#include "tensors.h"
#include "recurrent_layer.h"

namespace opennn
{

RecurrentLayer::RecurrentLayer() : Layer()
{
    set();

    layer_type = Type::Recurrent;
}


RecurrentLayer::RecurrentLayer(const Index& new_inputs_number, const Index& new_neurons_number, const Index& new_timesteps) : Layer()
{
    set(new_inputs_number, new_neurons_number, new_timesteps);

    layer_type = Type::Recurrent;
}


Index RecurrentLayer::get_inputs_number() const
{
    return input_weights.dimension(0);
}


Index RecurrentLayer::get_neurons_number() const
{
    return biases.size();
}


dimensions RecurrentLayer::get_output_dimensions() const
{
    Index neurons_number = get_neurons_number();

    return { neurons_number };
}


const Tensor<type, 3>& RecurrentLayer::get_hidden_states() const
{
    return hidden_states;
}


Index RecurrentLayer::get_parameters_number() const
{
    const Index neurons_number = get_neurons_number();
    const Index inputs_number = get_inputs_number();

    return  neurons_number * (1 + inputs_number + neurons_number);
}


Index RecurrentLayer::get_timesteps() const
{
    return time_steps;
}


Tensor<type, 1> RecurrentLayer::get_biases() const
{
    return biases;
}


const Tensor<type, 2>& RecurrentLayer::get_input_weights() const
{
    return input_weights;
}


const Tensor<type, 2>& RecurrentLayer::get_recurrent_weights() const
{
    return recurrent_weights;
}


Index RecurrentLayer::get_biases_number() const
{
    return biases.size();
}


Index RecurrentLayer::get_input_weights_number() const
{
    return input_weights.size();
}


Index RecurrentLayer::get_recurrent_weights_number() const
{
    return recurrent_weights.size();
}


Tensor<type, 1> RecurrentLayer::get_parameters() const
{
    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> parameters(parameters_number);

    #pragma omp parallel sections
    {
        #pragma omp section
        memcpy(parameters.data(),
            biases.data(),
            biases.size() * sizeof(type));

        #pragma omp section
        memcpy(parameters.data() + biases.size(),
            input_weights.data(),
            input_weights.size() * sizeof(type));

        #pragma omp section
        memcpy(parameters.data() + biases.size() + input_weights.size(),
            recurrent_weights.data(),
            recurrent_weights.size() * sizeof(type));
    }

    return parameters;
}


const RecurrentLayer::ActivationFunction& RecurrentLayer::get_activation_function() const
{
    return activation_function;
}


Tensor<type, 2> RecurrentLayer::get_biases(const Tensor<type, 1>& parameters) const
{
    const Index biases_number = get_biases_number();
    const Index input_weights_number = get_input_weights_number();

    Tensor<type, 1> new_biases(biases_number);

    new_biases = parameters.slice(Eigen::array<Index, 1>({input_weights_number}), Eigen::array<Index, 1>({biases_number}));

    Eigen::array<Index, 2> two_dim{{1, biases.dimension(1)}};

    return new_biases.reshape(two_dim);
}


Tensor<type, 2> RecurrentLayer::get_input_weights(const Tensor<type, 1>& parameters) const
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();
    const Index input_weights_number = get_input_weights_number();

    const Tensor<type, 1> new_inputs_weights
            = parameters.slice(Eigen::array<Index, 1>({0}), Eigen::array<Index, 1>({input_weights_number}));

    const Eigen::array<Index, 2> two_dim{{inputs_number, neurons_number}};

    return new_inputs_weights.reshape(two_dim);
}


Tensor<type, 2> RecurrentLayer::get_recurrent_weights(const Tensor<type, 1>& parameters) const
{
    const Index neurons_number = get_neurons_number();
    const Index recurrent_weights_number = recurrent_weights.size();

    const Index parameters_size = parameters.size();

    const Index start_recurrent_weights_number = (parameters_size - recurrent_weights_number);

    const Tensor<type, 1> new_synaptic_weights
            = parameters.slice(Eigen::array<Index, 1>({start_recurrent_weights_number}), Eigen::array<Index, 1>({recurrent_weights_number}));

    const Eigen::array<Index, 2> two_dim{{neurons_number, neurons_number}};

    return new_synaptic_weights.reshape(two_dim);
}


string RecurrentLayer::write_activation_function() const
{
    switch(activation_function)
    {
    case ActivationFunction::Logistic: return "Logistic";

    case ActivationFunction::HyperbolicTangent: return "HyperbolicTangent";

    case ActivationFunction::Linear: return "Linear";

    case ActivationFunction::RectifiedLinear: return "RectifiedLinear";

    case ActivationFunction::ScaledExponentialLinear: return "ScaledExponentialLinear";

    case ActivationFunction::SoftPlus: return "SoftPlus";

    case ActivationFunction::SoftSign: return "SoftSign";

    case ActivationFunction::HardSigmoid: return "HardSigmoid";

    case ActivationFunction::ExponentialLinear: return "ExponentialLinear";

    default:
        return string();
    }
}


const bool& RecurrentLayer::get_display() const
{
    return display;
}


void RecurrentLayer::set()
{
    set_default();
}


void RecurrentLayer::set(const Index& new_inputs_number, const Index& new_neurons_number, const Index& new_timesteps)
{
    biases.resize(new_neurons_number);

    input_weights.resize(new_inputs_number, new_neurons_number);

    recurrent_weights.resize(new_neurons_number, new_neurons_number);

    Index batch_size = 10;

    hidden_states.resize(batch_size, time_steps, new_neurons_number); 

    hidden_states.setConstant(type(0));

    time_steps = new_timesteps;

    set_parameters_random();

    set_default();
}


void RecurrentLayer::set(const RecurrentLayer& other_neuron_layer)
{
    activation_function = other_neuron_layer.activation_function;

    display = other_neuron_layer.display;

    set_default();
}


void RecurrentLayer::set_default()
{
    name = "recurrent_layer";

    display = true;

    layer_type = Type::Recurrent;
}


void RecurrentLayer::set_inputs_number(const Index& new_inputs_number)
{
    const Index neurons_number = get_neurons_number();

    input_weights.resize(new_inputs_number, neurons_number);
}


void RecurrentLayer::set_input_shape(const Tensor<Index, 1>& size)
{
    const Index new_size = size[0];

    set_inputs_number(new_size);
}


void RecurrentLayer::set_neurons_number(const Index& new_neurons_number)
{
    const Index inputs_number = get_inputs_number();

    biases.resize(new_neurons_number);

    input_weights.resize(inputs_number, new_neurons_number);

    recurrent_weights.resize(new_neurons_number, new_neurons_number);
}


void RecurrentLayer::set_timesteps(const Index& new_timesteps)
{
    time_steps = new_timesteps;
}


void RecurrentLayer::set_biases(const Tensor<type, 1>& new_biases)
{
    biases = new_biases;
}


void RecurrentLayer::set_input_weights(const Tensor<type, 2>& new_input_weights)
{
    input_weights = new_input_weights;
}


void RecurrentLayer::set_recurrent_weights(const Tensor<type, 2>& new_recurrent_weights)
{
    recurrent_weights = new_recurrent_weights;
}


void RecurrentLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    const Index biases_number = get_biases_number();
    const Index inputs_weights_number = get_input_weights_number();
    const Index recurrent_weights_number = get_recurrent_weights_number();

    #pragma omp parallel sections
    {
        #pragma omp section
            memcpy(biases.data(), new_parameters.data() + index, biases_number * sizeof(type));

        #pragma omp section
            memcpy(input_weights.data(), new_parameters.data() + index + biases_number, inputs_weights_number * sizeof(type));

        #pragma omp section
            memcpy(recurrent_weights.data(), new_parameters.data() + index + biases_number + inputs_weights_number, recurrent_weights_number * sizeof(type));
    }
}


void RecurrentLayer::set_activation_function(const RecurrentLayer::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


void RecurrentLayer::set_activation_function(const string& new_activation_function_name)
{
    if(new_activation_function_name == "Logistic")
    {
        activation_function = ActivationFunction::Logistic;
    }
    else if(new_activation_function_name == "HyperbolicTangent")
    {
        activation_function = ActivationFunction::HyperbolicTangent;
    }
    else if(new_activation_function_name == "Linear")
    {
        activation_function = ActivationFunction::Linear;
    }
    else if(new_activation_function_name == "RectifiedLinear")
    {
        activation_function = ActivationFunction::RectifiedLinear;
    }
    else if(new_activation_function_name == "ScaledExponentialLinear")
    {
        activation_function = ActivationFunction::ScaledExponentialLinear;
    }
    else if(new_activation_function_name == "SoftPlus")
    {
        activation_function = ActivationFunction::SoftPlus;
    }
    else if(new_activation_function_name == "SoftSign")
    {
        activation_function = ActivationFunction::SoftSign;
    }
    else if(new_activation_function_name == "HardSigmoid")
    {
        activation_function = ActivationFunction::HardSigmoid;
    }
    else if(new_activation_function_name == "ExponentialLinear")
    {
        activation_function = ActivationFunction::ExponentialLinear;
    }
    else
    {
        throw runtime_error("Unknown activation function: " + new_activation_function_name + ".\n");
    }
}


void RecurrentLayer::set_display(const bool& new_display)
{
    display = new_display;
}


void RecurrentLayer::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    input_weights.setConstant(value);

    recurrent_weights.setConstant(value);

    hidden_states.setZero();
}


void RecurrentLayer::set_parameters_random()
{
    set_random(biases);

    set_random(input_weights);

    set_random(recurrent_weights);
}


void RecurrentLayer::calculate_combinations(const Tensor<type, 2>& inputs,
                                            Tensor<type, 2>& combinations) const
{   
//    combinations.device(*thread_pool_device) = biases
//                                             + inputs.contract(input_weights, AT_B)
//                                             + hidden_states.contract(recurrent_weights, AT_B);

        // Compute the new hidden state: h_t = tanh(W_x * x_t + W_h * h_t + b)
//        current_hidden_states = (input_weights.contract(current_inputs), Eigen::array<Eigen::IndexPair<int>, 1>{{Eigen::IndexPair<int>(1, 1)}})
//            + recurrent_weights.contract(hidden_state, Eigen::array<Eigen::IndexPair<int>, 1>{{Eigen::IndexPair<int>(1, 1)}})
//            + biases.broadcast(Eigen::array<int, 2>{batch_size, 1}));
}


void RecurrentLayer::calculate_activations(Tensor<type, 2>& activations,
                                           Tensor<type, 2>& activations_derivatives) const
{
    switch(activation_function)
    {
        case ActivationFunction::Linear: linear(activations, activations_derivatives); return;

        case ActivationFunction::Logistic: logistic(activations, activations_derivatives); return;

        case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(activations, activations_derivatives); return;

        case ActivationFunction::RectifiedLinear: rectified_linear(activations, activations_derivatives); return;

        case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(activations, activations_derivatives); return;

        case ActivationFunction::SoftPlus: soft_plus(activations, activations_derivatives); return;

        case ActivationFunction::SoftSign: soft_sign(activations, activations_derivatives); return;

        case ActivationFunction::HardSigmoid: hard_sigmoid(activations, activations_derivatives); return;

        case ActivationFunction::ExponentialLinear: exponential_linear(activations, activations_derivatives); return;

        default: throw runtime_error("Unknown activation function");
    }
}


void RecurrentLayer::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                       LayerForwardPropagation* forward_propagation,
                                       const bool& is_training)
{
    const Index batch_size = inputs_pair(0).second[0];
    const Index time_steps = inputs_pair(0).second[1];
    const Index inputs_number = inputs_pair(0).second[2];

    RecurrentLayerForwardPropagation* recurrent_layer_forward_propagation
        = static_cast<RecurrentLayerForwardPropagation*>(forward_propagation);

    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(inputs_pair(0));

    Tensor<type, 2>& current_inputs = recurrent_layer_forward_propagation->current_inputs;

    Tensor<type, 3>& activations_derivatives = recurrent_layer_forward_propagation->activations_derivatives;

    Tensor<type, 2>& current_activations_derivatives = recurrent_layer_forward_propagation->current_activations_derivatives;

    const Index neurons_number = get_neurons_number();

    Tensor<type, 2> current_hidden_states(batch_size, neurons_number);
    current_hidden_states.setZero();

    for (Index time_step = 0; time_step < time_steps; time_step++) 
    {
        current_inputs = inputs.chip(time_step, 1);  

        calculate_combinations(current_inputs, current_hidden_states);

        if (is_training)
        {
            calculate_activations(current_hidden_states, empty);
        }
        else
        {
            calculate_activations(current_hidden_states, current_activations_derivatives);

            activations_derivatives.chip(time_step, 1) = current_activations_derivatives;
        }
        
        hidden_states.chip(time_step, 1) = current_hidden_states;
    }
}


void RecurrentLayer::back_propagate(const vector<pair<type*, dimensions>>& inputs_pair,
                                    const vector<pair<type*, dimensions>>& deltas_pair,
                                    LayerForwardPropagation* forward_propagation,
                                    LayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs_pair[0].second[0];
    const Index neurons_number = get_neurons_number();
    const Index inputs_number = get_inputs_number();

    RecurrentLayerForwardPropagation* recurrent_layer_forward_propagation =
            static_cast<RecurrentLayerForwardPropagation*>(forward_propagation);

    RecurrentLayerBackPropagation* recurrent_layer_back_propagation =
            static_cast<RecurrentLayerBackPropagation*>(back_propagation);

    // Forward propagation

    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(inputs_pair[0]);

    Tensor<type, 2>& current_inputs = recurrent_layer_forward_propagation->current_inputs;

    const Tensor<type, 3>& activations_derivatives = recurrent_layer_forward_propagation->activations_derivatives;

    Tensor<type, 2>& current_activations_derivatives = recurrent_layer_forward_propagation->current_activations_derivatives;

    // Back propagation

    const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(deltas_pair[0]);

    const bool& is_first_layer = recurrent_layer_back_propagation->is_first_layer;

    //Tensor<type, 1>& current_deltas = recurrent_layer_back_propagation->current_deltas;

    Tensor<type, 2>& combinations_derivatives = recurrent_layer_back_propagation->combinations_derivatives;
    Tensor<type, 1>& current_combinations_derivatives = recurrent_layer_back_propagation->current_combinations_derivatives;

    Tensor<type, 2>& combinations_biases_derivatives = recurrent_layer_back_propagation->combinations_biases_derivatives;
    combinations_biases_derivatives.setZero();

    Tensor<type, 3>& combinations_input_weights_derivatives = recurrent_layer_back_propagation->combinations_input_weights_derivatives;
    combinations_input_weights_derivatives.setZero();

    Tensor<type, 3>& combinations_recurrent_weights_derivatives = recurrent_layer_back_propagation->combinations_recurrent_weights_derivatives;
    combinations_recurrent_weights_derivatives.setZero();

    Tensor<type, 1>& biases_derivatives = recurrent_layer_back_propagation->biases_derivatives;
    biases_derivatives.setZero();

    Tensor<type, 2>& input_weights_derivatives = recurrent_layer_back_propagation->input_weights_derivatives;
    input_weights_derivatives.setZero();

    Tensor<type, 2>& recurrent_weights_derivatives = recurrent_layer_back_propagation->recurrent_weights_derivatives;
    recurrent_weights_derivatives.setZero();

    Tensor<type, 3>& input_derivatives = recurrent_layer_back_propagation->input_derivatives;

    const Eigen::array<IndexPair<Index>, 1> combinations_weights_indices = { IndexPair<Index>(2, 0) };

    for (Index time_step = 0; time_step < time_steps; time_step++)
        current_inputs = inputs.chip(time_step, 1);

/*
    for(Index sample_index = 0; sample_index < samples_number; sample_index++)
    {
        current_inputs.device(*thread_pool_device) = inputs.chip(sample_index, 0);

        current_deltas.device(*thread_pool_device) = deltas.chip(sample_index, 0);

        get_row(current_activations_derivatives, activations_derivatives, sample_index);

        if(sample_index % time_steps == 0)
        {
            combinations_biases_derivatives.setZero();
            combinations_input_weights_derivatives.setZero();
            combinations_recurrent_weights_derivatives.setZero();
        }
        else
        {
            // Combinations biases derivatives

            multiply_rows(combinations_biases_derivatives, current_activations_derivatives);

            combinations_biases_derivatives.device(*thread_pool_device) = combinations_biases_derivatives.contract(recurrent_weights, A_B);

            // Combinations weights derivatives

            multiply_matrices(thread_pool_device, combinations_input_weights_derivatives, current_activations_derivatives);

            combinations_input_weights_derivatives.device(*thread_pool_device) 
                = combinations_input_weights_derivatives.contract(recurrent_weights, combinations_weights_indices);

            // Combinations recurrent weights derivatives

            multiply_matrices(thread_pool_device, combinations_recurrent_weights_derivatives, current_activations_derivatives);

            combinations_recurrent_weights_derivatives.device(*thread_pool_device) 
                = combinations_recurrent_weights_derivatives.contract(recurrent_weights, combinations_weights_indices);
        }

        current_combinations_derivatives.device(*thread_pool_device) = current_deltas * current_activations_derivatives;

        combinations_derivatives.chip(sample_index, 0).device(*thread_pool_device) = current_combinations_derivatives;

        sum_diagonal(combinations_biases_derivatives, type(1));

        // Biases derivatives

        biases_derivatives.device(*thread_pool_device)
            += combinations_biases_derivatives.contract(current_combinations_derivatives, A_B);

//        combinations_input_weights_derivatives += current_inputs
//            .reshape(Eigen::array<Index, 2>({ inputs_number, 1 }))
//            .broadcast(Eigen::array<Index, 3>({ 1, neurons_number, 1 }));

//        for(Index neuron_index = 0; neuron_index < neurons_number; neuron_index++)
//            for(Index input_index = 0; input_index < inputs_number; input_index++)
//                combinations_input_weights_derivatives(input_index, neuron_index, neuron_index) += current_inputs(input_index);

        if(sample_index % time_steps != 0)
        {
            // @todo parallelize

            for(Index neuron_index = 0; neuron_index < neurons_number; neuron_index++)
                for(Index activation_index = 0; activation_index < neurons_number; activation_index++)
                    combinations_recurrent_weights_derivatives(activation_index, neuron_index, neuron_index)
                        += outputs(sample_index - 1, activation_index);

//            combinations_recurrent_weights_derivatives += outputs.chip(sample_index - 1, 0)
//                .reshape(Eigen::array<Index, 2>({ neurons_number, 1 }))
//                .broadcast(Eigen::array<Index, 3>({ 1, neurons_number, 1 }));
        }

        // Weights derivatives

        input_weights_derivatives.device(*thread_pool_device)
            += combinations_input_weights_derivatives.contract(current_combinations_derivatives, combinations_weights_indices);

        recurrent_weights_derivatives.device(*thread_pool_device)
            += combinations_recurrent_weights_derivatives.contract(current_combinations_derivatives, combinations_weights_indices);
    }
*/
    // Input derivatives

    if(!is_first_layer)
        input_derivatives.device(*thread_pool_device) = combinations_derivatives.contract(input_weights, A_BT);
}


void RecurrentLayer::insert_gradient(LayerBackPropagation* back_propagation,
                                     const Index& index,
                                     Tensor<type, 1>& gradient) const
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    type* gradient_data = gradient.data();

    RecurrentLayerBackPropagation* recurrent_layer_back_propagation =
        static_cast<RecurrentLayerBackPropagation*>(back_propagation);

    #pragma omp parallel sections
    {
        #pragma omp section
            memcpy(gradient_data + index,
                recurrent_layer_back_propagation->biases_derivatives.data(),
                neurons_number * sizeof(type));

        #pragma omp section
            memcpy(gradient_data + index + neurons_number,
                recurrent_layer_back_propagation->input_weights_derivatives.data(),
                inputs_number * neurons_number * sizeof(type));

        #pragma omp section
            memcpy(gradient_data + index + neurons_number + inputs_number * neurons_number,
                recurrent_layer_back_propagation->recurrent_weights_derivatives.data(),
                neurons_number * neurons_number * sizeof(type));
    }
}


string RecurrentLayer::write_expression(const Tensor<string, 1>& inputs_name,
                                        const Tensor<string, 1>& output_names) const
{
    ostringstream buffer;

    for(Index j = 0; j < output_names.size(); j++)
    {
        const Tensor<type, 1> synaptic_weights_column =  recurrent_weights.chip(j,1);

        buffer << output_names(j) << " = " << write_activation_function_expression() << "( " << biases(j) << " +";

        for(Index i = 0; i < inputs_name.size() - 1; i++)
        {
           buffer << " (" << inputs_name[i] << "*" << synaptic_weights_column(i) << ") +";
        }

        buffer << " (" << inputs_name[inputs_name.size() - 1] << "*" << synaptic_weights_column[inputs_name.size() - 1] << "));\n";
    }

    return buffer.str();
}


string RecurrentLayer::write_activation_function_expression() const
{
    switch(activation_function)
    {
        case ActivationFunction::HyperbolicTangent: return "tanh";

        case ActivationFunction::Linear: return string();

        default: return write_activation_function();
    }
}


void RecurrentLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Perceptron layer

    const tinyxml2::XMLElement* perceptron_layer_element = document.FirstChildElement("RecurrentLayer");

    if(!perceptron_layer_element)
        throw runtime_error("RecurrentLayer element is nullptr.\n");

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = perceptron_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
        throw runtime_error("InputsNumber element is nullptr.\n");

    if(inputs_number_element->GetText())
        set_inputs_number(Index(stoi(inputs_number_element->GetText())));

    // Neurons number

    const tinyxml2::XMLElement* neurons_number_element = perceptron_layer_element->FirstChildElement("NeuronsNumber");

    if(!neurons_number_element)
        throw runtime_error("NeuronsNumber element is nullptr.\n");

    if(neurons_number_element->GetText())
        set_neurons_number(Index(stoi(neurons_number_element->GetText())));

    // Activation function

    const tinyxml2::XMLElement* activation_function_element = perceptron_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
        throw runtime_error("ActivationFunction element is nullptr.\n");

    if(activation_function_element->GetText())
        set_activation_function(activation_function_element->GetText());

    // Parameters

    const tinyxml2::XMLElement* parameters_element = perceptron_layer_element->FirstChildElement("Parameters");

    if(!parameters_element)
        throw runtime_error("Parameters element is nullptr.\n");

    if(parameters_element->GetText())
        set_parameters(to_type_vector(parameters_element->GetText(), " "));
}


void RecurrentLayer::to_XML(tinyxml2::XMLPrinter& file_stream) const

{
    ostringstream buffer;

    // Perceptron layer

    file_stream.OpenElement("RecurrentLayer");

    // Inputs number

    file_stream.OpenElement("InputsNumber");
    file_stream.PushText(to_string(get_inputs_number()).c_str());
    file_stream.CloseElement();

    // Outputs number

    file_stream.OpenElement("NeuronsNumber");
    file_stream.PushText(to_string(get_neurons_number()).c_str());
    file_stream.CloseElement();

    // Activation function

    file_stream.OpenElement("ActivationFunction");
    file_stream.PushText(write_activation_function().c_str());
    file_stream.CloseElement();

    // Parameters

    file_stream.OpenElement("Parameters");
    file_stream.PushText(tensor_to_string(get_parameters()).c_str());
    file_stream.CloseElement();

    // Recurrent layer (end tag)

    file_stream.CloseElement();
}


pair<type*, dimensions> RecurrentLayerForwardPropagation::get_outputs_pair() const
{
    const Index neurons_number = layer->get_neurons_number();

    return pair<type*, dimensions>(outputs_data, { {batch_samples_number, neurons_number} });
}


void RecurrentLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    const Index neurons_number = layer->get_neurons_number();
    const Index inputs_number = layer->get_inputs_number();
    const Index time_steps = 0;

    batch_samples_number = new_batch_samples_number;

//    outputs_data = outputs.data();

    current_inputs.resize(batch_samples_number, inputs_number);
    current_activations_derivatives.resize(batch_samples_number, neurons_number);

    activations_derivatives.resize(batch_samples_number, time_steps, neurons_number);
}


void RecurrentLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = layer->get_neurons_number();
    const Index inputs_number = layer->get_inputs_number();

    //current_deltas.resize(neurons_number);

    combinations_biases_derivatives.resize(neurons_number, neurons_number);

    combinations_input_weights_derivatives.resize(inputs_number, neurons_number, neurons_number);

    combinations_recurrent_weights_derivatives.resize(neurons_number, neurons_number, neurons_number);

    combinations_derivatives.resize(batch_samples_number, neurons_number);
    current_combinations_derivatives.resize(neurons_number);

    biases_derivatives.resize(neurons_number);

    input_weights_derivatives.resize(inputs_number, neurons_number);

    recurrent_weights_derivatives.resize(neurons_number, neurons_number);

    const Index time_steps = 0;

    input_derivatives.resize(batch_samples_number, time_steps, inputs_number);

    inputs_derivatives = {{input_derivatives.data(), {batch_samples_number, inputs_number}}};
}

}

// OpenNN: Open Neural Networks Library.
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
