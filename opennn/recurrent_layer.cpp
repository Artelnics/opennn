//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "recurrent_layer.h"

namespace opennn
{

Recurrent::Recurrent(const dimensions& new_input_dimensions, 
                               const dimensions& new_output_dimensions) : Layer()
{
    set(new_input_dimensions, new_output_dimensions);
}


dimensions Recurrent::get_input_dimensions() const
{
    return { input_weights.dimension(0) };
}


dimensions Recurrent::get_output_dimensions() const
{
    return { biases.size() };
}


Index Recurrent::get_parameters_number() const
{
    return biases.size() + input_weights.size() + recurrent_weights.size();
}


Index Recurrent::get_timesteps() const
{
    return time_steps;
}


Tensor<type, 1> Recurrent::get_parameters() const
{
    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> parameters(parameters_number);

    Index index = 0;

    copy_to_vector(parameters, biases, index);
    copy_to_vector(parameters, input_weights, index);
    copy_to_vector(parameters, recurrent_weights, index);

    return parameters;
}


const Recurrent::Activation& Recurrent::get_activation_function() const
{
    return activation_function;
}


string Recurrent::get_activation_function_string() const
{
    switch(activation_function)
    {
    case Activation::Logistic: return "Logistic";

    case Activation::HyperbolicTangent: return "HyperbolicTangent";

    case Activation::Linear: return "Linear";

    case Activation::RectifiedLinear: return "RectifiedLinear";

    case Activation::ScaledExponentialLinear: return "ScaledExponentialLinear";

    case Activation::SoftPlus: return "SoftPlus";

    case Activation::SoftSign: return "SoftSign";

    case Activation::HardSigmoid: return "HardSigmoid";

    case Activation::ExponentialLinear: return "ExponentialLinear";

    default:
        return string();
    }
}


void Recurrent::set(const dimensions& new_input_dimensions, const dimensions& new_output_dimensions)
{

    biases.resize(new_output_dimensions[0]);

    input_weights.resize(new_input_dimensions[0], new_output_dimensions[0]);

    recurrent_weights.resize(new_output_dimensions[0], new_output_dimensions[0]);

    output_biases.resize(new_output_dimensions[0]);

    output_weights.resize(new_output_dimensions[0], new_output_dimensions[0]);

    Index samples_number = 10;

    hidden_states.resize(samples_number, new_output_dimensions[0]);

    hidden_states.setConstant(type(0));

    time_steps = get_timesteps();

    //set_parameters_random();

    name = "recurrent_layer";

    layer_type = Type::Recurrent;
/*
    biases.resize(new_neurons_number);

    input_weights.resize(new_inputs_number, new_neurons_number);

    recurrent_weights.resize(new_neurons_number, new_neurons_number);

    Index samples_number = 10;

    hidden_states.resize(samples_number, time_steps, new_neurons_number);

    hidden_states.setConstant(type(0));

    time_steps = new_timesteps;

    set_parameters_random();

    name = "recurrent_layer";

    layer_type = Type::Recurrent;
*/
}


void Recurrent::set_input_dimensions(const dimensions& new_input_dimensions)
{
    const Index outputs_number = get_outputs_number();

    input_weights.resize(new_input_dimensions[0], outputs_number);
}


void Recurrent::set_output_dimensions(const dimensions& new_output_dimensions)
{
    const Index inputs_number = get_inputs_number();

    biases.resize(new_output_dimensions[0]);

    input_weights.resize(inputs_number, new_output_dimensions[0]);

    recurrent_weights.resize(new_output_dimensions[0], new_output_dimensions[0]);

    output_biases.resize(new_output_dimensions[0]);

    output_weights.resize(new_output_dimensions[0], new_output_dimensions[0]);
}


void Recurrent::set_timesteps(const Index& new_timesteps)
{
    time_steps = new_timesteps;
}


void Recurrent::set_parameters(const Tensor<type, 1>& new_parameters, Index& index)
{
    copy_from_vector(biases, new_parameters, index);
    copy_from_vector(input_weights, new_parameters, index);
    copy_from_vector(recurrent_weights, new_parameters, index);
    copy_from_vector(output_biases, new_parameters, index);
    copy_from_vector(output_weights, new_parameters, index);
}


void Recurrent::set_activation_function(const Recurrent::Activation& new_activation_function)
{
    activation_function = new_activation_function;
}


void Recurrent::set_activation_function(const string& new_activation_function_name)
{
    if(new_activation_function_name == "Logistic")
        activation_function = Activation::Logistic;
    else if(new_activation_function_name == "HyperbolicTangent")
        activation_function = Activation::HyperbolicTangent;
    else if(new_activation_function_name == "Linear")
        activation_function = Activation::Linear;
    else if(new_activation_function_name == "RectifiedLinear")
        activation_function = Activation::RectifiedLinear;
    else if(new_activation_function_name == "ScaledExponentialLinear")
        activation_function = Activation::ScaledExponentialLinear;
    else if(new_activation_function_name == "SoftPlus")
        activation_function = Activation::SoftPlus;
    else if(new_activation_function_name == "SoftSign")
        activation_function = Activation::SoftSign;
    else if(new_activation_function_name == "HardSigmoid")
        activation_function = Activation::HardSigmoid;
    else if(new_activation_function_name == "ExponentialLinear")
        activation_function = Activation::ExponentialLinear;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function_name + ".\n");
}


void Recurrent::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    input_weights.setConstant(value);

    recurrent_weights.setConstant(value);

    hidden_states.setZero();

    output_weights.setConstant(value);

    output_biases.setConstant(value);
}


void Recurrent::set_parameters_random()
{
    set_random(biases);

    set_random(input_weights);

    set_random(recurrent_weights);

    set_random(output_weights);

    set_random(output_biases);
}

void Recurrent::calculate_combinations(const Tensor<type, 2>& inputs,
                                            Tensor<type, 2>& combinations) const
{
    Index samples_number = inputs.dimension(0);

    combinations.device(*thread_pool_device) = biases
                                             + inputs.contract(input_weights, AT_B)
                                             + hidden_states.contract(recurrent_weights, AT_B);

    // Compute the new hidden state: h_t = tanh(W_x * x_t + W_h * h_t + b)
    combinations = (input_weights.contract(inputs, Eigen::array<Eigen::IndexPair<Index>, 1>{{Eigen::IndexPair<Index>(1, 1)}})
        + recurrent_weights.contract(hidden_states, Eigen::array<Eigen::IndexPair<Index>, 1>{{Eigen::IndexPair<Index>(1, 1)}})
        + biases.broadcast(Eigen::array<Index, 2>{samples_number, 1}));

}


void Recurrent::calculate_activations(Tensor<type, 2>& activations,
                                           Tensor<type, 2>& activation_derivatives) const
{
    switch(activation_function)
    {
        case Activation::Linear: linear(activations, activation_derivatives); return;

        case Activation::Logistic: logistic(activations, activation_derivatives); return;

        case Activation::HyperbolicTangent: hyperbolic_tangent(activations, activation_derivatives); return;

        case Activation::RectifiedLinear: rectified_linear(activations, activation_derivatives); return;

        case Activation::ScaledExponentialLinear: scaled_exponential_linear(activations, activation_derivatives); return;

        case Activation::SoftPlus: soft_plus(activations, activation_derivatives); return;

        case Activation::SoftSign: soft_sign(activations, activation_derivatives); return;

        case Activation::HardSigmoid: hard_sigmoid(activations, activation_derivatives); return;

        case Activation::ExponentialLinear: exponential_linear(activations, activation_derivatives); return;

        default: throw runtime_error("Unknown activation function");
    }
}

void Recurrent::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                       unique_ptr<LayerForwardPropagation>& forward_propagation,
                                       const bool& is_training)

{
    const Index batch_size = input_pairs[0].second[0];
    const Index time_steps = input_pairs[0].second[1];
    const Index input_size = input_pairs[0].second[2];
    const Index output_size = get_outputs_number();

    TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    RecurrentLayerForwardPropagation* recurrent_layer_forward_propagation =
        static_cast<RecurrentLayerForwardPropagation*>(forward_propagation.get());

    Tensor<type, 2>& outputs = recurrent_layer_forward_propagation->outputs;

    outputs.resize(batch_size, output_size);
    outputs.setZero();

    hidden_states.resize(batch_size, output_size);
    hidden_states.setZero();

    for (Index time_step = 0; time_step < time_steps; time_step++)
    {
        Tensor<type, 3> current_inputs = inputs.slice(DSizes<Index, 3>{0, time_step, 0},
                                                      DSizes<Index, 3>{batch_size, 1, input_size})
                                             .reshape(DSizes<Index, 3>{batch_size, 1, input_size});

        multiply_matrices(thread_pool_device.get(), current_inputs, input_weights);
        sum_matrices(thread_pool_device.get(), biases, current_inputs);

        hidden_states = hidden_states * recurrent_weights;
        hidden_states += current_inputs.reshape(DSizes<Index, 2>{batch_size, input_size});

        calculate_activations(hidden_states, empty_2);

        outputs = hidden_states.contract(output_weights, Eigen::array<Eigen::IndexPair<Index>, 1>{{Eigen::IndexPair<Index>(1, 0)}}) + output_biases;

        calculate_activations(outputs, empty_2);

    }
}


void Recurrent::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                    const vector<pair<type*, dimensions>>& delta_pairs,
                                    unique_ptr<LayerForwardPropagation>& forward_propagation,
                                    unique_ptr<LayerBackPropagation>& back_propagation) const
{
/*
    const Index batch_size = input_pairs[0].second[0];
    const Index outputs_number = get_outputs_number();
    const Index inputs_number = get_inputs_number();

    RecurrentLayerForwardPropagation* recurrent_layer_forward_propagation =
            static_cast<RecurrentLayerForwardPropagation*>(forward_propagation.get());

    RecurrentBackPropagation* recurrent_back_propagation =
            static_cast<RecurrentBackPropagation*>(back_propagation.get());

    // Forward propagation

    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    Tensor<type, 2>& current_inputs = recurrent_layer_forward_propagation->current_inputs;

    const Tensor<type, 3>& activation_derivatives = recurrent_layer_forward_propagation->activation_derivatives;

    Tensor<type, 2>& current_activations_derivatives = recurrent_layer_forward_propagation->current_activations_derivatives;

    // Back propagation

    const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    const bool& is_first_layer = recurrent_back_propagation->is_first_layer;

    //Tensor<type, 1>& current_deltas = recurrent_back_propagation->current_deltas;

    Tensor<type, 2>& combination_deltas = recurrent_back_propagation->combination_deltas;
    Tensor<type, 1>& current_combinations_derivatives = recurrent_back_propagation->current_combinations_derivatives;

    Tensor<type, 2>& combinations_bias_derivatives = recurrent_back_propagation->combinations_bias_derivatives;
    combinations_bias_derivatives.setZero();

    Tensor<type, 3>& combinations_input_weight_derivatives = recurrent_back_propagation->combinations_input_weight_derivatives;
    combinations_input_weight_derivatives.setZero();

    Tensor<type, 3>& combinations_recurrent_weight_derivatives = recurrent_back_propagation->combinations_recurrent_weight_derivatives;
    combinations_recurrent_weight_derivatives.setZero();

    Tensor<type, 1>& bias_derivatives = recurrent_back_propagation->bias_derivatives;
    bias_derivatives.setZero();

    Tensor<type, 2>& input_weight_derivatives = recurrent_back_propagation->input_weight_derivatives;
    input_weight_derivatives.setZero();

    Tensor<type, 2>& recurrent_weight_derivatives = recurrent_back_propagation->recurrent_weight_derivatives;
    recurrent_weight_derivatives.setZero();

    Tensor<type, 3>& input_derivatives = recurrent_back_propagation->input_derivatives;

    const Eigen::array<IndexPair<Index>, 1> combinations_weights_indices = { IndexPair<Index>(2, 0) };

    for (Index time_step = 0; time_step < time_steps; time_step++)
        current_inputs = inputs.chip(time_step, 1);

    for(Index sample_index = 0; sample_index < samples_number; sample_index++)
    {
        current_inputs.device(*thread_pool_device) = inputs.chip(sample_index, 0);

        current_deltas.device(*thread_pool_device) = deltas.chip(sample_index, 0);

        get_row(current_activations_derivatives, activation_derivatives, sample_index);

        if(sample_index % time_steps == 0)
        {
            combinations_bias_derivatives.setZero();
            combinations_input_weight_derivatives.setZero();
            combinations_recurrent_weight_derivatives.setZero();
        }
        else
        {
            // Combinations biases derivatives

            multiply_rows(combinations_bias_derivatives, current_activations_derivatives);

            combinations_bias_derivatives.device(*thread_pool_device) = combinations_bias_derivatives.contract(recurrent_weights, A_B);

            // Combinations weights derivatives

            multiply_matrices(thread_pool_device, combinations_input_weight_derivatives, current_activations_derivatives);

            combinations_input_weight_derivatives.device(*thread_pool_device) 
                = combinations_input_weight_derivatives.contract(recurrent_weights, combinations_weights_indices);

            // Combinations recurrent weights derivatives

            multiply_matrices(thread_pool_device, combinations_recurrent_weight_derivatives, current_activations_derivatives);

            combinations_recurrent_weight_derivatives.device(*thread_pool_device) 
                = combinations_recurrent_weight_derivatives.contract(recurrent_weights, combinations_weights_indices);
        }

        current_combinations_derivatives.device(*thread_pool_device) = current_deltas * current_activations_derivatives;

        combination_deltas.chip(sample_index, 0).device(*thread_pool_device) = current_combinations_derivatives;

        sum_diagonal(combinations_bias_derivatives, type(1));

        // Biases derivatives

        bias_derivatives.device(*thread_pool_device)
            += combinations_bias_derivatives.contract(current_combinations_derivatives, A_B);

//        combinations_input_weight_derivatives += current_inputs
//            .reshape(Eigen::array<Index, 2>({ inputs_number, 1 }))
//            .broadcast(Eigen::array<Index, 3>({ 1, neurons_number, 1 }));

//        for(Index neuron_index = 0; neuron_index < neurons_number; neuron_index++)
//            for(Index input_index = 0; input_index < inputs_number; input_index++)
//                combinations_input_weight_derivatives(input_index, neuron_index, neuron_index) += current_inputs(input_index);

        if(sample_index % time_steps != 0)
        {
            // @todo parallelize

            for(Index neuron_index = 0; neuron_index < neurons_number; neuron_index++)
                for(Index activation_index = 0; activation_index < neurons_number; activation_index++)
                    combinations_recurrent_weight_derivatives(activation_index, neuron_index, neuron_index)
                        += outputs(sample_index - 1, activation_index);

//            combinations_recurrent_weight_derivatives += outputs.chip(sample_index - 1, 0)
//                .reshape(Eigen::array<Index, 2>({ neurons_number, 1 }))
//                .broadcast(Eigen::array<Index, 3>({ 1, neurons_number, 1 }));
        }

        // Weights derivatives

        input_weight_derivatives.device(*thread_pool_device)
            += combinations_input_weight_derivatives.contract(current_combinations_derivatives, combinations_weights_indices);

        recurrent_weight_derivatives.device(*thread_pool_device)
            += combinations_recurrent_weight_derivatives.contract(current_combinations_derivatives, combinations_weights_indices);
    }

    // Input derivatives

    if(!is_first_layer)
        input_derivatives.device(*thread_pool_device) = combination_deltas.contract(input_weights, A_BT);
*/
}


void Recurrent::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                Index& index,
                                Tensor<type, 1>& gradient) const
{
    RecurrentBackPropagation* recurrent_back_propagation =
        static_cast<RecurrentBackPropagation*>(back_propagation.get());

    copy_to_vector(gradient, recurrent_back_propagation->bias_derivatives, index);
    copy_to_vector(gradient, recurrent_back_propagation->input_weight_derivatives, index);
    copy_to_vector(gradient, recurrent_back_propagation->recurrent_weight_derivatives, index);
}


string Recurrent::get_expression(const vector<string>& input_names,
                                        const vector<string>& output_names) const
{
    ostringstream buffer;

    for(size_t j = 0; j < output_names.size(); j++)
    {
        const Tensor<type, 1> synaptic_weights_column =  recurrent_weights.chip(j,1);

        buffer << output_names[j] << " = " << get_activation_function_string_expression() << "( " << biases(j) << " +";

        for(size_t i = 0; i < input_names.size() - 1; i++)
           buffer << " (" << input_names[i] << "*" << synaptic_weights_column(i) << ") +";

        buffer << " (" << input_names[input_names.size() - 1] << "*" << synaptic_weights_column[input_names.size() - 1] << "));\n";
    }

    return buffer.str();
}


string Recurrent::get_activation_function_string_expression() const
{
    switch(activation_function)
    {
        case Activation::HyperbolicTangent: return "tanh";

        case Activation::Linear: return string();

        default: return get_activation_function_string();
    }
}


void Recurrent::from_XML(const XMLDocument& document)
{
    const XMLElement* recurrent_layer_element = document.FirstChildElement("Recurrent");

    if(!recurrent_layer_element)
        throw runtime_error("Recurrent layer element is nullptr.\n");

    set_input_dimensions({ read_xml_index(recurrent_layer_element, "InputsNumber") });
    set_output_dimensions({ read_xml_index(recurrent_layer_element, "NeuronsNumber") });
    set_activation_function(read_xml_string(recurrent_layer_element, "Activation"));
/*
    set_parameters(to_type_vector(read_xml_string(recurrent_layer_element, "Parameters"), " "));
*/
}


void Recurrent::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Recurrent");

    add_xml_element(printer, "InputsNumber", to_string(get_input_dimensions()[0]));
    add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions()[0]));
    add_xml_element(printer, "Activation", get_activation_function_string());
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();
}


RecurrentLayerForwardPropagation::RecurrentLayerForwardPropagation(const Index& new_batch_size, Layer* new_layer) : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> RecurrentLayerForwardPropagation::get_outputs_pair() const
{
    const Index outputs_number = layer->get_outputs_number();

    return {(type*)outputs.data(), {{batch_size, outputs_number}}};
}


void RecurrentLayerForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_dimensions()[0];
    const Index time_steps = 0;

    batch_size = new_batch_size;

    current_inputs.resize(batch_size, inputs_number);
    current_activations_derivatives.resize(batch_size, outputs_number);

    activation_derivatives.resize(batch_size, time_steps, outputs_number);

    outputs.resize(batch_size, outputs_number);
    outputs.setZero();
}


void RecurrentLayerForwardPropagation::print() const
{
}


void RecurrentBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    batch_size = new_batch_size;

    const Index inputs_number = layer->get_input_dimensions()[0];
    const Index outputs_number = layer->get_outputs_number();

    //current_deltas.resize(neurons_number);

    combinations_bias_derivatives.resize(outputs_number, outputs_number);

    combinations_input_weight_derivatives.resize(inputs_number, outputs_number, outputs_number);

    combinations_recurrent_weight_derivatives.resize(outputs_number, outputs_number, outputs_number);

    combination_deltas.resize(batch_size, outputs_number);
    current_combinations_derivatives.resize(outputs_number);

    bias_derivatives.resize(outputs_number);

    input_weight_derivatives.resize(inputs_number, outputs_number);

    recurrent_weight_derivatives.resize(outputs_number, outputs_number);

    const Index time_steps = 0;

    input_derivatives.resize(batch_size, time_steps, inputs_number);
}


void RecurrentBackPropagation::print() const
{

}


RecurrentBackPropagation::RecurrentBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<pair<type*, dimensions>> RecurrentBackPropagation::get_input_derivative_pairs() const
{
    const Index inputs_number = layer->get_input_dimensions()[0];

    return {{(type*)(input_derivatives.data()), {batch_size, inputs_number}}};
}

}

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
