//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "long_short_term_memory_layer.h"
#include "tensors.h"

namespace opennn
{

LongShortTermMemoryLayer::LongShortTermMemoryLayer(const Index& new_inputs_number, 
                                                   const Index& new_neurons_number, 
                                                   const Index& new_timesteps) : Layer()
{
    set(new_inputs_number, new_neurons_number, new_timesteps);

    layer_type = Type::LongShortTermMemory;
}


dimensions LongShortTermMemoryLayer::get_input_dimensions() const
{
    return { input_weights.dimension(0) };
}


dimensions LongShortTermMemoryLayer::get_output_dimensions() const
{
/*
    const Index neurons_number = get_neurons_number();

    return { neurons_number };
*/
    return {};
}


Index LongShortTermMemoryLayer::get_parameters_number() const
{
    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    return 4 * outputs_number * (1 + inputs_number + outputs_number);
}


Index LongShortTermMemoryLayer::get_timesteps() const
{
    return time_steps;
}


Tensor<type, 1> LongShortTermMemoryLayer::get_parameters() const
{
    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> parameters(parameters_number);

    Index index = 0;

    copy_to_vector(parameters, forget_biases, index);
    copy_to_vector(parameters, input_biases, index);
    copy_to_vector(parameters, state_biases, index);
    copy_to_vector(parameters, output_biases, index);
    copy_to_vector(parameters, forget_weights, index);
    copy_to_vector(parameters, input_weights, index);
    copy_to_vector(parameters, state_weights, index);
    copy_to_vector(parameters, output_weights, index);
    copy_to_vector(parameters, forget_recurrent_weights, index);
    copy_to_vector(parameters, input_recurrent_weights, index);
    copy_to_vector(parameters, state_recurrent_weights, index);
    copy_to_vector(parameters, output_recurrent_weights, index);

    return parameters;
}


const LongShortTermMemoryLayer::ActivationFunction& LongShortTermMemoryLayer::get_activation_function() const
{
    return activation_function;
}


const LongShortTermMemoryLayer::ActivationFunction& LongShortTermMemoryLayer::get_recurrent_activation_function() const
{
    return recurrent_activation_function;
}


string LongShortTermMemoryLayer::get_activation_function_string() const
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

    default: return string();
    }
}


string LongShortTermMemoryLayer::write_recurrent_activation_function() const
{
    switch(recurrent_activation_function)
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

    default: return string();
    }
}


void LongShortTermMemoryLayer::set(const Index& new_inputs_number, const Index& new_neurons_number, const Index& new_timesteps)
{
    input_biases.resize(new_neurons_number);
    forget_biases.resize(new_neurons_number);
    state_biases.resize(new_neurons_number);
    output_biases.resize(new_neurons_number);

    input_weights.resize(new_inputs_number, new_neurons_number);
    forget_weights.resize(new_inputs_number, new_neurons_number);
    state_weights.resize(new_inputs_number, new_neurons_number);
    output_weights.resize(new_inputs_number, new_neurons_number);

    input_recurrent_weights.resize(new_neurons_number, new_neurons_number);
    forget_recurrent_weights.resize(new_neurons_number, new_neurons_number);
    state_recurrent_weights.resize(new_neurons_number, new_neurons_number);
    output_recurrent_weights.resize(new_neurons_number, new_neurons_number);

    time_steps = new_timesteps;

    set_parameters_random();

    name = "long_short_term_memory_layer";

    layer_type = Type::LongShortTermMemory;
}


void LongShortTermMemoryLayer::set_input_dimensions(const dimensions& new_input_dimensions)
{
    const Index outputs_number = get_outputs_number();
    const Index time_steps = get_timesteps();

    set(new_input_dimensions[0], outputs_number, time_steps);
}


void LongShortTermMemoryLayer::set_output_dimensions(const dimensions& new_output_dimensions)
{
    const Index inputs_number = get_inputs_number();
    const Index time_steps = get_timesteps();

    set(inputs_number, new_output_dimensions[0], time_steps);
}


void LongShortTermMemoryLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    const Index outputs_number = get_outputs_number();
    const Index inputs_number = get_inputs_number();

    Index current_index = index;

    const type* new_parameters_data = new_parameters.data();

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            Index size = outputs_number;

            memcpy(forget_biases.data(), new_parameters_data + current_index, size * sizeof(type));
            current_index += size;

            memcpy(input_biases.data(), new_parameters_data + current_index, size * sizeof(type));
            current_index += size;

            memcpy(state_biases.data(), new_parameters_data + current_index, size * sizeof(type));
            current_index += size;

            memcpy(output_biases.data(), new_parameters_data + current_index, size * sizeof(type));
        }

        #pragma omp section
        {
            Index size = inputs_number * outputs_number;
            Index local_index = current_index + outputs_number * 4;

            memcpy(forget_weights.data(), new_parameters_data + local_index, size * sizeof(type));
            local_index += size;

            memcpy(input_weights.data(), new_parameters_data + local_index, size * sizeof(type));
            local_index += size;

            memcpy(state_weights.data(), new_parameters_data + local_index, size * sizeof(type));
            local_index += size;

            memcpy(output_weights.data(), new_parameters_data + local_index, size * sizeof(type));
        }

        #pragma omp section
        {
            Index size = outputs_number * outputs_number;
            Index local_index = current_index + outputs_number * 4 + inputs_number * outputs_number * 4;  // Skip bias and weights size

            memcpy(forget_recurrent_weights.data(), new_parameters_data + local_index, size * sizeof(type));
            local_index += size;

            memcpy(input_recurrent_weights.data(), new_parameters_data + local_index, size * sizeof(type));
            local_index += size;

            memcpy(state_recurrent_weights.data(), new_parameters_data + local_index, size * sizeof(type));
            local_index += size;

            memcpy(output_recurrent_weights.data(), new_parameters_data + local_index, size * sizeof(type));
        }
    }
}


void LongShortTermMemoryLayer::set_activation_function(const LongShortTermMemoryLayer::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


void LongShortTermMemoryLayer::set_activation_function(const string& new_activation_function_name)
{
    if(new_activation_function_name == "Logistic")
        activation_function = ActivationFunction::Logistic;
    else if(new_activation_function_name == "HyperbolicTangent")
        activation_function = ActivationFunction::HyperbolicTangent;
    else if(new_activation_function_name == "Linear")
        activation_function = ActivationFunction::Linear;
    else if(new_activation_function_name == "RectifiedLinear")
        activation_function = ActivationFunction::RectifiedLinear;
    else if(new_activation_function_name == "ScaledExponentialLinear")
        activation_function = ActivationFunction::ScaledExponentialLinear;
    else if(new_activation_function_name == "SoftPlus")
        activation_function = ActivationFunction::SoftPlus;
    else if(new_activation_function_name == "SoftSign")
        activation_function = ActivationFunction::SoftSign;
    else if(new_activation_function_name == "HardSigmoid")
        activation_function = ActivationFunction::HardSigmoid;
    else if(new_activation_function_name == "ExponentialLinear")
        activation_function = ActivationFunction::ExponentialLinear;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function_name + ".\n");
}


void LongShortTermMemoryLayer::set_recurrent_activation_function(const LongShortTermMemoryLayer::ActivationFunction& new_recurrent_activation_function)
{
    recurrent_activation_function = new_recurrent_activation_function;
}


void LongShortTermMemoryLayer::set_recurrent_activation_function(const string& new_recurrent_activation_function_name)
{
    if(new_recurrent_activation_function_name == "Logistic")
        recurrent_activation_function = ActivationFunction::Logistic;
    else if(new_recurrent_activation_function_name == "HyperbolicTangent")
        recurrent_activation_function = ActivationFunction::HyperbolicTangent;
    else if(new_recurrent_activation_function_name == "Linear")
        recurrent_activation_function = ActivationFunction::Linear;
    else if(new_recurrent_activation_function_name == "RectifiedLinear")
        recurrent_activation_function = ActivationFunction::RectifiedLinear;
    else if(new_recurrent_activation_function_name == "ScaledExponentialLinear")
        recurrent_activation_function = ActivationFunction::ScaledExponentialLinear;
    else if(new_recurrent_activation_function_name == "SoftPlus")
        recurrent_activation_function = ActivationFunction::SoftPlus;
    else if(new_recurrent_activation_function_name == "SoftSign")
        recurrent_activation_function = ActivationFunction::SoftSign;
    else if(new_recurrent_activation_function_name == "HardSigmoid")
        recurrent_activation_function = ActivationFunction::HardSigmoid;
    else if(new_recurrent_activation_function_name == "ExponentialLinear")
        recurrent_activation_function = ActivationFunction::ExponentialLinear;
    else
        throw runtime_error("Unknown activation function: " + new_recurrent_activation_function_name + ".\n");
}


void LongShortTermMemoryLayer::set_timesteps(const Index& new_timesteps)
{
    time_steps = new_timesteps;
}


void LongShortTermMemoryLayer::set_parameters_constant(const type& value)
{
    forget_biases.setConstant(value);
    input_biases.setConstant(value);
    state_biases.setConstant(value);
    output_biases.setConstant(value);

    forget_weights.setConstant(value);
    input_weights.setConstant(value);
    state_weights.setConstant(value);
    output_weights.setConstant(value);

    forget_recurrent_weights.setConstant(value);
    input_recurrent_weights.setConstant(value);
    state_recurrent_weights.setConstant(value);
    output_recurrent_weights.setConstant(value);
}


void LongShortTermMemoryLayer::set_parameters_random()
{
    set_random(forget_biases);

    set_random(input_biases);

    set_random(state_biases);

    set_random(output_biases);

    set_random(forget_weights);

    set_random(input_weights);

    set_random(state_weights);

    set_random(output_weights);

    set_random(forget_recurrent_weights);

    set_random(input_recurrent_weights);

    set_random(state_recurrent_weights);

    set_random(output_recurrent_weights);
}


void LongShortTermMemoryLayer::calculate_combinations(const Tensor<type, 1>& inputs,
                                                      const Tensor<type, 2>& weights,
                                                      const Tensor<type, 1>& hidden_states,
                                                      const Tensor<type, 2>& recurrent_weights,
                                                      const Tensor<type, 1>& biases,
                                                      Tensor<type, 1>& combinations) const
{
    combinations.device(*thread_pool_device) = inputs.contract(weights, AT_B)
                                             + hidden_states.contract(recurrent_weights, AT_B)
                                             + biases;
}


void LongShortTermMemoryLayer::calculate_activations(Tensor<type, 1>& activations, 
                                                     Tensor<type, 1>& activation_derivatives) const
{
    switch(activation_function)
    {
    case ActivationFunction::Linear: linear(activations, activation_derivatives); return;

    case ActivationFunction::Logistic: logistic(activations, activation_derivatives); return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(activations, activation_derivatives); return;

    case ActivationFunction::RectifiedLinear: rectified_linear(activations, activation_derivatives); return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(activations, activation_derivatives); return;

    case ActivationFunction::SoftPlus: soft_plus(activations, activation_derivatives); return;

    case ActivationFunction::SoftSign: soft_sign(activations, activation_derivatives); return;

    case ActivationFunction::HardSigmoid: hard_sigmoid(activations, activation_derivatives); return;

    case ActivationFunction::ExponentialLinear: exponential_linear(activations, activation_derivatives); return;

    default: throw runtime_error("Unknown activation function");
    }
}


void LongShortTermMemoryLayer::calculate_recurrent_activations(Tensor<type, 1>& activations, Tensor<type, 1>& activation_derivatives) const
{
    switch(recurrent_activation_function)
    {
    case ActivationFunction::Linear: linear(activations, activation_derivatives); return;

    case ActivationFunction::Logistic: logistic(activations, activation_derivatives); return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(activations, activation_derivatives); return;

    case ActivationFunction::RectifiedLinear: rectified_linear(activations, activation_derivatives); return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(activations, activation_derivatives); return;

    case ActivationFunction::SoftPlus: soft_plus(activations, activation_derivatives); return;

    case ActivationFunction::SoftSign: soft_sign(activations, activation_derivatives); return;

    case ActivationFunction::HardSigmoid: hard_sigmoid(activations, activation_derivatives); return;

    case ActivationFunction::ExponentialLinear: exponential_linear(activations, activation_derivatives); return;

    default: throw runtime_error("Unknown activation function");
    }
}


void LongShortTermMemoryLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                                 unique_ptr<LayerForwardPropagation>& forward_propagation,
                                                 const bool& is_training)
{

    const Index samples_number = input_pairs[0].second[0];

    LongShortTermMemoryLayerForwardPropagation* long_short_term_memory_layer_forward_propagation =
            static_cast<LongShortTermMemoryLayerForwardPropagation*>(forward_propagation.get());

    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);
    Tensor<type, 1>& current_inputs = long_short_term_memory_layer_forward_propagation->current_inputs;

    Tensor<type, 2, RowMajor>& forget_activations = long_short_term_memory_layer_forward_propagation->forget_activations;
    Tensor<type, 2, RowMajor>& forget_activations_derivatives = long_short_term_memory_layer_forward_propagation->forget_activations_derivatives;

    Tensor<type, 1>& current_forget_activations = long_short_term_memory_layer_forward_propagation->current_forget_activations;
    Tensor<type, 1>& current_forget_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives;

    Tensor<type, 2, RowMajor>& input_activations = long_short_term_memory_layer_forward_propagation->input_activations;
    Tensor<type, 2, RowMajor>& input_activations_derivatives = long_short_term_memory_layer_forward_propagation->input_activations_derivatives;

    Tensor<type, 1>& current_input_activations = long_short_term_memory_layer_forward_propagation->current_input_activations;
    Tensor<type, 1>& current_input_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_input_activations_derivatives;

    Tensor<type, 2, RowMajor>& state_activations = long_short_term_memory_layer_forward_propagation->state_activations;
    Tensor<type, 2, RowMajor>& state_activations_derivatives = long_short_term_memory_layer_forward_propagation->state_activations_derivatives;

    Tensor<type, 1>& current_state_activations = long_short_term_memory_layer_forward_propagation->current_state_activations;
    Tensor<type, 1>& current_state_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_state_activations_derivatives;

    Tensor<type, 2, RowMajor>& output_activations = long_short_term_memory_layer_forward_propagation->output_activations;
    Tensor<type, 2, RowMajor>& output_activations_derivatives = long_short_term_memory_layer_forward_propagation->output_activations_derivatives;

    Tensor<type, 1>& current_output_activations = long_short_term_memory_layer_forward_propagation->current_output_activations;
    Tensor<type, 1>& current_output_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_output_activations_derivatives;

    Tensor<type, 2, RowMajor>& cell_states= long_short_term_memory_layer_forward_propagation->cell_states;

    Tensor<type, 1>& previous_cell_states = long_short_term_memory_layer_forward_propagation->previous_cell_states;
    Tensor<type, 1>& current_cell_states = long_short_term_memory_layer_forward_propagation->current_cell_states;

    Tensor<type, 2, RowMajor>& hidden_states = long_short_term_memory_layer_forward_propagation->hidden_states;
    Tensor<type, 2, RowMajor>& hidden_states_activations_derivatives = long_short_term_memory_layer_forward_propagation->hidden_states_activations_derivatives;

    Tensor<type, 1>& previous_hidden_states = long_short_term_memory_layer_forward_propagation->previous_hidden_states;
    Tensor<type, 1>& current_hidden_states = long_short_term_memory_layer_forward_propagation->current_hidden_states;
    Tensor<type, 1>& current_hidden_states_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_hidden_states_activations_derivatives;

    Tensor<type, 2>& outputs = long_short_term_memory_layer_forward_propagation->outputs;

    for(Index i = 0; i < samples_number; i++)
    {
        if(i%time_steps == 0)
        {
            previous_cell_states.setZero();
            previous_hidden_states.setZero();
        }
        else
        {
            previous_cell_states = current_cell_states;
            previous_hidden_states = current_hidden_states;
        }

        current_inputs.device(*thread_pool_device) = inputs.chip(i, 0);

        calculate_combinations(current_inputs,
                               forget_weights,
                               previous_hidden_states,
                               forget_recurrent_weights,
                               forget_biases,
                               current_forget_activations);

        calculate_combinations(current_inputs,
                               input_weights,
                               previous_hidden_states,
                               input_recurrent_weights,
                               input_biases,
                               current_input_activations);

        calculate_combinations(current_inputs,
                               state_weights,
                               previous_hidden_states,
                               state_recurrent_weights,
                               state_biases,
                               current_state_activations);

        calculate_combinations(current_inputs,
                               output_weights,
                               previous_hidden_states,
                               output_recurrent_weights,
                               output_biases,
                               current_output_activations);

        if(is_training)
        {
            calculate_recurrent_activations(current_forget_activations,
                                            current_forget_activations_derivatives);

            calculate_recurrent_activations(current_input_activations,
                                            current_input_activations_derivatives);

            calculate_activations(current_state_activations,
                                  current_state_activations_derivatives);

            calculate_recurrent_activations(current_output_activations,
                                            current_output_activations_derivatives);

            set_row(forget_activations_derivatives, current_forget_activations_derivatives, i);

            set_row(input_activations_derivatives, current_input_activations_derivatives, i);

            set_row(state_activations_derivatives, current_state_activations_derivatives, i);

            set_row(output_activations_derivatives, current_output_activations_derivatives, i);
        }
        else
        {
            calculate_recurrent_activations(current_forget_activations, empty);

            calculate_recurrent_activations(current_input_activations, empty);

            calculate_activations(current_state_activations, empty);

            calculate_recurrent_activations(current_output_activations, empty);
        }

        set_row(forget_activations, current_forget_activations, i);

        set_row(input_activations, current_input_activations, i);

        set_row(state_activations, current_state_activations, i);
   
        set_row(output_activations, current_output_activations, i);

        // Cell states
        
        current_cell_states.device(*thread_pool_device)
            = current_forget_activations * previous_cell_states + current_input_activations * current_state_activations;

        set_row(cell_states, current_cell_states, i);

        // Hidden states

        if(is_training)
        {
/*
            calculate_activations_derivatives(current_cell_states,
                                              current_hidden_states,
                                              current_hidden_states_activations_derivatives);
*/
            set_row(hidden_states_activations_derivatives, current_hidden_states_activations_derivatives, i);
        }
        else
        {
/*
            calculate_activations(current_cell_states,
                                  current_hidden_states);
*/
        }

        current_hidden_states.device(*thread_pool_device) = current_output_activations * current_hidden_states;

        set_row(hidden_states, current_hidden_states, i);

        // Activations 2d

        outputs.chip(i, 0).device(*thread_pool_device) = current_hidden_states;
    }
}


void LongShortTermMemoryLayer::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                              const vector<pair<type*, dimensions>>& delta_pairs,
                                              unique_ptr<LayerForwardPropagation>& forward_propagation,
                                              unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);
    const TensorMap<Tensor<type, 2>> deltas = tensor_map_2(delta_pairs[0]);

     LongShortTermMemoryLayerForwardPropagation* long_short_term_memory_layer_forward_propagation =
            static_cast<LongShortTermMemoryLayerForwardPropagation*>(forward_propagation.get());

    LongShortTermMemoryLayerBackPropagation* long_short_term_memory_back_propagation =
            static_cast<LongShortTermMemoryLayerBackPropagation*>(back_propagation.get());
/*
    calculate_forget_parameter_derivatives(inputs,
                                            deltas,
                                            long_short_term_memory_layer_forward_propagation,
                                            long_short_term_memory_back_propagation);

    calculate_input_parameter_derivatives(inputs,
                                           deltas,
                                           long_short_term_memory_layer_forward_propagation,
                                           long_short_term_memory_back_propagation);

    calculate_state_parameter_derivatives(inputs,
                                           deltas,
                                           long_short_term_memory_layer_forward_propagation,
                                           long_short_term_memory_back_propagation);

    calculate_output_parameter_derivatives(inputs,
                                            deltas,
                                            long_short_term_memory_layer_forward_propagation,
                                            long_short_term_memory_back_propagation);
*/
}


void LongShortTermMemoryLayer::calculate_forget_parameter_derivatives(const Tensor<type, 2>& inputs,
                                                                      const Tensor<type, 2>& deltas,
                                                                      unique_ptr<LongShortTermMemoryLayerForwardPropagation>& long_short_term_memory_layer_forward_propagation,
                                                                      unique_ptr<LongShortTermMemoryLayerBackPropagation>& long_short_term_memory_back_propagation) const
{
/*
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = inputs_number * neurons_number;
    const Index samples_number = inputs.dimension(0);

    // Forward propagation

    Tensor<type, 1>& current_inputs = long_short_term_memory_layer_forward_propagation->current_inputs;

    const Tensor<type, 2, RowMajor>& forget_activations = long_short_term_memory_layer_forward_propagation->forget_activations;
    const Tensor<type, 2, RowMajor>& forget_activations_derivatives = long_short_term_memory_layer_forward_propagation->forget_activations_derivatives;

    Tensor<type, 1>& current_forget_activations = long_short_term_memory_layer_forward_propagation->current_forget_activations;
    Tensor<type, 1>& current_forget_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives;

    const Tensor<type, 2, RowMajor>& input_activations = long_short_term_memory_layer_forward_propagation->input_activations;
    const Tensor<type, 2, RowMajor>& input_activations_derivatives = long_short_term_memory_layer_forward_propagation->input_activations_derivatives;

    Tensor<type, 1>& current_input_activations = long_short_term_memory_layer_forward_propagation->current_input_activations;
    Tensor<type, 1>& current_input_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_input_activations_derivatives;

    const Tensor<type, 2, RowMajor>& state_activations = long_short_term_memory_layer_forward_propagation->state_activations;
    const Tensor<type, 2, RowMajor>& state_activations_derivatives = long_short_term_memory_layer_forward_propagation->state_activations_derivatives;

    Tensor<type, 1>& current_state_activations = long_short_term_memory_layer_forward_propagation->current_state_activations;
    Tensor<type, 1>& current_state_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_state_activations_derivatives;

    const Tensor<type, 2, RowMajor>& output_activations = long_short_term_memory_layer_forward_propagation->output_activations;
    const Tensor<type, 2, RowMajor>& output_activations_derivatives = long_short_term_memory_layer_forward_propagation->output_activations_derivatives;

    Tensor<type, 1>& current_output_activations = long_short_term_memory_layer_forward_propagation->current_output_activations;
    Tensor<type, 1>& current_output_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_output_activations_derivatives;

    const Tensor<type, 2, RowMajor>& cell_states = long_short_term_memory_layer_forward_propagation->cell_states;

    Tensor<type, 1>& previous_cell_states = long_short_term_memory_layer_forward_propagation->previous_cell_states;
    Tensor<type, 1>& current_cell_states = long_short_term_memory_layer_forward_propagation->current_cell_states;

    const Tensor<type, 2, RowMajor>& hidden_states = long_short_term_memory_layer_forward_propagation->hidden_states;
    const Tensor<type, 2, RowMajor>& hidden_states_activations_derivatives = long_short_term_memory_layer_forward_propagation->hidden_states_activations_derivatives;

    Tensor<type, 1>& previous_hidden_states = long_short_term_memory_layer_forward_propagation->previous_hidden_states;
    Tensor<type, 1>& current_hidden_states = long_short_term_memory_layer_forward_propagation->current_hidden_states;
    Tensor<type, 1>& current_hidden_states_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_hidden_states_activations_derivatives;

    // Back propagation

    Tensor<type, 1>& current_deltas = long_short_term_memory_back_propagation->current_deltas;

    Tensor<type, 2>& forget_combinations_weights_derivatives = long_short_term_memory_back_propagation->forget_combinations_weights_derivatives;
    Tensor<type, 2>& forget_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->forget_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& forget_combinations_biases_derivatives = long_short_term_memory_back_propagation->forget_combinations_biases_derivatives;

    Tensor<type, 2>& input_combinations_weights_derivatives = long_short_term_memory_back_propagation->input_combinations_weights_derivatives;
    Tensor<type, 2>& input_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->input_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& input_combinations_biases_derivatives = long_short_term_memory_back_propagation->input_combinations_biases_derivatives;

    Tensor<type, 2>& state_combinations_weights_derivatives = long_short_term_memory_back_propagation->state_combinations_weights_derivatives;
    Tensor<type, 2>& state_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->state_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& state_combinations_biases_derivatives = long_short_term_memory_back_propagation->state_combinations_biases_derivatives;

    Tensor<type, 2>& output_combinations_weights_derivatives = long_short_term_memory_back_propagation->output_combinations_weights_derivatives;
    Tensor<type, 2>& output_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->output_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& output_combinations_biases_derivatives = long_short_term_memory_back_propagation->output_combinations_biases_derivatives;

    Tensor<type, 2>& cell_states_weights_derivatives = long_short_term_memory_back_propagation->cell_states_weights_derivatives;
    Tensor<type, 2>& cell_states_recurrent_weights_derivatives = long_short_term_memory_back_propagation->cell_states_recurrent_weights_derivatives;
    Tensor<type, 2>& cell_states_biases_derivatives = long_short_term_memory_back_propagation->cell_states_biases_derivatives;

    Tensor<type, 2>& hidden_states_weights_derivatives = long_short_term_memory_back_propagation->hidden_states_weights_derivatives;
    Tensor<type, 2>& hidden_states_recurrent_weights_derivatives = long_short_term_memory_back_propagation->hidden_states_recurrent_weights_derivatives;
    Tensor<type, 2>& hidden_states_biases_derivatives = long_short_term_memory_back_propagation->hidden_states_biases_derivatives;

    Tensor<type, 1>& forget_weights_derivatives = long_short_term_memory_back_propagation->forget_weights_derivatives;
    Tensor<type, 1>& forget_recurrent_weights_derivatives = long_short_term_memory_back_propagation->forget_recurrent_weights_derivatives;
    Tensor<type, 1>& forget_biases_derivatives = long_short_term_memory_back_propagation->forget_biases_derivatives;
    forget_weights_derivatives.setZero();
    forget_recurrent_weights_derivatives.setZero();
    forget_biases_derivatives.setZero();

    Index weight_index = 0;
    Index recurrent_weight_index = 0;
    Index input_index = 0;
    Index neuron_index = 0;

    for(Index sample = 0; sample < samples_number; sample++)
    {
        current_inputs.device(*thread_pool_device) = inputs.chip(sample, 0);

        get_row(current_forget_activations, forget_activations, sample);
        get_row(current_forget_activations_derivatives, forget_activations_derivatives, sample);

        get_row(current_input_activations, input_activations, sample);
        get_row(current_input_activations_derivatives, input_activations_derivatives, sample);

        get_row(current_state_activations, state_activations, sample);
        get_row(current_state_activations_derivatives, state_activations_derivatives, sample);

        get_row(current_output_activations, output_activations, sample);
        get_row(current_output_activations_derivatives, output_activations_derivatives, sample);

        get_row(current_cell_states, cell_states, sample);

        get_row(current_hidden_states, hidden_states, sample);
        get_row(current_hidden_states_activations_derivatives, hidden_states_activations_derivatives, sample);

        current_deltas.device(*thread_pool_device) = deltas.chip(sample, 0);

        calculate_activations(current_cell_states);

        // FORGET PARAMETERS DERIVATIVES

        if(sample % time_steps == 0)
        {
            //set_derivatives_zero();

            previous_cell_states.setZero();
            previous_hidden_states.setZero();
        }
        else
        {
            get_row(previous_cell_states, cell_states, sample - 1);
            get_row(previous_hidden_states, hidden_states, sample - 1);

            // Forget combinations derivatives

            forget_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(forget_recurrent_weights, A_B);

            forget_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(forget_recurrent_weights, A_B);

            forget_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(forget_recurrent_weights, A_B);

            // Input combinations derivatives

            input_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(input_recurrent_weights, A_B);

            input_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(input_recurrent_weights, A_B);

            input_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(input_recurrent_weights, A_B);

            multiply_rows(input_combinations_weights_derivatives, current_input_activations_derivatives);

            multiply_rows(input_combinations_recurrent_weights_derivatives, current_input_activations_derivatives);

            multiply_rows(input_combinations_biases_derivatives, current_input_activations_derivatives);

            // State combinations derivatives

            state_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(state_recurrent_weights, A_B);

            state_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(state_recurrent_weights, A_B);

            state_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(state_recurrent_weights, A_B);

            multiply_rows(state_combinations_weights_derivatives, current_state_activations_derivatives);

            multiply_rows(state_combinations_recurrent_weights_derivatives, current_state_activations_derivatives);

            multiply_rows(state_combinations_biases_derivatives, current_state_activations_derivatives);

            // Output combinations derivatives

            output_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(output_recurrent_weights, A_B);

            output_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(output_recurrent_weights, A_B);

            output_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(output_recurrent_weights, A_B);

            multiply_rows(output_combinations_weights_derivatives, current_output_activations_derivatives);

            multiply_rows(output_combinations_recurrent_weights_derivatives, current_output_activations_derivatives);

            multiply_rows(output_combinations_biases_derivatives, current_output_activations_derivatives);
        }

        weight_index = 0;
        recurrent_weight_index = 0;

        input_index = 0;
        neuron_index = 0;

        for(Index i = 0; i < parameters_number; i++)
        {
            const type current_input = current_inputs(input_index);
            const type previous_hidden_state_activation = previous_hidden_states(neuron_index);

            forget_combinations_weights_derivatives(i, weight_index) += current_input;
            forget_combinations_recurrent_weights_derivatives(i, recurrent_weight_index) += previous_hidden_state_activation;
            forget_combinations_biases_derivatives(i, i) += type(1.0);

            input_index++;
            neuron_index++;

            if(input_index == inputs_number)
            {
                input_index = 0;
                weight_index++;
            }

            if(neuron_index == neurons_number)
            {
                neuron_index = 0;
                recurrent_weight_index++;
            }
        }

        // Forget weights derivatives

        multiply_rows(cell_states_weights_derivatives, current_forget_activations);

        multiply_rows(input_combinations_weights_derivatives, current_state_activations);

        cell_states_weights_derivatives.device(*thread_pool_device) += input_combinations_weights_derivatives;

        multiply_rows(state_combinations_weights_derivatives, current_input_activations);

        cell_states_weights_derivatives.device(*thread_pool_device) += state_combinations_weights_derivatives;

        multiply_rows(forget_combinations_weights_derivatives, current_forget_activations_derivatives * previous_cell_states);

        cell_states_weights_derivatives.device(*thread_pool_device) += forget_combinations_weights_derivatives;

        copy(cell_states_weights_derivatives.data(),
             cell_states_weights_derivatives.data() + cell_states_weights_derivatives.size(),
             hidden_states_weights_derivatives.data());

        multiply_rows(hidden_states_weights_derivatives, current_output_activations * current_hidden_states_activations_derivatives);

        multiply_rows(output_combinations_weights_derivatives, current_cell_states);

        hidden_states_weights_derivatives.device(*thread_pool_device) += output_combinations_weights_derivatives;

        forget_weights_derivatives.device(*thread_pool_device) += hidden_states_weights_derivatives.contract(current_deltas, A_B);

        // Forget recurrent weights derivatives

        if(sample % time_steps != 0)
        {
            multiply_rows(cell_states_recurrent_weights_derivatives, current_forget_activations);

            multiply_rows(input_combinations_recurrent_weights_derivatives, current_state_activations);

            cell_states_recurrent_weights_derivatives.device(*thread_pool_device) += input_combinations_recurrent_weights_derivatives;

            multiply_rows(state_combinations_recurrent_weights_derivatives, current_input_activations);

            cell_states_recurrent_weights_derivatives.device(*thread_pool_device) += state_combinations_recurrent_weights_derivatives;

            multiply_rows(forget_combinations_recurrent_weights_derivatives, current_forget_activations_derivatives * previous_cell_states);

            cell_states_recurrent_weights_derivatives.device(*thread_pool_device) += forget_combinations_recurrent_weights_derivatives;

            copy(cell_states_recurrent_weights_derivatives.data(),
                 cell_states_recurrent_weights_derivatives.data() + cell_states_recurrent_weights_derivatives.size(),
                 hidden_states_recurrent_weights_derivatives.data());

            multiply_rows(hidden_states_recurrent_weights_derivatives, current_output_activations * current_hidden_states_activations_derivatives);

            multiply_rows(output_combinations_weights_derivatives, current_cell_states);

            hidden_states_recurrent_weights_derivatives.device(*thread_pool_device) += output_combinations_recurrent_weights_derivatives;
        }

        forget_recurrent_weights_derivatives.device(*thread_pool_device) += hidden_states_recurrent_weights_derivatives.contract(current_deltas, A_B);

        // Forget biases derivatives

        multiply_rows(cell_states_biases_derivatives, current_forget_activations);

        multiply_rows(input_combinations_biases_derivatives, current_state_activations);

        cell_states_biases_derivatives.device(*thread_pool_device) += input_combinations_biases_derivatives;

        multiply_rows(state_combinations_biases_derivatives, current_input_activations);

        cell_states_biases_derivatives.device(*thread_pool_device) += state_combinations_biases_derivatives;

        multiply_rows(forget_combinations_biases_derivatives, current_forget_activations_derivatives * previous_cell_states);

        cell_states_biases_derivatives.device(*thread_pool_device) += forget_combinations_biases_derivatives;

        copy(cell_states_biases_derivatives.data(),
             cell_states_biases_derivatives.data() + cell_states_biases_derivatives.size(),
             hidden_states_biases_derivatives.data());

        multiply_rows(hidden_states_biases_derivatives, current_output_activations * current_hidden_states_activations_derivatives);

        multiply_rows(output_combinations_weights_derivatives, current_cell_states);

        hidden_states_biases_derivatives.device(*thread_pool_device) += output_combinations_biases_derivatives;

        forget_biases_derivatives.device(*thread_pool_device) += hidden_states_biases_derivatives.contract(current_deltas, A_B);
    }
    */
}


void LongShortTermMemoryLayer::calculate_input_parameter_derivatives(const Tensor<type, 2>& inputs,
                                                                     const Tensor<type, 2>& deltas,
                                                                     unique_ptr<LongShortTermMemoryLayerForwardPropagation>& long_short_term_memory_layer_forward_propagation,
                                                                     unique_ptr<LongShortTermMemoryLayerBackPropagation>& long_short_term_memory_back_propagation) const
{
/*
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = inputs_number * neurons_number;
    const Index samples_number = inputs.dimension(0);

    // Forward propagation

    Tensor<type, 1>& current_inputs = long_short_term_memory_layer_forward_propagation->current_inputs;

    const Tensor<type, 2, RowMajor>& forget_activations = long_short_term_memory_layer_forward_propagation->forget_activations;
    const Tensor<type, 2, RowMajor>& forget_activations_derivatives = long_short_term_memory_layer_forward_propagation->forget_activations_derivatives;

    Tensor<type, 1>& current_forget_activations = long_short_term_memory_layer_forward_propagation->current_forget_activations;
    Tensor<type, 1>& current_forget_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives;

    const Tensor<type, 2, RowMajor>& input_activations = long_short_term_memory_layer_forward_propagation->input_activations;
    const Tensor<type, 2, RowMajor>& input_activations_derivatives = long_short_term_memory_layer_forward_propagation->input_activations_derivatives;

    Tensor<type, 1>& current_input_activations = long_short_term_memory_layer_forward_propagation->current_input_activations;
    Tensor<type, 1>& current_input_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_input_activations_derivatives;

    const Tensor<type, 2, RowMajor>& state_activations = long_short_term_memory_layer_forward_propagation->state_activations;
    const Tensor<type, 2, RowMajor>& state_activations_derivatives = long_short_term_memory_layer_forward_propagation->state_activations_derivatives;

    Tensor<type, 1>& current_state_activations = long_short_term_memory_layer_forward_propagation->current_state_activations;
    Tensor<type, 1>& current_state_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_state_activations_derivatives;

    const Tensor<type, 2, RowMajor>& output_activations = long_short_term_memory_layer_forward_propagation->output_activations;
    const Tensor<type, 2, RowMajor>& output_activations_derivatives = long_short_term_memory_layer_forward_propagation->output_activations_derivatives;

    Tensor<type, 1>& current_output_activations = long_short_term_memory_layer_forward_propagation->current_output_activations;
    Tensor<type, 1>& current_output_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_output_activations_derivatives;

    const Tensor<type, 2, RowMajor>& cell_states = long_short_term_memory_layer_forward_propagation->cell_states;

    Tensor<type, 1>& previous_cell_states = long_short_term_memory_layer_forward_propagation->previous_cell_states;
    Tensor<type, 1>& current_cell_states = long_short_term_memory_layer_forward_propagation->current_cell_states;

    const Tensor<type, 2, RowMajor>& hidden_states = long_short_term_memory_layer_forward_propagation->hidden_states;
    const Tensor<type, 2, RowMajor>& hidden_states_activations_derivatives = long_short_term_memory_layer_forward_propagation->hidden_states_activations_derivatives;

    Tensor<type, 1>& previous_hidden_states = long_short_term_memory_layer_forward_propagation->previous_hidden_states;
    Tensor<type, 1>& current_hidden_states = long_short_term_memory_layer_forward_propagation->current_hidden_states;
    Tensor<type, 1>& current_hidden_states_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_hidden_states_activations_derivatives;

    // Back propagation

    Tensor<type, 1>& current_deltas = long_short_term_memory_back_propagation->current_deltas;

    Tensor<type, 2>& forget_combinations_weights_derivatives = long_short_term_memory_back_propagation->forget_combinations_weights_derivatives;
    Tensor<type, 2>& forget_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->forget_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& forget_combinations_biases_derivatives = long_short_term_memory_back_propagation->forget_combinations_biases_derivatives;

    Tensor<type, 2>& input_combinations_weights_derivatives = long_short_term_memory_back_propagation->input_combinations_weights_derivatives;
    Tensor<type, 2>& input_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->input_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& input_combinations_biases_derivatives = long_short_term_memory_back_propagation->input_combinations_biases_derivatives;

    Tensor<type, 2>& state_combinations_weights_derivatives = long_short_term_memory_back_propagation->state_combinations_weights_derivatives;
    Tensor<type, 2>& state_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->state_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& state_combinations_biases_derivatives = long_short_term_memory_back_propagation->state_combinations_biases_derivatives;

    Tensor<type, 2>& output_combinations_weights_derivatives = long_short_term_memory_back_propagation->output_combinations_weights_derivatives;
    Tensor<type, 2>& output_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->output_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& output_combinations_biases_derivatives = long_short_term_memory_back_propagation->output_combinations_biases_derivatives;

    Tensor<type, 2>& cell_states_weights_derivatives = long_short_term_memory_back_propagation->cell_states_weights_derivatives;
    Tensor<type, 2>& cell_states_recurrent_weights_derivatives = long_short_term_memory_back_propagation->cell_states_recurrent_weights_derivatives;
    Tensor<type, 2>& cell_states_biases_derivatives = long_short_term_memory_back_propagation->cell_states_biases_derivatives;

    Tensor<type, 2>& hidden_states_weights_derivatives = long_short_term_memory_back_propagation->hidden_states_weights_derivatives;
    Tensor<type, 2>& hidden_states_recurrent_weights_derivatives = long_short_term_memory_back_propagation->hidden_states_recurrent_weights_derivatives;
    Tensor<type, 2>& hidden_states_biases_derivatives = long_short_term_memory_back_propagation->hidden_states_biases_derivatives;

    Tensor<type, 1>& input_weights_derivatives = long_short_term_memory_back_propagation->input_weights_derivatives;
    Tensor<type, 1>& input_recurrent_weights_derivatives = long_short_term_memory_back_propagation->input_recurrent_weights_derivatives;
    Tensor<type, 1>& input_biases_derivatives = long_short_term_memory_back_propagation->input_biases_derivatives;

    input_weights_derivatives.setZero();
    input_recurrent_weights_derivatives.setZero();
    input_biases_derivatives.setZero();

    Index weight_index = 0;
    Index recurrent_weight_index = 0;
    Index input_index = 0;
    Index neuron_index = 0;

    for(Index sample = 0; sample < samples_number; sample++)
    {
        current_inputs.device(*thread_pool_device) = inputs.chip(sample, 0);

        get_row(current_forget_activations, forget_activations, sample);
        get_row(current_forget_activations_derivatives, forget_activations_derivatives, sample);

        get_row(current_input_activations, input_activations, sample);
        get_row(current_input_activations_derivatives, input_activations_derivatives, sample);

        get_row(current_state_activations, state_activations, sample);
        get_row(current_state_activations_derivatives, state_activations_derivatives, sample);

        get_row(current_output_activations, output_activations, sample);
        get_row(current_output_activations_derivatives, output_activations_derivatives, sample);

        get_row(current_cell_states, cell_states, sample);

        get_row(current_hidden_states, hidden_states, sample);
        get_row(current_hidden_states_activations_derivatives, hidden_states_activations_derivatives, sample);

        current_deltas.device(*thread_pool_device) = deltas.chip(sample, 0);

        calculate_activations(current_cell_states);

        // INPUT PARAMETERS DERIVATIVES

        if(sample % time_steps == 0)
        {
            //set_derivatives_zero();

            previous_cell_states.setZero();
            previous_hidden_states.setZero();
        }
        else
        {
            get_row(previous_cell_states, cell_states, sample - 1);
            get_row(previous_hidden_states, hidden_states, sample - 1);

            // Forget combinations derivatives

            forget_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(forget_recurrent_weights, A_B);

            forget_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(forget_recurrent_weights, A_B);

            forget_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(forget_recurrent_weights, A_B);

            multiply_rows(forget_combinations_weights_derivatives, current_forget_activations_derivatives);

            multiply_rows(forget_combinations_recurrent_weights_derivatives, current_forget_activations_derivatives);

            multiply_rows(forget_combinations_biases_derivatives, current_forget_activations_derivatives);

            // Input combinations derivatives

            input_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(input_recurrent_weights, A_B);

            input_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(input_recurrent_weights, A_B);

            input_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(input_recurrent_weights, A_B);

            // State combinations derivatives

            state_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(state_recurrent_weights, A_B);

            state_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(state_recurrent_weights, A_B);

            state_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(state_recurrent_weights, A_B);

            multiply_rows(state_combinations_weights_derivatives, current_state_activations_derivatives);

            multiply_rows(state_combinations_recurrent_weights_derivatives, current_state_activations_derivatives);

            multiply_rows(state_combinations_biases_derivatives, current_state_activations_derivatives);

            // Output combinations derivatives

            output_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(output_recurrent_weights, A_B);

            output_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(output_recurrent_weights, A_B);

            output_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(output_recurrent_weights, A_B);

            multiply_rows(output_combinations_weights_derivatives, current_output_activations_derivatives);

            multiply_rows(output_combinations_recurrent_weights_derivatives, current_output_activations_derivatives);

            multiply_rows(output_combinations_biases_derivatives, current_output_activations_derivatives);
        }

        weight_index = 0;
        recurrent_weight_index = 0;

        input_index = 0;
        neuron_index = 0;

        for(Index i = 0; i < parameters_number; i++)
        {
            const type current_input = current_inputs(input_index);
            const type previous_hidden_state_activation = previous_hidden_states(neuron_index);

            input_combinations_weights_derivatives(i, weight_index) += current_input;
            input_combinations_recurrent_weights_derivatives(i, recurrent_weight_index) += previous_hidden_state_activation;
            input_combinations_biases_derivatives(i, i) += type(1.0);

            input_index++;
            neuron_index++;

            if(input_index == inputs_number)
            {
                input_index = 0;
                weight_index++;
            }

            if(neuron_index == neurons_number)
            {
                neuron_index = 0;
                recurrent_weight_index++;
            }
        }

        // Input weights derivatives

        multiply_rows(cell_states_weights_derivatives, current_forget_activations);

        multiply_rows(input_combinations_weights_derivatives, current_input_activations_derivatives * current_state_activations);

        cell_states_weights_derivatives.device(*thread_pool_device) += input_combinations_weights_derivatives;

        multiply_rows(state_combinations_weights_derivatives, current_input_activations);

        cell_states_weights_derivatives.device(*thread_pool_device) += state_combinations_weights_derivatives;

        multiply_rows(forget_combinations_weights_derivatives, previous_cell_states);

        cell_states_weights_derivatives.device(*thread_pool_device) += forget_combinations_weights_derivatives;

        copy(cell_states_weights_derivatives.data(),
             cell_states_weights_derivatives.data() + cell_states_weights_derivatives.size(),
             hidden_states_weights_derivatives.data());

        multiply_rows(hidden_states_weights_derivatives, current_output_activations * current_hidden_states_activations_derivatives);

        multiply_rows(output_combinations_weights_derivatives, current_cell_states);

        hidden_states_weights_derivatives.device(*thread_pool_device) += output_combinations_weights_derivatives;

        input_weights_derivatives.device(*thread_pool_device) += hidden_states_weights_derivatives.contract(current_deltas, A_B);

        // Input recurrent weights derivatives

        if(sample % time_steps != 0)
        {
            multiply_rows(cell_states_recurrent_weights_derivatives, current_forget_activations);

            multiply_rows(input_combinations_recurrent_weights_derivatives, current_input_activations_derivatives * current_state_activations);

            cell_states_recurrent_weights_derivatives.device(*thread_pool_device) += input_combinations_recurrent_weights_derivatives;

            multiply_rows(state_combinations_recurrent_weights_derivatives, current_input_activations);

            cell_states_recurrent_weights_derivatives.device(*thread_pool_device) += state_combinations_recurrent_weights_derivatives;

            multiply_rows(forget_combinations_recurrent_weights_derivatives, previous_cell_states);

            cell_states_recurrent_weights_derivatives.device(*thread_pool_device) += forget_combinations_recurrent_weights_derivatives;

            copy(cell_states_recurrent_weights_derivatives.data(),
                 cell_states_recurrent_weights_derivatives.data() + cell_states_recurrent_weights_derivatives.size(),
                 hidden_states_recurrent_weights_derivatives.data());

            multiply_rows(hidden_states_recurrent_weights_derivatives, current_output_activations * current_hidden_states_activations_derivatives);

            multiply_rows(output_combinations_recurrent_weights_derivatives, current_cell_states);

            hidden_states_recurrent_weights_derivatives.device(*thread_pool_device) += output_combinations_recurrent_weights_derivatives;
        }

        input_recurrent_weights_derivatives.device(*thread_pool_device) += hidden_states_recurrent_weights_derivatives.contract(current_deltas, A_B);

        // Input biases derivatives

        multiply_rows(cell_states_biases_derivatives, current_forget_activations);

        multiply_rows(input_combinations_biases_derivatives, current_input_activations_derivatives * current_state_activations);

        cell_states_biases_derivatives.device(*thread_pool_device) += input_combinations_biases_derivatives;

        multiply_rows(state_combinations_biases_derivatives, current_input_activations);

        cell_states_biases_derivatives.device(*thread_pool_device) += state_combinations_biases_derivatives;

        multiply_rows(forget_combinations_biases_derivatives, previous_cell_states);

        cell_states_biases_derivatives.device(*thread_pool_device) += forget_combinations_biases_derivatives;

        copy(cell_states_biases_derivatives.data(),
             cell_states_biases_derivatives.data() + cell_states_biases_derivatives.size(),
             hidden_states_biases_derivatives.data());

        multiply_rows(hidden_states_biases_derivatives, current_output_activations * current_hidden_states_activations_derivatives);

        multiply_rows(output_combinations_biases_derivatives, current_cell_states);

        hidden_states_biases_derivatives.device(*thread_pool_device) += output_combinations_biases_derivatives;

        input_biases_derivatives.device(*thread_pool_device) += hidden_states_biases_derivatives.contract(current_deltas, A_B);
    }
*/
}


void LongShortTermMemoryLayer::calculate_state_parameter_derivatives(
        const Tensor<type, 2>& inputs,
        const Tensor<type, 2>& deltas,
        unique_ptr<LongShortTermMemoryLayerForwardPropagation>& long_short_term_memory_layer_forward_propagation,
        unique_ptr<LongShortTermMemoryLayerBackPropagation>& long_short_term_memory_back_propagation) const
{
/*
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = inputs_number * neurons_number;
    const Index samples_number = inputs.dimension(0);

    // Forward propagation

    Tensor<type, 1>& current_inputs = long_short_term_memory_layer_forward_propagation->current_inputs;

    const Tensor<type, 2, RowMajor>& forget_activations = long_short_term_memory_layer_forward_propagation->forget_activations;
    const Tensor<type, 2, RowMajor>& forget_activations_derivatives = long_short_term_memory_layer_forward_propagation->forget_activations_derivatives;

    Tensor<type, 1>& current_forget_activations = long_short_term_memory_layer_forward_propagation->current_forget_activations;
    Tensor<type, 1>& current_forget_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives;

    const Tensor<type, 2, RowMajor>& input_activations = long_short_term_memory_layer_forward_propagation->input_activations;
    const Tensor<type, 2, RowMajor>& input_activations_derivatives = long_short_term_memory_layer_forward_propagation->input_activations_derivatives;

    Tensor<type, 1>& current_input_activations = long_short_term_memory_layer_forward_propagation->current_input_activations;
    Tensor<type, 1>& current_input_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_input_activations_derivatives;

    const Tensor<type, 2, RowMajor>& state_activations = long_short_term_memory_layer_forward_propagation->state_activations;
    const Tensor<type, 2, RowMajor>& state_activations_derivatives = long_short_term_memory_layer_forward_propagation->state_activations_derivatives;

    Tensor<type, 1>& current_state_activations = long_short_term_memory_layer_forward_propagation->current_state_activations;
    Tensor<type, 1>& current_state_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_state_activations_derivatives;

    const Tensor<type, 2, RowMajor>& output_activations = long_short_term_memory_layer_forward_propagation->output_activations;
    const Tensor<type, 2, RowMajor>& output_activations_derivatives = long_short_term_memory_layer_forward_propagation->output_activations_derivatives;

    Tensor<type, 1>& current_output_activations = long_short_term_memory_layer_forward_propagation->current_output_activations;
    Tensor<type, 1>& current_output_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_output_activations_derivatives;

    const Tensor<type, 2, RowMajor>& cell_states = long_short_term_memory_layer_forward_propagation->cell_states;

    Tensor<type, 1>& previous_cell_states = long_short_term_memory_layer_forward_propagation->previous_cell_states;
    Tensor<type, 1>& current_cell_states = long_short_term_memory_layer_forward_propagation->current_cell_states;

    const Tensor<type, 2, RowMajor>& hidden_states = long_short_term_memory_layer_forward_propagation->hidden_states;
    const Tensor<type, 2, RowMajor>& hidden_states_activations_derivatives = long_short_term_memory_layer_forward_propagation->hidden_states_activations_derivatives;

    Tensor<type, 1>& previous_hidden_states = long_short_term_memory_layer_forward_propagation->previous_hidden_states;
    Tensor<type, 1>& current_hidden_states = long_short_term_memory_layer_forward_propagation->current_hidden_states;
    Tensor<type, 1>& current_hidden_states_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_hidden_states_activations_derivatives;

    // Back propagation

    Tensor<type, 1>& current_deltas = long_short_term_memory_back_propagation->current_deltas;

    Tensor<type, 2>& forget_combinations_weights_derivatives = long_short_term_memory_back_propagation->forget_combinations_weights_derivatives;
    Tensor<type, 2>& forget_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->forget_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& forget_combinations_biases_derivatives = long_short_term_memory_back_propagation->forget_combinations_biases_derivatives;

    Tensor<type, 2>& input_combinations_weights_derivatives = long_short_term_memory_back_propagation->input_combinations_weights_derivatives;
    Tensor<type, 2>& input_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->input_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& input_combinations_biases_derivatives = long_short_term_memory_back_propagation->input_combinations_biases_derivatives;

    Tensor<type, 2>& state_combinations_weights_derivatives = long_short_term_memory_back_propagation->state_combinations_weights_derivatives;
    Tensor<type, 2>& state_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->state_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& state_combinations_biases_derivatives = long_short_term_memory_back_propagation->state_combinations_biases_derivatives;

    Tensor<type, 2>& output_combinations_weights_derivatives = long_short_term_memory_back_propagation->output_combinations_weights_derivatives;
    Tensor<type, 2>& output_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->output_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& output_combinations_biases_derivatives = long_short_term_memory_back_propagation->output_combinations_biases_derivatives;

    Tensor<type, 2>& cell_states_weights_derivatives = long_short_term_memory_back_propagation->cell_states_weights_derivatives;
    Tensor<type, 2>& cell_states_recurrent_weights_derivatives = long_short_term_memory_back_propagation->cell_states_recurrent_weights_derivatives;
    Tensor<type, 2>& cell_states_biases_derivatives = long_short_term_memory_back_propagation->cell_states_biases_derivatives;

    Tensor<type, 2>& hidden_states_weights_derivatives = long_short_term_memory_back_propagation->hidden_states_weights_derivatives;
    Tensor<type, 2>& hidden_states_recurrent_weights_derivatives = long_short_term_memory_back_propagation->hidden_states_recurrent_weights_derivatives;
    Tensor<type, 2>& hidden_states_biases_derivatives = long_short_term_memory_back_propagation->hidden_states_biases_derivatives;

    Tensor<type, 1>& state_weights_derivatives = long_short_term_memory_back_propagation->state_weights_derivatives;
    Tensor<type, 1>& state_recurrent_weights_derivatives = long_short_term_memory_back_propagation->state_recurrent_weights_derivatives;
    Tensor<type, 1>& state_biases_derivatives = long_short_term_memory_back_propagation->state_biases_derivatives;
    state_weights_derivatives.setZero();
    state_recurrent_weights_derivatives.setZero();
    state_biases_derivatives.setZero();

    Index weight_index = 0;
    Index recurrent_weight_index = 0;
    Index input_index = 0;
    Index neuron_index = 0;

    for(Index sample = 0; sample < samples_number; sample++)
    {
        current_inputs.device(*thread_pool_device) = inputs.chip(sample, 0);

        get_row(current_forget_activations, forget_activations, sample);
        get_row(current_forget_activations_derivatives, forget_activations_derivatives, sample);

        get_row(current_input_activations, input_activations, sample);
        get_row(current_input_activations_derivatives, input_activations_derivatives, sample);

        get_row(current_state_activations, state_activations, sample);
        get_row(current_state_activations_derivatives, state_activations_derivatives, sample);

        get_row(current_output_activations, output_activations, sample);
        get_row(current_output_activations_derivatives, output_activations_derivatives, sample);

        get_row(current_cell_states, cell_states, sample);

        get_row(current_hidden_states, hidden_states, sample);
        get_row(current_hidden_states_activations_derivatives, hidden_states_activations_derivatives, sample);

        current_deltas.device(*thread_pool_device) = deltas.chip(sample, 0);

        calculate_activations(current_cell_states);

        // STATE PARAMETERS DERIVATIVES

        if(sample % time_steps == 0)
        {
            //set_derivatives_zero();

            previous_cell_states.setZero();
            previous_hidden_states.setZero();
        }
        else
        {
            get_row(previous_cell_states, cell_states, sample - 1);
            get_row(previous_hidden_states, hidden_states, sample - 1);

            // Forget combinations derivatives

            forget_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(forget_recurrent_weights, A_B);

            forget_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(forget_recurrent_weights, A_B);

            forget_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(forget_recurrent_weights, A_B);

            multiply_rows(forget_combinations_weights_derivatives, current_forget_activations_derivatives);

            multiply_rows(forget_combinations_recurrent_weights_derivatives, current_forget_activations_derivatives);

            multiply_rows(forget_combinations_biases_derivatives, current_forget_activations_derivatives);

            // Input combinations derivatives

            input_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(input_recurrent_weights, A_B);

            input_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(input_recurrent_weights, A_B);

            input_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(input_recurrent_weights, A_B);

            multiply_rows(input_combinations_weights_derivatives, current_input_activations_derivatives);

            multiply_rows(input_combinations_recurrent_weights_derivatives, current_input_activations_derivatives);

            multiply_rows(input_combinations_biases_derivatives, current_input_activations_derivatives);

            // State combinations derivatives

            state_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(state_recurrent_weights, A_B);

            state_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(state_recurrent_weights, A_B);

            state_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(state_recurrent_weights, A_B);

            // Output combinations derivatives

            output_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(output_recurrent_weights, A_B);

            output_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(output_recurrent_weights, A_B);

            output_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(output_recurrent_weights, A_B);

            multiply_rows(output_combinations_weights_derivatives, current_output_activations_derivatives);

            multiply_rows(output_combinations_recurrent_weights_derivatives, current_output_activations_derivatives);

            multiply_rows(output_combinations_biases_derivatives, current_output_activations_derivatives);
        }

        weight_index = 0;
        recurrent_weight_index = 0;

        input_index = 0;
        neuron_index = 0;

        for(Index i = 0; i < parameters_number; i++)
        {
            const type current_input = current_inputs(input_index);
            const type previous_hidden_state_activation = previous_hidden_states(neuron_index);

            state_combinations_weights_derivatives(i, weight_index) += current_input;
            state_combinations_recurrent_weights_derivatives(i, recurrent_weight_index) += previous_hidden_state_activation;
            state_combinations_biases_derivatives(i, i) += type(1.0);

            input_index++;
            neuron_index++;

            if(input_index == inputs_number)
            {
                input_index = 0;
                weight_index++;
            }

            if(neuron_index == neurons_number)
            {
                neuron_index = 0;
                recurrent_weight_index++;
            }
        }

        // State weights derivatives

        multiply_rows(cell_states_weights_derivatives, current_forget_activations);

        multiply_rows(input_combinations_weights_derivatives, current_state_activations);

        cell_states_weights_derivatives.device(*thread_pool_device) += input_combinations_weights_derivatives;

        multiply_rows(state_combinations_weights_derivatives, current_input_activations * current_state_activations_derivatives);

        cell_states_weights_derivatives.device(*thread_pool_device) += state_combinations_weights_derivatives;

        multiply_rows(forget_combinations_weights_derivatives, previous_cell_states);

        cell_states_weights_derivatives.device(*thread_pool_device) += forget_combinations_weights_derivatives;

        copy(cell_states_weights_derivatives.data(),
             cell_states_weights_derivatives.data() + cell_states_weights_derivatives.size(),
             hidden_states_weights_derivatives.data());

        multiply_rows(hidden_states_weights_derivatives, current_output_activations * current_hidden_states_activations_derivatives);

        multiply_rows(output_combinations_weights_derivatives, current_cell_states);

        hidden_states_weights_derivatives.device(*thread_pool_device) += output_combinations_weights_derivatives;

        state_weights_derivatives.device(*thread_pool_device) += hidden_states_weights_derivatives.contract(current_deltas, A_B);

        // State recurrent weights derivatives

        if(sample % time_steps != 0)
        {
            multiply_rows(cell_states_recurrent_weights_derivatives, current_forget_activations);

            multiply_rows(input_combinations_recurrent_weights_derivatives, current_state_activations);

            cell_states_recurrent_weights_derivatives.device(*thread_pool_device) += input_combinations_recurrent_weights_derivatives;

            multiply_rows(state_combinations_recurrent_weights_derivatives, current_input_activations * current_state_activations_derivatives);

            cell_states_recurrent_weights_derivatives.device(*thread_pool_device) += state_combinations_recurrent_weights_derivatives;

            multiply_rows(forget_combinations_recurrent_weights_derivatives, previous_cell_states);

            cell_states_recurrent_weights_derivatives.device(*thread_pool_device) += forget_combinations_recurrent_weights_derivatives;

            copy(cell_states_recurrent_weights_derivatives.data(),
                 cell_states_recurrent_weights_derivatives.data() + cell_states_recurrent_weights_derivatives.size(),
                 hidden_states_recurrent_weights_derivatives.data());

            multiply_rows(hidden_states_recurrent_weights_derivatives, current_output_activations * current_hidden_states_activations_derivatives);

            multiply_rows(output_combinations_recurrent_weights_derivatives, current_cell_states);

            hidden_states_recurrent_weights_derivatives.device(*thread_pool_device) += output_combinations_recurrent_weights_derivatives;
        }

        state_recurrent_weights_derivatives.device(*thread_pool_device) += hidden_states_recurrent_weights_derivatives.contract(current_deltas, A_B);

        // State biases derivatives

        multiply_rows(cell_states_biases_derivatives, current_forget_activations);

        multiply_rows(input_combinations_biases_derivatives, current_state_activations);

        cell_states_biases_derivatives.device(*thread_pool_device) += input_combinations_biases_derivatives;

        multiply_rows(state_combinations_biases_derivatives, current_input_activations * current_state_activations_derivatives);

        cell_states_biases_derivatives.device(*thread_pool_device) += state_combinations_biases_derivatives;

        multiply_rows(forget_combinations_biases_derivatives, previous_cell_states);

        cell_states_biases_derivatives.device(*thread_pool_device) += forget_combinations_biases_derivatives;

        copy(cell_states_biases_derivatives.data(),
             cell_states_biases_derivatives.data() + cell_states_biases_derivatives.size(),
             hidden_states_biases_derivatives.data());

        multiply_rows(hidden_states_biases_derivatives, current_output_activations * current_hidden_states_activations_derivatives);

        multiply_rows(output_combinations_biases_derivatives, current_cell_states);

        hidden_states_biases_derivatives.device(*thread_pool_device) += output_combinations_biases_derivatives;

        state_biases_derivatives.device(*thread_pool_device) += hidden_states_biases_derivatives.contract(current_deltas, A_B);
    }
*/
}


void LongShortTermMemoryLayer::calculate_output_parameter_derivatives(
    const Tensor<type, 2>& inputs,
    const Tensor<type, 2>& deltas,
    unique_ptr<LongShortTermMemoryLayerForwardPropagation>& long_short_term_memory_layer_forward_propagation,
    unique_ptr<LongShortTermMemoryLayerBackPropagation>& long_short_term_memory_back_propagation) const
{
/*
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = inputs_number * neurons_number;
    const Index samples_number = inputs.dimension(0);

    // Forward propagation

    Tensor<type, 1>& current_inputs = long_short_term_memory_layer_forward_propagation->current_inputs;

    const Tensor<type, 2, RowMajor>& forget_activations = long_short_term_memory_layer_forward_propagation->forget_activations;
    const Tensor<type, 2, RowMajor>& forget_activations_derivatives = long_short_term_memory_layer_forward_propagation->forget_activations_derivatives;

    Tensor<type, 1>& current_forget_activations = long_short_term_memory_layer_forward_propagation->current_forget_activations;
    Tensor<type, 1>& current_forget_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives;

    const Tensor<type, 2, RowMajor>& input_activations = long_short_term_memory_layer_forward_propagation->input_activations;
    const Tensor<type, 2, RowMajor>& input_activations_derivatives = long_short_term_memory_layer_forward_propagation->input_activations_derivatives;

    Tensor<type, 1>& current_input_activations = long_short_term_memory_layer_forward_propagation->current_input_activations;
    Tensor<type, 1>& current_input_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_input_activations_derivatives;

    const Tensor<type, 2, RowMajor>& state_activations = long_short_term_memory_layer_forward_propagation->state_activations;
    const Tensor<type, 2, RowMajor>& state_activations_derivatives = long_short_term_memory_layer_forward_propagation->state_activations_derivatives;

    Tensor<type, 1>& current_state_activations = long_short_term_memory_layer_forward_propagation->current_state_activations;
    Tensor<type, 1>& current_state_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_state_activations_derivatives;

    const Tensor<type, 2, RowMajor>& output_activations = long_short_term_memory_layer_forward_propagation->output_activations;
    const Tensor<type, 2, RowMajor>& output_activations_derivatives = long_short_term_memory_layer_forward_propagation->output_activations_derivatives;

    Tensor<type, 1>& current_output_activations = long_short_term_memory_layer_forward_propagation->current_output_activations;
    Tensor<type, 1>& current_output_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_output_activations_derivatives;

    const Tensor<type, 2, RowMajor>& cell_states = long_short_term_memory_layer_forward_propagation->cell_states;

    Tensor<type, 1>& previous_cell_states = long_short_term_memory_layer_forward_propagation->previous_cell_states;
    Tensor<type, 1>& current_cell_states = long_short_term_memory_layer_forward_propagation->current_cell_states;

    const Tensor<type, 2, RowMajor>& hidden_states = long_short_term_memory_layer_forward_propagation->hidden_states;
    const Tensor<type, 2, RowMajor>& hidden_states_activations_derivatives = long_short_term_memory_layer_forward_propagation->hidden_states_activations_derivatives;

    Tensor<type, 1>& previous_hidden_states = long_short_term_memory_layer_forward_propagation->previous_hidden_states;
    Tensor<type, 1>& current_hidden_states = long_short_term_memory_layer_forward_propagation->current_hidden_states;
    Tensor<type, 1>& current_hidden_states_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_hidden_states_activations_derivatives;

    // Back propagation

    Tensor<type, 1>& current_deltas = long_short_term_memory_back_propagation->current_deltas;

    Tensor<type, 2>& forget_combinations_weights_derivatives = long_short_term_memory_back_propagation->forget_combinations_weights_derivatives;
    Tensor<type, 2>& forget_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->forget_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& forget_combinations_biases_derivatives = long_short_term_memory_back_propagation->forget_combinations_biases_derivatives;

    Tensor<type, 2>& input_combinations_weights_derivatives = long_short_term_memory_back_propagation->input_combinations_weights_derivatives;
    Tensor<type, 2>& input_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->input_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& input_combinations_biases_derivatives = long_short_term_memory_back_propagation->input_combinations_biases_derivatives;

    Tensor<type, 2>& state_combinations_weights_derivatives = long_short_term_memory_back_propagation->state_combinations_weights_derivatives;
    Tensor<type, 2>& state_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->state_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& state_combinations_biases_derivatives = long_short_term_memory_back_propagation->state_combinations_biases_derivatives;

    Tensor<type, 2>& output_combinations_weights_derivatives = long_short_term_memory_back_propagation->output_combinations_weights_derivatives;
    Tensor<type, 2>& output_combinations_recurrent_weights_derivatives = long_short_term_memory_back_propagation->output_combinations_recurrent_weights_derivatives;
    Tensor<type, 2>& output_combinations_biases_derivatives = long_short_term_memory_back_propagation->output_combinations_biases_derivatives;

    Tensor<type, 2>& cell_states_weights_derivatives = long_short_term_memory_back_propagation->cell_states_weights_derivatives;
    Tensor<type, 2>& cell_states_recurrent_weights_derivatives = long_short_term_memory_back_propagation->cell_states_recurrent_weights_derivatives;
    Tensor<type, 2>& cell_states_biases_derivatives = long_short_term_memory_back_propagation->cell_states_biases_derivatives;

    Tensor<type, 2>& hidden_states_weights_derivatives = long_short_term_memory_back_propagation->hidden_states_weights_derivatives;
    Tensor<type, 2>& hidden_states_recurrent_weights_derivatives = long_short_term_memory_back_propagation->hidden_states_recurrent_weights_derivatives;
    Tensor<type, 2>& hidden_states_biases_derivatives = long_short_term_memory_back_propagation->hidden_states_biases_derivatives;

    Tensor<type, 1>& output_weights_derivatives = long_short_term_memory_back_propagation->output_weights_derivatives;
    Tensor<type, 1>& output_recurrent_weights_derivatives = long_short_term_memory_back_propagation->output_recurrent_weights_derivatives;
    Tensor<type, 1>& output_biases_derivatives = long_short_term_memory_back_propagation->output_biases_derivatives;
    output_weights_derivatives.setZero();
    output_recurrent_weights_derivatives.setZero();
    output_biases_derivatives.setZero();

    Index weight_index = 0;
    Index recurrent_weight_index = 0;
    Index input_index = 0;
    Index neuron_index = 0;

    for(Index sample = 0; sample < samples_number; sample++)
    {
        current_inputs.device(*thread_pool_device) = inputs.chip(sample, 0);

        get_row(current_forget_activations, forget_activations, sample);
        get_row(current_forget_activations_derivatives, forget_activations_derivatives, sample);

        get_row(current_input_activations, input_activations, sample);
        get_row(current_input_activations_derivatives, input_activations_derivatives, sample);

        get_row(current_state_activations, state_activations, sample);
        get_row(current_state_activations_derivatives, state_activations_derivatives, sample);

        get_row(current_output_activations, output_activations, sample);
        get_row(current_output_activations_derivatives, output_activations_derivatives, sample);

        get_row(current_cell_states, cell_states, sample);

        get_row(current_hidden_states, hidden_states, sample);
        get_row(current_hidden_states_activations_derivatives, hidden_states_activations_derivatives, sample);

        current_deltas.device(*thread_pool_device) = deltas.chip(sample, 0);

        calculate_activations(current_cell_states);

        // OUTPUT PARAMETERS DERIVATIVES

        if(sample % time_steps == 0)
        {
            //set_derivatives_zero();

            previous_cell_states.setZero();
            previous_hidden_states.setZero();
        }
        else
        {
            get_row(previous_cell_states, cell_states, sample - 1);
            get_row(previous_hidden_states, hidden_states, sample - 1);

            // Forget combinations derivatives

            forget_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(forget_recurrent_weights, A_B);

            forget_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(forget_recurrent_weights, A_B);

            forget_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(forget_recurrent_weights, A_B);

            multiply_rows(forget_combinations_weights_derivatives, current_forget_activations_derivatives);

            multiply_rows(forget_combinations_recurrent_weights_derivatives, current_forget_activations_derivatives);

            multiply_rows(forget_combinations_biases_derivatives, current_forget_activations_derivatives);

            // Input combinations derivatives

            input_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(input_recurrent_weights, A_B);

            input_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(input_recurrent_weights, A_B);

            input_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(input_recurrent_weights, A_B);

            multiply_rows(input_combinations_weights_derivatives, current_input_activations_derivatives);

            multiply_rows(input_combinations_recurrent_weights_derivatives, current_input_activations_derivatives);

            multiply_rows(input_combinations_biases_derivatives, current_input_activations_derivatives);

            // State combinations derivatives

            state_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(state_recurrent_weights, A_B);

            state_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(state_recurrent_weights, A_B);

            state_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(state_recurrent_weights, A_B);

            multiply_rows(state_combinations_weights_derivatives, current_state_activations_derivatives);

            multiply_rows(state_combinations_recurrent_weights_derivatives, current_state_activations_derivatives);

            multiply_rows(state_combinations_biases_derivatives, current_state_activations_derivatives);

            // Output combinations derivatives

            output_combinations_weights_derivatives.device(*thread_pool_device)
                = hidden_states_weights_derivatives.contract(output_recurrent_weights, A_B);

            output_combinations_recurrent_weights_derivatives.device(*thread_pool_device)
                = hidden_states_recurrent_weights_derivatives.contract(output_recurrent_weights, A_B);

            output_combinations_biases_derivatives.device(*thread_pool_device)
                = hidden_states_biases_derivatives.contract(output_recurrent_weights, A_B);
        }

        weight_index = 0;
        recurrent_weight_index = 0;

        input_index = 0;
        neuron_index = 0;

        for(Index i = 0; i < parameters_number; i++)
        {
            const type current_input = current_inputs(input_index);
            const type previous_hidden_state_activation = previous_hidden_states(neuron_index);

            output_combinations_weights_derivatives(i, weight_index) += current_input;
            output_combinations_recurrent_weights_derivatives(i, recurrent_weight_index) += previous_hidden_state_activation;
            output_combinations_biases_derivatives(i, i) += type(1.0);

            input_index++;
            neuron_index++;

            if(input_index == inputs_number)
            {
                input_index = 0;
                weight_index++;
            }

            if(neuron_index == neurons_number)
            {
                neuron_index = 0;
                recurrent_weight_index++;
            }
        }

        // Output weights derivatives

        multiply_rows(cell_states_weights_derivatives, current_forget_activations);

        multiply_rows(input_combinations_weights_derivatives, current_state_activations);

        cell_states_weights_derivatives.device(*thread_pool_device) += input_combinations_weights_derivatives;

        multiply_rows(state_combinations_weights_derivatives, current_input_activations);

        cell_states_weights_derivatives.device(*thread_pool_device) += state_combinations_weights_derivatives;

        multiply_rows(forget_combinations_weights_derivatives, previous_cell_states);

        cell_states_weights_derivatives.device(*thread_pool_device) += forget_combinations_weights_derivatives;

        copy(cell_states_weights_derivatives.data(),
             cell_states_weights_derivatives.data() + cell_states_weights_derivatives.size(),
             hidden_states_weights_derivatives.data());

        multiply_rows(hidden_states_weights_derivatives, current_output_activations * current_hidden_states_activations_derivatives);

        multiply_rows(output_combinations_weights_derivatives, current_output_activations_derivatives * current_cell_states);

        hidden_states_weights_derivatives.device(*thread_pool_device) += output_combinations_weights_derivatives;

        output_weights_derivatives.device(*thread_pool_device) += hidden_states_weights_derivatives.contract(current_deltas, A_B);

        // Output recurrent weights derivatives

        if(sample % time_steps != 0)
        {
            multiply_rows(cell_states_recurrent_weights_derivatives, current_forget_activations);

            multiply_rows(input_combinations_recurrent_weights_derivatives, current_state_activations);

            cell_states_recurrent_weights_derivatives.device(*thread_pool_device) += input_combinations_recurrent_weights_derivatives;

            multiply_rows(state_combinations_recurrent_weights_derivatives, current_input_activations);

            cell_states_recurrent_weights_derivatives.device(*thread_pool_device) += state_combinations_recurrent_weights_derivatives;

            multiply_rows(forget_combinations_recurrent_weights_derivatives, previous_cell_states);

            cell_states_recurrent_weights_derivatives.device(*thread_pool_device) += forget_combinations_recurrent_weights_derivatives;

            copy(cell_states_recurrent_weights_derivatives.data(),
                 cell_states_recurrent_weights_derivatives.data() + cell_states_recurrent_weights_derivatives.size(),
                 hidden_states_recurrent_weights_derivatives.data());

            multiply_rows(hidden_states_recurrent_weights_derivatives, current_output_activations * current_hidden_states_activations_derivatives);

            multiply_rows(output_combinations_recurrent_weights_derivatives, current_output_activations_derivatives * current_cell_states);

            hidden_states_recurrent_weights_derivatives.device(*thread_pool_device) += output_combinations_recurrent_weights_derivatives;
        }

        output_recurrent_weights_derivatives.device(*thread_pool_device) += hidden_states_recurrent_weights_derivatives.contract(current_deltas, A_B);

        // Output biases derivatives

        multiply_rows(cell_states_biases_derivatives, current_forget_activations);

        multiply_rows(input_combinations_biases_derivatives, current_state_activations);

        cell_states_biases_derivatives.device(*thread_pool_device) += input_combinations_biases_derivatives;

        multiply_rows(state_combinations_biases_derivatives, current_input_activations);

        cell_states_biases_derivatives.device(*thread_pool_device) += state_combinations_biases_derivatives;

        multiply_rows(forget_combinations_biases_derivatives, previous_cell_states);

        cell_states_biases_derivatives.device(*thread_pool_device) += forget_combinations_biases_derivatives;

        copy(cell_states_biases_derivatives.data(),
             cell_states_biases_derivatives.data() + cell_states_biases_derivatives.size(),
             hidden_states_biases_derivatives.data());

        multiply_rows(hidden_states_biases_derivatives, current_output_activations * current_hidden_states_activations_derivatives);

        multiply_rows(output_combinations_biases_derivatives, current_output_activations_derivatives * current_cell_states);

        hidden_states_biases_derivatives.device(*thread_pool_device) += output_combinations_biases_derivatives;

        output_biases_derivatives.device(*thread_pool_device) += hidden_states_biases_derivatives.contract(current_deltas, A_B);
    }
*/
}


void LongShortTermMemoryLayer::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                               Index& index,
                                               Tensor<type, 1>& gradient) const
{    
    LongShortTermMemoryLayerBackPropagation* long_short_term_memory_back_propagation =
            static_cast<LongShortTermMemoryLayerBackPropagation*>(back_propagation.get());

    // Biases

    copy_to_vector(gradient, long_short_term_memory_back_propagation->forget_biases_derivatives, index);
    copy_to_vector(gradient, long_short_term_memory_back_propagation->input_biases_derivatives, index);
    copy_to_vector(gradient, long_short_term_memory_back_propagation->state_biases_derivatives, index);
    copy_to_vector(gradient, long_short_term_memory_back_propagation->output_biases_derivatives, index);

    // Weights

    copy_to_vector(gradient, long_short_term_memory_back_propagation->forget_weights_derivatives, index);
    copy_to_vector(gradient, long_short_term_memory_back_propagation->input_weights_derivatives, index);
    copy_to_vector(gradient, long_short_term_memory_back_propagation->state_weights_derivatives, index);
    copy_to_vector(gradient, long_short_term_memory_back_propagation->output_weights_derivatives, index);

    // Recurrent weights

    copy_to_vector(gradient, long_short_term_memory_back_propagation->forget_recurrent_weights_derivatives, index);
    copy_to_vector(gradient, long_short_term_memory_back_propagation->input_recurrent_weights_derivatives, index);
    copy_to_vector(gradient, long_short_term_memory_back_propagation->state_recurrent_weights_derivatives, index);
    copy_to_vector(gradient, long_short_term_memory_back_propagation->output_recurrent_weights_derivatives, index);
}


string LongShortTermMemoryLayer::get_expression(const vector<string>& input_names, const vector<string>& output_names) const
{
    ostringstream buffer;

/*
    const Index neurons_number = get_neurons_number();
    const Index inputs_number = get_inputs_number();

    // Forget gate

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "forget_gate_" << to_string(i) << " = " << write_recurrent_activation_function_expression() << " (" << forget_biases[i] << " + ";

        for(Index j = 0; j < inputs_number; j++)
            buffer << input_names[j] << " * (" << forget_weights(j,i) << ") + ";

        for(Index k = 0; k < neurons_number-1; k++)
            buffer << "hidden_state_" << to_string(k) << "(t-1) * (" << forget_recurrent_weights(k,i) << ") + ";

        buffer << "hidden_state_" << to_string(neurons_number-1) << "(t-1) * (" << forget_recurrent_weights(neurons_number-1,i) << "));\n";
    }

    // Input gate

    for(Index i = 0; i < neurons_number; i++)
    {
       buffer << "input_gate_" << to_string(i) << " = " << write_recurrent_activation_function_expression() << " (" << input_biases[i] << " + ";

       for(Index j = 0; j < inputs_number; j++)
           buffer << input_names[j] << " * (" << input_weights(j,i) << ") + ";

       for(Index k = 0; k < neurons_number-1; k++)
           buffer << "hidden_state_" << to_string(k) << "(t-1) * (" << input_recurrent_weights(k,i) << ") + ";

       buffer << "hidden_state_" << to_string(neurons_number-1) << "(t-1) * (" << input_recurrent_weights(neurons_number-1,i) << "));\n";
    }

    // State gate
    for(Index i = 0; i < neurons_number; i++)
    {
       buffer << "state_gate_" << to_string(i) << " = " << get_activation_function_string_expression() << " (" << state_biases[i] << " + ";

       for(Index j = 0; j < inputs_number; j++)
           buffer << input_names[j] << " * (" << state_weights(j,i) << ") + ";

       for(Index k = 0; k < neurons_number-1; k++)
           buffer << "hidden_state_" << to_string(k) << "(t-1) * (" << state_recurrent_weights(k,i) << ") + ";

       buffer << "hidden_state_" << to_string(neurons_number-1) << "(t-1) * (" << state_recurrent_weights(neurons_number-1,i) << "));\n";
    }

    // Output gate

    for(Index i = 0; i < neurons_number; i++)
    {
       buffer << "output_gate_" << to_string(i) << " = " << write_recurrent_activation_function_expression() << " (" << output_biases[i] << " + ";

       for(Index j = 0; j < inputs_number; j++)
           buffer << input_names[j] << " * (" << output_weights(j,i) << ") + ";

       for(Index k = 0; k < neurons_number-1; k++)
           buffer << "hidden_state_" << to_string(k) << "(t-1) * (" << output_recurrent_weights(k,i) << ") + ";

       buffer << "hidden_state_" << to_string(neurons_number-1) << "(t-1) * (" << output_recurrent_weights(neurons_number-1,i) << "));\n";
    }

    // Cell state
    for(Index i = 0; i < neurons_number; i++)
        buffer << "cell_states_" << to_string(i) << "(t) = forget_gate_" << to_string(i) << " * cell_states_" << to_string(i) << "(t-1)+input_gate_" << to_string(i) << " * state_gate_" << to_string(i) << ";\n";

    // Hidden state

    for(Index i = 0; i < neurons_number; i++)
        buffer << "hidden_state_" << to_string(i) << "(t) = output_gate_" << to_string(i) << " * " << get_activation_function_string_expression() << "(cell_states_" << to_string(i) << ");\n";

    // Output

    for(Index i = 0; i < neurons_number; i++)
       buffer << output_names[i] << " = " << "hidden_state_" << to_string(i) << "(t);\n";
*/
    return buffer.str();
}


void LongShortTermMemoryLayer::from_XML(const XMLDocument& document)
{
    const XMLElement* lstm_layer_element = document.FirstChildElement("LongShortTermMemory");

    if(!lstm_layer_element)
        throw runtime_error("LongShortTermMemory element is nullptr.\n");

    set_name(read_xml_string(lstm_layer_element, "Name"));
    set_input_dimensions({ read_xml_index(lstm_layer_element, "InputsNumber") });
    set_output_dimensions({ read_xml_index(lstm_layer_element, "NeuronsNumber") });
    set_timesteps(read_xml_index(lstm_layer_element, "TimeStep"));
    set_activation_function(read_xml_string(lstm_layer_element, "ActivationFunction"));
    set_recurrent_activation_function(read_xml_string(lstm_layer_element, "RecurrentActivationFunction"));
/*
    set_parameters(to_type_vector(read_xml_string(lstm_layer_element, "Parameters"), " "));
*/
}


void LongShortTermMemoryLayer::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("LongShortTermMemory");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "InputsNumber", to_string(get_input_dimensions()[0]));
    add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions()[0]));
    add_xml_element(printer, "TimeStep", to_string(get_timesteps()));
    add_xml_element(printer, "ActivationFunction", get_activation_function_string());
    add_xml_element(printer, "RecurrentActivationFunction", write_recurrent_activation_function());
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();
}


string LongShortTermMemoryLayer::write_recurrent_activation_function_expression() const
{
    switch(recurrent_activation_function)
    {
    case ActivationFunction::HyperbolicTangent:
        return "tanh";
    case ActivationFunction::Linear:
        return string();
    default:
        return write_recurrent_activation_function();
    }
}


string LongShortTermMemoryLayer::get_activation_function_string_expression() const
{
    switch(activation_function)
    {
    case ActivationFunction::HyperbolicTangent:
        return "tanh";
    case ActivationFunction::Linear:
        return string();
    default:
        return get_activation_function_string();
    }
}


LongShortTermMemoryLayerForwardPropagation::LongShortTermMemoryLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


pair<type*, dimensions> LongShortTermMemoryLayerForwardPropagation::get_outputs_pair() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();

    return {(type*)outputs.data(), {{samples_number, output_dimensions[0]}}};
}


void LongShortTermMemoryLayerForwardPropagation::set(const Index& new_samples_number, Layer* new_layer)
{
    layer = new_layer;

    const Index inputs_number = layer->get_input_dimensions()[0];
    const Index outputs_number = layer->get_outputs_number();

    samples_number = new_samples_number;

    outputs.resize(samples_number, outputs_number);

    previous_cell_states.resize(outputs_number);
    previous_hidden_states.resize(outputs_number);

    current_inputs.resize(inputs_number);

    current_forget_activations.resize(outputs_number);
    current_input_activations.resize(outputs_number);
    current_state_activations.resize(outputs_number);
    current_output_activations.resize(outputs_number);

    current_cell_states.resize(outputs_number);
    current_hidden_states.resize(outputs_number);

    current_forget_activations_derivatives.resize(outputs_number);
    current_input_activations_derivatives.resize(outputs_number);
    current_state_activations_derivatives.resize(outputs_number);
    current_output_activations_derivatives.resize(outputs_number);

    current_hidden_states_activations_derivatives.resize(outputs_number);

    forget_activations.resize(samples_number, outputs_number);
    input_activations.resize(samples_number, outputs_number);
    state_activations.resize(samples_number, outputs_number);
    output_activations.resize(samples_number, outputs_number);

    cell_states.resize(samples_number, outputs_number);
    hidden_states.resize(samples_number, outputs_number);

    forget_activations_derivatives.resize(samples_number, outputs_number);
    input_activations_derivatives.resize(samples_number, outputs_number);
    state_activations_derivatives.resize(samples_number, outputs_number);
    output_activations_derivatives.resize(samples_number, outputs_number);

    hidden_states_activations_derivatives.resize(samples_number, outputs_number);
}


void LongShortTermMemoryLayerForwardPropagation::print() const
{
    cout << "Current inputs: " << endl
         << current_inputs << endl
         << "Current input activations: " << endl
         << current_input_activations << endl
         << "Current input activations derivatives: " << endl
         << current_input_activations_derivatives << endl;
}


void LongShortTermMemoryLayerBackPropagation::set(const Index& new_samples_number, Layer* new_layer)
{
    layer = new_layer;

    samples_number = new_samples_number;
/*
    const Index neurons_number = layer->get_neurons_number();
    const Index inputs_number = layer->get_inputs_number();

    current_deltas.resize(neurons_number);

    forget_weights_derivatives.resize(inputs_number * neurons_number);
    input_weights_derivatives.resize(inputs_number * neurons_number);
    state_weights_derivatives.resize(inputs_number * neurons_number);
    output_weights_derivatives.resize(inputs_number * neurons_number);

    hidden_states_weights_derivatives.resize(inputs_number * neurons_number, neurons_number);
    cell_states_weights_derivatives.resize(inputs_number * neurons_number, neurons_number);

    forget_recurrent_weights_derivatives.resize(neurons_number * neurons_number);
    input_recurrent_weights_derivatives.resize(neurons_number * neurons_number);
    state_recurrent_weights_derivatives.resize(neurons_number * neurons_number);
    output_recurrent_weights_derivatives.resize(neurons_number * neurons_number);

    hidden_states_recurrent_weights_derivatives.resize(neurons_number * neurons_number, neurons_number);
    cell_states_recurrent_weights_derivatives.resize(neurons_number * neurons_number, neurons_number);

    forget_biases_derivatives.resize(neurons_number);
    input_biases_derivatives.resize(neurons_number);
    state_biases_derivatives.resize(neurons_number);
    output_biases_derivatives.resize(neurons_number);

    hidden_states_biases_derivatives.resize(neurons_number, neurons_number);
    cell_states_biases_derivatives.resize(neurons_number, neurons_number);

    input_combinations_weights_derivatives.resize(inputs_number * neurons_number, neurons_number);
    forget_combinations_weights_derivatives.resize(inputs_number * neurons_number, neurons_number);
    state_combinations_weights_derivatives.resize(inputs_number * neurons_number, neurons_number);
    output_combinations_weights_derivatives.resize(inputs_number * neurons_number, neurons_number);

    input_combinations_recurrent_weights_derivatives.resize(neurons_number * neurons_number, neurons_number);
    forget_combinations_recurrent_weights_derivatives.resize(neurons_number * neurons_number, neurons_number);
    state_combinations_recurrent_weights_derivatives.resize(neurons_number * neurons_number, neurons_number);
    output_combinations_recurrent_weights_derivatives.resize(neurons_number * neurons_number, neurons_number);

    input_combinations_biases_derivatives.resize(neurons_number, neurons_number);
    forget_combinations_biases_derivatives.resize(neurons_number, neurons_number);
    state_combinations_biases_derivatives.resize(neurons_number, neurons_number);
    output_combinations_biases_derivatives.resize(neurons_number, neurons_number);

    input_derivatives.resize(samples_number, inputs_number);
*/
}


LongShortTermMemoryLayerBackPropagation::LongShortTermMemoryLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer);
}


vector<pair<type*, dimensions>> LongShortTermMemoryLayerBackPropagation::get_input_derivative_pairs() const
{
    const dimensions input_dimensions = layer->get_input_dimensions();

    return {{(type*)(input_derivatives.data()), {samples_number, input_dimensions[0]}}};
}


void LongShortTermMemoryLayerBackPropagation::set_derivatives_zero()
{
    forget_combinations_weights_derivatives.setZero();
    input_combinations_weights_derivatives.setZero();
    output_combinations_weights_derivatives.setZero();
    state_combinations_weights_derivatives.setZero();

    forget_combinations_recurrent_weights_derivatives.setZero();
    input_combinations_recurrent_weights_derivatives.setZero();
    output_combinations_recurrent_weights_derivatives.setZero();
    state_combinations_recurrent_weights_derivatives.setZero();

    forget_combinations_biases_derivatives.setZero();
    input_combinations_biases_derivatives.setZero();
    output_combinations_biases_derivatives.setZero();
    state_combinations_biases_derivatives.setZero();

    cell_states_weights_derivatives.setZero();
    cell_states_recurrent_weights_derivatives.setZero();
    cell_states_biases_derivatives.setZero();

    hidden_states_weights_derivatives.setZero();
    hidden_states_recurrent_weights_derivatives.setZero();
    hidden_states_biases_derivatives.setZero();
}


void LongShortTermMemoryLayerBackPropagation::print() const
{
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
