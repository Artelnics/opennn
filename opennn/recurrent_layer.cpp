//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "recurrent_layer.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a empty layer object, with no neurons.
/// This constructor also initializes the rest of class members to their default values.

RecurrentLayer::RecurrentLayer() : Layer()
{
    set();

    layer_type = Recurrent;
}


/// Layer architecture constructor.
/// It creates a layer object with given numbers of inputs and neurons.
/// The parameters are initialized at random.
/// This constructor also initializes the rest of class members to their default values.
/// @param new_inputs_number Number of inputs in the layer.
/// @param new_neurons_number Number of neurons in the layer.

RecurrentLayer::RecurrentLayer(const Index& new_inputs_number, const Index& new_neurons_number) : Layer()
{
    set(new_inputs_number, new_neurons_number);

    layer_type = Recurrent;
}


/// Destructor.
/// This destructor does not delete any pointer.

RecurrentLayer::~RecurrentLayer()
{
}


/// Returns the number of inputs to the layer.

Index RecurrentLayer::get_inputs_number() const
{
    return input_weights.dimension(0);
}


/// Returns the size of the neurons vector.

Index RecurrentLayer::get_neurons_number() const
{
    return biases.size();
}


/// Returns the hidden states of the layer.

const Tensor<type, 1>& RecurrentLayer::get_hidden_states() const
{
    return hidden_states;
}


/// Returns the number of parameters (biases and weights) of the layer.

Index RecurrentLayer::get_parameters_number() const
{
    const Index neurons_number = get_neurons_number();
    const Index inputs_number = get_inputs_number();

    return  neurons_number * (1 + inputs_number + neurons_number);
}


Index RecurrentLayer::get_timesteps() const
{
    return timesteps;
}


/// Returns the biases from all the recurrent neurons in the layer.
/// The format is a vector of real values.
/// The size of this vector is the number of neurons in the layer.

Tensor<type, 2> RecurrentLayer::get_biases() const
{
    return biases;
}


/// Returns the weights from the recurrent layer.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of columns is the number of inputs to the layer.

const Tensor<type, 2>& RecurrentLayer::get_input_weights() const
{
    return input_weights;
}


/// Returns the recurrent weights from the recurrent layer.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of columns is the number of neurons to the layer.

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


/// Returns a single vector with all the layer parameters.
/// The format is a vector of real values.
/// The size is the number of parameters in the layer.

Tensor<type, 1> RecurrentLayer::get_parameters() const
{
    const Tensor<type, 2> input_weights = get_input_weights();
    const Tensor<type, 2> recurrent_weights = get_recurrent_weights();
    const Tensor<type, 2> biases = get_biases();

    Tensor<type, 1> parameters(input_weights.size() + recurrent_weights.size() + biases.size());

    for(Index i = 0; i < input_weights.size(); i++)
    {
        fill_n(parameters.data()+i, 1, input_weights(i));
    }

    for(Index i = 0; i < biases.size(); i++)
    {
        fill_n(parameters.data()+ input_weights.size() +i, 1, biases(i));
    }

    for(Index i = 0; i < recurrent_weights.size(); i++)
    {
        fill_n(parameters.data()+ input_weights.size() + biases.size() +i, 1, recurrent_weights(i));
    }

    return parameters;
}


/// Returns the activation function of the layer.

const RecurrentLayer::ActivationFunction& RecurrentLayer::get_activation_function() const
{
    return activation_function;
}


/// Returns the biases from all the recurrent in the layer.
/// The format is a vector of real values.
/// The size of this vector is the number of neurons in the layer.

Tensor<type, 2> RecurrentLayer::get_biases(const Tensor<type, 1>& parameters) const
{
    const Index biases_number = get_biases_number();
    const Index input_weights_number = get_input_weights_number();

    Tensor<type,1> new_biases(biases_number);

    new_biases = parameters.slice(Eigen::array<Eigen::Index, 1>({input_weights_number}), Eigen::array<Eigen::Index, 1>({biases_number}));

    Eigen::array<Index, 2> two_dim{{1, biases.dimension(1)}};

    return new_biases.reshape(two_dim);
}


/// Returns the weights from the recurrent layer.
/// The format is a matrix of real values.
/// The number of rows is the number of inputs in the layer.
/// The number of columns is the number of neurons to the layer.

Tensor<type, 2> RecurrentLayer::get_input_weights(const Tensor<type, 1>& parameters) const
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();
    const Index input_weights_number = get_input_weights_number();

    Tensor<type, 1> new_inputs_weights = parameters.slice(Eigen::array<Eigen::Index, 1>({0}), Eigen::array<Eigen::Index, 1>({input_weights_number}));

    Eigen::array<Index, 2> two_dim{{inputs_number, neurons_number}};

    return new_inputs_weights.reshape(two_dim);
}


/// Returns the recurrent weights from the recurrent layer.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of columns is the number of neurons to the layer.

Tensor<type, 2> RecurrentLayer::get_recurrent_weights(const Tensor<type, 1>& parameters) const
{
    const Index neurons_number = get_neurons_number();
    const Index recurrent_weights_number = recurrent_weights.size();

    const Index parameters_size = parameters.size();

    const Index start_recurrent_weights_number = (parameters_size - recurrent_weights_number);

    Tensor<type, 1> new_synaptic_weights = parameters.slice(Eigen::array<Eigen::Index, 1>({start_recurrent_weights_number}), Eigen::array<Eigen::Index, 1>({recurrent_weights_number}));

    Eigen::array<Index, 2> two_dim{{neurons_number, neurons_number}};

    return new_synaptic_weights.reshape(two_dim);
}


/// Returns a string with the name of the layer activation function.
/// This can be: Logistic, HyperbolicTangent, Threshold, SymmetricThreshold, Linear, RectifiedLinear, ScaledExponentialLinear.

string RecurrentLayer::write_activation_function() const
{
    switch(activation_function)
    {
    case Logistic:
        return "Logistic";

    case HyperbolicTangent:
        return "HyperbolicTangent";

    case Threshold:
        return "Threshold";

    case SymmetricThreshold:
        return "SymmetricThreshold";

    case Linear:
        return "Linear";

    case RectifiedLinear:
        return "RectifiedLinear";

    case ScaledExponentialLinear:
        return "ScaledExponentialLinear";

    case SoftPlus:
        return "SoftPlus";

    case SoftSign:
        return "SoftSign";

    case HardSigmoid:
        return "HardSigmoid";

    case ExponentialLinear:
        return "ExponentialLinear";
    }

    return string();
}


/// Returns true if messages from this class are to be displayed on the screen,
/// or false if messages from this class are not to be displayed on the screen.

const bool& RecurrentLayer::get_display() const
{
    return display;
}


/// Sets an empty layer, wihtout any neuron.
/// It also sets the rest of members to their default values.

void RecurrentLayer::set()
{
    set_default();
}


/// Sets new numbers of inputs and neurons in the layer.
/// It also sets the rest of members to their default values.
/// @param new_inputs_number Number of inputs.
/// @param new_neurons_number Number of neuron.

void RecurrentLayer::set(const Index& new_inputs_number, const Index& new_neurons_number)
{
    biases.resize(1, new_neurons_number);

    input_weights.resize(new_inputs_number, new_neurons_number);

    recurrent_weights.resize(new_neurons_number, new_neurons_number);

    hidden_states.resize(new_neurons_number); // memory

    hidden_states.setConstant(0.0);

    set_default();
}


/// Sets the members of this neuron layer object with those from other neuron layer object.
/// @param other_neuron_layer RecurrentLayer object to be copied.

void RecurrentLayer::set(const RecurrentLayer& other_neuron_layer)
{
    activation_function = other_neuron_layer.activation_function;

    display = other_neuron_layer.display;

    set_default();
}


/// Sets those members not related to the vector of neurons to their default value.
/// <ul>
/// <li> Display: True.
/// <li> layer_type: neuron_Layer.
/// <li> trainable: True.
/// </ul>

void RecurrentLayer::set_default()
{
    display = true;
}


/// Sets a new number of inputs in the layer.
/// The new synaptic weights are initialized at random.
/// @param new_inputs_number Number of layer inputs.

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


/// Sets a new number neurons in the layer.
/// All the parameters are also initialized at random.
/// @param new_neurons_number New number of neurons in the layer.

void RecurrentLayer::set_neurons_number(const Index& new_neurons_number)
{
    const Index inputs_number = get_inputs_number();

    biases.resize(1, new_neurons_number);

    input_weights.resize(inputs_number, new_neurons_number);

    recurrent_weights.resize(new_neurons_number, new_neurons_number);
}


void RecurrentLayer::set_timesteps(const Index & new_timesteps)
{
    timesteps = new_timesteps;
}


void RecurrentLayer::set_biases(const Tensor<type, 2>& new_biases)
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


/// Sets the parameters of this layer.
/// @param new_parameters Parameters vector for that layer.

void RecurrentLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{

#ifdef __OPENNN_DEBUG__

    const Index parameters_number = get_parameters_number();

    const Index new_parameters_size = new_parameters.size();

    if(new_parameters_size != parameters_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "void set_parameters(const Tensor<type, 1>&) method.\n"
               << "Size of new parameters (" << new_parameters_size << ") must be equal to number of parameters (" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif
    const Index inputs_wieghts_number = get_input_weights_number();
    const Index biases_number = get_biases_number();
    const Index recurrent_weights_number = get_recurrent_weights_number();

    memcpy(input_weights.data(),
           new_parameters.data() + index,
           static_cast<size_t>(inputs_wieghts_number)*sizeof(type));

    memcpy(biases.data(),
           new_parameters.data() + inputs_wieghts_number + index,
           static_cast<size_t>(biases_number)*sizeof(type));

    memcpy(recurrent_weights.data(),
           new_parameters.data() + inputs_wieghts_number + biases_number + index,
           static_cast<size_t>(recurrent_weights_number)*sizeof(type));
}


/// This class sets a new activation(or transfer) function in a single layer.
/// @param new_activation_function Activation function for the layer.

void RecurrentLayer::set_activation_function(const RecurrentLayer::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


/// Sets a new activation(or transfer) function in a single layer.
/// The second argument is a string containing the name of the function("Logistic", "HyperbolicTangent", "Threshold", etc).
/// @param new_activation_function Activation function for that layer.

void RecurrentLayer::set_activation_function(const string& new_activation_function_name)
{
    if(new_activation_function_name == "Logistic")
    {
        activation_function = Logistic;
    }
    else if(new_activation_function_name == "HyperbolicTangent")
    {
        activation_function = HyperbolicTangent;
    }
    else if(new_activation_function_name == "Threshold")
    {
        activation_function = Threshold;
    }
    else if(new_activation_function_name == "SymmetricThreshold")
    {
        activation_function = SymmetricThreshold;
    }
    else if(new_activation_function_name == "Linear")
    {
        activation_function = Linear;
    }
    else if(new_activation_function_name == "RectifiedLinear")
    {
        activation_function = RectifiedLinear;
    }
    else if(new_activation_function_name == "ScaledExponentialLinear")
    {
        activation_function = ScaledExponentialLinear;
    }
    else if(new_activation_function_name == "SoftPlus")
    {
        activation_function = SoftPlus;
    }
    else if(new_activation_function_name == "SoftSign")
    {
        activation_function = SoftSign;
    }
    else if(new_activation_function_name == "HardSigmoid")
    {
        activation_function = HardSigmoid;
    }
    else if(new_activation_function_name == "ExponentialLinear")
    {
        activation_function = ExponentialLinear;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: neuron class.\n"
               << "void set_activation_function(const string&) method.\n"
               << "Unknown activation function: " << new_activation_function_name << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void RecurrentLayer::set_display(const bool& new_display)
{
    display = new_display;
}


/// Initializes the hidden states of in the layer of neurons with a given value.
/// @param value Hidden states initialization value.

void RecurrentLayer::initialize_hidden_states(const type& value)
{
    hidden_states.setConstant(value);
}


/// Initializes the biases of all the neurons in the layer of neurons with a given value.
/// @param value Biases initialization value.

void RecurrentLayer::set_biases_constant(const type& value)
{
    biases.setConstant(value);
}


/// Initializes the input weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Input weights initialization value.

void RecurrentLayer::initialize_input_weights(const type& value)
{
    input_weights.setConstant(value);
}


/// Initializes the recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Synaptic weights initialization value.

void RecurrentLayer::initialize_recurrent_weights(const type& value)
{
    recurrent_weights.setConstant(value);
}


/// @todo

void RecurrentLayer::initialize_input_weights_Glorot(const type&,const type&)
{
    input_weights.setRandom();
}


/// Initializes all the biases, input weights and recurrent weights in the neural newtork with a given value.
/// @param value Parameters initialization value.

void RecurrentLayer::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    input_weights.setConstant(value);

    recurrent_weights.setConstant(value);

    hidden_states.setZero();
}


/// Initializes all the biases and input weights in the layer of neurons at random with values
/// comprised between a minimum and a maximum values.
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

void RecurrentLayer::set_parameters_random()
{
    biases.setRandom();

    input_weights.setRandom();

    recurrent_weights.setRandom();
}


void RecurrentLayer::calculate_current_combinations(const Tensor<type, 1>& current_inputs,
                            Tensor<type, 1>& current_combinations)
{
    memcpy(current_combinations.data(), biases.data(), static_cast<size_t>(current_inputs.size())*sizeof(type));

    current_combinations.device(*thread_pool_device) += current_inputs.contract(input_weights, AT_B);

    current_combinations.device(*thread_pool_device) += hidden_states.contract(recurrent_weights, AT_B).eval();
}


void RecurrentLayer::calculate_current_activations(const Tensor<type, 1>& combinations_1d, Tensor<type, 1>& activations_1d) const
{

#ifdef __OPENNN_DEBUG__

const Index neurons_number = get_neurons_number();

const Index combinations_columns_number = combinations_1d.dimension(0);

if(combinations_columns_number != neurons_number)
{
ostringstream buffer;

buffer << "OpenNN Exception: RecurrentLayer class.\n"
       << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
       << "Number of combinations_2d columns (" << combinations_columns_number
       << ") must be equal to number of neurons (" << neurons_number << ").\n";

throw logic_error(buffer.str());
}

#endif

    switch(activation_function)
    {
        case Linear: return linear(combinations_1d, activations_1d);

        case Logistic: return logistic(combinations_1d, activations_1d);

        case HyperbolicTangent: return hyperbolic_tangent(combinations_1d, activations_1d);

        case Threshold: return threshold(combinations_1d, activations_1d);

        case SymmetricThreshold: return symmetric_threshold(combinations_1d, activations_1d);

        case RectifiedLinear: return rectified_linear(combinations_1d, activations_1d);

        case ScaledExponentialLinear: return scaled_exponential_linear(combinations_1d, activations_1d);

        case SoftPlus: return soft_plus(combinations_1d, activations_1d);

        case SoftSign: return soft_sign(combinations_1d, activations_1d);

        case HardSigmoid: return hard_sigmoid(combinations_1d, activations_1d);

        case ExponentialLinear: return exponential_linear(combinations_1d, activations_1d);
    }
}


void RecurrentLayer::calculate_current_activations_derivatives(const Tensor<type, 1>& combinations_1d,
                                       Tensor<type, 1>& activations_1d,
                                       Tensor<type, 1>& activations_derivatives_1d) const
{
     #ifdef __OPENNN_DEBUG__

     const Index neurons_number = get_neurons_number();

     const Index combinations_columns_number = combinations_1d.dimension(1);

     if(combinations_columns_number != neurons_number)
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "void calculate_activations_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
               << "Number of combinations_1d columns (" << combinations_columns_number
               << ") must be equal to number of neurons (" << neurons_number << ").\n";

        throw logic_error(buffer.str());
     }

     #endif

     switch(activation_function)
     {
         case Linear: linear_derivatives(combinations_1d, activations_1d,  activations_derivatives_1d); return;

         case Logistic: logistic_derivatives(combinations_1d, activations_1d,  activations_derivatives_1d); return;

         case HyperbolicTangent: hyperbolic_tangent_derivatives(combinations_1d, activations_1d,  activations_derivatives_1d); return;

         case Threshold: threshold_derivatives(combinations_1d, activations_1d,  activations_derivatives_1d); return;

         case SymmetricThreshold: symmetric_threshold_derivatives(combinations_1d, activations_1d,  activations_derivatives_1d); return;

         case RectifiedLinear: rectified_linear_derivatives(combinations_1d, activations_1d,  activations_derivatives_1d); return;

         case ScaledExponentialLinear: scaled_exponential_linear_derivatives(combinations_1d, activations_1d,  activations_derivatives_1d); return;

         case SoftPlus: soft_plus_derivatives(combinations_1d, activations_1d,  activations_derivatives_1d); return;

         case SoftSign: soft_sign_derivatives(combinations_1d, activations_1d,  activations_derivatives_1d); return;

         case HardSigmoid: hard_sigmoid_derivatives(combinations_1d, activations_1d,  activations_derivatives_1d); return;

         case ExponentialLinear: exponential_linear_derivatives(combinations_1d, activations_1d,  activations_derivatives_1d); return;
     }
}


void RecurrentLayer::forward_propagate(const Tensor<type, 2>& inputs, ForwardPropagation& forward_propagation)
{

    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();
    const Index samples_number = inputs.dimension(0);

    Tensor<type, 1> current_inputs(inputs_number);

    Tensor<type, 1> current_combinations(neurons_number);
    Tensor<type, 1> current_activations(neurons_number);
    Tensor<type, 1> current_activations_derivatives(neurons_number);

    for(Index i = 0; i < samples_number; i++)
    {
        if(i%timesteps == 0) hidden_states.setZero();

        current_inputs = inputs.chip(i, 1);

        calculate_current_combinations(current_inputs, current_combinations);

        calculate_current_activations_derivatives(current_combinations, hidden_states, current_activations_derivatives);

        for(Index j = 0; j < neurons_number; j++)
        {
            forward_propagation.combinations_2d(i,j) = current_combinations(j);
            forward_propagation.activations_2d(i,j) = hidden_states(j);
            forward_propagation.activations_derivatives_2d(i,j) = current_activations_derivatives(j);
        }
    }

}


Tensor<type, 2> RecurrentLayer::calculate_outputs(const Tensor<type, 2>& inputs)
{
#ifdef __OPENNN_DEBUG__

    const Index inputs_number = get_inputs_number();

    const Index inputs_columns_number = inputs.dimension(1);

    if(inputs_columns_number != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
               << "Number of columns("<<inputs_columns_number<<") of inputs matrix must be equal to number of inputs("<<inputs_number<<").\n";

        throw logic_error(buffer.str());
    }
#endif

    const Index samples_number = inputs.dimension(0);

    const Index neurons_number = get_neurons_number();

    Tensor<type, 1> current_inputs(neurons_number);

    Tensor<type, 1> current_outputs(neurons_number);

    Tensor<type, 2> outputs(samples_number, neurons_number);

    for(Index i = 0; i < samples_number; i++)
    {
        if(i%timesteps == 0) hidden_states.setZero();

        current_inputs = inputs.chip(i, 0);

        calculate_current_combinations(current_inputs, current_outputs);

        calculate_current_activations(current_outputs, hidden_states);

        for(Index j = 0; j < neurons_number; j++)
            outputs(i,j) = hidden_states(j);
    }

    return outputs;
}


void RecurrentLayer::calculate_hidden_delta(Layer* next_layer_pointer,
                            const Tensor<type, 2>&,
                            ForwardPropagation& forward_propagation,
                            const Tensor<type, 2>& next_layer_delta,
                            Tensor<type, 2>& hidden_delta) const
{

    const Type layer_type = next_layer_pointer->get_type();

    const Index neurons_number = next_layer_pointer->get_neurons_number();
    const Index inputs_number = next_layer_pointer->get_inputs_number();

    Tensor<type, 2> synaptic_weights(inputs_number, neurons_number);

    if(layer_type == Perceptron)
    {
        const PerceptronLayer* perceptron_layer = dynamic_cast<PerceptronLayer*>(next_layer_pointer);

        synaptic_weights= perceptron_layer->get_synaptic_weights();
    }
    else if(layer_type == Probabilistic)
    {
        const ProbabilisticLayer* probabilistic_layer = dynamic_cast<ProbabilisticLayer*>(next_layer_pointer);

        synaptic_weights = probabilistic_layer->get_synaptic_weights();
    }
/*
    Tensor<type, 2> hidden_delta(next_layer_delta.dimension(0), synaptic_weights.dimension(1));

    hidden_delta.device(*thread_pool_device) = next_layer_delta.contract(synaptic_weights, A_BT);

    hidden_delta.device(*thread_pool_device) = activations_derivatives*hidden_delta;
*/
}



Tensor<type, 1> RecurrentLayer::calculate_error_gradient(const Tensor<type, 2> & inputs,
        const Layer::ForwardPropagation& layers,
        const Tensor<type, 2> & deltas)
{
    const Index input_weights_number = get_input_weights_number();
    const Index recurrent_weights_number = get_recurrent_weights_number();

    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> error_gradient(parameters_number);

    // Input weights

//    error_gradient.embed(0, calculate_input_weights_error_gradient(inputs,layers,deltas));
    for(Index i = 0; i < calculate_input_weights_error_gradient(inputs,layers,deltas).size(); i++)
    {
        error_gradient(i) = (calculate_input_weights_error_gradient(inputs, layers, deltas))(i);
    }

    // Recurent weights

//    error_gradient.embed(input_weights_number, calculate_recurrent_weights_error_gradient(inputs,layers,deltas));
    for(Index i = 0; i < calculate_recurrent_weights_error_gradient(inputs, layers, deltas).size(); i++)
    {
        error_gradient(i + input_weights_number) = (calculate_recurrent_weights_error_gradient(inputs, layers, deltas))(i);
    }

    // Biases

//    error_gradient.embed(input_weights_number+recurrent_weights_number, calculate_biases_error_gradient(inputs,layers,deltas));
    for(Index i = 0; i < calculate_biases_error_gradient(inputs, layers, deltas).size(); i++)
    {
        error_gradient(i + input_weights_number+recurrent_weights_number) = (calculate_biases_error_gradient(inputs, layers, deltas))(i);
    }

    return error_gradient;

}


Tensor<type, 1> RecurrentLayer::calculate_input_weights_error_gradient(const Tensor<type, 2> & inputs,
        const Layer::ForwardPropagation& layers,
        const Tensor<type, 2> & deltas)
{
    const Index samples_number = inputs.dimension(0);
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    const Index parameters_number = inputs_number*neurons_number;

    // Derivatives of combinations_2d with respect to input weights

    Tensor<type, 2> combinations_weights_derivatives(parameters_number, neurons_number);

    Index column_index = 0;
    Index input_index = 0;

    Tensor<type, 1> input_weights_gradient(parameters_number);

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_inputs = inputs.chip(sample, 0);

        const Tensor<type, 1> current_layer_deltas = deltas.chip(sample, 0);//get_row(sample).to_column_matrix();

        if(sample%timesteps == 0)
        {
            combinations_weights_derivatives.setZero();
        }
        else
        {
            const Tensor<type, 1> previous_activation_derivatives = layers.activations_derivatives_2d.chip(sample-1, 0);//.get_row(sample-1);

//            combinations_weights_derivatives = dot(combinations_weights_derivatives.multiply_rows(previous_activation_derivatives), recurrent_weights);
            combinations_weights_derivatives = (multiply_rows(combinations_weights_derivatives, previous_activation_derivatives)).contract(recurrent_weights,A_B);
        }

        column_index = 0;
        input_index = 0;

        for(Index i = 0; i < parameters_number; i++)
        {
            combinations_weights_derivatives(i, column_index) += current_inputs(input_index);

            input_index++;

            if(input_index == inputs_number)
            {
                input_index = 0;
                column_index++;
            }
        }

//        input_weights_gradient += dot(combinations_weights_derivatives, current_layer_deltas).to_vector();
        input_weights_gradient += combinations_weights_derivatives.contract(current_layer_deltas, A_BT);

    }

    return input_weights_gradient;
}


Tensor<type, 1> RecurrentLayer::calculate_recurrent_weights_error_gradient(const Tensor<type, 2> &,
        const Layer::ForwardPropagation& forward_propagation,
        const Tensor<type, 2> & deltas)
{
    const Index samples_number = deltas.dimension(0);
    const Index neurons_number = get_neurons_number();

    const Index parameters_number = neurons_number*neurons_number;

    // Derivatives of combinations_2d with respect to recurrent weights

    Tensor<type, 2> combinations_recurrent_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 1> recurrent_weights_gradient(parameters_number);

    for(Index sample = 0; sample < samples_number-1; sample++)
    {
        Tensor<type, 1> current_activations = forward_propagation.activations_2d.chip(sample, 0);

        const Tensor<type, 1> next_layer_deltas = deltas.chip(sample+1,0);//get_row(sample+1).to_column_matrix();

        if((sample+1)%timesteps == 0)
        {
            combinations_recurrent_weights_derivatives.setZero();
        }
        else
        {
            const Tensor<type, 1> activation_derivatives = forward_propagation.activations_derivatives_2d.chip(sample, 0);

//            combinations_recurrent_weights_derivatives = dot(combinations_recurrent_weights_derivatives.multiply_rows(activation_derivatives), recurrent_weights);
            combinations_recurrent_weights_derivatives = (multiply_rows(combinations_recurrent_weights_derivatives, activation_derivatives)).contract(recurrent_weights,A_B);

            Index column_index = 0;
            Index activation_index = 0;

            for(Index i = 0; i < parameters_number; i++)
            {
                combinations_recurrent_weights_derivatives(i, column_index) += current_activations(activation_index);

                activation_index++;

                if(activation_index == neurons_number)
                {
                    activation_index = 0;
                    column_index++;
                }
            }
        }

//        recurrent_weights_gradient += dot(combinations_recurrent_weights_derivatives, next_layer_deltas).to_vector();
        recurrent_weights_gradient += combinations_recurrent_weights_derivatives.contract(next_layer_deltas, A_BT);
    }

    return recurrent_weights_gradient;
}



Tensor<type, 1> RecurrentLayer::calculate_biases_error_gradient(const Tensor<type, 2> & inputs,
        const Layer::ForwardPropagation& layers,
        const Tensor<type, 2> & deltas)
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();

    const Index biases_number = get_biases_number();

    // Derivatives of combinations_2d with respect to biases

    Tensor<type, 2> combinations_biases_derivatives(biases_number, neurons_number);

    Tensor<type, 1> biases_gradient(biases_number);
    biases_gradient.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_inputs = inputs.chip(sample, 0);

        const Tensor<type, 1> current_layer_deltas = deltas.chip(sample,0);//.get_row(sample).to_column_matrix();

        if(sample%timesteps == 0)
        {
            combinations_biases_derivatives.setZero();
        }
        else
        {
            const Tensor<type, 1> previous_activation_derivatives = layers.activations_derivatives_2d.chip(sample-1,0);//get_row(sample-1);

//            combinations_biases_derivatives = dot(combinations_biases_derivatives.multiply_rows(previous_activation_derivatives), recurrent_weights);
              combinations_biases_derivatives = (multiply_rows(combinations_biases_derivatives,previous_activation_derivatives)).contract(recurrent_weights, A_B);
        }

//        combinations_biases_derivatives.sum_diagonal(1.0);
        for(Index i = 0; i < biases_number; i++)
            combinations_biases_derivatives(i,i) += static_cast<type>(1.0);

//        biases_gradient += dot(combinations_biases_derivatives, current_layer_deltas).to_vector();
        biases_gradient += combinations_biases_derivatives.contract(current_layer_deltas, A_BT);
    }

    return biases_gradient;
}

Tensor<type, 2> RecurrentLayer::multiply_rows(const Tensor<type, 2>& matrix, const Tensor<type, 1>& vector) const
{
    const Index columns_number = matrix.dimension(1);
    const Index rows_number = matrix.dimension(0);

    Tensor<type, 2> new_matrix(rows_number, columns_number);

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
           new_matrix(i,j) = matrix(i,j) * vector(j);
        }
    }

    return new_matrix;
}


/// Returns a string with the expression of the inputs-outputs relationship of the layer.
/// @param inputs_names Vector of strings with the name of the layer inputs.
/// @param outputs_names Vector of strings with the name of the layer outputs.

string RecurrentLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    const Index inputs_number = get_inputs_number();
    const Index inputs_name_size = inputs_names.size();

    if(inputs_name_size != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const method.\n"
               << "Size of inputs name must be equal to number of layer inputs.\n";

        throw logic_error(buffer.str());
    }

    const Index outputs_name_size = outputs_names.size();

    if(outputs_name_size != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const method.\n"
               << "Size of outputs name must be equal to number of neurons.\n";

        throw logic_error(buffer.str());
    }

#endif

    ostringstream buffer;
    /*
       for(Index j = 0; j < outputs_names.size(); j++)
       {
           buffer << outputs_names[j] << " = " << write_activation_function_expression() << " (" << biases[j] << "+";

           for(Index i = 0; i < inputs_names.size() - 1; i++)
           {
               buffer << " (" << inputs_names[i] << "*" << input_weights.get_column(j)(i) << ")+";
           }

           buffer << " (" << inputs_names[inputs_names.size() - 1] << "*" << input_weights.get_column(j)[inputs_names.size() - 1] << "));\n";

           for(Index i = 0; i < outputs_names.size() - 1; i++)
           {
               buffer << " (hidden_states_" << std::to_string(i+1) << "*" << recurrent_weights.get_column(j)(i) << ")+";
           }

           buffer << " (hidden_states_" << std::to_string(outputs_names.size()) << "*" << recurrent_weights.get_column(j)[outputs_names.size() - 1] << "));\n";

       }
    */
    return buffer.str();
}


string RecurrentLayer::write_activation_function_expression() const
{
    switch(activation_function)
    {
        case HyperbolicTangent: return "tanh";

        case Linear: return "";

        default: return write_activation_function();
    }
}



void RecurrentLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Perceptron layer

    const tinyxml2::XMLElement* perceptron_layer_element = document.FirstChildElement("RecurrentLayer");

    if(!perceptron_layer_element)
    {
        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "RecurrentLayer element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = perceptron_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "InputsNumber element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(inputs_number_element->GetText())
    {
        set_inputs_number(static_cast<Index>(stoi(inputs_number_element->GetText())));
    }

    // Neurons number

    const tinyxml2::XMLElement* neurons_number_element = perceptron_layer_element->FirstChildElement("NeuronsNumber");

    if(!neurons_number_element)
    {
        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "NeuronsNumber element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(neurons_number_element->GetText())
    {
        set_neurons_number(static_cast<Index>(stoi(neurons_number_element->GetText())));
    }

    // Activation function

    const tinyxml2::XMLElement* activation_function_element = perceptron_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
    {
        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "ActivationFunction element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(activation_function_element->GetText())
    {
        set_activation_function(activation_function_element->GetText());
    }

    // Parameters

    const tinyxml2::XMLElement* parameters_element = perceptron_layer_element->FirstChildElement("Parameters");

    if(!parameters_element)
    {
        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Parameters element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(parameters_element->GetText())
    {
        const string parameters_string = parameters_element->GetText();

        set_parameters(to_type_vector(parameters_string, ' '));
    }
}


void RecurrentLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const

{
    ostringstream buffer;

    // Perceptron layer

    file_stream.OpenElement("RecurrentLayer");

    // Inputs number

    file_stream.OpenElement("InputsNumber");

    buffer.str("");
    buffer << get_inputs_number();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Outputs number

    file_stream.OpenElement("NeuronsNumber");

    buffer.str("");
    buffer << get_neurons_number();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Activation function

    file_stream.OpenElement("ActivationFunction");

    file_stream.PushText(write_activation_function().c_str());

    file_stream.CloseElement();

    // Parameters

    file_stream.OpenElement("Parameters");

    buffer.str("");

    const Tensor<type, 1> parameters = get_parameters();
    const Index parameters_size = parameters.size();

    for(Index i = 0; i < parameters_size; i++)
    {
        buffer << parameters(i);

        if(i != (parameters_size-1)) buffer << " ";
    }

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Recurrent layer (end tag)

    file_stream.CloseElement();
}


}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
