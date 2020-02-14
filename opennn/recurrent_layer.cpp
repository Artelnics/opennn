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


/// Copy constructor.
/// It creates a copy of an existing neuron layer object.
/// @param other_neuron_layer neuron layer object to be copied.

RecurrentLayer::RecurrentLayer(const RecurrentLayer& other_neuron_layer) : Layer()
{
    set(other_neuron_layer);
}


/// Destructor.
/// This destructor does not delete any pointer.

RecurrentLayer::~RecurrentLayer()
{
}


Tensor<Index, 1> RecurrentLayer::get_input_variables_dimensions() const
{
    return Tensor<Index, 1>();
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

Tensor<type, 1> RecurrentLayer::get_hidden_states() const
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

Tensor<type, 1> RecurrentLayer::get_biases() const
{
    return biases;
}


/// Returns the weights from the recurrent layer.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of columns is the number of inputs to the layer.

Tensor<type, 2> RecurrentLayer::get_input_weights() const
{
    return input_weights;
}


/// Returns the recurrent weights from the recurrent layer.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of columns is the number of neurons to the layer.

Tensor<type, 2> RecurrentLayer::get_recurrent_weights() const
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
    const Tensor<type, 1> biases = get_biases();
    /*
        return input_weights.to_vector().assemble(recurrent_weights.to_vector()).assemble(biases);
    */
    return Tensor<type, 1>();
}


/// Returns the activation function of the layer.

const RecurrentLayer::ActivationFunction& RecurrentLayer::get_activation_function() const
{
    return activation_function;
}


/// Returns the biases from all the recurrent in the layer.
/// The format is a vector of real values.
/// The size of this vector is the number of neurons in the layer.

Tensor<type, 1> RecurrentLayer::get_biases(const Tensor<type, 1>& parameters) const
{
    const Index biases_number = get_biases_number();
    /*
        return parameters.get_last(biases_number);
    */
    return Tensor<type, 1>();
}


/// Returns the weights from the recurrent layer.
/// The format is a matrix of real values.
/// The number of rows is the number of inputs in the layer.
/// The number of columns is the number of neurons to the layer.

Tensor<type, 2> RecurrentLayer::get_input_weights(const Tensor<type, 1>& parameters) const
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    const Index input_weights_number = input_weights.size();
    /*
        return parameters.get_first(input_weights_number).to_matrix(inputs_number, neurons_number);
    */
    return Tensor<type, 2>();

}


/// Returns the recurrent weights from the recurrent layer.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of columns is the number of neurons to the layer.

Tensor<type, 2> RecurrentLayer::get_recurrent_weights(const Tensor<type, 1>& parameters) const
{
    const Index neurons_number = get_neurons_number();

    const Index weights_number = input_weights.size();
    const Index recurrent_weights_number = recurrent_weights.size();
    /*
        return parameters.get_subvector(weights_number, weights_number + recurrent_weights_number - 1 ).to_matrix(neurons_number, neurons_number);
    */
    return Tensor<type, 2>();
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

    biases.resize(new_neurons_number);

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
    /*
    if(size.empty() || size.size() > 1)
    {
    //        throw exception(string("EXCEPTION: The new size is incompatible."));
    }
    */
    const Index new_size = size[0];

    set_inputs_number(new_size);

}


/// Sets a new number neurons in the layer.
/// All the parameters are also initialized at random.
/// @param new_neurons_number New number of neurons in the layer.

void RecurrentLayer::set_neurons_number(const Index& new_neurons_number)
{
    const Index inputs_number = get_inputs_number();

    biases.resize(new_neurons_number);

    input_weights.resize(inputs_number, new_neurons_number);

    recurrent_weights.resize(new_neurons_number, new_neurons_number);
}


void RecurrentLayer::set_timesteps(const Index & new_timesteps)
{
    timesteps = new_timesteps;
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


/// Sets the parameters of this layer.
/// @param new_parameters Parameters vector for that layer.

void RecurrentLayer::set_parameters(const Tensor<type, 1>& new_parameters)
{
    const Index parameters_number = get_parameters_number();

    const Index inputs_number = get_inputs_number();

    const Index neurons_number = get_neurons_number();

#ifdef __OPENNN_DEBUG__

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
    /*
       input_weights = new_parameters.get_subvector(0, inputs_number * neurons_number -1) .to_matrix(inputs_number, neurons_number);

       recurrent_weights = new_parameters.get_subvector(inputs_number * neurons_number,inputs_number * neurons_number + neurons_number * neurons_number -1) .to_matrix(neurons_number, neurons_number);

       biases = new_parameters.get_last(neurons_number);
    */
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

void RecurrentLayer::initialize_biases(const type& value)
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


void RecurrentLayer::initialize_input_weights_Glorot(const type& minimum,const type& maximum)
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


Tensor<type, 1> RecurrentLayer::calculate_combinations(const Tensor<type, 1>& inputs) const
{
#ifdef __OPENNN_DEBUG__

    const Index inputs_number = get_inputs_number();

    if(inputs.size() != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&) const method.\n"
               << "Size of layer inputs (" << inputs.size() << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif
    /*
        return dot(inputs, input_weights) + biases + dot(hidden_states, recurrent_weights);
    */

    return Tensor<type, 1>();
}


Tensor<type, 2> RecurrentLayer::calculate_combinations(const Tensor<type, 2>& inputs)
{
#ifdef __OPENNN_DEBUG__

    const Index inputs_number = get_inputs_number();

    const Index inputs_columns_number = inputs.dimension(1);

    if(inputs_columns_number != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 2> calculate_combinations(const Tensor<type, 2>&) const method.\n"
               << "Number of columns("<<inputs_columns_number<<") of inputs matrix must be equal to number of inputs("<<inputs_number<<").\n";

        throw logic_error(buffer.str());
    }
#endif

    const Index instances_number = inputs.dimension(0);

    const Index neurons_number = get_neurons_number();

    Tensor<type, 2> outputs(instances_number, neurons_number);

    for(Index i = 0; i < instances_number; i++)
    {
        if(i%timesteps == 0) hidden_states.setZero();
        /*
                const Tensor<type, 1> current_inputs = inputs.chip(i, 0);

                const Tensor<type, 1> combinations = calculate_combinations(current_inputs);

                const Tensor<type, 1> activations = calculate_activations(combinations);

                hidden_states = activations;

                outputs.set_row(i, combinations);
        */
    }

    return outputs;

    return Tensor<type, 2>();
}


Tensor<type, 1> RecurrentLayer::calculate_combinations(const Tensor<type, 1>& inputs, const Tensor<type, 1>& parameters) const
{
#ifdef __OPENNN_DEBUG__

    const Index inputs_number = get_inputs_number();

    if(inputs.size() != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of layer inputs (" << inputs.size() << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

        throw logic_error(buffer.str());
    }

    if(parameters.size() != get_parameters_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&) const method.\n"
               << "Size of layer parameters (" << parameters.size() << ") must be equal to number of layer parameters (" << get_parameters_number() << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    const Tensor<type, 2> new_input_weights = get_input_weights(parameters);
    const Tensor<type, 2> new_recurent_weights = get_recurrent_weights(parameters);
    const Tensor<type, 1> new_biases = get_biases(parameters);
    /*
        return dot(inputs, new_input_weights) + new_biases + dot(hidden_states,new_recurent_weights);
    */
    return Tensor<type, 1>();
}


Tensor<type, 1> RecurrentLayer::calculate_combinations(const Tensor<type, 1>& inputs, const Tensor<type, 1>& new_biases, const Tensor<type, 2>& new_input_weights, const Tensor<type, 2>& new_recurrent_weights) const
{
#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    const Index inputs_number = get_inputs_number();

    if(new_biases.size() != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>& , const Tensor<type, 2>&) const method.\n"
               << "Size of biases must be equal to number of neurons.\n";

        throw logic_error(buffer.str());
    }

    if(inputs.size() != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>& , const Tensor<type, 2>&) const method.\n"
               << "Size of layer inputs (" << inputs.size() << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

        throw logic_error(buffer.str());
    }

    if(new_input_weights.dimension(0) != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>& , const Tensor<type, 2>&) const method.\n"
               << "Rows number of input weights  (" << new_input_weights.dimension(0) << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

        throw logic_error(buffer.str());
    }


    if(new_input_weights.dimension(1) != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>& , const Tensor<type, 2>&) const method.\n"
               << "Columns number of weight  (" << new_input_weights.dimension(1) << ") must be equal to number of neurons number (" << neurons_number << ").\n";

        throw logic_error(buffer.str());
    }


    if(new_recurrent_weights.dimension(1) != neurons_number  || new_recurrent_weights.dimension(0) != neurons_number )
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>& , const Tensor<type, 2>&) const method.\n"
               << "Columns number of recurrent weight  (" <<new_recurrent_weights.dimension(1)  << ") must be equal to number of neurons number (" << neurons_number << ").\n"
               << "Rows number of recurrent weight  (" <<new_recurrent_weights.dimension(0)  << ") must be equal to number of neurons number (" << neurons_number << ").\n";

        throw logic_error(buffer.str());
    }


#endif

    Tensor<type, 1> combinations(get_neurons_number());
    /*
        combinations = dot(inputs, new_input_weights) + new_biases + dot(hidden_states, new_recurrent_weights);
    */
    return combinations ;
}


Tensor<type, 1> RecurrentLayer::calculate_activations(const Tensor<type, 1>& combinations) const
{
#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    const Index combinations_number = combinations.size();

    if(combinations_number != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 1> calculate_activations(const Tensor<type, 1>&) const method.\n"
               << "Size of combinations (" << combinations_number <<") must be equal to number of neurons("<< neurons_number <<") .\n";

        throw logic_error(buffer.str());
    }

#endif

    switch(activation_function)
    {
        /*
                case Linear: return linear(combinations);

                case Logistic: return logistic(combinations);

                case HyperbolicTangent: return hyperbolic_tangent(combinations);

                case Threshold: return threshold(combinations);

                case SymmetricThreshold: return symmetric_threshold(combinations);

                case RectifiedLinear: return rectified_linear(combinations);

                case ScaledExponentialLinear: return scaled_exponential_linear(combinations);

                case SoftPlus: return soft_plus(combinations);

                case SoftSign: return soft_sign(combinations);

                case HardSigmoid: return hard_sigmoid(combinations);

                case ExponentialLinear: return exponential_linear(combinations);
        */
    }

    return Tensor<type, 1>();
}


Tensor<type, 2> RecurrentLayer::calculate_activations(const Tensor<type, 2>& combinations) const
{
    switch(activation_function)
    {
        /*
                case Linear: return linear(combinations);

                case Logistic: return logistic(combinations);

                case HyperbolicTangent: return hyperbolic_tangent(combinations);

                case Threshold: return threshold(combinations);

                case SymmetricThreshold: return symmetric_threshold(combinations);

                case RectifiedLinear: return rectified_linear(combinations);

                case ScaledExponentialLinear: return scaled_exponential_linear(combinations);

                case SoftPlus: return soft_plus(combinations);

                case SoftSign: return soft_sign(combinations);

                case HardSigmoid: return hard_sigmoid(combinations);

                case ExponentialLinear: return exponential_linear(combinations);
        */
    }

    return Tensor<type, 2>();
}


Tensor<type, 2> RecurrentLayer::calculate_activations_derivatives(const Tensor<type, 2>& combinations) const
{
#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    const Index combinations_number = combinations.dimension(1);

    if(combinations_number != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 2> calculate_activations_derivatives(const Tensor<type, 2>&) const method.\n"
               << "Number of combinations (" << combinations_number << ") must be equal to number of neurons (" << neurons_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    switch(activation_function)
    {
        /*
                case Linear: return linear_derivatives(combinations);

                case Logistic: return logistic_derivatives(combinations);

                case HyperbolicTangent: return hyperbolic_tangent_derivatives(combinations);

                case Threshold: return threshold_derivatives(combinations);

                case SymmetricThreshold: return symmetric_threshold_derivatives(combinations);

                case RectifiedLinear: return rectified_linear_derivatives(combinations);

                case ScaledExponentialLinear: return scaled_exponential_linear_derivatives(combinations);

                case SoftPlus: return soft_plus_derivatives(combinations);

                case SoftSign: return soft_sign_derivatives(combinations);

                case HardSigmoid: return hard_sigmoid_derivatives(combinations);

                case ExponentialLinear: return exponential_linear_derivatives(combinations);
        */
    }

    return Tensor<type, 2> ();
}

void RecurrentLayer::update_hidden_states(const Tensor<type, 1>& inputs)
{
    /*
        const Tensor<type, 1> combinations = calculate_combinations(inputs);

        hidden_states = calculate_activations(combinations);
    */
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

    const Index instances_number = inputs.dimension(0);

    const Index neurons_number = get_neurons_number();

    Tensor<type, 2> outputs(instances_number, neurons_number);

    for(Index i = 0; i < instances_number; i++)
    {
        if(i%timesteps == 0) hidden_states.setZero();
        /*
                const Tensor<type, 1> current_inputs = inputs.chip(i, 0);

                const Tensor<type, 1> combinations = calculate_combinations(current_inputs);

                const Tensor<type, 1> activations = calculate_activations(combinations);

                outputs.set_row(i, activations);

                hidden_states = activations;
        */
    }

    return outputs;
}


Tensor<type, 2> RecurrentLayer::calculate_outputs(const Tensor<type, 2>& inputs, const Tensor<type, 1>& parameters)
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

    if(parameters.size() != get_parameters_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 1>&) const method.\n"
               << "Size of layer parameters (" << parameters.size() << ") must be equal to number of layer parameters (" << get_parameters_number() << ").\n";

        throw logic_error(buffer.str());
    }
#endif

    const Index instances_number = inputs.dimension(0);

    const Index neurons_number = get_neurons_number();
    /*
        Tensor<type, 2> outputs(Tensor<Index, 1>({instances_number, neurons_number}));

        for(Index i = 0; i < instances_number; i++)
        {
            if(i%timesteps == 0) hidden_states.setZero();

            const Tensor<type, 1> current_inputs = inputs.chip(i, 0);

            const Tensor<type, 1> combinations = calculate_combinations(current_inputs, parameters);

            const Tensor<type, 1> activations = calculate_activations(combinations);

            hidden_states = activations;

            outputs.set_row(i, activations);
          }

        return outputs;
    */
    return Tensor<type, 2>();
}


Tensor<type, 2> RecurrentLayer::calculate_outputs(const Tensor<type, 2>& inputs,
        const Tensor<type, 1>& new_biases,
        const Tensor<type, 2>& new_input_weights,
        const Tensor<type, 2>& new_recurrent_weights)
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

#ifdef __OPENNN_DEBUG__

    const Index inputs_columns_number = inputs.dimension(1);

    if(inputs_columns_number != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
               << "Number of columns("<<inputs_columns_number<<") of inputs matrix must be equal to number of inputs("<<inputs_number<<").\n";

        throw logic_error(buffer.str());
    }

    if(new_biases.size() != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>& , const Tensor<type, 2>&) const method.\n"
               << "Size of biases must be equal to number of neurons.\n";

        throw logic_error(buffer.str());
    }

    if(new_input_weights.dimension(0) != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>& , const Tensor<type, 2>&) const method.\n"
               << "Rows number of weight  (" << new_input_weights.dimension(0) << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

        throw logic_error(buffer.str());
    }


    if(new_input_weights.dimension(1) != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>& , const Tensor<type, 2>&) const method.\n"
               << "Columns number of weight  (" << new_input_weights.dimension(1) << ") must be equal to number of neurons number (" << neurons_number << ").\n";

        throw logic_error(buffer.str());
    }


    if(new_recurrent_weights.dimension(1) != neurons_number  || new_recurrent_weights.dimension(0) != neurons_number )
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: RecurrentLayer class.\n"
               << "Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>& , const Tensor<type, 2>&) const method.\n"
               << "Columns number of recurrent weight  (" <<new_recurrent_weights.dimension(1)  << ") must be equal to number of neurons number (" << neurons_number << ").\n"
               << "Rows number of recurrent weight  (" <<new_recurrent_weights.dimension(0)  << ") must be equal to number of neurons number (" << neurons_number << ").\n";

        throw logic_error(buffer.str());
    }
#endif

    const Index instances_number = inputs.dimension(0);
    /*
        Tensor<type, 2> outputs(Tensor<Index, 1>({instances_number, neurons_number}));

        for(Index i = 0; i < instances_number; i++)
        {
            if(i%timesteps == 0) hidden_states.setZero();

            const Tensor<type, 1> current_inputs = inputs.chip(i, 0);

            const Tensor<type, 1> combinations = calculate_combinations(current_inputs, new_biases, new_input_weights, new_recurrent_weights);

            const Tensor<type, 1> activations = calculate_activations(combinations);

            hidden_states = activations;

            outputs.set_row(i, activations);
          }

        return outputs;
    */
    return Tensor<type, 2>();
}


Tensor<type, 2> RecurrentLayer::calculate_hidden_delta(Layer* next_layer_pointer,
        const Tensor<type, 2>&,
        const Tensor<type, 2>& activations_derivatives,
        const Tensor<type, 2>& next_layer_delta) const
{

    const Type layer_type = next_layer_pointer->get_type();
    /*
        Tensor<type, 2> synaptic_weights_transpose;

        if(layer_type == Perceptron)
        {
            const PerceptronLayer* perceptron_layer = dynamic_cast<PerceptronLayer*>(next_layer_pointer);

            synaptic_weights_transpose = perceptron_layer->get_synaptic_weights_transpose();
        }
        else if(layer_type == Probabilistic)
        {
            const ProbabilisticLayer* probabilistic_layer = dynamic_cast<ProbabilisticLayer*>(next_layer_pointer);

            synaptic_weights_transpose = probabilistic_layer->get_synaptic_weights_transpose();
        }

        return activations_derivatives*dot(next_layer_delta, synaptic_weights_transpose);
    */
    return Tensor<type, 2>();
}


/*
Layer::ForwardPropagation RecurrentLayer::forward_propagate(const Tensor<type, 2>& inputs)
{
    ForwardPropagation layers;

    const Tensor<type, 2> combinations = calculate_combinations(inputs);

    layers.activations = calculate_activations(combinations);

    layers.activations_derivatives = calculate_activations_derivatives(combinations);

    return layers;
}
*/

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
        error_gradient(i) = (calculate_input_weights_error_gradient(inputs,layers,deltas))(i);
    }

    // Recurent weights

//    error_gradient.embed(input_weights_number, calculate_recurrent_weights_error_gradient(inputs,layers,deltas));
    for(Index i = 0; i < calculate_recurrent_weights_error_gradient(inputs,layers,deltas).size(); i++)
    {
        error_gradient(i + input_weights_number) = (calculate_recurrent_weights_error_gradient(inputs,layers,deltas))(i);
    }

    // Biases

//    error_gradient.embed(input_weights_number+recurrent_weights_number, calculate_biases_error_gradient(inputs,layers,deltas));
    for(Index i = 0; i < calculate_biases_error_gradient(inputs,layers,deltas).size(); i++)
    {
        error_gradient(i + input_weights_number+recurrent_weights_number) = (calculate_biases_error_gradient(inputs,layers,deltas))(i);
    }

    return error_gradient;

}


Tensor<type, 1> RecurrentLayer::calculate_input_weights_error_gradient(const Tensor<type, 2> & inputs,
        const Layer::ForwardPropagation& layers,
        const Tensor<type, 2> & deltas)
{
    const Index instances_number = inputs.dimension(0);
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    const Index parameters_number = inputs_number*neurons_number;

    // Derivatives of combinations with respect to input weights

    Tensor<type, 2> combinations_weights_derivatives(parameters_number, neurons_number);

    Index column_index = 0;
    Index input_index = 0;

    Tensor<type, 1> input_weights_gradient(parameters_number);

    for(Index instance = 0; instance < instances_number; instance++)
    {
        const Tensor<type, 1> current_inputs = inputs.chip(instance, 0);
        /*
                const Tensor<type, 2> current_layer_deltas = deltas.get_row(instance).to_column_matrix();

                if(instance%timesteps == 0)
                {
                    combinations_weights_derivatives.setZero();
                }
                else
                {
                    const Tensor<type, 1> previous_activation_derivatives = layers.activations_derivatives.get_row(instance-1);

                    combinations_weights_derivatives = dot(combinations_weights_derivatives.multiply_rows(previous_activation_derivatives), recurrent_weights);
                }

                column_index = 0;
                input_index = 0;

                for(Index i = 0; i < parameters_number; i++)
                {
                    combinations_weights_derivatives(i, column_index) += current_inputs[input_index];

                    input_index++;

                    if(input_index == inputs_number)
                    {
                        input_index = 0;
                        column_index++;
                    }
                }

                input_weights_gradient += dot(combinations_weights_derivatives, current_layer_deltas).to_vector();
        */
    }

    return input_weights_gradient;
}


Tensor<type, 1> RecurrentLayer::calculate_recurrent_weights_error_gradient(const Tensor<type, 2> &,
        const Layer::ForwardPropagation& forward_propagation,
        const Tensor<type, 2> & deltas)
{
    const Index instances_number = deltas.dimension(0);
    const Index neurons_number = get_neurons_number();

    const Index parameters_number = neurons_number*neurons_number;

    // Derivatives of combinations with respect to recurrent weights

    Tensor<type, 2> combinations_recurrent_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 1> recurrent_weights_gradient(parameters_number);
    /*
        for(Index instance = 0; instance < instances_number-1; instance++)
        {
            Tensor<type, 1> current_activations = forward_propagation.activations.chip(instance, 0);

            const Tensor<type, 2> next_layer_deltas = deltas.get_row(instance+1).to_column_matrix();

            if((instance+1)%timesteps == 0)
            {
                combinations_recurrent_weights_derivatives.setZero();
            }
            else
            {
                const Tensor<type, 1> activation_derivatives = forward_propagation.activations_derivatives.chip(instance, 0);

                combinations_recurrent_weights_derivatives = dot(combinations_recurrent_weights_derivatives.multiply_rows(activation_derivatives), recurrent_weights);

                Index column_index = 0;
                Index activation_index = 0;

                for(Index i = 0; i < parameters_number; i++)
                {
                    combinations_recurrent_weights_derivatives(i, column_index) += current_activations[activation_index];

                    activation_index++;

                    if(activation_index == neurons_number)
                    {
                        activation_index = 0;
                        column_index++;
                    }
                }
            }

            recurrent_weights_gradient += dot(combinations_recurrent_weights_derivatives, next_layer_deltas).to_vector();
        }
    */
    return recurrent_weights_gradient;
}



Tensor<type, 1> RecurrentLayer::calculate_biases_error_gradient(const Tensor<type, 2> & inputs,
        const Layer::ForwardPropagation& layers,
        const Tensor<type, 2> & deltas)
{
    const Index instances_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();

    const Index biases_number = get_biases_number();

    // Derivatives of combinations with respect to biases

    Tensor<type, 2> combinations_biases_derivatives(biases_number, neurons_number);

    Tensor<type, 1> biases_gradient(biases_number);
    /*
        for(Index instance = 0; instance < instances_number; instance++)
        {
            const Tensor<type, 1> current_inputs = inputs.chip(instance, 0);

            const Tensor<type, 2> current_layer_deltas = deltas.get_row(instance).to_column_matrix();

            if(instance%timesteps == 0)
            {
                combinations_biases_derivatives.setZero();
            }
            else
            {
                const Tensor<type, 1> previous_activation_derivatives = layers.activations_derivatives.get_row(instance-1);

                combinations_biases_derivatives = dot(combinations_biases_derivatives.multiply_rows(previous_activation_derivatives), recurrent_weights);
            }

            combinations_biases_derivatives.sum_diagonal(1.0);

            biases_gradient += dot(combinations_biases_derivatives, current_layer_deltas).to_vector();
        }
    */
    return biases_gradient;
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


string RecurrentLayer::object_to_string() const
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    ostringstream buffer;

    buffer << "Recurrent layer" << endl;
    buffer << "Inputs number: " << inputs_number << endl;
    buffer << "Neurons number: " << neurons_number << endl;
    buffer << "Activation function: " << write_activation_function() << endl;
    buffer << "Biases:\n " << biases << endl;
    buffer << "Input weights:\n" << input_weights << endl;
    buffer << "Recurrent synaptic weights:\n" << recurrent_weights << endl;
    buffer << "Hidden states:\n" << hidden_states << endl;

    return buffer.str();
}


string RecurrentLayer::write_activation_function_expression() const
{
    switch(activation_function)
    {
    case HyperbolicTangent:
    {
        return "tanh";
    }
    case Linear:
    {
        return "";
    }
    default:
    {
        return write_activation_function();
    }
    }
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
