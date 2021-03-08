//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// OpeNN Includes

#include "long_short_term_memory_layer.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a empty layer object, with no neurons.
/// This constructor also initializes the rest of class members to their default values.

LongShortTermMemoryLayer::LongShortTermMemoryLayer() : Layer()
{
    set();

    layer_type = LongShortTermMemory;
}


/// Layer architecture constructor.
/// It creates a layer object with given numbers of inputs and neurons.
/// The parameters are initialized at random.
/// This constructor also initializes the rest of class members to their default values.
/// @param new_inputs_number Number of inputs in the layer.
/// @param new_neurons_number Number of neurons in the layer.

LongShortTermMemoryLayer::LongShortTermMemoryLayer(const Index& new_inputs_number, const Index& new_neurons_number) : Layer()
{
    set(new_inputs_number, new_neurons_number);

    layer_type = LongShortTermMemory;
}


/// Destructor.
/// This destructor does not delete any pointer.

LongShortTermMemoryLayer::~LongShortTermMemoryLayer()
{
}


/// Returns the number of inputs to the layer.

Index LongShortTermMemoryLayer::get_inputs_number() const
{
    return input_weights.dimension(0);
}


/// Returns the size of the neurons vector.

Index LongShortTermMemoryLayer::get_neurons_number() const
{
    return output_biases.size();
}


/// Returns the number of parameters (biases, weights, recurrent weights) of the layer.

Index LongShortTermMemoryLayer::get_parameters_number() const
{
    Index neurons_number = get_neurons_number();
    Index inputs_number = get_inputs_number();

    return 4 * neurons_number * (1 + inputs_number + neurons_number);
}


/// Returns the forget biases from all the lstm in the layer.
/// The format is a vector of real values.
/// The size of this vector is the number of neurons in the layer.

Tensor<type, 1> LongShortTermMemoryLayer::get_forget_biases() const
{
    return forget_biases;
}


/// Returns the input biases from all the lstm in the layer.
/// The format is a vector of real values.
/// The size of this vector is the number of neurons in the layer.

Tensor<type, 1> LongShortTermMemoryLayer::get_input_biases() const
{
    return input_biases;
}


/// Returns the state biases from all the lstm in the layer.
/// The format is a vector of real values.
/// The size of this vector is the number of neurons in the layer.

Tensor<type, 1> LongShortTermMemoryLayer::get_state_biases() const
{
    return state_biases;
}


/// Returns the output biases from all the lstm in the layer.
/// The format is a vector of real values.
/// The size of this vector is the number of neurons in the layer.

Tensor<type, 1> LongShortTermMemoryLayer::get_output_biases() const
{
    return output_biases;
}

/// Returns the forget weights from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of inputs in the layer.
/// The number of columns is the number of neurons to the layer.
///
Tensor<type, 2> LongShortTermMemoryLayer::get_forget_weights() const
{
    return forget_weights;
}

/// Returns the input weights from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of inputs in the layer.
/// The number of columns is the number of neurons to the layer.

Tensor<type, 2> LongShortTermMemoryLayer::get_input_weights() const
{
    return input_weights;
}

/// Returns the state weights from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of inputs in the layer.
/// The number of columns is the number of neurons to the layer.

Tensor<type, 2> LongShortTermMemoryLayer::get_state_weights() const
{
    return state_weights;
}

/// Returns the output weights from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of inputs in the layer.
/// The number of columns is the number of neurons to the layer.

Tensor<type, 2> LongShortTermMemoryLayer::get_output_weights() const
{
    return output_weights;
}

/// Returns the forget recurrent weights from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of columns is the number of neurons to the layer.

Tensor<type, 2> LongShortTermMemoryLayer::get_forget_recurrent_weights() const
{
    return forget_recurrent_weights;
}

/// Returns the input recurrent weights from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of columns is the number of neurons to the layer.

Tensor<type, 2> LongShortTermMemoryLayer::get_input_recurrent_weights() const
{
    return input_recurrent_weights;
}

/// Returns the state recurrent weights from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of columns is the number of neurons to the layer.

Tensor<type, 2> LongShortTermMemoryLayer::get_state_recurrent_weights() const
{
    return state_recurrent_weights;
}


/// Returns the output recurrent weights from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of columns is the number of neurons to the layer.

Tensor<type, 2> LongShortTermMemoryLayer::get_output_recurrent_weights() const
{
    return output_recurrent_weights;
}

/// Returns the number of timesteps.

Index LongShortTermMemoryLayer::get_timesteps() const
{
    return timesteps;
}


/// Returns a single vector with all the layer parameters.
/// The format is a vector of real values.
/// The size is the number of parameters in the layer.

Tensor<type, 1> LongShortTermMemoryLayer::get_parameters() const
{
    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> parameters(parameters_number);

    Index current_position = 0;

    // Biases

    for(Index i = 0; i < forget_biases.size(); i++) fill_n(parameters.data()+current_position+i, 1, forget_biases(i));

    current_position += forget_biases.size();

    for(Index i = 0; i < input_biases.size(); i++) fill_n(parameters.data()+current_position+i, 1, input_biases(i));

    current_position += input_biases.size();

    for(Index i = 0; i < state_biases.size(); i++) fill_n(parameters.data()+current_position+i, 1, state_biases(i));

    current_position += state_biases.size();

    for(Index i = 0; i < output_biases.size(); i++) fill_n(parameters.data()+current_position+i, 1, output_biases(i));

    current_position += output_biases.size();

    // Weights

    for(Index i = 0; i < forget_weights.size(); i++) fill_n(parameters.data()+current_position+i, 1, forget_weights(i));

    current_position += forget_weights.size();

    for(Index i = 0; i < input_weights.size(); i++) fill_n(parameters.data()+current_position+i, 1, input_weights(i));

    current_position += input_weights.size();

    for(Index i = 0; i < state_weights.size(); i++) fill_n(parameters.data()+current_position+i, 1, state_weights(i));

    current_position += state_weights.size();

    for(Index i = 0; i < output_weights.size(); i++) fill_n(parameters.data()+current_position+i, 1, output_weights(i));

    current_position += output_weights.size();

    // Recurrent weights

    for(Index i = 0; i < forget_recurrent_weights.size(); i++) fill_n(parameters.data()+current_position+i, 1, forget_recurrent_weights(i));

    current_position += forget_recurrent_weights.size();

    for(Index i = 0; i < input_recurrent_weights.size(); i++) fill_n(parameters.data()+current_position+i, 1, input_recurrent_weights(i));

    current_position += input_recurrent_weights.size();

    for(Index i = 0; i < state_recurrent_weights.size(); i++) fill_n(parameters.data()+current_position+i, 1, state_recurrent_weights(i));

    current_position += state_recurrent_weights.size();

    for(Index i = 0; i < output_recurrent_weights.size(); i++) fill_n(parameters.data()+current_position+i, 1, output_recurrent_weights(i));

    return parameters;
}


/// Returns the activation function of the layer.

const LongShortTermMemoryLayer::ActivationFunction& LongShortTermMemoryLayer::get_activation_function() const
{
    return activation_function;
}


/// Returns the recurrent activation function of the layer.

const LongShortTermMemoryLayer::ActivationFunction& LongShortTermMemoryLayer::get_recurrent_activation_function() const
{
    return recurrent_activation_function;
}


/// Returns a string with the name of the layer activation function.
/// This can be: Logistic, HyperbolicTangent, Threshold, SymmetricThreshold, Linear, RectifiedLinear,
/// ScaledExponentialLinear.

string LongShortTermMemoryLayer::write_activation_function() const
{
    switch(activation_function)
    {
    case Logistic: return "Logistic";

    case HyperbolicTangent: return "HyperbolicTangent";

    case Threshold: return "Threshold";

    case SymmetricThreshold: return "SymmetricThreshold";

    case Linear: return "Linear";

    case RectifiedLinear: return "RectifiedLinear";

    case ScaledExponentialLinear: return "ScaledExponentialLinear";

    case SoftPlus: return "SoftPlus";

    case SoftSign: return "SoftSign";

    case HardSigmoid: return "HardSigmoid";

    case ExponentialLinear: return "ExponentialLinear";
    }

    return string();
}



/// Returns a string with the name of the layer recurrent activation function.
/// This can be: Logistic, HyperbolicTangent, Threshold, SymmetricThreshold, Linear, RectifiedLinear, ScaledExponentialLinear.

string LongShortTermMemoryLayer::write_recurrent_activation_function() const
{
    switch(recurrent_activation_function)
    {
    case Logistic: return "Logistic";

    case HyperbolicTangent: return "HyperbolicTangent";

    case Threshold: return "Threshold";

    case SymmetricThreshold: return "SymmetricThreshold";

    case Linear: return "Linear";

    case RectifiedLinear: return "RectifiedLinear";

    case ScaledExponentialLinear: return "ScaledExponentialLinear";

    case SoftPlus: return "SoftPlus";

    case SoftSign: return "SoftSign";

    case HardSigmoid: return "HardSigmoid";

    case ExponentialLinear: return "ExponentialLinear";
    }

    return string();
}


/// Returns true if messages from this class are to be displayed on the screen,
/// or false if messages from this class are not to be displayed on the screen.

const bool& LongShortTermMemoryLayer::get_display() const
{
    return display;
}


/// Sets an empty layer, wihtout any neuron.
/// It also sets the rest of members to their default values.

void LongShortTermMemoryLayer::set()
{
    set_default();
}


/// Sets new numbers of inputs and neurons in the layer.
/// It also sets the rest of members to their default values.
/// @param new_inputs_number Number of inputs.
/// @param new_neurons_number Number of neurons.

void LongShortTermMemoryLayer::set(const Index& new_inputs_number, const Index& new_neurons_number)
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

    hidden_states.resize(new_neurons_number); // memory
    hidden_states.setZero();

    cell_states.resize(new_neurons_number); // carry
    cell_states.setZero();

    set_parameters_random();

    set_default();
}


/// Sets the members of this neuron layer object with those from other neuron layer object.
/// @param other_neuron_layer LongShortTermMemoryLayer object to be copied.

void LongShortTermMemoryLayer::set(const LongShortTermMemoryLayer& other_neuron_layer)
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

void LongShortTermMemoryLayer::set_default()
{
    layer_name = "long_short_term_memory_layer";
    layer_type = LongShortTermMemory;
}

void LongShortTermMemoryLayer::set_layer_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}


/// Sets a new number of inputs in the layer.
/// The new biases, weights and recurrent weights are initialized at random.
/// @param new_inputs_number Number of layer inputs.

void LongShortTermMemoryLayer::set_inputs_number(const Index& new_inputs_number)
{
    const Index neurons_number = get_neurons_number();

    set(new_inputs_number, neurons_number);
}


/// Sets a new size of inputs in the layer.
/// The new biases, weights and recurrent weights are initialized at random.
/// @param size dimensions of layer inputs.

void LongShortTermMemoryLayer::set_input_shape(const Tensor<Index, 1>& size)
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

void LongShortTermMemoryLayer::set_neurons_number(const Index& new_neurons_number)
{
    const Index inputs_number = get_inputs_number();

    set(inputs_number, new_neurons_number);
}


/// Sets the forget biases of all lstm in the layer from a single vector.
/// @param new_forget_biases New set of forget biases in the layer.

void LongShortTermMemoryLayer::set_forget_biases(const Tensor<type, 1>& new_biases)
{
    forget_biases = new_biases;
}


/// Sets the input biases of all lstm in the layer from a single vector.
/// @param new_input_biases New set of input biases in the layer.
///
void LongShortTermMemoryLayer::set_input_biases(const Tensor<type, 1>& new_biases)
{
    input_biases = new_biases;
}



/// Sets the state biases of all lstm in the layer from a single vector.
/// @param new_state_biases New set of state biases in the layer.

void LongShortTermMemoryLayer::set_state_biases(const Tensor<type, 1>& new_biases)
{
    state_biases = new_biases;
}



/// Sets the output biases of all lstm in the layer from a single vector.
/// @param new_output_biases New set of output biases in the layer.

void LongShortTermMemoryLayer::set_output_biases(const Tensor<type, 1>& new_biases)
{
    output_biases = new_biases;
}


/// Sets the forget weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of inputs in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_forget_weights New set of forget weights in that layer.

void LongShortTermMemoryLayer::set_forget_weights(const Tensor<type, 2>& new_forget_weights)
{

    forget_weights = new_forget_weights;

}


/// Sets the input weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of inputs in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_input_weights New set of input weights in that layer.

void LongShortTermMemoryLayer::set_input_weights(const Tensor<type, 2>& new_input_weight)
{
    input_weights = new_input_weight;
}


/// Sets the state weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of inputs in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_state_weights New set of state weights in that layer.

void LongShortTermMemoryLayer::set_state_weights(const Tensor<type, 2>& new_state_weights)
{
    state_weights = new_state_weights;
}


/// Sets the output weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of inputs in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_output_weights New set of output weights in that layer.

void LongShortTermMemoryLayer::set_output_weights(const Tensor<type, 2>& new_output_weight)
{
    output_weights = new_output_weight;

}


/// Sets the forget recurrent weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of neurons in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_forget_recurrent_weights New set of forget recurrent weights in that layer.

void LongShortTermMemoryLayer::set_forget_recurrent_weights(const Tensor<type, 2>& new_forget_recurrent_weight)
{
    forget_recurrent_weights = new_forget_recurrent_weight;
}


/// Sets the input recurrent weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of neurons in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_input_recurrent_weights New set of input recurrent weights in that layer.


void LongShortTermMemoryLayer::set_input_recurrent_weights(const Tensor<type, 2>& new_input_recurrent_weight)
{
    input_recurrent_weights = new_input_recurrent_weight;
}


/// Sets the state recurrent weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of neurons in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_state_recurrent_weights New set of state recurrent weights in that layer.


void LongShortTermMemoryLayer::set_state_recurrent_weights(const Tensor<type, 2>& new_state_recurrent_weight)
{
    state_recurrent_weights = new_state_recurrent_weight;
}


/// Sets the output recurrent weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of neurons in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_output_recurrent_weights New set of output recurrent weights in that layer.

void LongShortTermMemoryLayer::set_output_recurrent_weights(const Tensor<type, 2>& new_output_recurrent_weight)
{
    output_recurrent_weights = new_output_recurrent_weight;
}


/// Sets the parameters of this layer.
/// @param new_parameters Parameters vector for that layer.

void LongShortTermMemoryLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    const Index neurons_number = get_neurons_number();
    const Index inputs_number = get_inputs_number();

#ifdef __OPENNN_DEBUG__

    const Index parameters_number = get_parameters_number();

    const Index new_parameters_size = new_parameters.size();

    if(new_parameters_size != parameters_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void set_parameters(const Tensor<type, 1>&) method.\n"
               << "Size of new parameters (" << new_parameters_size << ") must be equal to number of parameters (" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    Index current_index = index;

    // Biases

    Index size = neurons_number;

    memcpy(forget_biases.data(),
           new_parameters.data() + current_index,
           static_cast<size_t>(size)*sizeof(type));

    current_index += size;

    memcpy(input_biases.data(),
           new_parameters.data() + current_index,
           static_cast<size_t>(size)*sizeof(type));

    current_index += size;

    memcpy(state_biases.data(),
           new_parameters.data() + current_index,
           static_cast<size_t>(size)*sizeof(type));

    current_index += size;

    memcpy(output_biases.data(),
           new_parameters.data() + current_index,
           static_cast<size_t>(size)*sizeof(type));

    current_index += size;

    // Weights

    size = inputs_number*neurons_number;

    memcpy(forget_weights.data(),
           new_parameters.data() + current_index,
           static_cast<size_t>(size)*sizeof(type));

    current_index += size;

    memcpy(input_weights.data(),
           new_parameters.data() + current_index,
           static_cast<size_t>(size)*sizeof(type));

    current_index += size;

    memcpy(state_weights.data(),
           new_parameters.data() + current_index,
           static_cast<size_t>(size)*sizeof(type));

    current_index += size;

    memcpy(output_weights.data(),
           new_parameters.data() + current_index,
           static_cast<size_t>(size)*sizeof(type));

    current_index += size;

    // Recurrent weights

    size = neurons_number*neurons_number;

    memcpy(forget_recurrent_weights.data(),
           new_parameters.data() + current_index,
           static_cast<size_t>(size)*sizeof(type));

    current_index += size;

    memcpy(input_recurrent_weights.data(),
           new_parameters.data() + current_index,
           static_cast<size_t>(size)*sizeof(type));

    current_index += size;

    memcpy(state_recurrent_weights.data(),
           new_parameters.data() + current_index,
           static_cast<size_t>(size)*sizeof(type));

    current_index += size;

    memcpy(output_recurrent_weights.data(),
           new_parameters.data() + current_index,
           static_cast<size_t>(size)*sizeof(type));

    current_index += size;

    //       set_forget_weights(new_parameters.slice(Eigen::array<Eigen::Index, 1>({0}), Eigen::array<Eigen::Index, 1>({inputs_number * neurons_number})).reshape(Eigen::array<Index, 2>({inputs_number, neurons_number})));
    //       set_input_weights(new_parameters.slice(Eigen::array<Eigen::Index, 1>({inputs_number * neurons_number}), Eigen::array<Eigen::Index, 1>({inputs_number * neurons_number})).reshape(Eigen::array<Index, 2>({inputs_number, neurons_number})));
    //       set_state_weights(new_parameters.slice(Eigen::array<Eigen::Index, 1>({2*inputs_number * neurons_number}), Eigen::array<Eigen::Index, 1>({inputs_number * neurons_number})).reshape(Eigen::array<Index, 2>({inputs_number, neurons_number})));
    //       set_input_weights(new_parameters.slice(Eigen::array<Eigen::Index, 1>({3*inputs_number * neurons_number}), Eigen::array<Eigen::Index, 1>({inputs_number * neurons_number})).reshape(Eigen::array<Index, 2>({inputs_number, neurons_number})));

    //       set_forget_recurrent_weights(new_parameters.slice(Eigen::array<Eigen::Index, 1>({4*inputs_number * neurons_number}), Eigen::array<Eigen::Index, 1>({neurons_number*neurons_number})).reshape(Eigen::array<Index, 2>({neurons_number, neurons_number})));
    //       set_input_recurrent_weights(new_parameters.slice(Eigen::array<Eigen::Index, 1>({4 * inputs_number * neurons_number + neurons_number * neurons_number}), Eigen::array<Eigen::Index, 1>({neurons_number*neurons_number})).reshape(Eigen::array<Index, 2>({neurons_number, neurons_number})));
    //       set_state_recurrent_weights(new_parameters.slice(Eigen::array<Eigen::Index, 1>({4 * inputs_number * neurons_number + 2 * neurons_number * neurons_number}), Eigen::array<Eigen::Index, 1>({neurons_number*neurons_number})).reshape(Eigen::array<Index, 2>({neurons_number, neurons_number})));
    //       set_output_recurrent_weights(new_parameters.slice(Eigen::array<Eigen::Index, 1>({4 * inputs_number * neurons_number + 3 * neurons_number * neurons_number}), Eigen::array<Eigen::Index, 1>({neurons_number*neurons_number})).reshape(Eigen::array<Index, 2>({neurons_number, neurons_number})));

    //       set_forget_biases(new_parameters.slice(Eigen::array<Eigen::Index, 1>({4 * neurons_number * (inputs_number + neurons_number)}), Eigen::array<Eigen::Index, 1>({neurons_number})));
    //       set_input_biases(new_parameters.slice(Eigen::array<Eigen::Index, 1>({4 * neurons_number * (inputs_number + neurons_number) + neurons_number}), Eigen::array<Eigen::Index, 1>({neurons_number})));
    //       set_state_biases(new_parameters.slice(Eigen::array<Eigen::Index, 1>({4 * neurons_number * (inputs_number + neurons_number) + 2 * neurons_number}), Eigen::array<Eigen::Index, 1>({neurons_number})));
    //       set_output_biases(new_parameters.slice(Eigen::array<Eigen::Index, 1>({4 * neurons_number * (inputs_number + neurons_number) + 3 * neurons_number}), Eigen::array<Eigen::Index, 1>({neurons_number})));
}


/// This class sets a new activation(or transfer) function in a single layer.
/// @param new_activation_function Activation function for the layer.

void LongShortTermMemoryLayer::set_activation_function(const LongShortTermMemoryLayer::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


/// Sets a new activation(or transfer) function in a single layer.
/// The argument is a string containing the name of the function("Logistic", "HyperbolicTangent", "Threshold", etc).
/// @param new_activation_function Activation function for that layer.

void LongShortTermMemoryLayer::set_activation_function(const string& new_activation_function_name)
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


/// This class sets a new recurrent activation(or transfer) function in a single layer.
/// @param new_recurrent_activation_function Activation function for the layer.

void LongShortTermMemoryLayer::set_recurrent_activation_function(const LongShortTermMemoryLayer::ActivationFunction& new_recurrent_activation_function)
{
    recurrent_activation_function = new_recurrent_activation_function;
}


/// Sets a new recurrent activation(or transfer) function in a single layer.
/// The argument is a string containing the name of the function("Logistic", "HyperbolicTangent", "Threshold", etc).
/// @param new_recurrent_activation_function Recurrent activation function for that layer.

void LongShortTermMemoryLayer::set_recurrent_activation_function(const string& new_recurrent_activation_function_name)
{
    if(new_recurrent_activation_function_name == "Logistic")
    {
        recurrent_activation_function = Logistic;
    }
    else if(new_recurrent_activation_function_name == "HyperbolicTangent")
    {
        recurrent_activation_function = HyperbolicTangent;
    }
    else if(new_recurrent_activation_function_name == "Threshold")
    {
        recurrent_activation_function = Threshold;
    }
    else if(new_recurrent_activation_function_name == "SymmetricThreshold")
    {
        recurrent_activation_function = SymmetricThreshold;
    }
    else if(new_recurrent_activation_function_name == "Linear")
    {
        recurrent_activation_function = Linear;
    }
    else if(new_recurrent_activation_function_name == "RectifiedLinear")
    {
        recurrent_activation_function = RectifiedLinear;
    }
    else if(new_recurrent_activation_function_name == "ScaledExponentialLinear")
    {
        recurrent_activation_function = ScaledExponentialLinear;
    }
    else if(new_recurrent_activation_function_name == "SoftPlus")
    {
        recurrent_activation_function = SoftPlus;
    }
    else if(new_recurrent_activation_function_name == "SoftSign")
    {
        recurrent_activation_function = SoftSign;
    }
    else if(new_recurrent_activation_function_name == "HardSigmoid")
    {
        recurrent_activation_function = HardSigmoid;
    }
    else if(new_recurrent_activation_function_name == "ExponentialLinear")
    {
        recurrent_activation_function = ExponentialLinear;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: neuron class.\n"
               << "void set_recurrent_activation_function(const string&) method.\n"
               << "Unknown activation function: " << new_recurrent_activation_function_name << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Sets the timesteps of the layer from a Index.
/// @param new_timesteps New set of timesteps in the layer.

void LongShortTermMemoryLayer::set_timesteps(const Index & new_timesteps)
{
    timesteps = new_timesteps;
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void LongShortTermMemoryLayer::set_display(const bool& new_display)
{
    display = new_display;
}


/// Initializes the biases of all the neurons in the layer with a given value.
/// @param value Biases initialization value.

void LongShortTermMemoryLayer::initialize_biases(const type& value)
{
    forget_biases.setConstant(value);
    input_biases.setConstant(value);
    state_biases.setConstant(value);
    output_biases.setConstant(value);
}

/// Initializes the forget biases of all the neurons in the layer with a given value.
/// @param value Forget biases initialization value.

void LongShortTermMemoryLayer::initialize_forget_biases(const type& value)
{
    forget_biases.setConstant(value);
}


/// Initializes the input biases of all the neurons in the layer with a given value.
/// @param value Input biases initialization value.

void LongShortTermMemoryLayer::initialize_input_biases(const type& value)
{
    input_biases.setConstant(value);
}


/// Initializes the state biases of all the neurons in the layer with a given value.
/// @param value State biases initialization value.

void LongShortTermMemoryLayer::initialize_state_biases(const type& value)
{
    state_biases.setConstant(value);
}


/// Initializes the oputput biases of all the neurons in the layer with a given value.
/// @param value Output biases initialization value.

void LongShortTermMemoryLayer::initialize_output_biases(const type& value)
{
    output_biases.setConstant(value);
}

/// Initializes the weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Weights initialization value.

void LongShortTermMemoryLayer::initialize_weights(const type& value)
{
    forget_weights.setConstant(value);
    input_weights.setConstant(value);
    state_weights.setConstant(value);
    output_weights.setConstant(value);
}


/// Initializes the forget weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Forget weights initialization value.

void LongShortTermMemoryLayer::initialize_forget_weights(const type& value)
{
    forget_weights.setConstant(value);
}


/// Initializes the input weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Input weights initialization value.

void LongShortTermMemoryLayer::initialize_input_weights(const type& value)
{
    input_weights.setConstant(value);
}


/// Initializes the state weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value State weights initialization value.

void LongShortTermMemoryLayer::initialize_state_weights(const type& value)
{
    state_weights.setConstant(value);
}


/// Initializes the output weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Output weights initialization value.

void LongShortTermMemoryLayer::initialize_output_weights(const type & value)
{
    output_weights.setConstant(value);
}


/// Initializes the recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Recurrent weights initialization value.

void LongShortTermMemoryLayer::initialize_recurrent_weights(const type& value)
{
    forget_recurrent_weights.setConstant(value);
    input_recurrent_weights.setConstant(value);
    state_recurrent_weights.setConstant(value);
    output_recurrent_weights.setConstant(value);
}


/// Initializes the forget recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Forget recurrent weights initialization value.

void LongShortTermMemoryLayer::initialize_forget_recurrent_weights(const type& value)
{
    forget_recurrent_weights.setConstant(value);
}


/// Initializes the input recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Input recurrent weights initialization value.

void LongShortTermMemoryLayer::initialize_input_recurrent_weights(const type& value)
{
    input_recurrent_weights.setConstant(value);
}


/// Initializes the state recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value State recurrent weights initialization value.

void LongShortTermMemoryLayer::initialize_state_recurrent_weights(const type& value)
{
    state_recurrent_weights.setConstant(value);
}


/// Initializes the output recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Output recurrent weights initialization value.

void LongShortTermMemoryLayer::initialize_output_recurrent_weights(const type & value)
{
    output_recurrent_weights.setConstant(value);
}


/// Initializes hidden states of the layer with a given value.
/// @param value Hidden states initialization value.

void LongShortTermMemoryLayer::initialize_hidden_states(const type& value)
{
    hidden_states.setConstant(value);
}


/// Initializes cell states of the layer with a given value.
/// @param value Cell states initialization value.

void LongShortTermMemoryLayer::initialize_cell_states(const type& value)
{
    cell_states.setConstant(value);
}

/// @todo
void LongShortTermMemoryLayer::set_synaptic_weights_glorot()
{
    /*
    get_weights().setRandom(minimum, maximum);
    */
}


/// Initializes all the biases, weights and recurrent weights in the neural newtork with a given value.
/// @param value Parameters initialization value.

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

    hidden_states.setZero();

    cell_states.setZero();
}


/// Initializes all the biases, weights and recurrent weights in the neural newtork at random with values comprised
/// between -1 and +1.

void LongShortTermMemoryLayer::set_parameters_random()
{
    const type minimum = -1;
    const type maximum = 1;

    // Biases

    for(Index i = 0; i < forget_biases.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        forget_biases(i) = minimum + (maximum-minimum)*random;
    }

    for(Index i = 0; i < input_biases.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        input_biases(i) = minimum + (maximum-minimum)*random;
    }

    for(Index i = 0; i < state_biases.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        state_biases(i) = minimum + (maximum-minimum)*random;
    }

    for(Index i = 0; i < output_biases.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        output_biases(i) = minimum + (maximum-minimum)*random;
    }

    // Weights

    for(Index i = 0; i < forget_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        forget_weights(i) = minimum + (maximum-minimum)*random;
    }

    for(Index i = 0; i < input_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        input_weights(i) = minimum + (maximum-minimum)*random;
    }

    for(Index i = 0; i < state_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        state_weights(i) = minimum + (maximum-minimum)*random;
    }

    for(Index i = 0; i < output_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        output_weights(i) = minimum + (maximum-minimum)*random;
    }

    // Recurrent weights

    for(Index i = 0; i < forget_recurrent_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        forget_recurrent_weights(i) = minimum + (maximum-minimum)*random;
    }

    for(Index i = 0; i < input_recurrent_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        input_recurrent_weights(i) = minimum + (maximum-minimum)*random;
    }

    for(Index i = 0; i < state_recurrent_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        state_recurrent_weights(i) = minimum + (maximum-minimum)*random;
    }

    for(Index i = 0; i < output_recurrent_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        output_recurrent_weights(i) = minimum + (maximum-minimum)*random;
    }
}


void LongShortTermMemoryLayer::calculate_forget_combinations(const Tensor<type, 1>& inputs,
                                                             const Tensor<type, 2>&forget_weights,
                                                             const Tensor<type, 2>& forget_recurrent_weights,
                                                             const Tensor<type, 1>& forget_biases,
                                                             Tensor<type, 1>&forget_combinations_1d) const
{
#ifdef __OPENNN_DEBUG__

    const Index inputs_number = get_inputs_number();

    if(inputs.size() != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 1> calculate_forget_combinations(const Tensor<type, 1>&) const method.\n"
               << "Size of layer inputs (" << inputs.size() << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    forget_combinations_1d.device(*thread_pool_device) = inputs.contract(forget_weights, AT_B).eval();

    forget_combinations_1d.device(*thread_pool_device) += forget_biases;

    forget_combinations_1d.device(*thread_pool_device) += hidden_states.contract(forget_recurrent_weights, AT_B).eval();

}


void LongShortTermMemoryLayer::calculate_input_combinations(const Tensor<type, 1>& inputs,
                                                            const Tensor<type, 2>& input_weights,
                                                            const Tensor<type, 2>& input_recurrent_weights,
                                                            const Tensor<type, 1>& input_biases,
                                                            Tensor<type, 1>& input_combinations_1d) const
{
#ifdef __OPENNN_DEBUG__

    const Index inputs_number = get_inputs_number();

    if(inputs.size() != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 1> calculate_input_combinations(const Tensor<type, 1>&) const method.\n"
               << "Size of layer inputs (" << inputs.size() << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    input_combinations_1d.device(*thread_pool_device) = inputs.contract(input_weights, AT_B).eval();

    input_combinations_1d.device(*thread_pool_device) += input_biases;

    input_combinations_1d.device(*thread_pool_device) += hidden_states.contract(input_recurrent_weights, AT_B).eval();

}


void LongShortTermMemoryLayer::calculate_state_combinations(const Tensor<type, 1>& inputs,
                                                            const Tensor<type, 2>& state_weights,
                                                            const Tensor<type, 2>& state_recurrent_weights,
                                                            const Tensor<type, 1>& state_biases,
                                                            Tensor<type, 1>& state_combinations_1d) const
{
#ifdef __OPENNN_DEBUG__

    const Index inputs_number = get_inputs_number();

    if(inputs.size() != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 1> calculate_state_combinations(const Tensor<type, 1>&) const method.\n"
               << "Size of layer inputs (" << inputs.size() << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    state_combinations_1d.device(*thread_pool_device) = inputs.contract(state_weights, AT_B).eval();

    state_combinations_1d.device(*thread_pool_device) += state_biases;

    state_combinations_1d.device(*thread_pool_device) += hidden_states.contract(state_recurrent_weights, AT_B).eval();

}


void LongShortTermMemoryLayer::calculate_output_combinations(const Tensor<type, 1>& inputs,
                                                             const Tensor<type, 2>& output_weights,
                                                             const Tensor<type, 2>& output_recurrent_weights,
                                                             const Tensor<type, 1>& output_biases,
                                                             Tensor<type, 1>& output_combinations_1d) const
{

#ifdef __OPENNN_DEBUG__

    const Index inputs_number = get_inputs_number();

    if(inputs.size() != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 1> calculate_output_combinations(const Tensor<type, 1>&) const method.\n"
               << "Size of layer inputs (" << inputs.size() << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    output_combinations_1d.device(*thread_pool_device) = inputs.contract(output_weights, AT_B).eval();

    output_combinations_1d.device(*thread_pool_device) += output_biases;

    output_combinations_1d.device(*thread_pool_device) += hidden_states.contract(output_recurrent_weights, AT_B).eval();

}


void LongShortTermMemoryLayer::calculate_activations(const Tensor<type, 2>& combinations, Tensor<type, 2>& activations_2d) const
{
#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    const Index combinations_columns_number = combinations.dimension(1);

    if(combinations_columns_number != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 2> calculate_activations(const Tensor<type, 2>&) const method.\n"
               << "Number of columns("<< combinations_columns_number <<") of combinations must be equal to number of neurons("<<neurons_number<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    switch(activation_function)
    {
    case Linear:  linear(combinations, activations_2d);
        break;

    case Logistic:  logistic(combinations, activations_2d);
        break;

    case HyperbolicTangent:  hyperbolic_tangent(combinations, activations_2d);
        break;

    case Threshold:  threshold(combinations, activations_2d);
        break;

    case SymmetricThreshold:  symmetric_threshold(combinations, activations_2d);
        break;

    case RectifiedLinear:  rectified_linear(combinations, activations_2d);
        break;

    case ScaledExponentialLinear:  scaled_exponential_linear(combinations, activations_2d);
        break;

    case SoftPlus:  soft_plus(combinations, activations_2d);
        break;

    case SoftSign:  soft_sign(combinations, activations_2d);
        break;

    case HardSigmoid:  hard_sigmoid(combinations, activations_2d);
        break;

    case ExponentialLinear:  exponential_linear(combinations, activations_2d);
        break;
    }
}


void LongShortTermMemoryLayer::calculate_activations(const Tensor<type, 1>& combinations_1d, Tensor<type, 1>& activations_1d) const
{
#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    const Index combinations_columns_number = combinations_1d.size();

    if(combinations_columns_number != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 2> calculate_activations(const Tensor<type, 1>&) const method.\n"
               << "Size of combinations must be equal to number of neurons.\n";

        throw logic_error(buffer.str());
    }

#endif

    switch(activation_function)
    {
    case Linear:  linear(combinations_1d, activations_1d);
        break;

    case Logistic:  logistic(combinations_1d, activations_1d);
        break;

    case HyperbolicTangent:  hyperbolic_tangent(combinations_1d, activations_1d);
        break;

    case Threshold:  threshold(combinations_1d, activations_1d);
        break;

    case SymmetricThreshold:  symmetric_threshold(combinations_1d, activations_1d);
        break;

    case RectifiedLinear:  rectified_linear(combinations_1d, activations_1d);
        break;

    case ScaledExponentialLinear:  scaled_exponential_linear(combinations_1d, activations_1d);
        break;

    case SoftPlus:  soft_plus(combinations_1d, activations_1d);
        break;

    case SoftSign:  soft_sign(combinations_1d, activations_1d);
        break;

    case HardSigmoid:  hard_sigmoid(combinations_1d, activations_1d);
        break;

    case ExponentialLinear:  exponential_linear(combinations_1d, activations_1d);
        break;
    }
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_activations(const Tensor<type, 1>& combinations_1d) const
{
#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    const Index combinations_columns_number = combinations_1d.size();

    if(combinations_columns_number != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 2> calculate_activations(const Tensor<type, 1>&) const method.\n"
               << "Size of combinations must be equal to number of neurons.\n";

        throw logic_error(buffer.str());
    }

#endif

    Tensor<type, 1> activations_1d(combinations_1d.size());

    switch(activation_function)
    {
    case Linear:  linear(combinations_1d, activations_1d);
        break;

    case Logistic:  logistic(combinations_1d, activations_1d);
        break;

    case HyperbolicTangent:  hyperbolic_tangent(combinations_1d, activations_1d);
        break;

    case Threshold:  threshold(combinations_1d, activations_1d);
        break;

    case SymmetricThreshold:  symmetric_threshold(combinations_1d, activations_1d);
        break;

    case RectifiedLinear:  rectified_linear(combinations_1d, activations_1d);
        break;

    case ScaledExponentialLinear:  scaled_exponential_linear(combinations_1d, activations_1d);
        break;

    case SoftPlus:  soft_plus(combinations_1d, activations_1d);
        break;

    case SoftSign:  soft_sign(combinations_1d, activations_1d);
        break;

    case HardSigmoid:  hard_sigmoid(combinations_1d, activations_1d);
        break;

    case ExponentialLinear:  exponential_linear(combinations_1d, activations_1d);
        break;
    }

    return activations_1d;
}

void LongShortTermMemoryLayer::calculate_recurrent_activations(const Tensor<type, 2>& combinations, Tensor<type, 2>& activations_2d) const
{
#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    const Index combinations_columns_number = combinations.dimension(2);

    if(combinations_columns_number != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 2> calculate_recurrent_activations(const Tensor<type, 2>&) const method.\n"
               << "Number of columns("<< combinations_columns_number <<") of combinations must be equal to number of neurons("<<neurons_number<<").\n";

        throw logic_error(buffer.str());
    }

#endif


    switch(recurrent_activation_function)
    {
    case Linear:  linear(combinations, activations_2d);
        break;

    case Logistic:  logistic(combinations, activations_2d);
        break;

    case HyperbolicTangent:  hyperbolic_tangent(combinations, activations_2d);
        break;

    case Threshold:  threshold(combinations, activations_2d);
        break;

    case SymmetricThreshold:  symmetric_threshold(combinations, activations_2d);
        break;

    case RectifiedLinear:  rectified_linear(combinations, activations_2d);
        break;

    case ScaledExponentialLinear:  scaled_exponential_linear(combinations, activations_2d);
        break;

    case SoftPlus:  soft_plus(combinations, activations_2d);
        break;

    case SoftSign:  soft_sign(combinations, activations_2d);
        break;

    case HardSigmoid:  hard_sigmoid(combinations, activations_2d);
        break;

    case ExponentialLinear:  exponential_linear(combinations, activations_2d);
        break;
    }
}


void LongShortTermMemoryLayer::calculate_recurrent_activations(const Tensor<type, 1>& combinations_1d, Tensor<type, 1>& recurrent_activations_1d) const
{

#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    const Index combinations_columns_number = combinations_1d.size();

    if(combinations_columns_number != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 2> calculate_activations(const Tensor<type, 2>&) const method.\n"
               << "Size of combinations must be equal to number of neurons.\n";

        throw logic_error(buffer.str());
    }

#endif

    switch(recurrent_activation_function)
    {
    case Linear:  linear(combinations_1d, recurrent_activations_1d);
        break;

    case Logistic:  logistic(combinations_1d, recurrent_activations_1d);
        break;

    case HyperbolicTangent:  hyperbolic_tangent(combinations_1d, recurrent_activations_1d);
        break;

    case Threshold:  threshold(combinations_1d, recurrent_activations_1d);
        break;

    case SymmetricThreshold:  symmetric_threshold(combinations_1d, recurrent_activations_1d);
        break;

    case RectifiedLinear:  rectified_linear(combinations_1d, recurrent_activations_1d);
        break;

    case ScaledExponentialLinear:  scaled_exponential_linear(combinations_1d, recurrent_activations_1d);
        break;

    case SoftPlus:  soft_plus(combinations_1d, recurrent_activations_1d);
        break;

    case SoftSign:  soft_sign(combinations_1d, recurrent_activations_1d);
        break;

    case HardSigmoid:  hard_sigmoid(combinations_1d, recurrent_activations_1d);
        break;

    case ExponentialLinear:  exponential_linear(combinations_1d, recurrent_activations_1d);
        break;
    }
}


void LongShortTermMemoryLayer::calculate_activations_derivatives(const Tensor<type, 2>& combinations,
                                                                 Tensor<type, 2>& activations_2d,
                                                                 Tensor<type, 2>& activations_derivatives_2d) const
{
#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    const Index combinations_columns_number = combinations.dimension(1);

    if(combinations_columns_number != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 2> calculate_activations_derivatives(const Tensor<type, 2>&) const method.\n"
               << "Number of columns("<< combinations_columns_number <<") of combinations must be equal to number of neurons("<<neurons_number<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    switch(activation_function)
    {
    case Linear: return linear_derivatives(combinations, activations_2d, activations_derivatives_2d);

    case Logistic: return logistic_derivatives(combinations, activations_2d, activations_derivatives_2d);

    case HyperbolicTangent: return hyperbolic_tangent_derivatives(combinations, activations_2d, activations_derivatives_2d);

    case Threshold: return threshold_derivatives(combinations, activations_2d, activations_derivatives_2d);

    case SymmetricThreshold: return symmetric_threshold_derivatives(combinations, activations_2d, activations_derivatives_2d);

    case RectifiedLinear: return rectified_linear_derivatives(combinations, activations_2d, activations_derivatives_2d);

    case ScaledExponentialLinear: return scaled_exponential_linear_derivatives(combinations, activations_2d, activations_derivatives_2d);

    case SoftPlus: return soft_plus_derivatives(combinations, activations_2d, activations_derivatives_2d);

    case SoftSign: return soft_sign_derivatives(combinations, activations_2d, activations_derivatives_2d);

    case HardSigmoid: return hard_sigmoid_derivatives(combinations, activations_2d, activations_derivatives_2d);

    case ExponentialLinear: return exponential_linear_derivatives(combinations, activations_2d, activations_derivatives_2d);
    }

}



void LongShortTermMemoryLayer::calculate_activations_derivatives(const Tensor<type, 1>& combinations_1d,
                                                                 Tensor<type, 1>& activations_1d,
                                                                 Tensor<type, 1>& activations_derivatives_1d) const
{

#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    const Index combinations_columns_number = combinations_1d.size();

    if(combinations_columns_number != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 2> calculate_activations_derivatives(const Tensor<type, 2>&) const method.\n"
               << "Size of combinations must be equal to number of neurons.\n";

        throw logic_error(buffer.str());
    }

#endif

    switch(activation_function)
    {

    case Linear: return linear_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case Logistic: return logistic_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case HyperbolicTangent: return hyperbolic_tangent_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case Threshold: return threshold_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case SymmetricThreshold: return symmetric_threshold_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case RectifiedLinear: return rectified_linear_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case ScaledExponentialLinear: return scaled_exponential_linear_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case SoftPlus: return soft_plus_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case SoftSign: return soft_sign_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case HardSigmoid: return hard_sigmoid_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case ExponentialLinear: return exponential_linear_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    }
}

void LongShortTermMemoryLayer::calculate_recurrent_activations_derivatives(const Tensor<type, 1>& combinations_1d,
                                                                           Tensor<type, 1>& activations_1d,
                                                                           Tensor<type, 1>& activations_derivatives_1d) const
{
#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    const Index combinations_columns_number = combinations_1d.size();

    if(combinations_columns_number != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 2> calculate_recurrent_activations_derivatives(const Tensor<type, 2>&) const method.\n"
               << "Size of combinations must be equal to number of neurons.\n";

        throw logic_error(buffer.str());
    }

#endif

    switch(recurrent_activation_function)
    {
    case Linear: return linear_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case Logistic: return logistic_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case HyperbolicTangent: return hyperbolic_tangent_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case Threshold: return threshold_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case SymmetricThreshold: return symmetric_threshold_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case RectifiedLinear: return rectified_linear_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case ScaledExponentialLinear: return scaled_exponential_linear_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case SoftPlus: return soft_plus_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case SoftSign: return soft_sign_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case HardSigmoid: return hard_sigmoid_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    case ExponentialLinear: return exponential_linear_derivatives(combinations_1d, activations_1d, activations_derivatives_1d);

    }
}


Tensor<type, 2> LongShortTermMemoryLayer::calculate_outputs(const Tensor<type, 2>& inputs)
{
#ifdef __OPENNN_DEBUG__

    const Index inputs_number = get_inputs_number();

    const Index inputs_columns_number = inputs.dimension(1);

    if(inputs_columns_number != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
               << "Number of columns ("<<inputs_columns_number<<") of inputs matrix must be equal to number of inputs ("<<inputs_number<<").\n";

        throw logic_error(buffer.str());
    }
#endif

    const Index samples_number = inputs.dimension(0);

    const Index neurons_number = get_neurons_number();

    Tensor<type, 2> outputs(samples_number, neurons_number);

    Tensor<type, 1> forget_combinations(neurons_number);
    Tensor<type, 1> forget_activations(neurons_number);

    Tensor<type, 1> input_combinations(neurons_number);
    Tensor<type, 1> input_activations(neurons_number);

    Tensor<type, 1> state_combinations(neurons_number);
    Tensor<type, 1> state_activations(neurons_number);

    Tensor<type, 1> output_combinations(neurons_number);
    Tensor<type, 1> output_activations(neurons_number);

    for(Index i = 0; i < samples_number; i++)
    {
        if(i%timesteps == 0)
        {
            hidden_states.setZero();
            cell_states.setZero();
        }

        const Tensor<type, 1> current_inputs = inputs.chip(i, 0);

#pragma omp parallel
        {
            calculate_forget_combinations(current_inputs, forget_weights, forget_recurrent_weights, forget_biases, forget_combinations);
            calculate_recurrent_activations(forget_combinations, forget_activations);

            calculate_input_combinations(current_inputs, input_weights, input_recurrent_weights, input_biases, input_combinations);
            calculate_recurrent_activations(input_combinations, input_activations);

            calculate_state_combinations(current_inputs, state_weights, state_recurrent_weights, state_biases, state_combinations);
            calculate_activations(state_combinations, state_activations);

            calculate_output_combinations(current_inputs, output_weights, output_recurrent_weights, output_biases, output_combinations);
            calculate_recurrent_activations(output_combinations, output_activations);
        }

        cell_states = forget_activations * cell_states + input_activations * state_activations;
        calculate_activations(cell_states, hidden_states);
        hidden_states *= output_activations;

        for(Index j = 0; j < neurons_number; j++)
            outputs(i,j) = hidden_states(j);
    }

    return  outputs;
}


//void LongShortTermMemoryLayer::calculate_output_delta(ForwardPropagation* ,
//                                                      const Tensor<type, 2>& output_jacobian,
//                                                      Tensor<type, 2>& output_delta) const
//{
//    output_delta.device(*thread_pool_device) = output_jacobian;
//}

/*
void LongShortTermMemoryLayer::calculate_output_delta(ForwardPropagation* ,
                                                      const Tensor<type, 2>& output_jacobian,
                                                      BackPropagation* back_propagation) const
{
    LongShortTermMemoryLayerBackPropagation* long_short_term_memory_layer_back_propagation =
            static_cast<LongShortTermMemoryLayerBackPropagation*>(back_propagation);

    long_short_term_memory_layer_back_propagation->delta.device(*thread_pool_device) = output_jacobian;
}
*/

void LongShortTermMemoryLayer::calculate_hidden_delta(ForwardPropagation* forward_propagation,
                                                      BackPropagation* next_back_propagation,
                                                      BackPropagation* back_propagation) const
{
    LongShortTermMemoryLayerForwardPropagation* long_short_term_memory_layer_forward_propagation =
            static_cast<LongShortTermMemoryLayerForwardPropagation*>(forward_propagation);

    LongShortTermMemoryLayerBackPropagation* long_short_term_memory_layer_back_propagation =
            static_cast<LongShortTermMemoryLayerBackPropagation*>(back_propagation);

    switch(next_back_propagation->layer_pointer->get_type())
    {
    case Perceptron:
    {
        PerceptronLayer::PerceptronLayerBackPropagation* next_perceptron_layer_back_propagation =
                static_cast<PerceptronLayer::PerceptronLayerBackPropagation*>(next_back_propagation);

        calculate_hidden_delta_perceptron(long_short_term_memory_layer_forward_propagation,
                                          next_perceptron_layer_back_propagation,
                                          long_short_term_memory_layer_back_propagation);

    }
        break;

    case Probabilistic:
    {
        ProbabilisticLayer::ProbabilisticLayerBackPropagation* next_probabilistic_layer_back_propagation =
                static_cast<ProbabilisticLayer::ProbabilisticLayerBackPropagation*>(next_back_propagation);

        calculate_hidden_delta_probabilistic(long_short_term_memory_layer_forward_propagation,
                                             next_probabilistic_layer_back_propagation,
                                             long_short_term_memory_layer_back_propagation);

    }
        break;

    default: return;
    }
}


void LongShortTermMemoryLayer::calculate_hidden_delta_perceptron(LongShortTermMemoryLayerForwardPropagation* ,
                                                                 PerceptronLayer::PerceptronLayerBackPropagation* next_back_propagation,
                                                                 LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Tensor<type, 2>& next_synaptic_weights = static_cast<PerceptronLayer*>(back_propagation->layer_pointer)->get_synaptic_weights();

    back_propagation->delta.device(*thread_pool_device) = next_back_propagation->delta.contract(next_synaptic_weights, A_BT);
}


void LongShortTermMemoryLayer::calculate_hidden_delta_probabilistic(LongShortTermMemoryLayerForwardPropagation* ,
                                                                    ProbabilisticLayer::ProbabilisticLayerBackPropagation* next_back_propagation,
                                                                    LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Tensor<type, 2>& next_synaptic_weights = static_cast<ProbabilisticLayer*>(back_propagation->layer_pointer)->get_synaptic_weights();

    back_propagation->delta.device(*thread_pool_device) = next_back_propagation->delta.contract(next_synaptic_weights, A_BT);
}


/*
void LongShortTermMemoryLayer::calculate_hidden_delta(Layer* next_layer_pointer,
                                                      ForwardPropagation* ,
                                                      const Tensor<type, 2>& next_layer_delta,
                                                      Tensor<type, 2>& hidden_delta) const
{
    const Type next_layer_type = next_layer_pointer->get_type();

    switch (next_layer_type)
    {
        case Perceptron:

        calculate_hidden_delta_perceptron(next_layer_pointer, next_layer_delta, hidden_delta);

        return;

        case Probabilistic:

        calculate_hidden_delta_probabilistic(next_layer_pointer, next_layer_delta, hidden_delta);

        return;

        default:
        return;
    }
}


void LongShortTermMemoryLayer::calculate_hidden_delta_perceptron(Layer* next_layer_pointer,
                                                                 const Tensor<type, 2>& next_layer_delta,
                                                                 Tensor<type, 2>& hidden_delta) const
{
    const PerceptronLayer* next_perceptron_layer = dynamic_cast<PerceptronLayer*>(next_layer_pointer);

    const Tensor<type, 2>& next_synaptic_weights = next_perceptron_layer->get_synaptic_weights();

    hidden_delta.device(*thread_pool_device) = next_layer_delta.contract(next_synaptic_weights, A_BT);
}


void LongShortTermMemoryLayer::calculate_hidden_delta_probabilistic(Layer* next_layer_pointer,
                                                                    const Tensor<type, 2>& next_layer_delta,
                                                                    Tensor<type, 2>& hidden_delta) const
{
    const ProbabilisticLayer* next_probabilistic_layer = dynamic_cast<ProbabilisticLayer*>(next_layer_pointer);

    const Tensor<type, 2>& next_synaptic_weights = next_probabilistic_layer->get_synaptic_weights();

    hidden_delta.device(*thread_pool_device) = next_layer_delta.contract(next_synaptic_weights, A_BT);
}
*/

void LongShortTermMemoryLayer::forward_propagate(const Tensor<type, 2> &inputs, ForwardPropagation* forward_propagation)
{
    LongShortTermMemoryLayerForwardPropagation* long_short_term_memory_layer_forward_propagation = static_cast<LongShortTermMemoryLayerForwardPropagation*>(forward_propagation);

    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();

//    Tensor<type, 1> forget_combinations(neurons_number);
//    Tensor<type, 1> input_combinations(neurons_number);
//    Tensor<type, 1> state_combinations(neurons_number);
//    Tensor<type, 1> output_combinations(neurons_number);

//    Tensor<type, 1> forget_activations(neurons_number);
//    Tensor<type, 1> input_activations(neurons_number);
//    Tensor<type, 1> state_activations(neurons_number);
//    Tensor<type, 1> output_activations(neurons_number);

//    Tensor<type, 1> forget_activations_derivatives(neurons_number);
//    Tensor<type, 1> input_activations_derivatives(neurons_number);
//    Tensor<type, 1> state_activations_derivatives(neurons_number);
//    Tensor<type, 1> output_activations_derivatives(neurons_number);

//    Tensor<type, 1> hidden_states_derivatives(neurons_number);

    Index activations_copy_index = 0;
    Index derivatives_copy_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(i%timesteps == 0)
        {
            hidden_states.setZero();
            cell_states.setZero();
        }

        const Tensor<type, 1> current_inputs = inputs.chip(i,0);

        calculate_forget_combinations(current_inputs, forget_weights, forget_recurrent_weights, forget_biases, long_short_term_memory_layer_forward_propagation->current_forget_combinations);
        calculate_recurrent_activations_derivatives(long_short_term_memory_layer_forward_propagation->current_forget_combinations,
                                                    long_short_term_memory_layer_forward_propagation->current_forget_activations,
                                                    long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives);

        calculate_input_combinations(current_inputs, input_weights, input_recurrent_weights, input_biases, long_short_term_memory_layer_forward_propagation->current_input_combinations);
        calculate_recurrent_activations_derivatives(long_short_term_memory_layer_forward_propagation->current_input_combinations,
                                                    long_short_term_memory_layer_forward_propagation->current_input_activations,
                                                    long_short_term_memory_layer_forward_propagation->current_input_activations_derivatives);

        calculate_state_combinations(current_inputs, state_weights, state_recurrent_weights, state_biases, long_short_term_memory_layer_forward_propagation->current_state_combinations);
        calculate_recurrent_activations_derivatives(long_short_term_memory_layer_forward_propagation->current_state_combinations,
                                                    long_short_term_memory_layer_forward_propagation->current_state_activations,
                                                    long_short_term_memory_layer_forward_propagation->current_state_activations_derivatives);

        calculate_output_combinations(current_inputs, output_weights, output_recurrent_weights, output_biases, long_short_term_memory_layer_forward_propagation->current_output_combinations);
        calculate_recurrent_activations_derivatives(long_short_term_memory_layer_forward_propagation->current_output_combinations,
                                                    long_short_term_memory_layer_forward_propagation->current_output_activations,
                                                    long_short_term_memory_layer_forward_propagation->current_output_activations_derivatives);

        cell_states = long_short_term_memory_layer_forward_propagation->current_forget_activations * cell_states +
                long_short_term_memory_layer_forward_propagation->current_input_activations * long_short_term_memory_layer_forward_propagation->current_state_activations;
        calculate_activations_derivatives(cell_states, hidden_states, long_short_term_memory_layer_forward_propagation->current_hidden_states_derivatives);
        hidden_states *= long_short_term_memory_layer_forward_propagation->current_output_activations;

        // Activations 2d

        for(Index j = 0; j < neurons_number; j++) long_short_term_memory_layer_forward_propagation->activations(i,j) = hidden_states(j);

        // Forget (activations and activations derivatives)

        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_3d.data() + activations_copy_index,
               long_short_term_memory_layer_forward_propagation->current_forget_activations.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));
        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index,
               long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));

        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;


        // Input (activations and activations derivatives)

        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_3d.data() + activations_copy_index,
               long_short_term_memory_layer_forward_propagation->current_input_activations.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));
        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index,
               long_short_term_memory_layer_forward_propagation->current_input_activations_derivatives.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));

        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        // State (activations and activations derivatives)

        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_3d.data() + activations_copy_index,
               long_short_term_memory_layer_forward_propagation->current_state_activations.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));
        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index,
               long_short_term_memory_layer_forward_propagation->current_state_activations_derivatives.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));

        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        // Output (activations and activations derivatives)

        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_3d.data() + activations_copy_index,
               long_short_term_memory_layer_forward_propagation->current_output_activations.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));
        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index,
               long_short_term_memory_layer_forward_propagation->current_output_activations_derivatives.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));

        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        // Cell states (activations)

        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_3d.data() + activations_copy_index,
               cell_states.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));

        activations_copy_index += neurons_number;

        // Hidden states (activations and activations derivatives)

        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_3d.data() + activations_copy_index,
               hidden_states.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));

        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index,
               long_short_term_memory_layer_forward_propagation->current_hidden_states_derivatives.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));

        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;
    }
}


void LongShortTermMemoryLayer::forward_propagate(const Tensor<type, 2>& inputs, Tensor<type, 1> parameters, ForwardPropagation* forward_propagation)
{
    LongShortTermMemoryLayerForwardPropagation* long_short_term_memory_layer_forward_propagation = static_cast<LongShortTermMemoryLayerForwardPropagation*>(forward_propagation);

    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    const TensorMap<Tensor<type, 1>> forget_biases(parameters.data(), neurons_number);
    const TensorMap<Tensor<type, 1>> input_biases(parameters.data()+neurons_number, neurons_number);
    const TensorMap<Tensor<type, 1>> state_biases(parameters.data()+2*neurons_number, neurons_number);
    const TensorMap<Tensor<type, 1>> output_biases(parameters.data()+3*neurons_number, neurons_number);

    const TensorMap<Tensor<type, 2>> forget_weights(parameters.data()+4*neurons_number, inputs_number, neurons_number);
    const TensorMap<Tensor<type, 2>> input_weights(parameters.data()+4*neurons_number+inputs_number*neurons_number, inputs_number, neurons_number);
    const TensorMap<Tensor<type, 2>> state_weights(parameters.data()+4*neurons_number+2*inputs_number*neurons_number, inputs_number, neurons_number);
    const TensorMap<Tensor<type, 2>> output_weights(parameters.data()+4*neurons_number+3*inputs_number*neurons_number, inputs_number, neurons_number);

    const TensorMap<Tensor<type, 2>> forget_recurrent_weights(parameters.data()+4*neurons_number+4*inputs_number*neurons_number, neurons_number, neurons_number);
    const TensorMap<Tensor<type, 2>> input_recurrent_weights(parameters.data()+4*neurons_number+4*inputs_number*neurons_number+neurons_number*neurons_number, neurons_number, neurons_number);
    const TensorMap<Tensor<type, 2>> state_recurrent_weights(parameters.data()+4*neurons_number+4*inputs_number*neurons_number+2*neurons_number*neurons_number, neurons_number, neurons_number);
    const TensorMap<Tensor<type, 2>> output_recurrent_weights(parameters.data()+4*neurons_number+4*inputs_number*neurons_number+3*neurons_number*neurons_number, neurons_number, neurons_number);

    const Index samples_number = inputs.dimension(0);

    Tensor<type, 1> forget_combinations(neurons_number);
    Tensor<type, 1> input_combinations(neurons_number);
    Tensor<type, 1> state_combinations(neurons_number);
    Tensor<type, 1> output_combinations(neurons_number);

    Tensor<type, 1> forget_activations(neurons_number);
    Tensor<type, 1> input_activations(neurons_number);
    Tensor<type, 1> state_activations(neurons_number);
    Tensor<type, 1> output_activations(neurons_number);

    Tensor<type, 1> forget_activations_derivatives(neurons_number);
    Tensor<type, 1> input_activations_derivatives(neurons_number);
    Tensor<type, 1> state_activations_derivatives(neurons_number);
    Tensor<type, 1> output_activations_derivatives(neurons_number);

    Tensor<type, 1> hidden_states_derivatives(neurons_number);

    Index activations_copy_index = 0;
    Index derivatives_copy_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(i%timesteps == 0)
        {
            hidden_states.setZero();
            cell_states.setZero();
        }

        const Tensor<type, 1> current_inputs = inputs.chip(i,0);

        calculate_forget_combinations(current_inputs, forget_weights, forget_recurrent_weights, forget_biases, forget_combinations);
        calculate_recurrent_activations_derivatives(forget_combinations, forget_activations, forget_activations_derivatives);

        calculate_input_combinations(current_inputs, input_weights, input_recurrent_weights, input_biases, input_combinations);
        calculate_recurrent_activations_derivatives(input_combinations, input_activations, input_activations_derivatives);

        calculate_state_combinations(current_inputs, state_weights, state_recurrent_weights, state_biases, state_combinations);
        calculate_recurrent_activations_derivatives(state_combinations, state_activations, state_activations_derivatives);

        calculate_output_combinations(current_inputs, output_weights, output_recurrent_weights, output_biases, output_combinations);
        calculate_recurrent_activations_derivatives(output_combinations, output_activations, output_activations_derivatives);

        cell_states = forget_activations * cell_states + input_activations * state_activations;
        calculate_activations_derivatives(cell_states, hidden_states, hidden_states_derivatives);

        hidden_states *= output_activations;

        // Activations 2d

        for(Index j = 0; j < neurons_number; j++) long_short_term_memory_layer_forward_propagation->activations(i,j) = hidden_states(j);

        // Combinations?

        // Forget (activations and activations derivatives)

        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_3d.data() + activations_copy_index,
               forget_activations.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));
        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index,
               forget_activations_derivatives.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));

        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        // Input (activations and activations derivatives)

        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_3d.data() + activations_copy_index,
               input_activations.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));
        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index,
               input_activations_derivatives.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));

        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        // State (activations and activations derivatives)

        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_3d.data() + activations_copy_index,
               state_activations.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));
        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index,
               state_activations_derivatives.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));

        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        // Output (activations and activations derivatives)

        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_3d.data() + activations_copy_index,
               output_activations.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));
        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index,
               output_activations_derivatives.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));

        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        // Cell states activations

        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_3d.data() + activations_copy_index,
               cell_states.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));

        activations_copy_index += neurons_number;

        // Hidden states (activations and activations derivatives)

        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_3d.data() + activations_copy_index,
               hidden_states.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));

        memcpy(long_short_term_memory_layer_forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index,
               hidden_states_derivatives.data(),
               static_cast<size_t>(neurons_number)*sizeof (type));

        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;
    }
}


void LongShortTermMemoryLayer::insert_gradient(BackPropagation* back_propagation,
                                               const Index& index,
                                               Tensor<type, 1>& gradient) const
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    LongShortTermMemoryLayerBackPropagation* long_short_term_memory_layer_back_propagation =
            static_cast<LongShortTermMemoryLayerBackPropagation*>(back_propagation);

    // Biases

    memcpy(gradient.data() + index,
           long_short_term_memory_layer_back_propagation->forget_biases_derivatives.data(),
           static_cast<size_t>(neurons_number)*sizeof(type));

    memcpy(gradient.data() + index + neurons_number,
           long_short_term_memory_layer_back_propagation->input_biases_derivatives.data(),
           static_cast<size_t>(neurons_number)*sizeof(type));

    memcpy(gradient.data() + index + 2*neurons_number,
           long_short_term_memory_layer_back_propagation->state_biases_derivatives.data(),
           static_cast<size_t>(neurons_number)*sizeof(type));

    memcpy(gradient.data() + index + 3*neurons_number,
           long_short_term_memory_layer_back_propagation->output_biases_derivatives.data(),
           static_cast<size_t>(neurons_number)*sizeof(type));

    // Weights

    memcpy(gradient.data() + index + 4*neurons_number,
           long_short_term_memory_layer_back_propagation->forget_weights_derivatives.data(),
           static_cast<size_t>(inputs_number*neurons_number)*sizeof(type));

    memcpy(gradient.data() + index + 4*neurons_number + inputs_number*neurons_number,
           long_short_term_memory_layer_back_propagation->input_weights_derivatives.data(),
           static_cast<size_t>(inputs_number*neurons_number)*sizeof(type));

    memcpy(gradient.data() + index + 4*neurons_number + 2*inputs_number*neurons_number,
           long_short_term_memory_layer_back_propagation->state_weights_derivatives.data(),
           static_cast<size_t>(inputs_number*neurons_number)*sizeof(type));

    memcpy(gradient.data() + index + 4*neurons_number + 3*inputs_number*neurons_number,
           long_short_term_memory_layer_back_propagation->output_weights_derivatives.data(),
           static_cast<size_t>(inputs_number*neurons_number)*sizeof(type));

    // Recurrent weights

    memcpy(gradient.data() + index + 4*neurons_number + 4*inputs_number*neurons_number,
           long_short_term_memory_layer_back_propagation->forget_recurrent_weights_derivatives.data(),
           static_cast<size_t>(neurons_number*neurons_number)*sizeof(type));

    memcpy(gradient.data() + index + 4*neurons_number + 4*inputs_number*neurons_number + neurons_number*neurons_number,
           long_short_term_memory_layer_back_propagation->input_recurrent_weights_derivatives.data(),
           static_cast<size_t>(neurons_number*neurons_number)*sizeof(type));

    memcpy(gradient.data() + index + 4*neurons_number + 4*inputs_number*neurons_number + 2*neurons_number*neurons_number,
           long_short_term_memory_layer_back_propagation->state_recurrent_weights_derivatives.data(),
           static_cast<size_t>(neurons_number*neurons_number)*sizeof(type));

    memcpy(gradient.data() + index + 4*neurons_number + 4*inputs_number*neurons_number + 3*neurons_number*neurons_number,
           long_short_term_memory_layer_back_propagation->output_recurrent_weights_derivatives.data(),
           static_cast<size_t>(neurons_number*neurons_number)*sizeof(type));

}


void LongShortTermMemoryLayer::calculate_error_gradient(const Tensor<type, 2> &  inputs,
                                                        ForwardPropagation* forward_propagation,
                                                        BackPropagation* back_propagation) const
{
    LongShortTermMemoryLayerForwardPropagation* long_short_term_memory_layer_forward_propagation =
            static_cast<LongShortTermMemoryLayerForwardPropagation*>(forward_propagation);

    LongShortTermMemoryLayerBackPropagation* long_short_term_memory_layer_back_propagation =
            static_cast<LongShortTermMemoryLayerBackPropagation*>(back_propagation);

#pragma omp parallel
    {
        // Biases

        calculate_forget_biases_error_gradient(inputs, long_short_term_memory_layer_forward_propagation, long_short_term_memory_layer_back_propagation);

        calculate_input_biases_error_gradient(inputs, long_short_term_memory_layer_forward_propagation, long_short_term_memory_layer_back_propagation);

        calculate_state_biases_error_gradient(inputs, long_short_term_memory_layer_forward_propagation, long_short_term_memory_layer_back_propagation);

        calculate_output_biases_error_gradient(inputs, long_short_term_memory_layer_forward_propagation, long_short_term_memory_layer_back_propagation);

        // Weights

        calculate_forget_weights_error_gradient(inputs, long_short_term_memory_layer_forward_propagation, long_short_term_memory_layer_back_propagation);

        calculate_input_weights_error_gradient(inputs, long_short_term_memory_layer_forward_propagation, long_short_term_memory_layer_back_propagation);

        calculate_state_weights_error_gradient(inputs, long_short_term_memory_layer_forward_propagation, long_short_term_memory_layer_back_propagation);

        calculate_output_weights_error_gradient(inputs, long_short_term_memory_layer_forward_propagation, long_short_term_memory_layer_back_propagation);

        // Recurrent weights

        calculate_forget_recurrent_weights_error_gradient(inputs, long_short_term_memory_layer_forward_propagation, long_short_term_memory_layer_back_propagation);

        calculate_input_recurrent_weights_error_gradient(inputs, long_short_term_memory_layer_forward_propagation, long_short_term_memory_layer_back_propagation);

        calculate_state_recurrent_weights_error_gradient(inputs, long_short_term_memory_layer_forward_propagation, long_short_term_memory_layer_back_propagation);

        calculate_output_recurrent_weights_error_gradient(inputs, long_short_term_memory_layer_forward_propagation, long_short_term_memory_layer_back_propagation);
    }
}


void LongShortTermMemoryLayer::calculate_forget_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                       LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                       LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = inputs_number*neurons_number;

    Tensor<type, 2> input_combinations_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> forget_combinations_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> state_combinations_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> output_combinations_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 2> hidden_states_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> cell_state_weights_derivatives(parameters_number, neurons_number);

    input_combinations_weights_derivatives.setZero();
    forget_combinations_weights_derivatives.setZero();
    state_combinations_weights_derivatives.setZero();
    output_combinations_weights_derivatives.setZero();
    hidden_states_weights_derivatives.setZero();
    cell_state_weights_derivatives.setZero();

    Index column_index = 0;
    Index input_index = 0;

    Index activations_copy_index = 0;
    Index derivatives_copy_index = 0;

    Tensor<type, 1> current_forget_activations(neurons_number);
    Tensor<type, 1> current_forget_derivatives(neurons_number);
    Tensor<type, 1> current_input_activations(neurons_number);
    Tensor<type, 1> current_input_derivatives(neurons_number);
    Tensor<type, 1> current_state_activations(neurons_number);
    Tensor<type, 1> current_state_derivatives(neurons_number);
    Tensor<type, 1> current_output_activations(neurons_number);
    Tensor<type, 1> current_output_derivatives(neurons_number);
    Tensor<type, 1> current_cell_state_activations(neurons_number);
    Tensor<type, 1> current_hidden_derivatives(neurons_number);
    Tensor<type, 1> previous_cell_state_activations(neurons_number);

    back_propagation->forget_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_inputs = inputs.chip(sample, 0); // memcpy?
        const Tensor<type, 1> current_layer_deltas = back_propagation->delta.chip(sample,0); // memcpy?

        memcpy(current_forget_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_forget_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_input_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_input_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_state_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_output_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_output_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_cell_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += 2*neurons_number; // // 2* because of hidden state activations

        memcpy(current_hidden_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        derivatives_copy_index += neurons_number;

        if(sample%timesteps == 0)
        {
            previous_cell_state_activations.setZero();

            forget_combinations_weights_derivatives.setZero();
            input_combinations_weights_derivatives.setZero();
            output_combinations_weights_derivatives.setZero();
            state_combinations_weights_derivatives.setZero();

            cell_state_weights_derivatives.setZero();
        }
        else
        {
            memcpy(previous_cell_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-8*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            forget_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(forget_recurrent_weights, A_B);
            input_combinations_weights_derivatives = multiply_rows(hidden_states_weights_derivatives.contract(input_recurrent_weights, A_B), current_input_derivatives);
            state_combinations_weights_derivatives = multiply_rows(hidden_states_weights_derivatives.contract(state_recurrent_weights, A_B), current_state_derivatives);
            output_combinations_weights_derivatives = multiply_rows(hidden_states_weights_derivatives.contract(output_recurrent_weights, A_B), current_output_derivatives);
        }

        column_index = 0;
        input_index = 0;

        for(Index i = 0; i < parameters_number; i++)
        {
            forget_combinations_weights_derivatives(i, column_index) += current_inputs(input_index);

            input_index++;

            if(input_index == inputs_number)
            {
                input_index = 0;
                column_index++;
            }
        }

        cell_state_weights_derivatives = multiply_rows(cell_state_weights_derivatives, current_forget_activations);
        cell_state_weights_derivatives += multiply_rows(input_combinations_weights_derivatives, current_state_activations);
        cell_state_weights_derivatives += multiply_rows(state_combinations_weights_derivatives, current_input_activations);
        cell_state_weights_derivatives += multiply_rows(forget_combinations_weights_derivatives, (current_forget_derivatives*previous_cell_state_activations));

        hidden_states_weights_derivatives = multiply_rows(output_combinations_weights_derivatives, calculate_activations(current_cell_state_activations));
        hidden_states_weights_derivatives += multiply_rows(cell_state_weights_derivatives, current_output_activations*current_hidden_derivatives);

        back_propagation->forget_weights_derivatives += hidden_states_weights_derivatives.contract(current_layer_deltas, A_B);
    }
}


void LongShortTermMemoryLayer::calculate_input_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                      LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                      LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = inputs_number*neurons_number;

    Tensor<type, 2> input_combinations_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> forget_combinations_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> state_combinations_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> output_combinations_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 2> hidden_states_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> cell_state_weights_derivatives(parameters_number, neurons_number);

    input_combinations_weights_derivatives.setZero();
    forget_combinations_weights_derivatives.setZero();
    state_combinations_weights_derivatives.setZero();
    output_combinations_weights_derivatives.setZero();
    hidden_states_weights_derivatives.setZero();
    cell_state_weights_derivatives.setZero();

    Index column_index = 0;
    Index input_index = 0;

    Index activations_copy_index = 0;
    Index derivatives_copy_index = 0;

    Tensor<type, 1> current_forget_activations(neurons_number);
    Tensor<type, 1> current_forget_derivatives(neurons_number);
    Tensor<type, 1> current_input_activations(neurons_number);
    Tensor<type, 1> current_input_derivatives(neurons_number);
    Tensor<type, 1> current_state_activations(neurons_number);
    Tensor<type, 1> current_state_derivatives(neurons_number);
    Tensor<type, 1> current_output_activations(neurons_number);
    Tensor<type, 1> current_output_derivatives(neurons_number);
    Tensor<type, 1> current_cell_state_activations(neurons_number);
    Tensor<type, 1> current_hidden_derivatives(neurons_number);
    Tensor<type, 1> previous_cell_state_activations(neurons_number);

    back_propagation->input_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_inputs = inputs.chip(sample, 0); // memcpy?
        const Tensor<type, 1> current_layer_deltas = back_propagation->delta.chip(sample,0); // memcpy?

        memcpy(current_forget_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_forget_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_input_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_input_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_state_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_output_activations.data(),forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_output_derivatives.data(),forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_cell_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += 2*neurons_number; // // 2* because of hidden state activations

        memcpy(current_hidden_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        derivatives_copy_index += neurons_number;

        if(sample%timesteps == 0)
        {
            previous_cell_state_activations.setZero();

            forget_combinations_weights_derivatives.setZero();
            input_combinations_weights_derivatives.setZero();
            output_combinations_weights_derivatives.setZero();
            state_combinations_weights_derivatives.setZero();

            cell_state_weights_derivatives.setZero();
        }
        else
        {
            memcpy(previous_cell_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-8*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            forget_combinations_weights_derivatives = multiply_rows(hidden_states_weights_derivatives.contract(forget_recurrent_weights, A_B), current_forget_derivatives);
            input_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(input_recurrent_weights, A_B);
            state_combinations_weights_derivatives = multiply_rows(hidden_states_weights_derivatives.contract(state_recurrent_weights, A_B),current_state_derivatives);
            output_combinations_weights_derivatives = multiply_rows(hidden_states_weights_derivatives.contract(output_recurrent_weights, A_B), current_output_derivatives);
        }

        column_index = 0;
        input_index = 0;

        for(Index i = 0; i < parameters_number; i++)
        {
            input_combinations_weights_derivatives(i, column_index) += current_inputs[input_index];

            input_index++;

            if(input_index == inputs_number)
            {
                input_index = 0;
                column_index++;
            }
        }

        cell_state_weights_derivatives = multiply_rows(cell_state_weights_derivatives, current_forget_activations);//cell_state_weights_derivatives.multiply_rows(current_forget_activations);
        cell_state_weights_derivatives += multiply_rows(forget_combinations_weights_derivatives, previous_cell_state_activations);//input_combinations_weights_derivatives.multiply_rows(current_state_activations);
        cell_state_weights_derivatives += multiply_rows(state_combinations_weights_derivatives, current_input_activations);//state_combinations_weights_derivatives.multiply_rows(current_input_activations);
        cell_state_weights_derivatives += multiply_rows(input_combinations_weights_derivatives, (current_input_derivatives*current_state_activations));

        hidden_states_weights_derivatives = multiply_rows(output_combinations_weights_derivatives, calculate_activations(current_cell_state_activations));
        hidden_states_weights_derivatives += multiply_rows(cell_state_weights_derivatives, current_output_activations*current_hidden_derivatives);

        back_propagation->input_weights_derivatives += hidden_states_weights_derivatives.contract(current_layer_deltas, A_B);
    }
}


void LongShortTermMemoryLayer::calculate_state_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                      LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                      LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = inputs_number*neurons_number;

    Tensor<type, 2> input_combinations_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> forget_combinations_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> state_combinations_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> output_combinations_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 2> hidden_states_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> cell_state_weights_derivatives(parameters_number, neurons_number);

    input_combinations_weights_derivatives.setZero();
    forget_combinations_weights_derivatives.setZero();
    state_combinations_weights_derivatives.setZero();
    output_combinations_weights_derivatives.setZero();
    hidden_states_weights_derivatives.setZero();
    cell_state_weights_derivatives.setZero();

    Index column_index = 0;
    Index input_index = 0;

    Index activations_copy_index = 0;
    Index derivatives_copy_index = 0;

    Tensor<type, 1> current_forget_activations(neurons_number);
    Tensor<type, 1> current_forget_derivatives(neurons_number);
    Tensor<type, 1> current_input_activations(neurons_number);
    Tensor<type, 1> current_input_derivatives(neurons_number);
    Tensor<type, 1> current_state_activations(neurons_number);
    Tensor<type, 1> current_state_derivatives(neurons_number);
    Tensor<type, 1> current_output_activations(neurons_number);
    Tensor<type, 1> current_output_derivatives(neurons_number);
    Tensor<type, 1> current_cell_state_activations(neurons_number);
    Tensor<type, 1> current_hidden_derivatives(neurons_number);
    Tensor<type, 1> previous_cell_state_activations(neurons_number);

    back_propagation->state_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_inputs = inputs.chip(sample, 0); // memcpy?
        const Tensor<type, 1> current_layer_deltas = back_propagation->delta.chip(sample,0); // memcpy?

        memcpy(current_forget_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_forget_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_input_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_input_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_state_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_output_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_output_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_cell_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += 2*neurons_number; // // 2* because of hidden state activations

        memcpy(current_hidden_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        derivatives_copy_index += neurons_number;

        if(sample%timesteps == 0)
        {
            previous_cell_state_activations.setZero();

            forget_combinations_weights_derivatives.setZero();
            input_combinations_weights_derivatives.setZero();
            output_combinations_weights_derivatives.setZero();
            state_combinations_weights_derivatives.setZero();

            cell_state_weights_derivatives.setZero();
        }
        else
        {
            memcpy(previous_cell_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-8*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            forget_combinations_weights_derivatives = multiply_rows(hidden_states_weights_derivatives.contract(forget_recurrent_weights, A_B), current_forget_derivatives);
            input_combinations_weights_derivatives = multiply_rows(hidden_states_weights_derivatives.contract(input_recurrent_weights, A_B), current_input_derivatives);
            state_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(state_recurrent_weights, A_B);
            output_combinations_weights_derivatives = multiply_rows(hidden_states_weights_derivatives.contract(output_recurrent_weights, A_B), current_output_derivatives);
        }

        column_index = 0;
        input_index = 0;

        for(Index i = 0; i < parameters_number; i++)
        {
            state_combinations_weights_derivatives(i, column_index) += current_inputs[input_index];

            input_index++;

            if(input_index == inputs_number)
            {
                input_index = 0;
                column_index++;
            }
        }

        cell_state_weights_derivatives = multiply_rows(cell_state_weights_derivatives, current_forget_activations);//cell_state_weights_derivatives.multiply_rows(current_forget_activations);
        cell_state_weights_derivatives += multiply_rows(forget_combinations_weights_derivatives, previous_cell_state_activations);//input_combinations_weights_derivatives.multiply_rows(current_state_activations);
        cell_state_weights_derivatives += multiply_rows(input_combinations_weights_derivatives, current_state_activations);//state_combinations_weights_derivatives.multiply_rows(current_input_activations);
        cell_state_weights_derivatives += multiply_rows(state_combinations_weights_derivatives, (current_state_derivatives*current_input_activations));

        hidden_states_weights_derivatives = multiply_rows(output_combinations_weights_derivatives, calculate_activations(current_cell_state_activations));
        hidden_states_weights_derivatives += multiply_rows(cell_state_weights_derivatives, current_output_activations*current_hidden_derivatives);

        back_propagation->state_weights_derivatives += hidden_states_weights_derivatives.contract(current_layer_deltas, A_B);
    }
}


void LongShortTermMemoryLayer::calculate_output_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                       LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                       LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = inputs_number*neurons_number;

    Tensor<type, 2> input_combinations_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> forget_combinations_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> state_combinations_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> output_combinations_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 2> hidden_states_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> cell_state_weights_derivatives(parameters_number, neurons_number);

    input_combinations_weights_derivatives.setZero();
    forget_combinations_weights_derivatives.setZero();
    state_combinations_weights_derivatives.setZero();
    output_combinations_weights_derivatives.setZero();
    hidden_states_weights_derivatives.setZero();
    cell_state_weights_derivatives.setZero();

    Index column_index = 0;
    Index input_index = 0;

    Index activations_copy_index = 0;
    Index derivatives_copy_index = 0;

    Tensor<type, 1> current_forget_activations(neurons_number);
    Tensor<type, 1> current_forget_derivatives(neurons_number);
    Tensor<type, 1> current_input_activations(neurons_number);
    Tensor<type, 1> current_input_derivatives(neurons_number);
    Tensor<type, 1> current_state_activations(neurons_number);
    Tensor<type, 1> current_state_derivatives(neurons_number);
    Tensor<type, 1> current_output_activations(neurons_number);
    Tensor<type, 1> current_output_derivatives(neurons_number);
    Tensor<type, 1> current_cell_state_activations(neurons_number);
    Tensor<type, 1> current_hidden_derivatives(neurons_number);
    Tensor<type, 1> previous_cell_state_activations(neurons_number);

    back_propagation->output_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_inputs = inputs.chip(sample, 0); // memcpy?
        const Tensor<type, 1> current_layer_deltas = back_propagation->delta.chip(sample,0); // memcpy?

        memcpy(current_forget_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_forget_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_input_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_input_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_state_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_output_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_output_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_cell_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += 2*neurons_number; // // 2* because of hidden state activations

        memcpy(current_hidden_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        derivatives_copy_index += neurons_number;

        if(sample%timesteps == 0)
        {
            previous_cell_state_activations.setZero();

            forget_combinations_weights_derivatives.setZero();
            input_combinations_weights_derivatives.setZero();
            output_combinations_weights_derivatives.setZero();
            state_combinations_weights_derivatives.setZero();

            cell_state_weights_derivatives.setZero();
        }
        else
        {
            memcpy(previous_cell_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-8*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            forget_combinations_weights_derivatives = multiply_rows(hidden_states_weights_derivatives.contract(forget_recurrent_weights, A_B), current_forget_derivatives);
            input_combinations_weights_derivatives = multiply_rows(hidden_states_weights_derivatives.contract(input_recurrent_weights, A_B), current_input_derivatives);
            state_combinations_weights_derivatives = multiply_rows(hidden_states_weights_derivatives.contract(state_recurrent_weights, A_B), current_state_derivatives);
            output_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(output_recurrent_weights, A_B);
        }

        column_index = 0;
        input_index = 0;

        for(Index i = 0; i < parameters_number; i++)
        {
            output_combinations_weights_derivatives(i, column_index) += current_inputs[input_index];

            input_index++;

            if(input_index == inputs_number)
            {
                input_index = 0;
                column_index++;
            }
        }

        cell_state_weights_derivatives = multiply_rows(cell_state_weights_derivatives, current_forget_activations);
        cell_state_weights_derivatives += multiply_rows(forget_combinations_weights_derivatives, previous_cell_state_activations);
        cell_state_weights_derivatives += multiply_rows(state_combinations_weights_derivatives, current_input_activations);
        cell_state_weights_derivatives += multiply_rows(input_combinations_weights_derivatives, current_state_activations);

        hidden_states_weights_derivatives = multiply_rows(output_combinations_weights_derivatives, current_output_derivatives*calculate_activations(current_cell_state_activations));
        hidden_states_weights_derivatives += multiply_rows(cell_state_weights_derivatives, current_output_activations*current_hidden_derivatives);

        back_propagation->output_weights_derivatives += hidden_states_weights_derivatives.contract(current_layer_deltas, A_B);
    }
}


void LongShortTermMemoryLayer::calculate_forget_recurrent_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                 LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                                 LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number*neurons_number;

    Tensor<type, 1> forget_recurrent_weights_error_gradient(parameters_number);
    forget_recurrent_weights_error_gradient.setZero();

    Tensor<type, 2> input_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> forget_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> state_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> output_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 2> hidden_states_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> cell_state_recurrent_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 1> current_forget_activations(neurons_number);
    Tensor<type, 1> current_input_activations(neurons_number);
    Tensor<type, 1> current_state_activations(neurons_number);
    Tensor<type, 1> current_output_activations(neurons_number);
    Tensor<type, 1> current_cell_state_activations(neurons_number);

    Tensor<type, 1> current_forget_derivatives(neurons_number);
    Tensor<type, 1> current_input_derivatives(neurons_number);
    Tensor<type, 1> current_state_derivatives(neurons_number);
    Tensor<type, 1> current_output_derivatives(neurons_number);
    Tensor<type, 1> current_hidden_derivatives(neurons_number);

    Tensor<type, 1> previous_hidden_state_activations(neurons_number);
    Tensor<type, 1> previous_cell_state_activations(neurons_number);

    Index column_index = 0;
    Index activation_index = 0;

    Index activations_copy_index = 0;
    Index derivatives_copy_index = 0;

    back_propagation->forget_recurrent_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_layer_deltas = back_propagation->delta.chip(sample, 0);

        memcpy(current_forget_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_forget_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_input_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_input_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_state_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_output_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_output_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_cell_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += 2*neurons_number; // 2* because of hidden state activations

        memcpy(current_hidden_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        derivatives_copy_index += neurons_number;

        if(sample%timesteps == 0)
        {
            cell_state_recurrent_weights_derivatives.setZero();
            hidden_states_recurrent_weights_derivatives.setZero();
        }
        else
        {
            memcpy(previous_hidden_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-7*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            memcpy(previous_cell_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-8*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            forget_combinations_recurrent_weights_derivatives = hidden_states_recurrent_weights_derivatives.contract(forget_recurrent_weights, A_B);
            input_combinations_recurrent_weights_derivatives = multiply_rows(hidden_states_recurrent_weights_derivatives.contract(input_recurrent_weights, A_B), current_input_derivatives);
            state_combinations_recurrent_weights_derivatives = multiply_rows(hidden_states_recurrent_weights_derivatives.contract(state_recurrent_weights, A_B), current_state_derivatives);
            output_combinations_recurrent_weights_derivatives = multiply_rows(hidden_states_recurrent_weights_derivatives.contract(output_recurrent_weights, A_B), current_output_derivatives);

            column_index = 0;
            activation_index = 0;

            for(Index i = 0; i < parameters_number; i++)
            {
                forget_combinations_recurrent_weights_derivatives(i, column_index) += previous_hidden_state_activations[activation_index];

                activation_index++;

                if(activation_index == neurons_number)
                {
                    activation_index = 0;
                    column_index++;
                }
            }

            cell_state_recurrent_weights_derivatives = multiply_rows(cell_state_recurrent_weights_derivatives, current_forget_activations);
            cell_state_recurrent_weights_derivatives += multiply_rows(input_combinations_recurrent_weights_derivatives, current_state_activations);
            cell_state_recurrent_weights_derivatives += multiply_rows(state_combinations_recurrent_weights_derivatives, current_input_activations);
            cell_state_recurrent_weights_derivatives += multiply_rows(forget_combinations_recurrent_weights_derivatives, (current_forget_derivatives*previous_cell_state_activations));

            hidden_states_recurrent_weights_derivatives = multiply_rows(output_combinations_recurrent_weights_derivatives, calculate_activations(current_cell_state_activations));
            hidden_states_recurrent_weights_derivatives += multiply_rows(cell_state_recurrent_weights_derivatives, current_output_activations*current_hidden_derivatives);
        }

        back_propagation->forget_recurrent_weights_derivatives += hidden_states_recurrent_weights_derivatives.contract(current_layer_deltas, A_B);
    }
}


void LongShortTermMemoryLayer::calculate_input_recurrent_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                                LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number*neurons_number;

    Tensor<type, 1> forget_recurrent_weights_error_gradient(parameters_number);
    forget_recurrent_weights_error_gradient.setZero();

    Tensor<type, 2> input_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> forget_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> state_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> output_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 2> hidden_states_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> cell_state_recurrent_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 1> current_forget_activations(neurons_number);
    Tensor<type, 1> current_input_activations(neurons_number);
    Tensor<type, 1> current_state_activations(neurons_number);
    Tensor<type, 1> current_output_activations(neurons_number);
    Tensor<type, 1> current_cell_state_activations(neurons_number);

    Tensor<type, 1> current_forget_derivatives(neurons_number);
    Tensor<type, 1> current_input_derivatives(neurons_number);
    Tensor<type, 1> current_state_derivatives(neurons_number);
    Tensor<type, 1> current_output_derivatives(neurons_number);
    Tensor<type, 1> current_hidden_derivatives(neurons_number);

    Tensor<type, 1> previous_hidden_state_activations(neurons_number);
    Tensor<type, 1> previous_cell_state_activations(neurons_number);

    Index column_index = 0;
    Index activation_index = 0;

    Index activations_copy_index = 0;
    Index derivatives_copy_index = 0;

    back_propagation->input_recurrent_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_layer_deltas = back_propagation->delta.chip(sample, 0);

        memcpy(current_forget_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_forget_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_input_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_input_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_state_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_output_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_output_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_cell_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += 2*neurons_number; // 2* because of hidden state activations

        memcpy(current_hidden_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        derivatives_copy_index += neurons_number;

        if(sample%timesteps == 0)
        {
            cell_state_recurrent_weights_derivatives.setZero();
            hidden_states_recurrent_weights_derivatives.setZero();
        }
        else
        {
            memcpy(previous_hidden_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-7*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            memcpy(previous_cell_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-8*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            forget_combinations_recurrent_weights_derivatives = multiply_rows(hidden_states_recurrent_weights_derivatives.contract(forget_recurrent_weights, A_B), current_forget_derivatives);
            input_combinations_recurrent_weights_derivatives = hidden_states_recurrent_weights_derivatives.contract(input_recurrent_weights, A_B);
            state_combinations_recurrent_weights_derivatives = multiply_rows(hidden_states_recurrent_weights_derivatives.contract(state_recurrent_weights, A_B), current_state_derivatives);
            output_combinations_recurrent_weights_derivatives = multiply_rows(hidden_states_recurrent_weights_derivatives.contract(output_recurrent_weights, A_B), current_output_derivatives);

            column_index = 0;
            activation_index = 0;

            for(Index i = 0; i < parameters_number; i++)
            {
                input_combinations_recurrent_weights_derivatives(i, column_index) += previous_hidden_state_activations[activation_index];

                activation_index++;

                if(activation_index == neurons_number)
                {
                    activation_index = 0;
                    column_index++;
                }
            }

            cell_state_recurrent_weights_derivatives = multiply_rows(cell_state_recurrent_weights_derivatives, current_forget_activations);//cell_state_weights_derivatives.multiply_rows(current_forget_activations);
            cell_state_recurrent_weights_derivatives += multiply_rows(input_combinations_recurrent_weights_derivatives, current_input_derivatives*current_state_activations);//input_combinations_weights_derivatives.multiply_rows(current_state_activations);
            cell_state_recurrent_weights_derivatives += multiply_rows(state_combinations_recurrent_weights_derivatives, current_input_activations);//state_combinations_weights_derivatives.multiply_rows(current_input_activations);
            cell_state_recurrent_weights_derivatives += multiply_rows(forget_combinations_recurrent_weights_derivatives, previous_cell_state_activations);

            hidden_states_recurrent_weights_derivatives = multiply_rows(output_combinations_recurrent_weights_derivatives, calculate_activations(current_cell_state_activations));
            hidden_states_recurrent_weights_derivatives += multiply_rows(cell_state_recurrent_weights_derivatives, current_output_activations*current_hidden_derivatives);

        }

        back_propagation->input_recurrent_weights_derivatives += hidden_states_recurrent_weights_derivatives.contract(current_layer_deltas, A_B);
    }
}


void LongShortTermMemoryLayer::calculate_state_recurrent_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                                LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number*neurons_number;

    Tensor<type, 1> forget_recurrent_weights_error_gradient(parameters_number);
    forget_recurrent_weights_error_gradient.setZero();

    Tensor<type, 2> input_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> forget_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> state_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> output_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 2> hidden_states_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> cell_state_recurrent_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 1> current_forget_activations(neurons_number);
    Tensor<type, 1> current_input_activations(neurons_number);
    Tensor<type, 1> current_state_activations(neurons_number);
    Tensor<type, 1> current_output_activations(neurons_number);
    Tensor<type, 1> current_cell_state_activations(neurons_number);

    Tensor<type, 1> current_forget_derivatives(neurons_number);
    Tensor<type, 1> current_input_derivatives(neurons_number);
    Tensor<type, 1> current_state_derivatives(neurons_number);
    Tensor<type, 1> current_output_derivatives(neurons_number);
    Tensor<type, 1> current_hidden_derivatives(neurons_number);

    Tensor<type, 1> previous_hidden_state_activations(neurons_number);
    Tensor<type, 1> previous_cell_state_activations(neurons_number);

    Index column_index = 0;
    Index activation_index = 0;

    Index activations_copy_index = 0;
    Index derivatives_copy_index = 0;

    back_propagation->state_recurrent_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_layer_deltas = back_propagation->delta.chip(sample, 0);

        memcpy(current_forget_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_forget_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_input_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_input_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_state_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_output_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_output_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_cell_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += 2*neurons_number; // 2* because of hidden state activations

        memcpy(current_hidden_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        derivatives_copy_index += neurons_number;

        if(sample%timesteps == 0)
        {
            cell_state_recurrent_weights_derivatives.setZero();
            hidden_states_recurrent_weights_derivatives.setZero();
        }
        else
        {
            memcpy(previous_hidden_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-7*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            memcpy(previous_cell_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-8*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            forget_combinations_recurrent_weights_derivatives = multiply_rows(hidden_states_recurrent_weights_derivatives.contract(forget_recurrent_weights, A_B), current_forget_derivatives);
            input_combinations_recurrent_weights_derivatives = multiply_rows(hidden_states_recurrent_weights_derivatives.contract(input_recurrent_weights, A_B), current_input_derivatives);
            state_combinations_recurrent_weights_derivatives = hidden_states_recurrent_weights_derivatives.contract(state_recurrent_weights, A_B);
            output_combinations_recurrent_weights_derivatives = multiply_rows(hidden_states_recurrent_weights_derivatives.contract(output_recurrent_weights, A_B), current_output_derivatives);

            column_index = 0;
            activation_index = 0;

            for(Index i = 0; i < parameters_number; i++)
            {
                state_combinations_recurrent_weights_derivatives(i, column_index) += previous_hidden_state_activations[activation_index];

                activation_index++;

                if(activation_index == neurons_number)
                {
                    activation_index = 0;
                    column_index++;
                }
            }

            cell_state_recurrent_weights_derivatives = multiply_rows(cell_state_recurrent_weights_derivatives, current_forget_activations);//cell_state_weights_derivatives.multiply_rows(current_forget_activations);
            cell_state_recurrent_weights_derivatives += multiply_rows(input_combinations_recurrent_weights_derivatives, current_state_activations);//input_combinations_weights_derivatives.multiply_rows(current_state_activations);
            cell_state_recurrent_weights_derivatives += multiply_rows(state_combinations_recurrent_weights_derivatives, current_state_derivatives*current_input_activations);//state_combinations_weights_derivatives.multiply_rows(current_input_activations);
            cell_state_recurrent_weights_derivatives += multiply_rows(forget_combinations_recurrent_weights_derivatives, previous_cell_state_activations);

            hidden_states_recurrent_weights_derivatives = multiply_rows(output_combinations_recurrent_weights_derivatives, calculate_activations(current_cell_state_activations));
            hidden_states_recurrent_weights_derivatives += multiply_rows(cell_state_recurrent_weights_derivatives, current_output_activations*current_hidden_derivatives);

        }

        back_propagation->state_recurrent_weights_derivatives += hidden_states_recurrent_weights_derivatives.contract(current_layer_deltas, A_B);
    }
}


void LongShortTermMemoryLayer::calculate_output_recurrent_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                 LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                                 LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number*neurons_number;

    Tensor<type, 1> forget_recurrent_weights_error_gradient(parameters_number);
    forget_recurrent_weights_error_gradient.setZero();

    Tensor<type, 2> input_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> forget_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> state_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> output_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 2> hidden_states_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> cell_state_recurrent_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 1> current_forget_activations(neurons_number);
    Tensor<type, 1> current_input_activations(neurons_number);
    Tensor<type, 1> current_state_activations(neurons_number);
    Tensor<type, 1> current_output_activations(neurons_number);
    Tensor<type, 1> current_cell_state_activations(neurons_number);

    Tensor<type, 1> current_forget_derivatives(neurons_number);
    Tensor<type, 1> current_input_derivatives(neurons_number);
    Tensor<type, 1> current_state_derivatives(neurons_number);
    Tensor<type, 1> current_output_derivatives(neurons_number);
    Tensor<type, 1> current_hidden_derivatives(neurons_number);

    Tensor<type, 1> previous_hidden_state_activations(neurons_number);
    Tensor<type, 1> previous_cell_state_activations(neurons_number);

    Index column_index = 0;
    Index activation_index = 0;

    Index activations_copy_index = 0;
    Index derivatives_copy_index = 0;

    back_propagation->output_recurrent_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_layer_deltas = back_propagation->delta.chip(sample, 0);

        memcpy(current_forget_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_forget_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_input_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_input_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_state_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_output_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_output_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_cell_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += 2*neurons_number; // 2* because of hidden state activations

        memcpy(current_hidden_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        derivatives_copy_index += neurons_number;

        if(sample%timesteps == 0)
        {
            cell_state_recurrent_weights_derivatives.setZero();
            hidden_states_recurrent_weights_derivatives.setZero();
        }
        else
        {
            memcpy(previous_hidden_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-7*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            memcpy(previous_cell_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-8*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            forget_combinations_recurrent_weights_derivatives = multiply_rows(hidden_states_recurrent_weights_derivatives.contract(forget_recurrent_weights, A_B), current_forget_derivatives);
            input_combinations_recurrent_weights_derivatives = multiply_rows(hidden_states_recurrent_weights_derivatives.contract(input_recurrent_weights, A_B), current_input_derivatives);
            state_combinations_recurrent_weights_derivatives = multiply_rows(hidden_states_recurrent_weights_derivatives.contract(state_recurrent_weights, A_B), current_state_derivatives);
            output_combinations_recurrent_weights_derivatives = hidden_states_recurrent_weights_derivatives.contract(output_recurrent_weights, A_B);

            column_index = 0;
            activation_index = 0;

            for(Index i = 0; i < parameters_number; i++)
            {
                output_combinations_recurrent_weights_derivatives(i, column_index) += previous_hidden_state_activations[activation_index];

                activation_index++;

                if(activation_index == neurons_number)
                {
                    activation_index = 0;
                    column_index++;
                }
            }

            cell_state_recurrent_weights_derivatives = multiply_rows(cell_state_recurrent_weights_derivatives, current_forget_activations);//cell_state_weights_derivatives.multiply_rows(current_forget_activations);
            cell_state_recurrent_weights_derivatives += multiply_rows(input_combinations_recurrent_weights_derivatives, current_state_activations);//input_combinations_weights_derivatives.multiply_rows(current_state_activations);
            cell_state_recurrent_weights_derivatives += multiply_rows(state_combinations_recurrent_weights_derivatives, current_input_activations);//state_combinations_weights_derivatives.multiply_rows(current_input_activations);
            cell_state_recurrent_weights_derivatives += multiply_rows(forget_combinations_recurrent_weights_derivatives, previous_cell_state_activations);

            hidden_states_recurrent_weights_derivatives = multiply_rows(output_combinations_recurrent_weights_derivatives, current_output_derivatives*calculate_activations(current_cell_state_activations));
            hidden_states_recurrent_weights_derivatives += multiply_rows(cell_state_recurrent_weights_derivatives, current_output_activations*current_hidden_derivatives);

        }

        back_propagation->output_recurrent_weights_derivatives += hidden_states_recurrent_weights_derivatives.contract(current_layer_deltas, A_B);
    }
}


void LongShortTermMemoryLayer::calculate_forget_biases_error_gradient(const Tensor<type, 2>& inputs,
                                                                      LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                      LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number;

    Tensor<type, 2> input_combinations_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> forget_combinations_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> state_combinations_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> output_combinations_biases_derivatives(parameters_number, neurons_number);

    Tensor<type, 2> hidden_states_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> cell_state_biases_derivatives(parameters_number, neurons_number);

    input_combinations_biases_derivatives.setZero();
    forget_combinations_biases_derivatives.setZero();
    state_combinations_biases_derivatives.setZero();
    output_combinations_biases_derivatives.setZero();

    hidden_states_biases_derivatives.setZero();
    cell_state_biases_derivatives.setZero();

    Tensor<type, 1> current_forget_activations(neurons_number);
    Tensor<type, 1> current_input_activations(neurons_number);
    Tensor<type, 1> current_state_activations(neurons_number);
    Tensor<type, 1> current_output_activations(neurons_number);
    Tensor<type, 1> current_cell_state_activations(neurons_number);

    Tensor<type, 1> current_forget_derivatives(neurons_number);
    Tensor<type, 1> current_input_derivatives(neurons_number);
    Tensor<type, 1> current_state_derivatives(neurons_number);
    Tensor<type, 1> current_output_derivatives(neurons_number);
    Tensor<type, 1> current_hidden_derivatives(neurons_number);

    Tensor<type, 1> previous_cell_state_activations(neurons_number);

    Index activations_copy_index = 0;
    Index derivatives_copy_index = 0;

    back_propagation->forget_biases_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_layer_deltas = back_propagation->delta.chip(sample, 0);

        memcpy(current_forget_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_forget_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_input_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_input_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_state_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_output_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_output_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_cell_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += 2*neurons_number; // 2* because of hidden state activations

        memcpy(current_hidden_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        derivatives_copy_index += neurons_number;

        previous_cell_state_activations.setZero();

        if(sample%timesteps == 0)
        {
            forget_combinations_biases_derivatives.setZero();
            input_combinations_biases_derivatives.setZero();
            state_combinations_biases_derivatives.setZero();
            output_combinations_biases_derivatives.setZero();

            cell_state_biases_derivatives.setZero();
        }
        else
        {
            memcpy(previous_cell_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-8*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            forget_combinations_biases_derivatives = hidden_states_biases_derivatives.contract(forget_recurrent_weights, A_B);
            input_combinations_biases_derivatives = multiply_rows(hidden_states_biases_derivatives.contract(input_recurrent_weights, A_B), current_input_derivatives);
            state_combinations_biases_derivatives = multiply_rows(hidden_states_biases_derivatives.contract(state_recurrent_weights, A_B), current_state_derivatives);
            output_combinations_biases_derivatives = multiply_rows(hidden_states_biases_derivatives.contract(output_recurrent_weights, A_B), current_output_derivatives);
        }

        for(Index row = 0; row < parameters_number; row++) forget_combinations_biases_derivatives(row, row) += static_cast<type>(1.0);

        cell_state_biases_derivatives = multiply_rows(cell_state_biases_derivatives, current_forget_activations);
        cell_state_biases_derivatives += multiply_rows(input_combinations_biases_derivatives, current_state_activations);
        cell_state_biases_derivatives += multiply_rows(state_combinations_biases_derivatives, current_input_activations);
        cell_state_biases_derivatives += multiply_rows(forget_combinations_biases_derivatives, current_forget_derivatives*previous_cell_state_activations);

        hidden_states_biases_derivatives = multiply_rows(output_combinations_biases_derivatives, calculate_activations(current_cell_state_activations));
        hidden_states_biases_derivatives += multiply_rows(cell_state_biases_derivatives, current_output_activations*current_hidden_derivatives);

        back_propagation->forget_biases_derivatives += hidden_states_biases_derivatives.contract(current_layer_deltas, A_B);
    }
}


void LongShortTermMemoryLayer::calculate_input_biases_error_gradient(const Tensor<type, 2>& inputs,
                                                                     LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                     LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number;

    Tensor<type, 2> input_combinations_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> forget_combinations_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> state_combinations_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> output_combinations_biases_derivatives(parameters_number, neurons_number);

    Tensor<type, 2> hidden_states_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> cell_state_biases_derivatives(parameters_number, neurons_number);

    input_combinations_biases_derivatives.setZero();
    forget_combinations_biases_derivatives.setZero();
    state_combinations_biases_derivatives.setZero();
    output_combinations_biases_derivatives.setZero();

    hidden_states_biases_derivatives.setZero();
    cell_state_biases_derivatives.setZero();

    Tensor<type, 1> current_forget_activations(neurons_number);
    Tensor<type, 1> current_input_activations(neurons_number);
    Tensor<type, 1> current_state_activations(neurons_number);
    Tensor<type, 1> current_output_activations(neurons_number);
    Tensor<type, 1> current_cell_state_activations(neurons_number);

    Tensor<type, 1> current_forget_derivatives(neurons_number);
    Tensor<type, 1> current_input_derivatives(neurons_number);
    Tensor<type, 1> current_state_derivatives(neurons_number);
    Tensor<type, 1> current_output_derivatives(neurons_number);
    Tensor<type, 1> current_hidden_derivatives(neurons_number);

    Tensor<type, 1> previous_cell_state_activations(neurons_number);

    Index activations_copy_index = 0;
    Index derivatives_copy_index = 0;

    back_propagation->input_biases_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_layer_deltas = back_propagation->delta.chip(sample, 0);

        memcpy(current_forget_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_forget_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_input_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_input_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_state_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_output_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_output_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_cell_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += 2*neurons_number; // 2* because of hidden state activations

        memcpy(current_hidden_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        derivatives_copy_index += neurons_number;

        if(sample%timesteps == 0)
        {
            forget_combinations_biases_derivatives.setZero();
            input_combinations_biases_derivatives.setZero();
            state_combinations_biases_derivatives.setZero();
            output_combinations_biases_derivatives.setZero();

            previous_cell_state_activations.setZero();
            cell_state_biases_derivatives.setZero();
        }
        else
        {
            memcpy(previous_cell_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-8*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            forget_combinations_biases_derivatives = multiply_rows(hidden_states_biases_derivatives.contract(forget_recurrent_weights, A_B), current_forget_derivatives);
            input_combinations_biases_derivatives = hidden_states_biases_derivatives.contract(input_recurrent_weights, A_B);
            state_combinations_biases_derivatives = multiply_rows(hidden_states_biases_derivatives.contract(state_recurrent_weights, A_B), current_state_derivatives);
            output_combinations_biases_derivatives = multiply_rows(hidden_states_biases_derivatives.contract(output_recurrent_weights, A_B), current_output_derivatives);
        }

        for(Index row = 0; row < parameters_number; row++) input_combinations_biases_derivatives(row, row) += static_cast<type>(1.0);

        cell_state_biases_derivatives = multiply_rows(cell_state_biases_derivatives, current_forget_activations);
        cell_state_biases_derivatives += multiply_rows(forget_combinations_biases_derivatives, previous_cell_state_activations);
        cell_state_biases_derivatives += multiply_rows(state_combinations_biases_derivatives, current_input_activations);
        cell_state_biases_derivatives += multiply_rows(input_combinations_biases_derivatives, current_input_derivatives*current_state_activations);

        hidden_states_biases_derivatives = multiply_rows(output_combinations_biases_derivatives, calculate_activations(current_cell_state_activations));
        hidden_states_biases_derivatives += multiply_rows(cell_state_biases_derivatives, current_output_activations*current_hidden_derivatives);

        back_propagation->input_biases_derivatives += hidden_states_biases_derivatives.contract(current_layer_deltas, A_B);
    }
}



void LongShortTermMemoryLayer::calculate_state_biases_error_gradient(const Tensor<type, 2>& inputs,
                                                                     LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                     LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number;

    Tensor<type, 2> input_combinations_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> forget_combinations_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> state_combinations_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> output_combinations_biases_derivatives(parameters_number, neurons_number);

    Tensor<type, 2> hidden_states_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> cell_state_biases_derivatives(parameters_number, neurons_number);

    input_combinations_biases_derivatives.setZero();
    forget_combinations_biases_derivatives.setZero();
    state_combinations_biases_derivatives.setZero();
    output_combinations_biases_derivatives.setZero();

    hidden_states_biases_derivatives.setZero();
    cell_state_biases_derivatives.setZero();

    Tensor<type, 1> current_forget_activations(neurons_number);
    Tensor<type, 1> current_input_activations(neurons_number);
    Tensor<type, 1> current_state_activations(neurons_number);
    Tensor<type, 1> current_output_activations(neurons_number);
    Tensor<type, 1> current_cell_state_activations(neurons_number);

    Tensor<type, 1> current_forget_derivatives(neurons_number);
    Tensor<type, 1> current_input_derivatives(neurons_number);
    Tensor<type, 1> current_state_derivatives(neurons_number);
    Tensor<type, 1> current_output_derivatives(neurons_number);
    Tensor<type, 1> current_hidden_derivatives(neurons_number);

    Tensor<type, 1> previous_cell_state_activations(neurons_number);

    Index activations_copy_index = 0;
    Index derivatives_copy_index = 0;

    back_propagation->state_biases_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_layer_deltas = back_propagation->delta.chip(sample, 0);

        memcpy(current_forget_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_forget_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_input_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_input_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_state_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_output_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_output_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_cell_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += 2*neurons_number; // 2* because of hidden state activations

        memcpy(current_hidden_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        derivatives_copy_index += neurons_number;

        previous_cell_state_activations.setZero();

        if(sample%timesteps == 0)
        {
            forget_combinations_biases_derivatives.setZero();
            input_combinations_biases_derivatives.setZero();
            state_combinations_biases_derivatives.setZero();
            output_combinations_biases_derivatives.setZero();

            cell_state_biases_derivatives.setZero();
        }
        else
        {
            memcpy(previous_cell_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-8*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            forget_combinations_biases_derivatives = multiply_rows(hidden_states_biases_derivatives.contract(forget_recurrent_weights, A_B), current_forget_derivatives);
            input_combinations_biases_derivatives = multiply_rows(hidden_states_biases_derivatives.contract(input_recurrent_weights, A_B), current_input_derivatives);
            state_combinations_biases_derivatives = hidden_states_biases_derivatives.contract(state_recurrent_weights, A_B);
            output_combinations_biases_derivatives = multiply_rows(hidden_states_biases_derivatives.contract(output_recurrent_weights, A_B), current_output_derivatives);
        }

        for(Index row = 0; row < parameters_number; row++) state_combinations_biases_derivatives(row, row) += static_cast<type>(1.0);

        cell_state_biases_derivatives = multiply_rows(cell_state_biases_derivatives, current_forget_activations);
        cell_state_biases_derivatives += multiply_rows(forget_combinations_biases_derivatives, previous_cell_state_activations);
        cell_state_biases_derivatives += multiply_rows(input_combinations_biases_derivatives, current_state_activations);
        cell_state_biases_derivatives += multiply_rows(state_combinations_biases_derivatives, current_state_derivatives*current_input_activations);

        hidden_states_biases_derivatives = multiply_rows(output_combinations_biases_derivatives, calculate_activations(current_cell_state_activations));
        hidden_states_biases_derivatives += multiply_rows(cell_state_biases_derivatives, current_output_activations*current_hidden_derivatives);

        back_propagation->state_biases_derivatives += hidden_states_biases_derivatives.contract(current_layer_deltas, A_B);
    }
}


void LongShortTermMemoryLayer::calculate_output_biases_error_gradient(const Tensor<type, 2>& inputs,
                                                                      LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                      LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number;

    Tensor<type, 2> input_combinations_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> forget_combinations_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> state_combinations_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> output_combinations_biases_derivatives(parameters_number, neurons_number);

    Tensor<type, 2> hidden_states_biases_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> cell_state_biases_derivatives(parameters_number, neurons_number);

    input_combinations_biases_derivatives.setZero();
    forget_combinations_biases_derivatives.setZero();
    state_combinations_biases_derivatives.setZero();
    output_combinations_biases_derivatives.setZero();
    hidden_states_biases_derivatives.setZero();
    cell_state_biases_derivatives.setZero();

    Tensor<type, 1> current_forget_activations(neurons_number);
    Tensor<type, 1> current_input_activations(neurons_number);
    Tensor<type, 1> current_state_activations(neurons_number);
    Tensor<type, 1> current_output_activations(neurons_number);
    Tensor<type, 1> current_cell_state_activations(neurons_number);

    Tensor<type, 1> current_forget_derivatives(neurons_number);
    Tensor<type, 1> current_input_derivatives(neurons_number);
    Tensor<type, 1> current_state_derivatives(neurons_number);
    Tensor<type, 1> current_output_derivatives(neurons_number);
    Tensor<type, 1> current_hidden_derivatives(neurons_number);

    Tensor<type, 1> previous_cell_state_activations(neurons_number);

    Index activations_copy_index = 0;
    Index derivatives_copy_index = 0;

    back_propagation->output_biases_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_layer_deltas = back_propagation->delta.chip(sample, 0);

        memcpy(current_forget_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_forget_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_input_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_input_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_state_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_output_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        memcpy(current_output_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += neurons_number;
        derivatives_copy_index += neurons_number;

        memcpy(current_cell_state_activations.data(), forward_propagation->row_major_activations_3d.data() + activations_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        activations_copy_index += 2*neurons_number; //  2* because of hidden state activations

        memcpy(current_hidden_derivatives.data(), forward_propagation->row_major_activations_derivatives_3d.data() + derivatives_copy_index, static_cast<size_t>(neurons_number)*sizeof(type));
        derivatives_copy_index += neurons_number;

        previous_cell_state_activations.setZero();

        if(sample%timesteps == 0)
        {
            forget_combinations_biases_derivatives.setZero();
            input_combinations_biases_derivatives.setZero();
            state_combinations_biases_derivatives.setZero();
            output_combinations_biases_derivatives.setZero();

            cell_state_biases_derivatives.setZero();
        }
        else
        {
            memcpy(previous_cell_state_activations.data(),
                   forward_propagation->row_major_activations_3d.data() + (activations_copy_index-8*neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));
            forget_combinations_biases_derivatives = multiply_rows(hidden_states_biases_derivatives.contract(forget_recurrent_weights, A_B), current_forget_derivatives);
            input_combinations_biases_derivatives = multiply_rows(hidden_states_biases_derivatives.contract(input_recurrent_weights, A_B), current_input_derivatives);
            state_combinations_biases_derivatives = multiply_rows(hidden_states_biases_derivatives.contract(state_recurrent_weights, A_B), current_state_derivatives);
            output_combinations_biases_derivatives = hidden_states_biases_derivatives.contract(output_recurrent_weights, A_B);
        }

        for(Index row = 0; row < parameters_number; row++) output_combinations_biases_derivatives(row, row) += static_cast<type>(1.0);

        cell_state_biases_derivatives = multiply_rows(cell_state_biases_derivatives, current_forget_activations);
        cell_state_biases_derivatives += multiply_rows(forget_combinations_biases_derivatives, previous_cell_state_activations);
        cell_state_biases_derivatives += multiply_rows(state_combinations_biases_derivatives, current_input_activations);
        cell_state_biases_derivatives += multiply_rows(input_combinations_biases_derivatives, current_state_activations);

        hidden_states_biases_derivatives = multiply_rows(output_combinations_biases_derivatives, current_output_derivatives*calculate_activations(current_cell_state_activations));
        hidden_states_biases_derivatives += multiply_rows(cell_state_biases_derivatives, current_output_activations*current_hidden_derivatives);

        back_propagation->output_biases_derivatives += hidden_states_biases_derivatives.contract(current_layer_deltas, A_B);
    }
}

Tensor<type, 2> LongShortTermMemoryLayer::multiply_rows(const Tensor<type, 2>& matrix, const Tensor<type, 1>& vector) const
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

string LongShortTermMemoryLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{

    const Index neurons_number = get_neurons_number();

    const Index inputs_number = get_inputs_number();

#ifdef __OPENNN_DEBUG__

    const Index inputs_name_size = inputs_names.size();

    if(inputs_name_size != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const method.\n"
               << "Size of inputs name must be equal to number of layer inputs.\n";

        throw logic_error(buffer.str());
    }

    const Index outputs_name_size = outputs_names.size();

    if(outputs_name_size != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const method.\n"
               << "Size of outputs name must be equal to number of neurons.\n";

        throw logic_error(buffer.str());
    }

#endif

    ostringstream buffer;

    // Forget gate
    /*
       for(Index j = 0; j < neurons_number; j++)
       {
           buffer << "forget_gate_" << to_string(j+1) << " = " << write_recurrent_activation_function_expression() << " (" << forget_biases[j] << "+";

           for(Index i = 0; i < inputs_number; i++)
           {
               buffer << inputs_names[i] << "*" << forget_weights.get_column(j)(i) << "+";
           }

           for(Index k = 0; k < neurons_number-1; k++)
           {
               buffer << "hidden_state_" << to_string(k+1) << "(t-1)*" << forget_recurrent_weights.get_column(j)[k] << "+";
           }

           buffer << "hidden_state_" << to_string(neurons_number) << "(t-1)*" << forget_recurrent_weights.get_column(j)[neurons_number-1] << ");\n";
       }

       // Input gate

       for(Index j = 0; j < neurons_number; j++)
       {
           buffer << "input_gate_" << to_string(j+1) << " = " << write_recurrent_activation_function_expression() << " (" << input_biases[j] << "+";

           for(Index i = 0; i < inputs_number; i++)
           {
               buffer << inputs_names[i] << "*" << input_weights.get_column(j)(i) << "+";
           }

           for(Index k = 0; k < neurons_number-1; k++)
           {
               buffer << "hidden_state_" << to_string(k+1) << "(t-1)*" << input_recurrent_weights.get_column(j)[k] << "+";
           }

           buffer << "hidden_state_" << to_string(neurons_number) << "(t-1)*" << input_recurrent_weights.get_column(j)[neurons_number-1] << ");\n";
       }

       // State gate

       for(Index j = 0; j < neurons_number; j++)
       {
           buffer << "state_gate_" << to_string(j+1) << " = " << write_activation_function_expression() << " (" << state_biases[j] << "+";

           for(Index i = 0; i < inputs_number; i++)
           {
               buffer << inputs_names[i] << "*" << state_weights.get_column(j)(i) << "+";
           }

           for(Index k = 0; k < neurons_number-1; k++)
           {
               buffer << "hidden_state_" << to_string(k+1) << "(t-1)*" << state_recurrent_weights.get_column(j)[k] << "+";
           }

           buffer << "hidden_state_" << to_string(neurons_number) << "(t-1)*" << state_recurrent_weights.get_column(j)[neurons_number-1] << ");\n";
       }

       // Output gate

       for(Index j = 0; j < neurons_number; j++)
       {
           buffer << "output_gate_" << to_string(j+1) << " = " << write_recurrent_activation_function_expression() << " (" << output_biases[j] << "+";

           for(Index i = 0; i < inputs_number; i++)
           {
               buffer << inputs_names[i] << "*" << output_weights.get_column(j)(i) << "+";
           }

           for(Index k = 0; k < neurons_number-1; k++)
           {
               buffer << "hidden_state_" << to_string(k+1) << "(t-1)*" << output_recurrent_weights.get_column(j)[k] << "+";
           }

           buffer << "hidden_state_" << to_string(neurons_number) << "(t-1)*" << output_recurrent_weights.get_column(j)[neurons_number-1] << ");\n";
       }

       // Cell state

       for(Index i = 0; i < neurons_number; i++)
       {
            buffer << "cell_state_" << to_string(i+1) << "(t) = forget_gate_" << to_string(i+1) << "*cell_state_" << to_string(i+1) << "(t-1)+input_gate_" << to_string(i+1) << "*state_gate_" << to_string(i+1) << ";\n";
       }

       // Hidden state

       for(Index i = 0; i < neurons_number; i++)
       {
            buffer << "hidden_state_" << to_string(i+1) << "(t) = output_gate_" << to_string(i+1) << "*" << write_activation_function_expression() << "(cell_state_" << to_string(i+1) << ");\n";
       }

       // Output

       for(Index i = 0; i < neurons_number; i++)
       {
           buffer << outputs_names[i] << " = " << "hidden_state_" << to_string(i+1) << "(t);\n";
       }

       return buffer.str();
    */
    return string();
}

void LongShortTermMemoryLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Perceptron layer

    const tinyxml2::XMLElement* long_short_term_memory_layer_element = document.FirstChildElement("LongShortTermMemoryLayer");

    if(!long_short_term_memory_layer_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "PerceptronLayer element is nullptr.\n";

        throw logic_error(buffer.str());
    }


    // Layer name

    const tinyxml2::XMLElement* layer_name_element = long_short_term_memory_layer_element->FirstChildElement("LayerName");

    if(!layer_name_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "LayerName element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(layer_name_element->GetText())
    {
        set_layer_name(layer_name_element->GetText());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = long_short_term_memory_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "InputsNumber element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(inputs_number_element->GetText())
    {
        set_inputs_number(static_cast<Index>(stoi(inputs_number_element->GetText())));
    }

    // Neurons number

    const tinyxml2::XMLElement* neurons_number_element = long_short_term_memory_layer_element->FirstChildElement("NeuronsNumber");

    if(!neurons_number_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "NeuronsNumber element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(neurons_number_element->GetText())
    {
        set_neurons_number(static_cast<Index>(stoi(neurons_number_element->GetText())));
    }

    // Activation function

    const tinyxml2::XMLElement* activation_function_element = long_short_term_memory_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "ActivationFunction element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(activation_function_element->GetText())
    {
        set_activation_function(activation_function_element->GetText());
    }

    // Recurrent activation function

    const tinyxml2::XMLElement* recurrent_activation_function_element = long_short_term_memory_layer_element->FirstChildElement("RecurrentActivationFunction");

    if(!recurrent_activation_function_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "ActivationFunction element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(recurrent_activation_function_element->GetText())
    {
        set_recurrent_activation_function(recurrent_activation_function_element->GetText());
    }

    // Parameters

    const tinyxml2::XMLElement* parameters_element = long_short_term_memory_layer_element->FirstChildElement("Parameters");

    if(!parameters_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
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


void LongShortTermMemoryLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Long short term memory layer

    file_stream.OpenElement("LongShortTermMemoryLayer");

    // Layer name
    file_stream.OpenElement("LayerName");
    buffer.str("");
    buffer << layer_name;
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

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

    // Recurrent activation function

    file_stream.OpenElement("RecurrentActivationFunction");

    file_stream.PushText(write_recurrent_activation_function().c_str());

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

    // Long short term memory layer (end tag)

    file_stream.CloseElement();
}

string LongShortTermMemoryLayer::write_recurrent_activation_function_expression() const
{
    switch(recurrent_activation_function)
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
        return write_recurrent_activation_function();
    }
    }
}


string LongShortTermMemoryLayer::write_activation_function_expression() const
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
