//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// OpeNN Includes

#include "long_short_term_memory_layer.h"

namespace opennn
{

/// Default constructor.
/// It creates a empty layer object, with no neurons.
/// This constructor also initializes the rest of the class members to their default values.

LongShortTermMemoryLayer::LongShortTermMemoryLayer() : Layer()
{
    set();

    layer_type = Type::LongShortTermMemory;
}


/// Layer architecture constructor.
/// It creates a layer object with given numbers of inputs and neurons.
/// It also initializes the parameters at random.
/// This constructor also initializes the rest of the class members to their default values.
/// @param new_inputs_number Number of inputs in the layer.
/// @param new_neurons_number Number of neurons in the layer.

LongShortTermMemoryLayer::LongShortTermMemoryLayer(const Index& new_inputs_number, const Index& new_neurons_number) : Layer()
{
    set(new_inputs_number, new_neurons_number);

    layer_type = Type::LongShortTermMemory;
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

    Index current_position = forget_biases.size();

    // Biases

    copy(forget_biases.data(),
         forget_biases.data() + forget_biases.size(),
         parameters.data());

    copy(input_biases.data(),
         input_biases.data() + input_biases.size(),
         parameters.data() + current_position);

    current_position += input_biases.size();

    copy(state_biases.data(),
         state_biases.data() + state_biases.size(),
         parameters.data() + current_position);

    current_position += state_biases.size();

    copy(output_biases.data(),
         output_biases.data() + output_biases.size(),
         parameters.data() + current_position);

    current_position += output_biases.size();

    // Weights

    copy(forget_weights.data(),
         forget_weights.data() + forget_weights.size(),
         parameters.data() + current_position);

    current_position += forget_weights.size();

    copy(input_weights.data(),
         input_weights.data() + input_weights.size(),
         parameters.data() + current_position);

    current_position += input_weights.size();

    copy(state_weights.data(),
         state_weights.data() + state_weights.size(),
         parameters.data() + current_position);

    current_position += state_weights.size();

    copy(output_weights.data(),
         output_weights.data() + output_weights.size(),
         parameters.data() + current_position);

    current_position += output_weights.size();

    // Recurrent weights

    copy(forget_recurrent_weights.data(),
         forget_recurrent_weights.data() + forget_recurrent_weights.size(),
         parameters.data() + current_position);

    current_position += forget_recurrent_weights.size();

    copy(input_recurrent_weights.data(),
         input_recurrent_weights.data() + input_recurrent_weights.size(),
         parameters.data() + current_position);

    current_position += input_recurrent_weights.size();

    copy(state_recurrent_weights.data(),
         state_recurrent_weights.data() + state_recurrent_weights.size(),
         parameters.data() + current_position);

    current_position += state_recurrent_weights.size();

    copy(output_recurrent_weights.data(),
         output_recurrent_weights.data() + output_recurrent_weights.size(),
         parameters.data() + current_position);

    return parameters;
}


Tensor< TensorMap< Tensor<type, 1> >*, 1> LongShortTermMemoryLayer::get_layer_parameters()
{
    Tensor< TensorMap< Tensor<type, 1> >*, 1> layer_parameters(12);

    layer_parameters(0) = new TensorMap<Tensor<type, 1>>(input_biases.data(), input_biases.size());
    layer_parameters(1) = new TensorMap<Tensor<type, 1>>(forget_biases.data(), input_biases.size());
    layer_parameters(2) = new TensorMap<Tensor<type, 1>>(state_biases.data(), state_biases.size());
    layer_parameters(3) = new TensorMap<Tensor<type, 1>>(output_biases.data(), output_biases.size());
    layer_parameters(4) = new TensorMap<Tensor<type, 1>>(input_weights.data(), input_weights.size());
    layer_parameters(5) = new TensorMap<Tensor<type, 1>>(forget_weights.data(), forget_weights.size());
    layer_parameters(6) = new TensorMap<Tensor<type, 1>>(state_weights.data(), state_weights.size());
    layer_parameters(7) = new TensorMap<Tensor<type, 1>>(output_weights.data(), output_weights.size());
    layer_parameters(8) = new TensorMap<Tensor<type, 1>>(input_recurrent_weights.data(), input_recurrent_weights.size());
    layer_parameters(9) = new TensorMap<Tensor<type, 1>>(forget_recurrent_weights.data(), forget_recurrent_weights.size());
    layer_parameters(10) = new TensorMap<Tensor<type, 1>>(state_recurrent_weights.data(), state_recurrent_weights.size());
    layer_parameters(11) = new TensorMap<Tensor<type, 1>>(output_recurrent_weights.data(), output_recurrent_weights.size());

    return layer_parameters;
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
/// This can be Logistic, HyperbolicTangent, Threshold, SymmetricThreshold, Linear, RectifiedLinear,
/// ScaledExponentialLinear.

string LongShortTermMemoryLayer::write_activation_function() const
{
    switch(activation_function)
    {
    case ActivationFunction::Logistic: return "Logistic";

    case ActivationFunction::HyperbolicTangent: return "HyperbolicTangent";

    case ActivationFunction::Threshold: return "Threshold";

    case ActivationFunction::SymmetricThreshold: return "SymmetricThreshold";

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


/// Returns a string with the name of the layer recurrent activation function.
/// This can be Logistic, HyperbolicTangent, Threshold, SymmetricThreshold, Linear, RectifiedLinear, ScaledExponentialLinear.

string LongShortTermMemoryLayer::write_recurrent_activation_function() const
{
    switch(recurrent_activation_function)
    {
    case ActivationFunction::Logistic: return "Logistic";

    case ActivationFunction::HyperbolicTangent: return "HyperbolicTangent";

    case ActivationFunction::Threshold: return "Threshold";

    case ActivationFunction::SymmetricThreshold: return "SymmetricThreshold";

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


/// Returns true if messages from this class are displayed on the screen,
/// or false if messages from this class are not displayed on the screen.

const bool& LongShortTermMemoryLayer::get_display() const
{
    return display;
}


/// Sets an empty layer, wihtout any neuron.
/// It also sets the rest of the members to their default values.

void LongShortTermMemoryLayer::set()
{
    set_default();
}


/// Sets new numbers of inputs and neurons in the layer.
/// It also sets the rest of the members to their default values.
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
    layer_type = Type::LongShortTermMemory;
}


void LongShortTermMemoryLayer::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}


/// Sets a new number of inputs in the layer.
/// It inializes the new biases, weights and recurrent weights at random.
/// @param new_inputs_number Number of layer inputs.

void LongShortTermMemoryLayer::set_inputs_number(const Index& new_inputs_number)
{
    const Index neurons_number = get_neurons_number();

    set(new_inputs_number, neurons_number);
}


/// Sets a new size of inputs in the layer.
/// It initializes the new biases, weights and recurrent weights at random.
/// @param size dimensions of layer inputs.

void LongShortTermMemoryLayer::set_input_shape(const Tensor<Index, 1>& size)
{
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
#ifdef OPENNN_DEBUG
check_size(new_parameters, get_parameters_number(), LOG);
#endif

    const Index neurons_number = get_neurons_number();
    const Index inputs_number = get_inputs_number();

    Index current_index = index;

    // Biases

    Index size = neurons_number;

    copy(new_parameters.data() + current_index,
         new_parameters.data() + current_index + size,
         forget_biases.data());

    current_index += size;

    copy(new_parameters.data() + current_index,
         new_parameters.data() + current_index + size,
         input_biases.data());

    current_index += size;

    copy(new_parameters.data() + current_index,
         new_parameters.data() + current_index + size,
         state_biases.data());

    current_index += size;

    copy(new_parameters.data() + current_index,
         new_parameters.data() + current_index + size,
         output_biases.data());

    current_index += size;

    // Weights

    size = inputs_number*neurons_number;

    copy(new_parameters.data() + current_index,
         new_parameters.data() + current_index + size,
         forget_weights.data());

    current_index += size;

    copy(new_parameters.data() + current_index,
         new_parameters.data() + current_index + size,
         input_weights.data());

    current_index += size;

    copy(new_parameters.data() + current_index,
         new_parameters.data() + current_index + size,
         state_weights.data());

    current_index += size;

    copy(new_parameters.data() + current_index,
         new_parameters.data() + current_index + size,
         output_weights.data());

    current_index += size;

    // Recurrent weights

    size = neurons_number*neurons_number;

    copy(new_parameters.data() + current_index,
         new_parameters.data() + current_index + size,
         forget_recurrent_weights.data());

    current_index += size;

    copy(new_parameters.data() + current_index,
         new_parameters.data() + current_index + size,
         input_recurrent_weights.data());

    current_index += size;

    copy(new_parameters.data() + current_index,
         new_parameters.data() + current_index + size,
         state_recurrent_weights.data());

    current_index += size;

    copy(new_parameters.data() + current_index,
         new_parameters.data() + current_index + size,
         output_recurrent_weights.data());

    current_index += size;
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
        activation_function = ActivationFunction::Logistic;
    }
    else if(new_activation_function_name == "HyperbolicTangent")
    {
        activation_function = ActivationFunction::HyperbolicTangent;
    }
    else if(new_activation_function_name == "Threshold")
    {
        activation_function = ActivationFunction::Threshold;
    }
    else if(new_activation_function_name == "SymmetricThreshold")
    {
        activation_function = ActivationFunction::SymmetricThreshold;
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
        ostringstream buffer;

        buffer << "OpenNN Exception: neuron class.\n"
               << "void set_activation_function(const string&) method.\n"
               << "Unknown activation function: " << new_activation_function_name << ".\n";

        throw invalid_argument(buffer.str());
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
        recurrent_activation_function = ActivationFunction::Logistic;
    }
    else if(new_recurrent_activation_function_name == "HyperbolicTangent")
    {
        recurrent_activation_function = ActivationFunction::HyperbolicTangent;
    }
    else if(new_recurrent_activation_function_name == "Threshold")
    {
        recurrent_activation_function = ActivationFunction::Threshold;
    }
    else if(new_recurrent_activation_function_name == "SymmetricThreshold")
    {
        recurrent_activation_function = ActivationFunction::SymmetricThreshold;
    }
    else if(new_recurrent_activation_function_name == "Linear")
    {
        recurrent_activation_function = ActivationFunction::Linear;
    }
    else if(new_recurrent_activation_function_name == "RectifiedLinear")
    {
        recurrent_activation_function = ActivationFunction::RectifiedLinear;
    }
    else if(new_recurrent_activation_function_name == "ScaledExponentialLinear")
    {
        recurrent_activation_function = ActivationFunction::ScaledExponentialLinear;
    }
    else if(new_recurrent_activation_function_name == "SoftPlus")
    {
        recurrent_activation_function = ActivationFunction::SoftPlus;
    }
    else if(new_recurrent_activation_function_name == "SoftSign")
    {
        recurrent_activation_function = ActivationFunction::SoftSign;
    }
    else if(new_recurrent_activation_function_name == "HardSigmoid")
    {
        recurrent_activation_function = ActivationFunction::HardSigmoid;
    }
    else if(new_recurrent_activation_function_name == "ExponentialLinear")
    {
        recurrent_activation_function = ActivationFunction::ExponentialLinear;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: neuron class.\n"
               << "void set_recurrent_activation_function(const string&) method.\n"
               << "Unknown activation function: " << new_recurrent_activation_function_name << ".\n";

        throw invalid_argument(buffer.str());
    }
}


/// Sets the timesteps of the layer from a Index.
/// @param new_timesteps New set of timesteps in the layer.

void LongShortTermMemoryLayer::set_timesteps(const Index& new_timesteps)
{
    timesteps = new_timesteps;
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void LongShortTermMemoryLayer::set_display(const bool& new_display)
{
    display = new_display;
}


/// Initializes the biases of all the neurons in the layer with a given value.
/// @param value Biases initialization value.

void LongShortTermMemoryLayer::set_biases_constant(const type& value)
{
    forget_biases.setConstant(value);
    input_biases.setConstant(value);
    state_biases.setConstant(value);
    output_biases.setConstant(value);
}


/// Initializes the forget biases of all the neurons in the layer with a given value.
/// @param value Forget biases initialization value.

void LongShortTermMemoryLayer::set_forget_biases_constant(const type& value)
{
    forget_biases.setConstant(value);
}


/// Initializes the input biases of all the neurons in the layer with a given value.
/// @param value Input biases initialization value.

void LongShortTermMemoryLayer::set_input_biases_constant(const type& value)
{
    input_biases.setConstant(value);
}


/// Initializes the state biases of all the neurons in the layer with a given value.
/// @param value State biases initialization value.

void LongShortTermMemoryLayer::set_state_biases_constant(const type& value)
{
    state_biases.setConstant(value);
}


/// Initializes the oputput biases of all the neurons in the layer with a given value.
/// @param value Output biases initialization value.

void LongShortTermMemoryLayer::set_output_biases_constant(const type& value)
{
    output_biases.setConstant(value);
}


/// Initializes the weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Weights initialization value.

void LongShortTermMemoryLayer::set_weights_constant(const type& value)
{
    forget_weights.setConstant(value);
    input_weights.setConstant(value);
    state_weights.setConstant(value);
    output_weights.setConstant(value);
}


/// Initializes the forget weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Forget weights initialization value.

void LongShortTermMemoryLayer::set_forget_weights_constant(const type& value)
{
    forget_weights.setConstant(value);
}


/// Initializes the input weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Input weights initialization value.

void LongShortTermMemoryLayer::set_input_weights_constant(const type& value)
{
    input_weights.setConstant(value);
}


/// Initializes the state weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value State weights initialization value.

void LongShortTermMemoryLayer::set_state_weights_constant(const type& value)
{
    state_weights.setConstant(value);
}


/// Initializes the output weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Output weights initialization value.

void LongShortTermMemoryLayer::set_output_weights_constant(const type&  value)
{
    output_weights.setConstant(value);
}


/// Initializes the recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Recurrent weights initialization value.

void LongShortTermMemoryLayer::set_recurrent_weights_constant(const type& value)
{
    forget_recurrent_weights.setConstant(value);
    input_recurrent_weights.setConstant(value);
    state_recurrent_weights.setConstant(value);
    output_recurrent_weights.setConstant(value);
}


/// Initializes the forget recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Forget recurrent weights initialization value.

void LongShortTermMemoryLayer::set_forget_recurrent_weights_constant(const type& value)
{
    forget_recurrent_weights.setConstant(value);
}


/// Initializes the input recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Input recurrent weights initialization value.

void LongShortTermMemoryLayer::set_input_recurrent_weights_constant(const type& value)
{
    input_recurrent_weights.setConstant(value);
}


/// Initializes the state recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value State recurrent weights initialization value.

void LongShortTermMemoryLayer::set_state_recurrent_weights_constant(const type& value)
{
    state_recurrent_weights.setConstant(value);
}


/// Initializes the output recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Output recurrent weights initialization value.

void LongShortTermMemoryLayer::set_output_recurrent_weights_constant(const type&  value)
{
    output_recurrent_weights.setConstant(value);
}


/// Initializes hidden states of the layer with a given value.
/// @param value Hidden states initialization value.

void LongShortTermMemoryLayer::set_hidden_states_constant(const type& value)
{
    hidden_states.setConstant(value);
}


/// Initializes cell states of the layer with a given value.
/// @param value Cell states initialization value.

void LongShortTermMemoryLayer::set_cell_states_constant(const type& value)
{
    cell_states.setConstant(value);
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
    const type minimum = type(-0.2);
    const type maximum = type(0.2);

    // Biases

    for(Index i = 0; i < forget_biases.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        forget_biases(i) = minimum + (maximum - minimum)*random;
    }

    for(Index i = 0; i < input_biases.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        input_biases(i) = minimum + (maximum - minimum)*random;
    }

    for(Index i = 0; i < state_biases.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        state_biases(i) = minimum + (maximum - minimum)*random;
    }

    for(Index i = 0; i < output_biases.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        output_biases(i) = minimum + (maximum - minimum)*random;
    }

    // Weights

    for(Index i = 0; i < forget_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        forget_weights(i) = minimum + (maximum - minimum)*random;
    }

    for(Index i = 0; i < input_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        input_weights(i) = minimum + (maximum - minimum)*random;
    }

    for(Index i = 0; i < state_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        state_weights(i) = minimum + (maximum - minimum)*random;
    }

    for(Index i = 0; i < output_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        output_weights(i) = minimum + (maximum - minimum)*random;
    }

    // Recurrent weights

    for(Index i = 0; i < forget_recurrent_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        forget_recurrent_weights(i) = minimum + (maximum - minimum)*random;
    }

    for(Index i = 0; i < input_recurrent_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        input_recurrent_weights(i) = minimum + (maximum - minimum)*random;
    }

    for(Index i = 0; i < state_recurrent_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        state_recurrent_weights(i) = minimum + (maximum - minimum)*random;
    }

    for(Index i = 0; i < output_recurrent_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        output_recurrent_weights(i) = minimum + (maximum - minimum)*random;
    }
}


void LongShortTermMemoryLayer::calculate_combinations(type* inputs_data, const Tensor<Index, 1>& inputs_dimensions,
                                                      const Tensor<type, 2>& weights,
                                                      const Tensor<type, 2>& recurrent_weights,
                                                      const Tensor<type, 1>& biases,
                                                      type* combinations_data, const Tensor<Index, 1>& combinations_dimensions)
{

#ifdef OPENNN_DEBUG
    if(inputs_dimensions.size() != 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void calculate_combinations(type*, const Tensor<Index, 1>&, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 1>&, type*, const Tensor<Index, 1>&) method"
               << "Inputs rank must be equal to 1.\n";

        throw invalid_argument(buffer.str());

    }
    if(inputs_dimensions(0) != get_inputs_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void calculate_combinations(type*, const Tensor<Index, 1>&, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 1>&, type*, const Tensor<Index, 1>&) method"
               << "Inputs dimensions must be equal to inputs number, " << get_inputs_number() << ".\n";

        throw invalid_argument(buffer.str());
    }
#endif

    TensorMap<Tensor<type, 1>> inputs(inputs_data, inputs_dimensions(0));

    TensorMap<Tensor<type, 1>> combinations(combinations_data, combinations_dimensions(0));

    combinations.device(*thread_pool_device) = inputs.contract(weights, AT_B);

    combinations.device(*thread_pool_device) += biases;

    combinations.device(*thread_pool_device) += hidden_states.contract(recurrent_weights, AT_B);
}


void LongShortTermMemoryLayer::calculate_activations(type* combinations_data, Tensor<Index,1> &combinations_dimensions, type* activations_data, Tensor<Index,1> &activations_dimensions)
{
    switch(activation_function)
    {
    case ActivationFunction::Linear: linear(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::Logistic: logistic(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::Threshold: threshold(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::SymmetricThreshold: symmetric_threshold(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::RectifiedLinear: rectified_linear(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::SoftPlus: soft_plus(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::SoftSign: soft_sign(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::HardSigmoid: hard_sigmoid(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::ExponentialLinear: exponential_linear(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    default: rectified_linear(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;
    }
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_activations(Tensor<type, 1>& combinations_1d) const
{
#ifdef OPENNN_DEBUG

    const Index neurons_number = get_neurons_number();

    const Index combinations_columns_number = combinations_1d.size();

    if(combinations_columns_number != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 2> calculate_activations(const Tensor<type, 1>&) const method.\n"
               << "Size of combinations must be equal to number of neurons.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<type, 1> activations_1d(combinations_1d.size());

    Tensor<Index, 1> combinations_dimensions = get_dimensions(combinations_1d);
    Tensor<Index, 1> activations_dimensions = get_dimensions(activations_1d);

    switch(activation_function)
    {
    case ActivationFunction::Linear: linear(combinations_1d.data(), combinations_dimensions, activations_1d.data(), activations_dimensions); break;

    case ActivationFunction::Logistic: logistic(combinations_1d.data(), combinations_dimensions, activations_1d.data(), activations_dimensions); break;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(combinations_1d.data(), combinations_dimensions, activations_1d.data(), activations_dimensions); break;

    case ActivationFunction::Threshold: threshold(combinations_1d.data(), combinations_dimensions, activations_1d.data(), activations_dimensions); break;

    case ActivationFunction::SymmetricThreshold: symmetric_threshold(combinations_1d.data(), combinations_dimensions, activations_1d.data(), activations_dimensions); break;

    case ActivationFunction::RectifiedLinear: rectified_linear(combinations_1d.data(), combinations_dimensions, activations_1d.data(), activations_dimensions); break;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(combinations_1d.data(), combinations_dimensions, activations_1d.data(), activations_dimensions); break;

    case ActivationFunction::SoftPlus: soft_plus(combinations_1d.data(), combinations_dimensions, activations_1d.data(), activations_dimensions); break;

    case ActivationFunction::SoftSign: soft_sign(combinations_1d.data(), combinations_dimensions, activations_1d.data(), activations_dimensions); break;

    case ActivationFunction::HardSigmoid: hard_sigmoid(combinations_1d.data(), combinations_dimensions, activations_1d.data(), activations_dimensions); break;

    case ActivationFunction::ExponentialLinear: exponential_linear(combinations_1d.data(), combinations_dimensions, activations_1d.data(), activations_dimensions); break;

    default: rectified_linear(combinations_1d.data(), combinations_dimensions, activations_1d.data(), activations_dimensions); break;
    }

    return activations_1d;
}


void LongShortTermMemoryLayer::calculate_recurrent_activations(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,
                                                               type* activations_data, Tensor<Index, 1>& activations_dimensions)
{

    if(combinations_dimensions.size() != activations_dimensions.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void calculate_recurrent_activations(type*, Tensor<Index, 1>&, type*, Tensor<Index, 1>&) method.\n"
               << "Combinations and activations must have the same rank.\n";

        throw invalid_argument(buffer.str());
    }
    if(combinations_dimensions(combinations_dimensions.size() -1) != get_neurons_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void calculate_recurrent_activations(const Tensor<type, 2>&) const method.\n"
               << "Number of columns(" << combinations_dimensions(combinations_dimensions.size() -1) << ") of combinations must be equal to number of neurons(" << get_neurons_number() << ").\n";

        throw invalid_argument(buffer.str());
    }

    switch(recurrent_activation_function)
    {
    case ActivationFunction::Linear: linear(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::Logistic: logistic(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::Threshold: threshold(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::SymmetricThreshold: symmetric_threshold(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::RectifiedLinear: rectified_linear(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::SoftPlus: soft_plus(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::SoftSign: soft_sign(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::HardSigmoid: hard_sigmoid(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    case ActivationFunction::ExponentialLinear: exponential_linear(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;

    default: rectified_linear(combinations_data, combinations_dimensions, activations_data, activations_dimensions); return;
    }
}


void LongShortTermMemoryLayer::calculate_activations_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,
                                       type* activations_data, Tensor<Index, 1>& activations_dimensions,
                                       type* derivatives_data, Tensor<Index, 1>& derivatives_dimensions)
{

    const Index neurons_number = get_neurons_number();

    const Index combinations_columns_number = combinations_dimensions(combinations_dimensions.size() - 1);

    if(combinations_columns_number != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void calculate_activations_derivatives(type*, Tensor<Index, 1>&, type*, Tensor<Index, 1>&, type*, Tensor<Index, 1>&);\n"
               << "Number of columns(" << combinations_columns_number << ") of combinations must be equal to number of neurons(" << neurons_number << ").\n";

        throw invalid_argument(buffer.str());
    }

    switch(activation_function)
    {
    case ActivationFunction::Linear: linear_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::Logistic: logistic_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::Threshold: threshold_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::SymmetricThreshold: symmetric_threshold_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::RectifiedLinear: rectified_linear_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::SoftPlus: soft_plus_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::SoftSign: soft_sign_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::HardSigmoid: hard_sigmoid_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::ExponentialLinear: exponential_linear_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    default: rectified_linear_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    }
}

void LongShortTermMemoryLayer::calculate_recurrent_activations_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,
                                                 type* activations_data, Tensor<Index, 1>& activations_dimensions,
                                                 type* derivatives_data, Tensor<Index, 1>& derivatives_dimensions)
{
    const Index neurons_number = get_neurons_number();

    const Index combinations_columns_number = combinations_dimensions(combinations_dimensions.size() - 1);

    if(combinations_columns_number != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void calculate_activations_derivatives(type*, Tensor<Index, 1>&, type*, Tensor<Index, 1>&, type*, Tensor<Index, 1>&);\n"
               << "Number of columns(" << combinations_columns_number << ") of combinations must be equal to number of neurons(" << neurons_number << ").\n";

        throw invalid_argument(buffer.str());
    }

    switch(recurrent_activation_function)
    {
    case ActivationFunction::Linear: linear_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::Logistic: logistic_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::Threshold: threshold_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::SymmetricThreshold: symmetric_threshold_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::RectifiedLinear: rectified_linear_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::SoftPlus: soft_plus_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::SoftSign: soft_sign_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::HardSigmoid: hard_sigmoid_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    case ActivationFunction::ExponentialLinear: exponential_linear_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    default: rectified_linear_derivatives(combinations_data, combinations_dimensions, activations_data, activations_dimensions, derivatives_data, derivatives_dimensions); return;

    }
}


void LongShortTermMemoryLayer::calculate_outputs(type* inputs_data, const Tensor<Index, 1>& inputs_dimensions,
                                                 type* outputs_data, const Tensor<Index, 1>& outputs_dimensions)
{
#ifdef OPENNN_DEBUG

    const Index inputs_number = get_inputs_number();

    const Index inputs_columns_number = inputs_dimensions(1);

    if(inputs_columns_number != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
               << "Number of columns (" << inputs_columns_number << ") of inputs matrix must be equal to number of inputs (" << inputs_number << ").\n";

        throw invalid_argument(buffer.str());
    }
#endif

    const Index samples_number = inputs_dimensions(0);

    const Index neurons_number = get_neurons_number();

    if(outputs_dimensions(0) != samples_number || outputs_dimensions(1) != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void calculate_outputs(type*, Tensor<Index, 1>&, type*, Tensor<Index, 1>&)"
               << "Outputs dimensions must be equal to " << samples_number << " and " << neurons_number << ".\n";

        throw invalid_argument(buffer.str());
    }

    Tensor<type, 1> forget_combinations(neurons_number);
    Tensor<type, 1> forget_activations(neurons_number);

    Tensor<type, 1> input_combinations(neurons_number);
    Tensor<type, 1> input_activations(neurons_number);

    Tensor<type, 1> state_combinations(neurons_number);
    Tensor<type, 1> state_activations(neurons_number);

    Tensor<type, 1> output_combinations(neurons_number);
    Tensor<type, 1> output_activations(neurons_number);

    TensorMap<Tensor<type,2>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1));
    TensorMap<Tensor<type,2>> outputs(outputs_data, samples_number, neurons_number);

    Tensor<Index, 1> current_inputs_dimensions;

    Tensor<Index, 1> combinations_dimensions;
    Tensor<Index, 1> activations_dimensions;

    for(Index i = 0; i < samples_number; i++)
    {
        if(i%timesteps == 0)
        {
            hidden_states.setZero();
            cell_states.setZero();
        }

        Tensor<type, 1> current_inputs = inputs.chip(i, 0);

        current_inputs_dimensions = get_dimensions(current_inputs);

#pragma omp parallel
        {
            combinations_dimensions = get_dimensions(forget_combinations);
            activations_dimensions = get_dimensions(forget_activations);

            calculate_combinations(current_inputs.data(), current_inputs_dimensions, forget_weights, forget_recurrent_weights, forget_biases, forget_combinations.data(), combinations_dimensions);
            calculate_recurrent_activations(forget_combinations.data(), combinations_dimensions, forget_activations.data(), activations_dimensions);

            combinations_dimensions = get_dimensions(input_combinations);
            activations_dimensions = get_dimensions(input_activations);

            calculate_combinations(current_inputs.data(), current_inputs_dimensions, input_weights, input_recurrent_weights, input_biases, input_combinations.data(), combinations_dimensions);
            calculate_recurrent_activations(input_combinations.data(), combinations_dimensions, input_activations.data(), activations_dimensions);

            combinations_dimensions = get_dimensions(state_combinations);
            activations_dimensions = get_dimensions(state_activations);

            calculate_combinations(current_inputs.data(), current_inputs_dimensions, state_weights, state_recurrent_weights, state_biases, state_combinations.data(), combinations_dimensions);
            calculate_activations(state_combinations.data(), combinations_dimensions, state_activations.data(), activations_dimensions);

            combinations_dimensions = get_dimensions(output_combinations);
            activations_dimensions = get_dimensions(output_activations);

            calculate_combinations(current_inputs.data(), current_inputs_dimensions, output_weights, output_recurrent_weights, output_biases, output_combinations.data(), combinations_dimensions);
            calculate_recurrent_activations(output_combinations.data(), combinations_dimensions, output_activations.data(), activations_dimensions);
        }

        cell_states = forget_activations * cell_states + input_activations * state_activations;

        combinations_dimensions = get_dimensions(cell_states);
        activations_dimensions = get_dimensions(hidden_states);

        calculate_activations(cell_states.data(), combinations_dimensions, hidden_states.data(), activations_dimensions);
        hidden_states *= output_activations;

        for(Index j = 0; j < neurons_number; j++)
            outputs(i,j) = hidden_states(j);
    }
}


void LongShortTermMemoryLayer::calculate_hidden_delta(LayerForwardPropagation* next_forward_propagation,
                                                      LayerBackPropagation* next_back_propagation,
                                                      LayerBackPropagation* back_propagation) const
{
    LongShortTermMemoryLayerBackPropagation* long_short_term_memory_layer_back_propagation =
            static_cast<LongShortTermMemoryLayerBackPropagation*>(back_propagation);

    switch(next_back_propagation->layer_pointer->get_type())
    {
    case Type::Perceptron:
    {
        PerceptronLayerForwardPropagation* next_perceptron_layer_forward_propagation =
                static_cast<PerceptronLayerForwardPropagation*>(next_forward_propagation);

        PerceptronLayerBackPropagation* next_perceptron_layer_back_propagation =
                static_cast<PerceptronLayerBackPropagation*>(next_back_propagation);

        calculate_hidden_delta_perceptron(next_perceptron_layer_forward_propagation,
                                          next_perceptron_layer_back_propagation,
                                          long_short_term_memory_layer_back_propagation);
    }
        break;

    case Type::Probabilistic:
    {
        ProbabilisticLayerForwardPropagation* next_probabilistic_layer_forward_propagation =
                static_cast<ProbabilisticLayerForwardPropagation*>(next_forward_propagation);

        ProbabilisticLayerBackPropagation* next_probabilistic_layer_back_propagation =
                static_cast<ProbabilisticLayerBackPropagation*>(next_back_propagation);

        calculate_hidden_delta_probabilistic(next_probabilistic_layer_forward_propagation,
                                             next_probabilistic_layer_back_propagation,
                                             long_short_term_memory_layer_back_propagation);
    }
        break;

    default: return;
    }
}


void LongShortTermMemoryLayer::calculate_hidden_delta_perceptron(PerceptronLayerForwardPropagation* next_forward_propagation,
                                                                 PerceptronLayerBackPropagation* next_back_propagation,
                                                                 LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Tensor<type, 2>& next_synaptic_weights = static_cast<PerceptronLayer*>(next_back_propagation->layer_pointer)->get_synaptic_weights();

    const TensorMap<Tensor<type,2>> next_layer_deltas(next_back_propagation->deltas_data, next_back_propagation->deltas_dimensions(0), next_back_propagation->deltas_dimensions(1));
    TensorMap<Tensor<type,2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));

    deltas.device(*thread_pool_device) = (next_layer_deltas*next_forward_propagation->activations_derivatives).contract(next_synaptic_weights, A_BT);
}


void LongShortTermMemoryLayer::calculate_hidden_delta_probabilistic(ProbabilisticLayerForwardPropagation* next_forward_propagation,
                                                                    ProbabilisticLayerBackPropagation* next_back_propagation,
                                                                    LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const ProbabilisticLayer* probabilistic_layer_pointer = static_cast<ProbabilisticLayer*>(next_back_propagation->layer_pointer);

    const Tensor<type, 2>& next_synaptic_weights = probabilistic_layer_pointer->get_synaptic_weights();

    const TensorMap<Tensor<type, 2>> next_deltas(next_back_propagation->deltas_data, next_back_propagation->deltas_dimensions(0), next_back_propagation->deltas_dimensions(1));;
    TensorMap<Tensor<type, 2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));

    if(probabilistic_layer_pointer->get_neurons_number() == 1) // Binary
    {
        deltas.device(*thread_pool_device) =
                (next_deltas*next_forward_propagation->activations_derivatives).contract(next_synaptic_weights, A_BT);
    }
    else // Multiple
    {
        const Index samples_number = next_deltas.dimension(0);
        const Index outputs_number = next_deltas.dimension(1);
        const Index next_layer_neurons_number = probabilistic_layer_pointer->get_neurons_number();

        if(outputs_number != next_layer_neurons_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                   << "void calculate_hidden_delta_probabilistic(ProbabilisticLayerForwardPropagation*,ProbabilisticLayerBackPropagation*,PerceptronLayerBackPropagation*) const.\n"
                   << "Number of columns in delta (" << outputs_number << ") must be equal to number of neurons in probabilistic layer (" << next_layer_neurons_number << ").\n";

            throw invalid_argument(buffer.str());
        }

        if(next_forward_propagation->activations_derivatives.dimension(1) != next_layer_neurons_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                   << "void calculate_hidden_delta_probabilistic(ProbabilisticLayerForwardPropagation*,ProbabilisticLayerBackPropagation*,PerceptronLayerBackPropagation*) const.\n"
                   << "Dimension 1 of activations derivatives (" << outputs_number << ") must be equal to number of neurons in probabilistic layer (" << next_layer_neurons_number << ").\n";

            throw invalid_argument(buffer.str());
        }

        if(next_forward_propagation->activations_derivatives.dimension(2) != next_layer_neurons_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                   << "void calculate_hidden_delta_probabilistic(ProbabilisticLayerForwardPropagation*,ProbabilisticLayerBackPropagation*,PerceptronLayerBackPropagation*) const.\n"
                   << "Dimension 2 of activations derivatives (" << outputs_number << ") must be equal to number of neurons in probabilistic layer (" << next_layer_neurons_number << ").\n";

            throw invalid_argument(buffer.str());
        }

        const Index step = next_layer_neurons_number*next_layer_neurons_number;

        next_back_propagation->biases_derivatives.setZero();

        for(Index i = 0; i < samples_number; i++)
        {
            next_back_propagation->delta_row = next_deltas.chip(i,0);

            TensorMap< Tensor<type, 2> > activations_derivatives_matrix(next_forward_propagation->activations_derivatives.data() + i*step,
                                                                        next_layer_neurons_number, next_layer_neurons_number);

            next_back_propagation->error_combinations_derivatives.chip(i,0) =
                    next_back_propagation->delta_row.contract(activations_derivatives_matrix, AT_B);
        }

        deltas.device(*thread_pool_device) =
                (next_back_propagation->error_combinations_derivatives).contract(next_synaptic_weights, A_BT);
    }
}


// Forward propagate functions

void LongShortTermMemoryLayer::forward_propagate(type* inputs_data,
                                                 const Tensor<Index, 1>& inputs_dimensions,
                                                 LayerForwardPropagation* forward_propagation)
{
    LongShortTermMemoryLayerForwardPropagation* long_short_term_memory_layer_forward_propagation
            = static_cast<LongShortTermMemoryLayerForwardPropagation*>(forward_propagation);

    const Index samples_number = inputs_dimensions(0);
    const Index neurons_number = get_neurons_number();

    if(inputs_dimensions.size() != 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void forward_propagate(type*, const Tensor<Index, 1>&, LayerForwardPropagation*) final.\n"
               << "Inputs rank must be equal to 2.\n";

        throw invalid_argument(buffer.str());
    }

    TensorMap<Tensor<type, 2>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1));

    const Tensor<Index, 1> outputs_dimensions = forward_propagation->outputs_dimensions;

    const TensorMap<Tensor<type, 2>> outputs(forward_propagation->outputs_data, outputs_dimensions(0), outputs_dimensions(1));

    Tensor<Index, 1> current_inputs_dimensions(1);

    Tensor<Index, 1> combinations_dimensions;
    Tensor<Index, 1> activations_dimensions;
    Tensor<Index, 1> derivatives_dimensions;

    Index copy_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(i%timesteps == 0)
        {
            hidden_states.setZero();
            cell_states.setZero();
        }

        long_short_term_memory_layer_forward_propagation->current_inputs = inputs.chip(i,0);

        current_inputs_dimensions.setValues({long_short_term_memory_layer_forward_propagation->current_inputs.size()});

        combinations_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_combinations);

        calculate_combinations(long_short_term_memory_layer_forward_propagation->current_inputs.data(),
                               current_inputs_dimensions,
                               forget_weights,
                               forget_recurrent_weights,
                               forget_biases,
                               long_short_term_memory_layer_forward_propagation->current_forget_combinations.data(),
                               combinations_dimensions);

        combinations_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_combinations);
        activations_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_activations);
        derivatives_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives);

        calculate_recurrent_activations_derivatives(long_short_term_memory_layer_forward_propagation->current_forget_combinations.data(),
                                                    combinations_dimensions,
                                                    long_short_term_memory_layer_forward_propagation->current_forget_activations.data(),
                                                    activations_dimensions,
                                                    long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives.data(),
                                                    derivatives_dimensions);

        combinations_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_combinations);

        calculate_combinations(long_short_term_memory_layer_forward_propagation->current_inputs.data(),
                               current_inputs_dimensions,
                               input_weights,
                               input_recurrent_weights,
                               input_biases,
                               long_short_term_memory_layer_forward_propagation->current_input_combinations.data(),
                               combinations_dimensions);

        combinations_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_combinations);
        activations_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_activations);
        derivatives_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives);

        calculate_recurrent_activations_derivatives(long_short_term_memory_layer_forward_propagation->current_input_combinations.data(),
                                                    combinations_dimensions,
                                                    long_short_term_memory_layer_forward_propagation->current_input_activations.data(),
                                                    activations_dimensions,
                                                    long_short_term_memory_layer_forward_propagation->current_input_activations_derivatives.data(),
                                                    combinations_dimensions);

        combinations_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_combinations);

        calculate_combinations(long_short_term_memory_layer_forward_propagation->current_inputs.data(),
                               current_inputs_dimensions,
                               state_weights,
                               state_recurrent_weights,
                               state_biases,
                               long_short_term_memory_layer_forward_propagation->current_state_combinations.data(),
                               combinations_dimensions);

        combinations_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_combinations);
        activations_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_activations);
        derivatives_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives);

        calculate_recurrent_activations_derivatives(long_short_term_memory_layer_forward_propagation->current_state_combinations.data(),
                                                    combinations_dimensions,
                                                    long_short_term_memory_layer_forward_propagation->current_state_activations.data(),
                                                    activations_dimensions,
                                                    long_short_term_memory_layer_forward_propagation->current_state_activations_derivatives.data(),
                                                    derivatives_dimensions);

        combinations_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_combinations);

        calculate_combinations(long_short_term_memory_layer_forward_propagation->current_inputs.data(),
                               current_inputs_dimensions,
                               output_weights,
                               output_recurrent_weights,
                               output_biases,
                               long_short_term_memory_layer_forward_propagation->current_output_combinations.data(),
                               combinations_dimensions);

        combinations_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_combinations);
        activations_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_activations);
        derivatives_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives);

        calculate_recurrent_activations_derivatives(long_short_term_memory_layer_forward_propagation->current_output_combinations.data(),
                                                    combinations_dimensions,
                                                    long_short_term_memory_layer_forward_propagation->current_output_activations.data(),
                                                    activations_dimensions,
                                                    long_short_term_memory_layer_forward_propagation->current_output_activations_derivatives.data(),
                                                    derivatives_dimensions);

        cell_states = long_short_term_memory_layer_forward_propagation->current_forget_activations * cell_states +
                long_short_term_memory_layer_forward_propagation->current_input_activations * long_short_term_memory_layer_forward_propagation->current_state_activations;

        combinations_dimensions = get_dimensions(cell_states);
        activations_dimensions = get_dimensions(hidden_states);
        derivatives_dimensions = get_dimensions(long_short_term_memory_layer_forward_propagation->current_hidden_states_derivatives);

        calculate_activations_derivatives(cell_states.data(),
                                          combinations_dimensions,
                                          hidden_states.data(),
                                          activations_dimensions,
                                          long_short_term_memory_layer_forward_propagation->current_hidden_states_derivatives.data(),
                                          derivatives_dimensions);

        hidden_states *= long_short_term_memory_layer_forward_propagation->current_output_activations;

        // Activations 2d

        for(Index j = 0; j < neurons_number; j++) outputs(i,j) = hidden_states(j);

        // Forget (activations and activations derivatives)

        copy(long_short_term_memory_layer_forward_propagation->current_forget_activations.data(),
             long_short_term_memory_layer_forward_propagation->current_forget_activations.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->forget_activations.data() + copy_index);

        copy(long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives.data(),
             long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->forget_activations_derivatives.data() + copy_index);

        // Input (activations and activations derivatives)

        copy(long_short_term_memory_layer_forward_propagation->current_input_activations.data(),
             long_short_term_memory_layer_forward_propagation->current_input_activations.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->input_activations.data() + copy_index);

        copy(long_short_term_memory_layer_forward_propagation->current_input_activations_derivatives.data(),
             long_short_term_memory_layer_forward_propagation->current_input_activations_derivatives.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->input_activations_derivatives.data() + copy_index);

        // State (activations and activations derivatives)

        copy(long_short_term_memory_layer_forward_propagation->current_state_activations.data(),
             long_short_term_memory_layer_forward_propagation->current_state_activations.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->state_activations.data() + copy_index);

        copy(long_short_term_memory_layer_forward_propagation->current_state_activations_derivatives.data(),
             long_short_term_memory_layer_forward_propagation->current_state_activations_derivatives.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->state_activations_derivatives.data() + copy_index);

        // Output (activations and activations derivatives)

        copy(long_short_term_memory_layer_forward_propagation->current_output_activations.data(),
             long_short_term_memory_layer_forward_propagation->current_output_activations.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->output_activations.data() + copy_index);

        copy(long_short_term_memory_layer_forward_propagation->current_output_activations_derivatives.data(),
             long_short_term_memory_layer_forward_propagation->current_output_activations_derivatives.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->output_activations_derivatives.data() + copy_index);

        // Cell states (activations)

        copy(cell_states.data(),
             cell_states.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->cell_states_activations.data() + copy_index);

        // Hidden states (activations and activations derivatives)

        copy(hidden_states.data(),
             hidden_states.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->hidden_states_activations.data() + copy_index);

        copy(long_short_term_memory_layer_forward_propagation->current_hidden_states_derivatives.data(),
             long_short_term_memory_layer_forward_propagation->current_hidden_states_derivatives.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->hidden_states_activations_derivatives.data() + copy_index);

        copy_index += neurons_number;
    }
}


void LongShortTermMemoryLayer::forward_propagate(type* inputs_data, const Tensor<Index, 1>& inputs_dimensions, Tensor<type, 1>& parameters, LayerForwardPropagation* forward_propagation)
{

    if(inputs_dimensions.size() != 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void forward_propagate(type*, const Tensor<Index, 1>&, Tensor<type, 1>&, LayerForwardPropagation*) final.\n"
               << "Inputs rank must be equal to 2.\n";

        throw invalid_argument(buffer.str());
    }


    LongShortTermMemoryLayerForwardPropagation* long_short_term_memory_layer_forward_propagation
            = static_cast<LongShortTermMemoryLayerForwardPropagation*>(forward_propagation);

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

    const Index samples_number = inputs_dimensions(0);

    TensorMap<Tensor<type, 2>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1));

    const Tensor<Index, 1> outputs_dimensions = forward_propagation->outputs_dimensions;

    TensorMap<Tensor<type, 2>> outputs(forward_propagation->outputs_data, outputs_dimensions(0), outputs_dimensions(1));

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

    Tensor<Index, 1> current_inputs_dimensions;
    Tensor<Index, 1> combinations_dimensions;
    Tensor<Index, 1> activations_dimensions;
    Tensor<Index, 1> derivatives_dimensions;

    Index copy_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(i%timesteps == 0)
        {
            hidden_states.setZero();
            cell_states.setZero();
        }

        Tensor<type, 1> current_inputs = inputs.chip(i,0);
        current_inputs_dimensions = get_dimensions(current_inputs);

        combinations_dimensions = get_dimensions(forget_combinations);
        activations_dimensions = get_dimensions(forget_activations);
        derivatives_dimensions = get_dimensions(forget_activations_derivatives);

        calculate_combinations(current_inputs.data(), current_inputs_dimensions, forget_weights, forget_recurrent_weights, forget_biases, forget_combinations.data(), combinations_dimensions);
        calculate_recurrent_activations_derivatives(forget_combinations.data(),
                                                    combinations_dimensions,
                                                    forget_activations.data(),
                                                    activations_dimensions,
                                                    forget_activations_derivatives.data(),
                                                    derivatives_dimensions);

        combinations_dimensions = get_dimensions(input_combinations);
        activations_dimensions = get_dimensions(input_activations);
        derivatives_dimensions = get_dimensions(input_activations_derivatives);

        calculate_combinations(current_inputs.data(), current_inputs_dimensions, input_weights, input_recurrent_weights, input_biases, input_combinations.data(), combinations_dimensions);
        calculate_recurrent_activations_derivatives(input_combinations.data(),
                                                    combinations_dimensions,
                                                    input_activations.data(),
                                                    activations_dimensions,
                                                    input_activations_derivatives.data(),
                                                    derivatives_dimensions);

        combinations_dimensions = get_dimensions(state_combinations);
        activations_dimensions = get_dimensions(state_activations);
        derivatives_dimensions = get_dimensions(state_activations_derivatives);

        calculate_combinations(current_inputs.data(), current_inputs_dimensions, state_weights, state_recurrent_weights, state_biases, state_combinations.data(), combinations_dimensions);
        calculate_recurrent_activations_derivatives(state_combinations.data(),
                                                    combinations_dimensions,
                                                    state_activations.data(),
                                                    activations_dimensions,
                                                    state_activations_derivatives.data(),
                                                    derivatives_dimensions);

        combinations_dimensions = get_dimensions(output_combinations);
        activations_dimensions = get_dimensions(output_activations);
        derivatives_dimensions = get_dimensions(output_activations_derivatives);

        calculate_combinations(current_inputs.data(), current_inputs_dimensions, output_weights, output_recurrent_weights, output_biases, output_combinations.data(), combinations_dimensions);
        calculate_recurrent_activations_derivatives(output_combinations.data(),
                                                    combinations_dimensions,
                                                    output_activations.data(),
                                                    activations_dimensions,
                                                    output_activations_derivatives.data(),
                                                    derivatives_dimensions);

        combinations_dimensions = get_dimensions(cell_states);
        activations_dimensions = get_dimensions(hidden_states);
        derivatives_dimensions = get_dimensions(hidden_states_derivatives);

        cell_states = forget_activations * cell_states + input_activations * state_activations;
        calculate_activations_derivatives(cell_states.data(),
                                          combinations_dimensions,
                                          hidden_states.data(),
                                          activations_dimensions,
                                          hidden_states_derivatives.data(),
                                          derivatives_dimensions);

        hidden_states *= output_activations;

        // Activations 2d

        for(Index j = 0; j < neurons_number; j++) outputs(i,j) = hidden_states(j);

        // Forget (activations and activations derivatives)

        copy(long_short_term_memory_layer_forward_propagation->current_forget_activations.data(),
             long_short_term_memory_layer_forward_propagation->current_forget_activations.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->forget_activations.data() + copy_index);

        copy(long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives.data(),
             long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->forget_activations_derivatives.data() + copy_index);

        // Input (activations and activations derivatives)

        copy(long_short_term_memory_layer_forward_propagation->current_input_activations.data(),
             long_short_term_memory_layer_forward_propagation->current_input_activations.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->input_activations.data() + copy_index);

        copy(long_short_term_memory_layer_forward_propagation->current_input_activations_derivatives.data(),
             long_short_term_memory_layer_forward_propagation->current_input_activations_derivatives.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->input_activations_derivatives.data() + copy_index);

        // State (activations and activations derivatives)

        copy(long_short_term_memory_layer_forward_propagation->current_state_activations.data(),
             long_short_term_memory_layer_forward_propagation->current_state_activations.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->state_activations.data() + copy_index);

        copy(long_short_term_memory_layer_forward_propagation->current_state_activations_derivatives.data(),
             long_short_term_memory_layer_forward_propagation->current_state_activations_derivatives.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->state_activations_derivatives.data() + copy_index);

        // Output (activations and activations derivatives)

        copy(long_short_term_memory_layer_forward_propagation->current_output_activations.data(),
             long_short_term_memory_layer_forward_propagation->current_output_activations.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->output_activations.data() + copy_index);

        copy(long_short_term_memory_layer_forward_propagation->current_output_activations_derivatives.data(),
             long_short_term_memory_layer_forward_propagation->current_output_activations_derivatives.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->output_activations_derivatives.data() + copy_index);

        // Cell states (activations)

        copy(cell_states.data(),
             cell_states.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->cell_states_activations.data() + copy_index);

        // Hidden states (activations and activations derivatives)

        copy(hidden_states.data(),
             hidden_states.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->hidden_states_activations.data() + copy_index);

        copy(long_short_term_memory_layer_forward_propagation->current_hidden_states_derivatives.data(),
             long_short_term_memory_layer_forward_propagation->current_hidden_states_derivatives.data() + neurons_number,
             long_short_term_memory_layer_forward_propagation->hidden_states_activations_derivatives.data() + copy_index);

        copy_index += neurons_number;
    }
}


void LongShortTermMemoryLayer::insert_gradient(LayerBackPropagation* back_propagation,
                                               const Index& index,
                                               Tensor<type, 1>& gradient) const
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    LongShortTermMemoryLayerBackPropagation* long_short_term_memory_layer_back_propagation =
            static_cast<LongShortTermMemoryLayerBackPropagation*>(back_propagation);

    // Biases

    copy(long_short_term_memory_layer_back_propagation->forget_biases_derivatives.data(),
         long_short_term_memory_layer_back_propagation->forget_biases_derivatives.data() + neurons_number,
         gradient.data() + index);

    copy(long_short_term_memory_layer_back_propagation->input_biases_derivatives.data(),
         long_short_term_memory_layer_back_propagation->input_biases_derivatives.data() + neurons_number,
         gradient.data() + index + neurons_number);

    copy(long_short_term_memory_layer_back_propagation->state_biases_derivatives.data(),
         long_short_term_memory_layer_back_propagation->state_biases_derivatives.data() + neurons_number,
         gradient.data() + index + 2*neurons_number);

    copy(long_short_term_memory_layer_back_propagation->output_biases_derivatives.data(),
         long_short_term_memory_layer_back_propagation->output_biases_derivatives.data() + neurons_number,
         gradient.data() + index + 3*neurons_number);

    // Weights

    copy(long_short_term_memory_layer_back_propagation->forget_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->forget_weights_derivatives.data() + inputs_number*neurons_number,
         gradient.data() + index + 4*neurons_number);

    copy(long_short_term_memory_layer_back_propagation->input_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->input_weights_derivatives.data() + inputs_number*neurons_number,
         gradient.data() + index + 4*neurons_number + inputs_number*neurons_number);

    copy(long_short_term_memory_layer_back_propagation->state_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->state_weights_derivatives.data() + inputs_number*neurons_number,
         gradient.data() + index + 4*neurons_number + 2*inputs_number*neurons_number);

    copy(long_short_term_memory_layer_back_propagation->output_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->output_weights_derivatives.data() + inputs_number*neurons_number,
         gradient.data() + index + 4*neurons_number + 3*inputs_number*neurons_number);

    // Recurrent weights

    copy(long_short_term_memory_layer_back_propagation->forget_recurrent_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->forget_recurrent_weights_derivatives.data() + neurons_number*neurons_number,
         gradient.data() + index + 4*neurons_number + 4*inputs_number*neurons_number);

    copy(long_short_term_memory_layer_back_propagation->input_recurrent_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->input_recurrent_weights_derivatives.data() + neurons_number*neurons_number,
         gradient.data() + index + 4*neurons_number + 4*inputs_number*neurons_number + neurons_number*neurons_number);

    copy(long_short_term_memory_layer_back_propagation->state_recurrent_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->state_recurrent_weights_derivatives.data() + neurons_number*neurons_number,
         gradient.data() + index + 4*neurons_number + 4*inputs_number*neurons_number + 2*neurons_number*neurons_number);

    copy(long_short_term_memory_layer_back_propagation->output_recurrent_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->output_recurrent_weights_derivatives.data() + neurons_number*neurons_number,
         gradient.data() + index + 4*neurons_number + 4*inputs_number*neurons_number + 3*neurons_number*neurons_number);
}


void LongShortTermMemoryLayer::calculate_error_gradient(type* inputs_data,
                                                        LayerForwardPropagation* forward_propagation,
    LayerBackPropagation* back_propagation) const
{
    const Index batch_samples_number = back_propagation->batch_samples_number;

    LongShortTermMemoryLayerForwardPropagation* long_short_term_memory_layer_forward_propagation =
            static_cast<LongShortTermMemoryLayerForwardPropagation*>(forward_propagation);

    LongShortTermMemoryLayerBackPropagation* long_short_term_memory_layer_back_propagation =
            static_cast<LongShortTermMemoryLayerBackPropagation*>(back_propagation);

//#pragma omp parallel
    {
        // Biases

        const TensorMap<Tensor<type, 2>> inputs(inputs_data, batch_samples_number, get_inputs_number());


        calculate_forget_biases_error_gradient(inputs,
                                               long_short_term_memory_layer_forward_propagation,
                                               long_short_term_memory_layer_back_propagation);

        calculate_input_biases_error_gradient(inputs,
                                              long_short_term_memory_layer_forward_propagation,
                                              long_short_term_memory_layer_back_propagation);

        calculate_state_biases_error_gradient(inputs,
                                              long_short_term_memory_layer_forward_propagation,
                                              long_short_term_memory_layer_back_propagation);

        calculate_output_biases_error_gradient(inputs,
                                               long_short_term_memory_layer_forward_propagation,
                                               long_short_term_memory_layer_back_propagation);

        // Weights

        calculate_forget_weights_error_gradient(inputs,
                                                long_short_term_memory_layer_forward_propagation,
                                                long_short_term_memory_layer_back_propagation);

        calculate_input_weights_error_gradient(inputs,
                                               long_short_term_memory_layer_forward_propagation,
                                               long_short_term_memory_layer_back_propagation);

        calculate_state_weights_error_gradient(inputs,
                                               long_short_term_memory_layer_forward_propagation,
                                               long_short_term_memory_layer_back_propagation);

        calculate_output_weights_error_gradient(inputs,
                                                long_short_term_memory_layer_forward_propagation,
                                                long_short_term_memory_layer_back_propagation);

        // Recurrent weights

        calculate_forget_recurrent_weights_error_gradient(inputs,
                                                          long_short_term_memory_layer_forward_propagation,
                                                          long_short_term_memory_layer_back_propagation);

        calculate_input_recurrent_weights_error_gradient(inputs,
                                                         long_short_term_memory_layer_forward_propagation,
                                                         long_short_term_memory_layer_back_propagation);

        calculate_state_recurrent_weights_error_gradient(inputs,
                                                         long_short_term_memory_layer_forward_propagation,
                                                         long_short_term_memory_layer_back_propagation);

        calculate_output_recurrent_weights_error_gradient(inputs,
                                                          long_short_term_memory_layer_forward_propagation,
                                                          long_short_term_memory_layer_back_propagation);
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

    const TensorMap<Tensor<type, 2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));

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

    Index copy_index = 0;

    back_propagation->forget_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_inputs = inputs.chip(sample, 0); // memcpy?
        const Tensor<type, 1> current_layer_deltas = deltas.chip(sample,0); // memcpy?

        // Forget activations and derivatives

        copy(forward_propagation->forget_activations.data() + copy_index,
             forward_propagation->forget_activations.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations.data());

        copy(forward_propagation->forget_activations_derivatives.data() + copy_index,
             forward_propagation->forget_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations_derivatives.data());

        // Input activations and derivatives

        copy(forward_propagation->input_activations.data() + copy_index,
             forward_propagation->input_activations.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations.data());

        copy(forward_propagation->input_activations_derivatives.data() + copy_index,
             forward_propagation->input_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations_derivatives.data());

        // State activations and derivatives

        copy(forward_propagation->state_activations.data() + copy_index,
             forward_propagation->state_activations.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations.data());

        copy(forward_propagation->state_activations_derivatives.data() + copy_index,
             forward_propagation->state_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations_derivatives.data());

        // Output activations and derivatives

        copy(forward_propagation->output_activations.data() + copy_index,
             forward_propagation->output_activations.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations.data());

        copy(forward_propagation->output_activations_derivatives.data() + copy_index,
             forward_propagation->output_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations_derivatives.data());

        // Cell states and hidden states

        copy(forward_propagation->cell_states_activations.data() + copy_index,
             forward_propagation->cell_states_activations.data() + copy_index + neurons_number,
             forward_propagation->current_cell_state_activations.data());

        copy(forward_propagation->hidden_states_activations_derivatives.data() + copy_index,
             forward_propagation->hidden_states_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_hidden_states_derivatives.data());

        if(sample%timesteps == 0)
        {
            forward_propagation->previous_cell_state_activations.setZero();

            forget_combinations_weights_derivatives.setZero();
            input_combinations_weights_derivatives.setZero();
            output_combinations_weights_derivatives.setZero();
            state_combinations_weights_derivatives.setZero();

            cell_state_weights_derivatives.setZero();
        }
        else
        {
            copy(forward_propagation->cell_states_activations.data() + (copy_index-neurons_number),
                 forward_propagation->cell_states_activations.data() + (copy_index-neurons_number) + neurons_number,
                 forward_propagation->previous_cell_state_activations.data());

            forget_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(forget_recurrent_weights, A_B);

            input_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(input_recurrent_weights, A_B);
            multiply_rows(input_combinations_weights_derivatives, forward_propagation->current_input_activations_derivatives);

            state_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(state_recurrent_weights, A_B);
            multiply_rows(state_combinations_weights_derivatives, forward_propagation->current_state_activations_derivatives);

            output_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(output_recurrent_weights, A_B);
            multiply_rows(output_combinations_weights_derivatives, forward_propagation->current_output_activations_derivatives);
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

        multiply_rows(cell_state_weights_derivatives,
                      forward_propagation->current_forget_activations);

        multiply_rows(input_combinations_weights_derivatives,
                      forward_propagation->current_state_activations);

        cell_state_weights_derivatives += input_combinations_weights_derivatives;

        multiply_rows(state_combinations_weights_derivatives,
                      forward_propagation->current_input_activations);

        cell_state_weights_derivatives += state_combinations_weights_derivatives;

        multiply_rows(forget_combinations_weights_derivatives,
                      forward_propagation->current_forget_activations_derivatives*forward_propagation->previous_cell_state_activations);

        cell_state_weights_derivatives += forget_combinations_weights_derivatives;

        copy(cell_state_weights_derivatives.data(),
             cell_state_weights_derivatives.data() + cell_state_weights_derivatives.size(),
             hidden_states_weights_derivatives.data());

        multiply_rows(hidden_states_weights_derivatives,
                      forward_propagation->current_output_activations*forward_propagation->current_hidden_states_derivatives);

        multiply_rows(output_combinations_weights_derivatives,
                      calculate_activations(forward_propagation->current_cell_state_activations));
        hidden_states_weights_derivatives += output_combinations_weights_derivatives;

        back_propagation->forget_weights_derivatives += hidden_states_weights_derivatives.contract(current_layer_deltas, A_B);

        copy_index += neurons_number;
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

    const TensorMap<Tensor<type, 2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));

    input_combinations_weights_derivatives.setZero();
    forget_combinations_weights_derivatives.setZero();
    state_combinations_weights_derivatives.setZero();
    output_combinations_weights_derivatives.setZero();
    hidden_states_weights_derivatives.setZero();
    cell_state_weights_derivatives.setZero();

    Index column_index = 0;
    Index input_index = 0;

    Index copy_index = 0;

    back_propagation->input_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        forward_propagation->current_inputs = inputs.chip(sample, 0); // memcpy?

        back_propagation->current_layer_deltas = deltas.chip(sample,0); // memcpy?

        copy(forward_propagation->forget_activations.data() + copy_index,
             forward_propagation->forget_activations.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations.data());

        copy(forward_propagation->forget_activations_derivatives.data() + copy_index,
             forward_propagation->forget_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations_derivatives.data());

        copy(forward_propagation->input_activations.data() + copy_index,
             forward_propagation->input_activations.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations.data());

        copy(forward_propagation->input_activations_derivatives.data() + copy_index,
             forward_propagation->input_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations_derivatives.data());

        copy(forward_propagation->state_activations.data() + copy_index,
             forward_propagation->state_activations.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations.data());

        copy(forward_propagation->state_activations_derivatives.data() + copy_index,
             forward_propagation->state_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations_derivatives.data());

        copy(forward_propagation->output_activations.data() + copy_index,
             forward_propagation->output_activations.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations.data());

        copy(forward_propagation->output_activations_derivatives.data() + copy_index,
             forward_propagation->output_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations_derivatives.data());

        copy(forward_propagation->cell_states_activations.data() + copy_index,
             forward_propagation->cell_states_activations.data() + copy_index + neurons_number,
             forward_propagation->current_cell_state_activations.data());

        copy(forward_propagation->hidden_states_activations_derivatives.data() + copy_index,
             forward_propagation->hidden_states_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_hidden_states_derivatives.data());

        if(sample%timesteps == 0)
        {
            forward_propagation->previous_cell_state_activations.setZero();

            forget_combinations_weights_derivatives.setZero();
            input_combinations_weights_derivatives.setZero();
            output_combinations_weights_derivatives.setZero();
            state_combinations_weights_derivatives.setZero();

            cell_state_weights_derivatives.setZero();
            hidden_states_weights_derivatives.setZero();
        }
        else
        {
            copy(forward_propagation->cell_states_activations.data() + (copy_index-neurons_number),
                 forward_propagation->cell_states_activations.data() + (copy_index-neurons_number) + neurons_number,
                 forward_propagation->previous_cell_state_activations.data());

            forget_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(forget_recurrent_weights, A_B);

            multiply_rows(forget_combinations_weights_derivatives,
                          forward_propagation->current_forget_activations_derivatives);

            input_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(input_recurrent_weights, A_B);

            state_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(state_recurrent_weights, A_B);

            multiply_rows(state_combinations_weights_derivatives,
                          forward_propagation->current_state_activations_derivatives);

            output_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(output_recurrent_weights, A_B);

            multiply_rows(output_combinations_weights_derivatives,
                          forward_propagation->current_output_activations_derivatives);
        }

        column_index = 0;
        input_index = 0;

        for(Index i = 0; i < parameters_number; i++)
        {
            input_combinations_weights_derivatives(i, column_index) += forward_propagation->current_inputs[input_index];

            input_index++;

            if(input_index == inputs_number)
            {
                input_index = 0;
                column_index++;
            }
        }

        multiply_rows(cell_state_weights_derivatives,
                      forward_propagation->current_forget_activations);

        multiply_rows(forget_combinations_weights_derivatives,
                      forward_propagation->previous_cell_state_activations);

        cell_state_weights_derivatives += forget_combinations_weights_derivatives;

        multiply_rows(state_combinations_weights_derivatives,
                      forward_propagation->current_input_activations);

        cell_state_weights_derivatives += state_combinations_weights_derivatives;

        multiply_rows(input_combinations_weights_derivatives,
                      forward_propagation->current_input_activations_derivatives*forward_propagation->current_state_activations);

        cell_state_weights_derivatives += input_combinations_weights_derivatives;

        hidden_states_weights_derivatives = cell_state_weights_derivatives;

        multiply_rows(hidden_states_weights_derivatives,
                      forward_propagation->current_output_activations*forward_propagation->current_hidden_states_derivatives);

        multiply_rows(output_combinations_weights_derivatives,
                      calculate_activations(forward_propagation->current_cell_state_activations));

        hidden_states_weights_derivatives += output_combinations_weights_derivatives;

        back_propagation->input_weights_derivatives
                += hidden_states_weights_derivatives.contract(back_propagation->current_layer_deltas, A_B);

        copy_index += neurons_number;
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

    TensorMap<Tensor<type, 2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));

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

    Index copy_index = 0;

    back_propagation->state_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        forward_propagation->current_inputs = inputs.chip(sample, 0); // memcpy?

        back_propagation->current_layer_deltas = deltas.chip(sample,0); // memcpy?

        copy(forward_propagation->forget_activations.data() + copy_index,
             forward_propagation->forget_activations.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations.data());

        copy(forward_propagation->forget_activations_derivatives.data() + copy_index,
             forward_propagation->forget_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations_derivatives.data());

        copy(forward_propagation->input_activations.data() + copy_index,
             forward_propagation->input_activations.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations.data());

        copy(forward_propagation->input_activations_derivatives.data() + copy_index,
             forward_propagation->input_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations_derivatives.data());

        copy(forward_propagation->state_activations.data() + copy_index,
             forward_propagation->state_activations.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations.data());

        copy(forward_propagation->state_activations_derivatives.data() + copy_index,
             forward_propagation->state_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations_derivatives.data());

        copy(forward_propagation->output_activations.data() + copy_index,
             forward_propagation->output_activations.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations.data());

        copy(forward_propagation->output_activations_derivatives.data() + copy_index,
             forward_propagation->output_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations_derivatives.data());

        copy(forward_propagation->cell_states_activations.data() + copy_index,
             forward_propagation->cell_states_activations.data() + copy_index + neurons_number,
             forward_propagation->current_cell_state_activations.data());

        copy(forward_propagation->hidden_states_activations_derivatives.data() + copy_index,
             forward_propagation->hidden_states_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_hidden_states_derivatives.data());

        if(sample%timesteps == 0)
        {
            forward_propagation->previous_cell_state_activations.setZero();

            forget_combinations_weights_derivatives.setZero();
            input_combinations_weights_derivatives.setZero();
            output_combinations_weights_derivatives.setZero();
            state_combinations_weights_derivatives.setZero();

            cell_state_weights_derivatives.setZero();
        }
        else
        {
            copy(forward_propagation->cell_states_activations.data() + (copy_index - neurons_number),
            forward_propagation->cell_states_activations.data() + (copy_index - neurons_number) + neurons_number,
            forward_propagation->previous_cell_state_activations.data());

            forget_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(forget_recurrent_weights, A_B);
            multiply_rows(forget_combinations_weights_derivatives, forward_propagation->current_forget_activations_derivatives);

            input_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(input_recurrent_weights, A_B);
            multiply_rows(input_combinations_weights_derivatives, forward_propagation->current_input_activations_derivatives);

            state_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(state_recurrent_weights, A_B);

            output_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(output_recurrent_weights, A_B);
            multiply_rows(output_combinations_weights_derivatives, forward_propagation->current_output_activations_derivatives);
        }

        column_index = 0;
        input_index = 0;

        for(Index i = 0; i < parameters_number; i++)
        {
            state_combinations_weights_derivatives(i, column_index) += forward_propagation->current_inputs[input_index];

            input_index++;

            if(input_index == inputs_number)
            {
                input_index = 0;
                column_index++;
            }
        }

        multiply_rows(cell_state_weights_derivatives, forward_propagation->current_forget_activations);
        multiply_rows(forget_combinations_weights_derivatives, forward_propagation->previous_cell_state_activations);
        cell_state_weights_derivatives += forget_combinations_weights_derivatives;
        multiply_rows(input_combinations_weights_derivatives, forward_propagation->current_state_activations);
        cell_state_weights_derivatives += input_combinations_weights_derivatives;
        multiply_rows(state_combinations_weights_derivatives, (forward_propagation->current_state_activations_derivatives*forward_propagation->current_input_activations));
        cell_state_weights_derivatives += state_combinations_weights_derivatives;

        hidden_states_weights_derivatives = cell_state_weights_derivatives;
        multiply_rows(hidden_states_weights_derivatives, forward_propagation->current_output_activations*forward_propagation->current_hidden_states_derivatives);
        multiply_rows(output_combinations_weights_derivatives, calculate_activations(forward_propagation->current_cell_state_activations));
        hidden_states_weights_derivatives += output_combinations_weights_derivatives;

        back_propagation->state_weights_derivatives += hidden_states_weights_derivatives.contract(back_propagation->current_layer_deltas, A_B);

        copy_index += neurons_number;
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

    const TensorMap<Tensor<type, 2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));

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

    Index copy_index = 0;

    back_propagation->output_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        forward_propagation->current_inputs = inputs.chip(sample, 0); // memcpy?

        back_propagation->current_layer_deltas = deltas.chip(sample,0); // memcpy?

        copy(forward_propagation->forget_activations.data() + copy_index,
             forward_propagation->forget_activations.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations.data());

        copy(forward_propagation->forget_activations_derivatives.data() + copy_index,
             forward_propagation->forget_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations_derivatives.data());

        copy(forward_propagation->input_activations.data() + copy_index,
             forward_propagation->input_activations.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations.data());

        copy(forward_propagation->input_activations_derivatives.data() + copy_index,
             forward_propagation->input_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations_derivatives.data());

        copy(forward_propagation->state_activations.data() + copy_index,
             forward_propagation->state_activations.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations.data());

        copy(forward_propagation->state_activations_derivatives.data() + copy_index,
             forward_propagation->state_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations_derivatives.data());

        copy(forward_propagation->output_activations.data() + copy_index,
             forward_propagation->output_activations.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations.data());

        copy(forward_propagation->output_activations_derivatives.data() + copy_index,
             forward_propagation->output_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations_derivatives.data());

        copy(forward_propagation->cell_states_activations.data() + copy_index,
             forward_propagation->cell_states_activations.data() + copy_index + neurons_number,
             forward_propagation->current_cell_state_activations.data());

        copy(forward_propagation->hidden_states_activations_derivatives.data() + copy_index,
             forward_propagation->hidden_states_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_hidden_states_derivatives.data());

        if(sample%timesteps == 0)
        {
            forward_propagation->previous_cell_state_activations.setZero();

            forget_combinations_weights_derivatives.setZero();
            input_combinations_weights_derivatives.setZero();
            output_combinations_weights_derivatives.setZero();
            state_combinations_weights_derivatives.setZero();

            cell_state_weights_derivatives.setZero();
        }
        else
        {
            copy(forward_propagation->cell_states_activations.data() + (copy_index-neurons_number),
                 forward_propagation->cell_states_activations.data() + (copy_index-neurons_number) + neurons_number,
                 forward_propagation->previous_cell_state_activations.data());

            forget_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(forget_recurrent_weights, A_B);
            multiply_rows(forget_combinations_weights_derivatives, forward_propagation->current_forget_activations_derivatives);
            input_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(input_recurrent_weights, A_B);
            multiply_rows(input_combinations_weights_derivatives, forward_propagation->current_input_activations_derivatives);
            state_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(state_recurrent_weights, A_B);
            multiply_rows(state_combinations_weights_derivatives, forward_propagation->current_state_activations_derivatives);
            output_combinations_weights_derivatives = hidden_states_weights_derivatives.contract(output_recurrent_weights, A_B);
        }

        column_index = 0;
        input_index = 0;

        for(Index i = 0; i < parameters_number; i++)
        {
            output_combinations_weights_derivatives(i, column_index) += forward_propagation->current_inputs[input_index];

            input_index++;

            if(input_index == inputs_number)
            {
                input_index = 0;
                column_index++;
            }
        }

        multiply_rows(cell_state_weights_derivatives, forward_propagation->current_forget_activations);
        multiply_rows(forget_combinations_weights_derivatives, forward_propagation->previous_cell_state_activations);
        cell_state_weights_derivatives += forget_combinations_weights_derivatives;
        multiply_rows(state_combinations_weights_derivatives, forward_propagation->current_input_activations);
        cell_state_weights_derivatives += state_combinations_weights_derivatives;
        multiply_rows(input_combinations_weights_derivatives, forward_propagation->current_state_activations);
        cell_state_weights_derivatives += input_combinations_weights_derivatives;

         hidden_states_weights_derivatives = cell_state_weights_derivatives;
         multiply_rows(hidden_states_weights_derivatives, forward_propagation->current_output_activations*forward_propagation->current_hidden_states_derivatives);
         multiply_rows(output_combinations_weights_derivatives, forward_propagation->current_output_activations_derivatives*calculate_activations(forward_propagation->current_cell_state_activations));
        hidden_states_weights_derivatives += output_combinations_weights_derivatives;

        back_propagation->output_weights_derivatives += hidden_states_weights_derivatives.contract(back_propagation->current_layer_deltas, A_B);

        copy_index += neurons_number;
    }
}


void LongShortTermMemoryLayer::calculate_forget_recurrent_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                 LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                                 LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number*neurons_number;

    const TensorMap<Tensor<type, 2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));

    Tensor<type, 1> forget_recurrent_weights_error_gradient(parameters_number);
    forget_recurrent_weights_error_gradient.setZero();

    Tensor<type, 2> input_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> forget_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> state_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> output_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 2> hidden_states_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> cell_state_recurrent_weights_derivatives(parameters_number, neurons_number);

    Index column_index = 0;
    Index activation_index = 0;

    Index copy_index = 0;

    back_propagation->forget_recurrent_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        back_propagation->current_layer_deltas = deltas.chip(sample, 0);

        copy(forward_propagation->forget_activations.data() + copy_index,
             forward_propagation->forget_activations.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations.data());

        copy(forward_propagation->forget_activations_derivatives.data() + copy_index,
             forward_propagation->forget_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations_derivatives.data());

        copy(forward_propagation->input_activations.data() + copy_index,
             forward_propagation->input_activations.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations.data());

        copy(forward_propagation->input_activations_derivatives.data() + copy_index,
             forward_propagation->input_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations_derivatives.data());

        copy(forward_propagation->state_activations.data() + copy_index,
             forward_propagation->state_activations.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations.data());

        copy(forward_propagation->state_activations_derivatives.data() + copy_index,
             forward_propagation->state_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations_derivatives.data());

        copy(forward_propagation->output_activations.data() + copy_index,
             forward_propagation->output_activations.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations.data());

        copy(forward_propagation->output_activations_derivatives.data() + copy_index,
             forward_propagation->output_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations_derivatives.data());

        copy(forward_propagation->cell_states_activations.data() + copy_index,
             forward_propagation->cell_states_activations.data() + copy_index + neurons_number,
             forward_propagation->current_cell_state_activations.data());

        copy(forward_propagation->hidden_states_activations_derivatives.data() + copy_index,
             forward_propagation->hidden_states_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_hidden_states_derivatives.data());

        if(sample%timesteps == 0)
        {
            cell_state_recurrent_weights_derivatives.setZero();
            hidden_states_recurrent_weights_derivatives.setZero();
        }
        else
        {
            copy(forward_propagation->hidden_states_activations.data() + (copy_index-neurons_number),
                 forward_propagation->hidden_states_activations.data() + (copy_index-neurons_number) + neurons_number,
                 forward_propagation->previous_hidden_state_activations.data());

            copy(forward_propagation->cell_states_activations.data() + (copy_index-neurons_number),
                 forward_propagation->cell_states_activations.data() + (copy_index-neurons_number) + neurons_number,
                 forward_propagation->previous_cell_state_activations.data());

            forget_combinations_recurrent_weights_derivatives = hidden_states_recurrent_weights_derivatives.contract(forget_recurrent_weights, A_B);
            input_combinations_recurrent_weights_derivatives = hidden_states_recurrent_weights_derivatives.contract(input_recurrent_weights, A_B);
            multiply_rows(input_combinations_recurrent_weights_derivatives, forward_propagation->current_input_activations_derivatives);
            state_combinations_recurrent_weights_derivatives = hidden_states_recurrent_weights_derivatives.contract(state_recurrent_weights, A_B);
            multiply_rows(state_combinations_recurrent_weights_derivatives, forward_propagation->current_state_activations_derivatives);
            output_combinations_recurrent_weights_derivatives = hidden_states_recurrent_weights_derivatives.contract(output_recurrent_weights, A_B);
            multiply_rows(output_combinations_recurrent_weights_derivatives, forward_propagation->current_output_activations_derivatives);

            column_index = 0;
            activation_index = 0;

            for(Index i = 0; i < parameters_number; i++)
            {
                forget_combinations_recurrent_weights_derivatives(i, column_index) += forward_propagation->previous_hidden_state_activations[activation_index];

                activation_index++;

                if(activation_index == neurons_number)
                {
                    activation_index = 0;
                    column_index++;
                }
            }

            multiply_rows(cell_state_recurrent_weights_derivatives, forward_propagation->current_forget_activations);
            multiply_rows(input_combinations_recurrent_weights_derivatives, forward_propagation->current_state_activations);
            cell_state_recurrent_weights_derivatives += input_combinations_recurrent_weights_derivatives;
            multiply_rows(state_combinations_recurrent_weights_derivatives, forward_propagation->current_input_activations);
            cell_state_recurrent_weights_derivatives += state_combinations_recurrent_weights_derivatives;
            multiply_rows(forget_combinations_recurrent_weights_derivatives, (forward_propagation->current_forget_activations_derivatives*forward_propagation->previous_cell_state_activations));
            cell_state_recurrent_weights_derivatives += forget_combinations_recurrent_weights_derivatives;

            hidden_states_recurrent_weights_derivatives = cell_state_recurrent_weights_derivatives;
            multiply_rows(hidden_states_recurrent_weights_derivatives, forward_propagation->current_output_activations*forward_propagation->current_hidden_states_derivatives);
            multiply_rows(output_combinations_recurrent_weights_derivatives, calculate_activations(forward_propagation->current_cell_state_activations));
            hidden_states_recurrent_weights_derivatives += output_combinations_recurrent_weights_derivatives;
        }

        back_propagation->forget_recurrent_weights_derivatives += hidden_states_recurrent_weights_derivatives.contract(back_propagation->current_layer_deltas, A_B);

        copy_index += neurons_number;
    }
}


void LongShortTermMemoryLayer::calculate_input_recurrent_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                                LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number*neurons_number;

    const TensorMap<Tensor<type, 2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));

    Tensor<type, 1> forget_recurrent_weights_error_gradient(parameters_number);
    forget_recurrent_weights_error_gradient.setZero();

    Tensor<type, 2> input_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> forget_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> state_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> output_combinations_recurrent_weights_derivatives(parameters_number, neurons_number);

    Tensor<type, 2> hidden_states_recurrent_weights_derivatives(parameters_number, neurons_number);
    Tensor<type, 2> cell_state_recurrent_weights_derivatives(parameters_number, neurons_number);

    Index column_index = 0;
    Index activation_index = 0;

    Index copy_index = 0;

    back_propagation->input_recurrent_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        back_propagation->current_layer_deltas = deltas.chip(sample, 0);

        copy(forward_propagation->forget_activations.data() + copy_index,
             forward_propagation->forget_activations.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations.data());

        copy(forward_propagation->forget_activations_derivatives.data() + copy_index,
             forward_propagation->forget_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations_derivatives.data());

        copy(forward_propagation->input_activations.data() + copy_index,
             forward_propagation->input_activations.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations.data());

        copy(forward_propagation->input_activations_derivatives.data() + copy_index,
             forward_propagation->input_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations_derivatives.data());

        copy(forward_propagation->state_activations.data() + copy_index,
             forward_propagation->state_activations.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations.data());

        copy(forward_propagation->state_activations_derivatives.data() + copy_index,
             forward_propagation->state_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations_derivatives.data());

        copy(forward_propagation->output_activations.data() + copy_index,
             forward_propagation->output_activations.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations.data());

        copy(forward_propagation->output_activations_derivatives.data() + copy_index,
             forward_propagation->output_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations_derivatives.data());

        copy(forward_propagation->cell_states_activations.data() + copy_index,
             forward_propagation->cell_states_activations.data() + copy_index + neurons_number,
             forward_propagation->current_cell_state_activations.data());

        copy(forward_propagation->hidden_states_activations_derivatives.data() + copy_index,
             forward_propagation->hidden_states_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_hidden_states_derivatives.data());

        if(sample%timesteps == 0)
        {
            cell_state_recurrent_weights_derivatives.setZero();
            hidden_states_recurrent_weights_derivatives.setZero();
        }
        else
        {
            copy(forward_propagation->hidden_states_activations.data() + (copy_index-neurons_number),
                 forward_propagation->hidden_states_activations.data() + (copy_index-neurons_number) + neurons_number,
                 forward_propagation->previous_hidden_state_activations.data());

            copy(forward_propagation->cell_states_activations.data() + (copy_index-neurons_number),
                 forward_propagation->cell_states_activations.data() + (copy_index-neurons_number) + neurons_number,
                 forward_propagation->previous_cell_state_activations.data());

            forget_combinations_recurrent_weights_derivatives = hidden_states_recurrent_weights_derivatives.contract(forget_recurrent_weights, A_B);
            multiply_rows(forget_combinations_recurrent_weights_derivatives, forward_propagation->current_forget_activations_derivatives);
            input_combinations_recurrent_weights_derivatives = hidden_states_recurrent_weights_derivatives.contract(input_recurrent_weights, A_B);
            state_combinations_recurrent_weights_derivatives = hidden_states_recurrent_weights_derivatives.contract(state_recurrent_weights, A_B);
            multiply_rows(state_combinations_recurrent_weights_derivatives, forward_propagation->current_state_activations_derivatives);
            output_combinations_recurrent_weights_derivatives = hidden_states_recurrent_weights_derivatives.contract(output_recurrent_weights, A_B);
            multiply_rows(output_combinations_recurrent_weights_derivatives, forward_propagation->current_output_activations_derivatives);

            column_index = 0;
            activation_index = 0;

            for(Index i = 0; i < parameters_number; i++)
            {
                input_combinations_recurrent_weights_derivatives(i, column_index) += forward_propagation->previous_hidden_state_activations[activation_index];

                activation_index++;

                if(activation_index == neurons_number)
                {
                    activation_index = 0;
                    column_index++;
                }
            }

            multiply_rows(cell_state_recurrent_weights_derivatives, forward_propagation->current_forget_activations);
            multiply_rows(input_combinations_recurrent_weights_derivatives, forward_propagation->current_input_activations_derivatives*forward_propagation->current_state_activations);
            cell_state_recurrent_weights_derivatives += input_combinations_recurrent_weights_derivatives;
            multiply_rows(state_combinations_recurrent_weights_derivatives, forward_propagation->current_input_activations);
            cell_state_recurrent_weights_derivatives += state_combinations_recurrent_weights_derivatives;
            multiply_rows(forget_combinations_recurrent_weights_derivatives, forward_propagation->previous_cell_state_activations);
            cell_state_recurrent_weights_derivatives += forget_combinations_recurrent_weights_derivatives;

            hidden_states_recurrent_weights_derivatives = cell_state_recurrent_weights_derivatives;
            multiply_rows(hidden_states_recurrent_weights_derivatives, forward_propagation->current_output_activations*forward_propagation->current_hidden_states_derivatives);
            multiply_rows(output_combinations_recurrent_weights_derivatives, calculate_activations(forward_propagation->current_cell_state_activations));
            hidden_states_recurrent_weights_derivatives += output_combinations_recurrent_weights_derivatives;
        }

        back_propagation->input_recurrent_weights_derivatives += hidden_states_recurrent_weights_derivatives.contract(back_propagation->current_layer_deltas, A_B);

        copy_index += neurons_number;
    }
}


void LongShortTermMemoryLayer::calculate_state_recurrent_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                                LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number*neurons_number;

    const TensorMap<Tensor<type, 2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));

    Tensor<type, 1> forget_recurrent_weights_error_gradient(parameters_number);
    forget_recurrent_weights_error_gradient.setZero();

    Index column_index = 0;
    Index activation_index = 0;

    Index copy_index = 0;

    back_propagation->state_recurrent_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        back_propagation->current_layer_deltas = deltas.chip(sample, 0);

        copy(forward_propagation->forget_activations.data() + copy_index,
             forward_propagation->forget_activations.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations.data());

        copy(forward_propagation->forget_activations_derivatives.data() + copy_index,
             forward_propagation->forget_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations_derivatives.data());

        copy(forward_propagation->input_activations.data() + copy_index,
             forward_propagation->input_activations.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations.data());

        copy(forward_propagation->input_activations_derivatives.data() + copy_index,
             forward_propagation->input_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations_derivatives.data());

        copy(forward_propagation->state_activations.data() + copy_index,
             forward_propagation->state_activations.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations.data());

        copy(forward_propagation->state_activations_derivatives.data() + copy_index,
             forward_propagation->state_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations_derivatives.data());

        copy(forward_propagation->output_activations.data() + copy_index,
             forward_propagation->output_activations.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations.data());

        copy(forward_propagation->output_activations_derivatives.data() + copy_index,
             forward_propagation->output_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations_derivatives.data());

        copy(forward_propagation->cell_states_activations.data() + copy_index,
             forward_propagation->cell_states_activations.data() + copy_index + neurons_number,
             forward_propagation->current_cell_state_activations.data());

        copy(forward_propagation->hidden_states_activations_derivatives.data() + copy_index,
             forward_propagation->hidden_states_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_hidden_states_derivatives.data());

        if(sample%timesteps == 0)
        {
            back_propagation->cell_state_recurrent_weights_derivatives.setZero();
            back_propagation->hidden_states_recurrent_weights_derivatives.setZero();
        }
        else
        {
            copy(forward_propagation->hidden_states_activations.data() + (copy_index-neurons_number),
                 forward_propagation->hidden_states_activations.data() + (copy_index-neurons_number) + neurons_number,
                 forward_propagation->previous_hidden_state_activations.data());

            copy(forward_propagation->cell_states_activations.data() + (copy_index-neurons_number),
                 forward_propagation->cell_states_activations.data() + (copy_index-neurons_number) + neurons_number,
                 forward_propagation->previous_cell_state_activations.data());

            back_propagation->forget_combinations_recurrent_weights_derivatives = back_propagation->hidden_states_recurrent_weights_derivatives.contract(forget_recurrent_weights, A_B);
            multiply_rows(back_propagation->forget_combinations_recurrent_weights_derivatives, forward_propagation->current_forget_activations_derivatives);
            back_propagation->input_combinations_recurrent_weights_derivatives = back_propagation->hidden_states_recurrent_weights_derivatives.contract(input_recurrent_weights, A_B);
            multiply_rows(back_propagation->input_combinations_recurrent_weights_derivatives, forward_propagation->current_input_activations_derivatives);
            back_propagation->state_combinations_recurrent_weights_derivatives = back_propagation->hidden_states_recurrent_weights_derivatives.contract(state_recurrent_weights, A_B);
            back_propagation->state_combinations_recurrent_weights_derivatives = back_propagation->hidden_states_recurrent_weights_derivatives.contract(output_recurrent_weights, A_B);
            multiply_rows(back_propagation->state_combinations_recurrent_weights_derivatives, forward_propagation->current_output_activations_derivatives);

            column_index = 0;
            activation_index = 0;

            for(Index i = 0; i < parameters_number; i++)
            {
                back_propagation->state_combinations_recurrent_weights_derivatives(i, column_index) += forward_propagation->previous_hidden_state_activations[activation_index];

                activation_index++;

                if(activation_index == neurons_number)
                {
                    activation_index = 0;
                    column_index++;
                }
            }

            multiply_rows(back_propagation->cell_state_recurrent_weights_derivatives, forward_propagation->current_forget_activations);
            multiply_rows(back_propagation->input_combinations_recurrent_weights_derivatives, forward_propagation->current_state_activations);
            back_propagation->cell_state_recurrent_weights_derivatives += back_propagation->input_combinations_recurrent_weights_derivatives;
            multiply_rows(back_propagation->state_combinations_recurrent_weights_derivatives, forward_propagation->current_state_activations_derivatives*forward_propagation->current_input_activations);
            back_propagation->cell_state_recurrent_weights_derivatives += back_propagation->state_combinations_recurrent_weights_derivatives;
            multiply_rows(back_propagation->forget_combinations_recurrent_weights_derivatives, forward_propagation->previous_cell_state_activations);
            back_propagation->cell_state_recurrent_weights_derivatives += back_propagation->forget_combinations_recurrent_weights_derivatives;

            back_propagation->hidden_states_recurrent_weights_derivatives = back_propagation->cell_state_recurrent_weights_derivatives;
            multiply_rows(back_propagation->hidden_states_recurrent_weights_derivatives, forward_propagation->current_output_activations*forward_propagation->current_hidden_states_derivatives);
            multiply_rows(back_propagation->state_combinations_recurrent_weights_derivatives, calculate_activations(forward_propagation->current_cell_state_activations));
            back_propagation->hidden_states_recurrent_weights_derivatives += back_propagation->state_combinations_recurrent_weights_derivatives;
        }

        back_propagation->state_recurrent_weights_derivatives += back_propagation->hidden_states_recurrent_weights_derivatives.contract(back_propagation->current_layer_deltas, A_B);

        copy_index += neurons_number;
    }
}


void LongShortTermMemoryLayer::calculate_output_recurrent_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                 LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                                 LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number*neurons_number;

    const TensorMap<Tensor<type, 2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));


    Index column_index = 0;
    Index activation_index = 0;

    Index copy_index = 0;

    back_propagation->output_recurrent_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        back_propagation->current_layer_deltas = deltas.chip(sample, 0);

        copy(forward_propagation->forget_activations.data() + copy_index,
             forward_propagation->forget_activations.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations.data());

        copy(forward_propagation->forget_activations_derivatives.data() + copy_index,
             forward_propagation->forget_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations_derivatives.data());

        copy(forward_propagation->input_activations.data() + copy_index,
             forward_propagation->input_activations.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations.data());

        copy(forward_propagation->input_activations_derivatives.data() + copy_index,
             forward_propagation->input_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations_derivatives.data());

        copy(forward_propagation->state_activations.data() + copy_index,
             forward_propagation->state_activations.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations.data());

        copy(forward_propagation->state_activations_derivatives.data() + copy_index,
             forward_propagation->state_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations_derivatives.data());

        copy(forward_propagation->output_activations.data() + copy_index,
             forward_propagation->output_activations.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations.data());

        copy(forward_propagation->output_activations_derivatives.data() + copy_index,
             forward_propagation->output_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations_derivatives.data());

        copy(forward_propagation->cell_states_activations.data() + copy_index,
             forward_propagation->cell_states_activations.data() + copy_index + neurons_number,
             forward_propagation->current_cell_state_activations.data());

        copy(forward_propagation->hidden_states_activations_derivatives.data() + copy_index,
             forward_propagation->hidden_states_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_hidden_states_derivatives.data());

        if(sample%timesteps == 0)
        {
            back_propagation->cell_state_recurrent_weights_derivatives.setZero();
            back_propagation->hidden_states_recurrent_weights_derivatives.setZero();
        }
        else
        {
            copy(forward_propagation->hidden_states_activations.data() + (copy_index-neurons_number),
                 forward_propagation->hidden_states_activations.data() + (copy_index-neurons_number) + neurons_number,
                 forward_propagation->previous_hidden_state_activations.data());

            copy(forward_propagation->cell_states_activations.data() + (copy_index-neurons_number),
                 forward_propagation->cell_states_activations.data() + (copy_index-neurons_number) + neurons_number,
                 forward_propagation->previous_cell_state_activations.data());

            back_propagation->forget_combinations_recurrent_weights_derivatives = back_propagation->hidden_states_recurrent_weights_derivatives.contract(forget_recurrent_weights, A_B);
            multiply_rows(back_propagation->forget_combinations_recurrent_weights_derivatives, forward_propagation->current_forget_activations_derivatives);
            back_propagation->input_combinations_recurrent_weights_derivatives = back_propagation->hidden_states_recurrent_weights_derivatives.contract(input_recurrent_weights, A_B);
            multiply_rows(back_propagation->input_combinations_recurrent_weights_derivatives, forward_propagation->current_input_activations_derivatives);
            back_propagation->state_combinations_recurrent_weights_derivatives = back_propagation->hidden_states_recurrent_weights_derivatives.contract(state_recurrent_weights, A_B);
            multiply_rows(back_propagation->state_combinations_recurrent_weights_derivatives, forward_propagation->current_state_activations_derivatives);
            back_propagation->output_combinations_recurrent_weights_derivatives = back_propagation->hidden_states_recurrent_weights_derivatives.contract(output_recurrent_weights, A_B);

            column_index = 0;
            activation_index = 0;

            for(Index i = 0; i < parameters_number; i++)
            {
                back_propagation->output_combinations_recurrent_weights_derivatives(i, column_index) += forward_propagation->previous_hidden_state_activations[activation_index];

                activation_index++;

                if(activation_index == neurons_number)
                {
                    activation_index = 0;
                    column_index++;
                }
            }

            multiply_rows(back_propagation->cell_state_recurrent_weights_derivatives, forward_propagation->current_forget_activations);
            multiply_rows(back_propagation->input_combinations_recurrent_weights_derivatives, forward_propagation->current_state_activations);
            back_propagation->cell_state_recurrent_weights_derivatives += back_propagation->input_combinations_recurrent_weights_derivatives;
            multiply_rows(back_propagation->state_combinations_recurrent_weights_derivatives, forward_propagation->current_input_activations);
            back_propagation->cell_state_recurrent_weights_derivatives += back_propagation->state_combinations_recurrent_weights_derivatives;
            multiply_rows(back_propagation->forget_combinations_recurrent_weights_derivatives, forward_propagation->previous_cell_state_activations);
            back_propagation->cell_state_recurrent_weights_derivatives += back_propagation->forget_combinations_recurrent_weights_derivatives;

            back_propagation->hidden_states_recurrent_weights_derivatives = back_propagation->cell_state_recurrent_weights_derivatives;
            multiply_rows(back_propagation->cell_state_recurrent_weights_derivatives, forward_propagation->current_output_activations*forward_propagation->current_hidden_states_derivatives);
            multiply_rows(back_propagation->output_combinations_recurrent_weights_derivatives, forward_propagation->current_output_activations_derivatives*calculate_activations(forward_propagation->current_cell_state_activations));
            back_propagation->hidden_states_recurrent_weights_derivatives += back_propagation->output_combinations_recurrent_weights_derivatives;
        }

        back_propagation->output_recurrent_weights_derivatives += back_propagation->hidden_states_recurrent_weights_derivatives.contract(back_propagation->current_layer_deltas, A_B);

        copy_index += neurons_number;
    }
}


void LongShortTermMemoryLayer::calculate_forget_biases_error_gradient(const Tensor<type, 2>& inputs,
                                                                      LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                      LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number;

    const TensorMap<Tensor<type, 2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));

    back_propagation->input_combinations_biases_derivatives.setZero();
    back_propagation->forget_combinations_biases_derivatives.setZero();
    back_propagation->state_combinations_biases_derivatives.setZero();
    back_propagation->output_combinations_biases_derivatives.setZero();

    back_propagation->hidden_states_biases_derivatives.setZero();
    back_propagation->cell_state_biases_derivatives.setZero();

    Index copy_index = 0;

    back_propagation->forget_biases_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_layer_deltas = deltas.chip(sample, 0);

        copy(forward_propagation->forget_activations.data() + copy_index,
             forward_propagation->forget_activations.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations.data());

        copy(forward_propagation->forget_activations_derivatives.data() + copy_index,
             forward_propagation->forget_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations_derivatives.data());

        copy(forward_propagation->input_activations.data() + copy_index,
             forward_propagation->input_activations.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations.data());

        copy(forward_propagation->input_activations_derivatives.data() + copy_index,
             forward_propagation->input_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations_derivatives.data());

        copy(forward_propagation->state_activations.data() + copy_index,
             forward_propagation->state_activations.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations.data());

        copy(forward_propagation->state_activations_derivatives.data() + copy_index,
             forward_propagation->state_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations_derivatives.data());

        copy(forward_propagation->output_activations.data() + copy_index,
             forward_propagation->output_activations.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations.data());

        copy(forward_propagation->output_activations_derivatives.data() + copy_index,
             forward_propagation->output_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations_derivatives.data());

        copy(forward_propagation->cell_states_activations.data() + copy_index,
             forward_propagation->cell_states_activations.data() + copy_index + neurons_number,
             forward_propagation->current_cell_state_activations.data());

        copy(forward_propagation->hidden_states_activations_derivatives.data() + copy_index,
             forward_propagation->hidden_states_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_hidden_states_derivatives.data());

        if(sample%timesteps == 0)
        {
            back_propagation->forget_combinations_biases_derivatives.setZero();
            back_propagation->input_combinations_biases_derivatives.setZero();
            back_propagation->state_combinations_biases_derivatives.setZero();
            back_propagation->output_combinations_biases_derivatives.setZero();

            forward_propagation->previous_cell_state_activations.setZero();

            back_propagation->cell_state_biases_derivatives.setZero();
        }
        else
        {
            copy(forward_propagation->cell_states_activations.data() + (copy_index-neurons_number),
                 forward_propagation->cell_states_activations.data() + (copy_index-neurons_number) + neurons_number,
                 forward_propagation->previous_cell_state_activations.data());

            back_propagation->forget_combinations_biases_derivatives
                    = back_propagation->hidden_states_biases_derivatives.contract(forget_recurrent_weights, A_B);

            back_propagation->input_combinations_biases_derivatives
                    = back_propagation->hidden_states_biases_derivatives.contract(input_recurrent_weights, A_B);

            multiply_rows(back_propagation->input_combinations_biases_derivatives,
                          forward_propagation->current_input_activations_derivatives);

            back_propagation->state_combinations_biases_derivatives
                    = back_propagation->hidden_states_biases_derivatives.contract(state_recurrent_weights, A_B);

            multiply_rows(back_propagation->state_combinations_biases_derivatives,
                          forward_propagation->current_state_activations_derivatives);

            back_propagation->output_combinations_biases_derivatives
                    = back_propagation->hidden_states_biases_derivatives.contract(output_recurrent_weights, A_B);

            multiply_rows(back_propagation->output_combinations_biases_derivatives,
                          forward_propagation->current_output_activations_derivatives);
        }

        for(Index row = 0; row < parameters_number; row++) back_propagation->forget_combinations_biases_derivatives(row, row) += static_cast<type>(1.0);

        multiply_rows(back_propagation->cell_state_biases_derivatives,
                      forward_propagation->current_forget_activations);

        multiply_rows(back_propagation->input_combinations_biases_derivatives,
                      forward_propagation->current_state_activations);

        back_propagation->cell_state_biases_derivatives += back_propagation->input_combinations_biases_derivatives;

        multiply_rows(back_propagation->state_combinations_biases_derivatives,
                      forward_propagation->current_input_activations);

        back_propagation->cell_state_biases_derivatives += back_propagation->state_combinations_biases_derivatives;

        multiply_rows(back_propagation->forget_combinations_biases_derivatives,
                      forward_propagation->current_forget_activations_derivatives*forward_propagation->previous_cell_state_activations);

        back_propagation->cell_state_biases_derivatives += back_propagation->forget_combinations_biases_derivatives;

        back_propagation->hidden_states_biases_derivatives = back_propagation->cell_state_biases_derivatives;

        multiply_rows(back_propagation->hidden_states_biases_derivatives,
                      forward_propagation->current_output_activations*forward_propagation->current_hidden_states_derivatives);

        multiply_rows(back_propagation->output_combinations_biases_derivatives,
                      calculate_activations(forward_propagation->current_cell_state_activations));

        back_propagation->hidden_states_biases_derivatives += back_propagation->output_combinations_biases_derivatives;

        back_propagation->forget_biases_derivatives += back_propagation->hidden_states_biases_derivatives.contract(current_layer_deltas, A_B);

        copy_index += neurons_number;
    }
}


void LongShortTermMemoryLayer::calculate_input_biases_error_gradient(const Tensor<type, 2>& inputs,
                                                                     LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                     LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number;

    const TensorMap<Tensor<type, 2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));

    back_propagation->input_combinations_biases_derivatives.setZero();
    back_propagation->forget_combinations_biases_derivatives.setZero();
    back_propagation->state_combinations_biases_derivatives.setZero();
    back_propagation->output_combinations_biases_derivatives.setZero();

    back_propagation->hidden_states_biases_derivatives.setZero();
    back_propagation->cell_state_biases_derivatives.setZero();

    Tensor<type, 1> previous_cell_state_activations(neurons_number);

    Index copy_index = 0;

    back_propagation->input_biases_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        back_propagation->current_layer_deltas = deltas.chip(sample, 0);

        copy(forward_propagation->forget_activations.data() + copy_index,
             forward_propagation->forget_activations.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations.data());

        copy(forward_propagation->forget_activations_derivatives.data() + copy_index,
             forward_propagation->forget_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_forget_activations_derivatives.data());

        copy(forward_propagation->input_activations.data() + copy_index,
             forward_propagation->input_activations.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations.data());

        copy(forward_propagation->input_activations_derivatives.data() + copy_index,
             forward_propagation->input_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_input_activations_derivatives.data());

        copy(forward_propagation->state_activations.data() + copy_index,
             forward_propagation->state_activations.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations.data());

        copy(forward_propagation->state_activations_derivatives.data() + copy_index,
             forward_propagation->state_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_state_activations_derivatives.data());

        copy(forward_propagation->output_activations.data() + copy_index,
             forward_propagation->output_activations.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations.data());

        copy(forward_propagation->output_activations_derivatives.data() + copy_index,
             forward_propagation->output_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_output_activations_derivatives.data());

        copy(forward_propagation->cell_states_activations.data() + copy_index,
             forward_propagation->cell_states_activations.data() + copy_index + neurons_number,
             forward_propagation->current_cell_state_activations.data());

        copy(forward_propagation->hidden_states_activations_derivatives.data() + copy_index,
             forward_propagation->hidden_states_activations_derivatives.data() + copy_index + neurons_number,
             forward_propagation->current_hidden_states_derivatives.data());

        if(sample%timesteps == 0)
        {
            back_propagation->forget_combinations_biases_derivatives.setZero();
            back_propagation->input_combinations_biases_derivatives.setZero();
            back_propagation->state_combinations_biases_derivatives.setZero();
            back_propagation->output_combinations_biases_derivatives.setZero();

            previous_cell_state_activations.setZero();
            back_propagation->cell_state_biases_derivatives.setZero();
        }
        else
        {
            copy(forward_propagation->cell_states_activations.data() + (copy_index-neurons_number),
                 forward_propagation->cell_states_activations.data() + (copy_index-neurons_number) + neurons_number,
                 previous_cell_state_activations.data());

            back_propagation->forget_combinations_biases_derivatives
                    = back_propagation->hidden_states_biases_derivatives.contract(forget_recurrent_weights, A_B);

            multiply_rows(back_propagation->forget_combinations_biases_derivatives,
                          forward_propagation->current_forget_activations_derivatives);

            back_propagation->input_combinations_biases_derivatives
                    = back_propagation->hidden_states_biases_derivatives.contract(input_recurrent_weights, A_B);

            back_propagation->state_combinations_biases_derivatives
                    = back_propagation->hidden_states_biases_derivatives.contract(state_recurrent_weights, A_B);

            multiply_rows(back_propagation->state_combinations_biases_derivatives,
                          forward_propagation->current_state_activations_derivatives);

            back_propagation->output_combinations_biases_derivatives
                    = back_propagation->hidden_states_biases_derivatives.contract(output_recurrent_weights, A_B);

            multiply_rows(back_propagation->output_combinations_biases_derivatives,
                          forward_propagation->current_output_activations_derivatives);
        }

        for(Index row = 0; row < parameters_number; row++)
            back_propagation->input_combinations_biases_derivatives(row, row) += static_cast<type>(1.0);

        multiply_rows(back_propagation->cell_state_biases_derivatives,
                      forward_propagation->current_forget_activations);

        multiply_rows(back_propagation->forget_combinations_biases_derivatives, previous_cell_state_activations);

        back_propagation->cell_state_biases_derivatives += back_propagation->forget_combinations_biases_derivatives;

        multiply_rows(back_propagation->state_combinations_biases_derivatives,
                      forward_propagation->current_input_activations);

        back_propagation->cell_state_biases_derivatives += back_propagation->state_combinations_biases_derivatives;

        multiply_rows(back_propagation->input_combinations_biases_derivatives,
                      forward_propagation->current_input_activations_derivatives*forward_propagation->current_state_activations);

        back_propagation->cell_state_biases_derivatives += back_propagation->input_combinations_biases_derivatives;

        back_propagation->hidden_states_biases_derivatives = back_propagation->cell_state_biases_derivatives;

        multiply_rows(back_propagation->hidden_states_biases_derivatives,
                      forward_propagation->current_output_activations*forward_propagation->current_hidden_states_derivatives);

        multiply_rows(back_propagation->output_combinations_biases_derivatives,
                      calculate_activations(forward_propagation->current_cell_state_activations));

        back_propagation->hidden_states_biases_derivatives += back_propagation->output_combinations_biases_derivatives;

        back_propagation->input_biases_derivatives
                += back_propagation->hidden_states_biases_derivatives.contract(back_propagation->current_layer_deltas, A_B);

        copy_index += neurons_number;
    }
}


void LongShortTermMemoryLayer::calculate_state_biases_error_gradient(const Tensor<type, 2>& inputs,
                                                                     LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                     LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number;

    const TensorMap<Tensor<type, 2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));

    back_propagation->input_combinations_biases_derivatives.setZero();
    back_propagation->forget_combinations_biases_derivatives.setZero();
    back_propagation->state_combinations_biases_derivatives.setZero();
    back_propagation->output_combinations_biases_derivatives.setZero();

    back_propagation->hidden_states_biases_derivatives.setZero();
    back_propagation->cell_state_biases_derivatives.setZero();

    Index copy_index = 0;

    back_propagation->state_biases_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        const Tensor<type, 1> current_layer_deltas = deltas.chip(sample, 0);

        memcpy(forward_propagation->current_forget_activations.data(),
               forward_propagation->forget_activations.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_forget_activations_derivatives.data(),
               forward_propagation->forget_activations_derivatives.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_input_activations.data(),
               forward_propagation->input_activations.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_input_activations_derivatives.data(),
               forward_propagation->input_activations_derivatives.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_state_activations.data(),
               forward_propagation->state_activations.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_state_activations_derivatives.data(),
               forward_propagation->state_activations_derivatives.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_output_activations.data(),
               forward_propagation->output_activations.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_output_activations_derivatives.data(),
               forward_propagation->output_activations_derivatives.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_cell_state_activations.data(),
               forward_propagation->cell_states_activations.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_hidden_states_derivatives.data(),
               forward_propagation->hidden_states_activations_derivatives.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        if(sample%timesteps == 0)
        {
            back_propagation->forget_combinations_biases_derivatives.setZero();
            back_propagation->input_combinations_biases_derivatives.setZero();
            back_propagation->state_combinations_biases_derivatives.setZero();
            back_propagation->output_combinations_biases_derivatives.setZero();

            forward_propagation->previous_cell_state_activations.setZero();
            back_propagation->cell_state_biases_derivatives.setZero();
        }
        else
        {
            memcpy(forward_propagation->previous_cell_state_activations.data(),
                   forward_propagation->cell_states_activations.data() + (copy_index-neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            back_propagation->forget_combinations_biases_derivatives = back_propagation->hidden_states_biases_derivatives.contract(forget_recurrent_weights, A_B);

            multiply_rows(back_propagation->forget_combinations_biases_derivatives,
                          forward_propagation->current_forget_activations_derivatives);

            back_propagation->input_combinations_biases_derivatives = back_propagation->hidden_states_biases_derivatives.contract(input_recurrent_weights, A_B);

            multiply_rows(back_propagation->input_combinations_biases_derivatives,
                          forward_propagation->current_input_activations_derivatives);

            back_propagation->state_combinations_biases_derivatives = back_propagation->hidden_states_biases_derivatives.contract(state_recurrent_weights, A_B);

            back_propagation->output_combinations_biases_derivatives = back_propagation->hidden_states_biases_derivatives.contract(output_recurrent_weights, A_B);

            multiply_rows(back_propagation->output_combinations_biases_derivatives,
                          forward_propagation->current_output_activations_derivatives);
        }

        for(Index row = 0; row < parameters_number; row++) back_propagation->state_combinations_biases_derivatives(row, row) += static_cast<type>(1.0);

        multiply_rows(back_propagation->cell_state_biases_derivatives,
                      forward_propagation->current_forget_activations);

        multiply_rows(back_propagation->forget_combinations_biases_derivatives, forward_propagation->previous_cell_state_activations);

        back_propagation->cell_state_biases_derivatives += back_propagation->forget_combinations_biases_derivatives;

        multiply_rows(back_propagation->input_combinations_biases_derivatives,
                      forward_propagation->current_state_activations);

        back_propagation->cell_state_biases_derivatives += back_propagation->input_combinations_biases_derivatives;

        multiply_rows(back_propagation->state_combinations_biases_derivatives,
                      forward_propagation->current_state_activations_derivatives*forward_propagation->current_input_activations);

        back_propagation->cell_state_biases_derivatives += back_propagation->state_combinations_biases_derivatives;

        back_propagation->hidden_states_biases_derivatives = back_propagation->cell_state_biases_derivatives;

        multiply_rows(back_propagation->hidden_states_biases_derivatives,
                      forward_propagation->current_output_activations*forward_propagation->current_hidden_states_derivatives);

        multiply_rows(back_propagation->output_combinations_biases_derivatives,
                      calculate_activations(forward_propagation->current_cell_state_activations));

        back_propagation->hidden_states_biases_derivatives += back_propagation->output_combinations_biases_derivatives;

        back_propagation->state_biases_derivatives += back_propagation->hidden_states_biases_derivatives.contract(current_layer_deltas, A_B);

        copy_index += neurons_number;
    }
}


void LongShortTermMemoryLayer::calculate_output_biases_error_gradient(const Tensor<type, 2>& inputs,
                                                                      LongShortTermMemoryLayerForwardPropagation* forward_propagation,
                                                                      LongShortTermMemoryLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number;

    const TensorMap<Tensor<type, 2>> deltas(back_propagation->deltas_data, back_propagation->deltas_dimensions(0), back_propagation->deltas_dimensions(1));

    back_propagation->input_combinations_biases_derivatives.setZero();
    back_propagation->forget_combinations_biases_derivatives.setZero();
    back_propagation->state_combinations_biases_derivatives.setZero();
    back_propagation->output_combinations_biases_derivatives.setZero();
    back_propagation->hidden_states_biases_derivatives.setZero();
    back_propagation->cell_state_biases_derivatives.setZero();

    Index copy_index = 0;

    back_propagation->output_biases_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        back_propagation->current_layer_deltas = deltas.chip(sample, 0);

        memcpy(forward_propagation->current_forget_activations.data(),
               forward_propagation->forget_activations.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_forget_activations_derivatives.data(),
               forward_propagation->forget_activations_derivatives.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_input_activations.data(),
               forward_propagation->input_activations.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_input_activations_derivatives.data(),
               forward_propagation->input_activations_derivatives.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_state_activations.data(),
               forward_propagation->state_activations.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_state_activations_derivatives.data(),
               forward_propagation->state_activations_derivatives.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_output_activations.data(),
               forward_propagation->output_activations.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_output_activations_derivatives.data(),
               forward_propagation->output_activations_derivatives.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_cell_state_activations.data(),
               forward_propagation->cell_states_activations.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        memcpy(forward_propagation->current_hidden_states_derivatives.data(),
               forward_propagation->hidden_states_activations_derivatives.data()+copy_index,
               static_cast<size_t>(neurons_number)*sizeof(type));

        if(sample%timesteps == 0)
        {
            back_propagation->forget_combinations_biases_derivatives.setZero();
            back_propagation->input_combinations_biases_derivatives.setZero();
            back_propagation->state_combinations_biases_derivatives.setZero();
            back_propagation->output_combinations_biases_derivatives.setZero();

            forward_propagation->previous_cell_state_activations.setZero();
            back_propagation->cell_state_biases_derivatives.setZero();
        }
        else
        {
            memcpy(forward_propagation->previous_cell_state_activations.data(),
                   forward_propagation->cell_states_activations.data() + (copy_index-neurons_number),
                   static_cast<size_t>(neurons_number)*sizeof(type));

            back_propagation->forget_combinations_biases_derivatives = back_propagation->hidden_states_biases_derivatives.contract(forget_recurrent_weights, A_B);

            multiply_rows(back_propagation->forget_combinations_biases_derivatives,
                          forward_propagation->current_forget_activations_derivatives);

            back_propagation->input_combinations_biases_derivatives = back_propagation->hidden_states_biases_derivatives.contract(input_recurrent_weights, A_B);

            multiply_rows(back_propagation->input_combinations_biases_derivatives,
                          forward_propagation->current_input_activations_derivatives);

            back_propagation->state_combinations_biases_derivatives = back_propagation->hidden_states_biases_derivatives.contract(state_recurrent_weights, A_B);

            multiply_rows(back_propagation->state_combinations_biases_derivatives,
                          forward_propagation->current_state_activations_derivatives);

            back_propagation->output_combinations_biases_derivatives = back_propagation->hidden_states_biases_derivatives.contract(output_recurrent_weights, A_B);
        }

        for(Index row = 0; row < parameters_number; row++) back_propagation->output_combinations_biases_derivatives(row, row) += static_cast<type>(1.0);

        multiply_rows(back_propagation->cell_state_biases_derivatives,
                      forward_propagation->current_forget_activations);

        multiply_rows(back_propagation->forget_combinations_biases_derivatives, forward_propagation->previous_cell_state_activations);

        back_propagation->cell_state_biases_derivatives += back_propagation->forget_combinations_biases_derivatives;

        multiply_rows(back_propagation->state_combinations_biases_derivatives,
                      forward_propagation->current_input_activations);

        back_propagation->cell_state_biases_derivatives += back_propagation->state_combinations_biases_derivatives;

        multiply_rows(back_propagation->input_combinations_biases_derivatives,
                      forward_propagation->current_state_activations);

        back_propagation->cell_state_biases_derivatives += back_propagation->input_combinations_biases_derivatives;

        back_propagation->hidden_states_biases_derivatives = back_propagation->cell_state_biases_derivatives;

        multiply_rows(back_propagation->hidden_states_biases_derivatives,
                      forward_propagation->current_output_activations*forward_propagation->current_hidden_states_derivatives);

        multiply_rows(back_propagation->output_combinations_biases_derivatives,
                      forward_propagation->current_output_activations_derivatives*calculate_activations(forward_propagation->current_cell_state_activations));

        back_propagation->hidden_states_biases_derivatives += back_propagation->output_combinations_biases_derivatives;

        back_propagation->output_biases_derivatives += back_propagation->hidden_states_biases_derivatives.contract(back_propagation->current_layer_deltas, A_B);

        copy_index += neurons_number;
    }
}


/// Returns a string with the expression of the inputs-outputs relationship of the layer.
/// @param inputs_names Vector of strings with the name of the layer inputs.
/// @param outputs_names Vector of strings with the name of the layer outputs.
/// @todo Update this method.

string LongShortTermMemoryLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    const Index neurons_number = get_neurons_number();

    const Index inputs_number = get_inputs_number();

#ifdef OPENNN_DEBUG

    const Index inputs_name_size = inputs_names.size();

    if(inputs_name_size != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const method.\n"
               << "Size of inputs name must be equal to number of layer inputs.\n";

        throw invalid_argument(buffer.str());
    }

    const Index outputs_name_size = outputs_names.size();

    if(outputs_name_size != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const method.\n"
               << "Size of outputs name must be equal to number of neurons.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    ostringstream buffer;

        // Forget gate

        for(Index i = 0; i < neurons_number; i++)
        {
            buffer << "forget_gate_" << to_string(i) << " = " << write_recurrent_activation_function_expression() << " (" << forget_biases[i] << " + ";

            for(Index j = 0; j < inputs_number; j++)
            {
                buffer << inputs_names[j] << " * (" << forget_weights(j,i) << ") + ";
            }

            for(Index k = 0; k < neurons_number-1; k++)
            {
                buffer << "hidden_state_" << to_string(k) << "(t-1) * (" << forget_recurrent_weights(k,i) << ") + ";
            }

            buffer << "hidden_state_" << to_string(neurons_number-1) << "(t-1) * (" << forget_recurrent_weights(neurons_number-1,i) << ") );\n";
        }

       // Input gate

       for(Index i = 0; i < neurons_number; i++)
       {
           buffer << "input_gate_" << to_string(i) << " = " << write_recurrent_activation_function_expression() << " (" << input_biases[i] << " + ";

           for(Index j = 0; j < inputs_number; j++)
           {
               buffer << inputs_names[j] << " * (" << input_weights(j,i) << ") + ";
           }

           for(Index k = 0; k < neurons_number-1; k++)
           {
               buffer << "hidden_state_" << to_string(k) << "(t-1) * (" << input_recurrent_weights(k,i) << ") + ";
           }

           buffer << "hidden_state_" << to_string(neurons_number-1) << "(t-1) * (" << input_recurrent_weights(neurons_number-1,i) << ") );\n";
       }

       // State gate

       for(Index i = 0; i < neurons_number; i++)
       {
           buffer << "state_gate_" << to_string(i) << " = " << write_activation_function_expression() << " (" << state_biases[i] << " + ";

           for(Index j = 0; j < inputs_number; j++)
           {
               buffer << inputs_names[j] << " * (" << state_weights(j,i) << ") + ";
           }

           for(Index k = 0; k < neurons_number-1; k++)
           {
               buffer << "hidden_state_" << to_string(k) << "(t-1) * (" << state_recurrent_weights(k,i) << ") + ";
           }

           buffer << "hidden_state_" << to_string(neurons_number-1) << "(t-1) * (" << state_recurrent_weights(neurons_number-1,i) << ") );\n";
       }

       // Output gate

       for(Index i = 0; i < neurons_number; i++)
       {
           buffer << "output_gate_" << to_string(i) << " = " << write_recurrent_activation_function_expression() << " (" << output_biases[i] << " + ";

           for(Index j = 0; j < inputs_number; j++)
           {
               buffer << inputs_names[j] << " * (" << output_weights(j,i) << ") + ";
           }

           for(Index k = 0; k < neurons_number-1; k++)
           {
               buffer << "hidden_state_" << to_string(k) << "(t-1) * (" << output_recurrent_weights(k,i) << ") + ";
           }

           buffer << "hidden_state_" << to_string(neurons_number-1) << "(t-1) * (" << output_recurrent_weights(neurons_number-1,i) << ") );\n";
       }

       // Cell state

       for(Index i = 0; i < neurons_number; i++)
       {
            buffer << "cell_state_" << to_string(i) << "(t) = forget_gate_" << to_string(i) << " * cell_state_" << to_string(i) << "(t-1)+input_gate_" << to_string(i) << " * state_gate_" << to_string(i) << ";\n";
       }

       // Hidden state

       for(Index i = 0; i < neurons_number; i++)
       {
            buffer << "hidden_state_" << to_string(i) << "(t) = output_gate_" << to_string(i) << " * " << write_activation_function_expression() << "(cell_state_" << to_string(i) << ");\n";
       }

       // Output

       for(Index i = 0; i < neurons_number; i++)
       {
           buffer << outputs_names[i] << " = " << "hidden_state_" << to_string(i) << "(t);\n";
       }

       return buffer.str();
}

string LongShortTermMemoryLayer::write_expression_c() const
{
    ostringstream buffer;

    buffer << "vector<float> " << layer_name << "(const vector<float>& inputs)\n{" << endl;

    buffer << write_combinations_c();

    buffer << "\n\treturn long_short_term_memory_output;\n}" << endl;

    return buffer.str();
}


string LongShortTermMemoryLayer::write_combinations_c() const
{
    ostringstream buffer;

    const Index neurons_number = get_neurons_number();
    const Index inputs_number =  get_inputs_number();

    // Forget gate

    buffer << "\tvector<float> forget_gate_combinations(" << neurons_number << ");\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\tforget_gate_combinations[" << i << "] = " << forget_biases(i) << " + ";

        for(Index j = 0; j < inputs_number; j++)
        {
            buffer << " inputs[" << j << "] * (" << forget_weights(j,i) << ") + ";
        }

        for(Index k = 0; k < neurons_number-1; k++)
        {
            buffer << "hidden_states[" << k << "]" << " * (" << forget_recurrent_weights(k,i) << ") + ";
        }

        buffer << "hidden_states[" << neurons_number-1 << "]" << " * (" << forget_recurrent_weights(neurons_number-1,i) << "); \n";
    }

    buffer << endl;


    buffer << "\tvector<float> forget_gate_activations(" << neurons_number << ");\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\tforget_gate_activations[" << i << "] = ";

        switch(recurrent_activation_function)
        {
        case ActivationFunction::HyperbolicTangent:
            buffer << "tanh(forget_gate_combinations[" << i << "]);\n";
            break;

        case ActivationFunction::RectifiedLinear:
            buffer << "forget_gate_combinations[" << i << "] < 0.0 ? 0.0 : forget_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::Logistic:
            buffer << "1.0/(1.0 + exp(-forget_gate_combinations[" << i << "]));\n";
            break;

        case ActivationFunction::Threshold:
            buffer << "forget_gate_combinations[" << i << "] >= 0.0 ? 1.0 : 0.0;\n";
            break;

        case ActivationFunction::SymmetricThreshold:
            buffer << "forget_gate_combinations[" << i << "] >= 0.0 ? 1.0 : -1.0;\n";
            break;

        case ActivationFunction::Linear:
            buffer << "forget_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::ScaledExponentialLinear:
            buffer << "forget_gate_combinations[" << i << "] < 0.0 ? 1.0507*1.67326*(exp(forget_gate_combinations[" << i << "]) - 1.0) : 1.0507*forget_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::SoftPlus:
            buffer << "log(1.0 + exp(forget_gate_combinations[" << i << "]));\n";
            break;

        case ActivationFunction::SoftSign:
            buffer << "forget_gate_combinations[" << i << "] < 0.0 ? forget_gate_combinations[" << i << "]/(1.0 - forget_gate_combinations[" << i << "] ) : forget_gate_combinations[" << i << "]/(1.0 + forget_gate_combinations[" << i << "] );\n";
            break;

        case ActivationFunction::ExponentialLinear:
            buffer << "forget_gate_combinations[" << i << "] < 0.0 ? 1.0*(exp(forget_gate_combinations[" << i << "]) - 1.0) : forget_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::HardSigmoid:
            ///@todo
            break;

        default:
            buffer << "forget_gate_combinations[" << i << "] < 0.0 ? 0.0 : forget_gate_combinations[" << i << "];\n";
            break;
        }
    }

    buffer << endl;


    // Input gate

    buffer << "\tvector<float> input_gate_combinations(" << neurons_number << ");\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\tinput_gate_combinations[" << i << "] = " << input_biases(i) << " + ";

        for(Index j = 0; j < inputs_number; j++)
        {
            buffer << "inputs[" << j << "] * (" << input_weights(j,i) << ") + ";
        }

        for(Index k = 0; k < neurons_number-1; k++)
        {
            buffer << "hidden_states[" << k << "]" << " * (" << input_recurrent_weights(k,i) << ") + ";
        }

        buffer << "hidden_states[" << neurons_number-1 << "]" << " * (" << input_recurrent_weights(neurons_number-1,i) << "); \n";
    }

    buffer << endl;


    buffer << "\tvector<float> input_gate_activations(" << neurons_number << ");\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\tinput_gate_activations[" << i << "] = ";

        switch(recurrent_activation_function)
        {
        case ActivationFunction::HyperbolicTangent:
            buffer << "tanh(input_gate_combinations[" << i << "]);\n";
            break;

        case ActivationFunction::RectifiedLinear:
            buffer << "input_gate_combinations[" << i << "] < 0.0 ? 0.0 : input_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::Logistic:
            buffer << "1.0/(1.0 + exp(-input_gate_combinations[" << i << "]));\n";
            break;

        case ActivationFunction::Threshold:
            buffer << "input_gate_combinations[" << i << "] >= 0.0 ? 1.0 : 0.0;\n";
            break;

        case ActivationFunction::SymmetricThreshold:
            buffer << "input_gate_combinations[" << i << "] >= 0.0 ? 1.0 : -1.0;\n";
            break;

        case ActivationFunction::Linear:
            buffer << "input_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::ScaledExponentialLinear:
            buffer << "input_gate_combinations[" << i << "] < 0.0 ? 1.0507*1.67326*(exp(input_gate_combinations[" << i << "]) - 1.0) : 1.0507*input_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::SoftPlus:
            buffer << "log(1.0 + exp(input_gate_combinations[" << i << "]));\n";
            break;

        case ActivationFunction::SoftSign:
            buffer << "input_gate_combinations[" << i << "] < 0.0 ? input_gate_combinations[" << i << "]/(1.0 - input_gate_combinations[" << i << "] ) : input_gate_combinations[" << i << "]/(1.0 + input_gate_combinations[" << i << "] );\n";
            break;

        case ActivationFunction::ExponentialLinear:
            buffer << "input_gate_combinations[" << i << "] < 0.0 ? 1.0*(exp(input_gate_combinations[" << i << "]) - 1.0) : input_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::HardSigmoid:
            ///@todo
            break;

        default:
            buffer << "input_gate_combinations[" << i << "] < 0.0 ? 0.0 : input_gate_combinations[" << i << "];\n";
            break;
        }
    }

    buffer << endl;


    // State gate

    buffer << "\tvector<float> state_gate_combinations(" << neurons_number << ");\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\tstate_gate_combinations[" << i << "] = " << state_biases(i) << " + ";

        for(Index j = 0; j < inputs_number; j++)
        {
            buffer << "inputs[" << j << "] * (" << state_weights(j,i) << ") + ";
        }

        for(Index k = 0; k < neurons_number-1; k++)
        {
            buffer << "hidden_states[" << k << "]" << " * (" << state_recurrent_weights(k,i) << ") + ";
        }

        buffer << "hidden_states[" << neurons_number-1 << "]" << " * (" << state_recurrent_weights(neurons_number-1,i) << "); \n";
    }

    buffer << endl;


    buffer << "\tvector<float> state_gate_activations(" << neurons_number << ");\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\tstate_gate_activations[" << i << "] = ";

        switch(activation_function)
        {
        case ActivationFunction::HyperbolicTangent:
            buffer << "tanh(state_gate_combinations[" << i << "]);\n";
            break;

        case ActivationFunction::RectifiedLinear:
            buffer << "state_gate_combinations[" << i << "] < 0.0 ? 0.0 : state_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::Logistic:
            buffer << "1.0/(1.0 + exp(-state_gate_combinations[" << i << "]));\n";
            break;

        case ActivationFunction::Threshold:
            buffer << "state_gate_combinations[" << i << "] >= 0.0 ? 1.0 : 0.0;\n";
            break;

        case ActivationFunction::SymmetricThreshold:
            buffer << "state_gate_combinations[" << i << "] >= 0.0 ? 1.0 : -1.0;\n";
            break;

        case ActivationFunction::Linear:
            buffer << "state_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::ScaledExponentialLinear:
            buffer << "state_gate_combinations[" << i << "] < 0.0 ? 1.0507*1.67326*(exp(state_gate_combinations[" << i << "]) - 1.0) : 1.0507*state_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::SoftPlus:
            buffer << "log(1.0 + exp(state_gate_combinations[" << i << "]));\n";
            break;

        case ActivationFunction::SoftSign:
            buffer << "state_gate_combinations[" << i << "] < 0.0 ? state_gate_combinations[" << i << "]/(1.0 - state_gate_combinations[" << i << "] ) : state_gate_combinations[" << i << "]/(1.0 + state_gate_combinations[" << i << "] );\n";
            break;

        case ActivationFunction::ExponentialLinear:
            buffer << "state_gate_combinations[" << i << "] < 0.0 ? 1.0*(exp(state_gate_combinations[" << i << "]) - 1.0) : state_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::HardSigmoid:
            ///@todo
            break;

        default:
            buffer << "state_gate_combinations[" << i << "] < 0.0 ? 0.0 : state_gate_combinations[" << i << "];\n";
            break;
        }
    }

    buffer << endl;


    // Output gate

    buffer << "\tvector<float> output_gate_combinations(" << neurons_number << ");\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\toutput_gate_combinations[" << i << "] = " << output_biases(i) << " + ";

        for(Index j = 0; j < inputs_number; j++)
        {
            buffer << "inputs[" << j << "] * (" << output_weights(j,i) << ") + ";
        }

        for(Index k = 0; k < neurons_number-1; k++)
        {
            buffer << "hidden_states[" << k << "]" << " * (" << output_recurrent_weights(k,i) << ") + ";
        }

        buffer << "hidden_states[" << neurons_number-1 << "]" << " * (" << output_recurrent_weights(neurons_number-1,i) << "); \n";
    }

    buffer << endl;


    buffer << "\tvector<float> output_gate_activations(" << neurons_number << ");\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\toutput_gate_activations[" << i << "] = ";

        switch(recurrent_activation_function)
        {
        case ActivationFunction::HyperbolicTangent:
            buffer << "tanh(output_gate_combinations[" << i << "]);\n";
            break;

        case ActivationFunction::RectifiedLinear:
            buffer << "output_gate_combinations[" << i << "] < 0.0 ? 0.0 : output_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::Logistic:
            buffer << "1.0/(1.0 + exp(-output_gate_combinations[" << i << "]));\n";
            break;

        case ActivationFunction::Threshold:
            buffer << "output_gate_combinations[" << i << "] >= 0.0 ? 1.0 : 0.0;\n";
            break;

        case ActivationFunction::SymmetricThreshold:
            buffer << "output_gate_combinations[" << i << "] >= 0.0 ? 1.0 : -1.0;\n";
            break;

        case ActivationFunction::Linear:
            buffer << "output_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::ScaledExponentialLinear:
            buffer << "output_gate_combinations[" << i << "] < 0.0 ? 1.0507*1.67326*(exp(output_gate_combinations[" << i << "]) - 1.0) : 1.0507*output_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::SoftPlus:
            buffer << "log(1.0 + exp(output_gate_combinations[" << i << "]));\n";
            break;

        case ActivationFunction::SoftSign:
            buffer << "output_gate_combinations[" << i << "] < 0.0 ? output_gate_combinations[" << i << "]/(1.0 - output_gate_combinations[" << i << "] ) : output_gate_combinations[" << i << "]/(1.0 + output_gate_combinations[" << i << "] );\n";
            break;

        case ActivationFunction::ExponentialLinear:
            buffer << "output_gate_combinations[" << i << "] < 0.0 ? 1.0*(exp(output_gate_combinations[" << i << "]) - 1.0) : output_gate_combinations[" << i << "];\n";
            break;

        case ActivationFunction::HardSigmoid:
            ///@todo
            break;

        default:
            buffer << "output_gate_combinations[" << i << "] < 0.0 ? 0.0 : output_gate_combinations[" << i << "];\n";
            break;
        }
    }

    buffer << endl;


    // Cell State

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\tcell_states[" << i << "] = forget_gate_activations[" << i << "] * cell_states[" << i << "] + input_gate_activations[" << i << "] * state_gate_activations[" << i << "]; \n";
    }

    buffer << endl;


    buffer << "\tvector<float> cell_state_activations(" << neurons_number << ");\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\tcell_state_activations[" << i << "] = ";

        switch(activation_function)
        {
        case ActivationFunction::HyperbolicTangent:
            buffer << "tanh(cell_states[" << i << "]);\n";
            break;

        case ActivationFunction::RectifiedLinear:
            buffer << "cell_states[" << i << "] < 0.0 ? 0.0 : cell_states[" << i << "];\n";
            break;

        case ActivationFunction::Logistic:
            buffer << "1.0/(1.0 + exp(-cell_states[" << i << "]));\n";
            break;

        case ActivationFunction::Threshold:
            buffer << "cell_states[" << i << "] >= 0.0 ? 1.0 : 0.0;\n";
            break;

        case ActivationFunction::SymmetricThreshold:
            buffer << "cell_states[" << i << "] >= 0.0 ? 1.0 : -1.0;\n";
            break;

        case ActivationFunction::Linear:
            buffer << "cell_states[" << i << "];\n";
            break;

        case ActivationFunction::ScaledExponentialLinear:
            buffer << "cell_states[" << i << "] < 0.0 ? 1.0507*1.67326*(exp(cell_states[" << i << "]) - 1.0) : 1.0507*cell_states[" << i << "];\n";
            break;

        case ActivationFunction::SoftPlus:
            buffer << "log(1.0 + exp(cell_states[" << i << "]));\n";
            break;

        case ActivationFunction::SoftSign:
            buffer << "cell_states[" << i << "] < 0.0 ? cell_states[" << i << "]/(1.0 - cell_states[" << i << "] ) : cell_states[" << i << "]/(1.0 + cell_states[" << i << "] );\n";
            break;

        case ActivationFunction::ExponentialLinear:
            buffer << "cell_states[" << i << "] < 0.0 ? 1.0*(exp(cell_states[" << i << "]) - 1.0) : cell_states[" << i << "];\n";
            break;

        case ActivationFunction::HardSigmoid:
            ///@todo
            break;

        default:
            buffer << "cell_states[" << i << "] < 0.0 ? 0.0 : cell_states[" << i << "];\n";
            break;
        }
    }

    buffer << endl;

    // Hidden state

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\thidden_states[" << i << "] = output_gate_activations[" << i << "] * cell_state_activations[" << i << "];\n";
    }

    buffer << endl;


    // LSTM output

    buffer << "\tvector<float> long_short_term_memory_output(" << neurons_number << ");\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\tlong_short_term_memory_output[" << i << "] = hidden_states[" << i << "];\n";
    }

    return buffer.str();
}

string LongShortTermMemoryLayer::write_expression_python() const
{
    ostringstream buffer;

    buffer << "\tdef " << layer_name << "(self,inputs):\n" << endl;

    buffer << write_combinations_python();

    buffer << "\n\t\treturn long_short_term_memory_output;\n" << endl;

    return buffer.str();
}

string LongShortTermMemoryLayer::write_combinations_python() const
{
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    // Forget gate

    buffer << "\t\tforget_gate_combinations = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\t\tforget_gate_combinations[" << i << "] = " << forget_biases(i) << " + ";

        for(Index j = 0; j < inputs_number; j++)
        {
             buffer << "inputs[" << j << "] * (" << forget_weights(j,i) << ") + ";
        }

        for(Index k = 0; k < neurons_number-1; k++)
        {
             buffer << "self.hidden_states[" << k << "] * (" << forget_recurrent_weights(k,i) << ") + ";
        }

        buffer << "self.hidden_states[" << neurons_number-1 << "] * (" << forget_recurrent_weights(neurons_number-1,i) << ")";

        buffer << " " << endl;
    }

    buffer << "\t\t" << endl;


    buffer << "\t\tforget_gate_activations = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\t\tforget_gate_activations[" << i << "] = ";

        switch(recurrent_activation_function)
        {
        case ActivationFunction::HyperbolicTangent:
            buffer << "np.tanh(forget_gate_combinations[" << i << "])\n";
            break;

        case ActivationFunction::RectifiedLinear:
            buffer << "np.maximum(0.0, forget_gate_combinations[" << i << "])\n";
            break;

        case ActivationFunction::Logistic:
            buffer << "1.0/(1.0 + np.exp(-forget_gate_combinations[" << i << "]))\n";
            break;

        case ActivationFunction::Threshold:
            buffer << "1.0 if forget_gate_combinations[" << i << "] >= 0.0 else 0.0\n";
            break;

        case ActivationFunction::SymmetricThreshold:
            buffer << "1.0 if forget_gate_combinations[" << i << "] >= 0.0 else -1.0\n";
            break;

        case ActivationFunction::Linear:
            buffer << "forget_gate_combinations[" << i << "]\n";
            break;

        case ActivationFunction::ScaledExponentialLinear:
            buffer << "1.0507*1.67326*(np.exp(forget_gate_combinations[" << i << "]) - 1.0) if forget_gate_combinations[" << i << "] < 0.0 else 1.0507*forget_gate_combinations[" << i << "]\n";
            break;

        case ActivationFunction::SoftPlus:
            buffer << "np.log(1.0 + np.exp(forget_gate_combinations[" << i << "]))\n";
            break;

        case ActivationFunction::SoftSign:
            buffer << "forget_gate_combinations[" << i << "]/(1.0 - forget_gate_combinations[" << i << "] ) if forget_gate_combinations[" << i << "] < 0.0 else forget_gate_combinations[" << i << "]/(1.0 + forget_gate_combinations[" << i << "] )\n";
            break;

        case ActivationFunction::ExponentialLinear:
            buffer << "1.0*(np.exp(forget_gate_combinations[" << i << "]) - 1.0) if forget_gate_combinations[" << i << "] < 0.0 else forget_gate_combinations[" << i << "]\n";
            break;

        case ActivationFunction::HardSigmoid:
            ///@todo
            break;

        default:
            buffer << "np.maximum(0.0, forget_gate_combinations[" << i << "])\n";
            break;
        }
    }


    buffer << "\t\t" << endl;

    // Input gate

    buffer << "\t\tinput_gate_combinations = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\t\tinput_gate_combinations[" << i << "] = " << input_biases(i) << " + ";

        for(Index j = 0; j < inputs_number; j++)
        {
             buffer << "inputs[" << j << "] * (" << input_weights(j,i) << ") + ";
        }

        for(Index k = 0; k < neurons_number-1; k++)
        {
             buffer << "self.hidden_states[" << k << "] * (" << input_recurrent_weights(k,i) << ") + ";
        }

        buffer << "self.hidden_states[" << neurons_number-1 << "] * (" << input_recurrent_weights(neurons_number-1,i) << ")";

        buffer << " " << endl;
    }


    buffer << "\t\t" << endl;

    buffer << "\t\tinput_gate_activations = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\t\tinput_gate_activations[" << i << "] = ";

        switch(recurrent_activation_function)
        {
        case ActivationFunction::HyperbolicTangent:
            buffer << "np.tanh(input_gate_combinations[" << i << "])\n";
            break;

        case ActivationFunction::RectifiedLinear:
            buffer << "np.maximum(0.0, input_gate_combinations[" << i << "])\n";
            break;

        case ActivationFunction::Logistic:
            buffer << "1.0/(1.0 + np.exp(-input_gate_combinations[" << i << "]))\n";
            break;

        case ActivationFunction::Threshold:
            buffer << "1.0 if input_gate_combinations[" << i << "] >= 0.0 else 0.0\n";
            break;

        case ActivationFunction::SymmetricThreshold:
            buffer << "1.0 if input_gate_combinations[" << i << "] >= 0.0 else -1.0\n";
            break;

        case ActivationFunction::Linear:
            buffer << "input_gate_combinations[" << i << "]\n";
            break;

        case ActivationFunction::ScaledExponentialLinear:
            buffer << "1.0507*1.67326*(np.exp(input_gate_combinations[" << i << "]) - 1.0) if input_gate_combinations[" << i << "] < 0.0 else 1.0507*input_gate_combinations[" << i << "]\n";
            break;

        case ActivationFunction::SoftPlus:
            buffer << "np.log(1.0 + np.exp(input_gate_combinations[" << i << "]))\n";
            break;

        case ActivationFunction::SoftSign:
            buffer << "input_gate_combinations[" << i << "]/(1.0 - input_gate_combinations[" << i << "] ) if input_gate_combinations[" << i << "] < 0.0 else input_gate_combinations[" << i << "]/(1.0 + input_gate_combinations[" << i << "] )\n";
            break;

        case ActivationFunction::ExponentialLinear:
            buffer << "1.0*(np.exp(input_gate_combinations[" << i << "]) - 1.0) if input_gate_combinations[" << i << "] < 0.0 else input_gate_combinations[" << i << "]\n";
            break;

        case ActivationFunction::HardSigmoid:
            ///@todo
            break;

        default:
            buffer << "np.maximum(0.0, input_gate_combinations[" << i << "])\n";
            break;
        }
    }

    buffer << "\t\t" << endl;


    // State gate

    buffer << "\t\tstate_gate_combinations = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\t\tstate_gate_combinations[" << i << "] = " << state_biases(i) << " + ";

        for(Index j = 0; j < inputs_number; j++)
        {
             buffer << "inputs[" << j << "] * (" << state_weights(j,i) << ") + ";
        }

        for(Index k = 0; k < neurons_number-1; k++)
        {
             buffer << "self.hidden_states[" << k << "] * (" << state_recurrent_weights(k,i) << ") + ";
        }

        buffer << "self.hidden_states[" << neurons_number-1 << "] * (" << state_recurrent_weights(neurons_number-1,i) << ")";

        buffer << " " << endl;
    }


    buffer << "\t\t" << endl;

    buffer << "\t\tstate_gate_activations = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\t\tstate_gate_activations[" << i << "] = ";

        switch(activation_function)
        {
        case ActivationFunction::HyperbolicTangent:
            buffer << "np.tanh(state_gate_combinations[" << i << "])\n";
            break;

        case ActivationFunction::RectifiedLinear:
            buffer << "np.maximum(0.0, state_gate_combinations[" << i << "])\n";
            break;

        case ActivationFunction::Logistic:
            buffer << "1.0/(1.0 + np.exp(-state_gate_combinations[" << i << "]))\n";
            break;

        case ActivationFunction::Threshold:
            buffer << "1.0 if state_gate_combinations[" << i << "] >= 0.0 else 0.0\n";
            break;

        case ActivationFunction::SymmetricThreshold:
            buffer << "1.0 if state_gate_combinations[" << i << "] >= 0.0 else -1.0\n";
            break;

        case ActivationFunction::Linear:
            buffer << "state_gate_combinations[" << i << "]\n";
            break;

        case ActivationFunction::ScaledExponentialLinear:
            buffer << "1.0507*1.67326*(np.exp(state_gate_combinations[" << i << "]) - 1.0) if state_gate_combinations[" << i << "] < 0.0 else 1.0507*state_gate_combinations[" << i << "]\n";
            break;

        case ActivationFunction::SoftPlus:
            buffer << "np.log(1.0 + np.exp(state_gate_combinations[" << i << "]))\n";
            break;

        case ActivationFunction::SoftSign:
            buffer << "state_gate_combinations[" << i << "]/(1.0 - state_gate_combinations[" << i << "] ) if state_gate_combinations[" << i << "] < 0.0 else state_gate_combinations[" << i << "]/(1.0 + state_gate_combinations[" << i << "] )\n";
            break;

        case ActivationFunction::ExponentialLinear:
            buffer << "1.0*(np.exp(state_gate_combinations[" << i << "]) - 1.0) if state_gate_combinations[" << i << "] < 0.0 else state_gate_combinations[" << i << "]\n";
            break;

        case ActivationFunction::HardSigmoid:
            ///@todo
            break;

        default:
            buffer << "np.maximum(0.0, state_gate_combinations[" << i << "])\n";
            break;
        }
    }

    buffer << "\t\t" << endl;


    // Output gate

    buffer << "\t\toutput_gate_combinations = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\t\toutput_gate_combinations[" << i << "] = " << output_biases(i) << " + ";

        for(Index j = 0; j < inputs_number; j++)
        {
             buffer << "inputs[" << j << "] * (" << output_weights(j,i) << ") + ";
        }

        for(Index k = 0; k < neurons_number-1; k++)
        {
             buffer << "self.hidden_states[" << k << "] * (" << output_recurrent_weights(k,i) << ") + ";
        }

        buffer << "self.hidden_states[" << neurons_number-1 << "] * (" << output_recurrent_weights(neurons_number-1,i) << ")";

        buffer << " " << endl;
    }


    buffer << "\t\t" << endl;

    buffer << "\t\toutput_gate_activations = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\t\toutput_gate_activations[" << i << "] = ";

        switch(activation_function)
        {
        case ActivationFunction::HyperbolicTangent:
            buffer << "np.tanh(output_gate_combinations[" << i << "])\n";
            break;

        case ActivationFunction::RectifiedLinear:
            buffer << "np.maximum(0.0, output_gate_combinations[" << i << "])\n";
            break;

        case ActivationFunction::Logistic:
            buffer << "1.0/(1.0 + np.exp(-output_gate_combinations[" << i << "]))\n";
            break;

        case ActivationFunction::Threshold:
            buffer << "1.0 if output_gate_combinations[" << i << "] >= 0.0 else 0.0\n";
            break;

        case ActivationFunction::SymmetricThreshold:
            buffer << "1.0 if output_gate_combinations[" << i << "] >= 0.0 else -1.0\n";
            break;

        case ActivationFunction::Linear:
            buffer << "output_gate_combinations[" << i << "]\n";
            break;

        case ActivationFunction::ScaledExponentialLinear:
            buffer << "1.0507*1.67326*(np.exp(output_gate_combinations[" << i << "]) - 1.0) if output_gate_combinations[" << i << "] < 0.0 else 1.0507*output_gate_combinations[" << i << "]\n";
            break;

        case ActivationFunction::SoftPlus:
            buffer << "np.log(1.0 + np.exp(output_gate_combinations[" << i << "]))\n";
            break;

        case ActivationFunction::SoftSign:
            buffer << "output_gate_combinations[" << i << "]/(1.0 - output_gate_combinations[" << i << "] ) if output_gate_combinations[" << i << "] < 0.0 else output_gate_combinations[" << i << "]/(1.0 + output_gate_combinations[" << i << "] )\n";
            break;

        case ActivationFunction::ExponentialLinear:
            buffer << "1.0*(np.exp(output_gate_combinations[" << i << "]) - 1.0) if output_gate_combinations[" << i << "] < 0.0 else output_gate_combinations[" << i << "]\n";
            break;

        case ActivationFunction::HardSigmoid:
            ///@todo
            break;

        default:
            buffer << "np.maximum(0.0, output_gate_combinations[" << i << "])\n";
            break;
        }
    }

    buffer << "\t\t" << endl;


    // Cell states

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\t\tself.cell_states[" << i << "] = forget_gate_activations[" << i << "] * self.cell_states[" << i << "] + input_gate_activations[" << i << "] * state_gate_activations[" << i << "] \n";
    }

    buffer << " " << endl;

    buffer << "\t\t" << endl;

    buffer << "\t\tcell_state_activations = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\t\tcell_state_activations[" << i << "] = ";

        switch(activation_function)
        {
        case ActivationFunction::HyperbolicTangent:
            buffer << "np.tanh(self.cell_states[" << i << "])\n";
            break;

        case ActivationFunction::RectifiedLinear:
            buffer << "np.maximum(0.0, self.cell_states[" << i << "])\n";
            break;

        case ActivationFunction::Logistic:
            buffer << "1.0/(1.0 + np.exp(-self.cell_states[" << i << "]))\n";
            break;

        case ActivationFunction::Threshold:
            buffer << "1.0 if self.cell_states[" << i << "] >= 0.0 else 0.0\n";
            break;

        case ActivationFunction::SymmetricThreshold:
            buffer << "1.0 if self.cell_states[" << i << "] >= 0.0 else -1.0\n";
            break;

        case ActivationFunction::Linear:
            buffer << "self.cell_states[" << i << "]\n";
            break;

        case ActivationFunction::ScaledExponentialLinear:
            buffer << "1.0507*1.67326*(np.exp(self.cell_states[" << i << "]) - 1.0) if self.cell_states[" << i << "] < 0.0 else 1.0507*self.cell_states[" << i << "]\n";
            break;

        case ActivationFunction::SoftPlus:
            buffer << "np.log(1.0 + np.exp(self.cell_states[" << i << "]))\n";
            break;

        case ActivationFunction::SoftSign:
            buffer << "self.cell_states[" << i << "]/(1.0 - self.cell_states[" << i << "] ) if self.cell_states[" << i << "] < 0.0 else self.cell_states[" << i << "]/(1.0 + self.cell_states[" << i << "] )\n";
            break;

        case ActivationFunction::ExponentialLinear:
            buffer << "1.0*(np.exp(self.cell_states[" << i << "]) - 1.0) if self.cell_states[" << i << "] < 0.0 else self.cell_states[" << i << "]\n";
            break;

        case ActivationFunction::HardSigmoid:
            ///@todo
            break;

        default:
            buffer << "np.maximum(0.0, self.cell_states[" << i << "])\n";
            break;
        }
    }

    buffer << "\t\t" << endl;


    // Hidden state

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\t\tself.hidden_states[" << i << "] = output_gate_activations[" << i << "] * cell_state_activations[" << i << "]\n";
    }

    buffer << " " << endl;

    buffer << "\t\t" << endl;


    // LSTM output

    buffer << "\t\tlong_short_term_memory_output = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\t\tlong_short_term_memory_output[" << i << "] = self.hidden_states[" << i << "]\n";
    }

    return buffer.str();
}

void LongShortTermMemoryLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // LongShortTermMemoryLayer layer

    const tinyxml2::XMLElement* long_short_term_memory_layer_element = document.FirstChildElement("LongShortTermMemoryLayer");

    if(!long_short_term_memory_layer_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "PerceptronLayer element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Layer name

    const tinyxml2::XMLElement* layer_name_element = long_short_term_memory_layer_element->FirstChildElement("LayerName");

    if(!layer_name_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "LayerName element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(layer_name_element->GetText())
    {
        set_name(layer_name_element->GetText());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = long_short_term_memory_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "InputsNumber element is nullptr.\n";

        throw invalid_argument(buffer.str());
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

        throw invalid_argument(buffer.str());
    }

    if(neurons_number_element->GetText())
    {
        set_neurons_number(static_cast<Index>(stoi(neurons_number_element->GetText())));
    }

    // Time step

    const tinyxml2::XMLElement* time_step_element = long_short_term_memory_layer_element->FirstChildElement("TimeStep");

    if(!time_step_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "TimeStep element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    if(time_step_element->GetText())
    {
        set_timesteps(static_cast<Index>(stoi(time_step_element->GetText())));
    }

    // Activation function

    const tinyxml2::XMLElement* activation_function_element = long_short_term_memory_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "ActivationFunction element is nullptr.\n";

        throw invalid_argument(buffer.str());
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

        throw invalid_argument(buffer.str());
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

        throw invalid_argument(buffer.str());
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

    // Long short-term memory layer

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

    // Time step

    file_stream.OpenElement("TimeStep");

    buffer.str("");
    buffer << get_timesteps();

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

    // Long short-term memory layer (end tag)

    file_stream.CloseElement();
}

string LongShortTermMemoryLayer::write_recurrent_activation_function_expression() const
{
    switch(recurrent_activation_function)
    {
    case ActivationFunction::HyperbolicTangent:
    {
        return "tanh";
    }
    case ActivationFunction::Linear:
    {
        return string();
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
    case ActivationFunction::HyperbolicTangent:
    {
        return "tanh";
    }
    case ActivationFunction::Linear:
    {
        return string();
    }
    default:
    {
        return write_activation_function();
    }
    }
}
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
