//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// OpeNN Includes

#include "long_short_term_memory_layer.h"
#include "tensors.h"

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


dimensions LongShortTermMemoryLayer::get_output_dimensions() const
{
    Index neurons_number = get_neurons_number();

    return { neurons_number };
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
/// The number of raw_variables is the number of neurons to the layer.
///
Tensor<type, 2> LongShortTermMemoryLayer::get_forget_weights() const
{
    return forget_weights;
}

/// Returns the input weights from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of inputs in the layer.
/// The number of raw_variables is the number of neurons to the layer.

Tensor<type, 2> LongShortTermMemoryLayer::get_input_weights() const
{
    return input_weights;
}


/// Returns the state weights from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of inputs in the layer.
/// The number of raw_variables is the number of neurons to the layer.

Tensor<type, 2> LongShortTermMemoryLayer::get_state_weights() const
{
    return state_weights;
}

/// Returns the output weights from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of inputs in the layer.
/// The number of raw_variables is the number of neurons to the layer.

Tensor<type, 2> LongShortTermMemoryLayer::get_output_weights() const
{
    return output_weights;
}


/// Returns the forget recurrent weights from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of raw_variables is the number of neurons to the layer.

Tensor<type, 2> LongShortTermMemoryLayer::get_forget_recurrent_weights() const
{
    return forget_recurrent_weights;
}


/// Returns the input recurrent weights from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of raw_variables is the number of neurons to the layer.

Tensor<type, 2> LongShortTermMemoryLayer::get_input_recurrent_weights() const
{
    return input_recurrent_weights;
}


/// Returns the state recurrent weights from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of raw_variables is the number of neurons to the layer.

Tensor<type, 2> LongShortTermMemoryLayer::get_state_recurrent_weights() const
{
    return state_recurrent_weights;
}


/// Returns the output recurrent weights from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of raw_variables is the number of neurons to the layer.

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

    copy(/*execution::par,*/ 
         (type*)forget_biases.data(),
         (type*)forget_biases.data() + forget_biases.size(),
         (type*)parameters.data());

    copy(/*execution::par,*/ 
         input_biases.data(),
         input_biases.data() + input_biases.size(),
         parameters.data() + current_position);

    current_position += input_biases.size();

    copy(/*execution::par,*/ 
         state_biases.data(),
         state_biases.data() + state_biases.size(),
         parameters.data() + current_position);

    current_position += state_biases.size();

    copy(/*execution::par,*/ 
         output_biases.data(),
         output_biases.data() + output_biases.size(),
         parameters.data() + current_position);

    current_position += output_biases.size();

    // Weights

    copy(/*execution::par,*/ 
         forget_weights.data(),
         forget_weights.data() + forget_weights.size(),
         parameters.data() + current_position);

    current_position += forget_weights.size();

    copy(/*execution::par,*/ 
         input_weights.data(),
         input_weights.data() + input_weights.size(),
         parameters.data() + current_position);

    current_position += input_weights.size();

    copy(/*execution::par,*/ 
         state_weights.data(),
         state_weights.data() + state_weights.size(),
         parameters.data() + current_position);

    current_position += state_weights.size();

    copy(/*execution::par,*/ 
         output_weights.data(),
         output_weights.data() + output_weights.size(),
         parameters.data() + current_position);

    current_position += output_weights.size();

    // Recurrent weights

    copy(/*execution::par,*/ 
         forget_recurrent_weights.data(),
         forget_recurrent_weights.data() + forget_recurrent_weights.size(),
         parameters.data() + current_position);

    current_position += forget_recurrent_weights.size();

    copy(/*execution::par,*/ 
         input_recurrent_weights.data(),
         input_recurrent_weights.data() + input_recurrent_weights.size(),
         parameters.data() + current_position);

    current_position += input_recurrent_weights.size();

    copy(/*execution::par,*/ 
         state_recurrent_weights.data(),
         state_recurrent_weights.data() + state_recurrent_weights.size(),
         parameters.data() + current_position);

    current_position += state_recurrent_weights.size();

    copy(/*execution::par,*/ 
         output_recurrent_weights.data(),
         output_recurrent_weights.data() + output_recurrent_weights.size(),
         parameters.data() + current_position);

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
/// This can be Logistic, HyperbolicTangent, Linear, RectifiedLinear, ScaledExponentialLinear.

string LongShortTermMemoryLayer::write_activation_function() const
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


/// Returns a string with the name of the layer recurrent activation function.
/// This can be Logistic, HyperbolicTangent, Linear, RectifiedLinear, ScaledExponentialLinear.

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
/// The number of raw_variables is the number of neurons to the corresponding layer.
/// @param new_forget_weights New set of forget weights in that layer.

void LongShortTermMemoryLayer::set_forget_weights(const Tensor<type, 2>& new_forget_weights)
{
    forget_weights = new_forget_weights;
}


/// Sets the input weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of inputs in the corresponding layer.
/// The number of raw_variables is the number of neurons to the corresponding layer.
/// @param new_input_weights New set of input weights in that layer.

void LongShortTermMemoryLayer::set_input_weights(const Tensor<type, 2>& new_input_weight)
{
    input_weights = new_input_weight;
}


/// Sets the state weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of inputs in the corresponding layer.
/// The number of raw_variables is the number of neurons to the corresponding layer.
/// @param new_state_weights New set of state weights in that layer.

void LongShortTermMemoryLayer::set_state_weights(const Tensor<type, 2>& new_state_weights)
{
    state_weights = new_state_weights;
}


/// Sets the output weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of inputs in the corresponding layer.
/// The number of raw_variables is the number of neurons to the corresponding layer.
/// @param new_output_weights New set of output weights in that layer.

void LongShortTermMemoryLayer::set_output_weights(const Tensor<type, 2>& new_output_weight)
{
    output_weights = new_output_weight;

}


/// Sets the forget recurrent weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of neurons in the corresponding layer.
/// The number of raw_variables is the number of neurons to the corresponding layer.
/// @param new_forget_recurrent_weights New set of forget recurrent weights in that layer.

void LongShortTermMemoryLayer::set_forget_recurrent_weights(const Tensor<type, 2>& new_forget_recurrent_weight)
{
    forget_recurrent_weights = new_forget_recurrent_weight;
}


/// Sets the input recurrent weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of neurons in the corresponding layer.
/// The number of raw_variables is the number of neurons to the corresponding layer.
/// @param new_input_recurrent_weights New set of input recurrent weights in that layer.


void LongShortTermMemoryLayer::set_input_recurrent_weights(const Tensor<type, 2>& new_input_recurrent_weight)
{
    input_recurrent_weights = new_input_recurrent_weight;
}


/// Sets the state recurrent weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of neurons in the corresponding layer.
/// The number of raw_variables is the number of neurons to the corresponding layer.
/// @param new_state_recurrent_weights New set of state recurrent weights in that layer.

void LongShortTermMemoryLayer::set_state_recurrent_weights(const Tensor<type, 2>& new_state_recurrent_weight)
{
    state_recurrent_weights = new_state_recurrent_weight;
}


/// Sets the output recurrent weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of neurons in the corresponding layer.
/// The number of raw_variables is the number of neurons to the corresponding layer.
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

    Index current_index = index;

    const type* new_parameters_data = new_parameters.data();

    // Biases

    Index size = neurons_number;

    copy(/*execution::par,*/ 
        new_parameters_data + current_index,
         new_parameters_data + current_index + size,
         forget_biases.data());

    current_index += size;

    copy(/*execution::par,*/ 
         new_parameters_data + current_index,
         new_parameters_data + current_index + size,
         input_biases.data());

    current_index += size;

    copy(/*execution::par,*/ 
         new_parameters_data + current_index,
         new_parameters_data + current_index + size,
         state_biases.data());

    current_index += size;

    copy(/*execution::par,*/ 
         new_parameters_data + current_index,
         new_parameters_data + current_index + size,
         output_biases.data());

    current_index += size;

    // Weights

    size = inputs_number*neurons_number;

    copy(/*execution::par,*/ 
         new_parameters_data + current_index,
         new_parameters_data + current_index + size,
         forget_weights.data());

    current_index += size;

    copy(/*execution::par,*/ 
         new_parameters_data + current_index,
         new_parameters_data + current_index + size,
         input_weights.data());

    current_index += size;

    copy(/*execution::par,*/ 
         new_parameters_data + current_index,
         new_parameters_data + current_index + size,
         state_weights.data());

    current_index += size;

    copy(/*execution::par,*/ 
         new_parameters_data + current_index,
         new_parameters_data + current_index + size,
         output_weights.data());

    current_index += size;

    // Recurrent weights

    size = neurons_number*neurons_number;

    copy(/*execution::par,*/ 
         new_parameters_data + current_index,
         new_parameters_data + current_index + size,
         forget_recurrent_weights.data());

    current_index += size;

    copy(/*execution::par,*/ 
         new_parameters_data + current_index,
         new_parameters_data + current_index + size,
         input_recurrent_weights.data());

    current_index += size;

    copy(/*execution::par,*/ 
         new_parameters_data + current_index,
         new_parameters_data + current_index + size,
         state_recurrent_weights.data());

    current_index += size;

    copy(/*execution::par,*/ 
         new_parameters_data + current_index,
         new_parameters_data + current_index + size,
         output_recurrent_weights.data());         
}


/// This class sets a new activation(or transfer) function in a single layer.
/// @param new_activation_function Activation function for the layer.

void LongShortTermMemoryLayer::set_activation_function(const LongShortTermMemoryLayer::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


/// Sets a new activation(or transfer) function in a single layer.
/// The argument is a string containing the name of the function("Logistic", "HyperbolicTangent", etc).
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

        throw runtime_error(buffer.str());
    }
}


/// This class sets a new recurrent activation(or transfer) function in a single layer.
/// @param new_recurrent_activation_function Activation function for the layer.

void LongShortTermMemoryLayer::set_recurrent_activation_function(const LongShortTermMemoryLayer::ActivationFunction& new_recurrent_activation_function)
{
    recurrent_activation_function = new_recurrent_activation_function;
}


/// Sets a new recurrent activation(or transfer) function in a single layer.
/// The argument is a string containing the name of the function("Logistic", "HyperbolicTangent", etc).
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

        throw runtime_error(buffer.str());
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
    forget_biases.setRandom();

    input_biases.setRandom();

    state_biases.setRandom();

    output_biases.setRandom();

    forget_weights.setRandom();

    input_weights.setRandom();

    state_weights.setRandom();

    output_weights.setRandom();

    forget_recurrent_weights.setRandom();

    input_recurrent_weights.setRandom();

    state_recurrent_weights.setRandom();

    output_recurrent_weights.setRandom();
}


void LongShortTermMemoryLayer::calculate_combinations(const Tensor<type, 1>& inputs,
                                                      const Tensor<type, 2>& weights,
                                                      const Tensor<type, 2>& recurrent_weights,
                                                      const Tensor<type, 1>& biases,
                                                      Tensor<type, 1>& combinations)
{
    combinations.device(*thread_pool_device) = inputs.contract(weights, AT_B)
                                             + biases
                                             + hidden_states.contract(recurrent_weights, AT_B);
}


void LongShortTermMemoryLayer::calculate_activations(const Tensor<type, 1>& combinations,
                                                     Tensor<type, 1>& activations) const
{
    switch(activation_function)
    {
    case ActivationFunction::Linear: linear(combinations, activations); return;

    case ActivationFunction::Logistic: logistic(combinations, activations); return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(combinations, activations); return;

    case ActivationFunction::RectifiedLinear: rectified_linear(combinations, activations); return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(combinations, activations); return;

    case ActivationFunction::SoftPlus: soft_plus(combinations, activations); return;

    case ActivationFunction::SoftSign: soft_sign(combinations, activations); return;

    case ActivationFunction::HardSigmoid: hard_sigmoid(combinations, activations); return;

    case ActivationFunction::ExponentialLinear: exponential_linear(combinations, activations); return;

    default: rectified_linear(combinations, activations); return;
    }
}


void LongShortTermMemoryLayer::calculate_recurrent_activations(const Tensor<type, 1>& combinations,
                                                               Tensor<type, 1>& activations) const
{
    switch(recurrent_activation_function)
    {
    case ActivationFunction::Linear: linear(combinations, activations); return;

    case ActivationFunction::Logistic: logistic(combinations, activations); return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(combinations, activations); return;

    case ActivationFunction::RectifiedLinear: rectified_linear(combinations, activations); return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(combinations, activations); return;

    case ActivationFunction::SoftPlus: soft_plus(combinations, activations); return;

    case ActivationFunction::SoftSign: soft_sign(combinations, activations); return;

    case ActivationFunction::HardSigmoid: hard_sigmoid(combinations, activations); return;

    case ActivationFunction::ExponentialLinear: exponential_linear(combinations, activations); return;

    default: rectified_linear(combinations, activations); return;
    }
}


void LongShortTermMemoryLayer::calculate_activations_derivatives(const Tensor<type, 1>& combinations,
                                                                 Tensor<type, 1>& activations,
                                                                 Tensor<type, 1>& activations_derivatives) const
{
    switch(activation_function)
    {
    case ActivationFunction::Linear: linear_derivatives(combinations,
                                                        activations,
                                                        activations_derivatives);
        return;

    case ActivationFunction::Logistic: logistic_derivatives(combinations,
                                                            activations,
                                                            activations_derivatives);
        return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent_derivatives(combinations,
                                                                               activations,
                                                                               activations_derivatives);
        return;

    case ActivationFunction::RectifiedLinear: rectified_linear_derivatives(combinations,
                                                                           activations,
                                                                           activations_derivatives);
        return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear_derivatives(combinations,
                                                                                            activations,
                                                                                            activations_derivatives);
        return;

    case ActivationFunction::SoftPlus: soft_plus_derivatives(combinations,
                                                             activations,
                                                             activations_derivatives);
        return;

    case ActivationFunction::SoftSign: soft_sign_derivatives(combinations,
                                                             activations,
                                                             activations_derivatives);
        return;

    case ActivationFunction::HardSigmoid: hard_sigmoid_derivatives(combinations,
                                                                   activations,
                                                                   activations_derivatives); return;

    case ActivationFunction::ExponentialLinear: exponential_linear_derivatives(combinations,
                                                                               activations,
                                                                               activations_derivatives); return;

    default: rectified_linear_derivatives(combinations,
                                          activations,
                                          activations_derivatives); return;

    }
}


void LongShortTermMemoryLayer::calculate_recurrent_activations_derivatives(const Tensor<type, 1>& combinations,
                                                                           Tensor<type, 1>& activations,
                                                                           Tensor<type, 1>& activations_derivatives) const
{
    switch(recurrent_activation_function)
    {
    case ActivationFunction::Linear: 
        
        linear_derivatives(combinations, activations, activations_derivatives); 
        
        return;

    case ActivationFunction::Logistic: 
        
        logistic_derivatives(combinations, activations, activations_derivatives); 
        
        return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent_derivatives(combinations, activations, activations_derivatives); return;

    case ActivationFunction::RectifiedLinear: rectified_linear_derivatives(combinations, activations, activations_derivatives); return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear_derivatives(combinations, activations, activations_derivatives); return;

    case ActivationFunction::SoftPlus: soft_plus_derivatives(combinations, activations, activations_derivatives); return;

    case ActivationFunction::SoftSign: soft_sign_derivatives(combinations, activations, activations_derivatives); return;

    case ActivationFunction::HardSigmoid: hard_sigmoid_derivatives(combinations, activations, activations_derivatives); return;

    case ActivationFunction::ExponentialLinear: exponential_linear_derivatives(combinations, activations, activations_derivatives); return;

    default: rectified_linear_derivatives(combinations, activations, activations_derivatives); return;

    }
}


void LongShortTermMemoryLayer::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                                 LayerForwardPropagation* forward_propagation,
                                                 const bool& is_training)
{
    const Index samples_number = inputs_pair(0).second[0];

    const Index neurons_number = get_neurons_number();

    LongShortTermMemoryLayerForwardPropagation* long_short_term_memory_layer_forward_propagation
            = static_cast<LongShortTermMemoryLayerForwardPropagation*>(forward_propagation);

    const TensorMap<Tensor<type, 2>> inputs(inputs_pair(0).first, samples_number, neurons_number);
    Tensor<type, 1>& current_inputs = long_short_term_memory_layer_forward_propagation->current_inputs;

    Tensor<type, 2, RowMajor>& forget_activations = long_short_term_memory_layer_forward_propagation->forget_activations;
    Tensor<type, 2, RowMajor>& forget_activations_derivatives = long_short_term_memory_layer_forward_propagation->forget_activations_derivatives;

    Tensor<type, 1>& current_forget_combinations = long_short_term_memory_layer_forward_propagation->current_forget_combinations;
    Tensor<type, 1>& current_forget_activations = long_short_term_memory_layer_forward_propagation->current_forget_activations;
    Tensor<type, 1>& current_forget_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_forget_activations_derivatives;

    Tensor<type, 2, RowMajor>& input_activations = long_short_term_memory_layer_forward_propagation->input_activations;
    Tensor<type, 2, RowMajor>& input_activations_derivatives = long_short_term_memory_layer_forward_propagation->input_activations_derivatives;

    Tensor<type, 1>& current_input_combinations = long_short_term_memory_layer_forward_propagation->current_input_combinations;
    Tensor<type, 1>& current_input_activations = long_short_term_memory_layer_forward_propagation->current_input_activations;
    Tensor<type, 1>& current_input_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_input_activations_derivatives;

    Tensor<type, 2, RowMajor>& state_activations = long_short_term_memory_layer_forward_propagation->state_activations;
    Tensor<type, 2, RowMajor>& state_activations_derivatives = long_short_term_memory_layer_forward_propagation->state_activations_derivatives;

    Tensor<type, 1>& current_state_combinations = long_short_term_memory_layer_forward_propagation->current_state_combinations;
    Tensor<type, 1>& current_state_activations = long_short_term_memory_layer_forward_propagation->current_state_activations;
    Tensor<type, 1>& current_state_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_state_activations_derivatives;

    Tensor<type, 2, RowMajor>& output_activations = long_short_term_memory_layer_forward_propagation->output_activations;
    Tensor<type, 2, RowMajor>& output_activations_derivatives = long_short_term_memory_layer_forward_propagation->output_activations_derivatives;

    Tensor<type, 1>& current_output_combinations = long_short_term_memory_layer_forward_propagation->current_output_combinations;
    Tensor<type, 1>& current_output_activations = long_short_term_memory_layer_forward_propagation->current_output_activations;
    Tensor<type, 1>& current_output_activations_derivatives = long_short_term_memory_layer_forward_propagation->current_output_activations_derivatives;

    Tensor<type, 2, RowMajor>& cell_states = long_short_term_memory_layer_forward_propagation->cell_states;

    Tensor<type, 2, RowMajor>& hidden_states = long_short_term_memory_layer_forward_propagation->hidden_states;
    Tensor<type, 1>& current_hidden_states_derivatives = long_short_term_memory_layer_forward_propagation->current_hidden_states_derivatives;

    Tensor<type, 2>& outputs = long_short_term_memory_layer_forward_propagation->outputs;

    for(Index i = 0; i < samples_number; i++)
    {
        if(i%timesteps == 0)
        {
            hidden_states.setZero();
            cell_states.setZero();
        }

        current_inputs.device(*thread_pool_device) = inputs.chip(i, 0);

        calculate_combinations(current_inputs,
                               forget_weights,
                               forget_recurrent_weights,
                               forget_biases,
                               current_forget_combinations);

        calculate_combinations(current_inputs,
                               input_weights,
                               input_recurrent_weights,
                               input_biases,
                               current_input_combinations);

        calculate_combinations(current_inputs,
                               state_weights,
                               state_recurrent_weights,
                               state_biases,
                               current_state_combinations);

        calculate_combinations(current_inputs,
                               output_weights,
                               output_recurrent_weights,
                               output_biases,
                               current_output_combinations);

        if(is_training)
        {
            calculate_recurrent_activations_derivatives(current_forget_combinations,
                                                        current_forget_activations,
                                                        current_forget_activations_derivatives);

            calculate_recurrent_activations_derivatives(current_input_combinations,
                                                        current_input_activations,
                                                        current_input_activations_derivatives);

            calculate_activations_derivatives(current_state_combinations,
                                              current_state_activations,
                                              current_state_activations_derivatives);

            calculate_recurrent_activations_derivatives(current_output_combinations,
                                                        current_output_activations,
                                                        current_output_activations_derivatives);

            set_row(forget_activations_derivatives, current_forget_activations_derivatives, i);

            set_row(input_activations_derivatives, current_input_activations_derivatives, i);

            set_row(state_activations_derivatives, current_state_activations_derivatives, i);

            set_row(output_activations_derivatives, current_output_activations_derivatives, i);
        }
        else
        {
            calculate_recurrent_activations(current_forget_combinations,
                                            current_forget_activations);

            calculate_recurrent_activations(current_input_combinations,
                                            current_input_activations);

            calculate_activations(current_state_combinations,
                                  current_state_activations);

            calculate_recurrent_activations(current_output_combinations,
                                            current_output_activations);
        }

        set_row(forget_activations, current_forget_activations, i);

        set_row(input_activations, current_input_activations, i);

        set_row(state_activations, current_state_activations, i);
   
        set_row(output_activations, current_output_activations, i);

        // Cell states
        /*
        cell_states.device(*thread_pool_device)
            = current_forget_activations * cell_states + current_input_activations * current_state_activations;

        
        calculate_activations(cell_states,
        hidden_states);

        calculate_activations_derivatives(cell_states,
        hidden_states,
        current_hidden_states_derivatives);

        set_row(hidden_states_derivatives, current_hidden_states_derivatives, i);

        copy(/*execution::par,
        cell_states_data,
        cell_states_data + neurons_number,
        cell_states_data + copy_index);

        copy(/*execution::par,
        hidden_states_data,
        hidden_states_data + neurons_number,
        hidden_states_data + copy_index);

        // Hidden states

        hidden_states.device(*thread_pool_device) = hidden_states*current_output_activations;

        // Activations 2d

        outputs.chip(i, 0).device(*thread_pool_device) = hidden_states;
*/
    }
}


void LongShortTermMemoryLayer::calculate_error_gradient(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                                        const Tensor<pair<type*, dimensions>, 1>& deltas_pair,
                                                        LayerForwardPropagation* forward_propagation,
                                                        LayerBackPropagation* back_propagation) const
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = inputs_number * neurons_number;

    const TensorMap<Tensor<type, 2>> inputs(inputs_pair(0).first, inputs_pair(0).second[0], inputs_pair(0).second[1]);
    const Index samples_number = inputs.dimension(0);

    const TensorMap<Tensor<type, 2>> deltas(deltas_pair(0).first, deltas_pair(0).second[0], deltas_pair(0).second[1]);

    LongShortTermMemoryLayerForwardPropagation* long_short_term_memory_layer_forward_propagation =
            static_cast<LongShortTermMemoryLayerForwardPropagation*>(forward_propagation);

    LongShortTermMemoryLayerBackPropagation* long_short_term_memory_layer_back_propagation =
            static_cast<LongShortTermMemoryLayerBackPropagation*>(back_propagation);

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
    Tensor<type, 1>& current_cell_states = long_short_term_memory_layer_forward_propagation->current_cell_states;

    Tensor<type, 1>& previous_cell_states = long_short_term_memory_layer_forward_propagation->previous_cell_states;

    const Tensor<type, 2, RowMajor>& hidden_states_derivatives = long_short_term_memory_layer_forward_propagation->hidden_states_derivatives;
    Tensor<type, 1>& current_hidden_states_derivatives = long_short_term_memory_layer_forward_propagation->current_hidden_states_derivatives;

    Tensor<type, 1>& previous_hidden_state_activations = long_short_term_memory_layer_forward_propagation->previous_hidden_state_activations;

    // Back propagation

    Tensor<type, 1>& current_deltas = long_short_term_memory_layer_back_propagation->current_deltas;

    Tensor<type, 2>& input_combinations_weights_derivatives = long_short_term_memory_layer_back_propagation->input_combinations_weights_derivatives;
    input_combinations_weights_derivatives.setZero();

    Tensor<type, 2>& forget_combinations_weights_derivatives = long_short_term_memory_layer_back_propagation->forget_combinations_weights_derivatives;
    forget_combinations_weights_derivatives.setZero();

    Tensor<type, 2>& state_combinations_weights_derivatives = long_short_term_memory_layer_back_propagation->state_combinations_weights_derivatives;
    state_combinations_weights_derivatives.setZero();

    Tensor<type, 2>& output_combinations_weights_derivatives = long_short_term_memory_layer_back_propagation->output_combinations_weights_derivatives;
    output_combinations_weights_derivatives.setZero();

    Tensor<type, 2>& cell_states_weights_derivatives = long_short_term_memory_layer_back_propagation->cell_states_weights_derivatives;
    cell_states_weights_derivatives.setZero();

    Tensor<type, 2>& cell_states_biases_derivatives = long_short_term_memory_layer_back_propagation->cell_states_biases_derivatives;
    cell_states_biases_derivatives.setZero();

    Tensor<type, 2>& hidden_states_weights_derivatives = long_short_term_memory_layer_back_propagation->hidden_states_weights_derivatives;
    hidden_states_weights_derivatives.setZero();

    Tensor<type, 1>& forget_weights_derivatives = long_short_term_memory_layer_back_propagation->forget_weights_derivatives;
    forget_weights_derivatives.setZero();

    Tensor<type, 2>& cell_states_recurrent_weights_derivatives = long_short_term_memory_layer_back_propagation->cell_states_recurrent_weights_derivatives;
    cell_states_recurrent_weights_derivatives.setZero();

    Tensor<type, 2>& hidden_states_recurrent_weights_derivatives = long_short_term_memory_layer_back_propagation->hidden_states_recurrent_weights_derivatives;
    hidden_states_recurrent_weights_derivatives.setZero();

    Tensor<type, 2>& forget_combinations_recurrent_weights_derivatives = long_short_term_memory_layer_back_propagation->forget_combinations_recurrent_weights_derivatives;
    forget_combinations_recurrent_weights_derivatives.setZero();

    Tensor<type, 2>& input_combinations_recurrent_weights_derivatives = long_short_term_memory_layer_back_propagation->input_combinations_recurrent_weights_derivatives;
    input_combinations_recurrent_weights_derivatives.setZero();

    Tensor<type, 2>& state_combinations_recurrent_weights_derivatives = long_short_term_memory_layer_back_propagation->state_combinations_recurrent_weights_derivatives;
    state_combinations_recurrent_weights_derivatives.setZero();

    Tensor<type, 2>& output_combinations_recurrent_weights_derivatives = long_short_term_memory_layer_back_propagation->output_combinations_recurrent_weights_derivatives;
    output_combinations_recurrent_weights_derivatives.setZero();

    previous_cell_states.setZero();

//    input_weights_derivatives.setZero();

    Index raw_variable_index = 0;
    Index input_index = 0;
    Index neuron_index = 0;    

    for (Index sample = 0; sample < samples_number; sample++)
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

        get_row(current_hidden_states_derivatives, hidden_states_derivatives, sample);

        current_deltas.device(*thread_pool_device) = deltas.chip(sample, 0);

        if(sample % timesteps == 0)
        {
            forget_combinations_weights_derivatives.setZero();
            input_combinations_weights_derivatives.setZero();
            output_combinations_weights_derivatives.setZero();
            state_combinations_weights_derivatives.setZero();

            previous_cell_states.setZero();

            cell_states_weights_derivatives.setZero();
            cell_states_biases_derivatives.setZero();

            hidden_states_weights_derivatives.setZero();

            cell_states_recurrent_weights_derivatives.setZero();
            hidden_states_recurrent_weights_derivatives.setZero();

        }
        else
        {
/*
            previous_cell_states.device(*thread_pool_device) = cell_states.chip(sample, 0);
*/
            // Forget combinations weights derivatives

            forget_combinations_weights_derivatives.device(*thread_pool_device) = hidden_states_weights_derivatives.contract(forget_recurrent_weights, A_B);

            multiply_rows(forget_combinations_weights_derivatives, current_forget_activations_derivatives);

            // Input weights derivatives

            input_combinations_weights_derivatives.device(*thread_pool_device) = hidden_states_weights_derivatives.contract(input_recurrent_weights, A_B);

            multiply_rows(input_combinations_weights_derivatives, current_input_activations_derivatives);

            // State combinations weights derivatives

            state_combinations_weights_derivatives.device(*thread_pool_device) = hidden_states_weights_derivatives.contract(state_recurrent_weights, A_B);

            multiply_rows(state_combinations_weights_derivatives, current_state_activations_derivatives);

            // Output combinations weights derivatives

            output_combinations_weights_derivatives.device(*thread_pool_device) = hidden_states_weights_derivatives.contract(output_recurrent_weights, A_B);

            multiply_rows(output_combinations_weights_derivatives, current_output_activations_derivatives);
        }

        raw_variable_index = 0;
        input_index = 0;
        neuron_index = 0;

        for(Index i = 0; i < parameters_number; i++)
        {
            const type current_input = current_inputs(input_index);

            forget_combinations_weights_derivatives(i, raw_variable_index) += current_input;
            input_combinations_weights_derivatives(i, raw_variable_index) += current_input;
            state_combinations_weights_derivatives(i, raw_variable_index) += current_input;
            output_combinations_weights_derivatives(i, raw_variable_index) += current_input;

            input_index++;

            if (input_index == inputs_number)
            {
                input_index = 0;
                raw_variable_index++;
            }
    
            const type previous_hidden_state_activation = previous_hidden_state_activations(neuron_index);

            forget_combinations_recurrent_weights_derivatives(i, raw_variable_index) += previous_hidden_state_activation;

            input_combinations_recurrent_weights_derivatives(i, raw_variable_index) += previous_hidden_state_activation;

            state_combinations_recurrent_weights_derivatives(i, raw_variable_index) += previous_hidden_state_activation;

            output_combinations_recurrent_weights_derivatives(i, raw_variable_index) += previous_hidden_state_activation;

            neuron_index++;

            if (neuron_index == neurons_number)
            {
                neuron_index = 0;
                raw_variable_index++;
            }
        }

        // Cell states weights derivatives

        cell_states_weights_derivatives.device(*thread_pool_device) += input_combinations_weights_derivatives;

        multiply_rows(cell_states_weights_derivatives, current_forget_activations);        

        multiply_rows(input_combinations_weights_derivatives, current_state_activations);

        multiply_rows(state_combinations_weights_derivatives, current_input_activations);

        cell_states_weights_derivatives.device(*thread_pool_device) += state_combinations_weights_derivatives;

        multiply_rows(forget_combinations_weights_derivatives, current_forget_activations_derivatives*previous_cell_states);

        cell_states_weights_derivatives.device(*thread_pool_device) += forget_combinations_weights_derivatives;
/*
        copy(/*execution::par,
            cell_states_weights_derivatives.data(),
            cell_states_weights_derivatives.data() + cell_states_weights_derivatives.size(),
            hidden_states_weights_derivatives.data());
*/
        // Hidden states weights derivatives

        multiply_rows(hidden_states_weights_derivatives, current_output_activations*current_hidden_states_derivatives);

        calculate_activations(current_cell_states, current_cell_states);

        multiply_rows(output_combinations_weights_derivatives, current_cell_states);

        hidden_states_weights_derivatives.device(*thread_pool_device) += output_combinations_weights_derivatives;

        // Forget weights derivatives

        forget_weights_derivatives.device(*thread_pool_device) += hidden_states_weights_derivatives.contract(current_deltas, A_B);
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

    type* gradient_data = gradient.data();

    // Biases

    copy(/*execution::par,*/ 
        long_short_term_memory_layer_back_propagation->forget_biases_derivatives.data(),
         long_short_term_memory_layer_back_propagation->forget_biases_derivatives.data() + neurons_number,
        gradient_data + index);

    copy(/*execution::par,*/ 
        long_short_term_memory_layer_back_propagation->input_biases_derivatives.data(),
         long_short_term_memory_layer_back_propagation->input_biases_derivatives.data() + neurons_number,
        gradient_data + index + neurons_number);

    copy(/*execution::par,*/ 
        long_short_term_memory_layer_back_propagation->state_biases_derivatives.data(),
         long_short_term_memory_layer_back_propagation->state_biases_derivatives.data() + neurons_number,
        gradient_data + index + 2*neurons_number);

    copy(/*execution::par,*/ 
        long_short_term_memory_layer_back_propagation->output_biases_derivatives.data(),
         long_short_term_memory_layer_back_propagation->output_biases_derivatives.data() + neurons_number,
        gradient_data + index + 3*neurons_number);

    // Weights

    copy(/*execution::par,*/ 
        long_short_term_memory_layer_back_propagation->forget_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->forget_weights_derivatives.data() + inputs_number*neurons_number,
        gradient_data + index + 4*neurons_number);

    copy(/*execution::par,*/ 
        long_short_term_memory_layer_back_propagation->input_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->input_weights_derivatives.data() + inputs_number*neurons_number,
        gradient_data + index + 4*neurons_number + inputs_number*neurons_number);

    copy(/*execution::par,*/
        long_short_term_memory_layer_back_propagation->state_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->state_weights_derivatives.data() + inputs_number*neurons_number,
        gradient_data + index + 4*neurons_number + 2*inputs_number*neurons_number);

    copy(/*execution::par,*/ 
        long_short_term_memory_layer_back_propagation->output_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->output_weights_derivatives.data() + inputs_number*neurons_number,
        gradient_data + index + 4*neurons_number + 3*inputs_number*neurons_number);

    // Recurrent weights

    copy(/*execution::par,*/ 
        long_short_term_memory_layer_back_propagation->forget_recurrent_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->forget_recurrent_weights_derivatives.data() + neurons_number*neurons_number,
        gradient_data + index + 4*neurons_number + 4*inputs_number*neurons_number);

    copy(/*execution::par,*/ 
        long_short_term_memory_layer_back_propagation->input_recurrent_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->input_recurrent_weights_derivatives.data() + neurons_number*neurons_number,
        gradient_data + index + 4*neurons_number + 4*inputs_number*neurons_number + neurons_number*neurons_number);

    copy(/*execution::par,*/ 
        long_short_term_memory_layer_back_propagation->state_recurrent_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->state_recurrent_weights_derivatives.data() + neurons_number*neurons_number,
        gradient_data + index + 4*neurons_number + 4*inputs_number*neurons_number + 2*neurons_number*neurons_number);

    copy(/*execution::par,*/ 
        long_short_term_memory_layer_back_propagation->output_recurrent_weights_derivatives.data(),
         long_short_term_memory_layer_back_propagation->output_recurrent_weights_derivatives.data() + neurons_number*neurons_number,
        gradient_data + index + 4*neurons_number + 4*inputs_number*neurons_number + 3*neurons_number*neurons_number);
}



/// Returns a string with the expression of the inputs-outputs relationship of the layer.
/// @param inputs_names Vector of strings with the name of the layer inputs.
/// @param outputs_names Vector of strings with the name of the layer outputs.

string LongShortTermMemoryLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    const Index neurons_number = get_neurons_number();

    const Index inputs_number = get_inputs_number();

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
            buffer << "cell_states_" << to_string(i) << "(t) = forget_gate_" << to_string(i) << " * cell_states_" << to_string(i) << "(t-1)+input_gate_" << to_string(i) << " * state_gate_" << to_string(i) << ";\n";
       }

       // Hidden state

       for(Index i = 0; i < neurons_number; i++)
       {
            buffer << "hidden_state_" << to_string(i) << "(t) = output_gate_" << to_string(i) << " * " << write_activation_function_expression() << "(cell_states_" << to_string(i) << ");\n";
       }

       // Output

       for(Index i = 0; i < neurons_number; i++)
       {
           buffer << outputs_names[i] << " = " << "hidden_state_" << to_string(i) << "(t);\n";
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

        throw runtime_error(buffer.str());
    }

    // Layer name

    const tinyxml2::XMLElement* layer_name_element = long_short_term_memory_layer_element->FirstChildElement("LayerName");

    if(!layer_name_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "LayerName element is nullptr.\n";

        throw runtime_error(buffer.str());
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

        throw runtime_error(buffer.str());
    }

    if(inputs_number_element->GetText())
    {
        set_inputs_number(Index(stoi(inputs_number_element->GetText())));
    }

    // Neurons number

    const tinyxml2::XMLElement* neurons_number_element = long_short_term_memory_layer_element->FirstChildElement("NeuronsNumber");

    if(!neurons_number_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "NeuronsNumber element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(neurons_number_element->GetText())
    {
        set_neurons_number(Index(stoi(neurons_number_element->GetText())));
    }

    // Time step

    const tinyxml2::XMLElement* time_step_element = long_short_term_memory_layer_element->FirstChildElement("TimeStep");

    if(!time_step_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "TimeStep element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(time_step_element->GetText())
    {
        set_timesteps(Index(stoi(time_step_element->GetText())));
    }

    // Activation function

    const tinyxml2::XMLElement* activation_function_element = long_short_term_memory_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
    {
        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "ActivationFunction element is nullptr.\n";

        throw runtime_error(buffer.str());
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

        throw runtime_error(buffer.str());
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

        throw runtime_error(buffer.str());
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


pair<type*, dimensions> LongShortTermMemoryLayerForwardPropagation::get_outputs_pair() const
{
    return pair<type*, dimensions>();
}


void LongShortTermMemoryLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    const Index inputs_number = layer->get_inputs_number();
    const Index neurons_number = layer->get_neurons_number();

    batch_samples_number = new_batch_samples_number;

    outputs.resize(batch_samples_number, neurons_number);

    previous_hidden_state_activations.resize(neurons_number);
    previous_cell_states.resize(neurons_number);

    current_inputs.resize(inputs_number);

    current_forget_combinations.resize(neurons_number);
    current_input_combinations.resize(neurons_number);
    current_state_combinations.resize(neurons_number);
    current_output_combinations.resize(neurons_number);

    current_forget_activations.resize(neurons_number);
    current_input_activations.resize(neurons_number);
    current_state_activations.resize(neurons_number);
    current_output_activations.resize(neurons_number);

    current_cell_states.resize(neurons_number);

    current_forget_activations_derivatives.resize(neurons_number);
    current_input_activations_derivatives.resize(neurons_number);
    current_state_activations_derivatives.resize(neurons_number);
    current_output_activations_derivatives.resize(neurons_number);
    current_hidden_states_derivatives.resize(neurons_number);

    forget_activations.resize(batch_samples_number, neurons_number);
    input_activations.resize(batch_samples_number, neurons_number);
    state_activations.resize(batch_samples_number, neurons_number);
    output_activations.resize(batch_samples_number, neurons_number);
    cell_states.resize(batch_samples_number, neurons_number);
    hidden_states.resize(batch_samples_number, neurons_number);

    forget_activations_derivatives.resize(batch_samples_number, neurons_number);
    input_activations_derivatives.resize(batch_samples_number, neurons_number);
    state_activations_derivatives.resize(batch_samples_number, neurons_number);
    output_activations_derivatives.resize(batch_samples_number, neurons_number);
    cell_states_derivatives.resize(batch_samples_number, neurons_number);
    hidden_states_derivatives.resize(batch_samples_number, neurons_number);

}


void LongShortTermMemoryLayerBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = layer->get_neurons_number();
    const Index inputs_number = layer->get_inputs_number();

    current_deltas.resize(neurons_number);

    forget_weights_derivatives.resize(inputs_number * neurons_number);
    input_weights_derivatives.resize(inputs_number * neurons_number);
    state_weights_derivatives.resize(inputs_number * neurons_number);
    output_weights_derivatives.resize(inputs_number * neurons_number);

    forget_recurrent_weights_derivatives.resize(neurons_number * neurons_number);
    input_recurrent_weights_derivatives.resize(neurons_number * neurons_number);
    state_recurrent_weights_derivatives.resize(neurons_number * neurons_number);
    output_recurrent_weights_derivatives.resize(neurons_number * neurons_number);

    forget_biases_derivatives.resize(neurons_number);
    input_biases_derivatives.resize(neurons_number);
    state_biases_derivatives.resize(neurons_number);
    output_biases_derivatives.resize(neurons_number);

    input_combinations_biases_derivatives.resize(neurons_number, neurons_number);
    forget_combinations_biases_derivatives.resize(neurons_number, neurons_number);
    state_combinations_biases_derivatives.resize(neurons_number, neurons_number);
    output_combinations_biases_derivatives.resize(neurons_number, neurons_number);

    hidden_states_biases_derivatives.resize(neurons_number, neurons_number);
    cell_states_biases_derivatives.resize(neurons_number, neurons_number);

    input_combinations_weights_derivatives.resize(inputs_number * neurons_number, neurons_number);
    forget_combinations_weights_derivatives.resize(inputs_number * neurons_number, neurons_number);
    state_combinations_weights_derivatives.resize(inputs_number * neurons_number, neurons_number);
    output_combinations_weights_derivatives.resize(inputs_number * neurons_number, neurons_number);

    hidden_states_weights_derivatives.resize(inputs_number * neurons_number, neurons_number);
    cell_states_weights_derivatives.resize(inputs_number * neurons_number, neurons_number);

    input_combinations_recurrent_weights_derivatives.resize(neurons_number * neurons_number, neurons_number);
    forget_combinations_recurrent_weights_derivatives.resize(neurons_number * neurons_number, neurons_number);
    state_combinations_recurrent_weights_derivatives.resize(neurons_number * neurons_number, neurons_number);
    output_combinations_recurrent_weights_derivatives.resize(neurons_number * neurons_number, neurons_number);

    hidden_states_recurrent_weights_derivatives.resize(neurons_number * neurons_number, neurons_number);
    cell_states_recurrent_weights_derivatives.resize(neurons_number * neurons_number, neurons_number);

    input_derivatives.resize(batch_samples_number, inputs_number);

    inputs_derivatives.resize(1);
    inputs_derivatives(0).first = input_derivatives.data();
    inputs_derivatives(0).second = { batch_samples_number, inputs_number };
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
