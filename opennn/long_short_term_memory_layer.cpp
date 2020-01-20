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

LongShortTermMemoryLayer::LongShortTermMemoryLayer(const int& new_inputs_number, const int& new_neurons_number) : Layer()
{
   set(new_inputs_number, new_neurons_number);

   layer_type = LongShortTermMemory;
}
 

/// Copy constructor. 
/// It creates a copy of an existing neuron layer object.
/// @param other_neuron_layer neuron layer object to be copied.

LongShortTermMemoryLayer::LongShortTermMemoryLayer(const LongShortTermMemoryLayer& other_neuron_layer) : Layer()
{
   set(other_neuron_layer);
}


/// Destructor.
/// This destructor does not delete any pointer.

LongShortTermMemoryLayer::~LongShortTermMemoryLayer()
{
}


/// Returns the number of inputs to the layer.

int LongShortTermMemoryLayer::get_inputs_number() const
{
    return input_weights.dimension(0);
}


/// Returns the size of the neurons vector.

int LongShortTermMemoryLayer::get_neurons_number() const
{
   return output_biases.size();
}


/// Returns the number of parameters (biases, weights, recurrent weights) of the layer.

int LongShortTermMemoryLayer::get_parameters_number() const
{
    int neurons_number = get_neurons_number();
    int inputs_number = get_inputs_number();

    return 4 * neurons_number * (1 + inputs_number + neurons_number);
}

/// Returns the forget biases from all the lstm in the layer.
/// The format is a vector of real values.
/// The size of this vector is the number of neurons in the layer.
///
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

/// Returns the weights from the lstm.
/// The format is a tensor of real values.
/// Dimension(0) is the number of inputs in the layer.
/// Dimension(1) is the number of neurons to the layer.
/// Dimension(2) is 4.

Tensor<type, 2> LongShortTermMemoryLayer::get_weights() const
{
    const int inputs_number = get_inputs_number();
    const int neurons_number = get_neurons_number();
/*
    Tensor<type, 2> weights(inputs_number,neurons_number,4);

    weights.set_matrix(0, forget_weights);
    weights.set_matrix(1, input_weights);
    weights.set_matrix(2, state_weights);
    weights.set_matrix(3, output_weights);

    return weights;
*/
    return Tensor<type, 2>();
}


/// Returns the recurrent weights from the lstm.
/// The format is a tensor of real values.
/// Dimension(0) is the number of inputs in the layer.
/// Dimension(1) is the number of neurons to the layer.
/// Dimension(2) is 4.

Tensor<type, 2> LongShortTermMemoryLayer::get_recurrent_weights() const
{
    const int neurons_number = get_neurons_number();
/*
    Tensor<type, 2> recurrent_weights(neurons_number, neurons_number, 4);

    recurrent_weights.set_matrix(0, forget_recurrent_weights);
    recurrent_weights.set_matrix(1, input_recurrent_weights);
    recurrent_weights.set_matrix(2, state_recurrent_weights);
    recurrent_weights.set_matrix(3, output_recurrent_weights);

    return recurrent_weights;
*/
    return Tensor<type, 2>();
}


/// Returns the biases from the lstm.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of columns is 4.

Tensor<type, 2> LongShortTermMemoryLayer::get_biases() const
{
    const int neurons_number = get_neurons_number();

    Tensor<type, 2> biases(neurons_number,4);
/*
    biases.set_column(0,forget_biases);
    biases.set_column(1,input_biases);
    biases.set_column(2, state_biases);
    biases.set_column(3, output_biases);
*/
    return biases;
}


/// Returns the number of timesteps.

int LongShortTermMemoryLayer::get_timesteps() const
{
    return timesteps;
}


/// Returns a single vector with all the layer parameters. 
/// The format is a vector of real values. 
/// The size is the number of parameters in the layer. 

Tensor<type, 1> LongShortTermMemoryLayer::get_parameters() const
{
/*
    Tensor<type, 2> weights = get_weights();
    Tensor<type, 2> recurrent_weights = get_recurrent_weights();
    Tensor<type, 2> biases = get_biases();

    return weights.to_vector().assemble(recurrent_weights.to_vector()).assemble(biases.to_vector());
*/
    return Tensor<type, 1>();
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
/// This can be: Logistic, HyperbolicTangent, Threshold, SymmetricThreshold, Linear, RectifiedLinear, ScaledExponentialLinear.

string LongShortTermMemoryLayer::write_activation_function() const
{
   switch(activation_function)
   {
      case Logistic:
      {
         return "Logistic";
      }

      case HyperbolicTangent:
      {
         return "HyperbolicTangent";
      }

      case Threshold:
      {
         return "Threshold";
      }

      case SymmetricThreshold:
      {
         return "SymmetricThreshold";
      }

      case Linear:
      {
         return "Linear";
      }

      case RectifiedLinear:
      {
         return "RectifiedLinear";
      }

      case ScaledExponentialLinear:
      {
         return "ScaledExponentialLinear";
      }

      case SoftPlus:
      {
         return "SoftPlus";
      }

      case SoftSign:
      {
         return "SoftSign";
      }

      case HardSigmoid:
      {
         return "HardSigmoid";
      }

      case ExponentialLinear:
      {
         return "ExponentialLinear";
      }
    }

    return string();
}



/// Returns a string with the name of the layer recurrent activation function.
/// This can be: Logistic, HyperbolicTangent, Threshold, SymmetricThreshold, Linear, RectifiedLinear, ScaledExponentialLinear.

string LongShortTermMemoryLayer::write_recurrent_activation_function() const
{
   switch(recurrent_activation_function)
   {
      case Logistic:
      {
         return "Logistic";
      }

      case HyperbolicTangent:
      {
         return "HyperbolicTangent";
      }

      case Threshold:
      {
         return "Threshold";
      }

      case SymmetricThreshold:
      {
         return "SymmetricThreshold";
      }

      case Linear:
      {
         return "Linear";
      }

      case RectifiedLinear:
      {
         return "RectifiedLinear";
      }

      case ScaledExponentialLinear:
      {
         return "ScaledExponentialLinear";
      }

      case SoftPlus:
      {
         return "SoftPlus";
      }

      case SoftSign:
      {
         return "SoftSign";
      }

      case HardSigmoid:
      {
         return "HardSigmoid";
      }

      case ExponentialLinear:
      {
         return "ExponentialLinear";
      }
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

void LongShortTermMemoryLayer::set(const int& new_inputs_number, const int& new_neurons_number)
{
/*

    input_biases.set(new_neurons_number);
    forget_biases.set(new_neurons_number);
    state_biases.set(new_neurons_number);
    output_biases.set(new_neurons_number);

    input_weights.set(new_inputs_number, new_neurons_number);
    forget_weights.set(new_inputs_number, new_neurons_number);
    state_weights.set(new_inputs_number, new_neurons_number);
    output_weights.set(new_inputs_number, new_neurons_number);

    input_recurrent_weights.set(new_neurons_number, new_neurons_number);
    forget_recurrent_weights.set(new_neurons_number, new_neurons_number);
    state_recurrent_weights.set(new_neurons_number, new_neurons_number);
    output_recurrent_weights.set(new_neurons_number, new_neurons_number);

    hidden_states.set(new_neurons_number, 0.0); // memory

    cell_states.set(new_neurons_number, 0.0); // carry

    set_default();
*/
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
   display = true;
}


/// Sets a new number of inputs in the layer. 
/// The new biases, weights and recurrent weights are initialized at random.
/// @param new_inputs_number Number of layer inputs.

void LongShortTermMemoryLayer::set_inputs_number(const int& new_inputs_number)
{
    const int neurons_number = get_neurons_number();
    set(new_inputs_number, neurons_number);
}


/// Sets a new size of inputs in the layer.
/// The new biases, weights and recurrent weights are initialized at random.
/// @param size dimensions of layer inputs.

void LongShortTermMemoryLayer::set_input_shape(const vector<int>& size)
{
    if(size.empty() || size.size() > 1)
    {
//        throw exception(string("EXCEPTION: The new size is incompatible."));
    }

    const int new_size = size[0];

    set_inputs_number(new_size);
}


/// Sets a new number neurons in the layer.
/// All the parameters are also initialized at random.
/// @param new_neurons_number New number of neurons in the layer.

void LongShortTermMemoryLayer::set_neurons_number(const int& new_neurons_number)
{    
    const int inputs_number = get_inputs_number();

    set(inputs_number, new_neurons_number);
}


/// Sets the forget biases of all lstm in the layer from a single vector.
/// @param new_forget_biases New set of forget biases in the layer.

void LongShortTermMemoryLayer::set_forget_biases(const Tensor<type, 1>& new_biases)
{
/*
    forget_biases.set(new_biases);
    */
}


/// Sets the input biases of all lstm in the layer from a single vector.
/// @param new_input_biases New set of input biases in the layer.
///
void LongShortTermMemoryLayer::set_input_biases(const Tensor<type, 1>& new_biases)
{
/*
    input_biases.set(new_biases);
*/
}



/// Sets the state biases of all lstm in the layer from a single vector.
/// @param new_state_biases New set of state biases in the layer.

void LongShortTermMemoryLayer::set_state_biases(const Tensor<type, 1>& new_biases)
{
/*
    state_biases.set(new_biases);
*/
}



/// Sets the output biases of all lstm in the layer from a single vector.
/// @param new_output_biases New set of output biases in the layer.

void LongShortTermMemoryLayer::set_output_biases(const Tensor<type, 1>& new_biases)
{
/*
    output_biases.set(new_biases);
*/
}


/// Sets the forget weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of inputs in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_forget_weights New set of forget weights in that layer.

void LongShortTermMemoryLayer::set_forget_weights(const Tensor<type, 2>& new_forget_weight)
{
/*
    forget_weights.set(new_forget_weight);
*/
}


/// Sets the input weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of inputs in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_input_weights New set of input weights in that layer.

void LongShortTermMemoryLayer::set_input_weights(const Tensor<type, 2>& new_input_weight)
{
/*
    input_weights.set(new_input_weight);
*/
}


/// Sets the state weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of inputs in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_state_weights New set of state weights in that layer.

void LongShortTermMemoryLayer::set_state_weights(const Tensor<type, 2>& new_state_weight)
{
/*
    state_weights.set(new_state_weight);
*/
}


/// Sets the output weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of inputs in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_output_weights New set of output weights in that layer.

void LongShortTermMemoryLayer::set_output_weights(const Tensor<type, 2>& new_output_weight)
{
/*
    output_weights.set(new_output_weight);
*/
}


/// Sets the forget recurrent weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of neurons in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_forget_recurrent_weights New set of forget recurrent weights in that layer.

void LongShortTermMemoryLayer::set_forget_recurrent_weights(const Tensor<type, 2>& new_forget_recurrent_weight)
{
/*
    forget_recurrent_weights.set(new_forget_recurrent_weight);
*/
}


/// Sets the input recurrent weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of neurons in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_input_recurrent_weights New set of input recurrent weights in that layer.


void LongShortTermMemoryLayer::set_input_recurrent_weights(const Tensor<type, 2>& new_input_recurrent_weight)
{
/*
    input_recurrent_weights.set(new_input_recurrent_weight);
*/
}


/// Sets the state recurrent weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of neurons in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_state_recurrent_weights New set of state recurrent weights in that layer.


void LongShortTermMemoryLayer::set_state_recurrent_weights(const Tensor<type, 2>& new_state_recurrent_weight)
{
/*
    state_recurrent_weights.set(new_state_recurrent_weight);
*/
}


/// Sets the output recurrent weights of this lstm layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of neurons in the corresponding layer.
/// The number of columns is the number of neurons to the corresponding layer.
/// @param new_output_recurrent_weights New set of output recurrent weights in that layer.

void LongShortTermMemoryLayer::set_output_recurrent_weights(const Tensor<type, 2>& new_output_recurrent_weight)
{
/*
    output_recurrent_weights.set(new_output_recurrent_weight);
*/
}


/// Sets the parameters of this layer. 
/// @param new_parameters Parameters vector for that layer. 

void LongShortTermMemoryLayer::set_parameters(const Tensor<type, 1>& new_parameters)
{
    const int neurons_number = get_neurons_number();
    const int inputs_number = get_inputs_number();

   #ifdef __OPENNN_DEBUG__ 

    const int parameters_number = get_parameters_number();

    const int new_parameters_size = new_parameters.size();

   if(new_parameters_size != parameters_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
             << "void set_parameters(const Tensor<type, 1>&) method.\n"
             << "Size of new parameters (" << new_parameters_size << ") must be equal to number of parameters (" << parameters_number << ").\n";

	  throw logic_error(buffer.str());
   }

   #endif
/*
   set_forget_weights(new_parameters.get_subvector(0, inputs_number * neurons_number - 1).to_matrix(inputs_number, neurons_number));
   set_input_weights(new_parameters.get_subvector(inputs_number * neurons_number, 2 * inputs_number * neurons_number - 1).to_matrix(inputs_number, neurons_number));
   set_state_weights(new_parameters.get_subvector(2 * inputs_number * neurons_number, 3 * inputs_number * neurons_number - 1).to_matrix(inputs_number, neurons_number));
   set_output_weights(new_parameters.get_subvector(3 * inputs_number * neurons_number, 4 * inputs_number * neurons_number - 1).to_matrix(inputs_number, neurons_number));

   set_forget_recurrent_weights(new_parameters.get_subvector(4 * inputs_number * neurons_number,  4 * inputs_number * neurons_number + neurons_number * neurons_number -1).to_matrix(neurons_number, neurons_number));
   set_input_recurrent_weights(new_parameters.get_subvector(4 * inputs_number * neurons_number + neurons_number * neurons_number, 4 * inputs_number * neurons_number + 2 * neurons_number * neurons_number - 1).to_matrix(neurons_number, neurons_number));
   set_state_recurrent_weights(new_parameters.get_subvector(4 * inputs_number * neurons_number + 2 * neurons_number * neurons_number , 4 * inputs_number * neurons_number + 3 * neurons_number * neurons_number  - 1).to_matrix(neurons_number, neurons_number));
   set_output_recurrent_weights(new_parameters.get_subvector(4 * inputs_number * neurons_number + 3 * neurons_number * neurons_number , 4 * inputs_number * neurons_number + 4 * neurons_number * neurons_number  - 1).to_matrix(neurons_number, neurons_number));

   set_forget_biases(new_parameters.get_subvector(4 * neurons_number * (inputs_number + neurons_number),  4 * neurons_number * (inputs_number + neurons_number) + neurons_number -1));
   set_input_biases(new_parameters.get_subvector(4 * neurons_number * (inputs_number + neurons_number) + neurons_number, 4 * neurons_number * (inputs_number + neurons_number) + 2 * neurons_number - 1));
   set_state_biases(new_parameters.get_subvector(4 * neurons_number * (inputs_number + neurons_number) + 2 * neurons_number , 4 * neurons_number * (inputs_number + neurons_number) + 3 * neurons_number - 1));
   set_output_biases(new_parameters.get_subvector(4 * neurons_number * (inputs_number + neurons_number) + 3 * neurons_number, 4 * neurons_number * (inputs_number + neurons_number + 1) - 1));
*/
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


/// Sets the timesteps of the layer from a int.
/// @param new_timesteps New set of timesteps in the layer.

void LongShortTermMemoryLayer::set_timesteps(const int & new_timesteps)
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

void LongShortTermMemoryLayer::initialize_biases(const double& value)
{
    forget_biases.setConstant(value);
    input_biases.setConstant(value);
    state_biases.setConstant(value);
    output_biases.setConstant(value);
}

/// Initializes the forget biases of all the neurons in the layer with a given value.
/// @param value Forget biases initialization value.

void LongShortTermMemoryLayer::initialize_forget_biases(const double& value)
{
    forget_biases.setConstant(value);
}


/// Initializes the input biases of all the neurons in the layer with a given value.
/// @param value Input biases initialization value.

void LongShortTermMemoryLayer::initialize_input_biases(const double& value)
{
     input_biases.setConstant(value);
}


/// Initializes the state biases of all the neurons in the layer with a given value.
/// @param value State biases initialization value.

void LongShortTermMemoryLayer::initialize_state_biases(const double& value)
{
     state_biases.setConstant(value);
}


/// Initializes the oputput biases of all the neurons in the layer with a given value.
/// @param value Output biases initialization value.

void LongShortTermMemoryLayer::initialize_output_biases(const double& value)
{
    output_biases.setConstant(value);
}

/// Initializes the weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Weights initialization value.

void LongShortTermMemoryLayer::initialize_weights(const double& value)
{
    forget_weights.setConstant(value);
    input_weights.setConstant(value);
    state_weights.setConstant(value);
    output_weights.setConstant(value);
}


/// Initializes the forget weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Forget weights initialization value.

void LongShortTermMemoryLayer::initialize_forget_weights(const double& value)
{
    forget_weights.setConstant(value);
}


/// Initializes the input weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Input weights initialization value.

void LongShortTermMemoryLayer::initialize_input_weights(const double& value)
{
     input_weights.setConstant(value);
}


/// Initializes the state weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value State weights initialization value.

void LongShortTermMemoryLayer::initialize_state_weights(const double& value)
{
     state_weights.setConstant(value);
}


/// Initializes the output weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Output weights initialization value.

void LongShortTermMemoryLayer::initialize_output_weights(const double & value)
{
    output_weights.setConstant(value);
}


/// Initializes the recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Recurrent weights initialization value.

void LongShortTermMemoryLayer::initialize_recurrent_weights(const double& value)
{
    forget_recurrent_weights.setConstant(value);
    input_recurrent_weights.setConstant(value);
    state_recurrent_weights.setConstant(value);
    output_recurrent_weights.setConstant(value);
}


/// Initializes the forget recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Forget recurrent weights initialization value.

void LongShortTermMemoryLayer::initialize_forget_recurrent_weights(const double& value)
{
    forget_recurrent_weights.setConstant(value);
}


/// Initializes the input recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Input recurrent weights initialization value.

void LongShortTermMemoryLayer::initialize_input_recurrent_weights(const double& value)
{
     input_recurrent_weights.setConstant(value);
}


/// Initializes the state recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value State recurrent weights initialization value.

void LongShortTermMemoryLayer::initialize_state_recurrent_weights(const double& value)
{
     state_recurrent_weights.setConstant(value);
}


/// Initializes the output recurrent weights of all the neurons in the layer of neurons neuron with a given value.
/// @param value Output recurrent weights initialization value.

void LongShortTermMemoryLayer::initialize_output_recurrent_weights(const double & value)
{
    output_recurrent_weights.setConstant(value);
}


/// Initializes hidden states of the layer with a given value.
/// @param value Hidden states initialization value.

void LongShortTermMemoryLayer::initialize_hidden_states(const double& value)
{
    hidden_states.setConstant(value);
}


/// Initializes cell states of the layer with a given value.
/// @param value Cell states initialization value.

void LongShortTermMemoryLayer::initialize_cell_states(const double& value)
{
    cell_states.setConstant(value);
}


void LongShortTermMemoryLayer::initialize_weights_Glorot(const double& minimum,const double& maximum)
{
    /*
    get_weights().setRandom(minimum, maximum);
    */
}


/// Initializes all the biases, weights and recurrent weights in the neural newtork with a given value.
/// @param value Parameters initialization value. 

void LongShortTermMemoryLayer::initialize_parameters(const double& value)
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

void LongShortTermMemoryLayer::randomize_parameters_uniform()
{
/*
     forget_biases.setRandom(-1.0, 1.0);
     input_biases.setRandom(-1.0, 1.0);
     state_biases.setRandom(-1.0, 1.0);
     output_biases.setRandom(-1.0, 1.0);

     forget_weights.setRandom(-1.0, 1.0);
     input_weights.setRandom(-1.0, 1.0);
     state_weights.setRandom(-1.0, 1.0);
     output_weights.setRandom(-1.0, 1.0);

     forget_recurrent_weights.setRandom(-1.0, 1.0);
     input_recurrent_weights.setRandom(-1.0, 1.0);
     state_recurrent_weights.setRandom(-1.0, 1.0);
     output_recurrent_weights.setRandom(-1.0, 1.0);
*/
}


/// Initializes all the biases, weights and recurrent weights in the layer of neurons at random with values
/// comprised between a minimum and a maximum values.
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

void LongShortTermMemoryLayer::randomize_parameters_uniform(const double& minimum, const double& maximum)
{
/*
     forget_biases.setRandom(minimum, maximum);
     input_biases.setRandom(minimum, maximum);
     state_biases.setRandom(minimum, maximum);
     output_biases.setRandom(minimum, maximum);

     forget_weights.setRandom(minimum, maximum);
     input_weights.setRandom(minimum, maximum);
     state_weights.setRandom(minimum, maximum);
     output_weights.setRandom(minimum, maximum);

     forget_recurrent_weights.setRandom(minimum, maximum);
     input_recurrent_weights.setRandom(minimum, maximum);
     state_recurrent_weights.setRandom(minimum, maximum);
     output_recurrent_weights.setRandom(minimum, maximum);
*/
}


/// Initializes all the biases, weights and recurrent weights in the layer of neurons with random random values
/// chosen from a normal distribution with a given mean and a given standard deviation.
/// @param mean Mean of normal distribution.
/// @param standard_deviation Standard deviation of normal distribution.


void LongShortTermMemoryLayer::randomize_parameters_normal(const double& mean, const double& standard_deviation)
{
/*
    forget_biases.setRandom(mean, standard_deviation);
    input_biases.setRandom(mean, standard_deviation);
    state_biases.setRandom(mean, standard_deviation);
    output_biases.setRandom(mean, standard_deviation);

    forget_weights.setRandom(mean, standard_deviation);
    input_weights.setRandom(mean, standard_deviation);
    state_weights.setRandom(mean, standard_deviation);
    output_weights.setRandom(mean, standard_deviation);

    forget_recurrent_weights.setRandom(mean, standard_deviation);
    input_recurrent_weights.setRandom(mean, standard_deviation);
    state_recurrent_weights.setRandom(mean, standard_deviation);
    output_recurrent_weights.setRandom(mean, standard_deviation);
*/
}


/// Calculates the norm of a layer parameters vector. 

double LongShortTermMemoryLayer::calculate_parameters_norm() const
{
/*
    return(l2_norm(get_parameters()));
*/
    return 0.0;
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_forget_combinations(const Tensor<type, 1>& inputs) const
{

    #ifdef __OPENNN_DEBUG__

    const int inputs_number = get_inputs_number();

    if(inputs.size() != inputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
              << "Tensor<type, 1> calculate_forget_combinations(const Tensor<type, 1>&) const method.\n"
              << "Size of layer inputs (" << inputs.size() << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif
/*
    return dot(inputs, forget_weights) + forget_biases + dot(hidden_states, forget_recurrent_weights);
*/

    return Tensor<type, 1>();
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_input_combinations(const Tensor<type, 1>& inputs) const
{
    #ifdef __OPENNN_DEBUG__

    const int inputs_number = get_inputs_number();

    if(inputs.size() != inputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
              << "Tensor<type, 1> calculate_input_combinations(const Tensor<type, 1>&) const method.\n"
              << "Size of layer inputs (" << inputs.size() << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif
/*
    return dot(inputs, input_weights) + input_biases + dot(hidden_states, input_recurrent_weights);
*/
    return Tensor<type, 1>();

}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_state_combinations(const Tensor<type, 1>& inputs) const
{
    #ifdef __OPENNN_DEBUG__

    const int inputs_number = get_inputs_number();

    if(inputs.size() != inputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
              << "Tensor<type, 1> calculate_state_combinations(const Tensor<type, 1>&) const method.\n"
              << "Size of layer inputs (" << inputs.size() << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif
/*
    return dot(inputs, state_weights) + state_biases + dot(hidden_states, state_recurrent_weights);
*/
    return Tensor<type, 1>();

}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_output_combinations(const Tensor<type, 1>& inputs) const
{

    #ifdef __OPENNN_DEBUG__

    const int inputs_number = get_inputs_number();

    if(inputs.size() != inputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
              << "Tensor<type, 1> calculate_output_combinations(const Tensor<type, 1>&) const method.\n"
              << "Size of layer inputs (" << inputs.size() << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif
/*
    return dot(inputs, output_weights) + output_biases + dot(hidden_states, output_recurrent_weights);
*/
    return Tensor<type, 1>();

}


Tensor<type, 2> LongShortTermMemoryLayer::calculate_activations_states(const Tensor<type, 2>& inputs)
{
    /*
    const int instances_number = inputs.dimension(0);
    const int neurons_number = get_neurons_number();

    // forget activations, input activations, state activations, output activations, state, hidden state
    Tensor<type, 2> activations_states(instances_number,neurons_number,6);

    int forget_activations_index = 0;
    int input_activations_index = instances_number*neurons_number;
    int state_activations_index = 2*instances_number*neurons_number;
    int output_activations_index = 3*instances_number*neurons_number;
    int states_index = 4*instances_number*neurons_number;
    int hidden_states_index = 5*instances_number*neurons_number;

    for(int i = 0; i < instances_number; i++)
    {
        if(i%timesteps == 0)
        {
            hidden_states.setZero();
            cell_states.setZero();
        }

        const Tensor<type, 1> current_inputs = inputs.get_row(i);

        const Tensor<type, 1> forget_combinations = calculate_forget_combinations(current_inputs);
        const Tensor<type, 1> forget_activations = calculate_recurrent_activations(forget_combinations);

        const Tensor<type, 1> input_combinations = calculate_input_combinations(current_inputs);
        const Tensor<type, 1> input_activations = calculate_recurrent_activations(input_combinations);

        const Tensor<type, 1> state_combinations = calculate_state_combinations(current_inputs);
        const Tensor<type, 1> state_activations = calculate_activations(state_combinations);

        const Tensor<type, 1> output_combinations = calculate_output_combinations(current_inputs);
        const Tensor<type, 1> output_activations = calculate_recurrent_activations(output_combinations);

        cell_states = forget_activations * cell_states + input_activations * state_activations;
        hidden_states = output_activations * calculate_activations(cell_states);

        activations_states.embed(forget_activations_index, forget_activations);
        activations_states.embed(input_activations_index, input_activations);
        activations_states.embed(state_activations_index, state_activations);
        activations_states.embed(output_activations_index, output_activations);
        activations_states.embed(states_index, cell_states);
        activations_states.embed(hidden_states_index, hidden_states);

        forget_activations_index ++; //= neurons_number;
        input_activations_index ++; //= neurons_number;
        state_activations_index ++; //= neurons_number;
        output_activations_index ++; //= neurons_number;
        states_index ++; //= neurons_number;
        hidden_states_index ++; //= neurons_number;
    }

    return activations_states;
    */
    return Tensor<type, 2>();

}


Tensor<type, 2> LongShortTermMemoryLayer::calculate_activations(const Tensor<type, 2>& combinations) const
{
    #ifdef __OPENNN_DEBUG__

    const int neurons_number = get_neurons_number();

    const int combinations_columns_number = combinations.dimension(1);

    if(combinations_columns_number != neurons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
              << "Tensor<type, 2> calculate_activations(const Tensor<type, 2>&) const method.\n"
              << "Number of columns("<< combinations_columns_number <<") of combinations must be equal to number of neurons("<<neurons_number<<").\n";

       throw logic_error(buffer.str());
    }

    #endif
/*
    switch(activation_function)
    {
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
    }
*/
    return Tensor<type, 2>();
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_activations(const Tensor<type, 1>& combinations) const
{
    #ifdef __OPENNN_DEBUG__

    const int neurons_number = get_neurons_number();

    const int combinations_columns_number = combinations.size();

    if(combinations_columns_number != neurons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
              << "Tensor<type, 2> calculate_activations(const Tensor<type, 1>&) const method.\n"
              << "Size of combinations must be equal to number of neurons.\n";

       throw logic_error(buffer.str());
    }

    #endif
/*
    switch(activation_function)
    {
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
    }
*/
    return Tensor<type, 1>();
}


Tensor<type, 2> LongShortTermMemoryLayer::calculate_recurrent_activations(const Tensor<type, 2>& combinations) const
{
    /*
    #ifdef __OPENNN_DEBUG__

    const int neurons_number = get_neurons_number();

    const int combinations_columns_number = combinations.get_dimension(2);

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
    }
*/
    return Tensor<type, 2>();
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_recurrent_activations(const Tensor<type, 1>& combinations) const
{
    #ifdef __OPENNN_DEBUG__

    const int neurons_number = get_neurons_number();

    const int combinations_columns_number = combinations.size();

    if(combinations_columns_number != neurons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
              << "Tensor<type, 2> calculate_activations(const Tensor<type, 2>&) const method.\n"
              << "Size of combinations must be equal to number of neurons.\n";

       throw logic_error(buffer.str());
    }

    #endif
/*
    switch(recurrent_activation_function)
    {
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
    }
*/
    return Tensor<type, 1>();
}


Tensor<type, 2> LongShortTermMemoryLayer::calculate_activations_derivatives(const Tensor<type, 2>& combinations) const
{
    #ifdef __OPENNN_DEBUG__

    const int neurons_number = get_neurons_number();

    const int combinations_columns_number = combinations.dimension(1);

    if(combinations_columns_number != neurons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
              << "Tensor<type, 2> calculate_activations_derivatives(const Tensor<type, 2>&) const method.\n"
              << "Number of columns("<< combinations_columns_number <<") of combinations must be equal to number of neurons("<<neurons_number<<").\n";

       throw logic_error(buffer.str());
    }

    #endif
/*
    switch(activation_function)
    {
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
    }
*/
    return Tensor<type, 2>();
}




Tensor<type, 1> LongShortTermMemoryLayer::calculate_activations_derivatives(const Tensor<type, 1>& combination) const
{
    #ifdef __OPENNN_DEBUG__

    const int neurons_number = get_neurons_number();

    const int combinations_columns_number = combination.size();

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
        case Linear: return linear_derivatives(combination);

        case Logistic: return logistic_derivatives(combination);

        case HyperbolicTangent: return hyperbolic_tangent_derivatives(combination);

        case Threshold: return threshold_derivatives(combination);

        case SymmetricThreshold: return symmetric_threshold_derivatives(combination);

        case RectifiedLinear: return rectified_linear_derivatives(combination);

        case ScaledExponentialLinear: return scaled_exponential_linear_derivatives(combination);

        case SoftPlus: return soft_plus_derivatives(combination);

        case SoftSign: return soft_sign_derivatives(combination);

        case HardSigmoid: return hard_sigmoid_derivatives(combination);

        case ExponentialLinear: return exponential_linear_derivatives(combination);
    }

    return Tensor<type, 1>();
}

Tensor<type, 1> LongShortTermMemoryLayer::calculate_recurrent_activations_derivatives(const Tensor<type, 1>& combination) const
{
    #ifdef __OPENNN_DEBUG__

    const int neurons_number = get_neurons_number();

    const int combinations_columns_number = combination.size();

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
        case Linear: return linear_derivatives(combination);

        case Logistic: return logistic_derivatives(combination);

        case HyperbolicTangent: return hyperbolic_tangent_derivatives(combination);

        case Threshold: return threshold_derivatives(combination);

        case SymmetricThreshold: return symmetric_threshold_derivatives(combination);

        case RectifiedLinear: return rectified_linear_derivatives(combination);

        case ScaledExponentialLinear: return scaled_exponential_linear_derivatives(combination);

        case SoftPlus: return soft_plus_derivatives(combination);

        case SoftSign: return soft_sign_derivatives(combination);

        case HardSigmoid: return hard_sigmoid_derivatives(combination);

        case ExponentialLinear: return exponential_linear_derivatives(combination);
    }

    return Tensor<type, 1>();
}


void LongShortTermMemoryLayer::update_cell_states(const Tensor<type, 1>& inputs)
{   
    #ifdef __OPENNN_DEBUG__

    const int inputs_number = get_inputs_number();

    if(inputs.size() != inputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
              << "Void update_cell_states(const Tensor<type, 1>&) const method.\n"
              << "Size of layer inputs (" << inputs.size() << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    const Tensor<type, 1> forget_combinations = calculate_forget_combinations(inputs);
    Tensor<type, 1> forget_activations = calculate_recurrent_activations(forget_combinations);

    const Tensor<type, 1> input_combinations = calculate_input_combinations(inputs);
    const Tensor<type, 1> input_activations = calculate_recurrent_activations(input_combinations);

    const Tensor<type, 1> state_combinations = calculate_state_combinations(inputs);
    const Tensor<type, 1> state_activations = calculate_recurrent_activations(state_combinations);

    cell_states = forget_activations * cell_states + input_activations*state_activations;
}


void LongShortTermMemoryLayer::update_hidden_states(const Tensor<type, 1>& inputs)
{
    #ifdef __OPENNN_DEBUG__

    const int inputs_number = get_inputs_number();

    if(inputs.size() != inputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
              << "Void update_hidden_states(const Tensor<type, 1>&) const method.\n"
              << "Size of layer inputs (" << inputs.size() << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    const Tensor<type, 1> output_combinations = calculate_output_combinations(inputs);

    const Tensor<type, 1> output_activations = calculate_activations(output_combinations);

    hidden_states = output_activations*calculate_activations(cell_states);
}


Tensor<type, 2> LongShortTermMemoryLayer::calculate_outputs(const Tensor<type, 2>& inputs)
{
    /*
    #ifdef __OPENNN_DEBUG__

    const int inputs_number = get_inputs_number();

    const int inputs_columns_number = inputs.dimension(1);

    if(inputs_columns_number != inputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
              << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
              << "Number of columns ("<<inputs_columns_number<<") of inputs matrix must be equal to number of inputs ("<<inputs_number<<").\n";

       throw logic_error(buffer.str());
    }
    #endif

    const int instances_number = inputs.dimension(0);

    const int neurons_number = get_neurons_number();

    Tensor<type, 2> outputs(vector<int>({instances_number, neurons_number}));

    Tensor<type, 1> forget_combinations;
    Tensor<type, 1> forget_activations;
    Tensor<type, 1> input_combinations;
    Tensor<type, 1> input_activations;
    Tensor<type, 1> state_combinations;
    Tensor<type, 1> state_activations;
    Tensor<type, 1> output_combinations;
    Tensor<type, 1> output_activations;

    for(int i = 0; i < instances_number; i++)
    {
        if(i%timesteps == 0)
        {
            hidden_states.setZero();
            cell_states.setZero();
        }

        const Tensor<type, 1> current_inputs = inputs.get_row(i);

#pragma omp parallel
        {
            forget_combinations = calculate_forget_combinations(current_inputs);
            forget_activations = calculate_recurrent_activations(forget_combinations);

            input_combinations = calculate_input_combinations(current_inputs);
            input_activations  = calculate_recurrent_activations(input_combinations);

            state_combinations = calculate_state_combinations(current_inputs);
            state_activations  = calculate_activations(state_combinations);

            output_combinations = calculate_output_combinations(current_inputs);
            output_activations  = calculate_recurrent_activations(output_combinations);
        }

        cell_states = forget_activations * cell_states + input_activations * state_activations;
        hidden_states = output_activations * calculate_activations(cell_states);

        outputs.set_row(i, hidden_states);
      }

    return outputs;
*/
    return Tensor<type, 2>();
}


Tensor<type, 2> LongShortTermMemoryLayer::calculate_outputs(const Tensor<type, 2>& inputs, const Tensor<type, 1>& parameters)
{
/*
    const int inputs_number = get_inputs_number();

    #ifdef __OPENNN_DEBUG__

    const int inputs_columns_number = inputs.dimension(1);

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

        buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
               << "Tensor<type, 1> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 1>&) const method.\n"
               << "Parameters size("<<parameters.size() <<")must be equal to number of parameters("<< get_parameters_number() << ").\n";

        throw logic_error(buffer.str());
    }
    #endif

    const int instances_number = inputs.dimension(0);
    const int neurons_number = get_neurons_number();

    const Tensor<type, 2> new_forget_weights = parameters.get_subvector(0, inputs_number * neurons_number - 1).to_matrix(inputs_number, neurons_number);
    const Tensor<type, 2> new_input_weights = parameters.get_subvector(inputs_number * neurons_number, 2 * inputs_number * neurons_number - 1).to_matrix(inputs_number, neurons_number);
    const Tensor<type, 2> new_state_weights = parameters.get_subvector(2 * inputs_number * neurons_number, 3 * inputs_number * neurons_number - 1).to_matrix(inputs_number, neurons_number);
    const Tensor<type, 2> new_output_weights = parameters.get_subvector(3 * inputs_number * neurons_number, 4 * inputs_number * neurons_number - 1).to_matrix(inputs_number, neurons_number);

    const Tensor<type, 2> new_forget_recurrent_weights = parameters.get_subvector(4 * inputs_number * neurons_number,  4 * inputs_number * neurons_number + neurons_number * neurons_number -1).to_matrix(neurons_number, neurons_number);
    const Tensor<type, 2> new_input_recurrent_weights = parameters.get_subvector(4 * inputs_number * neurons_number + neurons_number * neurons_number, 4 * inputs_number * neurons_number + 2 * neurons_number * neurons_number - 1).to_matrix(neurons_number, neurons_number);
    const Tensor<type, 2> new_state_recurrent_weights = parameters.get_subvector(4 * inputs_number * neurons_number + 2 * neurons_number * neurons_number , 4 * inputs_number * neurons_number + 3 * neurons_number * neurons_number  - 1).to_matrix(neurons_number, neurons_number);
    const Tensor<type, 2> new_output_recurrent_weights = parameters.get_subvector(4 * inputs_number * neurons_number + 3 * neurons_number * neurons_number , 4 * inputs_number * neurons_number + 4 * neurons_number * neurons_number  - 1).to_matrix(neurons_number, neurons_number);

    const Tensor<type, 1> new_forget_biases = parameters.get_subvector(4 * neurons_number * (inputs_number + neurons_number),  4 * neurons_number * (inputs_number + neurons_number) + neurons_number -1);
    const Tensor<type, 1> new_input_biases = parameters.get_subvector(4 * neurons_number * (inputs_number + neurons_number) + neurons_number, 4 * neurons_number * (inputs_number + neurons_number) + 2 * neurons_number - 1);
    const Tensor<type, 1> new_state_biases = parameters.get_subvector(4 * neurons_number * (inputs_number + neurons_number) + 2 * neurons_number , 4 * neurons_number * (inputs_number + neurons_number) + 3 * neurons_number - 1);
    const Tensor<type, 1> new_output_biases = parameters.get_subvector(4 * neurons_number * (inputs_number + neurons_number) + 3 * neurons_number, 4 * neurons_number * (inputs_number + neurons_number + 1) - 1);

    Tensor<type, 2> outputs(vector<int>({instances_number, neurons_number}));

    Tensor<type, 1> forget_combinations;
    Tensor<type, 1> forget_activations;

    Tensor<type, 1> input_combinations;
    Tensor<type, 1> input_activations;

    Tensor<type, 1> state_combinations;
    Tensor<type, 1> state_activations;

    Tensor<type, 1> output_combinations;
    Tensor<type, 1> output_activations;

    for(int i = 0; i < instances_number; i++)
    {
        if(i%timesteps == 0)
        {
            hidden_states.setZero();
            cell_states.setZero();
        }

        const Tensor<type, 1> current_inputs = inputs.get_row(i);

#pragma omp parallel
        {
            forget_combinations = dot(current_inputs, new_forget_weights) + new_forget_biases + dot(hidden_states, new_forget_recurrent_weights);
            forget_activations = calculate_recurrent_activations(forget_combinations);

            input_combinations = dot(current_inputs, new_input_weights) + new_input_biases + dot(hidden_states, new_input_recurrent_weights);
            input_activations = calculate_recurrent_activations(input_combinations);

            state_combinations = dot(current_inputs, new_state_weights) + new_state_biases + dot(hidden_states, new_state_recurrent_weights);
            state_activations = calculate_activations(state_combinations);

            output_combinations = dot(current_inputs, new_output_weights) + new_output_biases + dot(hidden_states, new_output_recurrent_weights);
            output_activations = calculate_recurrent_activations(output_combinations);
        }

        cell_states = forget_activations * cell_states + input_activations * state_activations;
        hidden_states = output_activations * calculate_activations(cell_states);

        outputs.set_row(i, hidden_states);
      }

    return outputs;
*/
    return Tensor<type, 2>();
}


Tensor<type, 2> LongShortTermMemoryLayer::calculate_outputs(const Tensor<type, 2>& inputs, const Tensor<type, 2>& new_biases, const Tensor<type, 2>& new_weights, const Tensor<type, 2>& new_recurrent_weights)
{
    const int neurons_number = get_neurons_number();

    #ifdef __OPENNN_DEBUG__

    const int inputs_number = get_inputs_number();

    const int inputs_columns_number = inputs.dimension(1);

    if(inputs_columns_number != inputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: RecurrentLayer class.\n"
              << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
              << "Number of columns("<<inputs_columns_number<<") of inputs matrix must be equal to number of inputs("<<inputs_number<<").\n";

       throw logic_error(buffer.str());
    }

    if(new_biases.dimension(0) != neurons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: RecurrentLayer class.\n"
              << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&).\n"
              << "Number of rows of biases must be equal to number of neurons.\n";

       throw logic_error(buffer.str());
    }

    if(new_biases.dimension(1) != 4)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: RecurrentLayer class.\n"
              << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&).\n"
              << "Number of columns("<< new_biases.dimension(1) <<") of biases must be equal to number 4 .\n";

       throw logic_error(buffer.str());
    }

    if(new_weights.dimension(0) != inputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: RecurrentLayer class.\n"
              << "Tensor<type, 1> calculate_combinations(const Tensor<type, 1>&, const Tensor<type, 1>&, const Tensor<type, 2>& , const Tensor<type, 2>&) const method.\n"
              << "Rows number of weight  (" << new_weights.dimension(0) << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

       throw logic_error(buffer.str());
    }


    if(new_weights.dimension(1) != neurons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: RecurrentLayer class.\n"
              << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&).\n"
              << "Columns number of weight  (" << new_weights.dimension(1) << ") must be equal to number of neurons number (" << neurons_number << ").\n";

       throw logic_error(buffer.str());
    }


    if(new_recurrent_weights.dimension(0) != neurons_number  || new_recurrent_weights.dimension(1) != neurons_number )
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: RecurrentLayer class.\n"
              << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&, const Tensor<type, 2>&).\n"
              << "Columns number of recurrent weight  (" <<new_recurrent_weights.dimension(1)  << ") must be equal to number of neurons number (" << neurons_number << ").\n"
              << "Rows number of recurrent weight  (" <<new_recurrent_weights.dimension(0)  << ") must be equal to number of neurons number (" << neurons_number << ").\n";

       throw logic_error(buffer.str());
     }
     #endif
/*
     const Tensor<type, 2> new_forget_weights(new_weights.get_matrix(0));
     const Tensor<type, 2> new_input_weights(new_weights.get_matrix(1));
     const Tensor<type, 2> new_state_weights(new_weights.get_matrix(2));
     const Tensor<type, 2> new_output_weights(new_weights.get_matrix(3));

     const Tensor<type, 2> new_forget_recurrent_weights(new_recurrent_weights.get_matrix(0));
     const Tensor<type, 2> new_input_recurrent_weights(new_recurrent_weights.get_matrix(1));
     const Tensor<type, 2> new_state_recurrent_weights(new_recurrent_weights.get_matrix(2));
     const Tensor<type, 2> new_output_recurrent_weights(new_recurrent_weights.get_matrix(3));

     const Tensor<type, 1> new_forget_biases(new_biases.get_column(0));
     const Tensor<type, 1> new_input_biases(new_biases.get_column(1));
     const Tensor<type, 1> new_state_biases(new_biases.get_column(2));
     const Tensor<type, 1> new_output_biases(new_biases.get_column(3));

     const int instances_number = inputs.dimension(0);

     Tensor<type, 2> outputs(vector<int>({instances_number, neurons_number}));

     Tensor<type, 1> forget_combinations;
     Tensor<type, 1> forget_activations;

     Tensor<type, 1> input_combinations;
     Tensor<type, 1> input_activations;

     Tensor<type, 1> state_combinations;
     Tensor<type, 1> state_activations;

     Tensor<type, 1> output_combinations;
     Tensor<type, 1> output_activations;

     for(int i = 0; i < instances_number; i++)
     {
         if(i%timesteps == 0)
         {
             hidden_states.setZero();
             cell_states.setZero();
         }

         const Tensor<type, 1> current_inputs = inputs.get_row(i);

 #pragma omp parallel
         {
             forget_combinations = dot(current_inputs, new_forget_weights) + new_forget_biases + dot(hidden_states, new_forget_recurrent_weights);
             forget_activations = calculate_recurrent_activations(forget_combinations);

             input_combinations = dot(current_inputs, new_input_weights) + new_input_biases + dot(hidden_states, new_input_recurrent_weights);
             input_activations = calculate_recurrent_activations(input_combinations);

             state_combinations = dot(current_inputs, new_state_weights) + new_state_biases + dot(hidden_states, new_state_recurrent_weights);
             state_activations = calculate_activations(state_combinations);

             output_combinations = dot(current_inputs, new_output_weights) + new_output_biases + dot(hidden_states, new_output_recurrent_weights);
             output_activations = calculate_recurrent_activations(output_combinations);
         }


         cell_states = forget_activations * cell_states + input_activations * state_activations;
         hidden_states = output_activations * calculate_activations(cell_states);

         outputs.set_row(i, hidden_states);
     }

     return outputs;
*/
        return Tensor<type, 2>();
}


Layer::ForwardPropagation LongShortTermMemoryLayer::calculate_forward_propagation(const Tensor<type, 2>& inputs)
{
    /*
    const int instances_number = inputs.dimension(0);
    const int neurons_number = get_neurons_number();

    Tensor<type, 2> activations(instances_number,neurons_number);

    // forget, input, state, output and tanh(cell_states) derivatives
    Tensor<type, 2> activations_derivatives(instances_number,neurons_number, 5);
    activations_derivatives.setZero();

    int forget_activations_index = 0;
    int input_activations_index = instances_number*neurons_number;
    int state_activations_index = 2*instances_number*neurons_number;
    int output_activations_index = 3*instances_number*neurons_number;
    int hidden_states_index = 4*instances_number*neurons_number;

    Tensor<type, 1> forget_combinations;
    Tensor<type, 1> forget_activations;
    Tensor<type, 1> forget_activations_derivatives;

    Tensor<type, 1> input_combinations;
    Tensor<type, 1> input_activations;
    Tensor<type, 1> input_activations_derivatives;

    Tensor<type, 1> state_combinations;
    Tensor<type, 1> state_activations;
    Tensor<type, 1> state_activations_derivatives;

    Tensor<type, 1> output_combinations;
    Tensor<type, 1> output_activations;
    Tensor<type, 1> output_activations_derivatives;

    for(int i = 0; i < instances_number; i++)
    {
        if(i%timesteps == 0)
        {
            hidden_states.setZero();
            cell_states.setZero();
        }

        const Tensor<type, 1> current_inputs = inputs.get_row(i);

 #pragma omp parallel
        {
            forget_combinations = calculate_forget_combinations(current_inputs);
            forget_activations = calculate_recurrent_activations(forget_combinations);
            forget_activations_derivatives = calculate_recurrent_activations_derivatives(forget_combinations);

            input_combinations = calculate_input_combinations(current_inputs);
            input_activations = calculate_recurrent_activations(input_combinations);
            input_activations_derivatives = calculate_recurrent_activations_derivatives(input_combinations);

            state_combinations = calculate_state_combinations(current_inputs);
            state_activations = calculate_activations(state_combinations);
            state_activations_derivatives = calculate_activations_derivatives(state_combinations);

            output_combinations = calculate_output_combinations(current_inputs);
            output_activations = calculate_recurrent_activations(output_combinations);
            output_activations_derivatives = calculate_recurrent_activations_derivatives(output_combinations);
        }

        cell_states = forget_activations * cell_states + input_activations * state_activations;
        hidden_states = output_activations * calculate_activations(cell_states);
        const Tensor<type, 1> hidden_states_derivatives = calculate_activations_derivatives(cell_states);

        activations.set_row(i,hidden_states);

        activations_derivatives.embed(forget_activations_index, forget_activations_derivatives);
        activations_derivatives.embed(input_activations_index, input_activations_derivatives);
        activations_derivatives.embed(state_activations_index, state_activations_derivatives);
        activations_derivatives.embed(output_activations_index, output_activations_derivatives);
        activations_derivatives.embed(hidden_states_index, hidden_states_derivatives);

        forget_activations_index++;
        input_activations_index++;
        state_activations_index++;
        output_activations_index++;
        hidden_states_index++;
    }

    Layer::ForwardPropagation layers;

    layers.activations = activations;
    layers.activations_derivatives = activations_derivatives;

    return layers;
    */
    return Layer::ForwardPropagation();
}


Tensor<type, 2> LongShortTermMemoryLayer::calculate_output_delta(const Tensor<type, 2> &, const Tensor<type, 2> & output_gradient) const
{
    return output_gradient;
}


Tensor<type, 2> LongShortTermMemoryLayer::calculate_hidden_delta(Layer* next_layer_pointer,
                                                                const Tensor<type, 2>&,
                                                                const Tensor<type, 2>&,
                                                                const Tensor<type, 2>& next_layer_delta) const
{
    /*
    const Type layer_type = next_layer_pointer->get_type();

    MatrixXd synaptic_weights_transpose;

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

    return dot(next_layer_delta, synaptic_weights_transpose);
    */
    return Tensor<type, 2>();
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_error_gradient(const Tensor<type, 2> &  inputs,
                                                                  const Layer::ForwardPropagation& layers,
                                                                  const Tensor<type, 2> & deltas)
{
    /*
    const int parameters_number = get_parameters_number();

    const int neurons_number = get_neurons_number();
    const int inputs_number = get_inputs_number();

    // For each layer

    const int weights_number = inputs_number*neurons_number;
    const int recurrent_weights_number = neurons_number*neurons_number;
    const int biases_number = neurons_number;

    Tensor<type, 1> error_gradient(parameters_number, 0.0);

    const Tensor<type, 2> activations_states = calculate_activations_states(inputs);


    #pragma omp parallel
    {
        // Forget weights

        error_gradient.embed(0, calculate_forget_weights_error_gradient(inputs,layers,deltas,activations_states));

        // Input weights

        error_gradient.embed(weights_number, calculate_input_weights_error_gradient(inputs,layers,deltas,activations_states));

        // State weights

        error_gradient.embed(2*weights_number, calculate_state_weights_error_gradient(inputs,layers,deltas,activations_states));

        // Output weights

        error_gradient.embed(3*weights_number, calculate_output_weights_error_gradient(inputs,layers,deltas,activations_states));

        // Forget recurrent weights

        error_gradient.embed(4*weights_number, calculate_forget_recurrent_weights_error_gradient(inputs,layers,deltas,activations_states));

        // Input recurrent weights

        error_gradient.embed(4*weights_number+recurrent_weights_number, calculate_input_recurrent_weights_error_gradient(inputs,layers,deltas,activations_states));

        // State recurrent weights

        error_gradient.embed(4*weights_number+2*recurrent_weights_number, calculate_state_recurrent_weights_error_gradient(inputs,layers,deltas,activations_states));

        // Output recurrent weights

        error_gradient.embed(4*weights_number+3*recurrent_weights_number, calculate_output_recurrent_weights_error_gradient(inputs,layers,deltas,activations_states));

        // Forget biases

        error_gradient.embed(4*weights_number+4*recurrent_weights_number, calculate_forget_biases_error_gradient(inputs,layers,deltas,activations_states));

        // Input biases

        error_gradient.embed(4*weights_number+4*recurrent_weights_number+biases_number, calculate_input_biases_error_gradient(inputs,layers,deltas,activations_states));

        // State biases

        error_gradient.embed(4*weights_number+4*recurrent_weights_number+2*biases_number, calculate_state_biases_error_gradient(inputs,layers,deltas,activations_states));

        // Output biases

        error_gradient.embed(4*weights_number+4*recurrent_weights_number+3*biases_number, calculate_output_biases_error_gradient(inputs,layers,deltas,activations_states));
    }

    return error_gradient;
    */
    return Tensor<type, 1>();
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_forget_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                 const Layer::ForwardPropagation& layers,
                                                                                 const Tensor<type, 2>& deltas,
                                                                                 const Tensor<type, 2>& activations_states)
 {
    /*
    const int instances_number = inputs.dimension(0);
    const int inputs_number = get_inputs_number();
    const int neurons_number = get_neurons_number();
    const int parameters_number = inputs_number*neurons_number;

     Tensor<type, 1> forget_weights_error_gradient(parameters_number, 0.0);

     Tensor<type, 2> input_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);
     Tensor<type, 2> forget_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);
     Tensor<type, 2> state_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);
     Tensor<type, 2> output_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);

     Tensor<type, 2> hidden_states_weights_derivatives(parameters_number, neurons_number,  0.0);
     Tensor<type, 2> cell_state_weights_derivatives(parameters_number, neurons_number,  0.0);

     const Tensor<type, 2> forget_activations = activations_states.get_matrix(0);
     const Tensor<type, 2> input_activations = activations_states.get_matrix(1);
     const Tensor<type, 2> state_activations = activations_states.get_matrix(2);
     const Tensor<type, 2> output_activations = activations_states.get_matrix(3);
     const Tensor<type, 2> cell_state_activations = activations_states.get_matrix(4);

     const Tensor<type, 2> forget_derivatives = layers.activations_derivatives.get_matrix(0);
     const Tensor<type, 2> input_derivatives = layers.activations_derivatives.get_matrix(1);
     const Tensor<type, 2> state_derivatives = layers.activations_derivatives.get_matrix(2);
     const Tensor<type, 2> output_derivatives = layers.activations_derivatives.get_matrix(3);
     const Tensor<type, 2> hidden_derivatives = layers.activations_derivatives.get_matrix(4);

     int column_index = 0;
     int input_index = 0;

     for(int instance = 0; instance < instances_number; instance++)
     {
         const Tensor<type, 1> current_inputs = inputs.get_row(instance);

         const Tensor<type, 2> current_layer_deltas = deltas.get_row(instance).to_column_matrix();

         const Tensor<type, 1> current_forget_activations = forget_activations.get_row(instance);
         const Tensor<type, 1> current_input_activations = input_activations.get_row(instance);
         const Tensor<type, 1> current_state_activations = state_activations.get_row(instance);
         const Tensor<type, 1> current_output_activations = output_activations.get_row(instance);
         const Tensor<type, 1> current_cell_state_activations = cell_state_activations.get_row(instance);

         const Tensor<type, 1> current_forget_derivatives = forget_derivatives.get_row(instance);
         const Tensor<type, 1> current_input_derivatives = input_derivatives.get_row(instance);
         const Tensor<type, 1> current_state_derivatives = state_derivatives.get_row(instance);
         const Tensor<type, 1> current_output_derivatives = output_derivatives.get_row(instance);
         const Tensor<type, 1> current_hidden_derivatives = hidden_derivatives.get_row(instance);

         Tensor<type, 1> previous_cell_state_activations(neurons_number,0.0);

         if(instance%timesteps == 0)
         {
             forget_combinations_weights_derivatives.setZero();
             input_combinations_weights_derivatives.setZero();
             output_combinations_weights_derivatives.setZero();
             state_combinations_weights_derivatives.setZero();

             cell_state_weights_derivatives.setZero();
         }
         else
         {
             previous_cell_state_activations = cell_state_activations.get_row(instance-1);

             forget_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, forget_recurrent_weights);
             input_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, input_recurrent_weights).multiply_rows(current_input_derivatives);
             state_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, state_recurrent_weights).multiply_rows(current_state_derivatives);
             output_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, output_recurrent_weights).multiply_rows(current_output_derivatives);
         }

         column_index = 0;
         input_index = 0;

         for(int i = 0; i < parameters_number; i++)
         {
             forget_combinations_weights_derivatives(i, column_index) += current_inputs[input_index];

             input_index++;

             if(input_index == inputs_number)
             {
                 input_index = 0;
                 column_index++;
             }
         }

         cell_state_weights_derivatives = cell_state_weights_derivatives.multiply_rows(current_forget_activations)
                                        + input_combinations_weights_derivatives.multiply_rows(current_state_activations)
                                        + state_combinations_weights_derivatives.multiply_rows(current_input_activations)
                                        + forget_combinations_weights_derivatives.multiply_rows((current_forget_derivatives*previous_cell_state_activations));

         hidden_states_weights_derivatives =
                 (output_combinations_weights_derivatives.multiply_rows(calculate_activations(current_cell_state_activations)) +
                 cell_state_weights_derivatives.multiply_rows(current_output_activations*current_hidden_derivatives));

         forget_weights_error_gradient += dot(hidden_states_weights_derivatives, current_layer_deltas).to_vector();
     }

     return forget_weights_error_gradient;
     */
     return Tensor<type, 1>();
}



Tensor<type, 1> LongShortTermMemoryLayer::calculate_input_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                const Layer::ForwardPropagation& layers,
                                                                                const Tensor<type, 2>& deltas,
                                                                                const Tensor<type, 2>& activations_states)
{
    /*
    const int instances_number = inputs.dimension(0);
    const int inputs_number = get_inputs_number();
    const int neurons_number = get_neurons_number();
    const int parameters_number = inputs_number*neurons_number;

    Tensor<type, 1> input_weights_error_gradient(parameters_number, 0.0);

    Tensor<type, 2> input_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> forget_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> state_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> output_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);

    Tensor<type, 2> hidden_states_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> cell_state_weights_derivatives(parameters_number, neurons_number,  0.0);

    const Tensor<type, 2> forget_activations = activations_states.get_matrix(0);
    const Tensor<type, 2> input_activations = activations_states.get_matrix(1);
    const Tensor<type, 2> state_activations = activations_states.get_matrix(2);
    const Tensor<type, 2> output_activations = activations_states.get_matrix(3);
    const Tensor<type, 2> cell_state_activations = activations_states.get_matrix(4);

    const Tensor<type, 2> forget_derivatives = layers.activations_derivatives.get_matrix(0);
    const Tensor<type, 2> input_derivatives = layers.activations_derivatives.get_matrix(1);
    const Tensor<type, 2> state_derivatives = layers.activations_derivatives.get_matrix(2);
    const Tensor<type, 2> output_derivatives = layers.activations_derivatives.get_matrix(3);
    const Tensor<type, 2> hidden_derivatives = layers.activations_derivatives.get_matrix(4);

    int column_index = 0;
    int input_index = 0;

    for(int instance = 0; instance < instances_number; instance++)
    {
        const Tensor<type, 1> current_inputs = inputs.get_row(instance);

        const Tensor<type, 2> current_layer_deltas = deltas.get_row(instance).to_column_matrix();

        const Tensor<type, 1> current_forget_activations = forget_activations.get_row(instance);
        const Tensor<type, 1> current_input_activations = input_activations.get_row(instance);
        const Tensor<type, 1> current_state_activations = state_activations.get_row(instance);
        const Tensor<type, 1> current_output_activations = output_activations.get_row(instance);
        const Tensor<type, 1> current_cell_state_activations = cell_state_activations.get_row(instance);

        const Tensor<type, 1> current_forget_derivatives = forget_derivatives.get_row(instance);
        const Tensor<type, 1> current_input_derivatives = input_derivatives.get_row(instance);
        const Tensor<type, 1> current_state_derivatives = state_derivatives.get_row(instance);
        const Tensor<type, 1> current_output_derivatives = output_derivatives.get_row(instance);
        const Tensor<type, 1> current_hidden_derivatives = hidden_derivatives.get_row(instance);

        Tensor<type, 1> previous_cell_state_activations(neurons_number, 0.0);

        if(instance%timesteps == 0)
        {
            forget_combinations_weights_derivatives.setZero();
            input_combinations_weights_derivatives.setZero();
            output_combinations_weights_derivatives.setZero();
            state_combinations_weights_derivatives.setZero();

            cell_state_weights_derivatives.setZero();
        }
        else
        {
            previous_cell_state_activations = cell_state_activations.get_row(instance-1);

            forget_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, forget_recurrent_weights).multiply_rows(current_forget_derivatives);
            input_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, input_recurrent_weights);
            state_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, state_recurrent_weights).multiply_rows(current_state_derivatives);
            output_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, output_recurrent_weights).multiply_rows(current_output_derivatives);
        }

        column_index = 0;
        input_index = 0;

        for(int i = 0; i < parameters_number; i++)
        {
            input_combinations_weights_derivatives(i, column_index) += current_inputs[input_index];

            input_index++;

            if(input_index == inputs_number)
            {
                input_index = 0;
                column_index++;
            }
        }

        cell_state_weights_derivatives = cell_state_weights_derivatives.multiply_rows(current_forget_activations)
                                       + forget_combinations_weights_derivatives.multiply_rows(previous_cell_state_activations)
                                       + state_combinations_weights_derivatives.multiply_rows(current_input_activations)
                                       + input_combinations_weights_derivatives.multiply_rows((current_input_derivatives*current_state_activations));

        hidden_states_weights_derivatives =
                (output_combinations_weights_derivatives.multiply_rows(calculate_activations(current_cell_state_activations)) +
                cell_state_weights_derivatives.multiply_rows(current_output_activations*current_hidden_derivatives));

        input_weights_error_gradient += dot(hidden_states_weights_derivatives, current_layer_deltas).to_vector();
    }

    return input_weights_error_gradient;
    */
    return Tensor<type, 1>();
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_state_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                const Layer::ForwardPropagation& layers,
                                                                                const Tensor<type, 2>& deltas,
                                                                                const Tensor<type, 2>& activations_states)
{
    /*
    const int instances_number = inputs.dimension(0);
    const int inputs_number = get_inputs_number();
    const int neurons_number = get_neurons_number();
    const int parameters_number = inputs_number*neurons_number;

    Tensor<type, 1> state_weights_error_gradient(parameters_number, 0.0);

    Tensor<type, 2> input_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> forget_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> state_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> output_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);

    Tensor<type, 2> hidden_states_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> cell_state_weights_derivatives(parameters_number, neurons_number,  0.0);

    const Tensor<type, 2> forget_activations = activations_states.get_matrix(0);
    const Tensor<type, 2> input_activations = activations_states.get_matrix(1);
    const Tensor<type, 2> state_activations = activations_states.get_matrix(2);
    const Tensor<type, 2> output_activations = activations_states.get_matrix(3);
    const Tensor<type, 2> cell_state_activations = activations_states.get_matrix(4);

    const Tensor<type, 2> forget_derivatives = layers.activations_derivatives.get_matrix(0);
    const Tensor<type, 2> input_derivatives = layers.activations_derivatives.get_matrix(1);
    const Tensor<type, 2> state_derivatives = layers.activations_derivatives.get_matrix(2);
    const Tensor<type, 2> output_derivatives = layers.activations_derivatives.get_matrix(3);
    const Tensor<type, 2> hidden_derivatives = layers.activations_derivatives.get_matrix(4);

    int column_index = 0;
    int input_index = 0;

    for(int instance = 0; instance < instances_number; instance++)
    {
        const Tensor<type, 1> current_inputs = inputs.get_row(instance);

        const Tensor<type, 2> current_layer_deltas = deltas.get_row(instance).to_column_matrix();

        const Tensor<type, 1> current_forget_activations = forget_activations.get_row(instance);
        const Tensor<type, 1> current_input_activations = input_activations.get_row(instance);
        const Tensor<type, 1> current_state_activations = state_activations.get_row(instance);
        const Tensor<type, 1> current_output_activations = output_activations.get_row(instance);
        const Tensor<type, 1> current_cell_state_activations = cell_state_activations.get_row(instance);

        const Tensor<type, 1> current_forget_derivatives = forget_derivatives.get_row(instance);
        const Tensor<type, 1> current_input_derivatives = input_derivatives.get_row(instance);
        const Tensor<type, 1> current_state_derivatives = state_derivatives.get_row(instance);
        const Tensor<type, 1> current_output_derivatives = output_derivatives.get_row(instance);
        const Tensor<type, 1> current_hidden_derivatives = hidden_derivatives.get_row(instance);

        Tensor<type, 1> previous_cell_state_activations(neurons_number, 0.0);

        if(instance%timesteps == 0)
        {
            forget_combinations_weights_derivatives.setZero();
            input_combinations_weights_derivatives.setZero();
            output_combinations_weights_derivatives.setZero();
            state_combinations_weights_derivatives.setZero();

            cell_state_weights_derivatives.setZero();
        }
        else
        {
            previous_cell_state_activations = cell_state_activations.get_row(instance-1);

            forget_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, forget_recurrent_weights).multiply_rows(current_forget_derivatives);
            input_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, input_recurrent_weights).multiply_rows(current_input_derivatives);
            state_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, state_recurrent_weights);
            output_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, output_recurrent_weights).multiply_rows(current_output_derivatives);
        }

        column_index = 0;
        input_index = 0;

        for(int i = 0; i < parameters_number; i++)
        {
            state_combinations_weights_derivatives(i, column_index) += current_inputs[input_index];

            input_index++;

            if(input_index == inputs_number)
            {
                input_index = 0;
                column_index++;
            }
        }

        cell_state_weights_derivatives = cell_state_weights_derivatives.multiply_rows(current_forget_activations)
                                       + forget_combinations_weights_derivatives.multiply_rows(previous_cell_state_activations)
                                       + input_combinations_weights_derivatives.multiply_rows(current_state_activations)
                                       + state_combinations_weights_derivatives.multiply_rows(current_state_derivatives*current_input_activations);

        hidden_states_weights_derivatives =
                (output_combinations_weights_derivatives.multiply_rows(calculate_activations(current_cell_state_activations)) +
                cell_state_weights_derivatives.multiply_rows(current_output_activations*current_hidden_derivatives));

        state_weights_error_gradient += dot(hidden_states_weights_derivatives, current_layer_deltas).to_vector();
    }

    return state_weights_error_gradient;
    */
    return Tensor<type, 1>();
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_output_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                 const Layer::ForwardPropagation& layers,
                                                                                 const Tensor<type, 2>& deltas,
                                                                                 const Tensor<type, 2>& activations_states)
 {
    /*
     const int instances_number = inputs.dimension(0);
     const int inputs_number = get_inputs_number();
     const int neurons_number = get_neurons_number();
     const int parameters_number = inputs_number*neurons_number;

     Tensor<type, 1> output_weights_error_gradient(parameters_number, 0.0);

     Tensor<type, 2> input_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);
     Tensor<type, 2> forget_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);
     Tensor<type, 2> state_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);
     Tensor<type, 2> output_combinations_weights_derivatives(parameters_number, neurons_number,  0.0);

     Tensor<type, 2> hidden_states_weights_derivatives(parameters_number, neurons_number,  0.0);
     Tensor<type, 2> cell_state_weights_derivatives(parameters_number, neurons_number,  0.0);

     const Tensor<type, 2> forget_activations = activations_states.get_matrix(0);
     const Tensor<type, 2> input_activations = activations_states.get_matrix(1);
     const Tensor<type, 2> state_activations = activations_states.get_matrix(2);
     const Tensor<type, 2> output_activations = activations_states.get_matrix(3);
     const Tensor<type, 2> cell_state_activations = activations_states.get_matrix(4);

     const Tensor<type, 2> forget_derivatives = layers.activations_derivatives.get_matrix(0);
     const Tensor<type, 2> input_derivatives = layers.activations_derivatives.get_matrix(1);
     const Tensor<type, 2> state_derivatives = layers.activations_derivatives.get_matrix(2);
     const Tensor<type, 2> output_derivatives = layers.activations_derivatives.get_matrix(3);
     const Tensor<type, 2> hidden_derivatives = layers.activations_derivatives.get_matrix(4);

     int column_index = 0;
     int input_index = 0;

     for(int instance = 0; instance < instances_number; instance++)
     {
         const Tensor<type, 1> current_inputs = inputs.get_row(instance);

         const Tensor<type, 2> current_layer_deltas = deltas.get_row(instance).to_column_matrix();

         const Tensor<type, 1> current_forget_activations = forget_activations.get_row(instance);
         const Tensor<type, 1> current_input_activations = input_activations.get_row(instance);
         const Tensor<type, 1> current_state_activations = state_activations.get_row(instance);
         const Tensor<type, 1> current_output_activations = output_activations.get_row(instance);
         const Tensor<type, 1> current_cell_state_activations = cell_state_activations.get_row(instance);

         const Tensor<type, 1> current_forget_derivatives = forget_derivatives.get_row(instance);
         const Tensor<type, 1> current_input_derivatives = input_derivatives.get_row(instance);
         const Tensor<type, 1> current_state_derivatives = state_derivatives.get_row(instance);
         const Tensor<type, 1> current_output_derivatives = output_derivatives.get_row(instance);
         const Tensor<type, 1> current_hidden_derivatives = hidden_derivatives.get_row(instance);

         Tensor<type, 1> previous_cell_state_activations(neurons_number, 0.0);

         if(instance%timesteps == 0)
         {
             forget_combinations_weights_derivatives.setZero();
             input_combinations_weights_derivatives.setZero();
             output_combinations_weights_derivatives.setZero();
             state_combinations_weights_derivatives.setZero();

             cell_state_weights_derivatives.setZero();
         }
         else
         {
             previous_cell_state_activations = cell_state_activations.get_row(instance-1);

             forget_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, forget_recurrent_weights).multiply_rows(current_forget_derivatives);
             input_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, input_recurrent_weights).multiply_rows(current_input_derivatives);
             state_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, state_recurrent_weights).multiply_rows(current_state_derivatives);
             output_combinations_weights_derivatives = dot(hidden_states_weights_derivatives, output_recurrent_weights);
         }

         column_index = 0;
         input_index = 0;

         for(int i = 0; i < parameters_number; i++)
         {
             output_combinations_weights_derivatives(i, column_index) += current_inputs[input_index];

             input_index++;

             if(input_index == inputs_number)
             {
                 input_index = 0;
                 column_index++;
             }
         }

         cell_state_weights_derivatives = cell_state_weights_derivatives.multiply_rows(current_forget_activations)
                                        + forget_combinations_weights_derivatives.multiply_rows(previous_cell_state_activations)
                                        + state_combinations_weights_derivatives.multiply_rows(current_input_activations)
                                        + input_combinations_weights_derivatives.multiply_rows(current_state_activations);

         hidden_states_weights_derivatives =
                 (output_combinations_weights_derivatives.multiply_rows(current_output_derivatives*calculate_activations(current_cell_state_activations)) +
                 cell_state_weights_derivatives.multiply_rows(current_output_activations*current_hidden_derivatives));

         output_weights_error_gradient += dot(hidden_states_weights_derivatives, current_layer_deltas).to_vector();
     }

     return output_weights_error_gradient;
*/
     return Tensor<type, 1>();
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_forget_recurrent_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                           const Layer::ForwardPropagation& layers,
                                                                                           const Tensor<type, 2>& deltas,
                                                                                           const Tensor<type, 2>& activations_states)
{
    /*
    const int instances_number = inputs.dimension(0);
    const int neurons_number = get_neurons_number();
    const int parameters_number = neurons_number*neurons_number;

    Tensor<type, 1> forget_recurrent_weights_error_gradient(parameters_number, 0.0);

    Tensor<type, 2> input_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> forget_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> state_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> output_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);

    Tensor<type, 2> hidden_states_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> cell_state_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);

    const Tensor<type, 2> forget_activations = activations_states.get_matrix(0);
    const Tensor<type, 2> input_activations = activations_states.get_matrix(1);
    const Tensor<type, 2> state_activations = activations_states.get_matrix(2);
    const Tensor<type, 2> output_activations = activations_states.get_matrix(3);
    const Tensor<type, 2> cell_state_activations = activations_states.get_matrix(4);
    const Tensor<type, 2> hidden_state_activations = activations_states.get_matrix(5);

    const Tensor<type, 2> forget_derivatives = layers.activations_derivatives.get_matrix(0);
    const Tensor<type, 2> input_derivatives = layers.activations_derivatives.get_matrix(1);
    const Tensor<type, 2> state_derivatives = layers.activations_derivatives.get_matrix(2);
    const Tensor<type, 2> output_derivatives = layers.activations_derivatives.get_matrix(3);
    const Tensor<type, 2> hidden_derivatives = layers.activations_derivatives.get_matrix(4);

    int column_index = 0;
    int activation_index = 0;

    for(int instance = 0; instance < instances_number; instance++)
    {
        const Tensor<type, 2> current_layer_deltas = deltas.get_row(instance).to_column_matrix();

        const Tensor<type, 1> current_forget_activations = forget_activations.get_row(instance);
        const Tensor<type, 1> current_input_activations = input_activations.get_row(instance);
        const Tensor<type, 1> current_state_activations = state_activations.get_row(instance);
        const Tensor<type, 1> current_output_activations = output_activations.get_row(instance);
        const Tensor<type, 1> current_cell_state_activations = cell_state_activations.get_row(instance);

        const Tensor<type, 1> current_forget_derivatives = forget_derivatives.get_row(instance);
        const Tensor<type, 1> current_input_derivatives = input_derivatives.get_row(instance);
        const Tensor<type, 1> current_state_derivatives = state_derivatives.get_row(instance);
        const Tensor<type, 1> current_output_derivatives = output_derivatives.get_row(instance);
        const Tensor<type, 1> current_hidden_derivatives = hidden_derivatives.get_row(instance);

        if(instance%timesteps == 0)
        {
            cell_state_recurrent_weights_derivatives.setZero();
            hidden_states_recurrent_weights_derivatives.setZero();
        }
        else
        {
            const Tensor<type, 1> previous_hidden_state_activations = hidden_state_activations.get_row(instance-1);

            const Tensor<type, 1> previous_cell_state_activations = cell_state_activations.get_row(instance-1);

            forget_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, forget_recurrent_weights);
            input_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, input_recurrent_weights).multiply_rows(current_input_derivatives);
            state_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, state_recurrent_weights).multiply_rows(current_state_derivatives);
            output_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, output_recurrent_weights).multiply_rows(current_output_derivatives);

            column_index = 0;
            activation_index = 0;

            for(int i = 0; i < parameters_number; i++)
            {
                forget_combinations_recurrent_weights_derivatives(i, column_index) += previous_hidden_state_activations[activation_index];

                activation_index++;

                if(activation_index == neurons_number)
                {
                    activation_index = 0;
                    column_index++;
                }
            }

            cell_state_recurrent_weights_derivatives = cell_state_recurrent_weights_derivatives.multiply_rows(current_forget_activations)
                                                     + input_combinations_recurrent_weights_derivatives.multiply_rows(current_state_activations)
                                                     + state_combinations_recurrent_weights_derivatives.multiply_rows(current_input_activations)
                                                     + forget_combinations_recurrent_weights_derivatives.multiply_rows((current_forget_derivatives*previous_cell_state_activations));

            hidden_states_recurrent_weights_derivatives =
                    (output_combinations_recurrent_weights_derivatives.multiply_rows(calculate_activations(current_cell_state_activations)) +
                    cell_state_recurrent_weights_derivatives.multiply_rows(current_output_activations*current_hidden_derivatives));
        }

        forget_recurrent_weights_error_gradient += dot(hidden_states_recurrent_weights_derivatives, current_layer_deltas).to_vector();
    }

    return forget_recurrent_weights_error_gradient;
*/
    return Tensor<type, 1>();
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_input_recurrent_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                          const Layer::ForwardPropagation& layers,
                                                                                          const Tensor<type, 2>& deltas,
                                                                                          const Tensor<type, 2>& activations_states)
{
    /*
   const int instances_number = inputs.dimension(0);
   const int neurons_number = get_neurons_number();
   const int parameters_number = neurons_number*neurons_number;

   Tensor<type, 1> input_recurrent_weights_error_gradient(parameters_number, 0.0);

   Tensor<type, 2> input_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> forget_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> state_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> output_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);

   Tensor<type, 2> hidden_states_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> cell_state_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);

   const Tensor<type, 2> forget_activations = activations_states.get_matrix(0);
   const Tensor<type, 2> input_activations = activations_states.get_matrix(1);
   const Tensor<type, 2> state_activations = activations_states.get_matrix(2);
   const Tensor<type, 2> output_activations = activations_states.get_matrix(3);
   const Tensor<type, 2> cell_state_activations = activations_states.get_matrix(4);
   const Tensor<type, 2> hidden_state_activations = activations_states.get_matrix(5);

   const Tensor<type, 2> forget_derivatives = layers.activations_derivatives.get_matrix(0);
   const Tensor<type, 2> input_derivatives = layers.activations_derivatives.get_matrix(1);
   const Tensor<type, 2> state_derivatives = layers.activations_derivatives.get_matrix(2);
   const Tensor<type, 2> output_derivatives = layers.activations_derivatives.get_matrix(3);
   const Tensor<type, 2> hidden_derivatives = layers.activations_derivatives.get_matrix(4);

   int column_index = 0;
   int activation_index = 0;

   for(int instance = 0; instance < instances_number; instance++)
   {
       const Tensor<type, 2> current_layer_deltas = deltas.get_row(instance).to_column_matrix();

       const Tensor<type, 1> current_forget_activations = forget_activations.get_row(instance);
       const Tensor<type, 1> current_input_activations = input_activations.get_row(instance);
       const Tensor<type, 1> current_state_activations = state_activations.get_row(instance);
       const Tensor<type, 1> current_output_activations = output_activations.get_row(instance);
       const Tensor<type, 1> current_cell_state_activations = cell_state_activations.get_row(instance);

       const Tensor<type, 1> current_forget_derivatives = forget_derivatives.get_row(instance);
       const Tensor<type, 1> current_input_derivatives = input_derivatives.get_row(instance);
       const Tensor<type, 1> current_state_derivatives = state_derivatives.get_row(instance);
       const Tensor<type, 1> current_output_derivatives = output_derivatives.get_row(instance);
       const Tensor<type, 1> current_hidden_derivatives = hidden_derivatives.get_row(instance);

       if(instance%timesteps == 0)
       {
           cell_state_recurrent_weights_derivatives.setZero();
           hidden_states_recurrent_weights_derivatives.setZero();
       }
       else
       {
           const Tensor<type, 1> previous_hidden_state_activations = hidden_state_activations.get_row(instance-1);
           const Tensor<type, 1> previous_cell_state_activations = cell_state_activations.get_row(instance-1);

           forget_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, forget_recurrent_weights).multiply_rows(current_forget_derivatives);
           input_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, input_recurrent_weights);
           state_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, state_recurrent_weights).multiply_rows(current_state_derivatives);
           output_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, output_recurrent_weights).multiply_rows(current_output_derivatives);

           column_index = 0;
           activation_index = 0;

           for(int i = 0; i < parameters_number; i++)
           {
               input_combinations_recurrent_weights_derivatives(i, column_index) += previous_hidden_state_activations[activation_index];

               activation_index++;

               if(activation_index == neurons_number)
               {
                   activation_index = 0;
                   column_index++;
               }
           }

           cell_state_recurrent_weights_derivatives = cell_state_recurrent_weights_derivatives.multiply_rows(current_forget_activations)
                                                    + input_combinations_recurrent_weights_derivatives.multiply_rows(current_input_derivatives*current_state_activations)
                                                    + state_combinations_recurrent_weights_derivatives.multiply_rows(current_input_activations)
                                                    + forget_combinations_recurrent_weights_derivatives.multiply_rows(previous_cell_state_activations);

           hidden_states_recurrent_weights_derivatives =
                   (output_combinations_recurrent_weights_derivatives.multiply_rows(calculate_activations(current_cell_state_activations)) +
                   cell_state_recurrent_weights_derivatives.multiply_rows(current_output_activations*current_hidden_derivatives));
       }

       input_recurrent_weights_error_gradient += dot(hidden_states_recurrent_weights_derivatives, current_layer_deltas).to_vector();
   }

   return input_recurrent_weights_error_gradient;
*/
    return Tensor<type, 1>();
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_state_recurrent_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                          const Layer::ForwardPropagation& layers,
                                                                                          const Tensor<type, 2>& deltas,
                                                                                          const Tensor<type, 2>& activations_states)
{
    /*
   const int instances_number = inputs.dimension(0);
   const int neurons_number = get_neurons_number();
   const int parameters_number = neurons_number*neurons_number;

   Tensor<type, 1> state_recurrent_weights_error_gradient(parameters_number, 0.0);

   Tensor<type, 2> input_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> forget_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> state_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> output_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);

   Tensor<type, 2> hidden_states_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> cell_state_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);

   const Tensor<type, 2> forget_activations = activations_states.get_matrix(0);
   const Tensor<type, 2> input_activations = activations_states.get_matrix(1);
   const Tensor<type, 2> state_activations = activations_states.get_matrix(2);
   const Tensor<type, 2> output_activations = activations_states.get_matrix(3);
   const Tensor<type, 2> cell_state_activations = activations_states.get_matrix(4);
   const Tensor<type, 2> hidden_state_activations = activations_states.get_matrix(5);

   const Tensor<type, 2> forget_derivatives = layers.activations_derivatives.get_matrix(0);
   const Tensor<type, 2> input_derivatives = layers.activations_derivatives.get_matrix(1);
   const Tensor<type, 2> state_derivatives = layers.activations_derivatives.get_matrix(2);
   const Tensor<type, 2> output_derivatives = layers.activations_derivatives.get_matrix(3);
   const Tensor<type, 2> hidden_derivatives = layers.activations_derivatives.get_matrix(4);

   int column_index = 0;
   int activation_index = 0;

   for(int instance = 0; instance < instances_number; instance++)
   {
       const Tensor<type, 2> current_layer_deltas = deltas.get_row(instance).to_column_matrix();

       const Tensor<type, 1> current_forget_activations = forget_activations.get_row(instance);
       const Tensor<type, 1> current_input_activations = input_activations.get_row(instance);
       const Tensor<type, 1> current_state_activations = state_activations.get_row(instance);
       const Tensor<type, 1> current_output_activations = output_activations.get_row(instance);
       const Tensor<type, 1> current_cell_state_activations = cell_state_activations.get_row(instance);

       const Tensor<type, 1> current_forget_derivatives = forget_derivatives.get_row(instance);
       const Tensor<type, 1> current_input_derivatives = input_derivatives.get_row(instance);
       const Tensor<type, 1> current_state_derivatives = state_derivatives.get_row(instance);
       const Tensor<type, 1> current_output_derivatives = output_derivatives.get_row(instance);
       const Tensor<type, 1> current_hidden_derivatives = hidden_derivatives.get_row(instance);

       if(instance%timesteps == 0)
       {
           cell_state_recurrent_weights_derivatives.setZero();
           hidden_states_recurrent_weights_derivatives.setZero();
       }
       else
       {
           const Tensor<type, 1> previous_hidden_state_activations = hidden_state_activations.get_row(instance-1);
           const Tensor<type, 1> previous_cell_state_activations = cell_state_activations.get_row(instance-1);

           forget_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, forget_recurrent_weights).multiply_rows(current_forget_derivatives);
           input_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, input_recurrent_weights).multiply_rows(current_input_derivatives);
           state_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, state_recurrent_weights);
           output_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, output_recurrent_weights).multiply_rows(current_output_derivatives);

           column_index = 0;
           activation_index = 0;

           for(int i = 0; i < parameters_number; i++)
           {
               state_combinations_recurrent_weights_derivatives(i, column_index) += previous_hidden_state_activations[activation_index];

               activation_index++;

               if(activation_index == neurons_number)
               {
                   activation_index = 0;
                   column_index++;
               }
           }

           cell_state_recurrent_weights_derivatives = cell_state_recurrent_weights_derivatives.multiply_rows(current_forget_activations)
                                                    + input_combinations_recurrent_weights_derivatives.multiply_rows(current_state_activations)
                                                    + state_combinations_recurrent_weights_derivatives.multiply_rows(current_state_derivatives*current_input_activations)
                                                    + forget_combinations_recurrent_weights_derivatives.multiply_rows(previous_cell_state_activations);

           hidden_states_recurrent_weights_derivatives =
                   (output_combinations_recurrent_weights_derivatives.multiply_rows(calculate_activations(current_cell_state_activations)) +
                   cell_state_recurrent_weights_derivatives.multiply_rows(current_output_activations*current_hidden_derivatives));
       }

       state_recurrent_weights_error_gradient += dot(hidden_states_recurrent_weights_derivatives, current_layer_deltas).to_vector();
   }

   return state_recurrent_weights_error_gradient;
   */
   return Tensor<type, 1>();
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_output_recurrent_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                                           const Layer::ForwardPropagation& layers,
                                                                                           const Tensor<type, 2>& deltas,
                                                                                           const Tensor<type, 2>& activations_states)
 {
/*
    const int instances_number = inputs.dimension(0);
    const int neurons_number = get_neurons_number();
    const int parameters_number = neurons_number*neurons_number;

    Tensor<type, 1> output_recurrent_weights_error_gradient(parameters_number, 0.0);

    Tensor<type, 2> input_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> forget_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> state_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> output_combinations_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);

    Tensor<type, 2> hidden_states_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> cell_state_recurrent_weights_derivatives(parameters_number, neurons_number,  0.0);

    const Tensor<type, 2> forget_activations = activations_states.get_matrix(0);
    const Tensor<type, 2> input_activations = activations_states.get_matrix(1);
    const Tensor<type, 2> state_activations = activations_states.get_matrix(2);
    const Tensor<type, 2> output_activations = activations_states.get_matrix(3);
    const Tensor<type, 2> cell_state_activations = activations_states.get_matrix(4);
    const Tensor<type, 2> hidden_state_activations = activations_states.get_matrix(5);

    const Tensor<type, 2> forget_derivatives = layers.activations_derivatives.get_matrix(0);
    const Tensor<type, 2> input_derivatives = layers.activations_derivatives.get_matrix(1);
    const Tensor<type, 2> state_derivatives = layers.activations_derivatives.get_matrix(2);
    const Tensor<type, 2> output_derivatives = layers.activations_derivatives.get_matrix(3);
    const Tensor<type, 2> hidden_derivatives = layers.activations_derivatives.get_matrix(4);

    int column_index = 0;
    int activation_index = 0;

    for(int instance = 0; instance < instances_number; instance++)
    {
        const Tensor<type, 2> current_layer_deltas = deltas.get_row(instance).to_column_matrix();

        const Tensor<type, 1> current_forget_activations = forget_activations.get_row(instance);
        const Tensor<type, 1> current_input_activations = input_activations.get_row(instance);
        const Tensor<type, 1> current_state_activations = state_activations.get_row(instance);
        const Tensor<type, 1> current_output_activations = output_activations.get_row(instance);
        const Tensor<type, 1> current_cell_state_activations = cell_state_activations.get_row(instance);

        const Tensor<type, 1> current_forget_derivatives = forget_derivatives.get_row(instance);
        const Tensor<type, 1> current_input_derivatives = input_derivatives.get_row(instance);
        const Tensor<type, 1> current_state_derivatives = state_derivatives.get_row(instance);
        const Tensor<type, 1> current_output_derivatives = output_derivatives.get_row(instance);
        const Tensor<type, 1> current_hidden_derivatives = hidden_derivatives.get_row(instance);

        if(instance%timesteps == 0)
        {
            cell_state_recurrent_weights_derivatives.setZero();
            hidden_states_recurrent_weights_derivatives.setZero();
        }
        else
        {
            const Tensor<type, 1> previous_hidden_state_activations = hidden_state_activations.get_row(instance-1);
            const Tensor<type, 1> previous_cell_state_activations = cell_state_activations.get_row(instance-1);

            forget_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, forget_recurrent_weights).multiply_rows(current_forget_derivatives);
            input_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, input_recurrent_weights).multiply_rows(current_input_derivatives);
            state_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, state_recurrent_weights).multiply_rows(current_state_derivatives);
            output_combinations_recurrent_weights_derivatives = dot(hidden_states_recurrent_weights_derivatives, output_recurrent_weights);

            column_index = 0;
            activation_index = 0;

            for(int i = 0; i < parameters_number; i++)
            {
                output_combinations_recurrent_weights_derivatives(i, column_index) += previous_hidden_state_activations[activation_index];

                activation_index++;

                if(activation_index == neurons_number)
                {
                    activation_index = 0;
                    column_index++;
                }
            }

            cell_state_recurrent_weights_derivatives = cell_state_recurrent_weights_derivatives.multiply_rows(current_forget_activations)
                                                     + input_combinations_recurrent_weights_derivatives.multiply_rows(current_state_activations)
                                                     + state_combinations_recurrent_weights_derivatives.multiply_rows(current_input_activations)
                                                     + forget_combinations_recurrent_weights_derivatives.multiply_rows(previous_cell_state_activations);

            hidden_states_recurrent_weights_derivatives =
                    (output_combinations_recurrent_weights_derivatives.multiply_rows(current_output_derivatives*calculate_activations(current_cell_state_activations)) +
                    cell_state_recurrent_weights_derivatives.multiply_rows(current_output_activations*current_hidden_derivatives));
        }

        output_recurrent_weights_error_gradient += dot(hidden_states_recurrent_weights_derivatives, current_layer_deltas).to_vector();
    }

    return output_recurrent_weights_error_gradient;
*/
    return Tensor<type, 1>();
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_forget_biases_error_gradient(const Tensor<type, 2>& inputs,
                                                                                const Layer::ForwardPropagation& layers,
                                                                                const Tensor<type, 2>& deltas,
                                                                                const Tensor<type, 2>& activations_states)
{
    /*
    const int instances_number = inputs.dimension(0);
    const int neurons_number = get_neurons_number();
    const int parameters_number = neurons_number;

    Tensor<type, 1> forget_biases_error_gradient(parameters_number, 0.0);

    Tensor<type, 2> input_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> forget_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> state_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> output_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);

    Tensor<type, 2> hidden_states_biases_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> cell_state_biases_derivatives(parameters_number, neurons_number,  0.0);

    const Tensor<type, 2> forget_activations = activations_states.get_matrix(0);
    const Tensor<type, 2> input_activations = activations_states.get_matrix(1);
    const Tensor<type, 2> state_activations = activations_states.get_matrix(2);
    const Tensor<type, 2> output_activations = activations_states.get_matrix(3);
    const Tensor<type, 2> cell_state_activations = activations_states.get_matrix(4);

    const Tensor<type, 2> forget_derivatives = layers.activations_derivatives.get_matrix(0);
    const Tensor<type, 2> input_derivatives = layers.activations_derivatives.get_matrix(1);
    const Tensor<type, 2> state_derivatives = layers.activations_derivatives.get_matrix(2);
    const Tensor<type, 2> output_derivatives = layers.activations_derivatives.get_matrix(3);
    const Tensor<type, 2> hidden_derivatives = layers.activations_derivatives.get_matrix(4);

    for(int instance = 0; instance < instances_number; instance++)
    {
        const Tensor<type, 2> current_layer_deltas = deltas.get_row(instance).to_column_matrix();

        const Tensor<type, 1> current_forget_activations = forget_activations.get_row(instance);
        const Tensor<type, 1> current_input_activations = input_activations.get_row(instance);
        const Tensor<type, 1> current_state_activations = state_activations.get_row(instance);
        const Tensor<type, 1> current_output_activations = output_activations.get_row(instance);
        const Tensor<type, 1> current_cell_state_activations = cell_state_activations.get_row(instance);

        const Tensor<type, 1> current_forget_derivatives = forget_derivatives.get_row(instance);
        const Tensor<type, 1> current_input_derivatives = input_derivatives.get_row(instance);
        const Tensor<type, 1> current_state_derivatives = state_derivatives.get_row(instance);
        const Tensor<type, 1> current_output_derivatives = output_derivatives.get_row(instance);
        const Tensor<type, 1> current_hidden_derivatives = hidden_derivatives.get_row(instance);

        Tensor<type, 1> previous_cell_state_activations(neurons_number,0.0);

        if(instance%timesteps == 0)
        {
            forget_combinations_biases_derivatives.setZero();
            input_combinations_biases_derivatives.setZero();
            state_combinations_biases_derivatives.setZero();
            output_combinations_biases_derivatives.setZero();

            cell_state_biases_derivatives.setZero();
        }
        else
        {
            previous_cell_state_activations = cell_state_activations.get_row(instance-1);

            forget_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, forget_recurrent_weights);
            input_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, input_recurrent_weights).multiply_rows(current_input_derivatives);
            state_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, state_recurrent_weights).multiply_rows(current_state_derivatives);
            output_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, output_recurrent_weights).multiply_rows(current_output_derivatives);
        }

        forget_combinations_biases_derivatives.sum_diagonal(1.0);

        cell_state_biases_derivatives = cell_state_biases_derivatives.multiply_rows(current_forget_activations)
                                      + input_combinations_biases_derivatives.multiply_rows(current_state_activations)
                                      + state_combinations_biases_derivatives.multiply_rows(current_input_activations)
                                      + forget_combinations_biases_derivatives.multiply_rows((current_forget_derivatives*previous_cell_state_activations));

        hidden_states_biases_derivatives =
                (output_combinations_biases_derivatives.multiply_rows(calculate_activations(current_cell_state_activations)) +
                cell_state_biases_derivatives.multiply_rows(current_output_activations*current_hidden_derivatives));

        forget_biases_error_gradient += dot(hidden_states_biases_derivatives, current_layer_deltas).to_vector();
    }

    return forget_biases_error_gradient;
    */
    return Tensor<type, 1>();
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_input_biases_error_gradient(const Tensor<type, 2>& inputs,
                                                                               const Layer::ForwardPropagation& layers,
                                                                               const Tensor<type, 2>& deltas,
                                                                               const Tensor<type, 2>& activations_states)
{
/*
    const int instances_number = inputs.dimension(0);
    const int neurons_number = get_neurons_number();
    const int parameters_number = neurons_number;

   Tensor<type, 1> input_biases_error_gradient(parameters_number, 0.0);

   Tensor<type, 2> input_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> forget_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> state_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> output_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);

   Tensor<type, 2> hidden_states_biases_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> cell_state_biases_derivatives(parameters_number, neurons_number,  0.0);

   const Tensor<type, 2> forget_activations = activations_states.get_matrix(0);
   const Tensor<type, 2> input_activations = activations_states.get_matrix(1);
   const Tensor<type, 2> state_activations = activations_states.get_matrix(2);
   const Tensor<type, 2> output_activations = activations_states.get_matrix(3);
   const Tensor<type, 2> cell_state_activations = activations_states.get_matrix(4);

   const Tensor<type, 2> forget_derivatives = layers.activations_derivatives.get_matrix(0);
   const Tensor<type, 2> input_derivatives = layers.activations_derivatives.get_matrix(1);
   const Tensor<type, 2> state_derivatives = layers.activations_derivatives.get_matrix(2);
   const Tensor<type, 2> output_derivatives = layers.activations_derivatives.get_matrix(3);
   const Tensor<type, 2> hidden_derivatives = layers.activations_derivatives.get_matrix(4);

   for(int instance = 0; instance < instances_number; instance++)
   {
       const Tensor<type, 2> current_layer_deltas = deltas.get_row(instance).to_column_matrix();

       const Tensor<type, 1> current_forget_activations = forget_activations.get_row(instance);
       const Tensor<type, 1> current_input_activations = input_activations.get_row(instance);
       const Tensor<type, 1> current_state_activations = state_activations.get_row(instance);
       const Tensor<type, 1> current_output_activations = output_activations.get_row(instance);
       const Tensor<type, 1> current_cell_state_activations = cell_state_activations.get_row(instance);

       const Tensor<type, 1> current_forget_derivatives = forget_derivatives.get_row(instance);
       const Tensor<type, 1> current_input_derivatives = input_derivatives.get_row(instance);
       const Tensor<type, 1> current_state_derivatives = state_derivatives.get_row(instance);
       const Tensor<type, 1> current_output_derivatives = output_derivatives.get_row(instance);
       const Tensor<type, 1> current_hidden_derivatives = hidden_derivatives.get_row(instance);

       Tensor<type, 1> previous_cell_state_activations(neurons_number, 0.0);

       if(instance%timesteps == 0)
       {
           forget_combinations_biases_derivatives.setZero();
           input_combinations_biases_derivatives.setZero();
           state_combinations_biases_derivatives.setZero();
           output_combinations_biases_derivatives.setZero();

           cell_state_biases_derivatives.setZero();
       }
       else
       {
           previous_cell_state_activations = cell_state_activations.get_row(instance-1);

           forget_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, forget_recurrent_weights).multiply_rows(current_forget_derivatives);
           input_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, input_recurrent_weights);
           state_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, state_recurrent_weights).multiply_rows(current_state_derivatives);
           output_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, output_recurrent_weights).multiply_rows(current_output_derivatives);
       }

       input_combinations_biases_derivatives.sum_diagonal(1.0);

       cell_state_biases_derivatives = cell_state_biases_derivatives.multiply_rows(current_forget_activations)
                                      + forget_combinations_biases_derivatives.multiply_rows(previous_cell_state_activations)
                                      + state_combinations_biases_derivatives.multiply_rows(current_input_activations)
                                      + input_combinations_biases_derivatives.multiply_rows((current_input_derivatives*current_state_activations));

       hidden_states_biases_derivatives =
               (output_combinations_biases_derivatives.multiply_rows(calculate_activations(current_cell_state_activations)) +
               cell_state_biases_derivatives.multiply_rows(current_output_activations*current_hidden_derivatives));

       input_biases_error_gradient += dot(hidden_states_biases_derivatives, current_layer_deltas).to_vector();
   }

   return input_biases_error_gradient;
*/
    return Tensor<type, 1>();
}



Tensor<type, 1> LongShortTermMemoryLayer::calculate_state_biases_error_gradient(const Tensor<type, 2>& inputs,
                                                                               const Layer::ForwardPropagation& layers,
                                                                               const Tensor<type, 2>& deltas,
                                                                               const Tensor<type, 2>& activations_states)
{
/*
    const int instances_number = inputs.dimension(0);
    const int neurons_number = get_neurons_number();
    const int parameters_number = neurons_number;

   Tensor<type, 1> state_biases_error_gradient(parameters_number, 0.0);

   Tensor<type, 2> input_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> forget_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> state_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> output_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);

   Tensor<type, 2> hidden_states_biases_derivatives(parameters_number, neurons_number,  0.0);
   Tensor<type, 2> cell_state_biases_derivatives(parameters_number, neurons_number,  0.0);

   const Tensor<type, 2> forget_activations = activations_states.get_matrix(0);
   const Tensor<type, 2> input_activations = activations_states.get_matrix(1);
   const Tensor<type, 2> state_activations = activations_states.get_matrix(2);
   const Tensor<type, 2> output_activations = activations_states.get_matrix(3);
   const Tensor<type, 2> cell_state_activations = activations_states.get_matrix(4);

   const Tensor<type, 2> forget_derivatives = layers.activations_derivatives.get_matrix(0);
   const Tensor<type, 2> input_derivatives = layers.activations_derivatives.get_matrix(1);
   const Tensor<type, 2> state_derivatives = layers.activations_derivatives.get_matrix(2);
   const Tensor<type, 2> output_derivatives = layers.activations_derivatives.get_matrix(3);
   const Tensor<type, 2> hidden_derivatives = layers.activations_derivatives.get_matrix(4);

   for(int instance = 0; instance < instances_number; instance++)
   {
       const Tensor<type, 2> current_layer_deltas = deltas.get_row(instance).to_column_matrix();

       const Tensor<type, 1> current_forget_activations = forget_activations.get_row(instance);
       const Tensor<type, 1> current_input_activations = input_activations.get_row(instance);
       const Tensor<type, 1> current_state_activations = state_activations.get_row(instance);
       const Tensor<type, 1> current_output_activations = output_activations.get_row(instance);
       const Tensor<type, 1> current_cell_state_activations = cell_state_activations.get_row(instance);

       const Tensor<type, 1> current_forget_derivatives = forget_derivatives.get_row(instance);
       const Tensor<type, 1> current_input_derivatives = input_derivatives.get_row(instance);
       const Tensor<type, 1> current_state_derivatives = state_derivatives.get_row(instance);
       const Tensor<type, 1> current_output_derivatives = output_derivatives.get_row(instance);
       const Tensor<type, 1> current_hidden_derivatives = hidden_derivatives.get_row(instance);

       Tensor<type, 1> previous_cell_state_activations(neurons_number, 0.0);

       if(instance%timesteps == 0)
       {
           forget_combinations_biases_derivatives.setZero();
           input_combinations_biases_derivatives.setZero();
           state_combinations_biases_derivatives.setZero();
           output_combinations_biases_derivatives.setZero();

           cell_state_biases_derivatives.setZero();
       }
       else
       {
           previous_cell_state_activations = cell_state_activations.get_row(instance-1);

           forget_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, forget_recurrent_weights).multiply_rows(current_forget_derivatives);
           input_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, input_recurrent_weights).multiply_rows(current_input_derivatives);
           state_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, state_recurrent_weights);
           output_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, output_recurrent_weights).multiply_rows(current_output_derivatives);
       }

       state_combinations_biases_derivatives.sum_diagonal(1.0);

       cell_state_biases_derivatives = cell_state_biases_derivatives.multiply_rows(current_forget_activations)
                                      + forget_combinations_biases_derivatives.multiply_rows(previous_cell_state_activations)
                                      + input_combinations_biases_derivatives.multiply_rows(current_state_activations)
                                      + state_combinations_biases_derivatives.multiply_rows((current_state_derivatives*current_input_activations));

       hidden_states_biases_derivatives =
               (output_combinations_biases_derivatives.multiply_rows(calculate_activations(current_cell_state_activations)) +
               cell_state_biases_derivatives.multiply_rows(current_output_activations*current_hidden_derivatives));

       state_biases_error_gradient += dot(hidden_states_biases_derivatives, current_layer_deltas).to_vector();
   }

   return state_biases_error_gradient;
*/
    return Tensor<type, 1>();
}


Tensor<type, 1> LongShortTermMemoryLayer::calculate_output_biases_error_gradient(const Tensor<type, 2>& inputs,
                                                                                const Layer::ForwardPropagation& layers,
                                                                                const Tensor<type, 2>& deltas,
                                                                                const Tensor<type, 2>& activations_states)
{
/*
    const int instances_number = inputs.dimension(0);
    const int neurons_number = get_neurons_number();
    const int parameters_number = neurons_number;

    Tensor<type, 1> output_biases_error_gradient(parameters_number, 0.0);

    Tensor<type, 2> input_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> forget_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> state_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> output_combinations_biases_derivatives(parameters_number, neurons_number,  0.0);

    Tensor<type, 2> hidden_states_biases_derivatives(parameters_number, neurons_number,  0.0);
    Tensor<type, 2> cell_state_biases_derivatives(parameters_number, neurons_number,  0.0);

    const Tensor<type, 2> forget_activations = activations_states.get_matrix(0);
    const Tensor<type, 2> input_activations = activations_states.get_matrix(1);
    const Tensor<type, 2> state_activations = activations_states.get_matrix(2);
    const Tensor<type, 2> output_activations = activations_states.get_matrix(3);
    const Tensor<type, 2> cell_state_activations = activations_states.get_matrix(4);

    const Tensor<type, 2> forget_derivatives = layers.activations_derivatives.get_matrix(0);
    const Tensor<type, 2> input_derivatives = layers.activations_derivatives.get_matrix(1);
    const Tensor<type, 2> state_derivatives = layers.activations_derivatives.get_matrix(2);
    const Tensor<type, 2> output_derivatives = layers.activations_derivatives.get_matrix(3);
    const Tensor<type, 2> hidden_derivatives = layers.activations_derivatives.get_matrix(4);

    for(int instance = 0; instance < instances_number; instance++)
    {
        const Tensor<type, 2> current_layer_deltas = deltas.get_row(instance).to_column_matrix();

        const Tensor<type, 1> current_forget_activations = forget_activations.get_row(instance);
        const Tensor<type, 1> current_input_activations = input_activations.get_row(instance);
        const Tensor<type, 1> current_state_activations = state_activations.get_row(instance);
        const Tensor<type, 1> current_output_activations = output_activations.get_row(instance);
        const Tensor<type, 1> current_cell_state_activations = cell_state_activations.get_row(instance);

        const Tensor<type, 1> current_forget_derivatives = forget_derivatives.get_row(instance);
        const Tensor<type, 1> current_input_derivatives = input_derivatives.get_row(instance);
        const Tensor<type, 1> current_state_derivatives = state_derivatives.get_row(instance);
        const Tensor<type, 1> current_output_derivatives = output_derivatives.get_row(instance);
        const Tensor<type, 1> current_hidden_derivatives = hidden_derivatives.get_row(instance);

        Tensor<type, 1> previous_cell_state_activations(neurons_number,0.0);

        if(instance%timesteps == 0)
        {
            forget_combinations_biases_derivatives.setZero();
            input_combinations_biases_derivatives.setZero();
            state_combinations_biases_derivatives.setZero();
            output_combinations_biases_derivatives.setZero();

            cell_state_biases_derivatives.setZero();
        }
        else
        {
            previous_cell_state_activations = cell_state_activations.get_row(instance-1);

            forget_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, forget_recurrent_weights).multiply_rows(current_forget_derivatives);
            input_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, input_recurrent_weights).multiply_rows(current_input_derivatives);
            state_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, state_recurrent_weights).multiply_rows(current_state_derivatives);
            output_combinations_biases_derivatives = dot(hidden_states_biases_derivatives, output_recurrent_weights);
        }

        output_combinations_biases_derivatives.sum_diagonal(1.0);

        cell_state_biases_derivatives = cell_state_biases_derivatives.multiply_rows(current_forget_activations)
                                      + forget_combinations_biases_derivatives.multiply_rows(previous_cell_state_activations)
                                      + state_combinations_biases_derivatives.multiply_rows(current_input_activations)
                                      + input_combinations_biases_derivatives.multiply_rows(current_state_activations);

        hidden_states_biases_derivatives =
                (output_combinations_biases_derivatives.multiply_rows(current_output_derivatives*calculate_activations(current_cell_state_activations)) +
                cell_state_biases_derivatives.multiply_rows(current_output_activations*current_hidden_derivatives));

        output_biases_error_gradient += dot(hidden_states_biases_derivatives, current_layer_deltas).to_vector();
    }

    return output_biases_error_gradient;
*/
    return Tensor<type, 1>();
}


/// Returns a string with the expression of the inputs-outputs relationship of the layer.
/// @param inputs_names Vector of strings with the name of the layer inputs. 
/// @param outputs_names Vector of strings with the name of the layer outputs. 

string LongShortTermMemoryLayer::write_expression(const vector<string>& inputs_names, const vector<string>& outputs_names) const
{
/*
    const int neurons_number = get_neurons_number();

    const int inputs_number = get_inputs_number();

   #ifdef __OPENNN_DEBUG__ 

   const int inputs_name_size = inputs_names.size();

   if(inputs_name_size != inputs_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
             << "string write_expression(const vector<string>&, const vector<string>&) const method.\n"
             << "Size of inputs name must be equal to number of layer inputs.\n";

	  throw logic_error(buffer.str());
   }

   const int outputs_name_size = outputs_names.size();

   if(outputs_name_size != neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: LongShortTermMemoryLayer class.\n"
             << "string write_expression(const vector<string>&, const vector<string>&) const method.\n"
             << "Size of outputs name must be equal to number of neurons.\n";

	  throw logic_error(buffer.str());
   }

   #endif

   ostringstream buffer;

   // Forget gate

   for(int j = 0; j < neurons_number; j++)
   {
       buffer << "forget_gate_" << to_string(j+1) << " = " << write_recurrent_activation_function_expression() << " (" << forget_biases[j] << "+";

       for(int i = 0; i < inputs_number; i++)
       {
           buffer << inputs_names[i] << "*" << forget_weights.get_column(j)[i] << "+";
       }

       for(int k = 0; k < neurons_number-1; k++)
       {
           buffer << "hidden_state_" << to_string(k+1) << "(t-1)*" << forget_recurrent_weights.get_column(j)[k] << "+";
       }

       buffer << "hidden_state_" << to_string(neurons_number) << "(t-1)*" << forget_recurrent_weights.get_column(j)[neurons_number-1] << ");\n";
   }

   // Input gate

   for(int j = 0; j < neurons_number; j++)
   {
       buffer << "input_gate_" << to_string(j+1) << " = " << write_recurrent_activation_function_expression() << " (" << input_biases[j] << "+";

       for(int i = 0; i < inputs_number; i++)
       {
           buffer << inputs_names[i] << "*" << input_weights.get_column(j)[i] << "+";
       }

       for(int k = 0; k < neurons_number-1; k++)
       {
           buffer << "hidden_state_" << to_string(k+1) << "(t-1)*" << input_recurrent_weights.get_column(j)[k] << "+";
       }

       buffer << "hidden_state_" << to_string(neurons_number) << "(t-1)*" << input_recurrent_weights.get_column(j)[neurons_number-1] << ");\n";
   }

   // State gate

   for(int j = 0; j < neurons_number; j++)
   {
       buffer << "state_gate_" << to_string(j+1) << " = " << write_activation_function_expression() << " (" << state_biases[j] << "+";

       for(int i = 0; i < inputs_number; i++)
       {
           buffer << inputs_names[i] << "*" << state_weights.get_column(j)[i] << "+";
       }

       for(int k = 0; k < neurons_number-1; k++)
       {
           buffer << "hidden_state_" << to_string(k+1) << "(t-1)*" << state_recurrent_weights.get_column(j)[k] << "+";
       }

       buffer << "hidden_state_" << to_string(neurons_number) << "(t-1)*" << state_recurrent_weights.get_column(j)[neurons_number-1] << ");\n";
   }

   // Output gate

   for(int j = 0; j < neurons_number; j++)
   {
       buffer << "output_gate_" << to_string(j+1) << " = " << write_recurrent_activation_function_expression() << " (" << output_biases[j] << "+";

       for(int i = 0; i < inputs_number; i++)
       {
           buffer << inputs_names[i] << "*" << output_weights.get_column(j)[i] << "+";
       }

       for(int k = 0; k < neurons_number-1; k++)
       {
           buffer << "hidden_state_" << to_string(k+1) << "(t-1)*" << output_recurrent_weights.get_column(j)[k] << "+";
       }

       buffer << "hidden_state_" << to_string(neurons_number) << "(t-1)*" << output_recurrent_weights.get_column(j)[neurons_number-1] << ");\n";
   }

   // Cell state

   for(int i = 0; i < neurons_number; i++)
   {
        buffer << "cell_state_" << to_string(i+1) << "(t) = forget_gate_" << to_string(i+1) << "*cell_state_" << to_string(i+1) << "(t-1)+input_gate_" << to_string(i+1) << "*state_gate_" << to_string(i+1) << ";\n";
   }

   // Hidden state

   for(int i = 0; i < neurons_number; i++)
   {
        buffer << "hidden_state_" << to_string(i+1) << "(t) = output_gate_" << to_string(i+1) << "*" << write_activation_function_expression() << "(cell_state_" << to_string(i+1) << ");\n";
   }

   // Output

   for(int i = 0; i < neurons_number; i++)
   {
       buffer << outputs_names[i] << " = " << "hidden_state_" << to_string(i+1) << "(t);\n";
   }

   return buffer.str();
*/
    return string();
}


string LongShortTermMemoryLayer::object_to_string() const
{
    const int inputs_number = get_inputs_number();
    const int neurons_number = get_neurons_number();

    ostringstream buffer;

    buffer << "Neuron layer" << endl;
    buffer << "Inputs number: " << inputs_number << endl;
    buffer << "Activation function: " << write_activation_function() << endl;
    buffer << "neurons number: " << neurons_number << endl;
    buffer << "Biases:\n " << get_biases() << endl;
    buffer << "Weights:\n" << get_weights()<<endl;
    buffer << " Recurrent weights:\n" << get_recurrent_weights()<<endl;
    buffer << "Hidden states:\n" << hidden_states << endl;
    buffer << "Cell states:\n" << cell_states << endl;

    return buffer.str();
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
