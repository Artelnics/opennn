//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   C L A S S                           
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "perceptron_layer.h"

namespace OpenNN
{

/// Default constructor. 
/// It creates a empty layer object, with no perceptrons.
/// This constructor also initializes the rest of class members to their default values.

PerceptronLayer::PerceptronLayer() : Layer()
{
   set();

   layer_type = Perceptron;
}


/// Layer architecture constructor. 
/// It creates a layer object with given numbers of inputs and perceptrons. 
/// The parameters are initialized at random. 
/// This constructor also initializes the rest of class members to their default values.
/// @param new_inputs_number Number of inputs in the layer.
/// @param new_neurons_number Number of perceptrons in the layer.

PerceptronLayer::PerceptronLayer(const int& new_inputs_number, const int& new_neurons_number,
                                 const PerceptronLayer::ActivationFunction& new_activation_function) : Layer()
{
   set(new_inputs_number, new_neurons_number, new_activation_function);

   layer_type = Perceptron;
}
 

/// Copy constructor. 
/// It creates a copy of an existing perceptron layer object. 
/// @param other_perceptron_layer Perceptron layer object to be copied.

PerceptronLayer::PerceptronLayer(const PerceptronLayer& other_perceptron_layer) : Layer()
{
   set(other_perceptron_layer);

   layer_type = Perceptron;
}


/// Destructor.
/// This destructor does not delete any pointer.

PerceptronLayer::~PerceptronLayer()
{
}


Tensor<int, 1> PerceptronLayer::get_input_variables_dimensions() const
{
    const int inputs_number = get_inputs_number();

    return Tensor<int, 1>(inputs_number);
}


/// Returns the number of inputs to the layer.

int PerceptronLayer::get_inputs_number() const
{
    return synaptic_weights.dimension(0);

}


/// Returns the number of neurons in the layer.

int PerceptronLayer::get_neurons_number() const
{
    return biases.size();
}


int PerceptronLayer::get_biases_number() const
{
    return biases.size();
}


int PerceptronLayer::get_synaptic_weights_number() const
{
    return synaptic_weights.size();
}

/// Returns the number of parameters(biases and synaptic weights) of the layer.

int PerceptronLayer::get_parameters_number() const
{
    return biases.size() + synaptic_weights.size();
}


/// Returns the biases from all the perceptrons in the layer. 
/// The format is a vector of real values. 
/// The size of this vector is the number of neurons in the layer.

const Tensor<type, 2>& PerceptronLayer::get_biases() const
{   
   return biases;
}


/// Returns the synaptic weights from the perceptrons. 
/// The format is a matrix of real values. 
/// The number of rows is the number of neurons in the layer. 
/// The number of columns is the number of inputs to the layer. 

const Tensor<type, 2>& PerceptronLayer::get_synaptic_weights() const
{
   return synaptic_weights;
}


Tensor<type, 2> PerceptronLayer::get_synaptic_weights(const Tensor<type, 1>& parameters) const
{
    const int inputs_number = get_inputs_number();
    const int neurons_number = get_neurons_number();

    const int synaptic_weights_number = synaptic_weights.size();
/*
    return parameters.get_first(synaptic_weights_number).to_matrix(inputs_number, neurons_number);
*/
    return Tensor<type, 2>();
}


Tensor<type, 2> PerceptronLayer::get_synaptic_weights_transpose() const
{
/*
    return synaptic_weights.transpose();
*/
    return Tensor<type, 2>();
}


Tensor<type, 1> PerceptronLayer::get_biases(const Tensor<type, 1>& parameters) const
{
    const int biases_number = biases.size();
/*
    return parameters.get_last(biases_number);
*/
    return Tensor<type, 1>();
}


/// Returns a single vector with all the layer parameters. 
/// The format is a vector of real values. 
/// The size is the number of parameters in the layer. 

Tensor<type, 1> PerceptronLayer::get_parameters() const
{
//    return synaptic_weights.to_vector().assemble(biases);

    return Tensor<type, 1>();
}


/// Returns the activation function of the layer.
/// The activation function of a layer is the activation function of all perceptrons in it.

const PerceptronLayer::ActivationFunction& PerceptronLayer::get_activation_function() const
{
    return activation_function;
}


/// Returns a string with the name of the layer activation function.
/// This can be: Logistic, HyperbolicTangent, Threshold, SymmetricThreshold, Linear, RectifiedLinear, ScaledExponentialLinear.

string PerceptronLayer::write_activation_function() const
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


/// Returns true if messages from this class are to be displayed on the screen, 
/// or false if messages from this class are not to be displayed on the screen.

const bool& PerceptronLayer::get_display() const
{
   return display;
}


/// Sets an empty layer, wihtout any perceptron.
/// It also sets the rest of members to their default values. 

void PerceptronLayer::set()
{
    biases.resize(0, 0);

    synaptic_weights.resize(0, 0);

   set_default();
}


/// Sets new numbers of inputs and perceptrons in the layer.
/// It also sets the rest of members to their default values. 
/// @param new_inputs_number Number of inputs.
/// @param new_neurons_number Number of perceptron neurons.

void PerceptronLayer::set(const int& new_inputs_number, const int& new_neurons_number,
                          const PerceptronLayer::ActivationFunction& new_activation_function)
{
    biases = Tensor<type, 2>(1, new_neurons_number);

    biases.setRandom();

    synaptic_weights = Tensor<type, 2>(new_inputs_number, new_neurons_number);

    synaptic_weights.setRandom();

    activation_function = new_activation_function;

    set_default();
}


/// Sets the members of this perceptron layer object with those from other perceptron layer object. 
/// @param other_perceptron_layer PerceptronLayer object to be copied.

void PerceptronLayer::set(const PerceptronLayer& other_perceptron_layer)
{   
   biases = other_perceptron_layer.biases;

   synaptic_weights = other_perceptron_layer.synaptic_weights;

   activation_function = other_perceptron_layer.activation_function;

   display = other_perceptron_layer.display;

   set_default();
}


/// Sets those members not related to the vector of perceptrons to their default value. 
/// <ul>
/// <li> Display: True.
/// <li> layer_type: Perceptron_Layer.
/// <li> trainable: True.
/// </ul> 

void PerceptronLayer::set_default()
{
   display = true;

   layer_type = Perceptron;
}


/// Sets a new number of inputs in the layer. 
/// The new synaptic weights are initialized at random. 
/// @param new_inputs_number Number of layer inputs.
 
void PerceptronLayer::set_inputs_number(const int& new_inputs_number)
{
    const int neurons_number = get_neurons_number();

    biases.resize(1, neurons_number);

    synaptic_weights.resize(new_inputs_number, neurons_number);

}


/// Sets a new number perceptrons in the layer. 
/// All the parameters are also initialized at random.
/// @param new_neurons_number New number of neurons in the layer.

void PerceptronLayer::set_neurons_number(const int& new_neurons_number)
{    
    const int inputs_number = get_inputs_number();

    biases.resize(1, new_neurons_number);

    synaptic_weights.resize(inputs_number, new_neurons_number);
}


/// Sets the biases of all perceptrons in the layer from a single vector.
/// @param new_biases New set of biases in the layer. 

void PerceptronLayer::set_biases(const Tensor<type, 2>& new_biases)
{
    biases = new_biases;
}


/// Sets the synaptic weights of this perceptron layer from a single matrix.
/// The format is a matrix of real numbers. 
/// The number of rows is the number of neurons in the corresponding layer. 
/// The number of columns is the number of inputs to the corresponding layer. 
/// @param new_synaptic_weights New set of synaptic weights in that layer. 

void PerceptronLayer::set_synaptic_weights(const Tensor<type, 2>& new_synaptic_weights)
{
    synaptic_weights = new_synaptic_weights;
}


/// Sets the parameters of this layer. 
/// @param new_parameters Parameters vector for that layer. 

void PerceptronLayer::set_parameters(const Tensor<type, 1>& new_parameters)
{
    const int neurons_number = get_neurons_number();
    const int inputs_number = get_inputs_number();

    const int parameters_number = get_parameters_number();

   #ifdef __OPENNN_DEBUG__ 

    const int new_parameters_size = new_parameters.size();

   if(new_parameters_size != parameters_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "void set_parameters(const Tensor<type, 1>&) method.\n"
             << "Size of new parameters (" << new_parameters_size << ") must be equal to number of parameters (" << parameters_number << ").\n";

	  throw logic_error(buffer.str());
   }

   #endif

/*
   synaptic_weights = new_parameters.get_subvector(0, inputs_number*neurons_number-1).to_matrix(inputs_number, neurons_number);

   biases = new_parameters.get_subvector(inputs_number*neurons_number, parameters_number-1);
*/
}


/// This class sets a new activation(or transfer) function in a single layer. 
/// @param new_activation_function Activation function for the layer.

void PerceptronLayer::set_activation_function(const PerceptronLayer::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


/// Sets a new activation(or transfer) function in a single layer. 
/// The second argument is a string containing the name of the function("Logistic", "HyperbolicTangent", "Threshold", etc).
/// @param new_activation_function Activation function for that layer. 

void PerceptronLayer::set_activation_function(const string& new_activation_function_name)
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

       buffer << "OpenNN Exception: PerceptronLayer class.\n"
              << "void set_activation_function(const string&) method.\n"
              << "Unknown activation function: " << new_activation_function_name << ".\n";

       throw logic_error(buffer.str());
    }
}


/// Sets a new display value. 
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void PerceptronLayer::set_display(const bool& new_display)
{
   display = new_display;
}


/// Makes the perceptron layer to have one more input.

void PerceptronLayer::grow_input()
{
    const int new_inputs_number = get_inputs_number() + 1;

    set_inputs_number(new_inputs_number);
}


/// Makes the perceptron layer to have one more perceptron.

void PerceptronLayer::grow_perceptron()
{
    const int new_neurons_number = get_neurons_number() + 1;

    set_neurons_number(new_neurons_number);
}


/// Makes the perceptron layer to have perceptrons_added more perceptrons.
/// @param neurons_added Number of perceptrons to be added.

void PerceptronLayer::grow_perceptrons(const int& neurons_added)
{
    const int new_neurons_number = get_neurons_number() + neurons_added;

    set_neurons_number(new_neurons_number);
}


/// This method removes a given input from the layer of perceptrons.
/// @param index Index of input to be pruned.

void PerceptronLayer::prune_input(const int& index)
{
    #ifdef __OPENNN_DEBUG__

    const int inputs_number = get_inputs_number();

    if(index >= inputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: PerceptronLayer class.\n"
              << "void prune_input(const int&) method.\n"
              << "Index of input is equal or greater than number of inputs.\n";

       throw logic_error(buffer.str());
    }

    #endif    
/*
    synaptic_weights = synaptic_weights.delete_row(index);
*/
}


/// This method removes a given perceptron from the layer.
/// @param index Index of perceptron to be pruned.

void PerceptronLayer::prune_neuron(const int& index)
{
    #ifdef __OPENNN_DEBUG__

    const int neurons_number = get_neurons_number();

    if(index >= neurons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: PerceptronLayer class.\n"
              << "void prune_neuron(const int&) method.\n"
              << "Index of perceptron is equal or greater than number of perceptrons.\n";

       throw logic_error(buffer.str());
    }

    #endif
/*
    biases = biases.delete_index(index);

    synaptic_weights = synaptic_weights.delete_column(index);
*/
}


/// Initializes the biases of all the perceptrons in the layer of perceptrons with a given value. 
/// @param value Biases initialization value. 

void PerceptronLayer::initialize_biases(const double& value)
{
/*
    biases.setConstant(value);
*/
}


/// Initializes the synaptic weights of all the perceptrons in the layer of perceptrons with a given value.
/// @param value Synaptic weights initialization value. 

void PerceptronLayer::initialize_synaptic_weights(const double& value) 
{
/*
    synaptic_weights.setConstant(value);
*/
}


/// Initializes the synaptic weights of all the perceptrons in the layer of perceptrons with glorot uniform distribution.

void PerceptronLayer::initialize_synaptic_weights_glorot_uniform()
{
    int fan_in;
    int fan_out;

    double scale = 1.0;
    double limit;

    fan_in = synaptic_weights.dimension(0);
    fan_out = synaptic_weights.dimension(1);

    scale /= ((fan_in + fan_out) / 2.0);
    limit = sqrt(3.0 * scale);
/*
    synaptic_weights.setRandom(-limit, limit);
*/
}


/// Initializes all the biases and synaptic weights in the neural newtork with a given value.
/// @param value Parameters initialization value. 

void PerceptronLayer::initialize_parameters(const double& value)
{
/*
    biases.setConstant(value);

    synaptic_weights.setConstant(value);
*/
}


/// Initializes all the biases and synaptic weights in the neural newtork at random with values comprised 
/// between -1 and +1.

void PerceptronLayer::randomize_parameters_uniform()
{
/*
   biases.setRandom(-1.0, 1.0);

   synaptic_weights.setRandom(-1.0, 1.0);
*/
}


/// Initializes all the biases and synaptic weights in the layer of perceptrons at random with values 
/// comprised between a minimum and a maximum values.
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

void PerceptronLayer::randomize_parameters_uniform(const double& minimum, const double& maximum)
{
/*
    biases.setRandom(minimum, maximum);

    synaptic_weights.setRandom(minimum, maximum);
*/
}


/// Initializes all the biases and synaptic weights in the newtork with random values chosen from a 
/// normal distribution with mean 0 and standard deviation 1.

void PerceptronLayer::randomize_parameters_normal()
{
/*
    biases.setRandom();

    synaptic_weights.setRandom();
*/
}


/// Initializes all the biases and synaptic weights in the layer of perceptrons with random random values 
/// chosen from a normal distribution with a given mean and a given standard deviation.
/// @param mean Mean of normal distribution.
/// @param standard_deviation Standard deviation of normal distribution.

void PerceptronLayer::randomize_parameters_normal(const double& mean, const double& standard_deviation)
{
    biases.setRandom();

    synaptic_weights.setRandom();
}


/// Calculates the norm of a layer parameters vector. 

double PerceptronLayer::calculate_parameters_norm() const
{
/*
   return l2_norm(get_parameters());
*/
    return 0.0;
}


Tensor<type, 2> PerceptronLayer::calculate_combinations(const Tensor<type, 2>& inputs) const
{
/*
    return linear_combinations(inputs, synaptic_weights, biases);
*/
    return Tensor<type, 2>();
}


Tensor<type, 2> PerceptronLayer::calculate_combinations(const Tensor<type, 2>& inputs, const Tensor<type, 1>& parameters) const
{
/*
    const Tensor<type, 2> new_synaptic_weights = get_synaptic_weights(parameters);
    const Tensor<type, 1> new_biases = get_biases(parameters);

    return calculate_combinations(inputs, new_biases, new_synaptic_weights);
*/
    return Tensor<type, 2>();
}


Tensor<type, 2> PerceptronLayer::calculate_combinations(const Tensor<type, 2>& inputs, const Tensor<type, 1>& new_biases, const Tensor<type, 2>& new_synaptic_weights) const
{
/*
    return linear_combinations(inputs, new_synaptic_weights, new_biases);
*/
    return Tensor<type, 2>();
}


Tensor<type, 2> PerceptronLayer::calculate_activations(const Tensor<type, 2>& combinations) const
{
    #ifdef __OPENNN_DEBUG__

    const int neurons_number = get_neurons_number();

    const int combinations_columns_number = combinations.dimension(1);

    if(combinations_columns_number != neurons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: PerceptronLayer class.\n"
              << "Tensor<type, 2> calculate_activations(const Tensor<type, 2>&) const method.\n"
              << "Number of combinations columns (" << combinations_columns_number << ") must be equal to number of neurons (" << neurons_number << ").\n";

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

    return Tensor<type, 2>();
}


Tensor<type, 2> PerceptronLayer::calculate_activations_derivatives(const Tensor<type, 2>& combinations) const
{
    #ifdef __OPENNN_DEBUG__

    const int neurons_number = get_neurons_number();

    const int combinations_columns_number = combinations.dimension(1);

    if(combinations_columns_number != neurons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: PerceptronLayer class.\n"
              << "Tensor<type, 2> calculate_activations_derivatives(const Tensor<type, 2>&) const method.\n"
              << "Number of combinations columns (" << combinations_columns_number << ") must be equal to number of neurons (" << neurons_number << ").\n";

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

    return Tensor<type, 2>();
}


Tensor<type, 2> PerceptronLayer::calculate_outputs(const Tensor<type, 2>& inputs)
{
    const int inputs_dimensions_number = inputs.rank();

    #ifdef __OPENNN_DEBUG__

    if(inputs_dimensions_number > 4)
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: PerceptronLayer class.\n"
           << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
           << "Number of dimensions (" << inputs_dimensions_number << ") must be less than or equal to 4.\n";

    throw logic_error(buffer.str());
    }

    #endif
/*
    Tensor<type, 2> reshaped_inputs = inputs.to_2d_tensor();

   #ifdef __OPENNN_DEBUG__

   const int inputs_number = get_inputs_number();

   const int inputs_columns_number = reshaped_inputs.dimension(1);

   if(inputs_columns_number != inputs_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
             << "Number of columns (" << inputs_columns_number << ") must be equal to number of inputs (" << inputs_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

    Tensor<type, 2> outputs = linear_combinations(reshaped_inputs, synaptic_weights, biases);

    switch(activation_function)
    {
        case PerceptronLayer::Linear:
        {
             // do nothing
        }
        break;

        case PerceptronLayer::HyperbolicTangent:
        {
             transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return tanh(value);});
        }
        break;

       case PerceptronLayer::Logistic:
       {
            transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return 1.0 / (1.0 + exp(-value));});
       }
       break;

       case PerceptronLayer::Threshold:
       {
            transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return value < 0.0 ? 0.0 : 1.0;});
       }
       break;

       case PerceptronLayer::SymmetricThreshold:
       {
            transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return value < 0.0 ? -1.0 : 1.0;});
       }
       break;

       case PerceptronLayer::RectifiedLinear:
       {
            transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return value < 0.0 ? 0.0 : value;});
       }
       break;

       case PerceptronLayer::ScaledExponentialLinear:
       {
            transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return value < 0.0 ? 1.0507 * 1.67326 * (exp(value) - 1.0) :  1.0507 * value;});
       }
       break;

       case PerceptronLayer::SoftPlus:
       {
            transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return log(1 + exp(value));});
       }
       break;

       case PerceptronLayer::SoftSign:
       {
            transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return value < 0.0 ?  value/(1.0-value) : value/(1.0 + value);});
       }
       break;

       case PerceptronLayer::ExponentialLinear:
       {
            transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return value < 0.0 ?  1.0 * (exp(value)- 1.0) : value;});
        }
       break;

       case PerceptronLayer::HardSigmoid:
       {
            transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){if(value < -2.5){return 0.0;}else if(value > 2.5){return 1.0;}else{return 0.2*value + 0.5;}});
       }
       break;

    }

    return outputs;
*/
    return Tensor<type, 2>();
}


Tensor<type, 2> PerceptronLayer::calculate_outputs(const Tensor<type, 2>& inputs, const Tensor<type, 1>& parameters)
{
/*
    const Tensor<type, 2> synaptic_weights = get_synaptic_weights(parameters);
    const Tensor<type, 1> biases = get_biases(parameters);

    return calculate_outputs(inputs, biases, synaptic_weights);
*/
    return Tensor<type, 2>();
}


Tensor<type, 2> PerceptronLayer::calculate_outputs(const Tensor<type, 2>& inputs, const Tensor<type, 1>& new_biases, const Tensor<type, 2>& new_synaptic_weights) const
{
    const int inputs_dimensions_number = inputs.rank();

    #ifdef __OPENNN_DEBUG__

    if(inputs_dimensions_number > 4)
    {
    ostringstream buffer;

    buffer << "OpenNN Exception: PerceptronLayer class.\n"
           << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
           << "Number of dimensions (" << inputs_dimensions_number << ") must be less than or equal to 4.\n";

    throw logic_error(buffer.str());
    }

    #endif
/*
    Tensor<type, 2> reshaped_inputs = inputs.to_2d_tensor();

   #ifdef __OPENNN_DEBUG__

   const int inputs_number = get_inputs_number();

   const int inputs_columns_number = reshaped_inputs.dimension(1);

   if(inputs_columns_number != inputs_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
             << "Number of columns (" << inputs_columns_number << ") must be equal to number of inputs (" << inputs_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   Tensor<type, 2> outputs(linear_combinations(reshaped_inputs, new_synaptic_weights, new_biases));

   switch(activation_function)
   {
       case Linear:
       {
             // Do nothing
       }
       break;

       case HyperbolicTangent:
       {
            transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return tanh(value);});
       }
       break;

      case Logistic:
      {
           transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return 1.0 / (1.0 + exp(-value));});
      }
      break;

      case Threshold:
      {
           transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return value < 0.0 ? 0.0 : 1.0;});
      }
      break;

      case SymmetricThreshold:
      {
           transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return value < 0.0 ? -1.0 : 1.0;});
      }
      break;

      case RectifiedLinear:
      {
           transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return value < 0.0 ? 0.0 : value;});
      }
      break;

      case ScaledExponentialLinear:
      {
           transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return value < 0.0 ? 1.0507 * 1.67326 * (exp(value) - 1.0) :  1.0507 * value;});
      }
      break;

      case SoftPlus:
      {
           transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return log(1 + exp(value));});
      }
      break;

      case SoftSign:
      {
           transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return value < 0.0 ?  value / (1 - value) :  value / (1 + value);});
      }
      break;

      case ExponentialLinear:
      {
           transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return value < 0.0 ?  1.0 * (exp(value)- 1) : value;});
      }
      break;

      case HardSigmoid:
      {
           transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){if(value < -2.5){return 0.0;}else if(value > 2.5){return 1.0;}else{return 0.2*value + 0.5;}});
      }
      break;
   }

   return outputs;
*/
    return Tensor<type, 2>();
}


Layer::ForwardPropagation PerceptronLayer::calculate_forward_propagation(const Tensor<type, 2>& inputs)
{
    ForwardPropagation forward_propagation;
/*
    Tensor<type, 2> combinations;

    if(inputs.rank() != 2)
    {
        combinations = calculate_combinations(inputs.to_2d_tensor());
    }
    else
    {
        combinations = calculate_combinations(inputs);
    }

    layers.activations = calculate_activations(combinations);

    layers.activations_derivatives = calculate_activations_derivatives(combinations);
*/
    return forward_propagation;
}


Tensor<type, 2> PerceptronLayer::calculate_output_delta(const Tensor<type, 2>& activations_derivatives, const Tensor<type, 2>& output_gradient) const
{
    return activations_derivatives*output_gradient;
}


Tensor<type, 2> PerceptronLayer::calculate_hidden_delta(Layer* next_layer_pointer,
                                                       const Tensor<type, 2>&,
                                                       const Tensor<type, 2>& activations_derivatives,
                                                       const Tensor<type, 2>& next_layer_delta) const
{

    const Type layer_type = next_layer_pointer->get_type();

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


/*
    return activations_derivatives*dot(next_layer_delta, synaptic_weights_transpose);
*/
    return Tensor<type, 2>();
}


/// Calculates the gradient error from the layer.
/// Returns the gradient of the objective, according to the objective type.
/// That gradient is the vector of partial derivatives of the objective with respect to the parameters.
/// The size is thus the number of parameters.
/// @param layer_deltas Tensor with layers delta.
/// @param inputs Tensor with layers inputs.

Tensor<type, 1> PerceptronLayer::calculate_error_gradient(const Tensor<type, 2>& inputs,
                                                         const Layer::ForwardPropagation& ,
                                                         const Tensor<type, 2>& deltas)
{
    const int inputs_number = get_inputs_number();
    const int neurons_number = get_neurons_number();

    const int parameters_number = get_parameters_number();

    const int synaptic_weights_number = neurons_number*inputs_number;

    Tensor<type, 1> layer_error_gradient(parameters_number);

    // Synaptic weights
/*
    layer_error_gradient.embed(0, dot(reshaped_inputs.to_matrix().calculate_transpose(), reshaped_deltas).to_vector());

    // Biases

    layer_error_gradient.embed(synaptic_weights_number, reshaped_deltas.to_matrix().calculate_columns_sum());
*/
    return layer_error_gradient;

}


/// Returns a string with the expression of the inputs-outputs relationship of the layer.
/// @param inputs_names vector of strings with the name of the layer inputs. 
/// @param outputs_names vector of strings with the name of the layer outputs. 

string PerceptronLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
   #ifdef __OPENNN_DEBUG__ 

   const int neurons_number = get_neurons_number();

   const int inputs_number = get_inputs_number(); 
   const int inputs_name_size = inputs_names.size();

   if(inputs_name_size != inputs_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const method.\n"
             << "Size of inputs name must be equal to number of layer inputs.\n";

	  throw logic_error(buffer.str());
   }

   const int outputs_name_size = outputs_names.size();

   if(outputs_name_size != neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const method.\n"
             << "Size of outputs name must be equal to number of perceptrons.\n";

	  throw logic_error(buffer.str());
   }

   #endif

   ostringstream buffer;

   for(int j = 0; j < outputs_names.size(); j++)
   {
/*
       buffer << outputs_names[j] << " = " << write_activation_function_expression() << " (" << biases[j] << "+";

       for(int i = 0; i < inputs_names.size() - 1; i++)
       {

           buffer << " (" << inputs_names[i] << "*" << synaptic_weights.get_column(j)[i] << ")+";
       }

       buffer << " (" << inputs_names[inputs_names.size() - 1] << "*" << synaptic_weights.get_column(j)[inputs_names.size() - 1] << "));\n";
*/
   }

   return buffer.str();
}


string PerceptronLayer::object_to_string() const
{
    const int inputs_number = get_inputs_number();
    const int neurons_number = get_neurons_number();

    ostringstream buffer;

    buffer << "Perceptron layer" << endl;
    buffer << "Inputs number: " << inputs_number << endl;
    buffer << "Activation function: " << write_activation_function() << endl;
    buffer << "Neurons number: " << neurons_number << endl;
    buffer << "Biases:\n " << biases << endl;
    buffer << "Synaptic_weights:\n" << synaptic_weights;

    return buffer.str();
}


void PerceptronLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Perceptron layer

    const tinyxml2::XMLElement* perceptron_layer_element = document.FirstChildElement("PerceptronLayer");

    if(!perceptron_layer_element)
    {
        buffer << "OpenNN Exception: PerceptronLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "PerceptronLayer element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = document.FirstChildElement("InputsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: PerceptronLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "InputsNumber element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(inputs_number_element->GetText())
    {
        set_inputs_number(static_cast<size_t>(stoi(inputs_number_element->GetText())));
    }

    // Neurons number

    const tinyxml2::XMLElement* neurons_number_element = document.FirstChildElement("NeuronsNumber");

    if(!neurons_number_element)
    {
        buffer << "OpenNN Exception: PerceptronLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "NeuronsNumber element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(neurons_number_element->GetText())
    {
        set_neurons_number(static_cast<size_t>(stoi(neurons_number_element->GetText())));
    }

    // Activation function

    const tinyxml2::XMLElement* activation_function_element = document.FirstChildElement("ActivationFunction");

    if(!activation_function_element)
    {
        buffer << "OpenNN Exception: PerceptronLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "ActivationFunction element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(activation_function_element->GetText())
    {
        set_activation_function(activation_function_element->GetText());
    }

    // Parameters

    const tinyxml2::XMLElement* parameters_element = document.FirstChildElement("Parameters");

    if(!parameters_element)
    {
        buffer << "OpenNN Exception: PerceptronLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Parameters element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(parameters_element->GetText())
    {
        const string parameters_string = parameters_element->GetText();
//@todo
//        set_parameters(to_double_vector(parameters_string, ' '));
    }
}


void PerceptronLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Perceptron layer

    file_stream.OpenElement("PerceptronLayer");

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
    buffer << get_parameters();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Peceptron layer (end tag)

    file_stream.CloseElement();
}


string PerceptronLayer::write_activation_function_expression() const
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
