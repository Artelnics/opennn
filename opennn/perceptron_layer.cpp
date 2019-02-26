/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P E R C E P T R O N   L A Y E R   C L A S S                                                                */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "perceptron_layer.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a empty layer object, with no perceptrons.
/// This constructor also initializes the rest of class members to their default values.

PerceptronLayer::PerceptronLayer()
{
   set();
}


// ARCHITECTURE CONSTRUCTOR

/// Layer architecture constructor. 
/// It creates a layer object with given numbers of inputs and perceptrons. 
/// The parameters are initialized at random. 
/// This constructor also initializes the rest of class members to their default values.
/// @param new_inputs_number Number of inputs in the layer.
/// @param new_perceptrons_number Number of perceptrons in the layer. 

PerceptronLayer::PerceptronLayer(const size_t& new_inputs_number, const size_t& new_perceptrons_number,
                                 const PerceptronLayer::ActivationFunction& new_activation_function)
{
   set(new_inputs_number, new_perceptrons_number, new_activation_function);
}
 

// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing perceptron layer object. 
/// @param other_perceptron_layer Perceptron layer object to be copied.

PerceptronLayer::PerceptronLayer(const PerceptronLayer& other_perceptron_layer)
{
   set(other_perceptron_layer);
}


// DESTRUCTOR

/// Destructor.
/// This destructor does not delete any pointer.

PerceptronLayer::~PerceptronLayer()
{
}


// ASSIGNMENT OPERATOR

/// Assignment operator. 
/// It assigns to this object the members of an existing perceptron layer object.
/// @param other_perceptron_layer Perceptron layer object to be assigned.

PerceptronLayer& PerceptronLayer::operator = (const PerceptronLayer& other_perceptron_layer)
{
   if(this != &other_perceptron_layer) 
   {
      display = other_perceptron_layer.display;
   }

   return(*this);
}


// EQUAL TO OPERATOR


/// Equal to operator. 
/// It compares this object with another object of the same class. 
/// It returns true if the members of the two objects have the same values, and false otherwise.
/// @ param other_perceptron_layer Perceptron layer to be compared with.

bool PerceptronLayer::operator == (const PerceptronLayer& other_perceptron_layer) const
{
   if(display == other_perceptron_layer.display)
   {
      return(true);
   }
   else
   {
      return(false);
   }
}


// METHODS


/// Returns the number of inputs to the layer.

size_t PerceptronLayer::get_inputs_number() const
{
    return synaptic_weights.get_rows_number();
}


/// Returns the size of the perceptrons vector.

size_t PerceptronLayer::get_perceptrons_number() const
{
   return biases.size();
}


/// Returns the number of parameters(biases and synaptic weights) of the layer.

size_t PerceptronLayer::get_parameters_number() const
{
   return biases.size() + synaptic_weights.size();
}


/// Returns the biases from all the perceptrons in the layer. 
/// The format is a vector of real values. 
/// The size of this vector is the number of neurons in the layer.

const Vector<double>& PerceptronLayer::get_biases() const
{   
   return(biases);
}


/// Returns the synaptic weights from the perceptrons. 
/// The format is a matrix of real values. 
/// The number of rows is the number of neurons in the layer. 
/// The number of columns is the number of inputs to the layer. 

const Matrix<double>& PerceptronLayer::get_synaptic_weights() const
{
   return(synaptic_weights);
}


Matrix<double> PerceptronLayer::get_synaptic_weights(const Vector<double>& parameters) const
{
    const size_t inputs_number = get_inputs_number();
    const size_t perceptrons_number = get_perceptrons_number();

    const size_t synaptic_weights_number = synaptic_weights.size();

    return parameters.get_first(synaptic_weights_number).to_matrix(inputs_number, perceptrons_number);
}


Vector<double> PerceptronLayer::get_biases(const Vector<double>& parameters) const
{
    const size_t biases_number = biases.size();

    return parameters.get_last(biases_number);
}



/// Returns a single vector with all the layer parameters. 
/// The format is a vector of real values. 
/// The size is the number of parameters in the layer. 

Vector<double> PerceptronLayer::get_parameters() const
{
/*
    const size_t parameters_number = get_parameters_number();

    const size_t inputs_number = get_inputs_number();
    const size_t perceptrons_number = get_perceptrons_number();

    Vector<double> parameters(parameters_number);

    size_t index = 0;

    for(size_t i = 0; i < perceptrons_number; i++)
    {
        parameters[index] = biases[i];

        index++;

        for(size_t j = 0; j < inputs_number; j++)
        {
            parameters[index] = synaptic_weights(j,i);

            index++;
        }
    }
*/    

    return synaptic_weights.to_vector().assemble(biases);
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
         return("Logistic");
      }

      case HyperbolicTangent:
      {
         return("HyperbolicTangent");
      }

      case Threshold:
      {
         return("Threshold");
      }

      case SymmetricThreshold:
      {
         return("SymmetricThreshold");
      }

      case Linear:
      {
         return("Linear");
      }

      case RectifiedLinear:
      {
         return("RectifiedLinear");
      }

      case ScaledExponentialLinear:
      {
         return("ScaledExponentialLinear");
      }

      case SoftPlus:
      {
         return("SoftPlus");
      }

      case SoftSign:
      {
         return("SoftSign");
      }

      case HardSigmoid:
      {
         return("HardSigmoid");
      }

      case ExponentialLinear:
      {
         return("ExponentialLinear");
      }

    }

    return string();
}


/// Returns true if messages from this class are to be displayed on the screen, 
/// or false if messages from this class are not to be displayed on the screen.

const bool& PerceptronLayer::get_display() const
{
   return(display);
}


/// Sets an empty layer, wihtout any perceptron.
/// It also sets the rest of members to their default values. 

void PerceptronLayer::set()
{
   set_default();
}


/// Sets new numbers of inputs and perceptrons in the layer.
/// It also sets the rest of members to their default values. 
/// @param new_inputs_number Number of inputs.
/// @param new_perceptrons_number Number of perceptron neurons.

void PerceptronLayer::set(const size_t& new_inputs_number, const size_t& new_perceptrons_number,
                          const PerceptronLayer::ActivationFunction& new_activation_function)
{
    biases.set(new_perceptrons_number);

    biases.randomize_normal();

    synaptic_weights.set(new_inputs_number, new_perceptrons_number);

    synaptic_weights.randomize_normal();
   
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
}


/// Sets those members not related to the vector of perceptrons to their default value. 
/// <ul>
/// <li> Display: True.
/// </ul> 

void PerceptronLayer::set_default()
{
   display = true;
}


/// Sets a new number of inputs in the layer. 
/// The new synaptic weights are initialized at random. 
/// @param new_inputs_number Number of layer inputs.
 
void PerceptronLayer::set_inputs_number(const size_t& new_inputs_number)
{
    const size_t perceptrons_number = get_perceptrons_number();

    biases.set(perceptrons_number);

    synaptic_weights.set(new_inputs_number, perceptrons_number);
}


/// Sets a new number perceptrons in the layer. 
/// All the parameters are also initialized at random.
/// @param new_perceptrons_number New number of neurons in the layer.

void PerceptronLayer::set_perceptrons_number(const size_t& new_perceptrons_number)
{    
    const size_t inputs_number = get_inputs_number();

    biases.set(new_perceptrons_number);
    synaptic_weights.set(inputs_number, new_perceptrons_number);
}


/// Sets the biases of all perceptrons in the layer from a single vector.
/// @param new_biases New set of biases in the layer. 

void PerceptronLayer::set_biases(const Vector<double>& new_biases)
{
    biases = new_biases;
}


/// Sets the synaptic weights of this perceptron layer from a single matrix.
/// The format is a matrix of real numbers. 
/// The number of rows is the number of neurons in the corresponding layer. 
/// The number of columns is the number of inputs to the corresponding layer. 
/// @param new_synaptic_weights New set of synaptic weights in that layer. 

void PerceptronLayer::set_synaptic_weights(const Matrix<double>& new_synaptic_weights)
{
    synaptic_weights = new_synaptic_weights;
}


/// Sets the parameters of this layer. 
/// @param new_parameters Parameters vector for that layer. 

void PerceptronLayer::set_parameters(const Vector<double>& new_parameters)
{
    const size_t perceptrons_number = get_perceptrons_number();
    const size_t inputs_number = get_inputs_number();

    const size_t parameters_number = get_parameters_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

    const size_t new_parameters_size = new_parameters.size();

   if(new_parameters_size != parameters_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "void set_parameters(const Vector<double>&) method.\n"
             << "Size of new parameters (" << new_parameters_size << ") must be equal to number of parameters (" << parameters_number << ").\n";

	  throw logic_error(buffer.str());
   }

   #endif

//   size_t index = 0;

//   for(size_t i = 0; i < perceptrons_number; i++)
//   {
//       biases[i] = new_parameters[index];
//       index++;

//       for(size_t j = 0; j < inputs_number; j++)
//       {
//           synaptic_weights(j,i) = new_parameters[index];
//           index++;
//       }
//   }

   synaptic_weights = new_parameters.get_subvector(0, inputs_number*perceptrons_number-1).to_matrix(inputs_number, perceptrons_number);

   biases = new_parameters.get_subvector(inputs_number*perceptrons_number, parameters_number-1);
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

       buffer << "OpenNN Exception: Perceptron class.\n"
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
}


/// Makes the perceptron layer to have one more perceptron.

void PerceptronLayer::grow_perceptron()
{
}


/// Makes the perceptron layer to have perceptrons_added more perceptrons.
/// @param perceptrons_added Number of perceptrons to be added.

void PerceptronLayer::grow_perceptrons(const size_t&)
{
}


/// This method removes a given input from the layer of perceptrons.
/// @param index Index of input to be pruned.
/// @todo

void PerceptronLayer::prune_input(const size_t& index)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = get_inputs_number();

    if(index >= inputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: PerceptronLayer class.\n"
              << "void prune_input(const size_t&) method.\n"
              << "Index of input is equal or greater than number of inputs.\n";

       throw logic_error(buffer.str());
    }

    #endif
}


/// This method removes a given perceptron from the layer.
/// @param index Index of perceptron to be pruned.
/// @todo

void PerceptronLayer::prune_perceptron(const size_t& index)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t perceptrons_number = get_perceptrons_number();

    if(index >= perceptrons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: PerceptronLayer class.\n"
              << "void prune_perceptron(const size_t&) method.\n"
              << "Index of perceptron is equal or greater than number of perceptrons.\n";

       throw logic_error(buffer.str());
    }

    #endif

}


/// Initializes the perceptron layer with a random number of inputs and a randon number of perceptrons.
/// That can be useful for testing purposes. 

void PerceptronLayer::initialize_random()
{
   const size_t inputs_number = rand()%10 + 1;
   const size_t perceptrons_number = rand()%10 + 1;

   set(inputs_number, perceptrons_number, PerceptronLayer::HyperbolicTangent);
   
   set_display(true);
}


/// Initializes the biases of all the perceptrons in the layer of perceptrons with a given value. 
/// @param value Biases initialization value. 

void PerceptronLayer::initialize_biases(const double& value)
{
    biases.initialize(value);
}


/// Initializes the synaptic weights of all the perceptrons in the layer of perceptrons perceptron with a given value. 
/// @param value Synaptic weights initialization value. 

void PerceptronLayer::initialize_synaptic_weights(const double& value) 
{
    synaptic_weights.initialize(value);
}

void PerceptronLayer::initialize_synaptic_weights_Glorot(const double& minimum,const double& maximum)
{
    synaptic_weights.randomize_uniform(minimum,maximum);
}


/// Initializes all the biases and synaptic weights in the neural newtork with a given value.
/// @param value Parameters initialization value. 

void PerceptronLayer::initialize_parameters(const double& value)
{
    biases.initialize(value);

    synaptic_weights.initialize(value);
}


/// Initializes all the biases and synaptic weights in the neural newtork at random with values comprised 
/// between -1 and +1.

void PerceptronLayer::randomize_parameters_uniform()
{
   biases.randomize_uniform();

   synaptic_weights.randomize_uniform();
}


/// Initializes all the biases and synaptic weights in the layer of perceptrons at random with values 
/// comprised between a minimum and a maximum values.
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

void PerceptronLayer::randomize_parameters_uniform(const double& minimum, const double& maximum)
{
    biases.randomize_uniform(minimum, maximum);

    synaptic_weights.randomize_uniform(minimum, maximum);
}


/// Initializes all the biases and synaptic weights in the layer of perceptrons at random, with values 
/// comprised between different minimum and maximum numbers for each parameter.
/// @param minimum Vector of minimum initialization values.
/// @param maximum Vector of maximum initialization values.

void PerceptronLayer::randomize_parameters_uniform(const Vector<double>& minimum, const Vector<double>& maximum)
{
    biases.randomize_uniform(minimum, maximum);

    synaptic_weights.randomize_uniform(minimum, maximum);
}


/// Initializes all the biases and synaptic weights in the layer of perceptrons at random, with values 
/// comprised between a different minimum and maximum numbers for each parameter.
/// All minimum are maximum initialization values must be given from a vector of two real vectors.
/// The first element must contain the minimum inizizalization value for each parameter.
/// The second element must contain the maximum inizizalization value for each parameter.
/// @param minimum_maximum Vector of minimum and maximum initialization values.

void PerceptronLayer::randomize_parameters_uniform(const Vector< Vector<double> >& minimum_maximum)
{
   const size_t parameters_number = get_parameters_number();

   Vector<double> parameters(parameters_number);

   parameters.randomize_uniform(minimum_maximum[0], minimum_maximum[1]);

   set_parameters(parameters);
}


/// Initializes all the biases and synaptic weights in the newtork with random values chosen from a 
/// normal distribution with mean 0 and standard deviation 1.

void PerceptronLayer::randomize_parameters_normal()
{
    biases.randomize_normal();

    synaptic_weights.randomize_normal();
}


/// Initializes all the biases and synaptic weights in the layer of perceptrons with random random values 
/// chosen from a normal distribution with a given mean and a given standard deviation.
/// @param mean Mean of normal distribution.
/// @param standard_deviation Standard deviation of normal distribution.

void PerceptronLayer::randomize_parameters_normal(const double& mean, const double& standard_deviation)
{
    biases.randomize_normal(mean, standard_deviation);

    synaptic_weights.randomize_normal(mean, standard_deviation);
}


/// Initializes all the biases an synaptic weights in the layer of perceptrons with random values chosen 
/// from normal distributions with different mean and standard deviation for each parameter.
/// @param mean Vector of mean values.
/// @param standard_deviation Vector of standard deviation values.

void PerceptronLayer::randomize_parameters_normal(const Vector<double>& mean, const Vector<double>& standard_deviation)
{
    biases.randomize_normal(mean, standard_deviation);

    synaptic_weights.randomize_normal(mean, standard_deviation);
}


/// Initializes all the biases and synaptic weights in the layer of perceptrons with random values chosen 
/// from normal distributions with different mean and standard deviation for each parameter.
/// All mean and standard deviation values are given from a vector of two real vectors.
/// The first element must contain the mean value for each parameter.
/// The second element must contain the standard deviation value for each parameter.
/// @param mean_standard_deviation Vector of mean and standard deviation values.

void PerceptronLayer::randomize_parameters_normal(const Vector< Vector<double> >& mean_standard_deviation)
{
   const size_t parameters_number = get_parameters_number();

   Vector<double> parameters(parameters_number);

   parameters.randomize_normal(mean_standard_deviation[0], mean_standard_deviation[1]);

   set_parameters(parameters);
}

/// Calculates the norm of a layer parameters vector. 

double PerceptronLayer::calculate_parameters_norm() const
{
   return(get_parameters().calculate_L2_norm());
}


Matrix<double> PerceptronLayer::calculate_combinations(const Matrix<double>& inputs) const
{
    return inputs.calculate_linear_combinations(synaptic_weights, biases);
}


Matrix<double> PerceptronLayer::calculate_combinations(const Matrix<double>& inputs, const Vector<double>& parameters) const
{
    const Matrix<double> new_synaptic_weights = get_synaptic_weights(parameters);
    const Vector<double> new_biases = get_biases(parameters);

    return calculate_combinations(inputs, new_biases, new_synaptic_weights);
}


Matrix<double> PerceptronLayer::calculate_combinations(const Matrix<double>& inputs, const Vector<double>& new_biases, const Matrix<double>& new_synaptic_weights) const
{
    return inputs.calculate_linear_combinations(new_synaptic_weights, new_biases);
}


Matrix<double> PerceptronLayer::calculate_activations(const Matrix<double>& combinations) const
{

    #ifdef __OPENNN_DEBUG__

    const size_t perceptrons_number = get_perceptrons_number();

    const size_t combinations_columns_number = combinations.get_columns_number();

    if(combinations_columns_number != perceptrons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: PerceptronLayer class.\n"
              << "Matrix<double> calculate_activations(const Matrix<double>&) const method.\n"
              << "Number of columns of combinations must be equal to number of neurons.\n";

       throw logic_error(buffer.str());
    }

    #endif

    switch(activation_function)
    {
        case PerceptronLayer::Linear:
        {
             return linear(combinations);
        }

        case PerceptronLayer::Logistic:
        {
             return logistic(combinations);
        }

        case PerceptronLayer::HyperbolicTangent:
        {
             return hyperbolic_tangent(combinations);
        }

        case PerceptronLayer::Threshold:
        {
             return threshold(combinations);
        }

        case PerceptronLayer::SymmetricThreshold:
        {
             return symmetric_threshold(combinations);
        }

        case PerceptronLayer::RectifiedLinear:
        {
             return rectified_linear(combinations);
        }

        case PerceptronLayer::ScaledExponentialLinear:
        {
             return scaled_exponential_linear(combinations);
        }

        case PerceptronLayer::SoftPlus:
        {
             return soft_plus(combinations);
        }

        case PerceptronLayer::SoftSign:
        {
             return soft_sign(combinations);
        }

        case PerceptronLayer::HardSigmoid:
        {
             return hard_sigmoid(combinations);
        }

        case PerceptronLayer::ExponentialLinear:
        {
             return exponential_linear(combinations);
        }
    }

    return Matrix<double>();
}


Matrix<double> PerceptronLayer::calculate_activations_derivatives(const Matrix<double>& combinations) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t perceptrons_number = get_perceptrons_number();

    const size_t combinations_columns_number = combinations.get_columns_number();

    if(combinations_columns_number != perceptrons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: PerceptronLayer class.\n"
              << "Matrix<double> calculate_activations_derivatives(const Matrix<double>&) const method.\n"
              << "Number of columns of combination must be equal to number of neurons.\n";

       throw logic_error(buffer.str());
    }

    #endif

    switch(activation_function)
    {
        case PerceptronLayer::Linear:
        {
             return linear_derivatives(combinations);
        }

        case PerceptronLayer::Logistic:
        {
             return logistic_derivatives(combinations);
        }

        case PerceptronLayer::HyperbolicTangent:
        {
             return hyperbolic_tangent_derivatives(combinations);
        }

        case PerceptronLayer::Threshold:
        {
             return threshold_derivatives(combinations);
        }

        case PerceptronLayer::SymmetricThreshold:
        {
             return symmetric_threshold_derivatives(combinations);
        }

        case PerceptronLayer::RectifiedLinear:
        {
             return rectified_linear_derivatives(combinations);
        }

        case PerceptronLayer::ScaledExponentialLinear:
        {
             return scaled_exponential_linear_derivate(combinations);
        }

        case PerceptronLayer::SoftPlus:
        {
             return soft_plus_derivatives(combinations);
        }

        case PerceptronLayer::SoftSign:
        {
             return soft_sign_derivatives(combinations);
        }

        case PerceptronLayer::HardSigmoid:
        {
             return hard_sigmoid_derivatives(combinations);
        }

        case PerceptronLayer::ExponentialLinear:
        {
             return exponential_linear_derivatives(combinations);
        }
    }

//    return Matrix<double>();
}


Matrix<double> PerceptronLayer::calculate_outputs(const Matrix<double>& inputs) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = get_inputs_number();

    const size_t inputs_columns_number = inputs.get_columns_number();

    if(inputs_columns_number != inputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: PerceptronLayer class.\n"
              << "Matrix<double> calculate_outputs(const Matrix<double>&) const method.\n"
              << "Number of columns of inputs matrix must be equal to number of inputs.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Matrix<double> outputs(inputs.calculate_linear_combinations(synaptic_weights, biases));

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
}


Matrix<double> PerceptronLayer::calculate_outputs(const Matrix<double>& inputs, const Vector<double>& new_biases, const Matrix<double>& new_synaptic_weights) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t inputs_size = inputs.get_columns_number();

   const size_t inputs_number = get_inputs_number();

   if(inputs_size != inputs_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "Vector<double> calculate_outputs(const Vector<double>&, const Vector<double>&) const method.\n"
             << "Size of layer inputs (" << inputs_size << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

  Matrix<double> outputs(inputs.calculate_linear_combinations(new_synaptic_weights, new_biases));

   switch(activation_function)
   {
       case PerceptronLayer::Linear:
       {
             // Do nothing
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
           transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return value < 0.0 ?  value / (1 - value) :  value / (1 + value);});
      }
      break;

      case PerceptronLayer::ExponentialLinear:
      {
           transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){return value < 0.0 ?  1.0 * (exp(value)- 1) : value;});
      }
      break;

      case PerceptronLayer::HardSigmoid:
      {
           transform(outputs.begin(), outputs.end(), outputs.begin(), [](const double &value){if(value < -2.5){return 0.0;}else if(value > 2.5){return 1.0;}else{return 0.2*value + 0.5;}});
      }
      break;
   }

   return outputs;
}


/// Returns the Jacobian matrix of a layer for a given inputs to that layer.
/// This is composed by the derivatives of the layer outputs with respect to their inputs.
/// The number of rows is the number of neurons in the layer.
/// The number of columns is the number of inputs to that layer.
/// @param inputs Input to layer.

Vector< Matrix<double> > PerceptronLayer::calculate_Jacobian(const Matrix<double>& inputs) const
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_number = get_inputs_number();
    const size_t inputs_size = inputs.size();

    if(inputs_size != inputs_number)
    {
       ostringstream buffer;
       buffer << "OpenNN Exception: PerceptronLayer class.\n"
              << "Matrix<double> calculate_Jacobian(const Vector<double>&) const method.\n"
              << "Size of inputs must be equal to number of inputs to layer.\n";
       throw logic_error(buffer.str());
    }
    #endif

    const Matrix<double> combinations = calculate_combinations(inputs);

    const Matrix<double> activations_derivatives = calculate_activations_derivatives(combinations);

    return(synaptic_weights.multiply_rows(activations_derivatives));
}



/// Returns a string with the expression of the inputs-outputs relationship of the layer.
/// @param inputs_name Vector of strings with the name of the layer inputs. 
/// @param outputs_name Vector of strings with the name of the layer outputs. 

string PerceptronLayer::write_expression(const Vector<string>& inputs_name, const Vector<string>& outputs_name) const
{
   const size_t perceptrons_number = get_perceptrons_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__ 

   const size_t inputs_number = get_inputs_number(); 
   const size_t inputs_name_size = inputs_name.size();

   if(inputs_name_size != inputs_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "string write_expression(const Vector<string>&, const Vector<string>&) const method.\n"
             << "Size of inputs name must be equal to number of layer inputs.\n";

	  throw logic_error(buffer.str());
   }

   const size_t outputs_name_size = outputs_name.size();

   if(outputs_name_size != perceptrons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: PerceptronLayer class.\n"
             << "string write_expression(const Vector<string>&, const Vector<string>&) const method.\n"
             << "Size of outputs name must be equal to number of perceptrons.\n";

	  throw logic_error(buffer.str());
   }

   #endif

   ostringstream buffer;

//   cout << "Synaptic Weights: " << synaptic_weights << endl;
//   cout << "Biases: " << biases << endl;
//   cout << "Activation Function: " << activation_function << ", " << write_activation_function_expression() << endl << endl;

   for(size_t j = 0; j < outputs_name.size(); j++)
   {
       buffer << outputs_name[j] << " = " << write_activation_function_expression() << "(" << biases[j] << "+";
       for(size_t i = 0; i < inputs_name.size() - 1; i++)
       {
           buffer << "(" << inputs_name[i] << "*" << synaptic_weights.get_column(j)[i] << ")+";
       }
       buffer << "(" << inputs_name[inputs_name.size() - 1] << "*" << synaptic_weights.get_column(j)[inputs_name.size() - 1] << "));\n";
   }

//   for(size_t i = 0; i < perceptrons_number; i++)
//   {
//      buffer << perceptrons[i].write_expression(inputs_name, outputs_name[i]);
//   }

   return(buffer.str());
}

string PerceptronLayer::object_to_string() const
{
    const size_t inputs_number = get_inputs_number();
    const size_t perceptrons_number = get_perceptrons_number();

    ostringstream buffer;

    buffer << "Perceptron layer" << endl;
    buffer << "Inputs number: " << inputs_number << endl;
    buffer << "Activation function: " << write_activation_function() << endl;
    buffer << "Perceptrons number: " << perceptrons_number << endl;
    buffer << "Biases:\n " << biases << endl;
    buffer << "Synaptic_weights:\n" << synaptic_weights;

    return buffer.str();
}

string PerceptronLayer::write_activation_function_expression() const
{
//    const string function = write_activation_function();
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
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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
