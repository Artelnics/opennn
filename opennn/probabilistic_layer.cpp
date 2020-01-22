//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   C L A S S                     
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "probabilistic_layer.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a probabilistic layer object with zero probabilistic neurons.
/// It does not has Synaptic weights or Biases

ProbabilisticLayer::ProbabilisticLayer()
{
    set();
}


/// Probabilistic neurons number constructor.
/// It creates a probabilistic layer with a given size.
/// @param new_neurons_number Number of neurons in the layer.

ProbabilisticLayer::ProbabilisticLayer(const int& new_inputs_number, const int& new_neurons_number)
{
    set(new_inputs_number, new_neurons_number);

    if(new_neurons_number > 1)
    {
        activation_function = Softmax;
    }
}


/// Copy constructor.
/// It creates a copy of an existing probabilistic layer object.
/// @param other_probabilistic_layer Probabilistic layer to be copied.

ProbabilisticLayer::ProbabilisticLayer(const ProbabilisticLayer& other_probabilistic_layer)
{
    set(other_probabilistic_layer);
}


/// Destructor.
/// This destructor does not delete any pointer.

ProbabilisticLayer::~ProbabilisticLayer()
{
}


Tensor<int, 1> ProbabilisticLayer::get_input_variables_dimensions() const
{
    const int inputs_number = get_inputs_number();

    return Tensor<int, 1>(inputs_number);
}


int ProbabilisticLayer::get_inputs_number() const
{
    return synaptic_weights.dimension(0);
}


int ProbabilisticLayer::get_neurons_number() const
{
    return biases.size();
}


/// Returns the decision threshold.

const double& ProbabilisticLayer::get_decision_threshold() const
{
    return decision_threshold;
}


/// Returns the method to be used for interpreting the outputs as probabilistic values.
/// The methods available for that are Binary, Probability, Competitive and Softmax.

const ProbabilisticLayer::ActivationFunction& ProbabilisticLayer::get_activation_function() const
{
    return activation_function;
}


/// Returns a string with the probabilistic method for the outputs
///("Competitive", "Softmax" or "NoProbabilistic").

string ProbabilisticLayer::write_activation_function() const
{
    if(activation_function == Binary)
    {
        return "Binary";
    }
    else if(activation_function == Logistic)
    {
        return "Logistic";
    }
    else if(activation_function == Competitive)
    {
        return "Competitive";
    }
    else if(activation_function == Softmax)
    {
        return "Softmax";
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "string write_activation_function() const method.\n"
               << "Unknown probabilistic method.\n";

        throw logic_error(buffer.str());
    }
}


/// Returns a string with the probabilistic method for the outputs to be included in some text
///("competitive", "softmax" or "no probabilistic").

string ProbabilisticLayer::write_activation_function_text() const
{
    if(activation_function == Binary)
    {
        return "binary";
    }
    else if(activation_function == Logistic)
    {
        return "logistic";
    }
    else if(activation_function == Competitive)
    {
        return "competitive";
    }
    else if(activation_function == Softmax)
    {
        return "softmax";
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "string write_activation_function_text() const method.\n"
               << "Unknown probabilistic method.\n";

        throw logic_error(buffer.str());
    }
}


/// Returns true if messages from this class are to be displayed on the screen, or false if messages
/// from this class are not to be displayed on the screen.

const bool& ProbabilisticLayer::get_display() const
{
    return display;
}


/// Returns the biases of the layer.

const Tensor<type, 1>& ProbabilisticLayer::get_biases() const
{
    return biases;
}


/// Returns the synaptic weights of the layer.

const Tensor<type, 2>& ProbabilisticLayer::get_synaptic_weights() const
{
    return synaptic_weights;
}


/// Returns the biases from a given vector of paramters for the layer.
/// @param parameters Parameters of the layer.

Tensor<type, 1> ProbabilisticLayer::get_biases(const Tensor<type, 1>& parameters) const
{
    const int biases_number = biases.size();
/*
    return parameters.get_last(biases_number);
*/
    return Tensor<type, 1>();
}


/// Returns the synaptic weights from a given vector of paramters for the layer.
/// @param parameters Parameters of the layer.

Tensor<type, 2> ProbabilisticLayer::get_synaptic_weights(const Tensor<type, 1>& parameters) const
{
    const int inputs_number = get_inputs_number();
    const int neurons_number = get_neurons_number();

    const int synaptic_weights_number = synaptic_weights.size();
/*
    return parameters.get_first(synaptic_weights_number).to_matrix(inputs_number, neurons_number);
*/
    return Tensor<type, 2>();

}


Tensor<type, 2> ProbabilisticLayer::get_synaptic_weights_transpose() const
{
    //return synaptic_weights.transpose();

    return Tensor<type, 2>();
}


/// Returns the number of parameters(biases and synaptic weights) of the layer.

int ProbabilisticLayer::get_parameters_number() const
{
    return biases.size() + synaptic_weights.size();
}


/// Returns a single vector with all the layer parameters.
/// The format is a vector of real values.
/// The size is the number of parameters in the layer.

Tensor<type, 1> ProbabilisticLayer::get_parameters() const
{
/*
    return synaptic_weights.to_vector().assemble(biases);
*/
    return Tensor<type, 1>();
}


/// Sets a probabilistic layer with zero probabilistic neurons.
/// It also sets the rest of members to their default values.

void ProbabilisticLayer::set()
{
/*
    biases.resize(0);
*/
    synaptic_weights.resize(0,0);

    set_default();
}


/// Resizes the size of the probabilistic layer.
/// It also sets the rest of class members to their default values.
/// @param new_neurons_number New size for the probabilistic layer.

void ProbabilisticLayer::set(const int& new_inputs_number, const int& new_neurons_number)
{
    biases.resize(new_neurons_number);

    biases.setRandom();
/*
    synaptic_weights.resize(new_inputs_number, new_neurons_number);
*/
    synaptic_weights.setRandom();

    set_default();
}


/// Sets this object to be equal to another object of the same class.
/// @param other_probabilistic_layer Probabilistic layer object to be copied.

void ProbabilisticLayer::set(const ProbabilisticLayer& other_probabilistic_layer)
{
    set_default();

    activation_function = other_probabilistic_layer.activation_function;

    decision_threshold = other_probabilistic_layer.decision_threshold;

    display = other_probabilistic_layer.display;
}


void ProbabilisticLayer::set_inputs_number(const int& new_inputs_number)
{
    const int neurons_number = get_neurons_number();

    biases.resize(neurons_number);
/*
    synaptic_weights.resize(new_inputs_number, neurons_number);
*/
}


void ProbabilisticLayer::set_neurons_number(const int& new_neurons_number)
{
    const int inputs_number = get_inputs_number();

    biases = Tensor<type, 1>(new_neurons_number);

    synaptic_weights = Tensor<type, 2>(inputs_number, new_neurons_number);
}


void ProbabilisticLayer::set_biases(const Tensor<type, 1>& new_biases)
{
    biases = Tensor<type, 1>(new_biases);
}


void ProbabilisticLayer::set_synaptic_weights(const Tensor<type, 2>& new_synaptic_weights)
{

    synaptic_weights = Tensor<type, 2>(new_synaptic_weights);

}


void ProbabilisticLayer::set_parameters(const Tensor<type, 1>& new_parameters)
{
    const int neurons_number = get_neurons_number();
    const int inputs_number = get_inputs_number();

    const int parameters_number = get_parameters_number();

   #ifdef __OPENNN_DEBUG__

    const int new_parameters_size = new_parameters.size();

   if(new_parameters_size != parameters_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
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


/// Sets a new threshold value for discriminating between two classes.
/// @param new_decision_threshold New discriminating value. It must be comprised between 0 and 1.

void ProbabilisticLayer::set_decision_threshold(const double& new_decision_threshold)
{
#ifdef __OPENNN_DEBUG__

    if(new_decision_threshold <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_decision_threshold(const double&) method.\n"
               << "Decision threshold(" << decision_threshold << ") must be greater than zero.\n";

        throw logic_error(buffer.str());
    }
    else if(new_decision_threshold >= 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_decision_threshold(const double&) method.\n"
               << "Decision threshold(" << decision_threshold << ") must be less than one.\n";

        throw logic_error(buffer.str());
    }

#endif

    decision_threshold = new_decision_threshold;
}


/// Sets the members to their default values:
/// <ul>
/// <li> Probabilistic method: Softmax.
/// <li> Display: True.
/// </ul>

void ProbabilisticLayer::set_default()
{
    layer_type = Probabilistic;

    const int neurons_number = get_neurons_number();

    if(neurons_number == 1)
    {
        activation_function = Logistic;
    }
    else
    {
        activation_function = Softmax;
    }

    decision_threshold = 0.5;

    display = true;
}


/// Sets the chosen method for probabilistic postprocessing.
/// Current probabilistic methods include Binary, Probability, Competitive and Softmax.
/// @param new_activation_function Method for interpreting the outputs as probabilistic values.

void ProbabilisticLayer::set_activation_function(const ActivationFunction& new_activation_function)
{
#ifdef __OPENNN_DEBUG__

    const int neurons_number = get_neurons_number();

    if(neurons_number == 1 && new_activation_function == Competitive)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_activation_function(const ActivationFunction&) method.\n"
               << "Activation function cannot be Competitive when the number of neurons is 1.\n";

        throw logic_error(buffer.str());
    }

    if(neurons_number == 1 && new_activation_function == Softmax)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_activation_function(const ActivationFunction&) method.\n"
               << "Activation function cannot be Softmax when the number of neurons is 1.\n";

        throw logic_error(buffer.str());
    }

    if(neurons_number != 1 && new_activation_function == Binary)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_activation_function(const ActivationFunction&) method.\n"
               << "Activation function cannot be Binary when the number of neurons is greater than 1.\n";

        throw logic_error(buffer.str());
    }

    if(neurons_number != 1 && new_activation_function == Logistic)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_activation_function(const ActivationFunction&) method.\n"
               << "Activation function cannot be Logistic when the number of neurons is greater than 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    activation_function = new_activation_function;
}


/// Sets a new method for probabilistic processing from a string with the name.
/// Current probabilistic methods include Competitive and Softmax.
/// @param new_activation_function Method for interpreting the outputs as probabilistic values.

void ProbabilisticLayer::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "Binary")
    {
        set_activation_function(Binary);
    }
    else if(new_activation_function == "Logistic")
    {
        set_activation_function(Logistic);
    }
    else if(new_activation_function == "Competitive")
    {
        set_activation_function(Competitive);
    }
    else if(new_activation_function == "Softmax")
    {
        set_activation_function(Softmax);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_activation_function(const string&) method.\n"
               << "Unknown probabilistic method: " << new_activation_function << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void ProbabilisticLayer::set_display(const bool& new_display)
{
    display = new_display;
}


/// Removes a probabilistic neuron from the probabilistic layer.

void ProbabilisticLayer::prune_neuron(const int& index)
{
/*
    biases = biases.delete_index(index);

    synaptic_weights = synaptic_weights.delete_column(index);
*/
}


/// Initializes the biases of all the neurons in the probabilistic layer with a given value.
/// @param value Biases initialization value.

void ProbabilisticLayer::initialize_biases(const double& value)
{
    biases.setConstant(value);
}


/// Initializes the synaptic weights of all the neurons in the probabilistic layer with a given value.
/// @param value Synaptic weights initialization value.

void ProbabilisticLayer::initialize_synaptic_weights(const double& value)
{
    synaptic_weights.setConstant(value);
}


void ProbabilisticLayer::initialize_synaptic_weights_Glorot(const double& minimum,const double& maximum)
{
    synaptic_weights.setRandom();
}


/// Initializes all the biases and synaptic weights in the neural newtork with a given value.
/// @param value Parameters initialization value.

void ProbabilisticLayer::initialize_parameters(const double& value)
{
    biases.setConstant(value);

    synaptic_weights.setConstant(value);
}


/// Initializes all the biases and synaptic weights in the neural newtork at random with values comprised
/// between -1 and +1.

void ProbabilisticLayer::randomize_parameters_uniform()
{
   biases.setRandom();

   synaptic_weights.setRandom();
}


/// Initializes all the biases and synaptic weights in the probabilistic layer at random with values
/// comprised between a minimum and a maximum values.
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

void ProbabilisticLayer::randomize_parameters_uniform(const double& minimum, const double& maximum)
{
    biases.setRandom();

    synaptic_weights.setRandom();

}


/// Initializes all the biases and synaptic weights in the newtork with random values chosen from a
/// normal distribution with mean 0 and standard deviation 1.

void ProbabilisticLayer::randomize_parameters_normal()
{
    biases.setRandom();

    synaptic_weights.setRandom();
}


/// Initializes all the biases and synaptic weights in the probabilistic layer with random random values
/// chosen from a normal distribution with a given mean and a given standard deviation.
/// @param mean Mean of normal distribution.
/// @param standard_deviation Standard deviation of normal distribution.

void ProbabilisticLayer::randomize_parameters_normal(const double& mean, const double& standard_deviation)
{
    biases.setRandom();

    synaptic_weights.setRandom();
}


Tensor<type, 2> ProbabilisticLayer::calculate_combinations(const Tensor<type, 2>& inputs) const
{
/*
    const int inputs_dimensions_number = inputs.rank();

    Tensor<type, 2> reshaped_inputs = inputs;

    if(inputs_dimensions_number != 2) reshaped_inputs = inputs.to_2d_tensor();

    return linear_combinations(reshaped_inputs, synaptic_weights, biases);
*/

    return Tensor<type, 2>();
}


/// This method processes the input to the probabilistic layer in order to obtain a set of outputs which
/// can be interpreted as probabilities.
/// This posprocessing is performed according to the probabilistic method to be used.
/// @param inputs Set of inputs to the probabilistic layer.

Tensor<type, 2> ProbabilisticLayer::calculate_outputs(const Tensor<type, 2>& inputs)
{
/*
    const int output_rows_number = inputs.dimension(0);
    const int output_columns_number = get_neurons_number();

    Tensor<type, 2> combinations(Tensor<int, 1>({output_rows_number, output_columns_number}));

    const int inputs_dimensions_number = inputs.rank();

    if(inputs_dimensions_number == 2)
    {
        combinations = linear_combinations(inputs, synaptic_weights, biases);
    }
    else
    {
        const Tensor<type, 2> reshaped_inputs = inputs.to_2d_tensor();

        combinations = linear_combinations(reshaped_inputs, synaptic_weights, biases);
    }

    switch(activation_function)
    {
        case Binary:
        {
            return binary(combinations);
        }

        case Logistic:
        {
            return logistic(combinations);
        }

        case Competitive:
        {
            return competitive(combinations);
        }

        case Softmax:
        {
            return softmax(combinations);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
           << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
           << "Unknown probabilistic method.\n";

    throw logic_error(buffer.str());
*/
    return Tensor<type, 2>();
}


/// This method processes the input to the probabilistic layer for a given set of parameters in order to obtain a set of outputs which
/// can be interpreted as probabilities.
/// This posprocessing is performed according to the probabilistic method to be used.
/// @param inputs Set of inputs to the probabilistic layer
/// @param parameters Set of parameters of the probabilistic layer

Tensor<type, 2> ProbabilisticLayer::calculate_outputs(const Tensor<type, 2>& inputs, const Tensor<type, 1>& parameters)
{
/*
    const Tensor<type, 2> synaptic_weights = get_synaptic_weights(parameters);
    const Tensor<type, 1> biases = get_biases(parameters);

    return calculate_outputs(inputs, biases, synaptic_weights);
*/

    return Tensor<type, 2>();
}


/// This method processes the input to the probabilistic layer for a given set of biases and synaptic weights in order to obtain a set of outputs which
/// can be interpreted as probabilities.
/// This posprocessing is performed according to the probabilistic method to be used.
/// @param inputs Set of inputs to the probabilistic layer
/// @param biases Set of biases of the probabilistic layer
/// @param synaptic_weights Set of synaptic weights of the probabilistic layer

Tensor<type, 2> ProbabilisticLayer::calculate_outputs(const Tensor<type, 2>& inputs, const Tensor<type, 1>& biases, const Tensor<type, 2>& synaptic_weights) const
{
#ifdef __OPENNN_DEBUG__

 const int inputs_dimensions_number = inputs.rank();

 if(inputs_dimensions_number > 4)
 {
    ostringstream buffer;

    buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
           << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 1>&, const Tensor<type, 2>&) const method.\n"
           << "Inputs dimensions number (" << inputs_dimensions_number << ") must not be greater than 4.\n";

    throw logic_error(buffer.str());
 }

#endif
/*
Tensor<type, 2> reshaped_inputs = inputs.to_2d_tensor();

#ifdef __OPENNN_DEBUG__

const int inputs_columns_number = reshaped_inputs.dimension(1);

const int inputs_number = get_inputs_number();

if(inputs_columns_number != inputs_number)
{
   ostringstream buffer;

   buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
          << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 1>&, const Tensor<type, 2>&) const method.\n"
          << "Size of layer inputs (" << inputs_columns_number << ") must be equal to number of layer inputs (" << inputs_number << ").\n";

   throw logic_error(buffer.str());
}

#endif

    Tensor<type, 2> combinations(linear_combinations(reshaped_inputs,synaptic_weights,biases));

    switch(activation_function)
    {
        case Binary:
        {
            return binary(combinations);
        }

        case Logistic:
        {

            return logistic(combinations);
        }

        case Competitive:
        {
            return competitive(combinations);
        }

        case Softmax:
        {
            return softmax(combinations);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
           << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&, const Tensor<type, 1>&, const Tensor<type, 2>&) const method.\n"
           << "Unknown probabilistic method.\n";

    throw logic_error(buffer.str());
*/
    return Tensor<type, 2>();
}


Layer::ForwardPropagation ProbabilisticLayer::calculate_forward_propagation(const Tensor<type, 2>& inputs)
{
    ForwardPropagation forward_propagation;

    const Tensor<type, 2> combinations = calculate_combinations(inputs);

    forward_propagation.activations = calculate_activations(combinations);
    forward_propagation.activations_derivatives = calculate_activations_derivatives(combinations);

    return forward_propagation;
}


Tensor<type, 2> ProbabilisticLayer::calculate_output_delta(const Tensor<type, 2>& activations_derivatives,
                                                          const Tensor<type, 2>& output_gradient) const
{/*
    const int neurons_number = get_neurons_number();

    if(neurons_number == 1)
    {
        return activations_derivatives*output_gradient;
    }
    else
    {
        return dot_2d_3d(output_gradient, activations_derivatives);
    }
    */
    return Tensor<type, 2>();
}


/// Calculates the gradient error from the layer.
/// Returns the gradient of the objective, according to the objective type.
/// That gradient is the vector of partial derivatives of the objective with respect to the parameters.
/// The size is thus the number of parameters.
/// @param layer_deltas Tensor with layers delta.
/// @param layer_inputs Tensor with layers inputs.

Tensor<type, 1> ProbabilisticLayer::calculate_error_gradient(const Tensor<type, 2>& layer_inputs,
                                                            const Layer::ForwardPropagation&,
                                                            const Tensor<type, 2>& layer_deltas)
{
/*
    Tensor<type, 2> reshaped_inputs = layer_inputs.to_2d_tensor();

    Tensor<type, 2> reshaped_deltas = layer_deltas.to_2d_tensor();

    const int inputs_number = get_inputs_number();
    const int neurons_number = get_neurons_number();

    const int synaptic_weights_number = neurons_number*inputs_number;

    const int parameters_number = get_parameters_number();

    Tensor<type, 1> error_gradient(parameters_number, 0.0);

    // Synaptic weights

    error_gradient.embed(0, dot(reshaped_inputs.to_matrix().calculate_transpose(), reshaped_deltas).to_vector());

    // Biases

    error_gradient.embed(synaptic_weights_number, reshaped_deltas.to_matrix().calculate_columns_sum());

    return error_gradient;
*/
    return Tensor<type, 1>();
}


/// Returns a string representation of the current probabilistic layer object.

string ProbabilisticLayer::object_to_string() const
{
    ostringstream buffer;

    buffer << "Probabilistic layer\n"
           << "Activation function: " << write_activation_function() << "\n";

    return buffer.str();
}


Tensor<type, 2> ProbabilisticLayer::calculate_activations(const Tensor<type, 2>& combinations) const
{
    #ifdef __OPENNN_DEBUG__

    const int dimensions_number = combinations.rank();

    if(dimensions_number != 2)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
              << "Tensor<type, 2> calculate_activations(const Tensor<type, 2>&) const method.\n"
              << "Dimensions of combinations (" << dimensions_number << ") must be 2.\n";

       throw logic_error(buffer.str());
    }

    const int neurons_number = get_neurons_number();

    const int combinations_columns_number = combinations.dimension(1);

    if(combinations_columns_number != neurons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
              << "Tensor<type, 2> calculate_activations(const Tensor<type, 2>&) const method.\n"
              << "Number of combinations columns (" << combinations_columns_number << ") must be equal to number of neurons (" << neurons_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    switch(activation_function)
    {
        case Binary: return binary(combinations);

        case Logistic: return logistic(combinations);

        case Competitive: return competitive(combinations);

        case Softmax: return softmax(combinations);
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
           << "Tensor<type, 2> calculate_activations(const Tensor<type, 2>&) const method.\n"
           << "Unknown probabilistic method.\n";

    throw logic_error(buffer.str());
}


Tensor<type, 2> ProbabilisticLayer::calculate_activations_derivatives(const Tensor<type, 2>& combinations) const
{
    #ifdef __OPENNN_DEBUG__

    const int dimensions_number = combinations.rank();

    if(dimensions_number != 2)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
              << "Tensor<type, 2> calculate_activations_derivatives(const Tensor<type, 2>&) const method.\n"
              << "Dimensions of combinations (" << dimensions_number << ") must be 2.\n";

       throw logic_error(buffer.str());
    }

    const int neurons_number = get_neurons_number();

    const int combinations_columns_number = combinations.dimension(1);

    if(combinations_columns_number != neurons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
              << "Tensor<type, 2> calculate_activations_derivatives(const Tensor<type, 2>&) const method.\n"
              << "Number of combinations columns (" << combinations_columns_number << ") must be equal to number of neurons (" << neurons_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    switch(activation_function)
    {
        case Binary:
        {
             ostringstream buffer;

             buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                    << "Binary derivative doesn't exist.\n";

             throw logic_error(buffer.str());
        }

        case Logistic:
        {
                return logistic_derivatives(combinations);
        }

        case Competitive:
        {
             ostringstream buffer;

             buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                    << "Competitive derivative doesn't exist.\n";

             throw logic_error(buffer.str());
        }

        case Softmax:
        {
            return softmax_derivatives(combinations);
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
           << "Tensor<type, 2> calculate_activations_derivatives(const Tensor<type, 2>&) const method.\n"
           << "Unknown probabilistic method.\n";

    throw logic_error(buffer.str());
}


/// Serializes the probabilistic layer object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this element.

tinyxml2::XMLDocument* ProbabilisticLayer::to_XML() const
{
    ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    tinyxml2::XMLElement* root_element = document->NewElement("ProbabilisticLayer");

    document->InsertFirstChild(root_element);

    tinyxml2::XMLElement* element = nullptr;
    tinyxml2::XMLText* text = nullptr;

    // Activation function
    {
        element = document->NewElement("ActivationFunction");
        root_element->LinkEndChild(element);

        text = document->NewText(write_activation_function().c_str());
        element->LinkEndChild(text);
    }

    // Probabilistic neurons number
    {
        element = document->NewElement("DecisionThreshold");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << decision_threshold;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Display
    //   {
    //      element = document->NewElement("Display");
    //      root_element->LinkEndChild(element);

    //      buffer.str("");
    //      buffer << display;

    //      text = document->NewText(buffer.str().c_str());
    //      element->LinkEndChild(text);
    //   }

    return document;
}


/// Serializes the probabilistic layer object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void ProbabilisticLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Probabilistic layer

    file_stream.OpenElement("ProbabilisticLayer");

    // Inputs number

    file_stream.OpenElement("InputsNumber");

    buffer.str("");
    buffer << get_inputs_number();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Neurons number

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

    // Decision threshold

    file_stream.OpenElement("DecisionThreshold");

    buffer.str("");
    buffer << decision_threshold;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Probabilistic layer (end tag)

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this probabilistic layer object.
/// @param document XML document containing the member data.

void ProbabilisticLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Probabilistic layer

    const tinyxml2::XMLElement* probabilistic_layer_element = document.FirstChildElement("ProbabilisticLayer");

    if(!probabilistic_layer_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Probabilistic layer element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = document.FirstChildElement("InputsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Inputs number element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(inputs_number_element->GetText())
    {
        set_inputs_number(static_cast<size_t>(stoi(inputs_number_element->GetText())));
    }

    // Neurons number

    const tinyxml2::XMLElement* neurons_number_element = document.FirstChildElement("NeuronsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Neurons number element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(neurons_number_element->GetText())
    {
        set_neurons_number(static_cast<size_t>(stoi(neurons_number_element->GetText())));
    }

    // Activation function

    const tinyxml2::XMLElement* activation_function_element = probabilistic_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Activation function element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(activation_function_element->GetText())
    {
        set_activation_function(activation_function_element->GetText());
    }

    // Parameters

    const tinyxml2::XMLElement* parameters_element = probabilistic_layer_element->FirstChildElement("Parameters");

    if(!parameters_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
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

    // Decision threshold

    const tinyxml2::XMLElement* decision_threshold_element = probabilistic_layer_element->FirstChildElement("DecisionThreshold");

    if(!decision_threshold_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Decision threshold element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(decision_threshold_element->GetText())
    {
        set_decision_threshold(atof(decision_threshold_element->GetText()));
    }

    // Display

    const tinyxml2::XMLElement* display_element = probabilistic_layer_element->FirstChildElement("Display");

    if(display_element)
    {
        string new_display_string = display_element->GetText();

        try
        {
            set_display(new_display_string != "0");
        }
        catch(const logic_error& e)
        {
            cerr << e.what() << endl;
        }
    }
}


/// Returns a string with the expression of the binary probabilistic outputs function.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

string ProbabilisticLayer::write_binary_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;

    buffer.str("");
/*
    buffer << outputs_names.vector_to_string(',') << " = binary(" << inputs_names.vector_to_string(',') << ");\n";
*/
    return buffer.str();
}


/// Returns a string with the expression of the probability outputs function.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

string ProbabilisticLayer::write_probability_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;
/*
    buffer << outputs_names.vector_to_string(',') << " = probability(" << inputs_names.vector_to_string(',') << ");\n";
*/
    return buffer.str();
}


/// Returns a string with the expression of the competitive probabilistic outputs function.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

string ProbabilisticLayer::write_competitive_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;
/*
    buffer << outputs_names.vector_to_string(',') << " = competitive(" << inputs_names.vector_to_string(',') << ");\n";
*/
    return buffer.str();
}


/// Returns a string with the expression of the softmax probabilistic outputs function.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

string ProbabilisticLayer::write_softmax_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;
/*
    buffer << outputs_names.vector_to_string(',') << " = softmax(" << inputs_names.vector_to_string(',') << ");\n";
*/
    return buffer.str();
}


/// Returns a string with the expression of the no probabilistic outputs function.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

string ProbabilisticLayer::write_no_probabilistic_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;
/*
    buffer << outputs_names.vector_to_string(',') << " = (" << inputs_names.vector_to_string(',') << ");\n";
*/
    return buffer.str();
}


/// Returns a string with the expression of the probabilistic outputs function,
/// depending on the probabilistic method to be used.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

string ProbabilisticLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    switch(activation_function)
    {
        case Binary:
        {
            return write_binary_expression(inputs_names, outputs_names);
        }

        case Logistic:
        {
            return(write_probability_expression(inputs_names, outputs_names));
        }

        case Competitive:
        {
            return(write_competitive_expression(inputs_names, outputs_names));
        }

        case Softmax:
        {
            return(write_softmax_expression(inputs_names, outputs_names));
        }

    }// end switch

    // Default

    ostringstream buffer;

    buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
           << "string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const method.\n"
           << "Unknown probabilistic method.\n";

    throw logic_error(buffer.str());
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
