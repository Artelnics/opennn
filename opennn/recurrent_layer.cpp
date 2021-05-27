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

Tensor<type, 1> RecurrentLayer::get_biases() const
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
    const Index parameters_number = get_parameters_number();

    Tensor<type, 1> parameters(parameters_number);

    Index current_position = 0;

    // Biases

    for(Index i = 0; i < biases.size(); i++) fill_n(parameters.data()+current_position+i, 1, biases(i));

    current_position += biases.size();

    // Weights

    for(Index i = 0; i < input_weights.size(); i++) fill_n(parameters.data()+current_position+i, 1, input_weights(i));

    current_position += input_weights.size();

    // Recurrent weights

    for(Index i = 0; i < recurrent_weights.size(); i++) fill_n(parameters.data()+current_position+i, 1, recurrent_weights(i));

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

    const Tensor<type, 1> new_inputs_weights
            = parameters.slice(Eigen::array<Eigen::Index, 1>({0}), Eigen::array<Eigen::Index, 1>({input_weights_number}));

    const Eigen::array<Index, 2> two_dim{{inputs_number, neurons_number}};

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

    const Tensor<type, 1> new_synaptic_weights
            = parameters.slice(Eigen::array<Eigen::Index, 1>({start_recurrent_weights_number}), Eigen::array<Eigen::Index, 1>({recurrent_weights_number}));

    const Eigen::array<Index, 2> two_dim{{neurons_number, neurons_number}};

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
    biases.resize(new_neurons_number);

    input_weights.resize(new_inputs_number, new_neurons_number);

    recurrent_weights.resize(new_neurons_number, new_neurons_number);

    hidden_states.resize(new_neurons_number); // memory

    hidden_states.setConstant(0.0);

    set_parameters_random();

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
    layer_name = "recurrent_layer";

    display = true;

    layer_type = Recurrent;
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

    biases.resize(new_neurons_number);

    input_weights.resize(inputs_number, new_neurons_number);

    recurrent_weights.resize(new_neurons_number, new_neurons_number);
}


void RecurrentLayer::set_timesteps(const Index& new_timesteps)
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

void RecurrentLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
#ifdef OPENNN_DEBUG

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

    const Index biases_number = get_biases_number();
    const Index inputs_weights_number = get_input_weights_number();
    const Index recurrent_weights_number = get_recurrent_weights_number();

    memcpy(biases.data(),
           new_parameters.data() + index,
           static_cast<size_t>(biases_number)*sizeof(type));

    memcpy(input_weights.data(),
           new_parameters.data() + index + biases_number,
           static_cast<size_t>(inputs_weights_number)*sizeof(type));

    memcpy(recurrent_weights.data(),
           new_parameters.data() + biases_number + inputs_weights_number + index,
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

void RecurrentLayer::initialize_input_weights_Glorot(const type&, const type&)
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
/// comprised between -1 and 1 values.

void RecurrentLayer::set_parameters_random()
{
    const type maximum = 1;
    const type minimum = -1;

    // Biases

    for(Index i = 0; i < biases.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        biases(i) = minimum + (maximum-minimum)*random;
    }

    // Weights

    for(Index i = 0; i < input_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        input_weights(i) = minimum + (maximum-minimum)*random;
    }

    // Recurrent weights

    for(Index i = 0; i < recurrent_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        recurrent_weights(i) = minimum + (maximum-minimum)*random;
    }
}


void RecurrentLayer::calculate_combinations(const Tensor<type, 1>& inputs,
                                                    const Tensor<type, 2>& input_weights,
                                                    const Tensor<type, 2>& recurrent_weights,
                                                    const Tensor<type, 1>& biases,
                                                    Tensor<type, 1>& combinations) const
{
    combinations.device(*thread_pool_device) = inputs.contract(input_weights, AT_B);

    combinations.device(*thread_pool_device) += biases;

    combinations.device(*thread_pool_device) += hidden_states.contract(recurrent_weights, AT_B);
}


void RecurrentLayer::calculate_activations(const Tensor<type, 1>& combinations_1d,
                                           Tensor<type, 1>& activations_1d) const
{
#ifdef OPENNN_DEBUG

const Index neurons_number = get_neurons_number();

const Index combinations_columns_number = combinations_1d.dimension(0);

if(combinations_columns_number != neurons_number)
{
ostringstream buffer;

buffer << "OpenNN Exception: RecurrentLayer class.\n"
       << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
       << "Number of combinations columns (" << combinations_columns_number
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


void RecurrentLayer::calculate_activations_derivatives(const Tensor<type, 1>& combinations_1d,
                                                       Tensor<type, 1>& activations_1d,
                                                       Tensor<type, 1>& activations_derivatives_1d) const
{
     #ifdef OPENNN_DEBUG

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


void RecurrentLayer::forward_propagate(const Tensor<type, 2>& inputs, LayerForwardPropagation* forward_propagation)
{
    RecurrentLayerForwardPropagation* recurrent_layer_forward_propagation = static_cast<RecurrentLayerForwardPropagation*>(forward_propagation);

    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();

    for(Index i = 0; i < samples_number; i++)
    {
        if(i%timesteps == 0) hidden_states.setZero();

        recurrent_layer_forward_propagation->current_inputs = inputs.chip(i, 0);

        calculate_combinations(recurrent_layer_forward_propagation->current_inputs,
                               input_weights,
                               recurrent_weights,
                               biases,
                               recurrent_layer_forward_propagation->current_combinations);

        calculate_activations_derivatives(recurrent_layer_forward_propagation->current_combinations,
                                          hidden_states,
                                          recurrent_layer_forward_propagation->current_activations_derivatives);

        for(Index j = 0; j < neurons_number; j++)
        {
            recurrent_layer_forward_propagation->combinations(i,j) = recurrent_layer_forward_propagation->current_combinations(j);
            recurrent_layer_forward_propagation->activations(i,j) = hidden_states(j);
            recurrent_layer_forward_propagation->activations_derivatives(i,j) = recurrent_layer_forward_propagation->current_activations_derivatives(j);
        }
    }
}


void RecurrentLayer::forward_propagate(const Tensor<type, 2>&inputs,
                                       Tensor<type, 1> parameters,
                                       LayerForwardPropagation* forward_propagation)
{
    RecurrentLayerForwardPropagation* recurrent_layer_forward_propagation
            = static_cast<RecurrentLayerForwardPropagation*>(forward_propagation);

    const Index neurons_number = get_neurons_number();
    const Index inputs_number = get_inputs_number();

    const TensorMap<Tensor<type, 1>> biases(parameters.data(), neurons_number);
    const TensorMap<Tensor<type, 2>> input_weights(parameters.data()+neurons_number, inputs_number, neurons_number);
    const TensorMap<Tensor<type, 2>> recurrent_weights(parameters.data()+neurons_number+inputs_number*neurons_number, neurons_number, neurons_number);

    const Index samples_number = inputs.dimension(0);

    for(Index i = 0; i < samples_number; i++)
    {
        if(i%timesteps == 0) hidden_states.setZero();

        recurrent_layer_forward_propagation->current_inputs = inputs.chip(i, 0);

        calculate_combinations(recurrent_layer_forward_propagation->current_inputs,
                               input_weights,
                               recurrent_weights,
                               biases,
                               recurrent_layer_forward_propagation->current_combinations);

        calculate_activations_derivatives(recurrent_layer_forward_propagation->current_combinations,
                                          hidden_states,
                                          recurrent_layer_forward_propagation->current_activations_derivatives);

        for(Index j = 0; j < neurons_number; j++)
        {
            recurrent_layer_forward_propagation->combinations(i,j)
                    = recurrent_layer_forward_propagation->current_combinations(j);

            recurrent_layer_forward_propagation->activations(i,j) = hidden_states(j);

            recurrent_layer_forward_propagation->activations_derivatives(i,j)
                    = recurrent_layer_forward_propagation->current_activations_derivatives(j);
        }
    }
}


Tensor<type, 2> RecurrentLayer::calculate_outputs(const Tensor<type, 2>& inputs)
{
#ifdef OPENNN_DEBUG

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

        calculate_combinations(current_inputs, input_weights, recurrent_weights, biases, current_outputs);

        calculate_activations(current_outputs, hidden_states);

        for(Index j = 0; j < neurons_number; j++)
            outputs(i,j) = hidden_states(j);
    }

    return outputs;
}


void RecurrentLayer::calculate_hidden_delta(LayerForwardPropagation* next_layer_forward_propagation,
                                            LayerBackPropagation* next_layer_back_propagation,
                                            LayerBackPropagation* current_layer_back_propagation) const
{
    RecurrentLayerBackPropagation* recurrent_layer_back_propagation =
            static_cast<RecurrentLayerBackPropagation*>(current_layer_back_propagation);

    switch(next_layer_back_propagation->layer_pointer->get_type())
    {
    case Perceptron:
    {
        PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation =
                static_cast<PerceptronLayerForwardPropagation*>(next_layer_forward_propagation);

        PerceptronLayerBackPropagation* perceptron_layer_back_propagation =
                static_cast<PerceptronLayerBackPropagation*>(next_layer_back_propagation);

        calculate_hidden_delta_perceptron(perceptron_layer_forward_propagation,
                                          perceptron_layer_back_propagation,
                                          recurrent_layer_back_propagation);
    }
        break;

    case Probabilistic:
    {
        ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation =
                static_cast<ProbabilisticLayerForwardPropagation*>(next_layer_forward_propagation);

        ProbabilisticLayerBackPropagation* probabilistic_layer_back_propagation =
                static_cast<ProbabilisticLayerBackPropagation*>(next_layer_back_propagation);

        calculate_hidden_delta_probabilistic(probabilistic_layer_forward_propagation,
                                             probabilistic_layer_back_propagation,
                                             recurrent_layer_back_propagation);
    }
        break;

    default: return;
    }
}


void RecurrentLayer::calculate_hidden_delta_perceptron(PerceptronLayerForwardPropagation* next_forward_propagation,
                                                       PerceptronLayerBackPropagation* next_back_propagation,
                                                       RecurrentLayerBackPropagation* back_propagation) const
{
    const Tensor<type, 2>& next_synaptic_weights
            = static_cast<PerceptronLayer*>(next_back_propagation->layer_pointer)->get_synaptic_weights();

    back_propagation->delta.device(*thread_pool_device) =
            (next_back_propagation->delta*next_forward_propagation->activations_derivatives).contract(next_synaptic_weights, A_BT);
}


void RecurrentLayer::calculate_hidden_delta_probabilistic(ProbabilisticLayerForwardPropagation* next_forward_propagation,
                                                          ProbabilisticLayerBackPropagation* next_back_propagation,
                                                          RecurrentLayerBackPropagation* back_propagation) const
{
    const ProbabilisticLayer* probabilistic_layer_pointer = static_cast<ProbabilisticLayer*>(next_back_propagation->layer_pointer);

    const Tensor<type, 2>& next_synaptic_weights = probabilistic_layer_pointer->get_synaptic_weights();

    if(probabilistic_layer_pointer->get_neurons_number() == 1) // Binary
    {
        back_propagation->delta.device(*thread_pool_device) =
                (next_back_propagation->delta*next_forward_propagation->activations_derivatives).contract(next_synaptic_weights, A_BT);
    }
    else // Multiple
    {
        const Index samples_number = next_back_propagation->delta.dimension(0);
        const Index outputs_number = next_back_propagation->delta.dimension(1);
        const Index next_layer_neurons_number = probabilistic_layer_pointer->get_neurons_number();

        if(outputs_number != next_layer_neurons_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                   << "void calculate_hidden_delta_probabilistic(ProbabilisticLayerForwardPropagation*,ProbabilisticLayerBackPropagation*,RecurrentLayerBackPropagation*) const.\n"
                   << "Number of columns in delta (" << outputs_number << ") must be equal to number of neurons in probabilistic layer (" << next_layer_neurons_number << ").\n";

            throw logic_error(buffer.str());
        }

        if(next_forward_propagation->activations_derivatives.dimension(1) != next_layer_neurons_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                   << "void calculate_hidden_delta_probabilistic(ProbabilisticLayerForwardPropagation*,ProbabilisticLayerBackPropagation*,RecurrentLayerBackPropagation*) const.\n"
                   << "Dimension 1 of activations derivatives (" << outputs_number << ") must be equal to number of neurons in probabilistic layer (" << next_layer_neurons_number << ").\n";

            throw logic_error(buffer.str());
        }

        if(next_forward_propagation->activations_derivatives.dimension(2) != next_layer_neurons_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                   << "void calculate_hidden_delta_probabilistic(ProbabilisticLayerForwardPropagation*,ProbabilisticLayerBackPropagation*,RecurrentLayerBackPropagation*) const.\n"
                   << "Dimension 2 of activations derivatives (" << outputs_number << ") must be equal to number of neurons in probabilistic layer (" << next_layer_neurons_number << ").\n";

            throw logic_error(buffer.str());
        }

        const Index step = next_layer_neurons_number*next_layer_neurons_number;

        next_back_propagation->biases_derivatives.setZero();

        for(Index i = 0; i < samples_number; i++)
        {
            next_back_propagation->delta_row = next_back_propagation->delta.chip(i,0);

            TensorMap< Tensor<type, 2> > activations_derivatives_matrix(next_forward_propagation->activations_derivatives.data() + i*step,
                                                                        next_layer_neurons_number, next_layer_neurons_number);

            next_back_propagation->error_combinations_derivatives.chip(i,0) =
                    next_back_propagation->delta_row.contract(activations_derivatives_matrix, AT_B);
        }

        back_propagation->delta.device(*thread_pool_device) =
                (next_back_propagation->error_combinations_derivatives).contract(next_synaptic_weights, A_BT);
    }
}


void RecurrentLayer::insert_gradient(LayerBackPropagation* back_propagation, const Index& index, Tensor<type, 1>& gradient) const
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    RecurrentLayerBackPropagation* recurrent_layer_back_propagation
            = static_cast<RecurrentLayerBackPropagation*>(back_propagation);

    // Biases

    memcpy(gradient.data() + index,
           recurrent_layer_back_propagation->biases_derivatives.data(),
           static_cast<size_t>(neurons_number)*sizeof(type));

    // Input weights

    memcpy(gradient.data() + index + neurons_number,
           recurrent_layer_back_propagation->input_weights_derivatives.data(),
           static_cast<size_t>(inputs_number*neurons_number)*sizeof(type));

    // Recurrent weights

    memcpy(gradient.data() + index + neurons_number + inputs_number*neurons_number,
           recurrent_layer_back_propagation->recurrent_weights_derivatives.data(),
           static_cast<size_t>(neurons_number*neurons_number)*sizeof(type));
}


void RecurrentLayer::calculate_error_gradient(const Tensor<type, 2>& inputs,
                                              LayerForwardPropagation* forward_propagation,
                                              LayerBackPropagation* back_propagation) const
{
    RecurrentLayerForwardPropagation* recurrent_layer_forward_propagation =
            static_cast<RecurrentLayerForwardPropagation*>(forward_propagation);

    RecurrentLayerBackPropagation* recurrent_layer_back_propagation =
            static_cast<RecurrentLayerBackPropagation*>(back_propagation);


    calculate_biases_error_gradient(inputs, recurrent_layer_forward_propagation, recurrent_layer_back_propagation);

    calculate_input_weights_error_gradient(inputs, recurrent_layer_forward_propagation, recurrent_layer_back_propagation);

    calculate_recurrent_weights_error_gradient(inputs, recurrent_layer_forward_propagation, recurrent_layer_back_propagation);

}


void RecurrentLayer::calculate_biases_error_gradient(const Tensor<type, 2>& inputs,
                                                     RecurrentLayerForwardPropagation* forward_propagation,
                                                     RecurrentLayerBackPropagation* back_propagation) const
{
    // Derivatives of combinations with respect to biases

    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = neurons_number;

    back_propagation->combinations_biases_derivatives.setZero();

    back_propagation->biases_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        back_propagation->current_layer_deltas = back_propagation->delta.chip(sample,0);

        if(sample%timesteps == 0)
        {
            back_propagation->combinations_biases_derivatives.setZero();
        }
        else
        {
            multiply_rows(back_propagation->combinations_biases_derivatives, forward_propagation->current_activations_derivatives);

            back_propagation->combinations_biases_derivatives
                    = back_propagation->combinations_biases_derivatives.contract(recurrent_weights, A_B).eval();
        }

        forward_propagation->current_activations_derivatives
                = forward_propagation->activations_derivatives.chip(sample, 0);

        for(Index i = 0; i < parameters_number; i++) back_propagation->combinations_biases_derivatives(i,i) += static_cast<type>(1);

        back_propagation->biases_derivatives += back_propagation->combinations_biases_derivatives
                .contract(back_propagation->current_layer_deltas*forward_propagation->current_activations_derivatives, A_B);
    }
}


void RecurrentLayer::calculate_input_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                            RecurrentLayerForwardPropagation* forward_propagation,
                                                            RecurrentLayerBackPropagation* back_propagation) const
{
    // Derivatives of combinations with respect to input weights

    const Index samples_number = inputs.dimension(0);
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();
    const Index parameters_number = inputs_number*neurons_number;

    Index column_index = 0;
    Index input_index = 0;

    back_propagation->combinations_weights_derivatives.setZero();
    back_propagation->input_weights_derivatives.setZero();

    for(Index sample = 0; sample < samples_number; sample++)
    {
        forward_propagation->current_inputs = inputs.chip(sample, 0);

        back_propagation->current_layer_deltas = back_propagation->delta.chip(sample, 0);

        if(sample%timesteps == 0)
        {
            back_propagation->combinations_weights_derivatives.setZero();
        }
        else
        {
            multiply_rows(back_propagation->combinations_weights_derivatives, forward_propagation->current_activations_derivatives);

            back_propagation->combinations_weights_derivatives
                    = back_propagation->combinations_weights_derivatives.contract(recurrent_weights, A_B).eval();
        }

        forward_propagation->current_activations_derivatives
                = forward_propagation->activations_derivatives.chip(sample, 0);

        column_index = 0;
        input_index = 0;

        for(Index i = 0; i < parameters_number; i++)
        {
            back_propagation->combinations_weights_derivatives(i, column_index) += forward_propagation->current_inputs(input_index);

            input_index++;

            if(input_index == inputs_number)
            {
                input_index = 0;
                column_index++;
            }
        }

        back_propagation->input_weights_derivatives += back_propagation->combinations_weights_derivatives
                .contract(back_propagation->current_layer_deltas*forward_propagation->current_activations_derivatives, A_B);
    }
}


void RecurrentLayer::calculate_recurrent_weights_error_gradient(const Tensor<type, 2>& inputs,
                                                                RecurrentLayerForwardPropagation* forward_propagation,
                                                                RecurrentLayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs.dimension(0);
    const Index neurons_number = get_neurons_number();

    const Index parameters_number = neurons_number*neurons_number;

    // Derivatives of combinations with respect to recurrent weights

    back_propagation->combinations_recurrent_weights_derivatives.setZero();

    back_propagation->recurrent_weights_derivatives.setZero();

    Index column_index = 0;
    Index activation_index = 0;

    for(Index sample = 0; sample < samples_number; sample++)
    {
        back_propagation->current_layer_deltas = back_propagation->delta.chip(sample,0);

        if(sample%timesteps == 0)
        {
            back_propagation->combinations_recurrent_weights_derivatives.setZero();
        }
        else
        {
            forward_propagation->previous_activations = forward_propagation->activations.chip(sample-1, 0);

            multiply_rows(back_propagation->combinations_recurrent_weights_derivatives, forward_propagation->current_activations_derivatives);

            back_propagation->combinations_recurrent_weights_derivatives
                    = back_propagation->combinations_recurrent_weights_derivatives.contract(recurrent_weights,A_B).eval();

            column_index = 0;
            activation_index = 0;

            for(Index i = 0; i < parameters_number; i++)
            {
                back_propagation->combinations_recurrent_weights_derivatives(i, column_index)
                        += forward_propagation->previous_activations(activation_index);

                activation_index++;

                if(activation_index == neurons_number)
                {
                    activation_index = 0;
                    column_index++;
                }
            }
        }

        forward_propagation->current_activations_derivatives = forward_propagation->activations_derivatives.chip(sample, 0);

        back_propagation->recurrent_weights_derivatives += back_propagation->combinations_recurrent_weights_derivatives
                .contract(back_propagation->current_layer_deltas*forward_propagation->current_activations_derivatives, A_B);
    }

}


/// Returns a string with the expression of the inputs-outputs relationship of the layer.
/// @param inputs_names Vector of strings with the name of the layer inputs.
/// @param outputs_names Vector of strings with the name of the layer outputs.
/// @todo Implement method

string RecurrentLayer::write_expression(const Tensor<string, 1>& inputs_names,
                                        const Tensor<string, 1>& outputs_names) const
{
#ifdef OPENNN_DEBUG

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

    for(Index j = 0; j < outputs_names.size(); j++)
    {
        const Tensor<type, 1> synaptic_weights_column =  recurrent_weights.chip(j,1);


        buffer << outputs_names(j) << " = " << write_activation_function_expression() << "( " << biases(j) << " +";

        for(Index i = 0; i < inputs_names.size() - 1; i++)
        {
           buffer << " (" << inputs_names[i] << "*" << synaptic_weights_column(i) << ") +";
        }

        buffer << " (" << inputs_names[inputs_names.size() - 1] << "*" << synaptic_weights_column[inputs_names.size() - 1] << ") );\n";
    }

    return buffer.str();
}


string RecurrentLayer::write_activation_function_expression() const
{
    switch(activation_function)
    {
        case HyperbolicTangent: return "tanh";

        case Linear: return string();

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


string RecurrentLayer::write_combinations_python() const
{
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    buffer << "\t\tcombinations = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\t\tcombinations[" << i << "] = " << biases(i);

        for(Index j = 0; j < neurons_number; j++)
        {
             buffer << " +" << recurrent_weights(j, i) << "*self.hidden_states[" << j << "]";
        }

        for(Index j = 0; j < inputs_number; j++)
        {
             buffer << " +" << input_weights(j, i) << "*inputs[" << j << "]";
        }

        buffer << " " << endl;
    }

    buffer << "\t\t" << endl;

    return buffer.str();
}


string RecurrentLayer::write_activations_python() const
{
    ostringstream buffer;

    const Index neurons_number = get_neurons_number();

    buffer << "\t\tactivations = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\t\tactivations[" << i << "] = ";

        switch(activation_function)
        {

        case HyperbolicTangent:
            buffer << "np.tanh(combinations[" << i << "])\n";
            break;

        case RectifiedLinear:
            buffer << "np.maximum(0.0, combinations[" << i << "])\n";
            break;

        case Logistic:
            buffer << "1.0/(1.0 + np.exp(-combinations[" << i << "]))\n";
            break;

        case Threshold:
            buffer << "1.0 if combinations[" << i << "] >= 0.0 else 0.0\n";
            break;

        case SymmetricThreshold:
            buffer << "1.0 if combinations[" << i << "] >= 0.0 else -1.0\n";
            break;

        case Linear:
            buffer << "combinations[" << i << "]\n";
            break;

        case ScaledExponentialLinear:
            buffer << "1.0507*1.67326*(np.exp(combinations[" << i << "]) - 1.0) if combinations[" << i << "] < 0.0 else 1.0507*combinations[" << i << "]\n";
            break;

        case SoftPlus:
            buffer << "np.log(1.0 + np.exp(combinations[" << i << "]))\n";
            break;

        case SoftSign:
            buffer << "combinations[" << i << "]/(1.0 - combinations[" << i << "] ) if combinations[" << i << "] < 0.0 else combinations[" << i << "]/(1.0 + combinations[" << i << "] )\n";
            break;

        case ExponentialLinear:
            buffer << "1.0*(np.exp(combinations[" << i << "]) - 1.0) if combinations[" << i << "] < 0.0 else combinations[" << i << "]\n";
            break;

        case HardSigmoid:
            ///@todo
            break;

        }

    }

    buffer << "\t\tself.hidden_states = activations" << endl;

    return buffer.str();
}


string RecurrentLayer::write_expression_python() const
{
    ostringstream buffer;

    buffer << "\tdef " << layer_name << "(self,inputs):\n" << endl;

    buffer << write_combinations_python();

    buffer << write_activations_python();

    buffer << "\n\t\treturn activations;\n" << endl;

    return buffer.str();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
