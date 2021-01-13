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

ProbabilisticLayer::ProbabilisticLayer(const Index& new_inputs_number, const Index& new_neurons_number)
{
    set(new_inputs_number, new_neurons_number);

    if(new_neurons_number > 1)
    {
        activation_function = Softmax;
    }
}


/// Destructor.
/// This destructor does not delete any pointer.

ProbabilisticLayer::~ProbabilisticLayer()
{
}


Index ProbabilisticLayer::get_inputs_number() const
{
    return synaptic_weights.dimension(0);
}


Index ProbabilisticLayer::get_neurons_number() const
{
    return biases.size();
}


Index ProbabilisticLayer::get_biases_number() const
{
    return biases.size();
}


/// Returns the number of layer's synaptic weights

Index ProbabilisticLayer::get_synaptic_weights_number() const
{
    return synaptic_weights.size();
}


/// Returns the decision threshold.

const type& ProbabilisticLayer::get_decision_threshold() const
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

const Tensor<type, 2>& ProbabilisticLayer::get_biases() const
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

Tensor<type, 2> ProbabilisticLayer::get_biases(Tensor<type, 1>& parameters) const
{
    const Index neurons_number = get_neurons_number();
/*
    Tensor<type, 2> bias_tensor(1, biases_number);

    Index index = parameters.size()-1;

    for(Index i = 0; i < biases_number; i++)
    {
        bias_tensor(0, i) = parameters(index);

        index--;
    }
    */
    const TensorMap < Tensor<type, 2> > bias_tensor(parameters.data(),  1, neurons_number);

    return bias_tensor;
}


/// Returns the synaptic weights from a given vector of paramters for the layer.
/// @param parameters Parameters of the layer.

Tensor<type, 2> ProbabilisticLayer::get_synaptic_weights(Tensor<type, 1>& parameters) const
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();
    const Index biases_number = get_biases_number();
/*
    const Index synaptic_weights_number = synaptic_weights.size();

    Tensor<type, 2> synaptic_weights_tensor(inputs_number, neurons_number);

    for(Index i = 0; i < synaptic_weights_number; i++)
    {
        synaptic_weights_tensor(i) = parameters(i);
    }
*/
    const TensorMap< Tensor<type, 2> > synaptic_weights_tensor(parameters.data()+biases_number, inputs_number, neurons_number);

    return  synaptic_weights_tensor;
}


/// Returns the number of parameters(biases and synaptic weights) of the layer.

Index ProbabilisticLayer::get_parameters_number() const
{
    return biases.size() + synaptic_weights.size();
}


/// Returns a single vector with all the layer parameters.
/// The format is a vector of real values.
/// The size is the number of parameters in the layer.

Tensor<type, 1> ProbabilisticLayer::get_parameters() const
{

//    Eigen::array<Index, 1> one_dim_weight{{synaptic_weights.dimension(0)*synaptic_weights.dimension(1)}};

//    Eigen::array<Index, 1> one_dim_bias{{biases.dimension(0)*biases.dimension(1)}};

//    Tensor<type, 1> synaptic_weights_vector = synaptic_weights.reshape(one_dim_weight);

//    Tensor<type, 1> biases_vector = biases.reshape(one_dim_bias);

    Tensor<type, 1> parameters(synaptic_weights.size() + biases.size());
/*
    for(Index i = 0; i < biases_vector.size(); i++)
    {
        fill_n(parameters.data()+i, 1, biases_vector(i));
    }

    for(Index i = 0; i < synaptic_weights_vector.size(); i++)
    {
        fill_n(parameters.data()+ biases_vector.size() +i, 1, synaptic_weights_vector(i));
    }
*/
    for(Index i = 0; i < biases.size(); i++)
    {
        fill_n(parameters.data()+i, 1, biases(i));
    }

    for(Index i = 0; i < synaptic_weights.size(); i++)
    {
        fill_n(parameters.data()+ biases.size() +i, 1, synaptic_weights(i));
    }

    return parameters;

}


/// Sets a probabilistic layer with zero probabilistic neurons.
/// It also sets the rest of members to their default values.

void ProbabilisticLayer::set()
{
    biases.resize(0, 0);

    synaptic_weights.resize(0,0);

    set_default();
}


/// Resizes the size of the probabilistic layer.
/// It also sets the rest of class members to their default values.
/// @param new_neurons_number New size for the probabilistic layer.

void ProbabilisticLayer::set(const Index& new_inputs_number, const Index& new_neurons_number)
{
    biases.resize(1, new_neurons_number);

    synaptic_weights.resize(new_inputs_number, new_neurons_number);

    set_parameters_random();

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


void ProbabilisticLayer::set_inputs_number(const Index& new_inputs_number)
{
    const Index neurons_number = get_neurons_number();

    biases.resize(1, neurons_number);

    synaptic_weights.resize(new_inputs_number, neurons_number);
}


void ProbabilisticLayer::set_neurons_number(const Index& new_neurons_number)
{
    const Index inputs_number = get_inputs_number();

    biases.resize(1, new_neurons_number);

    synaptic_weights.resize(inputs_number, new_neurons_number);
}


void ProbabilisticLayer::set_biases(const Tensor<type, 2>& new_biases)
{
    biases = new_biases;
}


void ProbabilisticLayer::set_synaptic_weights(const Tensor<type, 2>& new_synaptic_weights)
{
    synaptic_weights = new_synaptic_weights;
}


void ProbabilisticLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{/*
    const Index neurons_number = get_neurons_number();
    const Index inputs_number = get_inputs_number();

    const Index parameters_number = get_parameters_number();

#ifdef __OPENNN_DEBUG__

    const Index new_parameters_size = new_parameters.size();

    if(new_parameters_size != parameters_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_parameters(const Tensor<type, 1>&) method.\n"
               << "Size of new parameters ("
               << new_parameters_size << ") must be equal to number of parameters ("
               << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif
*/
    const Index biases_number = biases.size();
    const Index synaptic_weights_number = synaptic_weights.size();

    memcpy(biases.data(), new_parameters.data() + index, static_cast<size_t>(biases_number)*sizeof(type));
    memcpy(synaptic_weights.data(), new_parameters.data() + biases_number + index, static_cast<size_t>(synaptic_weights_number)*sizeof(type));
}


/// Sets a new threshold value for discriminating between two classes.
/// @param new_decision_threshold New discriminating value. It must be comprised between 0 and 1.

void ProbabilisticLayer::set_decision_threshold(const type& new_decision_threshold)
{
#ifdef __OPENNN_DEBUG__

    if(new_decision_threshold <= 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_decision_threshold(const type&) method.\n"
               << "Decision threshold(" << decision_threshold << ") must be greater than zero.\n";

        throw logic_error(buffer.str());
    }
    else if(new_decision_threshold >= 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_decision_threshold(const type&) method.\n"
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
    layer_name = "probabilistic_layer";

    layer_type = Probabilistic;

    const Index neurons_number = get_neurons_number();

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

    const Index neurons_number = get_neurons_number();

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


/// Initializes the biases of all the neurons in the probabilistic layer with a given value.
/// @param value Biases initialization value.

void ProbabilisticLayer::set_biases_constant(const type& value)
{
    biases.setConstant(value);
}


/// Initializes the synaptic weights of all the neurons in the probabilistic layer with a given value.
/// @param value Synaptic weights initialization value.

void ProbabilisticLayer::set_synaptic_weights_constant(const type& value)
{
    synaptic_weights.setConstant(value);
}


void ProbabilisticLayer::set_synaptic_weights_constant_Glorot()
{
    synaptic_weights.setRandom();
}


/// Initializes all the biases and synaptic weights in the neural newtork with a given value.
/// @param value Parameters initialization value.

void ProbabilisticLayer::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    synaptic_weights.setConstant(value);
}


/// Initializes all the biases and synaptic weights in the neural newtork at random with values comprised
/// between -1 and +1.

void ProbabilisticLayer::set_parameters_random()
{
    const type minimum = -1;
    const type maximum = 1;

    for(Index i = 0; i < biases.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        biases(i) = minimum +(maximum-minimum)*random;
    }

    for(Index i = 0; i < synaptic_weights.size(); i++)
    {
        const type random = static_cast<type>(rand()/(RAND_MAX+1.0));

        synaptic_weights(i) = minimum +(maximum-minimum)*random;
    }
}

/// Initializes the synaptic weights with glorot uniform distribution.

void ProbabilisticLayer::set_synaptic_weights_glorot()
{
    Index fan_in;
    Index fan_out;

    type scale = 1.0;
    type limit;

    fan_in = synaptic_weights.dimension(0);
    fan_out = synaptic_weights.dimension(1);

    scale /= ((fan_in + fan_out) / static_cast<type>(2.0));
    limit = sqrt(static_cast<type>(3.0) * scale);

//    biases.setRandom<Eigen::internal::UniformRandomGenerator<type>>();
    biases.setZero();

    synaptic_weights.setRandom<Eigen::internal::UniformRandomGenerator<type>>();

    Eigen::Tensor<type, 0> min_weight = synaptic_weights.minimum();
    Eigen::Tensor<type, 0> max_weight = synaptic_weights.maximum();

    synaptic_weights = (synaptic_weights - synaptic_weights.constant(min_weight(0))) / (synaptic_weights.constant(max_weight(0))- synaptic_weights.constant(min_weight(0)));
    synaptic_weights = (synaptic_weights * synaptic_weights.constant(2. * limit)) - synaptic_weights.constant(limit);
}

void ProbabilisticLayer::insert_parameters(const Tensor<type, 1>& parameters, const Index& )
{
    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    memcpy(biases.data() , parameters.data(), static_cast<size_t>(biases_number)*sizeof(type));
    memcpy(synaptic_weights.data(), parameters.data() + biases_number, static_cast<size_t>(synaptic_weights_number)*sizeof(type));
}


void ProbabilisticLayer::calculate_combinations(const Tensor<type, 2>& inputs,
                            const Tensor<type, 2>& biases,
                            const Tensor<type, 2>& synaptic_weights,
                            Tensor<type, 2>& combinations_2d) const
{
    const Index batch_samples_number = inputs.dimension(0);
    const Index biases_number = get_neurons_number();

    for(Index i = 0; i < biases_number; i++)
    {
        fill_n(combinations_2d.data()+i*batch_samples_number, batch_samples_number, biases(i));
    }

    combinations_2d.device(*thread_pool_device) += inputs.contract(synaptic_weights, A_B);

}


// Activations

void ProbabilisticLayer::calculate_activations(const Tensor<type, 2>& combinations_2d, Tensor<type, 2>& activations_2d) const
{
     #ifdef __OPENNN_DEBUG__

     const Index dimensions_number = combinations_2d.rank();

     if(dimensions_number != 2)
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
               << "Dimensions of combinations_2d (" << dimensions_number << ") must be 2.\n";

        throw logic_error(buffer.str());
     }

     const Index neurons_number = get_neurons_number();

     const Index combinations_columns_number = combinations_2d.dimension(1);

     if(combinations_columns_number != neurons_number)
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
               << "Number of combinations_2d columns (" << combinations_columns_number << ") must be equal to number of neurons (" << neurons_number << ").\n";

        throw logic_error(buffer.str());
     }

     #endif

     switch(activation_function)
     {
         case Binary: binary(combinations_2d, activations_2d); return;

         case Logistic: logistic(combinations_2d, activations_2d); return;

         case Competitive: competitive(combinations_2d, activations_2d); return;

         case Softmax: softmax(combinations_2d, activations_2d); return;
     }

     ostringstream buffer;

     buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
            << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
            << "Unknown probabilistic method.\n";

     throw logic_error(buffer.str());
}

void ProbabilisticLayer::calculate_activations_derivatives(const Tensor<type, 2>& combinations_2d,
                                       Tensor<type, 2>& activations,
                                       Tensor<type, 3>& activations_derivatives) const
{
     #ifdef __OPENNN_DEBUG__

     const Index neurons_number = get_neurons_number();

     const Index combinations_columns_number = combinations_2d.dimension(1);

     if(combinations_columns_number != neurons_number)
     {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void calculate_activations_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
               << "Number of combinations_2d columns (" << combinations_columns_number
               << ") must be equal to number of neurons (" << neurons_number << ").\n";

        throw logic_error(buffer.str());
     }

     #endif

     switch(activation_function)
     {
         case Logistic: logistic_derivatives(combinations_2d, activations, activations_derivatives); return;

         case Softmax: softmax_derivatives(combinations_2d, activations, activations_derivatives); return;

         default: return;
    }
}

/// This method processes the input to the probabilistic layer in order to obtain a set of outputs which
/// can be interpreted as probabilities.
/// This posprocessing is performed according to the probabilistic method to be used.
/// @param inputs Set of inputs to the probabilistic layer.

Tensor<type, 2> ProbabilisticLayer::calculate_outputs(const Tensor<type, 2>& inputs)
{
    const Index batch_size = inputs.dimension(0);
    const Index outputs_number = get_neurons_number();

    Tensor<type, 2> outputs(batch_size, outputs_number);

    calculate_combinations(inputs, biases, synaptic_weights, outputs);

    calculate_activations(outputs, outputs);

    return outputs;
}


void ProbabilisticLayer::forward_propagate(const Tensor<type, 2>& inputs, ForwardPropagation& forward_propagation) const
{
    calculate_combinations(inputs, biases, synaptic_weights, forward_propagation.combinations_2d);
    calculate_activations_derivatives(forward_propagation.combinations_2d,
                                      forward_propagation.activations_2d,
                                      forward_propagation.activations_derivatives_3d);
}


void ProbabilisticLayer::forward_propagate(const Tensor<type, 2>& inputs,
                                   Tensor<type, 1> potential_parameters,
                                   ForwardPropagation& forward_propagation) const
{
    const Index neurons_number = get_neurons_number();
    const Index inputs_number = get_inputs_number();

#ifdef __OPENNN_DEBUG__

    if(inputs_number != inputs.dimension(1))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void forward_propagate(const Tensor<type, 2>&, Tensor<type, 1>&, ForwardPropagation&) method.\n"
               << "Number of inputs columns (" << inputs.dimension(1) << ") must be equal to number of inputs ("
               << inputs_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    const TensorMap<Tensor<type, 2>> potential_biases(potential_parameters.data(), neurons_number, 1);

    const TensorMap<Tensor<type, 2>> potential_synaptic_weights(potential_parameters.data()+neurons_number,
                                                                inputs_number, neurons_number);

    calculate_combinations(inputs, potential_biases, potential_synaptic_weights, forward_propagation.combinations_2d);

    calculate_activations_derivatives(forward_propagation.combinations_2d,
                                      forward_propagation.activations_2d,
                                      forward_propagation.activations_derivatives_3d);
}




void ProbabilisticLayer::calculate_output_delta(ForwardPropagation& forward_propagation,
                            const Tensor<type, 2>& output_gradient,
                            Tensor<type, 2>& output_delta) const
{
    const Index neurons_number = get_neurons_number();
    const Index batch_samples_number = forward_propagation.activations_derivatives_3d.dimension(0);

    if(neurons_number == 1)
    {
        TensorMap< Tensor<type, 2> > activations_derivatives(forward_propagation.activations_derivatives_3d.data(), batch_samples_number, neurons_number);

        output_delta.device(*thread_pool_device) = activations_derivatives*output_gradient;
    }
    else
    {
        const Index outputs_number = output_gradient.dimension(1); // outputs_number = neurons_number and activations.dimension(1)

        if(outputs_number != neurons_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                   << "void calculate_output_delta(ForwardPropagation& ,const Tensor<type, 2>& ,Tensor<type, 2>& ) const.\n"
                   << "Number of columns in output gradient (" << outputs_number << ") must be equal to number of neurons in probabilistic layer (" << neurons_number << ").\n";

            throw logic_error(buffer.str());
        }

        if(forward_propagation.activations_derivatives_3d.dimension(1) != neurons_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                   << "void calculate_output_delta(ForwardPropagation& ,const Tensor<type, 2>& ,Tensor<type, 2>& ) const.\n"
                   << "Dimension 1 of activations derivatives 3d (" << outputs_number << ") must be equal to number of neurons in probabilistic layer (" << neurons_number << ").\n";

            throw logic_error(buffer.str());
        }

        if(forward_propagation.activations_derivatives_3d.dimension(2) != neurons_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
                   << "void calculate_output_delta(ForwardPropagation& ,const Tensor<type, 2>& ,Tensor<type, 2>& ) const.\n"
                   << "Dimension 2 of activations derivatives 3d (" << outputs_number << ") must be equal to number of neurons in probabilistic layer (" << neurons_number << ").\n";

            throw logic_error(buffer.str());
        }

        Tensor<type, 1> output_gradient_row(neurons_number);
        Tensor<type, 1> output_delta_row(neurons_number);

        Index index = 0;
        Index step = neurons_number*neurons_number;

        for(Index i = 0; i < batch_samples_number; i++)
        {
            output_gradient_row = output_gradient.chip(i,0);

            TensorMap< Tensor<type, 2> > activations_derivatives_matrix(forward_propagation.activations_derivatives_3d.data() + index,
                                                                        neurons_number, neurons_number);

            output_delta_row.device(*thread_pool_device) = output_gradient_row.contract(activations_derivatives_matrix, AT_B);

            for(Index j = 0; j < neurons_number; j++)
            {
                output_delta(i,j) = output_delta_row(j);
            }

            index += step;
        }
    }
}


// Gradient methods

void ProbabilisticLayer::calculate_error_gradient(const Tensor<type, 2>& inputs,
                                                  const Layer::ForwardPropagation&,
                                                  Layer::BackPropagation& back_propagation) const
{
    back_propagation.biases_derivatives.device(*thread_pool_device) = back_propagation.delta.sum(Eigen::array<Index, 1>({0}));

    back_propagation.synaptic_weights_derivatives.device(*thread_pool_device) = inputs.contract(back_propagation.delta, AT_B);
}


void ProbabilisticLayer::insert_gradient(const BackPropagation& back_propagation, const Index& index, Tensor<type, 1>& gradient) const
{
    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    memcpy(gradient.data() + index,
           back_propagation.biases_derivatives.data(),
           static_cast<size_t>(biases_number)*sizeof(type));

    memcpy(gradient.data() + index + biases_number,
           back_propagation.synaptic_weights_derivatives.data(),
           static_cast<size_t>(synaptic_weights_number)*sizeof(type));
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

    const Tensor<type, 1> parameters = get_parameters();
    const Index parameters_size = parameters.size();

    for(Index i = 0; i < parameters_size; i++)
    {
        buffer << parameters(i);

        if(i != (parameters_size-1)) buffer << " ";
    }

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

    const tinyxml2::XMLElement* inputs_number_element = probabilistic_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Inputs number element is nullptr.\n" /* << inputs_number_element->GetText()*/;

        throw logic_error(buffer.str());
    }

    Index new_inputs_number;

    if(inputs_number_element->GetText())
    {
        new_inputs_number = static_cast<Index>(stoi(inputs_number_element->GetText()));
    }

    // Neurons number

    const tinyxml2::XMLElement* neurons_number_element = probabilistic_layer_element->FirstChildElement("NeuronsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Neurons number element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    Index new_neurons_number;

    if(neurons_number_element->GetText())
    {
        new_neurons_number = static_cast<Index>(stoi(neurons_number_element->GetText()));
    }

    set(new_inputs_number, new_neurons_number);

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

        set_parameters(to_type_vector(parameters_string, ' '));
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
        set_decision_threshold(static_cast<type>(atof(decision_threshold_element->GetText())));
    }

    // Display

    const tinyxml2::XMLElement* display_element = probabilistic_layer_element->FirstChildElement("Display");

    if(display_element)
    {
        const string new_display_string = display_element->GetText();

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

    for(Index j = 0; j < outputs_names.size(); j++)
    {
        buffer << outputs_names(j) << " = binary(" << inputs_names(j) << ");\n";
    }
    return buffer.str();
}


/// Returns a string with the expression of the probability outputs function.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

string ProbabilisticLayer::write_logistic_expression(const Tensor<string, 1>& inputs_names,
        const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;

    for(Index j = 0; j < outputs_names.size(); j++)
    {
        buffer << outputs_names(j) << " = logistic(" << inputs_names(j) << ");\n";
    }
    return buffer.str();
}


/// Returns a string with the expression of the competitive probabilistic outputs function.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

string ProbabilisticLayer::write_competitive_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;

    for(Index j = 0; j < outputs_names.size(); j++)
    {
        buffer << outputs_names(j) << " = competitive(" << inputs_names(j) << ");\n";
    }
    return buffer.str();
}


/// Returns a string with the expression of the softmax probabilistic outputs function.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

string ProbabilisticLayer::write_softmax_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;

    for(Index j = 0; j < outputs_names.size(); j++)
    {
        buffer << outputs_names(j) << " = softmax(" << inputs_names(j) << ");\n";
    }

    return buffer.str();
}


/// Returns a string with the expression of the no probabilistic outputs function.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

string ProbabilisticLayer::write_no_probabilistic_expression(const Tensor<string, 1>& inputs_names,
        const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;

    for(Index j = 0; j < outputs_names.size(); j++)
    {
        buffer << outputs_names(j) << " = (" << inputs_names(j) << ");\n";
    }
    return buffer.str();
}


string ProbabilisticLayer::write_combinations_c() const
{
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    buffer << "\tvector<float> combinations(" << neurons_number << ");\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\tcombinations[" << i << "] = " << biases(i);

        for(Index j = 0; j < inputs_number; j++)
        {
             buffer << " +" << synaptic_weights(j, i) << "*inputs[" << j << "]";
        }

        buffer << ";" << endl;
    }

    return buffer.str();
}


string ProbabilisticLayer::write_activations_c() const
{
    ostringstream buffer;

    const Index neurons_number = get_neurons_number();

    buffer << "\n\tvector<float> activations(" << neurons_number << ");\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        switch(activation_function)
        {
        case Binary:
            buffer << "\tactivations[" << i << "] = combinations[" << i << "] < 0.5 ? 0.0 : 1.0;\n";
            break;

        case Logistic:
            buffer << "\tactivations[" << i << "] = 1.0/(1.0 + exp(-combinations[" << i << "]));\n";
            break;

        case Competitive:
            ///@todo
            break;

        case Softmax:

            if(i == 0)
            {
                buffer << "\tfloat sum = 0;\n" << endl;

                buffer << "\tsum = ";

                for(Index i = 0; i < neurons_number; i++)
                {
                    buffer << "exp(combinations[" << i << "])";

                    if(i != neurons_number-1) buffer << " + ";
                }

                buffer << ";\n" << endl;

                for(Index i = 0; i < neurons_number; i++)
                {
                    buffer << "\tactivations[" << i << "] = exp(combinations[" << i << "])/sum;\n";
                }

            }
            break;
        }
    }

    return buffer.str();
}


string ProbabilisticLayer::write_combinations_python() const
{
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    buffer << "\tcombinations = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\tcombinations[" << i << "] = " << biases(i);

        for(Index j = 0; j < inputs_number; j++)
        {
             buffer << " +" << synaptic_weights(j, i) << "*inputs[" << j << "]";
        }

        buffer << " " << endl;
    }

    buffer << "\t" << endl;

    return buffer.str();
}


string ProbabilisticLayer::write_activations_python() const
{
    ostringstream buffer;

    const Index neurons_number = get_neurons_number();

    buffer << "\tactivations = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        switch(activation_function)
        {
        case Binary:
            buffer << "\tactivations[" << i << "] = 0.0 if combinations[" << i << "] < 0.5 else 1.0\n";
            break;

        case Logistic:
            buffer << "\tactivations[" << i << "] = 1.0/(1.0 + np.exp(-combinations[" << i << "]));\n";
            break;

        case Competitive:

            if(i == 0)
            {
                buffer << "\tfor i, value in enumerate(combinations):"<<endl;

                buffer <<"\t\tif(max(combinations) == value):"<<endl;

                buffer <<"\t\t\tactivations[i] = 1"<<endl;

                buffer <<"\t\telse:"<<endl;

                buffer <<"\t\t\tactivations[i] = 0"<<endl;
            }

            break;

        case Softmax:

            if(i == 0)
            {
                buffer << "\tsum_ = 0;\n" << endl;

                buffer << "\tsum_ = ";

                for(Index i = 0; i < neurons_number; i++)
                {
                    buffer << "np.exp(combinations[" << i << "])";

                    if(i != neurons_number-1) buffer << " + ";
                }

                buffer << ";\n" << endl;

                for(Index i = 0; i < neurons_number; i++)
                {
                    buffer << "\tactivations[" << i << "] = np.exp(combinations[" << i << "])/sum_;\n";
                }

            }
            break;



        }
    }

    return buffer.str();
}


string ProbabilisticLayer::write_combinations(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "\t" << "probabilistic_layer_combinations_" << to_string(i) << " = " << biases(i);

        for(Index j = 0; j < inputs_number; j++)
        {
             buffer << " +" << synaptic_weights(j, i) << "*" << inputs_names(j) << "";
        }

        buffer << " " << endl;
    }

    buffer << "\t" << endl;

    return buffer.str();
}


string ProbabilisticLayer::write_activations(const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;

    const Index neurons_number = get_neurons_number();

    for(Index i = 0; i < neurons_number; i++)
    {
        switch(activation_function)
        {
        case Binary:
            {
                buffer << "\tif" << "probabilistic_layer_combinations_" << to_string(i) << " < 0.5, " << outputs_names(i) << "= 0.0. Else " << outputs_names(i) << " = 1.0\n";
            }
            break;

        case Logistic:
            {
                buffer <<  outputs_names(i) << " = 1.0/(1.0 + exp(-" <<  "probabilistic_layer_combinations_" << to_string(i) << ");\n";
            }
            break;

        case Competitive:
            if(i == 0)
            {
                buffer << "\tfor each probabilistic_layer_combinations_i:"<<endl;

                buffer <<"\t\tif probabilistic_layer_combinations_i is equal to max(probabilistic_layer_combinations_i):"<<endl;

                buffer <<"\t\t\tactivations[i] = 1"<<endl;

                buffer <<"\t\telse:"<<endl;

                buffer <<"\t\t\tactivations[i] = 0"<<endl;
            }

            break;

        case Softmax:

            if(i == 0)
            {
                buffer << "\tsum_ = ";

                for(Index i = 0; i < neurons_number; i++)
                {
                    buffer << "exp(probabilistic_layer_combinations_"  << to_string(i);

                    if(i != neurons_number-1) buffer << " + ";
                }

                buffer << ";\n" << endl;

                for(Index i = 0; i < neurons_number; i++)
                {
                    buffer << "\t" << outputs_names(i) << " = exp(probabilistic_layer_combinations_" << to_string(i) <<")/sum_;\n";
                }

            }
            break;



        }
    }

    return buffer.str();
}

string ProbabilisticLayer::write_expression_c() const
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    ostringstream buffer;

    buffer << "vector<float> " << layer_name << "(const vector<float>& inputs)\n{" << endl;

    buffer << write_combinations_c();

    buffer << write_activations_c();

    buffer << "\n\treturn activations;\n}" << endl;

    return buffer.str();
}


string ProbabilisticLayer::write_expression_python() const
{
    ostringstream buffer;

    buffer << "def " << layer_name << "(inputs):\n" << endl;

    buffer << write_combinations_python();

    buffer << write_activations_python();

    buffer << "\n\treturn activations;\n" << endl;

    return buffer.str();
}

string ProbabilisticLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;

    buffer << write_combinations(inputs_names, outputs_names);

    buffer << write_activations(outputs_names);

    return buffer.str();
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
