//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "perceptron_layer_3d.h"
#include "tensors.h"

namespace opennn
{

/// Default constructor.
/// It creates a empty layer object, with no perceptrons.
/// This constructor also initializes the rest of the class members to their default values.


PerceptronLayer3D::PerceptronLayer3D() : Layer()
{
    set();

    layer_type = Type::Perceptron3D;
}


/// Layer architecture constructor.
/// It creates a layer object with given numbers of inputs and perceptrons.
/// It initializes the parameters at random.
/// This constructor also initializes the rest of the class members to their default values.
/// @param new_inputs_number Number of inputs in the layer.
/// @param new_neurons_number Number of perceptrons in the layer.

PerceptronLayer3D::PerceptronLayer3D(const Index& new_inputs_number, const Index& new_inputs_depth, const Index& new_neurons_number,
                                 const PerceptronLayer3D::ActivationFunction& new_activation_function) : Layer()
{
    set(new_inputs_number, new_inputs_depth, new_neurons_number, new_activation_function);

    layer_type = Type::Perceptron3D;

    layer_name = "perceptron_layer_3d";
}


/// Returns the number of inputs to the layer.

Index PerceptronLayer3D::get_inputs_number() const
{
    return inputs_number;
}


Index PerceptronLayer3D::get_inputs_depth() const
{
    return synaptic_weights.dimension(0);
}


void PerceptronLayer3D::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


/// Returns the number of neurons in the layer.

Index PerceptronLayer3D::get_neurons_number() const
{
    return biases.size();
}


dimensions PerceptronLayer3D::get_outputs_dimensions() const
{
    Index neurons_number = get_neurons_number();

    return { inputs_number, neurons_number };
}


Index PerceptronLayer3D::get_biases_number() const
{
    return biases.size();
}


/// Returns the number of layer's synaptic weights

Index PerceptronLayer3D::get_synaptic_weights_number() const
{
    return synaptic_weights.size();
}


/// Returns the number of parameters (biases and synaptic weights) of the layer.

Index PerceptronLayer3D::get_parameters_number() const
{
    return biases.size() + synaptic_weights.size();
}


type PerceptronLayer3D::get_dropout_rate() const
{
    return dropout_rate;
}

/// Returns the biases from all the perceptrons in the layer.
/// The format is a vector of real values.
/// The size of this vector is the number of neurons in the layer.

const Tensor<type, 1>& PerceptronLayer3D::get_biases() const
{
    return biases;
}



/// Returns the synaptic weights from the perceptrons.
/// The format is a matrix of real values.
/// The number of rows is the number of neurons in the layer.
/// The number of raw_variables is the number of inputs to the layer.

const Tensor<type, 2>& PerceptronLayer3D::get_synaptic_weights() const
{
    return synaptic_weights;
}


Tensor<type, 2> PerceptronLayer3D::get_synaptic_weights(const Tensor<type, 1>& parameters) const
{
    const Index inputs_depth = get_inputs_depth();

    const Index neurons_number = get_neurons_number();

    const Index synaptic_weights_number = get_synaptic_weights_number();

    const Index parameters_size = parameters.size();

    const Index start_synaptic_weights_number = (parameters_size - synaptic_weights_number);

    const Tensor<type, 1> new_synaptic_weights = parameters.slice(Eigen::array<Eigen::Index, 1>({start_synaptic_weights_number}), Eigen::array<Eigen::Index, 1>({synaptic_weights_number}));

    const Eigen::array<Index, 2> two_dim{{inputs_depth, neurons_number}};

    return new_synaptic_weights.reshape(two_dim);
}


Tensor<type, 2> PerceptronLayer3D::get_biases(const Tensor<type, 1>& parameters) const
{
    const Index biases_number = biases.size();

    const Tensor<type, 1> new_biases = parameters.slice(Eigen::array<Eigen::Index, 1>({0}), Eigen::array<Eigen::Index, 1>({biases_number}));

    const Eigen::array<Index, 2> two_dim{{1, biases.dimension(1)}};

    return new_biases.reshape(two_dim);

}


/// Returns a single vector with all the layer parameters.
/// The format is a vector of real values.
/// The size is the number of parameters in the layer.

Tensor<type, 1> PerceptronLayer3D::get_parameters() const
{
    Tensor<type, 1> parameters(synaptic_weights.size() + biases.size());

    copy(/*execution::par,*/ 
        synaptic_weights.data(),
        synaptic_weights.data() + synaptic_weights.size(),
        parameters.data());

    copy(/*execution::par,*/ 
        biases.data(),
        biases.data() + biases.size(),
        parameters.data() + synaptic_weights.size());

    return parameters;
}


/// Returns the activation function of the layer.
/// The activation function of a layer is the activation function of all perceptrons in it.

const PerceptronLayer3D::ActivationFunction& PerceptronLayer3D::get_activation_function() const
{
    return activation_function;
}


/// Returns a string with the name of the layer activation function.
/// This can be Logistic, HyperbolicTangent, Linear, RectifiedLinear, ScaledExponentialLinear.

string PerceptronLayer3D::write_activation_function() const
{
    switch(activation_function)
    {
    /*case ActivationFunction::Logistic:
        return "Logistic";*/

    case ActivationFunction::HyperbolicTangent:
        return "HyperbolicTangent";

    case ActivationFunction::Linear:
        return "Linear";

    case ActivationFunction::RectifiedLinear:
        return "RectifiedLinear";

    /*case ActivationFunction::ScaledExponentialLinear:
        return "ScaledExponentialLinear";

    case ActivationFunction::SoftPlus:
        return "SoftPlus";

    case ActivationFunction::SoftSign:
        return "SoftSign";

    case ActivationFunction::HardSigmoid:
        return "HardSigmoid";

    case ActivationFunction::ExponentialLinear:
        return "ExponentialLinear";*/
    }

    return string();
}


/// Returns true if messages from this class are displayed on the screen,
/// or false if messages from this class are not displayed on the screen.

const bool& PerceptronLayer3D::get_display() const
{
    return display;
}


/// Sets an empty layer, wihtout any perceptron.
/// It also sets the rest of the members to their default values.

void PerceptronLayer3D::set()
{
    biases.resize(0);

    synaptic_weights.resize(0, 0);

    set_default();
}


/// Sets new numbers of inputs and perceptrons in the layer.
/// It also sets the rest of the members to their default values.
/// @param new_inputs_number Number of inputs.
/// @param new_neurons_number Number of perceptron neurons.

void PerceptronLayer3D::set(const Index& new_inputs_number, 
                            const Index& new_inputs_depth, 
                            const Index& new_neurons_number,
                            const PerceptronLayer3D::ActivationFunction& new_activation_function)
{
    inputs_number = new_inputs_number;

    biases.resize(new_neurons_number);

    synaptic_weights.resize(new_inputs_depth, new_neurons_number);

    set_parameters_glorot();

    activation_function = new_activation_function;

    set_default();
}


/// Sets those members not related to the vector of perceptrons to their default value.
/// <ul>
/// <li> Display: True.
/// <li> layer_type: Perceptron_Layer.
/// <li> trainable: True.
/// </ul>

void PerceptronLayer3D::set_default()
{
    layer_name = "perceptron_layer_3d";

    display = true;

    layer_type = Type::Perceptron3D;

    dropout_rate = 0;
}


void PerceptronLayer3D::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}


/// Sets a new number of inputs in the layer.
/// It also initializes the new synaptic weights at random.
/// @param new_inputs_number Number of layer inputs.

void PerceptronLayer3D::set_inputs_number(const Index& new_inputs_number)
{
    inputs_number = new_inputs_number;
}


void PerceptronLayer3D::set_inputs_depth(const Index& new_inputs_depth)
{
    const Index neurons_number = get_neurons_number();

    biases.resize(neurons_number);

    synaptic_weights.resize(new_inputs_depth, neurons_number);
}


/// Sets a new number perceptrons in the layer.
/// All the parameters are also initialized at random.
/// @param new_neurons_number New number of neurons in the layer.

void PerceptronLayer3D::set_neurons_number(const Index& new_neurons_number)
{
    const Index inputs_depth = get_inputs_depth();

    biases.resize(new_neurons_number);

    synaptic_weights.resize(inputs_depth, new_neurons_number);
}


/// Sets the biases of all perceptrons in the layer from a single vector.
/// @param new_biases New set of biases in the layer.

void PerceptronLayer3D::set_biases(const Tensor<type, 1>& new_biases)
{
    biases = new_biases;
}


/// Sets the synaptic weights of this perceptron layer from a single matrix.
/// The format is a matrix of real numbers.
/// The number of rows is the number of neurons in the corresponding layer.
/// The number of raw_variables is the number of inputs to the corresponding layer.
/// @param new_synaptic_weights New set of synaptic weights in that layer.

void PerceptronLayer3D::set_synaptic_weights(const Tensor<type, 2>& new_synaptic_weights)
{
    synaptic_weights = new_synaptic_weights;
}


/// Sets the parameters of this layer.

void PerceptronLayer3D::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    copy(/*execution::par,*/ 
        new_parameters.data() + index,
        new_parameters.data() + index + synaptic_weights.size(),
        synaptic_weights.data());

    copy(/*execution::par,*/ 
        new_parameters.data() + index + synaptic_weights.size(),
        new_parameters.data() + index + synaptic_weights.size() + biases.size(),
        biases.data());
}


/// This class sets a new activation(or transfer) function in a single layer.
/// @param new_activation_function Activation function for the layer.

void PerceptronLayer3D::set_activation_function(const PerceptronLayer3D::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}

/// Sets a new activation(or transfer) function in a single layer.
/// The second argument is a string containing the name of the function("Logistic", "HyperbolicTangent", etc).
/// @param new_activation_function Activation function for that layer.

void PerceptronLayer3D::set_activation_function(const string& new_activation_function_name)
{
    if(new_activation_function_name == "Logistic")
    {
        //activation_function = ActivationFunction::Logistic;
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
        //activation_function = ActivationFunction::ScaledExponentialLinear;
    }
    else if(new_activation_function_name == "SoftPlus")
    {
        //activation_function = ActivationFunction::SoftPlus;
    }
    else if(new_activation_function_name == "SoftSign")
    {
        //activation_function = ActivationFunction::SoftSign;
    }
    else if(new_activation_function_name == "HardSigmoid")
    {
        //activation_function = ActivationFunction::HardSigmoid;
    }
    else if(new_activation_function_name == "ExponentialLinear")
    {
        //activation_function = ActivationFunction::ExponentialLinear;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PerceptronLayer3D class.\n"
               << "void set_activation_function(const string&) method.\n"
               << "Unknown activation function: " << new_activation_function_name << ".\n";

        throw runtime_error(buffer.str());
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void PerceptronLayer3D::set_display(const bool& new_display)
{
    display = new_display;
}


/// Initializes the biases of all the perceptrons in the layer of perceptrons with a given value.
/// @param value Biases initialization value.

void PerceptronLayer3D::set_biases_constant(const type& value)
{
    biases.setConstant(value);
}


/// Initializes the synaptic weights of all the perceptrons in the layer of perceptrons with a given value.
/// @param value Synaptic weights initialization value.

void PerceptronLayer3D::set_synaptic_weights_constant(const type& value)
{
    synaptic_weights.setConstant(value);
}


/// Initializes all the biases and synaptic weights in the neural newtork with a given value.
/// @param value Parameters initialization value.

void PerceptronLayer3D::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    synaptic_weights.setConstant(value);
}


/// Initializes all the biases and synaptic weights in the neural newtork at random with values comprised
/// between -1 and +1.

void PerceptronLayer3D::set_parameters_random()
{
    biases.setRandom();

    synaptic_weights.setRandom();
}


/// Initializes the biases to zeroes and the synaptic weights with the Glorot Uniform initializer

void PerceptronLayer3D::set_parameters_glorot()
{
    biases.setZero();

    const type limit = sqrt(6 / type(get_inputs_depth() + get_neurons_number()));

    const type minimum = -limit;
    const type maximum = limit;

#pragma omp parallel for

    for (Index i = 0; i < synaptic_weights.size(); i++)
    {
        const type random = static_cast<type>(rand() / (RAND_MAX + 1.0));

        synaptic_weights(i) = minimum + (maximum - minimum) * random;
    }
}


void PerceptronLayer3D::calculate_combinations(const Tensor<type, 3>& inputs,
                                             const Tensor<type, 1>& biases,
                                             const Tensor<type, 2>& synaptic_weights,
                                             Tensor<type, 3>& combinations) const
{
    const Eigen::array<IndexPair<Index>, 1> contraction_indices = {IndexPair<Index>(2, 0)};

    combinations.device(*thread_pool_device) = inputs.contract(synaptic_weights, contraction_indices);

    sum_matrices(thread_pool_device, biases, combinations);
}


void PerceptronLayer3D::dropout(Tensor<type, 3>& outputs) const 
{
    type* outputs_data = outputs.data();

    const Index batch_samples_number = outputs.dimension(0);
    const Index inputs_number = outputs.dimension(1);
    const Index outputs_number = outputs.dimension(2);

    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    type random;

    for(Index neuron_index = 0; neuron_index < outputs_number; neuron_index++)
    {
        TensorMap<Tensor<type, 2>> matrix(outputs_data + neuron_index*batch_samples_number*inputs_number,
                                          batch_samples_number, inputs_number);

        random = calculate_random_uniform(type(0), type(1));

        if (random < dropout_rate)
        {
            matrix.setZero();
        }
        else
        {
            matrix.device(*thread_pool_device) = matrix * scaling_factor;
        }                   
    }
}


void PerceptronLayer3D::calculate_activations(const Tensor<type, 3>& combinations,
                                              Tensor<type, 3>& activations) const
{
    switch(activation_function)
    {
    case ActivationFunction::Linear: linear(combinations, activations); return;

//    case ActivationFunction::Logistic: logistic(combinations, activations); return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(combinations, activations); return;

    case ActivationFunction::RectifiedLinear: rectified_linear(combinations, activations); return;

//    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(combinations, activations); return;

//    case ActivationFunction::SoftPlus: soft_plus(combinations, activations); return;

//    case ActivationFunction::SoftSign: soft_sign(combinations, activations); return;

//    case ActivationFunction::HardSigmoid: hard_sigmoid(combinations, activations); return;

//    case ActivationFunction::ExponentialLinear: exponential_linear(combinations, activations); return;

    default: return;
    }
}


void PerceptronLayer3D::calculate_activations_derivatives(const Tensor<type, 3>& combinations,
                                                        Tensor<type, 3>& activations,
                                                        Tensor<type, 3>& activations_derivatives) const
{
    switch(activation_function)
    {
    case ActivationFunction::Linear: linear_derivatives(combinations,
                                                        activations,
                                                        activations_derivatives);
        return;

//    case ActivationFunction::Logistic: logistic_derivatives(combinations,
//                                                            activations,
//                                                            activations_derivatives);
//        return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent_derivatives(combinations,
                                                                               activations,
                                                                               activations_derivatives);
        return;

    case ActivationFunction::RectifiedLinear: rectified_linear_derivatives(combinations,
                                                                           activations,
                                                                           activations_derivatives);
        return;

//    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear_derivatives(combinations,
//                                                                                            activations,
//                                                                                            activations_derivatives);
//        return;

//    case ActivationFunction::SoftPlus: soft_plus_derivatives(combinations,
//                                                             activations,
//                                                             activations_derivatives);
//        return;

//    case ActivationFunction::SoftSign: soft_sign_derivatives(combinations,
//                                                             activations,
//                                                             activations_derivatives);
//        return;

//    case ActivationFunction::HardSigmoid: hard_sigmoid_derivatives(combinations,
//                                                                   activations,
//                                                                   activations_derivatives);
//        return;

//    case ActivationFunction::ExponentialLinear: exponential_linear_derivatives(combinations,
//                                                                               activations,
//                                                                               activations_derivatives);
//        return;

    default:

        return;
    }
}


void PerceptronLayer3D::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                        LayerForwardPropagation* layer_forward_propagation,
                                        const bool& is_training)
{
    const TensorMap<Tensor<type, 3>> inputs(inputs_pair(0).first, inputs_pair(0).second[0], inputs_pair(0).second[1], inputs_pair(0).second[2]);

    PerceptronLayer3DForwardPropagation* perceptron_layer_3d_forward_propagation =
        static_cast<PerceptronLayer3DForwardPropagation*>(layer_forward_propagation);

    Tensor<type, 3>& outputs = perceptron_layer_3d_forward_propagation->outputs;

    calculate_combinations(inputs,
                           biases,
                           synaptic_weights,
                           outputs);

    if(is_training)
    {
        if (dropout_rate > type(0))
        {
            dropout(outputs);
        }

        Tensor<type, 3>& activations_derivatives = perceptron_layer_3d_forward_propagation->activations_derivatives;

        calculate_activations_derivatives(outputs,
                                          outputs,
                                          activations_derivatives);
    }
    else
    {
        calculate_activations(outputs,
                              outputs);
    }
}


void PerceptronLayer3D::back_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                       const Tensor<pair<type*, dimensions>, 1>& deltas_pair,
                                       LayerForwardPropagation* forward_propagation,
                                       LayerBackPropagation* back_propagation) const
{
    const TensorMap<Tensor<type, 3>> inputs(inputs_pair(0).first,
                                            inputs_pair(0).second[0],
                                            inputs_pair(0).second[1],
                                            inputs_pair(0).second[2]);

    if (deltas_pair.size() > 1)     add_deltas(deltas_pair);

    const TensorMap<Tensor<type, 3>> deltas(deltas_pair(0).first,
                                            deltas_pair(0).second[0],
                                            deltas_pair(0).second[1],
                                            deltas_pair(0).second[2]);

    // Forward propagation

    const PerceptronLayer3DForwardPropagation* perceptron_layer_3d_forward_propagation =
            static_cast<PerceptronLayer3DForwardPropagation*>(forward_propagation);

    const Tensor<type, 3>& activations_derivatives = perceptron_layer_3d_forward_propagation->activations_derivatives;

    // Back propagation

    PerceptronLayer3DBackPropagation* perceptron_layer_3d_back_propagation =
            static_cast<PerceptronLayer3DBackPropagation*>(back_propagation);

    Tensor<type, 3>& error_combinations_derivatives = perceptron_layer_3d_back_propagation->error_combinations_derivatives;

    Tensor<type, 3>& input_derivatives = perceptron_layer_3d_back_propagation->input_derivatives;

    Tensor<type, 1>& biases_derivatives = perceptron_layer_3d_back_propagation->biases_derivatives;
    Tensor<type, 2>& synaptic_weights_derivatives = perceptron_layer_3d_back_propagation->synaptic_weights_derivatives;
   
    const Eigen::array<IndexPair<Index>, 2> double_contraction_indices = { IndexPair<Index>(0, 0), IndexPair<Index>(1, 1) };

    const Eigen::array<IndexPair<Index>, 1> single_contraction_indices = { IndexPair<Index>(2, 1) };

    error_combinations_derivatives.device(*thread_pool_device) 
        = deltas * activations_derivatives;

    biases_derivatives.device(*thread_pool_device)
        = error_combinations_derivatives.sum(Eigen::array<Index, 2>({0, 1}));

    synaptic_weights_derivatives.device(*thread_pool_device)
        = inputs.contract(error_combinations_derivatives, double_contraction_indices);

    input_derivatives.device(*thread_pool_device) 
        = error_combinations_derivatives.contract(synaptic_weights, single_contraction_indices);
}


void PerceptronLayer3D::add_deltas(const Tensor<pair<type*, dimensions>, 1>& deltas_pair) const
{
    TensorMap<Tensor<type, 3>> deltas(deltas_pair(0).first,
                                      deltas_pair(0).second[0],
                                      deltas_pair(0).second[1],
                                      deltas_pair(0).second[2]);

    for (Index i = 1; i < deltas_pair.size(); i++)
    {
        const TensorMap<Tensor<type, 3>> other_deltas(deltas_pair(i).first,
                                                      deltas_pair(i).second[0],
                                                      deltas_pair(i).second[1],
                                                      deltas_pair(i).second[2]);

        deltas.device(*thread_pool_device) += other_deltas;
    }
}



void PerceptronLayer3D::insert_gradient(LayerBackPropagation* back_propagation,
                                      const Index& index,
                                      Tensor<type, 1>& gradient) const
{
    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    PerceptronLayer3DBackPropagation* perceptron_layer_back_propagation =
            static_cast<PerceptronLayer3DBackPropagation*>(back_propagation);

    const type* synaptic_weights_derivatives_data = perceptron_layer_back_propagation->synaptic_weights_derivatives.data();
    const type* biases_derivatives_data = perceptron_layer_back_propagation->biases_derivatives.data();
    type* gradient_data = gradient.data();

    copy(/*execution::par,*/ 
        synaptic_weights_derivatives_data,
         synaptic_weights_derivatives_data + synaptic_weights_number,
         gradient_data + index);

    copy(/*execution::par,*/ 
        biases_derivatives_data,
         biases_derivatives_data + biases_number,
         gradient_data + index + synaptic_weights_number);
}


void PerceptronLayer3D::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Perceptron layer

    const tinyxml2::XMLElement* perceptron_layer_element = document.FirstChildElement("PerceptronLayer3D");

    if(!perceptron_layer_element)
    {
        buffer << "OpenNN Exception: PerceptronLayer3D class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "PerceptronLayer3D element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Layer name

    const tinyxml2::XMLElement* layer_name_element = perceptron_layer_element->FirstChildElement("LayerName");

    if(!layer_name_element)
    {
        buffer << "OpenNN Exception: PerceptronLayer3D class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "LayerName element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(layer_name_element->GetText())
    {
        set_name(layer_name_element->GetText());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = perceptron_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: PerceptronLayer3D class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "InputsNumber element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(inputs_number_element->GetText())
    {
        set_inputs_number(Index(stoi(inputs_number_element->GetText())));
    }

    // Inputs depth

    const tinyxml2::XMLElement* inputs_depth_element = perceptron_layer_element->FirstChildElement("InputsDepth");

    if(!inputs_depth_element)
    {
        buffer << "OpenNN Exception: PerceptronLayer3D class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "InputsDepth element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(inputs_depth_element->GetText())
    {
        set_inputs_depth(Index(stoi(inputs_depth_element->GetText())));
    }

    // Neurons number

    const tinyxml2::XMLElement* neurons_number_element = perceptron_layer_element->FirstChildElement("NeuronsNumber");

    if(!neurons_number_element)
    {
        buffer << "OpenNN Exception: PerceptronLayer3D class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "NeuronsNumber element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(neurons_number_element->GetText())
    {
        set_neurons_number(Index(stoi(neurons_number_element->GetText())));
    }

    // Activation function

    const tinyxml2::XMLElement* activation_function_element = perceptron_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
    {
        buffer << "OpenNN Exception: PerceptronLayer3D class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "ActivationFunction element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(activation_function_element->GetText())
    {
        set_activation_function(activation_function_element->GetText());
    }

    // Parameters

    const tinyxml2::XMLElement* parameters_element = perceptron_layer_element->FirstChildElement("Parameters");

    if(!parameters_element)
    {
        buffer << "OpenNN Exception: PerceptronLayer3D class.\n"
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


void PerceptronLayer3D::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Perceptron layer

    file_stream.OpenElement("PerceptronLayer3D");

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

    // Inputs depth
    file_stream.OpenElement("InputsDepth");

    buffer.str("");
    buffer << get_inputs_depth();

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

    // Peceptron layer (end tag)

    file_stream.CloseElement();
}


pair<type*, dimensions> PerceptronLayer3DForwardPropagation::get_outputs_pair() const
{
    PerceptronLayer3D* perceptron_layer_3d = static_cast<PerceptronLayer3D*>(layer);

    const Index neurons_number = perceptron_layer_3d->get_neurons_number();

    const Index inputs_number = perceptron_layer_3d->get_inputs_number();

    return pair<type*, dimensions>(outputs_data, { batch_samples_number, inputs_number, neurons_number });
}

void PerceptronLayer3DForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    PerceptronLayer3D* perceptron_layer_3d = static_cast<PerceptronLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = perceptron_layer_3d->get_neurons_number();

    const Index inputs_number = perceptron_layer_3d->get_inputs_number();

    outputs.resize(batch_samples_number, inputs_number, neurons_number);

    outputs_data = outputs.data();

    activations_derivatives.resize(batch_samples_number, inputs_number, neurons_number);
}

void PerceptronLayer3DBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    PerceptronLayer3D* perceptron_layer_3d = static_cast<PerceptronLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = perceptron_layer_3d->get_neurons_number();
    const Index inputs_number = perceptron_layer_3d->get_inputs_number();
    const Index inputs_depth = perceptron_layer_3d->get_inputs_depth();

    biases_derivatives.resize(neurons_number);

    synaptic_weights_derivatives.resize(inputs_depth, neurons_number);

    error_combinations_derivatives.resize(batch_samples_number, inputs_number, neurons_number);

    input_derivatives.resize(batch_samples_number, inputs_number, inputs_depth);

    inputs_derivatives.resize(1);
    inputs_derivatives(0).first = input_derivatives.data();
    inputs_derivatives(0).second = { batch_samples_number, inputs_number, inputs_depth };
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
