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

PerceptronLayer::PerceptronLayer(const Index& new_inputs_number, const Index& new_neurons_number,
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


Tensor<Index, 1> PerceptronLayer::get_input_variables_dimensions() const
{
    const Index inputs_number = get_inputs_number();

    return Tensor<Index, 1>(inputs_number);
}


/// Returns the number of inputs to the layer.

Index PerceptronLayer::get_inputs_number() const
{
    return synaptic_weights.dimension(0);

}


/// Returns the number of neurons in the layer.

Index PerceptronLayer::get_neurons_number() const
{
    return biases.size();
}


Index PerceptronLayer::get_biases_number() const
{
    return biases.size();
}


Index PerceptronLayer::get_synaptic_weights_number() const
{
    return synaptic_weights.size();
}

/// Returns the number of parameters(biases and synaptic weights) of the layer.

Index PerceptronLayer::get_parameters_number() const
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
    const Index inputs_number = get_inputs_number();

    const Index neurons_number = get_neurons_number();

    const Index synaptic_weights_number = get_synaptic_weights_number();

    Tensor<type, 1> new_synaptic_weights = parameters.slice(Eigen::array<Eigen::Index, 1>({0}), Eigen::array<Eigen::Index, 1>({synaptic_weights_number}));

    Eigen::array<Index, 2> two_dim{{inputs_number, neurons_number}};

    return new_synaptic_weights.reshape(two_dim);

}



Tensor<type, 2> PerceptronLayer::get_biases(const Tensor<type, 1>& parameters) const
{
    const Index biases_number = biases.size();

    const Index parameters_size = parameters.size();

    const Index start_bias = (parameters_size - biases_number);

    Tensor<type,1> new_biases(biases_number);

    new_biases = parameters.slice(Eigen::array<Eigen::Index, 1>({start_bias}), Eigen::array<Eigen::Index, 1>({biases_number}));

    Eigen::array<Index, 2> two_dim{{1, biases.dimension(1)}};

    return new_biases.reshape(two_dim);

}


/// Returns a single vector with all the layer parameters.
/// The format is a vector of real values.
/// The size is the number of parameters in the layer.

Tensor<type, 1> PerceptronLayer:: get_parameters() const
{
    Eigen::array<Index, 1> one_dim_weight{{synaptic_weights.dimension(0)*synaptic_weights.dimension(1)}};

    Eigen::array<Index, 1> one_dim_bias{biases.dimension(1)};

    Tensor<type, 1> synaptic_weights_vector = synaptic_weights.reshape(one_dim_weight);

    Tensor<type, 1> biases_vector = biases.reshape(one_dim_bias);

    Tensor<type, 1> parameters(synaptic_weights_vector.size() + biases_vector.size());

    Index index = 0;

    for(Index i = 0; i < synaptic_weights_vector.dimension(0); i++)
    {
        parameters(i) = synaptic_weights_vector(i);

        index++;
    }

    for(Index i=0; i< biases_vector.dimension(0); i++)
    {
        parameters(i + index) = biases_vector(i);
    }

    return parameters;
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

void PerceptronLayer::set(const Index& new_inputs_number, const Index& new_neurons_number,
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

void PerceptronLayer::set_inputs_number(const Index& new_inputs_number)
{
    const Index neurons_number = get_neurons_number();

    biases.resize(neurons_number,1);

    synaptic_weights.resize(new_inputs_number, neurons_number);
}


/// Sets a new number perceptrons in the layer.
/// All the parameters are also initialized at random.
/// @param new_neurons_number New number of neurons in the layer.

void PerceptronLayer::set_neurons_number(const Index& new_neurons_number)
{
    const Index inputs_number = get_inputs_number();

    biases.resize(new_neurons_number, 1);

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
#ifdef __OPENNN_DEBUG__

    const Index new_parameters_size = new_parameters.size();
    const Index parameters_number = get_parameters_number();

    if(new_parameters_size != parameters_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PerceptronLayer class.\n"
               << "void set_parameters(const Tensor<type, 1>&) method.\n"
               << "Size of new parameters (" << new_parameters_size << ") must be equal to number of parameters (" << parameters_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    memcpy(synaptic_weights.data(), new_parameters.data(), static_cast<size_t>(synaptic_weights_number)*sizeof(type));
    memcpy(biases.data(), new_parameters.data() + synaptic_weights_number, static_cast<size_t>(biases_number)*sizeof(type));
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


/// Initializes the biases of all the perceptrons in the layer of perceptrons with a given value.
/// @param value Biases initialization value.

void PerceptronLayer::initialize_biases(const type& value)
{
    biases.setConstant(value);
}


/// Initializes the synaptic weights of all the perceptrons in the layer of perceptrons with a given value.
/// @param value Synaptic weights initialization value.

void PerceptronLayer::initialize_synaptic_weights(const type& value)
{
    synaptic_weights.setConstant(value);
}


/// Initializes the synaptic weights of all the perceptrons in the layer of perceptrons with glorot uniform distribution.

void PerceptronLayer::initialize_synaptic_weights_glorot_uniform()
{
    Index fan_in;
    Index fan_out;

    type scale = 1.0;
    type limit;

    fan_in = synaptic_weights.dimension(0);
    fan_out = synaptic_weights.dimension(1);

    scale /= ((fan_in + fan_out) / static_cast<type>(2.0));
    limit = sqrt(static_cast<type>(3.0) * scale);

    /*
        synaptic_weights.setRandom(-limit, limit);
    */
}


/// Initializes all the biases and synaptic weights in the neural newtork with a given value.
/// @param value Parameters initialization value.

void PerceptronLayer::set_parameters_constant(const type& value)
{

    biases.setConstant(value);

    synaptic_weights.setConstant(value);

}


/// Initializes all the biases and synaptic weights in the neural newtork at random with values comprised
/// between -1 and +1.

void PerceptronLayer::set_parameters_random()
{
    biases.setRandom();

    synaptic_weights.setRandom();
}


Tensor<type, 2> PerceptronLayer::calculate_outputs(const Tensor<type, 2>& inputs)
{
    const Index inputs_dimensions_number = inputs.rank();

#ifdef __OPENNN_DEBUG__

    if(inputs_dimensions_number != 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PerceptronLayer class.\n"
               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
               << "Number of dimensions (" << inputs_dimensions_number << ") must be equal to 2.\n";

        throw logic_error(buffer.str());
    }

    const Index inputs_number = get_inputs_number();

    const Index inputs_columns_number = inputs.dimension(1);

    if(inputs_columns_number != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PerceptronLayer class.\n"
               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
               << "Number of columns (" << inputs_columns_number << ") must be equal to number of inputs (" << inputs_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    const Index batch_size = inputs.dimension(0);
    const Index outputs_number = get_neurons_number();

    Tensor<type, 2> outputs(batch_size, outputs_number);

    calculate_combinations(inputs, biases, synaptic_weights, outputs);

    calculate_activations(outputs, outputs);

    return outputs;
}


Tensor<type, 2> PerceptronLayer::calculate_outputs(const Tensor<type, 2>& inputs, const Tensor<type, 1>& parameters)
{
//    const Tensor<type, 2> synaptic_weights = get_synaptic_weights(parameters);
//    const Tensor<type, 2> biases = get_biases(parameters);

//    return calculate_outputs(inputs, biases, synaptic_weights);

    return Tensor<type, 2>();
}


/// Returns a string with the expression of the inputs-outputs relationship of the layer.
/// @param inputs_names vector of strings with the name of the layer inputs.
/// @param outputs_names vector of strings with the name of the layer outputs.

string PerceptronLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    const Index inputs_number = get_inputs_number();
    const Index inputs_name_size = inputs_names.size();

    if(inputs_name_size != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PerceptronLayer class.\n"
               << "string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const method.\n"
               << "Size of inputs name must be equal to number of layer inputs.\n";

        throw logic_error(buffer.str());
    }

    const Index outputs_name_size = outputs_names.size();

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

    for(Index j = 0; j < outputs_names.size(); j++)
    {
        /*
               buffer << outputs_names[j] << " = " << write_activation_function_expression() << " (" << biases[j] << "+";

               for(Index i = 0; i < inputs_names.size() - 1; i++)
               {

                   buffer << " (" << inputs_names[i] << "*" << synaptic_weights.get_column(j)(i) << ")+";
               }

               buffer << " (" << inputs_names[inputs_names.size() - 1] << "*" << synaptic_weights.get_column(j)[inputs_names.size() - 1] << "));\n";
        */
    }

    return buffer.str();
}


string PerceptronLayer::object_to_string() const
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

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
        set_inputs_number(static_cast<Index>(stoi(inputs_number_element->GetText())));
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
        set_neurons_number(static_cast<Index>(stoi(neurons_number_element->GetText())));
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
//        set_parameters(to_type_vector(parameters_string, ' '));
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


string PerceptronLayer::write_activation_function_expression() const
{
    switch(activation_function)
    {
    case HyperbolicTangent:
        return "tanh";

    case Linear:
        return "";

    default:
        return write_activation_function();
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
