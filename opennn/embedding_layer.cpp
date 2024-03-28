//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "embedding_layer.h"

namespace opennn
{

/// Default constructor.
/// It creates a empty layer object.
/// This constructor also initializes the rest of the class members to their default values.


EmbeddingLayer::EmbeddingLayer() : Layer()
{
    set();

    layer_type = Type::Embedding;
}


/// Layer architecture constructor.
/// It creates a layer object with given input dimension, input length and embedding depth.
/// It initializes the parameters at random.
/// This constructor also initializes the rest of the class members to their default values.

EmbeddingLayer::EmbeddingLayer(const Index& new_input_dim,
                               const Index& new_input_length,
                               const Index& new_depth,
                               const bool& new_positional_encoding,
                               const PerceptronLayer::ActivationFunction& new_activation_function) : Layer()
{
    set(new_input_dim, new_input_length, new_depth, new_positional_encoding, new_activation_function);

    layer_type = Type::Embedding;

    layer_name = "embedding_layer";
}


/// Returns the dimension (maximum value + 1) of the input to the layer.

Index EmbeddingLayer::get_input_dim() const
{
    return input_dim;
}


/// Returns the length of the input to the layer.

Index EmbeddingLayer::get_input_length() const
{
    return input_length;
}


/// Returns the embedding depth to be used in the layer.

Index EmbeddingLayer::get_depth() const
{
    return depth;
}


/// Each returns the layer's Perceptron sub-layer.

PerceptronLayer EmbeddingLayer::get_perceptron_layer() const
{
    return perceptron_layer;
}


/// Returns the number of parameters of the layer.

Index EmbeddingLayer::get_parameters_number() const
{
    return perceptron_layer.get_parameters_number();
}


/// Returns the activation function of the layer.
/// The activation function of a layer is the activation function of all perceptrons in it.

const PerceptronLayer::ActivationFunction& EmbeddingLayer::get_activation_function() const
{
    return activation_function;
}


/// Returns a string with the name of the layer activation function.
/// This can be Logistic, HyperbolicTangent, Threshold, SymmetricThreshold, Linear, RectifiedLinear, ScaledExponentialLinear.

string EmbeddingLayer::write_activation_function() const
{
    switch(activation_function)
    {
    case PerceptronLayer::ActivationFunction::Logistic:
        return "Logistic";

    case PerceptronLayer::ActivationFunction::HyperbolicTangent:
        return "HyperbolicTangent";

    case PerceptronLayer::ActivationFunction::Threshold:
        return "Threshold";

    case PerceptronLayer::ActivationFunction::SymmetricThreshold:
        return "SymmetricThreshold";

    case PerceptronLayer::ActivationFunction::Linear:
        return "Linear";

    case PerceptronLayer::ActivationFunction::RectifiedLinear:
        return "RectifiedLinear";

    case PerceptronLayer::ActivationFunction::ScaledExponentialLinear:
        return "ScaledExponentialLinear";

    case PerceptronLayer::ActivationFunction::SoftPlus:
        return "SoftPlus";

    case PerceptronLayer::ActivationFunction::SoftSign:
        return "SoftSign";

    case PerceptronLayer::ActivationFunction::HardSigmoid:
        return "HardSigmoid";

    case PerceptronLayer::ActivationFunction::ExponentialLinear:
        return "ExponentialLinear";
    }

    return string();
}


/// Returns true if messages from this class are displayed on the screen,
/// or false if messages from this class are not displayed on the screen.

const bool& EmbeddingLayer::get_display() const
{
    return display;
}


/// Sets an empty layer.
/// It also sets the rest of the members to their default values.

void EmbeddingLayer::set()
{
    input_dim = 0;

    input_length = 0;

    depth = 0;

    positional_encoding = false;

    perceptron_layer.set();

    set_default();
}


/// Sets new input dimension, input length, embedding depth and activation function of the layer.
/// It also sets the rest of the members to their default values.

void EmbeddingLayer::set(const Index& new_input_dim,
                         const Index& new_input_length,
                         const Index& new_depth,
                         const bool& new_positional_encoding,
                         const PerceptronLayer::ActivationFunction& new_activation_function)
{
    input_dim = new_input_dim;

    input_length = new_input_length;

    depth = new_depth;

    set_perceptron();

    positional_encoding = new_positional_encoding;

    activation_function = new_activation_function;

    set_default();
}


/// Sets those members not related to the perceptrons to their default value.

void EmbeddingLayer::set_default()
{
    layer_name = "embedding_layer";

    display = true;

    layer_type = Type::Embedding;
}


void EmbeddingLayer::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}


/// Sets a new input dim in the layer.

void EmbeddingLayer::set_input_dim(const Index& new_input_dim)
{
    input_dim = new_input_dim;

    set_perceptron();
}


/// Sets a new input length in the layer.

void EmbeddingLayer::set_input_length(const Index& new_input_length)
{
    input_length = new_input_length;
}


/// Sets a new embedding depth in the layer.

void EmbeddingLayer::set_depth(const Index& new_depth)
{
    depth = new_depth;

    set_perceptron();
}


/// Sets the perceptron sub-layer according to the layer's parameters.

void EmbeddingLayer::set_perceptron()
{
    perceptron_layer.set(input_dim, depth, activation_function);
}

/// This class sets a new activation(or transfer) function in the layer.

void EmbeddingLayer::set_activation_function(const PerceptronLayer::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


/// Sets a new activation(or transfer) function in a single layer.
/// The second argument is a string containing the name of the function("Logistic", "HyperbolicTangent", "Threshold", etc).
/// @param new_activation_function Activation function for that layer.

void EmbeddingLayer::set_activation_function(const string& new_activation_function_name)
{
    if(new_activation_function_name == "Logistic")
    {
        activation_function = PerceptronLayer::ActivationFunction::Logistic;
    }
    else if(new_activation_function_name == "HyperbolicTangent")
    {
        activation_function = PerceptronLayer::ActivationFunction::HyperbolicTangent;
    }
    else if(new_activation_function_name == "Threshold")
    {
        activation_function = PerceptronLayer::ActivationFunction::Threshold;
    }
    else if(new_activation_function_name == "SymmetricThreshold")
    {
        activation_function = PerceptronLayer::ActivationFunction::SymmetricThreshold;
    }
    else if(new_activation_function_name == "Linear")
    {
        activation_function = PerceptronLayer::ActivationFunction::Linear;
    }
    else if(new_activation_function_name == "RectifiedLinear")
    {
        activation_function = PerceptronLayer::ActivationFunction::RectifiedLinear;
    }
    else if(new_activation_function_name == "ScaledExponentialLinear")
    {
        activation_function = PerceptronLayer::ActivationFunction::ScaledExponentialLinear;
    }
    else if(new_activation_function_name == "SoftPlus")
    {
        activation_function = PerceptronLayer::ActivationFunction::SoftPlus;
    }
    else if(new_activation_function_name == "SoftSign")
    {
        activation_function = PerceptronLayer::ActivationFunction::SoftSign;
    }
    else if(new_activation_function_name == "HardSigmoid")
    {
        activation_function = PerceptronLayer::ActivationFunction::HardSigmoid;
    }
    else if(new_activation_function_name == "ExponentialLinear")
    {
        activation_function = PerceptronLayer::ActivationFunction::ExponentialLinear;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: PerceptronLayer class.\n"
               << "void set_activation_function(const string&) method.\n"
               << "Unknown activation function: " << new_activation_function_name << ".\n";

        throw invalid_argument(buffer.str());
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void EmbeddingLayer::set_display(const bool& new_display)
{
    display = new_display;
}


/// Calculates one-hot encoding, of dimension = input_dim, of an input row (assuming all input values are integers)
/// @return Matrix of one-hot encodings of all values in input_row

Tensor<type, 2> EmbeddingLayer::one_hot_encode_row(const Tensor<type, 1>& input_row)
{
    Tensor<type, 2> one_hot_encoded_input_row(input_length, input_dim);
    one_hot_encoded_input_row.setZero();

    const Tensor<type, 0> max_input = input_row.maximum();

    if(max_input(0) >= type(input_dim))
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: EmbeddingLayer class.\n"
               << "void EmbeddingLayer::one_hot_encode_row(const Tensor<Index, 1>&)\n"
               << "All input values must be less than " << input_dim << " (" << max_input(0) << ").\n";
        throw invalid_argument(buffer.str());
    }

#pragma omp parallel for
    for(Index i = 0; i < input_length; i++)
        one_hot_encoded_input_row(i, Index(input_row(i))) = 1;

    return one_hot_encoded_input_row;
}


/// Looks up embedding of an input row, by passing its one-hot encoding through a perceptron layer (that corresponds to the lookup table)
/// Saves the embedding matrix of the row in outputs_data of the given perceptron layer forward propagation structure

void EmbeddingLayer::lookup_embedding(const Tensor<type, 1>& input_row,
                                    PerceptronLayerForwardPropagation* perceptron_forward_propagation,
                                    const bool& is_training)
{
    Tensor<type, 2> one_hot_encoded_input_row = one_hot_encode_row(input_row);

    Tensor<type*, 1> one_hot_encoded_input_row_data(1);
    one_hot_encoded_input_row_data.setValues({one_hot_encoded_input_row.data()});

    Tensor<Tensor<Index, 1>, 1> one_hot_encoded_input_row_dimensions;
    one_hot_encoded_input_row_dimensions.setValues({get_dimensions(one_hot_encoded_input_row)});

//    perceptron_layer.forward_propagate(one_hot_encoded_input_row_data,
//                                       one_hot_encoded_input_row_dimensions, // input_length is batch_size
//                                       perceptron_forward_propagation,
//                                       is_training);
};


/// Builds positional encoding matrix with dimensions (input_length, depth) of the layer.

const Tensor<type, 2> EmbeddingLayer::build_positional_encoding_matrix()
{
    Tensor<type, 2> positional_encoding_matrix(input_length, depth);
    positional_encoding_matrix.setZero();

    const Index half_depth = type(depth)/2;

#pragma omp parallel for collapse(2)
    for(Index i = 0; i < input_length; i++)
    {
        for(Index j = 0; j < half_depth - 1; j++)
        {
            positional_encoding_matrix(i, 2*j) = sin( (i + 1) / pow(100000, (j + 1) / half_depth) );
            positional_encoding_matrix(i, 2*j+1) = cos( (i + 1) / pow(100000, (j + 1) / half_depth) );
        }
    }

#pragma omp parallel for
    for(Index i = 0; i < input_length; i++)
    {
        if(depth % 2 == 0)
        {
            positional_encoding_matrix(i, depth - 2) = sin( (i+1) / 10000 );
            positional_encoding_matrix(i, depth - 1) = cos( (i+1) / 10000 );
        }
        else
        {
            positional_encoding_matrix(i, depth - 1) = sin( (i+1) / 10000 );
        }
    }

    return positional_encoding_matrix;

};


void EmbeddingLayer::forward_propagate(const Tensor<DynamicTensor<type>, 1>& inputs,
                                        LayerForwardPropagation* forward_propagation,
                                        const bool& is_training)
{
    if(inputs(0).get_dimension(1) != input_length)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: EmbeddingLayer class.\n"
               << "void EmbeddingLayer::forward_propagate(const Tensor<DynamicTensor<type>, 1>&, type*, Tensor<Index, 1>&)\n"
               << "Inputs columns number must be equal to " << input_length << ", (" << inputs(0).get_dimension(1) << ").\n";
        throw invalid_argument(buffer.str());
    }

    const Index batch_size = inputs(0).get_dimension(0);

    const TensorMap<Tensor<type, 2>> inputs_tensor_map = inputs(0).to_tensor_map<2>();

    EmbeddingLayerForwardPropagation* embedding_layer_forward_propagation
        = static_cast<EmbeddingLayerForwardPropagation*>(forward_propagation);

    PerceptronLayerForwardPropagation perceptron_layer_forward_propagation =
        embedding_layer_forward_propagation->perceptron_forward_propagation;

    Tensor<type, 1> input_row(input_length);

//#pragma omp parallel for
    for(Index i = 0; i < batch_size; i++)
    {
        input_row = inputs_tensor_map.chip(i, 0);
        lookup_embedding(input_row, &perceptron_layer_forward_propagation, is_training);

        memcpy(embedding_layer_forward_propagation->outputs(0).get_data() + i*input_length*depth,
               perceptron_layer_forward_propagation.outputs(0).get_data(),
               static_cast<size_t>(input_length*depth*sizeof(type)));
    }

    if(positional_encoding)
    {
        TensorMap<Tensor<type, 3>> outputs = embedding_layer_forward_propagation->outputs(0).to_tensor_map<3>();

        const Tensor<type, 2> positional_encoding_matrix = build_positional_encoding_matrix();
#pragma omp parallel for
        for(Index i = 0; i < batch_size; i++)
            outputs.chip(i, 0) += positional_encoding_matrix;
    }
}

/*
void EmbeddingLayer::forward_propagate(type* inputs_data,
                                        const Tensor<Index, 1>& inputs_dimensions,
                                        Tensor<type, 1>& potential_parameters,
                                        LayerForwardPropagation* forward_propagation)
{
#ifdef OPENNN_DEBUG
    if(inputs_dimensions(1) != get_inputs_number())
    {
        ostringstream buffer;
        buffer << "OpenNN Exception:" << LOG << endl
               << "void forward_propagate(type*, const Tensor<Index, 1>&, Tensor<type, 1>&, LayerForwardPropagation*) final method.\n"
               << "Inputs columns number must be equal to " << get_inputs_number() << ", (inputs number).\n";

        throw invalid_argument(buffer.str());
    }

    check_size(potential_parameters, get_parameters_number(), LOG);
#endif

    const TensorMap<Tensor<type, 2>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1));

    const Index neurons_number = get_neurons_number();

    const Index inputs_number = get_inputs_number();

    const TensorMap<Tensor<type, 2>> potential_biases(potential_parameters.data(), 1, neurons_number);

    const TensorMap<Tensor<type, 2>> potential_synaptic_weights(potential_parameters.data()+neurons_number, inputs_number, neurons_number);

    PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation
        = static_cast<PerceptronLayerForwardPropagation*>(forward_propagation);

    const Tensor<Index, 1> activations_dimensions = perceptron_layer_forward_propagation->outputs_dimensions;

    const Tensor<Index, 1> combinations_dimensions = get_dimensions(perceptron_layer_forward_propagation->combinations);

    const Tensor<Index, 1> derivatives_dimensions = get_dimensions(perceptron_layer_forward_propagation->activations_derivatives);


    calculate_combinations(inputs,
                           potential_biases,
                           potential_synaptic_weights,
                           perceptron_layer_forward_propagation->get_combinations_data());


    calculate_activations_derivatives(perceptron_layer_forward_propagation->combinations.data(),
                                      combinations_dimensions,
                                      perceptron_layer_forward_propagation->outputs_data,
                                      activations_dimensions,
                                      perceptron_layer_forward_propagation->activations_derivatives.data(),
                                      derivatives_dimensions);
}
*/


string EmbeddingLayer::write_activation_function_expression() const
{
    switch(activation_function)
    {
    case PerceptronLayer::ActivationFunction::Threshold:
        return "threshold";

    case PerceptronLayer::ActivationFunction::SymmetricThreshold:
        return "symmetric_threshold";

    case PerceptronLayer::ActivationFunction::Logistic:
        return "logistic";

    case PerceptronLayer::ActivationFunction::HyperbolicTangent:
        return "tanh";

    case PerceptronLayer::ActivationFunction::Linear:
        return string();

    case PerceptronLayer::ActivationFunction::RectifiedLinear:
        return "ReLU";

    case PerceptronLayer::ActivationFunction::ExponentialLinear:
        return "ELU";

    case PerceptronLayer::ActivationFunction::ScaledExponentialLinear:
        return "SELU";

    case PerceptronLayer::ActivationFunction::SoftPlus:
        return "soft_plus";

    case PerceptronLayer::ActivationFunction::SoftSign:
        return "soft_sign";

    case PerceptronLayer::ActivationFunction::HardSigmoid:
        return "hard_sigmoid";

    default:
        return string();
    }
}

/// @todo
///// Returns a string with the expression of the inputs-outputs relationship of the layer.
///// @param inputs_names vector of strings with the name of the layer inputs.
///// @param outputs_names vector of strings with the name of the layer outputs.

//string PerceptronLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
//{
//#ifdef OPENNN_DEBUG
//    //    check_size(inputs_names, get_inputs_number(), LOG);
//    //    check_size(outputs_names, get_neurons_number(), LOG);
//#endif

//    ostringstream buffer;

//    for(Index j = 0; j < outputs_names.size(); j++)
//    {
//        const Tensor<type, 1> synaptic_weights_column =  synaptic_weights.chip(j,1);

//        buffer << outputs_names[j] << " = " << write_activation_function_expression() << "( " << biases(0,j) << " +";

//        for(Index i = 0; i < inputs_names.size() - 1; i++)
//        {
//            buffer << " (" << inputs_names[i] << "*" << synaptic_weights_column(i) << ") +";
//        }

//        buffer << " (" << inputs_names[inputs_names.size() - 1] << "*" << synaptic_weights_column[inputs_names.size() - 1] << ") );\n";
//    }

//    return buffer.str();
//}


//void PerceptronLayer::from_XML(const tinyxml2::XMLDocument& document)
//{
//    ostringstream buffer;

//    // Perceptron layer

//    const tinyxml2::XMLElement* perceptron_layer_element = document.FirstChildElement("PerceptronLayer");

//    if(!perceptron_layer_element)
//    {
//        buffer << "OpenNN Exception: PerceptronLayer class.\n"
//               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//               << "PerceptronLayer element is nullptr.\n";

//        throw invalid_argument(buffer.str());
//    }

//    // Layer name

//    const tinyxml2::XMLElement* layer_name_element = perceptron_layer_element->FirstChildElement("LayerName");

//    if(!layer_name_element)
//    {
//        buffer << "OpenNN Exception: PerceptronLayer class.\n"
//               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//               << "LayerName element is nullptr.\n";

//        throw invalid_argument(buffer.str());
//    }

//    if(layer_name_element->GetText())
//    {
//        set_name(layer_name_element->GetText());
//    }

//    // Inputs number

//    const tinyxml2::XMLElement* inputs_number_element = perceptron_layer_element->FirstChildElement("InputsNumber");

//    if(!inputs_number_element)
//    {
//        buffer << "OpenNN Exception: PerceptronLayer class.\n"
//               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//               << "InputsNumber element is nullptr.\n";

//        throw invalid_argument(buffer.str());
//    }

//    if(inputs_number_element->GetText())
//    {
//        set_inputs_number(static_cast<Index>(stoi(inputs_number_element->GetText())));
//    }

//    // Neurons number

//    const tinyxml2::XMLElement* neurons_number_element = perceptron_layer_element->FirstChildElement("NeuronsNumber");

//    if(!neurons_number_element)
//    {
//        buffer << "OpenNN Exception: PerceptronLayer class.\n"
//               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//               << "NeuronsNumber element is nullptr.\n";

//        throw invalid_argument(buffer.str());
//    }

//    if(neurons_number_element->GetText())
//    {
//        set_neurons_number(static_cast<Index>(stoi(neurons_number_element->GetText())));
//    }

//    // Activation function

//    const tinyxml2::XMLElement* activation_function_element = perceptron_layer_element->FirstChildElement("ActivationFunction");

//    if(!activation_function_element)
//    {
//        buffer << "OpenNN Exception: PerceptronLayer class.\n"
//               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//               << "ActivationFunction element is nullptr.\n";

//        throw invalid_argument(buffer.str());
//    }

//    if(activation_function_element->GetText())
//    {
//        set_activation_function(activation_function_element->GetText());
//    }

//    // Parameters

//    const tinyxml2::XMLElement* parameters_element = perceptron_layer_element->FirstChildElement("Parameters");

//    if(!parameters_element)
//    {
//        buffer << "OpenNN Exception: PerceptronLayer class.\n"
//               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//               << "Parameters element is nullptr.\n";

//        throw invalid_argument(buffer.str());
//    }

//    if(parameters_element->GetText())
//    {
//        const string parameters_string = parameters_element->GetText();

//        set_parameters(to_type_vector(parameters_string, ' '));
//    }
//}


//void PerceptronLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
//{
//    ostringstream buffer;

//    // Perceptron layer

//    file_stream.OpenElement("PerceptronLayer");

//    // Layer name
//    file_stream.OpenElement("LayerName");
//    buffer.str("");
//    buffer << layer_name;
//    file_stream.PushText(buffer.str().c_str());
//    file_stream.CloseElement();

//    // Inputs number
//    file_stream.OpenElement("InputsNumber");

//    buffer.str("");
//    buffer << get_inputs_number();

//    file_stream.PushText(buffer.str().c_str());

//    file_stream.CloseElement();

//    // Outputs number

//    file_stream.OpenElement("NeuronsNumber");

//    buffer.str("");
//    buffer << get_neurons_number();

//    file_stream.PushText(buffer.str().c_str());

//    file_stream.CloseElement();

//    // Activation function

//    file_stream.OpenElement("ActivationFunction");

//    file_stream.PushText(write_activation_function().c_str());

//    file_stream.CloseElement();

//    // Parameters

//    file_stream.OpenElement("Parameters");

//    buffer.str("");

//    const Tensor<type, 1> parameters = get_parameters();
//    const Index parameters_size = parameters.size();

//    for(Index i = 0; i < parameters_size; i++)
//    {
//        buffer << parameters(i);

//        if(i != (parameters_size-1)) buffer << " ";
//    }

//    file_stream.PushText(buffer.str().c_str());

//    file_stream.CloseElement();

//    // Peceptron layer (end tag)

//    file_stream.CloseElement();
//}


}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
