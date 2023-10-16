//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   A T T E N T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "multihead_attention_layer.h"

namespace opennn
{

/// Default constructor.
/// It creates a empty layer object.
/// This constructor also initializes the rest of the class members to their default values.


MultiheadAttentionLayer::MultiheadAttentionLayer() : Layer()
{
    set();

    layer_type = Type::MultiheadAttention;
}


/// Layer architecture constructor.
/// It creates a layer object with given input size, embedding depth and number of attention heads.
/// It initializes the parameters at random.
/// This constructor also initializes the rest of the class members to their default values.

MultiheadAttentionLayer::MultiheadAttentionLayer(const Index& new_input_size,
                                                 const Index& embedding_depth,
                                                 const Index& number_of_heads,
                                                 const MultiheadAttentionLayer::ActivationFunction& new_activation_function) : Layer()
{
    set(new_input_size, embedding_depth, number_of_heads, new_activation_function);

    layer_type = Type::MultiheadAttention;

    layer_name = "multihead_attention_layer";
}


/// Returns the size of the input to the layer.

Index MultiheadAttentionLayer::get_input_size() const
{
    return input_size;
}


/// Returns the embedding depth used in the layer.

Index MultiheadAttentionLayer::get_depth() const
{
    return depth;
}


/// Returns the number of attention heads of the layer.

Index MultiheadAttentionLayer::get_number_of_heads() const
{
    return number_of_heads;
}


/// Each returns one of layer's Perceptron sub-layers.

PerceptronLayer MultiheadAttentionLayer::get_input_perceptron() const
{
    return input_perceptron_layer;
}

PerceptronLayer MultiheadAttentionLayer::get_context_perceptron() const
{
    return context_perceptron_layer;
}

PerceptronLayer MultiheadAttentionLayer::get_output_perceptron() const
{
    return output_perceptron_layer;
}


/// Returns the number of parameters of the layer.

Index MultiheadAttentionLayer::get_parameters_number() const
{
    Index input_perceptron_parameters = input_perceptron_layer.get_parameters_number();
    Index context_perceptron_parameters = context_perceptron_layer.get_parameters_number();
    Index output_perceptron_parameters = output_perceptron_layer.get_parameters_number();
    return input_perceptron_parameters + context_perceptron_parameters + output_perceptron_parameters;
}


/// Returns the activation function of the layer.
/// The activation function of a layer is the activation function of all perceptrons in it.

const MultiheadAttentionLayer::ActivationFunction& MultiheadAttentionLayer::get_activation_function() const
{
    return activation_function;
}


/// Returns a string with the name of the layer activation function.
/// This can be Logistic, HyperbolicTangent, Threshold, SymmetricThreshold, Linear, RectifiedLinear, ScaledExponentialLinear.

string MultiheadAttentionLayer::write_activation_function() const
{
    switch(activation_function)
    {
    case ActivationFunction::Logistic:
        return "Logistic";

    case ActivationFunction::HyperbolicTangent:
        return "HyperbolicTangent";

    case ActivationFunction::Threshold:
        return "Threshold";

    case ActivationFunction::SymmetricThreshold:
        return "SymmetricThreshold";

    case ActivationFunction::Linear:
        return "Linear";

    case ActivationFunction::RectifiedLinear:
        return "RectifiedLinear";

    case ActivationFunction::ScaledExponentialLinear:
        return "ScaledExponentialLinear";

    case ActivationFunction::SoftPlus:
        return "SoftPlus";

    case ActivationFunction::SoftSign:
        return "SoftSign";

    case ActivationFunction::HardSigmoid:
        return "HardSigmoid";

    case ActivationFunction::ExponentialLinear:
        return "ExponentialLinear";
    }

    return string();
}


/// Returns true if messages from this class are displayed on the screen,
/// or false if messages from this class are not displayed on the screen.

const bool& MultiheadAttentionLayer::get_display() const
{
    return display;
}


/// Sets an empty layer.
/// It also sets the rest of the members to their default values.

void MultiheadAttentionLayer::set()
{
    input_size = 0;

    depth = 0;

    number_of_heads = 0;

    input_perceptron_layer.set();
    context_perceptron_layer.set();
    output_perceptron_layer.set();

    set_default();
}


/// Sets new input size, embedding depth, number of attention heads and activation function of the layer.
/// It also sets the rest of the members to their default values.

void MultiheadAttentionLayer::set(const Index& new_input_size, const Index& new_depth, const Index& new_number_of_heads,
                          const MultiheadAttentionLayer::ActivationFunction& new_activation_function)
{
    input_size = new_input_size;

    depth = new_depth;

    number_of_heads = new_number_of_heads;

    set_perceptrons();

    activation_function = new_activation_function;

    set_default();
}


/// Sets those members not related to the perceptrons to their default value.

void MultiheadAttentionLayer::set_default()
{
    layer_name = "multihead_attention_layer";

    display = true;

    layer_type = Type::MultiheadAttention;
}


void MultiheadAttentionLayer::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
}


/// Sets a new input size in the layer.

void MultiheadAttentionLayer::set_input_size(const Index& new_input_size)
{
    input_size = new_input_size;

    set_perceptrons();
}


/// Sets a new embedding depth in the layer.

void MultiheadAttentionLayer::set_depth(const Index& new_depth)
{
    depth = new_depth;

    set_perceptrons();
}


/// Sets a new number of attention heads in the layer.

void MultiheadAttentionLayer::set_number_of_heads(const Index& new_number_of_heads)
{
    number_of_heads = new_number_of_heads;

    set_perceptrons();
}


/// Sets the perceptron sub-layers according to the layer's parameters.

void MultiheadAttentionLayer::set_perceptrons()
{
    input_perceptron_layer.set(input_size*depth, input_size*depth*number_of_heads);
    context_perceptron_layer.set(input_size*depth, input_size*depth*number_of_heads);
    output_perceptron_layer.set(input_size*depth*number_of_heads, input_size*depth);
}

/// This class sets a new activation(or transfer) function in the layer.

void MultiheadAttentionLayer::set_activation_function(const MultiheadAttentionLayer::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


/// Sets a new activation(or transfer) function in a single layer.
/// The second argument is a string containing the name of the function("Logistic", "HyperbolicTangent", "Threshold", etc).
/// @param new_activation_function Activation function for that layer.

void MultiheadAttentionLayer::set_activation_function(const string& new_activation_function_name)
{
    if(new_activation_function_name == "Logistic")
    {
        activation_function = ActivationFunction::Logistic;
    }
    else if(new_activation_function_name == "HyperbolicTangent")
    {
        activation_function = ActivationFunction::HyperbolicTangent;
    }
    else if(new_activation_function_name == "Threshold")
    {
        activation_function = ActivationFunction::Threshold;
    }
    else if(new_activation_function_name == "SymmetricThreshold")
    {
        activation_function = ActivationFunction::SymmetricThreshold;
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

void MultiheadAttentionLayer::set_display(const bool& new_display)
{
    display = new_display;
}


/// Computes the attention scores by comparing (via dot product) input and context.
/// Attention scores must be computed separately for each batch element and each attention head.

void MultiheadAttentionLayer::compute_attention_scores(type* input_data,
                                                       const Tensor<Index, 1>& input_dimensions,
                                                       type* context_data,
                                                       const Tensor<Index, 1>& context_dimensions,
                                                       type* attention_scores_data)
{
    if(input_dimensions(0) != context_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void compute_attention_scores(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*) method.\n"
               << "Input batch size (" << input_dimensions(0) << ") and context batch size (" << context_dimensions(0) << ") not equal.\n";

        throw invalid_argument(buffer.str());
    }

    const Index batch_size = input_dimensions(0);

    if(input_size*depth*number_of_heads != input_dimensions(1))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void compute_attention_scores(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*) method.\n"
               << "Invalid input dimensions: " << input_dimensions << ".\n";

        throw invalid_argument(buffer.str());
    }

    if(input_size*depth*number_of_heads != context_dimensions(1))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void compute_attention_scores(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*) method.\n"
               << "Invalid context dimensions: " << context_dimensions << ".\n";

        throw invalid_argument(buffer.str());
    }

    TensorMap<Tensor<type, 4>>  input(input_data, batch_size, input_size, depth, number_of_heads);
    const TensorMap<Tensor<type, 4>>  context(context_data, batch_size, input_size, depth, number_of_heads);

    const Tensor<type, 4> scaled_input = input / input.setConstant(sqrt(depth));

    TensorMap<Tensor<type, 4>> attention_scores(attention_scores_data, batch_size, input_size, input_size, number_of_heads);

    Tensor<type, 3> input_i(input_size, depth, number_of_heads);
    Tensor<type, 3> context_i(input_size, depth, number_of_heads);

    Tensor<type, 2> input_ij(input_size, depth);
    Tensor<type, 2> context_ij(input_size, depth);

//    Tensor<type, 4> raw_attention_scores(batch_size, input_size, input_size, number_of_heads);
//    Tensor<Index, 1> attention_scores_dimensions = {input_size, input_size};

#pragma omp parallel for collapse(2)
        for(Index i = 0; i < batch_size; i++)
        {
            input_i = scaled_input.chip(i, 0);
            context_i = context.chip(i, 0);

                for(Index j = 0; j < number_of_heads ; j++)
                {
                    input_ij = input_i.chip(j, 2);
                    context_ij = context_i.chip(j, 2);

                    attention_scores.chip(i, 0).chip(j, 2) = input_ij.contract(context_ij, A_BT);
//                    raw_attention_scores.chip(i, 0).chip(j, 2) = input_ij.contract(context_ij, A_BT);
//                    softmax(raw_attention_scores.data(), attention_scores_dimensions,
//                            attention_scores.chip(i, 0).chip(j, 2).data(), attention_scores_dimensions);

                }
        };
    /// @todo softmax attention scores (rows of each A_ij)
    /// softmax();
    /// maybe add dropout?
}


void MultiheadAttentionLayer::compute_attention_output(type* value_data,
                                                       const Tensor<Index, 1>& value_dimensions,
                                                       type* attention_scores_data,
                                                       const Tensor<Index, 1>& attention_scores_dimensions,
                                                       type* attention_output_data)
{    

    if(value_dimensions(0) != attention_scores_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void compute_attention_output(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*) method.\n"
               << "Value batch size (" << value_dimensions(0) << ") and attention scores batch size (" << attention_scores_dimensions(0) << ") not equal.\n";

        throw invalid_argument(buffer.str());
    }

    const Index batch_size = value_dimensions(0);

    if(input_size*depth*number_of_heads != value_dimensions(1))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void compute_attention_output(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*) method.\n"
               << "Invalid value dimensions: " << value_dimensions << ".\n";

        throw invalid_argument(buffer.str());
    }

    if(input_size != attention_scores_dimensions(1) || input_size != attention_scores_dimensions(2) || number_of_heads != attention_scores_dimensions(3))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void compute_attention_output(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&, type*) method.\n"
               << "Invalid attention scores dimensions: " << attention_scores_dimensions << ".\n";

        throw invalid_argument(buffer.str());
    }

    TensorMap<Tensor<type, 4>> value(value_data, batch_size, input_size, depth, number_of_heads);
    TensorMap<Tensor<type, 4>> attention_scores(attention_scores_data, batch_size, input_size, input_size, number_of_heads);

    TensorMap<Tensor<type, 4>> attention_output(attention_output_data, batch_size, input_size, depth, number_of_heads);

    Tensor<type, 3> value_i(input_size, depth, number_of_heads);
    Tensor<type, 3> attention_scores_i(input_size, input_size, number_of_heads);

    Tensor<type, 2> value_ij(input_size, depth);
    Tensor<type, 2> attention_scores_ij(input_size, input_size);

#pragma omp parallel for collapse(2)
        for(Index i = 0; i < batch_size; i++)
        {
            value_i = value.chip(i, 0);
            attention_scores_i = attention_scores.chip(i, 0);

                for(Index j = 0; j < number_of_heads ; j++)
                {
                    value_ij = value_i.chip(j, 2);
                    attention_scores_ij = attention_scores_i.chip(j, 2);
                    attention_output.chip(i, 0).chip(j, 2) = value_ij.contract(attention_scores_ij, AT_B);
                }
        };
}


void MultiheadAttentionLayer::forward_propagate(type* inputs_data,
                                        const Tensor<Index,1>& inputs_dimensions,
                                        LayerForwardPropagation* forward_propagation,
                                        bool& switch_train)
{

    if(inputs_dimensions(1) != input_size*depth*2)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: MultiheadAttentionLayer class.\n"
               << "void MultiheadAttentionLayer::forward_propagate(type*, const Tensor<Index, 1>&, type*, Tensor<Index, 1>&)\n"
               << "Inputs columns number must be equal to " << input_size*depth*2 << ", (" << inputs_dimensions(1) << ").\n";
        throw invalid_argument(buffer.str());
    }

    const Index batch_size = inputs_dimensions(0);

    MultiheadAttentionLayerForwardPropagation* multihead_attention_layer_forward_propagation
        = static_cast<MultiheadAttentionLayerForwardPropagation*>(forward_propagation);

    Tensor<Index, 1> flattened_input_dimension(2);

    flattened_input_dimension.setValues({batch_size, input_size*depth});

    PerceptronLayerForwardPropagation input_perceptron_forward_propagation =
        multihead_attention_layer_forward_propagation->input_perceptron_forward_propagation;

    input_perceptron_layer.forward_propagate(inputs_data,
                                              flattened_input_dimension,
                                              &input_perceptron_forward_propagation,
                                              switch_train);

    PerceptronLayerForwardPropagation context_perceptron_forward_propagation =
        multihead_attention_layer_forward_propagation->context_perceptron_forward_propagation;

    context_perceptron_layer.forward_propagate(inputs_data + batch_size*input_size*depth,
                                                     flattened_input_dimension,
                                                     &context_perceptron_forward_propagation,
                                                     switch_train);

    type* attention_scores_data = multihead_attention_layer_forward_propagation->get_attention_scores_data();

    compute_attention_scores(input_perceptron_forward_propagation.outputs_data,
                             input_perceptron_forward_propagation.outputs_dimensions,
                             context_perceptron_forward_propagation.outputs_data,
                             context_perceptron_forward_propagation.outputs_dimensions,
                             attention_scores_data);

    const Tensor<Index, 1> attention_scores_dimensions = get_dimensions(multihead_attention_layer_forward_propagation->attention_scores);

    type* attention_output_data = multihead_attention_layer_forward_propagation->get_attention_output_data();

    compute_attention_output(context_perceptron_forward_propagation.outputs_data,
                             context_perceptron_forward_propagation.outputs_dimensions,
                             attention_scores_data,
                             attention_scores_dimensions,
                             attention_output_data);

    PerceptronLayerForwardPropagation output_perceptron_forward_propagation =
        multihead_attention_layer_forward_propagation->output_perceptron_forward_propagation;

    Tensor<Index, 1> flattened_attention_output_dimension(2);

    flattened_attention_output_dimension.setValues({batch_size, input_size*depth*number_of_heads});

    output_perceptron_layer.forward_propagate(attention_output_data,
                                              flattened_attention_output_dimension,
                                              &output_perceptron_forward_propagation,
                                              switch_train);

    memcpy(multihead_attention_layer_forward_propagation->outputs_data,
           output_perceptron_forward_propagation.outputs_data,
           static_cast<size_t>(batch_size*input_size*depth*sizeof(type)));
}

/*
void PerceptronLayer::forward_propagate(type* inputs_data,
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


string MultiheadAttentionLayer::write_activation_function_expression() const
{
    switch(activation_function)
    {
    case ActivationFunction::Threshold:
        return "threshold";

    case ActivationFunction::SymmetricThreshold:
        return "symmetric_threshold";

    case ActivationFunction::Logistic:
        return "logistic";

    case ActivationFunction::HyperbolicTangent:
        return "tanh";

    case ActivationFunction::Linear:
        return string();

    case ActivationFunction::RectifiedLinear:
        return "ReLU";

    case ActivationFunction::ExponentialLinear:
        return "ELU";

    case ActivationFunction::ScaledExponentialLinear:
        return "SELU";

    case ActivationFunction::SoftPlus:
        return "soft_plus";

    case ActivationFunction::SoftSign:
        return "soft_sign";

    case ActivationFunction::HardSigmoid:
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
