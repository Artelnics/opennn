//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn_strings.h"
#include "probabilistic_layer.h"

namespace opennn
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
        activation_function = ActivationFunction::Softmax;
    }
}


void ProbabilisticLayer::set_name(const string& new_layer_name)
{
    layer_name = new_layer_name;
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
/// ("Competitive", "Softmax" or "NoProbabilistic").

string ProbabilisticLayer::write_activation_function() const
{
    if(activation_function == ActivationFunction::Binary)
    {
        return "Binary";
    }
    else if(activation_function == ActivationFunction::Logistic)
    {
        return "Logistic";
    }
    else if(activation_function == ActivationFunction::Competitive)
    {
        return "Competitive";
    }
    else if(activation_function == ActivationFunction::Softmax)
    {
        return "Softmax";
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "string write_activation_function() const method.\n"
               << "Unknown probabilistic method.\n";

        throw runtime_error(buffer.str());
    }
}


/// Returns a string with the probabilistic method for the outputs to be included in some text
/// ("competitive", "softmax" or "no probabilistic").

string ProbabilisticLayer::write_activation_function_text() const
{
    if(activation_function == ActivationFunction::Binary)
    {
        return "binary";
    }
    else if(activation_function == ActivationFunction::Logistic)
    {
        return "logistic";
    }
    else if(activation_function == ActivationFunction::Competitive)
    {
        return "competitive";
    }
    else if(activation_function == ActivationFunction::Softmax)
    {
        return "softmax";
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "string write_activation_function_text() const method.\n"
               << "Unknown probabilistic method.\n";

        throw runtime_error(buffer.str());
    }
}


/// Returns true if messages from this class are displayed on the screen, or false if messages
/// from this class are not displayed on the screen.

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


/// Returns the number of parameters (biases and synaptic weights) of the layer.

Index ProbabilisticLayer::get_parameters_number() const
{
    return biases.size() + synaptic_weights.size();
}


/// Returns a single vector with all the layer parameters.
/// The format is a vector of real values.
/// The size is the number of parameters in the layer.

Tensor<type, 1> ProbabilisticLayer::get_parameters() const
{
    Tensor<type, 1> parameters(synaptic_weights.size() + biases.size());

    copy(execution::par, 
        synaptic_weights.data(),
        synaptic_weights.data() + synaptic_weights.size(),
        parameters.data());

    copy(execution::par, 
        biases.data(),
        biases.data() + biases.size(),
        parameters.data() + synaptic_weights.size());

    return parameters;
}


/// Sets a probabilistic layer with zero probabilistic neurons.
/// It also sets the rest of the members to their default values.

void ProbabilisticLayer::set()
{
    biases.resize(0);

    synaptic_weights.resize(0,0);

    set_default();
}


/// Resizes the size of the probabilistic layer.
/// It also sets the rest of the class members to their default values.
/// @param new_neurons_number New size for the probabilistic layer.

void ProbabilisticLayer::set(const Index& new_inputs_number, const Index& new_neurons_number)
{
    biases.resize(new_neurons_number);

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

    biases.resize(neurons_number);

    synaptic_weights.resize(new_inputs_number, neurons_number);
}


void ProbabilisticLayer::set_neurons_number(const Index& new_neurons_number)
{
    const Index inputs_number = get_inputs_number();

    biases.resize(new_neurons_number);

    synaptic_weights.resize(inputs_number, new_neurons_number);
}


void ProbabilisticLayer::set_biases(const Tensor<type, 1>& new_biases)
{
    biases = new_biases;
}


void ProbabilisticLayer::set_synaptic_weights(const Tensor<type, 2>& new_synaptic_weights)
{
    synaptic_weights = new_synaptic_weights;
}


void ProbabilisticLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    const Index biases_number = biases.size();
    const Index synaptic_weights_number = synaptic_weights.size();

    copy(execution::par, 
        new_parameters.data() + index,
        new_parameters.data() + index + synaptic_weights_number,
        synaptic_weights.data());

    copy(execution::par, 
        new_parameters.data() + index + synaptic_weights_number,
        new_parameters.data() + index + synaptic_weights_number + biases_number,
        biases.data());
}


/// Sets a new threshold value for discriminating between two classes.
/// @param new_decision_threshold New discriminating value. It must be comprised between 0 and 1.

void ProbabilisticLayer::set_decision_threshold(const type& new_decision_threshold)
{
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

    layer_type = Layer::Type::Probabilistic;

    const Index neurons_number = get_neurons_number();

    if(neurons_number == 1)
    {
        activation_function = ActivationFunction::Logistic;
    }
    else
    {
        activation_function = ActivationFunction::Softmax;
    }

    decision_threshold = type(0.5);

    display = true;
}


/// Sets the chosen method for probabilistic postprocessing.
/// Current probabilistic methods include Binary, Probability, Competitive and Softmax.
/// @param new_activation_function Method for interpreting the outputs as probabilistic values.

void ProbabilisticLayer::set_activation_function(const ActivationFunction& new_activation_function)
{
#ifdef OPENNN_DEBUG

    const Index neurons_number = get_neurons_number();

    if(neurons_number == 1 && new_activation_function == ActivationFunction::Competitive)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_activation_function(const ActivationFunction&) method.\n"
               << "Activation function cannot be Competitive when the number of neurons is 1.\n";

        throw runtime_error(buffer.str());
    }

    if(neurons_number == 1 && new_activation_function == ActivationFunction::Softmax)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_activation_function(const ActivationFunction&) method.\n"
               << "Activation function cannot be Softmax when the number of neurons is 1.\n";

        throw runtime_error(buffer.str());
    }

    if(neurons_number != 1 && new_activation_function == ActivationFunction::Binary)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_activation_function(const ActivationFunction&) method.\n"
               << "Activation function cannot be Binary when the number of neurons is greater than 1.\n";

        throw runtime_error(buffer.str());
    }

    if(neurons_number != 1 && new_activation_function == ActivationFunction::Logistic)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_activation_function(const ActivationFunction&) method.\n"
               << "Activation function cannot be Logistic when the number of neurons is greater than 1.\n";

        throw runtime_error(buffer.str());
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
        set_activation_function(ActivationFunction::Binary);
    }
    else if(new_activation_function == "Logistic")
    {
        set_activation_function(ActivationFunction::Logistic);
    }
    else if(new_activation_function == "Competitive")
    {
        set_activation_function(ActivationFunction::Competitive);
    }
    else if(new_activation_function == "Softmax")
    {
        set_activation_function(ActivationFunction::Softmax);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_activation_function(const string&) method.\n"
               << "Unknown probabilistic method: " << new_activation_function << ".\n";

        throw runtime_error(buffer.str());
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
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
    biases.setRandom();

    synaptic_weights.setRandom();
}


void ProbabilisticLayer::insert_parameters(const Tensor<type, 1>& parameters, const Index&)
{
    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    copy(execution::par, 
        parameters.data(),
         parameters.data() + biases_number,
         biases.data());

    copy(execution::par, 
        parameters.data() + biases_number,
         parameters.data() + biases_number + synaptic_weights_number,
         synaptic_weights.data());
}


void ProbabilisticLayer::calculate_combinations(const Tensor<type, 2>& inputs,
                                                Tensor<type, 2>& combinations) const
{
    combinations.device(*thread_pool_device) = inputs.contract(synaptic_weights, A_B);

    sum_columns(thread_pool_device, biases, combinations);
}


void ProbabilisticLayer::calculate_activations(const Tensor<type, 2>& combinations,
                                               Tensor<type, 2>& activations) const
{
    switch(activation_function)
    {
    case ActivationFunction::Binary: binary(combinations, activations); return;

    case ActivationFunction::Logistic: logistic(combinations, activations); return;

    case ActivationFunction::Competitive: competitive(combinations, activations); return;

    case ActivationFunction::Softmax: softmax(combinations, activations); return;

    default: return;
    }
}


void ProbabilisticLayer::calculate_activations_derivatives(const Tensor<type, 2>& combinations,
                                                           Tensor<type, 2>& activations,
                                                           Tensor<type, 2>& activations_derivatives) const
{

    switch(activation_function)
    {
    case ActivationFunction::Logistic:
        
        logistic_derivatives(combinations,
                             activations,
                             activations_derivatives);

        return;

    default:

        return;
    }
}



void ProbabilisticLayer::calculate_activations_derivatives(const Tensor<type, 2>& combinations,
                                                           Tensor<type, 2>& activations,
                                                           Tensor<type, 3>& activations_derivatives) const
{

    switch (activation_function)
    {
    case ActivationFunction::Softmax:

        softmax_derivatives(combinations,
                            activations,
                            activations_derivatives);
        return;

    default:

        return;
    }
}


void ProbabilisticLayer::forward_propagate(const pair<type*, dimensions>& inputs_pair,
                                           LayerForwardPropagation* forward_propagation,
                                           const bool& is_training)
{
    const Index neurons_number = get_neurons_number();

    const TensorMap<Tensor<type, 2>> inputs(inputs_pair.first, inputs_pair.second[0][0], inputs_pair.second[0][1]);

    ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation
            = static_cast<ProbabilisticLayerForwardPropagation*>(forward_propagation);

    Tensor<type, 2>& outputs = probabilistic_layer_forward_propagation->outputs;

    calculate_combinations(inputs, outputs);

    if(is_training)
    {
        if (neurons_number == 1)
        {
            Tensor<type, 2>& activations_derivatives_2d = probabilistic_layer_forward_propagation->activations_derivatives_2d;

            calculate_activations_derivatives(outputs,
                                              outputs,
                                              activations_derivatives_2d);
        }
        else
        {
            Tensor<type, 3>& activations_derivatives_3d = probabilistic_layer_forward_propagation->activations_derivatives_3d;

            calculate_activations_derivatives(outputs,
                                              outputs,
                                              activations_derivatives_3d);
        }
        
    }
    else
    {
        calculate_activations(outputs,
                              outputs);
    }
}


void ProbabilisticLayer::calculate_error_gradient(const pair<type*, dimensions>& inputs_pair,
                                                  LayerForwardPropagation* forward_propagation,
                                                  LayerBackPropagation* back_propagation) const
{
    const Index samples_number = inputs_pair.second[0][0];
    const Index neurons_number = get_neurons_number();

    const TensorMap<Tensor<type, 2>> inputs(inputs_pair.first, samples_number, neurons_number);

    // Forward propagation

    ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation =
            static_cast<ProbabilisticLayerForwardPropagation*>(forward_propagation);

    // Back propagation

    ProbabilisticLayerBackPropagation* probabilistic_layer_back_propagation =
            static_cast<ProbabilisticLayerBackPropagation*>(back_propagation);

    const Tensor<type,2>& deltas = probabilistic_layer_back_propagation->deltas;

    Tensor<type, 2>& error_combinations_derivatives = probabilistic_layer_back_propagation->error_combinations_derivatives;

    if (neurons_number == 1)
    {
        const Tensor<type, 2>& activations_derivatives_2d = probabilistic_layer_forward_propagation->activations_derivatives_2d;

        calculate_error_combinations_derivatives(deltas, activations_derivatives_2d, error_combinations_derivatives);
    }
    else
    {
        const Tensor<type, 3>& activations_derivatives_3d = probabilistic_layer_forward_propagation->activations_derivatives_3d;

        calculate_error_combinations_derivatives(deltas, activations_derivatives_3d, error_combinations_derivatives);
    }

    Tensor<type, 1>& biases_derivatives = probabilistic_layer_back_propagation->biases_derivatives;
    Tensor<type, 2>& synaptic_weights_derivatives = probabilistic_layer_back_propagation->synaptic_weights_derivatives;

    const Eigen::array<Index, 1> sum_dimensions({0});

    synaptic_weights_derivatives.device(*thread_pool_device) = inputs.contract(error_combinations_derivatives, AT_B);

    biases_derivatives.device(*thread_pool_device) = error_combinations_derivatives.sum(sum_dimensions);
}


void ProbabilisticLayer::insert_gradient(LayerBackPropagation* back_propagation,
                                         const Index& index,
                                         Tensor<type, 1>& gradient) const
{
    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    const ProbabilisticLayerBackPropagation* probabilistic_layer_back_propagation =
            static_cast<ProbabilisticLayerBackPropagation*>(back_propagation);

    const type* synaptic_weights_derivatives_data = probabilistic_layer_back_propagation->synaptic_weights_derivatives.data();
    const type* biases_derivatives_data = probabilistic_layer_back_propagation->biases_derivatives.data();

    copy(execution::par, 
         synaptic_weights_derivatives_data,
         synaptic_weights_derivatives_data + synaptic_weights_number,
         gradient.data() + index);

    copy(execution::par, 
         biases_derivatives_data,
         biases_derivatives_data + biases_number,
         gradient.data() + index + synaptic_weights_number);
}


void ProbabilisticLayer::calculate_error_combinations_derivatives(const Tensor<type, 2>& deltas,
                                                                  const Tensor<type, 2>& activations_derivatives_2d,
                                                                  Tensor<type, 2>& error_combinations_derivatives) const
{
    error_combinations_derivatives.device(*thread_pool_device) = deltas * activations_derivatives_2d;
}



void ProbabilisticLayer::calculate_error_combinations_derivatives(const Tensor<type, 2>& deltas,
                                                                  const Tensor<type, 3>& activations_derivatives_3d,
                                                                  Tensor<type, 2>& error_combinations_derivatives) const
{
    const Index samples_number = deltas.dimension(0);

    const Index neurons_number = get_neurons_number();

    type* activations_derivatives_3d_data = (type*)activations_derivatives_3d.data();

    const Index step = neurons_number * neurons_number;

    for (Index i = 0; i < samples_number; i++)
    {
        const Tensor<type, 1> deltas_row = deltas.chip(i, 0);

        error_combinations_derivatives.chip(i, 0).device(*thread_pool_device)
            = deltas_row.contract(activations_derivatives_3d.chip(i, 0), AT_B);
    }
}


void ProbabilisticLayer::calculate_squared_errors_Jacobian_lm(const Tensor<type, 2>& inputs,
                                                              LayerForwardPropagation* forward_propagation,
                                                              LayerBackPropagationLM* back_propagation)
{

    const Index samples_number = inputs.dimension(0);

    const Index inputs_number = get_inputs_number();

    const Index neurons_number = get_neurons_number();

    const Index synaptic_weights_number = get_synaptic_weights_number();

    // Forward propagation

    ProbabilisticLayerForwardPropagation* probabilistic_layer_forward_propagation =
            static_cast<ProbabilisticLayerForwardPropagation*>(forward_propagation);

    const Tensor<type, 2>& activations_derivatives_2d = probabilistic_layer_forward_propagation->activations_derivatives_2d;
    const Tensor<type, 3>& activations_derivatives_3d = probabilistic_layer_forward_propagation->activations_derivatives_3d;

    // Back propagation

    ProbabilisticLayerBackPropagationLM* probabilistic_layer_back_propagation_lm =
            static_cast<ProbabilisticLayerBackPropagationLM*>(back_propagation);

    const Tensor<type, 2>& deltas = probabilistic_layer_back_propagation_lm->deltas;

    Tensor<type, 2>& error_combinations_derivatives = probabilistic_layer_back_propagation_lm->error_combinations_derivatives;

    if (neurons_number == 1)
    {
        calculate_error_combinations_derivatives(deltas, activations_derivatives_2d, error_combinations_derivatives);
    }
    else
    {
        calculate_error_combinations_derivatives(deltas, activations_derivatives_3d, error_combinations_derivatives);
    }

    Tensor<type, 2>& squared_errors_Jacobian = probabilistic_layer_back_propagation_lm->squared_errors_Jacobian;

    Index synaptic_weight_index = 0;

    for (Index neuron_index = 0; neuron_index < neurons_number; neuron_index++)
    {
        const TensorMap<Tensor<type, 1>> error_combinations_derivatives_neuron = tensor_map(error_combinations_derivatives, neuron_index);

        for (Index input_index = 0; input_index < inputs_number; input_index++)
        {
            const TensorMap<Tensor<type, 1>> input = tensor_map(inputs, input_index);

            TensorMap<Tensor<type, 1>> squared_errors_jacobian_synaptic_weight = tensor_map(squared_errors_Jacobian, synaptic_weight_index);

            squared_errors_jacobian_synaptic_weight.device(*thread_pool_device) = error_combinations_derivatives_neuron * input;

            synaptic_weight_index++;
        }

        // bias

        Index bias_index = synaptic_weights_number + neuron_index;

        TensorMap<Tensor<type, 1>> squared_errors_jacobian_bias = tensor_map(squared_errors_Jacobian, bias_index);

        squared_errors_jacobian_bias.device(*thread_pool_device) = error_combinations_derivatives_neuron;
    }
}


void ProbabilisticLayer::insert_squared_errors_Jacobian_lm(LayerBackPropagationLM* back_propagation,
                                                           const Index& index,
                                                           Tensor<type, 2>& squared_errors_Jacobian) const
{
    ProbabilisticLayerBackPropagationLM* probabilistic_layer_back_propagation_lm =
            static_cast<ProbabilisticLayerBackPropagationLM*>(back_propagation);

    const Index batch_samples_number = back_propagation->batch_samples_number;

    const Index layer_parameters_number = get_parameters_number();

    copy(execution::par, 
         probabilistic_layer_back_propagation_lm->squared_errors_Jacobian.data(),
         probabilistic_layer_back_propagation_lm->squared_errors_Jacobian.data()+ layer_parameters_number*batch_samples_number,
         squared_errors_Jacobian.data() + index);
}


/// Serializes the probabilistic layer object into an XML document of the TinyXML library without keeping the DOM tree in memory.
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

        throw runtime_error(buffer.str());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = probabilistic_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Inputs number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    Index new_inputs_number;

    if(inputs_number_element->GetText())
    {
        new_inputs_number = Index(stoi(inputs_number_element->GetText()));
    }

    // Neurons number

    const tinyxml2::XMLElement* neurons_number_element = probabilistic_layer_element->FirstChildElement("NeuronsNumber");

    if(!inputs_number_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Neurons number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    Index new_neurons_number;

    if(neurons_number_element->GetText())
    {
        new_neurons_number = Index(stoi(neurons_number_element->GetText()));
    }

    set(new_inputs_number, new_neurons_number);

    // Activation function

    const tinyxml2::XMLElement* activation_function_element = probabilistic_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Activation function element is nullptr.\n";

        throw runtime_error(buffer.str());
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

        throw runtime_error(buffer.str());
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

        throw runtime_error(buffer.str());
    }

    if(decision_threshold_element->GetText())
    {
        set_decision_threshold(type(atof(decision_threshold_element->GetText())));
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
        catch(const exception& e)
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


string ProbabilisticLayer::write_combinations(const Tensor<string, 1>& inputs_names) const
{
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    for(Index i = 0; i < neurons_number; i++)
    {
        buffer << "probabilistic_layer_combinations_" << to_string(i) << " = " << biases(i);

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
        case ActivationFunction::Binary:
        {
            buffer << "\tif" << "probabilistic_layer_combinations_" << to_string(i) << " < 0.5, " << outputs_names(i) << "= 0.0. Else " << outputs_names(i) << " = 1.0\n";
        }
            break;

        case ActivationFunction::Logistic:
        {
            buffer <<  outputs_names(i) << " = 1.0/(1.0 + exp(-" <<  "probabilistic_layer_combinations_" << to_string(i) << ") );\n";
        }
            break;

        case ActivationFunction::Competitive:
            if(i == 0)
            {
                buffer << "\tfor each probabilistic_layer_combinations_i:"<<endl;

                buffer <<"\t\tif probabilistic_layer_combinations_i is equal to max(probabilistic_layer_combinations_i):"<<endl;

                buffer <<"\t\t\tactivations[i] = 1"<<endl;

                buffer <<"\t\telse:"<<endl;

                buffer <<"\t\t\tactivations[i] = 0"<<endl;
            }

            break;

        case ActivationFunction::Softmax:

            if(i == 0)
            {
                buffer << "sum = ";

                for(Index i = 0; i < neurons_number; i++)
                {
                    buffer << "exp(probabilistic_layer_combinations_" << to_string(i) << ")";

                    if(i != neurons_number-1) buffer << " + ";
                }

                buffer << ";\n" << endl;

                for(Index i = 0; i < neurons_number; i++)
                {
                    buffer << outputs_names(i) << " = exp(probabilistic_layer_combinations_" << to_string(i) <<")/sum;\n";
                }

            }
            break;
        default:
            break;
        }
    }

    return buffer.str();
}


string ProbabilisticLayer::write_expression(const Tensor<string, 1>& inputs_names,
                                            const Tensor<string, 1>& outputs_names) const
{
    ostringstream buffer;

    buffer << write_combinations(inputs_names);

    buffer << write_activations(outputs_names);

    return buffer.str();
}

ProbabilisticLayerForwardPropagation::ProbabilisticLayerForwardPropagation()
    : LayerForwardPropagation()
{

}


ProbabilisticLayerForwardPropagation::ProbabilisticLayerForwardPropagation(
    const Index& new_batch_samples_number, Layer *new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


ProbabilisticLayerForwardPropagation::~ProbabilisticLayerForwardPropagation()
{

}


std::pair<type *, dimensions> ProbabilisticLayerForwardPropagation::get_outputs_pair() const 
{
    const Index neurons_number = layer->get_neurons_number();

    return pair<type *, dimensions>(outputs_data, {{batch_samples_number, neurons_number}});
}


void ProbabilisticLayerForwardPropagation::set(const Index &new_batch_samples_number, Layer *new_layer) 
{
    layer = new_layer;

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = layer->get_neurons_number();

    outputs.resize(batch_samples_number, neurons_number);

    outputs_data = outputs.data();

    activations_derivatives_2d.resize(0, 0);
    activations_derivatives_3d.resize(0, 0, 0);

    if (neurons_number == 1)
    {
        activations_derivatives_2d.resize(batch_samples_number, neurons_number);
    }
    else
    {
        activations_derivatives_3d.resize(batch_samples_number, neurons_number, neurons_number);
    }
}


void ProbabilisticLayerForwardPropagation::print() const 
{
    cout << "Probabilistic layer forward-propagation" << endl;

    cout << "Outputs:" << endl;
    cout << outputs << endl;

    const Index neurons_number = layer->get_neurons_number();

    if (neurons_number == 1)
    {
        cout << "Activations derivatives:" << endl;
        cout << activations_derivatives_2d << endl;
    }
    else
    {
        cout << "Activations derivatives:" << endl;
        cout << activations_derivatives_3d << endl;
    }
}


ProbabilisticLayerBackPropagation::ProbabilisticLayerBackPropagation() : LayerBackPropagation() 
{
}


ProbabilisticLayerBackPropagation::~ProbabilisticLayerBackPropagation() 
{

}


ProbabilisticLayerBackPropagation::ProbabilisticLayerBackPropagation(const Index &new_batch_samples_number, Layer *new_layer)
    : LayerBackPropagation() 
{
    set(new_batch_samples_number, new_layer);
}


std::pair<type *, dimensions> ProbabilisticLayerBackPropagation::get_deltas_pair() const 
{
    const Index neurons_number = layer->get_neurons_number();

    return pair<type *, dimensions>(deltas_data, {{batch_samples_number, neurons_number}});
}


void ProbabilisticLayerBackPropagation::set(const Index &new_batch_samples_number, Layer *new_layer) 
{
    layer = new_layer;

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = layer->get_neurons_number();
    const Index inputs_number = layer->get_inputs_number();

    deltas.resize(batch_samples_number, neurons_number);

    deltas_data = deltas.data();

    biases_derivatives.resize(neurons_number);

    synaptic_weights_derivatives.resize(inputs_number, neurons_number);

    deltas_row.resize(neurons_number);
    activations_derivatives_matrix.resize(neurons_number, neurons_number);

    error_combinations_derivatives.resize(batch_samples_number, neurons_number);

    error_combinations_derivatives.resize(batch_samples_number, neurons_number);
}


void ProbabilisticLayerBackPropagation::print() const 
{
    cout << "Deltas:" << endl;
    cout << deltas << endl;

    cout << "Biases derivatives:" << endl;
    cout << biases_derivatives << endl;

    cout << "Synaptic weights derivatives:" << endl;
    cout << synaptic_weights_derivatives << endl;
}
} // namespace opennn

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
