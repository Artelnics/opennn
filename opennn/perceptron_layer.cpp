//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "perceptron_layer.h"
#include "tensors.h"
#include "strings_utilities.h"

#include <iostream>
#include <map>
#include <functional>

namespace opennn
{

PerceptronLayer::PerceptronLayer() : Layer()
{
    set();

    layer_type = Type::Perceptron;
}


PerceptronLayer::PerceptronLayer(const dimensions& new_input_dimensions, const dimensions& new_output_dimensions,
                                 const ActivationFunction& new_activation_function)
{
    set(new_input_dimensions[0], new_output_dimensions[0], new_activation_function);
}


Index PerceptronLayer::get_inputs_number() const
{
    return synaptic_weights.dimension(0);
}


void PerceptronLayer::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


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


Index PerceptronLayer::get_parameters_number() const
{
    return biases.size() + synaptic_weights.size();
}


type PerceptronLayer::get_dropout_rate() const
{
    return dropout_rate;
}


dimensions PerceptronLayer::get_output_dimensions() const
{
    const Index neurons_number = get_neurons_number();

    return { neurons_number };
}


const Tensor<type, 1>& PerceptronLayer::get_biases() const
{
    return biases;
}


const Tensor<type, 2>& PerceptronLayer::get_synaptic_weights() const
{
    return synaptic_weights;
}


Tensor<type, 1> PerceptronLayer::get_parameters() const
{
    const Index synaptic_weights_number = synaptic_weights.size();
    const Index biases_number = biases.size();

    Tensor<type, 1> parameters(synaptic_weights_number + biases_number);

    memcpy(parameters.data(), synaptic_weights.data(),synaptic_weights_number*sizeof(type));

    memcpy(parameters.data() + synaptic_weights_number, biases.data(), biases_number*sizeof(type));

    return parameters;
}


const PerceptronLayer::ActivationFunction& PerceptronLayer::get_activation_function() const
{
    return activation_function;
}


string PerceptronLayer::write_activation_function() const
{
    switch(activation_function)
    {
    case ActivationFunction::Logistic:
        return "Logistic";

    case ActivationFunction::HyperbolicTangent:
        return "HyperbolicTangent";

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


const bool& PerceptronLayer::get_display() const
{
    return display;
}


void PerceptronLayer::set()
{
    biases.resize(0);

    synaptic_weights.resize(0, 0);

    set_default();
}


void PerceptronLayer::set(const Index& new_inputs_number, const Index& new_neurons_number,
                          const PerceptronLayer::ActivationFunction& new_activation_function)
{
    biases.resize(new_neurons_number);

    synaptic_weights.resize(new_inputs_number, new_neurons_number);

    set_parameters_random();

    activation_function = new_activation_function;

    set_default();
}


void PerceptronLayer::set_default()
{
    name = "perceptron_layer";

    display = true;

    layer_type = Type::Perceptron;
}


void PerceptronLayer::set_inputs_number(const Index& new_inputs_number)
{
    const Index neurons_number = get_neurons_number();

    biases.resize(neurons_number);

    synaptic_weights.resize(new_inputs_number, neurons_number);
}


void PerceptronLayer::set_neurons_number(const Index& new_neurons_number)
{
    const Index inputs_number = get_inputs_number();

    biases.resize(new_neurons_number);

    synaptic_weights.resize(inputs_number, new_neurons_number);
}


void PerceptronLayer::set_biases(const Tensor<type, 1>& new_biases)
{
    biases = new_biases;
}


void PerceptronLayer::set_synaptic_weights(const Tensor<type, 2>& new_synaptic_weights)
{
    synaptic_weights = new_synaptic_weights;
}


void PerceptronLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{   
    const type* new_parameters_data = new_parameters.data();
    type* synaptic_weights_data = synaptic_weights.data();
    type* biases_data = biases.data();

    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    memcpy(synaptic_weights_data, new_parameters_data + index, synaptic_weights_number*sizeof(type));

    memcpy(biases_data, new_parameters_data + index + synaptic_weights_number, biases_number*sizeof(type));
}


void PerceptronLayer::set_activation_function(const PerceptronLayer::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


void PerceptronLayer::set_activation_function(const string& new_activation_function_name)
{
    if(new_activation_function_name == "Logistic")
    {
        activation_function = ActivationFunction::Logistic;
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
        throw runtime_error("Unknown activation function: " + new_activation_function_name + ".\n");
    }
}


void PerceptronLayer::set_display(const bool& new_display)
{
    display = new_display;
}


void PerceptronLayer::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    synaptic_weights.setConstant(value);
}


void PerceptronLayer::set_parameters_random()
{
    set_random(biases);

    set_random(synaptic_weights);
}


void PerceptronLayer::calculate_combinations(const Tensor<type, 2>& inputs,
                                             Tensor<type, 2>& combinations) const
{
    combinations.device(*thread_pool_device) = inputs.contract(synaptic_weights, A_B);

    sum_columns(thread_pool_device, biases, combinations);
}


void PerceptronLayer::dropout(Tensor<type, 2>& outputs) const
{  
    const Index outputs_number = outputs.dimension(1);

    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    type random;

    for(Index neuron_index = 0; neuron_index < outputs_number; neuron_index++)
    {
        TensorMap<Tensor<type, 1>> column = tensor_map(outputs, neuron_index);

        random = calculate_random_uniform(type(0), type(1));

        random < dropout_rate ? column.setZero()
                              : column = column*scaling_factor;
    }
}


void PerceptronLayer::calculate_activations(Tensor<type, 2>& activations,
                                            Tensor<type, 2>& activations_derivatives) const
{
    switch(activation_function)
    {
    case ActivationFunction::Linear: linear(activations, activations_derivatives); return;

    case ActivationFunction::Logistic: logistic(activations, activations_derivatives);return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(activations, activations_derivatives); return;

	case ActivationFunction::RectifiedLinear: rectified_linear(activations, activations_derivatives); return;        

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(activations, activations_derivatives); return;

    case ActivationFunction::SoftPlus: soft_plus(activations, activations_derivatives);return;

    case ActivationFunction::SoftSign: soft_sign(activations, activations_derivatives); return;

    case ActivationFunction::HardSigmoid: hard_sigmoid(activations, activations_derivatives); return;

    case ActivationFunction::ExponentialLinear: exponential_linear(activations, activations_derivatives); return;

    default: return;
    }
}


void PerceptronLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                        LayerForwardPropagation* layer_forward_propagation,
                                        const bool& is_training)
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);

    PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation =
        static_cast<PerceptronLayerForwardPropagation*>(layer_forward_propagation);

    Tensor<type, 2>& outputs = perceptron_layer_forward_propagation->outputs;

    calculate_combinations(inputs,
                           outputs);

    if(is_training && dropout_rate > type(0))
    {
        dropout(outputs);
    }

    if(is_training)
    {
        Tensor<type, 2>& activations_derivatives = perceptron_layer_forward_propagation->activations_derivatives;

        calculate_activations(outputs, activations_derivatives);
    }
    else
    {
        Tensor<type, 2> empty;

        calculate_activations(outputs, empty);
    }
}


void PerceptronLayer::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                               const vector<pair<type*, dimensions>>& deltas_pair,
                                               LayerForwardPropagation* forward_propagation,
                                               LayerBackPropagation* back_propagation) const
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);
    const TensorMap<Tensor<type, 2>> deltas = tensor_map_2(deltas_pair[0]);

    // Forward propagation

    const PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation =
            static_cast<PerceptronLayerForwardPropagation*>(forward_propagation);

    const Tensor<type, 2>& activations_derivatives = perceptron_layer_forward_propagation->activations_derivatives;

    // Back propagation

    PerceptronLayerBackPropagation* perceptron_layer_back_propagation =
            static_cast<PerceptronLayerBackPropagation*>(back_propagation);
    
    Tensor<type, 2>& combinations_derivatives = perceptron_layer_back_propagation->combinations_derivatives;

    Tensor<type, 2>& synaptic_weights_derivatives = perceptron_layer_back_propagation->synaptic_weights_derivatives;

    Tensor<type, 1>& biases_derivatives = perceptron_layer_back_propagation->biases_derivatives;

    const bool& is_first_layer = perceptron_layer_back_propagation->is_first_layer;

    Tensor<type, 2>& input_derivatives = perceptron_layer_back_propagation->input_derivatives;
    
    combinations_derivatives.device(*thread_pool_device) = deltas * activations_derivatives;

    biases_derivatives.device(*thread_pool_device) = combinations_derivatives.sum(Eigen::array<Index, 1>({0}));

    synaptic_weights_derivatives.device(*thread_pool_device) = inputs.contract(combinations_derivatives, AT_B);

    if(!is_first_layer)
        input_derivatives.device(*thread_pool_device) = combinations_derivatives.contract(synaptic_weights, A_BT);
}


void PerceptronLayer::back_propagate_lm(const vector<pair<type*, dimensions>>& input_pairs,
                                        const vector<pair<type*, dimensions>>& deltas_pair,
                                        LayerForwardPropagation* forward_propagation,
                                        LayerBackPropagationLM* back_propagation) const
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);
    const TensorMap<Tensor<type, 2>> deltas = tensor_map_2(deltas_pair[0]);

    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    const Index synaptic_weights_number = get_synaptic_weights_number();

    // Forward propagation

    const PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation =
        static_cast<PerceptronLayerForwardPropagation*>(forward_propagation);

    const Tensor<type, 2>& activations_derivatives = perceptron_layer_forward_propagation->activations_derivatives;

    // Back propagation

    PerceptronLayerBackPropagationLM* perceptron_layer_back_propagation_lm =
        static_cast<PerceptronLayerBackPropagationLM*>(back_propagation);

    Tensor<type, 2>& combinations_derivatives = perceptron_layer_back_propagation_lm->combinations_derivatives;

    Tensor<type, 2>& squared_errors_Jacobian = perceptron_layer_back_propagation_lm->squared_errors_Jacobian;

    bool& is_first_layer = perceptron_layer_back_propagation_lm->is_first_layer;

    Tensor<type, 2>& input_derivatives = perceptron_layer_back_propagation_lm->input_derivatives;

    // Parameters derivatives
    
    combinations_derivatives.device(*thread_pool_device) = deltas * activations_derivatives;

    Index synaptic_weight_index = 0;

    for(Index neuron_index = 0; neuron_index < neurons_number; neuron_index++)
    {
        const TensorMap<Tensor<type, 1>> combinations_derivatives_neuron = tensor_map(combinations_derivatives, neuron_index);

        for(Index input_index = 0; input_index < inputs_number; input_index++)
        {
            const TensorMap<Tensor<type, 1>> input = tensor_map(inputs, input_index);

            TensorMap<Tensor<type, 1>> squared_errors_jacobian_synaptic_weight = tensor_map(squared_errors_Jacobian, synaptic_weight_index);

            squared_errors_jacobian_synaptic_weight.device(*thread_pool_device) = combinations_derivatives_neuron * input;

            synaptic_weight_index++;
        }

        // bias

        const Index bias_index = synaptic_weights_number + neuron_index;

        TensorMap<Tensor<type, 1>> squared_errors_jacobian_bias = tensor_map(squared_errors_Jacobian, bias_index);

        squared_errors_jacobian_bias.device(*thread_pool_device) = combinations_derivatives_neuron;
    }

    // Input derivatives

    if(!is_first_layer)
        input_derivatives.device(*thread_pool_device) = combinations_derivatives.contract(synaptic_weights, A_BT);
}


void PerceptronLayer::insert_gradient(LayerBackPropagation* back_propagation,
                                      const Index& index,
                                      Tensor<type, 1>& gradient) const
{
    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    PerceptronLayerBackPropagation* perceptron_layer_back_propagation =
        static_cast<PerceptronLayerBackPropagation*>(back_propagation);

    const Tensor<type, 2>& synaptic_weights_derivatives = perceptron_layer_back_propagation->synaptic_weights_derivatives;
    const Tensor<type, 1>& biases_derivatives = perceptron_layer_back_propagation->biases_derivatives;

    const type* synaptic_weights_derivatives_data = synaptic_weights_derivatives.data();
    const type* biases_derivatives_data = biases_derivatives.data();
    type* gradient_data = gradient.data();

    #pragma omp parallel sections
    {
        #pragma omp section
        memcpy(gradient_data + index, synaptic_weights_derivatives_data, synaptic_weights_number * sizeof(type));

        #pragma omp section
        memcpy(gradient_data + index + synaptic_weights_number, biases_derivatives_data, biases_number * sizeof(type));
    }
}


void PerceptronLayer::insert_squared_errors_Jacobian_lm(LayerBackPropagationLM* back_propagation,
                                                        const Index& index,
                                                        Tensor<type, 2>& squared_errors_Jacobian) const
{
    const Index layer_parameters_number = get_parameters_number();
    const Index batch_samples_number = back_propagation->batch_samples_number;

    PerceptronLayerBackPropagationLM* perceptron_layer_back_propagation_lm =
        static_cast<PerceptronLayerBackPropagationLM*>(back_propagation);

    type* squared_errors_Jacobian_data = perceptron_layer_back_propagation_lm->squared_errors_Jacobian.data();

    memcpy(squared_errors_Jacobian_data + index, squared_errors_Jacobian_data, layer_parameters_number * batch_samples_number*sizeof(type));
}


string PerceptronLayer::write_expression(const Tensor<string, 1>& inputs_name,
                                         const Tensor<string, 1>& output_names) const
{
    ostringstream buffer;

    for(Index j = 0; j < output_names.size(); j++)
    {
        const Tensor<type, 1> synaptic_weights_column =  synaptic_weights.chip(j,1);

        buffer << output_names[j] << " = " << write_activation_function_expression() << "( " << biases(j) << " +";

        for(Index i = 0; i < inputs_name.size() - 1; i++)
        {
            buffer << " (" << inputs_name[i] << "*" << synaptic_weights_column(i) << ") +";
        }

        buffer << " (" << inputs_name[inputs_name.size() - 1] << "*" << synaptic_weights_column[inputs_name.size() - 1] << "));\n";
    }

    return buffer.str();
}


void PerceptronLayer::print() const
{
    cout << "Perceptron layer" << endl;

    cout << "Inputs number: " << get_inputs_number() << endl;

    cout << "Neurons number: " << get_neurons_number() << endl;

    cout << "Synaptic weights dimensions: " << synaptic_weights.dimensions() << endl;
}


void PerceptronLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    // Perceptron layer

    const tinyxml2::XMLElement* perceptron_layer_element = document.FirstChildElement("PerceptronLayer");

    if(!perceptron_layer_element)
        throw runtime_error("PerceptronLayer element is nullptr.\n");

    // Layer name

    const tinyxml2::XMLElement* layer_name_element = perceptron_layer_element->FirstChildElement("Name");

    if(!layer_name_element)
        throw runtime_error("LayerName element is nullptr.\n");

    if(layer_name_element->GetText())
        set_name(layer_name_element->GetText());

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = perceptron_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
        throw runtime_error("InputsNumber element is nullptr.\n");

    if(inputs_number_element->GetText())
        set_inputs_number(Index(stoi(inputs_number_element->GetText())));

    // Neurons number

    const tinyxml2::XMLElement* neurons_number_element = perceptron_layer_element->FirstChildElement("NeuronsNumber");

    if(!neurons_number_element)
        throw runtime_error("NeuronsNumber element is nullptr.\n");

    if(neurons_number_element->GetText())
        set_neurons_number(Index(stoi(neurons_number_element->GetText())));

    // Activation function

    const tinyxml2::XMLElement* activation_function_element = perceptron_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
        throw runtime_error("ActivationFunction element is nullptr.\n");

    if(activation_function_element->GetText())
        set_activation_function(activation_function_element->GetText());

    // Parameters

    const tinyxml2::XMLElement* parameters_element = perceptron_layer_element->FirstChildElement("Parameters");

    if(!parameters_element)
        throw runtime_error("Parameters element is nullptr.\n");

    if(parameters_element->GetText())
        set_parameters(to_type_vector(parameters_element->GetText(), " "));
}


void PerceptronLayer::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Perceptron layer

    file_stream.OpenElement("PerceptronLayer");

    // Layer name

    file_stream.OpenElement("Name");
    file_stream.PushText(name.c_str());
    file_stream.CloseElement();

    // Inputs number

    file_stream.OpenElement("InputsNumber");
    file_stream.PushText(to_string(get_inputs_number()).c_str());
    file_stream.CloseElement();

    // Outputs number

    file_stream.OpenElement("NeuronsNumber");
    file_stream.PushText(to_string(get_neurons_number()).c_str());
    file_stream.CloseElement();

    // Activation function

    file_stream.OpenElement("ActivationFunction");
    file_stream.PushText(write_activation_function().c_str());
    file_stream.CloseElement();

    // Parameters

    file_stream.OpenElement("Parameters");
    file_stream.PushText(tensor_to_string(get_parameters()).c_str());
    file_stream.CloseElement();

    // Peceptron layer (end tag)

    file_stream.CloseElement();
}


string PerceptronLayer::write_activation_function_expression() const
{
    switch(activation_function)
    {
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


void PerceptronLayerForwardPropagation::set(const Index &new_batch_samples_number, Layer *new_layer)
{
    layer = new_layer;
    
    batch_samples_number = new_batch_samples_number;
    
    const Index neurons_number = layer->get_neurons_number();
    
    outputs.resize(batch_samples_number, neurons_number);
    
    outputs_data = outputs.data();
    
    activations_derivatives.resize(batch_samples_number, neurons_number);
}


pair<type *, dimensions> PerceptronLayerForwardPropagation::get_outputs_pair() const
{
    const Index neurons_number = layer->get_neurons_number();
    
    return pair<type *, dimensions>(outputs_data, {{batch_samples_number, neurons_number}});
}


PerceptronLayerForwardPropagation::PerceptronLayerForwardPropagation()
    : LayerForwardPropagation()
{

}


PerceptronLayerForwardPropagation::PerceptronLayerForwardPropagation(const Index &new_batch_samples_number,
                                                                     Layer *new_layer)
: LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


PerceptronLayerForwardPropagation::~PerceptronLayerForwardPropagation()
{

}


void PerceptronLayerForwardPropagation::print() const
{
    cout << "Outputs:" << endl;
    cout << outputs << endl;
    
    cout << "Activations derivatives:" << endl;
    cout << activations_derivatives << endl;
}


PerceptronLayerBackPropagation::PerceptronLayerBackPropagation()
    : LayerBackPropagation()
{

}


PerceptronLayerBackPropagation::PerceptronLayerBackPropagation(const Index &new_batch_samples_number, Layer *new_layer)
    : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer);
}


PerceptronLayerBackPropagation::~PerceptronLayerBackPropagation()
{

}


void PerceptronLayerBackPropagation::set(const Index &new_batch_samples_number,
                                         Layer *new_layer)
{
    layer = new_layer;
    
    batch_samples_number = new_batch_samples_number;
    
    const Index neurons_number = layer->get_neurons_number();
    const Index inputs_number = layer->get_inputs_number();

    combinations_derivatives.resize(batch_samples_number, neurons_number);
    combinations_derivatives.setZero();

    biases_derivatives.resize(neurons_number);
    biases_derivatives.setZero();

    synaptic_weights_derivatives.resize(inputs_number, neurons_number);
    synaptic_weights_derivatives.setZero();

    input_derivatives.resize(batch_samples_number, inputs_number);
}


vector<pair<type*, dimensions>> PerceptronLayerBackPropagation::get_input_derivative_pairs() const
{
    const Index inputs_number = layer->get_inputs_number();

    return { {(type*)(input_derivatives.data()), {batch_samples_number, inputs_number}} };
}


void PerceptronLayerBackPropagation::print() const
{
    cout << "Error combinations derivatives:" << endl;
    cout << combinations_derivatives << endl;

    cout << "Biases derivatives:" << endl;
    cout << biases_derivatives << endl;
    
    cout << "Synaptic weights derivatives:" << endl;
    cout << synaptic_weights_derivatives << endl;
}


PerceptronLayerBackPropagationLM::PerceptronLayerBackPropagationLM()
    : LayerBackPropagationLM()
{

}


PerceptronLayerBackPropagationLM::PerceptronLayerBackPropagationLM(const Index &new_batch_samples_number,
                                                                   Layer *new_layer)
    : LayerBackPropagationLM()
{
    set(new_batch_samples_number, new_layer);
}


PerceptronLayerBackPropagationLM::~PerceptronLayerBackPropagationLM()
{

}


void PerceptronLayerBackPropagationLM::set(const Index &new_batch_samples_number, Layer *new_layer)
{
    layer = new_layer;

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = layer->get_neurons_number();
    const Index inputs_number = layer->get_inputs_number();
    const Index parameters_number = layer->get_parameters_number();

    squared_errors_Jacobian.resize(batch_samples_number, parameters_number);

    combinations_derivatives.resize(batch_samples_number, neurons_number);

    input_derivatives.resize(batch_samples_number, inputs_number);
}


vector<pair<type*, dimensions>> PerceptronLayerBackPropagationLM::get_input_derivative_pairs() const
{
    const Index inputs_number = layer->get_inputs_number();

    return {{(type*)(input_derivatives.data()), {batch_samples_number, inputs_number}}};
}


void PerceptronLayerBackPropagationLM::print() const
{
    cout << "Squared errors Jacobian: " << endl;
    cout << squared_errors_Jacobian << endl;
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
