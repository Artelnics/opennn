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

namespace opennn
{

PerceptronLayer::PerceptronLayer(const dimensions& new_input_dimensions,
                                 const dimensions& new_output_dimensions,
                                 const ActivationFunction& new_activation_function,
                                 const string& new_layer_name) : Layer()
{
    set(new_input_dimensions,
        new_output_dimensions,
        new_activation_function,
        new_layer_name);
}


dimensions PerceptronLayer::get_input_dimensions() const
{
    return { synaptic_weights.dimension(0) };
}


dimensions PerceptronLayer::get_output_dimensions() const
{
    return { biases.size() };
}


void PerceptronLayer::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


Index PerceptronLayer::get_parameters_number() const
{
    return biases.size() + synaptic_weights.size();
}


type PerceptronLayer::get_dropout_rate() const
{
    return dropout_rate;
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


string PerceptronLayer::get_activation_function_string() const
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


void PerceptronLayer::set(const dimensions& new_input_dimensions,
                          const dimensions& new_output_dimensions,
                          const PerceptronLayer::ActivationFunction& new_activation_function,
                          const string& new_name)
{

    if (new_input_dimensions.size() != 1) 
        throw runtime_error("Input dimensions size is not 1");

    if (new_output_dimensions.size() != 1)
        throw runtime_error("Output dimensions size is not 1");   

    biases.resize(new_output_dimensions[0]);    
    synaptic_weights.resize(new_input_dimensions[0], new_output_dimensions[0]);

    set_parameters_random();

    set_activation_function(new_activation_function);

    set_name(new_name);
    
    layer_type = Layer::Type::Perceptron;
}


void PerceptronLayer::set_input_dimensions(const dimensions& new_input_dimensions)
{
    const Index inputs_number = new_input_dimensions[0];
    const Index outputs_number = get_outputs_number();

    biases.resize(outputs_number);

    synaptic_weights.resize(inputs_number, outputs_number);
}


void PerceptronLayer::set_output_dimensions(const dimensions& new_output_dimensions)
{
    const Index inputs_number = get_inputs_number();
    const Index neurons_number = new_output_dimensions[0];

    biases.resize(neurons_number);

    synaptic_weights.resize(inputs_number, neurons_number);
}


void PerceptronLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{   
    const type* new_parameters_data = new_parameters.data();
    type* synaptic_weights_data = synaptic_weights.data();
    type* biases_data = biases.data();

    const Index biases_number = biases.size();
    const Index synaptic_weights_number = synaptic_weights.size();

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
        activation_function = ActivationFunction::Logistic;
    else if(new_activation_function_name == "HyperbolicTangent")
        activation_function = ActivationFunction::HyperbolicTangent;
    else if(new_activation_function_name == "Linear")
        activation_function = ActivationFunction::Linear;
    else if(new_activation_function_name == "RectifiedLinear")
        activation_function = ActivationFunction::RectifiedLinear;
    else if(new_activation_function_name == "ScaledExponentialLinear")
        activation_function = ActivationFunction::ScaledExponentialLinear;
    else if(new_activation_function_name == "SoftPlus")
        activation_function = ActivationFunction::SoftPlus;
    else if(new_activation_function_name == "SoftSign")
        activation_function = ActivationFunction::SoftSign;
    else if(new_activation_function_name == "HardSigmoid")
        activation_function = ActivationFunction::HardSigmoid;
    else if(new_activation_function_name == "ExponentialLinear")
        activation_function = ActivationFunction::ExponentialLinear;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function_name + ".\n");
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

    sum_columns(thread_pool_device.get(), biases, combinations);

}


void PerceptronLayer::dropout(Tensor<type, 2>& outputs) const
{  
    const Index outputs_number = outputs.dimension(1);

    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    for(Index neuron_index = 0; neuron_index < outputs_number; neuron_index++)
    {
        TensorMap<Tensor<type, 1>> column = tensor_map(outputs, neuron_index);

        get_random_type(type(0), type(1)) < dropout_rate ? column.setZero()
                              : column = column*scaling_factor;
    }
}


void PerceptronLayer::calculate_activations(Tensor<type, 2>& activations,
                                            Tensor<type, 2>& activation_derivatives) const
{
    switch(activation_function)
    {
    case ActivationFunction::Linear: linear(activations, activation_derivatives); return;

    case ActivationFunction::Logistic: logistic(activations, activation_derivatives);return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(activations, activation_derivatives); return;

    case ActivationFunction::RectifiedLinear: rectified_linear(activations, activation_derivatives); return;

    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(activations, activation_derivatives); return;

    case ActivationFunction::SoftPlus: soft_plus(activations, activation_derivatives);return;

    case ActivationFunction::SoftSign: soft_sign(activations, activation_derivatives); return;

    case ActivationFunction::HardSigmoid: hard_sigmoid(activations, activation_derivatives); return;

    case ActivationFunction::ExponentialLinear: exponential_linear(activations, activation_derivatives); return;

    default: return;
    }
}


void PerceptronLayer::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                        unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                        const bool& is_training)
{

    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);

    if(!is_training)
        cerr << "Inputs:\n" << inputs << endl;

    PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation =
        static_cast<PerceptronLayerForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 2>& outputs = perceptron_layer_forward_propagation->outputs;

    calculate_combinations(inputs,
                           outputs);

    // @todo
    // if(is_training && dropout_rate > type(0))
    //     dropout(outputs);

    if(is_training)
    {
        Tensor<type, 2>& activation_derivatives = perceptron_layer_forward_propagation->activation_derivatives;

        calculate_activations(outputs, activation_derivatives);
    }
    else
    {
        Tensor<type, 2> empty;

        calculate_activations(outputs, empty);
    }

    if(!is_training)
        cout << "Outputs:\n" << outputs << endl;

}


void PerceptronLayer::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                     const vector<pair<type*, dimensions>>& delta_pairs,
                                     unique_ptr<LayerForwardPropagation>& forward_propagation,
                                     unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);
    const TensorMap<Tensor<type, 2>> deltas = tensor_map_2(delta_pairs[0]);

    // Forward propagation

    const PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation =
        static_cast<PerceptronLayerForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 2>& activation_derivatives = perceptron_layer_forward_propagation->activation_derivatives;

    // Back propagation

    PerceptronLayerBackPropagation* perceptron_layer_back_propagation =
        static_cast<PerceptronLayerBackPropagation*>(back_propagation.get());
    
    Tensor<type, 2>& combination_derivatives = perceptron_layer_back_propagation->combination_derivatives;

    Tensor<type, 2>& synaptic_weight_derivatives = perceptron_layer_back_propagation->synaptic_weight_derivatives;

    Tensor<type, 1>& bias_derivatives = perceptron_layer_back_propagation->bias_derivatives;

    const bool& is_first_layer = perceptron_layer_back_propagation->is_first_layer;

    Tensor<type, 2>& input_derivatives = perceptron_layer_back_propagation->input_derivatives;
    
    combination_derivatives.device(*thread_pool_device) = deltas * activation_derivatives;

    bias_derivatives.device(*thread_pool_device) = combination_derivatives.sum(sum_dimensions_1);

    synaptic_weight_derivatives.device(*thread_pool_device) = inputs.contract(combination_derivatives, AT_B);

    if(!is_first_layer)
        input_derivatives.device(*thread_pool_device) = combination_derivatives.contract(synaptic_weights, A_BT);
}


void PerceptronLayer::back_propagate_lm(const vector<pair<type*, dimensions>>& input_pairs,
                                        const vector<pair<type*, dimensions>>& delta_pairs,
                                        unique_ptr<LayerForwardPropagation>& forward_propagation,
                                        unique_ptr<LayerBackPropagationLM>& back_propagation) const
{
    const TensorMap<Tensor<type, 2>> inputs = tensor_map_2(input_pairs[0]);
    const TensorMap<Tensor<type, 2>> deltas = tensor_map_2(delta_pairs[0]);

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    const Index synaptic_weights_number = synaptic_weights.size();

    // Forward propagation

    const PerceptronLayerForwardPropagation* perceptron_layer_forward_propagation =
        static_cast<PerceptronLayerForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 2>& activation_derivatives
        = perceptron_layer_forward_propagation->activation_derivatives;

    // Back propagation

    PerceptronLayerBackPropagationLM* perceptron_layer_back_propagation_lm =
        static_cast<PerceptronLayerBackPropagationLM*>(back_propagation.get());

    Tensor<type, 2>& combination_derivatives = perceptron_layer_back_propagation_lm->combination_derivatives;

    Tensor<type, 2>& squared_errors_Jacobian = perceptron_layer_back_propagation_lm->squared_errors_Jacobian;

    const bool& is_first_layer = perceptron_layer_back_propagation_lm->is_first_layer;

    Tensor<type, 2>& input_derivatives = perceptron_layer_back_propagation_lm->input_derivatives;
    
    combination_derivatives.device(*thread_pool_device) = deltas * activation_derivatives;

    Index synaptic_weight_index = 0;

    for(Index neuron_index = 0; neuron_index < outputs_number; neuron_index++)
    {
        const TensorMap<Tensor<type, 1>> combinations_derivatives_neuron 
            = tensor_map(combination_derivatives, neuron_index);

        for(Index input_index = 0; input_index < inputs_number; input_index++)
        {
            const TensorMap<Tensor<type, 1>> input = tensor_map(inputs, input_index);

            TensorMap<Tensor<type, 1>> squared_errors_jacobian_synaptic_weight 
                = tensor_map(squared_errors_Jacobian, synaptic_weight_index++);

            squared_errors_jacobian_synaptic_weight.device(*thread_pool_device) 
                = combinations_derivatives_neuron * input;
        }

        const Index bias_index = synaptic_weights_number + neuron_index;

        TensorMap<Tensor<type, 1>> squared_errors_jacobian_bias 
            = tensor_map(squared_errors_Jacobian, bias_index);

        squared_errors_jacobian_bias.device(*thread_pool_device) = combinations_derivatives_neuron;
    }

    if(!is_first_layer)
        input_derivatives.device(*thread_pool_device) 
        = combination_derivatives.contract(synaptic_weights, A_BT);
}


void PerceptronLayer::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                      const Index& index,
                                      Tensor<type, 1>& gradient) const
{
    const Index biases_number = biases.size();
    const Index synaptic_weights_number = synaptic_weights.size();

    PerceptronLayerBackPropagation* perceptron_layer_back_propagation =
        static_cast<PerceptronLayerBackPropagation*>(back_propagation.get());

    const Tensor<type, 2>& synaptic_weight_derivatives = perceptron_layer_back_propagation->synaptic_weight_derivatives;
    const Tensor<type, 1>& bias_derivatives = perceptron_layer_back_propagation->bias_derivatives;

    const type* synaptic_weights_derivatives_data = synaptic_weight_derivatives.data();
    const type* biases_derivatives_data = bias_derivatives.data();
    type* gradient_data = gradient.data();

    #pragma omp parallel sections
    {
        #pragma omp section
        memcpy(gradient_data + index, synaptic_weights_derivatives_data, synaptic_weights_number * sizeof(type));

        #pragma omp section
        memcpy(gradient_data + index + synaptic_weights_number, biases_derivatives_data, biases_number * sizeof(type));
    }
}


void PerceptronLayer::insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>& back_propagation,
                                                        const Index& index,
                                                        Tensor<type, 2>& squared_errors_Jacobian) const
{
    const Index parameters_number = get_parameters_number();
    const Index batch_samples_number = back_propagation->batch_samples_number;

    PerceptronLayerBackPropagationLM* perceptron_layer_back_propagation_lm =
        static_cast<PerceptronLayerBackPropagationLM*>(back_propagation.get());

    type* this_squared_errors_Jacobian_data = perceptron_layer_back_propagation_lm->squared_errors_Jacobian.data();

    memcpy(squared_errors_Jacobian.data() + index,
           this_squared_errors_Jacobian_data,
           parameters_number * batch_samples_number * sizeof(type));
}


string PerceptronLayer::get_expression(const vector<string>& new_input_names,
                                       const vector<string>& new_output_names) const
{
    const vector<string> input_names = new_input_names.empty()
       ? get_default_input_names()
       : new_input_names;

    const vector<string> output_names = new_output_names.empty()
        ? get_default_output_names()
        : new_output_names;

    const Index inputs_number = get_inputs_number();
    const Index outputs_number = get_outputs_number();

    ostringstream buffer;

    for(Index j = 0; j < outputs_number; j++)
    {
        const TensorMap<Tensor<type, 1>> synaptic_weights_column = tensor_map(synaptic_weights, j);

        buffer << output_names[j] << " = " << get_activation_function_string_expression() << "(" << biases(j) << "+";

        for(Index i = 0; i < inputs_number - 1; i++)
            buffer << synaptic_weights_column(i) << "*" << input_names[i] << "+";

        buffer << synaptic_weights_column(inputs_number - 1) << "*" << input_names[inputs_number - 1]  << ");\n";
    }

    return buffer.str();
}


void PerceptronLayer::print() const
{
    cout << "Perceptron layer" << endl
         << "Input dimensions: " << get_input_dimensions()[0] << endl
         << "Output dimensions: " << get_output_dimensions()[0] << endl
         << "Biases dimensions: " << biases.dimensions() << endl
         << "Synaptic weights dimensions: " << synaptic_weights.dimensions() << endl;

    cout << "Biases:" << endl;
    cout << biases << endl;
    cout << "Synaptic weights:" << endl;
    cout << synaptic_weights << endl;

    cout << "Activation function:" << endl;
    cout << get_activation_function_string() << endl;
}


void PerceptronLayer::from_XML(const XMLDocument& document)
{
    const XMLElement* perceptron_layer_element = document.FirstChildElement("Perceptron");

    if(!perceptron_layer_element)
        throw runtime_error("PerceptronLayer element is nullptr.\n");

    set_name(read_xml_string(perceptron_layer_element, "Name"));
    set_input_dimensions({ read_xml_index(perceptron_layer_element, "InputsNumber") });
    set_output_dimensions({ read_xml_index(perceptron_layer_element, "NeuronsNumber") });
    set_activation_function(read_xml_string(perceptron_layer_element, "ActivationFunction"));
    set_parameters(to_type_vector(read_xml_string(perceptron_layer_element, "Parameters"), " "), 0);
}


void PerceptronLayer::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Perceptron");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "InputsNumber", to_string(get_input_dimensions()[0]));
    add_xml_element(printer, "NeuronsNumber", to_string(get_output_dimensions()[0]));
    add_xml_element(printer, "ActivationFunction", get_activation_function_string());
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();  
}


string PerceptronLayer::get_activation_function_string_expression() const
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

    if (!layer) return;

    const Index outputs_number = layer->get_outputs_number();

    outputs.resize(batch_samples_number, outputs_number);

    activation_derivatives.resize(batch_samples_number, outputs_number);

    activation_derivatives.setConstant((type)NAN);
}


pair<type *, dimensions> PerceptronLayerForwardPropagation::get_outputs_pair() const
{
    const dimensions output_dimensions = layer->get_output_dimensions();
    
    return pair<type *, dimensions>((type*)outputs.data(), {{batch_samples_number, output_dimensions[0]}});
}


PerceptronLayerForwardPropagation::PerceptronLayerForwardPropagation(const Index &new_batch_samples_number,
                                                                     Layer *new_layer)
: LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


void PerceptronLayerForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl
         << "Activation derivatives:" << endl
         << activation_derivatives << endl;
}


PerceptronLayerBackPropagation::PerceptronLayerBackPropagation(const Index &new_batch_samples_number, Layer *new_layer)
    : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer);
}


void PerceptronLayerBackPropagation::set(const Index &new_batch_samples_number,
                                         Layer *new_layer)
{
    layer = new_layer;
    
    batch_samples_number = new_batch_samples_number;
    
    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_dimensions()[0];

    combination_derivatives.resize(batch_samples_number, outputs_number);
    combination_derivatives.setZero();

    bias_derivatives.resize(outputs_number);
    bias_derivatives.setZero();

    synaptic_weight_derivatives.resize(inputs_number, outputs_number);
    synaptic_weight_derivatives.setZero();

    input_derivatives.resize(batch_samples_number, inputs_number);
}


vector<pair<type*, dimensions>> PerceptronLayerBackPropagation::get_input_derivative_pairs() const
{
    const Index inputs_number = layer->get_input_dimensions()[0];

    return { {(type*)(input_derivatives.data()), {batch_samples_number, inputs_number}} };
}


void PerceptronLayerBackPropagation::print() const
{
    cout << "Error combinations derivatives:" << endl
         << combination_derivatives << endl
         << "Biases derivatives:" << endl
         << bias_derivatives << endl
         << "Synaptic weights derivatives:" << endl
         << synaptic_weight_derivatives << endl;
}


PerceptronLayerBackPropagationLM::PerceptronLayerBackPropagationLM(const Index &new_batch_samples_number,
                                                                   Layer *new_layer)
    : LayerBackPropagationLM()
{
    set(new_batch_samples_number, new_layer);
}


void PerceptronLayerBackPropagationLM::set(const Index &new_batch_samples_number, Layer *new_layer)
{
    layer = new_layer;

    batch_samples_number = new_batch_samples_number;

    const Index outputs_number = layer->get_outputs_number();
    const Index inputs_number = layer->get_input_dimensions()[0];
    const Index parameters_number = layer->get_parameters_number();

    combination_derivatives.resize(batch_samples_number, outputs_number);

    squared_errors_Jacobian.resize(batch_samples_number, parameters_number);

    input_derivatives.resize(batch_samples_number, inputs_number);
}


vector<pair<type*, dimensions>> PerceptronLayerBackPropagationLM::get_input_derivative_pairs() const
{
    const Index inputs_number = layer->get_input_dimensions()[0];

    return {{(type*)(input_derivatives.data()), {batch_samples_number, inputs_number}}};
}


void PerceptronLayerBackPropagationLM::print() const
{
    cout << "Combination derivatives: " << endl
        << combination_derivatives << endl;
    cout << "Squared errors Jacobian: " << endl
        << squared_errors_Jacobian << endl;
    cout << "Input derivatives: " << endl
        << input_derivatives << endl;
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
