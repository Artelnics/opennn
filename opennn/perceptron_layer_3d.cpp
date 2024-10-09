//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>

#include "strings_utilities.h"
#include "perceptron_layer_3d.h"
#include "tensors.h"

namespace opennn
{

PerceptronLayer3D::PerceptronLayer3D() : Layer()
{
    set();

    layer_type = Type::PerceptronLayer3D;
}


PerceptronLayer3D::PerceptronLayer3D(const Index& new_inputs_number,
                                     const Index& new_inputs_depth,
                                     const Index& new_neurons_number,
                                     const PerceptronLayer3D::ActivationFunction& new_activation_function) : Layer()
{
    set(new_inputs_number, new_inputs_depth, new_neurons_number, new_activation_function);

    layer_type = Type::PerceptronLayer3D;

    name = "perceptron_layer_3d";
}


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


Index PerceptronLayer3D::get_neurons_number() const
{
    return biases.size();
}


dimensions PerceptronLayer3D::get_output_dimensions() const
{
    return { inputs_number, get_neurons_number() };
}


Index PerceptronLayer3D::get_biases_number() const
{
    return biases.size();
}


Index PerceptronLayer3D::get_synaptic_weights_number() const
{
    return synaptic_weights.size();
}


Index PerceptronLayer3D::get_parameters_number() const
{
    return biases.size() + synaptic_weights.size();
}


type PerceptronLayer3D::get_dropout_rate() const
{
    return dropout_rate;
}


const Tensor<type, 1>& PerceptronLayer3D::get_biases() const
{
    return biases;
}


const Tensor<type, 2>& PerceptronLayer3D::get_synaptic_weights() const
{
    return synaptic_weights;
}


Tensor<type, 1> PerceptronLayer3D::get_parameters() const
{
    Tensor<type, 1> parameters(synaptic_weights.size() + biases.size());

    memcpy(parameters.data(), synaptic_weights.data(), synaptic_weights.size()*sizeof(type));

    memcpy(parameters.data() + synaptic_weights.size(), biases.data(), biases.size()*sizeof(type));

    return parameters;
}


const PerceptronLayer3D::ActivationFunction& PerceptronLayer3D::get_activation_function() const
{
    return activation_function;
}


string PerceptronLayer3D::write_activation_function() const
{
    switch(activation_function)
    {
    case ActivationFunction::HyperbolicTangent:
        return "HyperbolicTangent";

    case ActivationFunction::Linear:
        return "Linear";

    case ActivationFunction::RectifiedLinear:
        return "RectifiedLinear";
    /*
    case ActivationFunction::Logistic:
        return "Logistic";

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
*/
    }

    return string();
}


const bool& PerceptronLayer3D::get_display() const
{
    return display;
}


void PerceptronLayer3D::set()
{
    biases.resize(0);

    synaptic_weights.resize(0, 0);

    set_default();
}


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


void PerceptronLayer3D::set_default()
{
    name = "perceptron_layer_3d";

    display = true;

    layer_type = Type::PerceptronLayer3D;

    dropout_rate = 0;
}


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


void PerceptronLayer3D::set_neurons_number(const Index& new_neurons_number)
{
    const Index inputs_depth = get_inputs_depth();

    biases.resize(new_neurons_number);

    synaptic_weights.resize(inputs_depth, new_neurons_number);
}


void PerceptronLayer3D::set_biases(const Tensor<type, 1>& new_biases)
{
    biases = new_biases;
}


void PerceptronLayer3D::set_synaptic_weights(const Tensor<type, 2>& new_synaptic_weights)
{
    synaptic_weights = new_synaptic_weights;
}


void PerceptronLayer3D::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    #pragma omp parallel sections
    {
        #pragma omp section
        memcpy(synaptic_weights.data(), new_parameters.data() + index, synaptic_weights.size()*sizeof(type));

        #pragma omp section
        memcpy(biases.data(), new_parameters.data() + index + synaptic_weights.size(), biases.size()*sizeof(type));
    }
}


void PerceptronLayer3D::set_activation_function(const PerceptronLayer3D::ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


void PerceptronLayer3D::set_activation_function(const string& new_activation_function_name)
{
    if (new_activation_function_name == "Logistic")
        ;//activation_function = ActivationFunction::Logistic;
    else if (new_activation_function_name == "HyperbolicTangent")
        activation_function = ActivationFunction::HyperbolicTangent;
    else if (new_activation_function_name == "Linear")
        activation_function = ActivationFunction::Linear;
    else if (new_activation_function_name == "RectifiedLinear")
        activation_function = ActivationFunction::RectifiedLinear;
    else if (new_activation_function_name == "ScaledExponentialLinear")
        ;//activation_function = ActivationFunction::ScaledExponentialLinear;
    else if (new_activation_function_name == "SoftPlus")
        ;//activation_function = ActivationFunction::SoftPlus;
    else if (new_activation_function_name == "SoftSign")
        ;//activation_function = ActivationFunction::SoftSign;
    else if (new_activation_function_name == "HardSigmoid")
        ;//activation_function = ActivationFunction::HardSigmoid;
    else if (new_activation_function_name == "ExponentialLinear")
        ;//activation_function = ActivationFunction::ExponentialLinear;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function_name + ".\n");
}


void PerceptronLayer3D::set_display(const bool& new_display)
{
    display = new_display;
}


void PerceptronLayer3D::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    synaptic_weights.setConstant(value);
}


void PerceptronLayer3D::set_parameters_random()
{
    set_random(biases);

    set_random(synaptic_weights);
}


void PerceptronLayer3D::set_parameters_glorot()
{
    biases.setZero();

    const type limit = sqrt(6 / type(get_inputs_depth() + get_neurons_number()));

    const type minimum = -limit;
    const type maximum = limit;

    #pragma omp parallel for

    for(Index i = 0; i < synaptic_weights.size(); i++)
        synaptic_weights(i) = minimum + (maximum - minimum)*type(rand() / (RAND_MAX + 1.0));
}


void PerceptronLayer3D::calculate_combinations(const Tensor<type, 3>& inputs,
                                               Tensor<type, 3>& combinations) const
{
    const Eigen::array<IndexPair<Index>, 1> contraction_indices = {IndexPair<Index>(2, 0)};

    combinations.device(*thread_pool_device) = inputs.contract(synaptic_weights, contraction_indices);

    sum_matrices(thread_pool_device, biases, combinations);
}


void PerceptronLayer3D::dropout(Tensor<type, 3>& outputs) const 
{
    /*
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

        if(random < dropout_rate)
            matrix.setZero();
        else
            matrix.device(*thread_pool_device) = matrix * scaling_factor;
    }
    */
    const type scaling_factor = type(1) / (type(1) - dropout_rate);

    #pragma omp parallel for

    for(Index i = 0; i < outputs.size(); i++)
        outputs(i) = (calculate_random_uniform(type(0), type(1)) < dropout_rate) 
            ? 0 
            : outputs(i) * scaling_factor;
}


void PerceptronLayer3D::calculate_activations(Tensor<type, 3>& activations, Tensor<type, 3>& activations_derivatives) const
{
    switch(activation_function)
    {
    case ActivationFunction::Linear: linear(activations, activations_derivatives); return;

//    case ActivationFunction::Logistic: logistic(activations, activations_derivatives); return;

    case ActivationFunction::HyperbolicTangent: hyperbolic_tangent(activations, activations_derivatives); return;

    case ActivationFunction::RectifiedLinear: rectified_linear(activations, activations_derivatives); return;

//    case ActivationFunction::ScaledExponentialLinear: scaled_exponential_linear(activations, activations_derivatives); return;

//    case ActivationFunction::SoftPlus: soft_plus(activations, activations_derivatives); return;

//    case ActivationFunction::SoftSign: soft_sign(activations, activations_derivatives); return;

//    case ActivationFunction::HardSigmoid: hard_sigmoid(activations, activations_derivatives); return;

//    case ActivationFunction::ExponentialLinear: exponential_linear(activations, activations_derivatives); return;

    default: return;
    }
}


void PerceptronLayer3D::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                          LayerForwardPropagation* layer_forward_propagation,
                                          const bool& is_training)
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    PerceptronLayer3DForwardPropagation* perceptron_layer_3d_forward_propagation =
        static_cast<PerceptronLayer3DForwardPropagation*>(layer_forward_propagation);

    Tensor<type, 3>& outputs = perceptron_layer_3d_forward_propagation->outputs;

    calculate_combinations(inputs,
                           outputs);

    if(is_training)
    {
        if(dropout_rate > type(0))
            dropout(outputs);

        Tensor<type, 3>& activations_derivatives = perceptron_layer_3d_forward_propagation->activations_derivatives;

        calculate_activations(outputs, activations_derivatives);
    }
    else
    {
        calculate_activations(outputs, empty);
    }
}


void PerceptronLayer3D::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                       const vector<pair<type*, dimensions>>& deltas_pair,
                                       LayerForwardPropagation* forward_propagation,
                                       LayerBackPropagation* back_propagation) const
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    if(deltas_pair.size() > 1)     
        add_deltas(deltas_pair);

    const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(deltas_pair[0]);

    // Forward propagation

    const PerceptronLayer3DForwardPropagation* perceptron_layer_3d_forward_propagation =
            static_cast<PerceptronLayer3DForwardPropagation*>(forward_propagation);

    const Tensor<type, 3>& activations_derivatives = perceptron_layer_3d_forward_propagation->activations_derivatives;

    // Back propagation

    PerceptronLayer3DBackPropagation* perceptron_layer_3d_back_propagation =
            static_cast<PerceptronLayer3DBackPropagation*>(back_propagation);

    Tensor<type, 3>& combinations_derivatives = perceptron_layer_3d_back_propagation->combinations_derivatives;

    Tensor<type, 1>& biases_derivatives = perceptron_layer_3d_back_propagation->biases_derivatives;
    Tensor<type, 2>& synaptic_weights_derivatives = perceptron_layer_3d_back_propagation->synaptic_weights_derivatives;

    Tensor<type, 3>& input_derivatives = perceptron_layer_3d_back_propagation->input_derivatives;

    const Eigen::array<IndexPair<Index>, 2> double_contraction_indices = { IndexPair<Index>(0, 0), IndexPair<Index>(1, 1) };
    const Eigen::array<IndexPair<Index>, 1> single_contraction_indices = { IndexPair<Index>(2, 1) };

    combinations_derivatives.device(*thread_pool_device) 
        = deltas * activations_derivatives;

    biases_derivatives.device(*thread_pool_device)
        = combinations_derivatives.sum(Eigen::array<Index, 2>({0, 1}));

    synaptic_weights_derivatives.device(*thread_pool_device)
        = inputs.contract(combinations_derivatives, double_contraction_indices);

    input_derivatives.device(*thread_pool_device) 
        = combinations_derivatives.contract(synaptic_weights, single_contraction_indices);
}


void PerceptronLayer3D::add_deltas(const vector<pair<type*, dimensions>>& deltas_pair) const
{
    TensorMap<Tensor<type, 3>> deltas = tensor_map_3(deltas_pair[0]);

    for(Index i = 1; i < static_cast<Index>(deltas_pair.size()); i++)
    {
        const TensorMap<Tensor<type, 3>> other_deltas = tensor_map_3(deltas_pair[i]);

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

    #pragma omp parallel sections
    {
        #pragma omp section
        memcpy(gradient_data + index, synaptic_weights_derivatives_data, synaptic_weights_number * sizeof(type));

        #pragma omp section
        memcpy(gradient_data + index + synaptic_weights_number, biases_derivatives_data, biases_number * sizeof(type));
    }
}


void PerceptronLayer3D::from_XML(const tinyxml2::XMLDocument& document)
{
    // Perceptron layer

    const tinyxml2::XMLElement* perceptron_layer_element = document.FirstChildElement("PerceptronLayer3D");

    if(!perceptron_layer_element)
        throw runtime_error("PerceptronLayer3D element is nullptr.\n");

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

    // Inputs depth

    const tinyxml2::XMLElement* inputs_depth_element = perceptron_layer_element->FirstChildElement("InputsDepth");

    if(!inputs_depth_element)
        throw runtime_error("InputsDepth element is nullptr.\n");

    if(inputs_depth_element->GetText())
        set_inputs_depth(Index(stoi(inputs_depth_element->GetText())));

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


void PerceptronLayer3D::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Perceptron layer

    file_stream.OpenElement("PerceptronLayer3D");

    // Layer name

    file_stream.OpenElement("Name");
    file_stream.PushText(name.c_str());
    file_stream.CloseElement();

    // Inputs number

    file_stream.OpenElement("InputsNumber");
    file_stream.PushText(to_string(get_inputs_number()).c_str());
    file_stream.CloseElement();

    // Inputs depth

    file_stream.OpenElement("InputsDepth");
    file_stream.PushText(to_string(get_inputs_depth()).c_str());
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


void PerceptronLayer3DForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl
         << "Activations derivatives:" << endl
         << activations_derivatives << endl;
}


pair<type*, dimensions> PerceptronLayer3DForwardPropagation::get_outputs_pair() const
{
    PerceptronLayer3D* perceptron_layer_3d = static_cast<PerceptronLayer3D*>(layer);

    const Index neurons_number = perceptron_layer_3d->get_neurons_number();

    const Index inputs_number = perceptron_layer_3d->get_inputs_number();

    return { outputs_data, { batch_samples_number, inputs_number, neurons_number } };
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

    combinations_derivatives.resize(batch_samples_number, inputs_number, neurons_number);

    input_derivatives.resize(batch_samples_number, inputs_number, inputs_depth);
}


vector<pair<type*, dimensions>> PerceptronLayer3DBackPropagation::get_input_derivative_pairs() const
{
    PerceptronLayer3D* perceptron_layer_3d = static_cast<PerceptronLayer3D*>(layer);

    const Index inputs_number = perceptron_layer_3d->get_inputs_number();
    const Index inputs_depth = perceptron_layer_3d->get_inputs_depth();

    return {{(type*)(input_derivatives.data()),
            {batch_samples_number, inputs_number, inputs_depth}}};
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
