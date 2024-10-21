//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P R O B A B I L I S T I C   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "strings_utilities.h"
#include "probabilistic_layer_3d.h"

namespace opennn
{

ProbabilisticLayer3D::ProbabilisticLayer3D()
{
    set();
}


ProbabilisticLayer3D::ProbabilisticLayer3D(const Index& new_inputs_number, const Index& new_inputs_depth, const Index& new_neurons_number)
{
    set(new_inputs_number, new_inputs_depth, new_neurons_number);
}


Index ProbabilisticLayer3D::get_inputs_number() const
{
    return inputs_number;
}


Index ProbabilisticLayer3D::get_inputs_depth() const
{
    return synaptic_weights.dimension(0);
}


Index ProbabilisticLayer3D::get_neurons_number() const
{
    return biases.size();
}


dimensions ProbabilisticLayer3D::get_output_dimensions() const
{
    return { inputs_number, get_neurons_number() };
}


Index ProbabilisticLayer3D::get_biases_number() const
{
    return biases.size();
}


Index ProbabilisticLayer3D::get_synaptic_weights_number() const
{
    return synaptic_weights.size();
}


const type& ProbabilisticLayer3D::get_decision_threshold() const
{
    return decision_threshold;
}


const ProbabilisticLayer3D::ActivationFunction& ProbabilisticLayer3D::get_activation_function() const
{
    return activation_function;
}


string ProbabilisticLayer3D::write_activation_function() const
{
    switch (activation_function)
    {
    case ActivationFunction::Competitive:
        return "Competitive";
    case ActivationFunction::Softmax:
        return "Softmax";
    default:
        throw runtime_error("Unknown probabilistic method.\n");
    }
}


string ProbabilisticLayer3D::write_activation_function_text() const
{
    switch (activation_function)
    {
    case ActivationFunction::Competitive:
        return "competitive";
    case ActivationFunction::Softmax:
        return "softmax";
    default:
        throw runtime_error("Unknown probabilistic method.\n");
    }
}


const bool& ProbabilisticLayer3D::get_display() const
{
    return display;
}


Index ProbabilisticLayer3D::get_parameters_number() const
{
    return biases.size() + synaptic_weights.size();
}


Tensor<type, 1> ProbabilisticLayer3D::get_parameters() const
{
    Tensor<type, 1> parameters(synaptic_weights.size() + biases.size());

    memcpy(parameters.data(), synaptic_weights.data(), synaptic_weights.size()*sizeof(type));

    memcpy(parameters.data() + synaptic_weights.size(), biases.data(), biases.size()*sizeof(type));

    return parameters;
}


void ProbabilisticLayer3D::set()
{
    inputs_number = 0;
    
    biases.resize(0);

    synaptic_weights.resize(0,0);

    set_default();
}


void ProbabilisticLayer3D::set(const Index& new_inputs_number, const Index& new_inputs_depth, const Index& new_neurons_number)
{
    inputs_number = new_inputs_number;

    biases.resize(new_neurons_number);

    synaptic_weights.resize(new_inputs_depth, new_neurons_number);

    set_parameters_glorot();

    set_default();
}


void ProbabilisticLayer3D::set(const ProbabilisticLayer3D& other_probabilistic_layer)
{
    set_default();

    activation_function = other_probabilistic_layer.activation_function;

    decision_threshold = other_probabilistic_layer.decision_threshold;

    display = other_probabilistic_layer.display;
}


void ProbabilisticLayer3D::set_inputs_number(const Index& new_inputs_number)
{
    inputs_number = new_inputs_number;
}


void ProbabilisticLayer3D::set_inputs_depth(const Index& new_inputs_depth)
{
    const Index neurons_number = get_neurons_number();

    biases.resize(neurons_number);

    synaptic_weights.resize(new_inputs_depth, neurons_number);
}


void ProbabilisticLayer3D::set_neurons_number(const Index& new_neurons_number)
{
    const Index inputs_depth = get_inputs_depth();

    biases.resize(new_neurons_number);

    synaptic_weights.resize(inputs_depth, new_neurons_number);
}


void ProbabilisticLayer3D::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    const Index biases_number = biases.size();
    const Index synaptic_weights_number = synaptic_weights.size();

    #pragma omp parallel sections
    {
        #pragma omp section
        memcpy(synaptic_weights.data(), new_parameters.data() + index, synaptic_weights_number*sizeof(type));

        #pragma omp section
        memcpy(biases.data(), new_parameters.data() + index + synaptic_weights_number, biases_number*sizeof(type));
    }
}


void ProbabilisticLayer3D::set_decision_threshold(const type& new_decision_threshold)
{
    decision_threshold = new_decision_threshold;
}


void ProbabilisticLayer3D::set_default()
{
    name = "probabilistic_layer_3d";
    
    layer_type = Layer::Type::Probabilistic3D;
    
    activation_function = ActivationFunction::Softmax;

    decision_threshold = type(0.5);

    display = true;
}


void ProbabilisticLayer3D::set_activation_function(const ActivationFunction& new_activation_function)
{
    activation_function = new_activation_function;
}


void ProbabilisticLayer3D::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "Competitive")
        set_activation_function(ActivationFunction::Competitive);
    else if(new_activation_function == "Softmax")
        set_activation_function(ActivationFunction::Softmax);
    else
        throw runtime_error("Unknown probabilistic method: " + new_activation_function + ".\n");
}


void ProbabilisticLayer3D::set_display(const bool& new_display)
{
    display = new_display;
}


void ProbabilisticLayer3D::set_parameters_constant(const type& value)
{
    biases.setConstant(value);

    synaptic_weights.setConstant(value);
}


void ProbabilisticLayer3D::set_parameters_random()
{
    set_random(biases);

    set_random(synaptic_weights);
}


void ProbabilisticLayer3D::set_parameters_glorot()
{
    biases.setZero();
    
    const type limit = sqrt(6 / type(get_inputs_depth() + get_neurons_number()));

    const type minimum = -limit;
    const type maximum = limit;
    
    #pragma omp parallel for

    for(Index i = 0; i < synaptic_weights.size(); i++)
        synaptic_weights(i) = minimum + (maximum - minimum) * type(rand() / (RAND_MAX + 1.0));
}


void ProbabilisticLayer3D::calculate_combinations(const Tensor<type, 3>& inputs,
                                                  Tensor<type, 3>& combinations) const
{
    combinations.device(*thread_pool_device) = inputs.contract(synaptic_weights, contraction_indices);

    sum_matrices(thread_pool_device, biases, combinations);
}


void ProbabilisticLayer3D::calculate_activations(Tensor<type, 3>& activations) const
{
    switch(activation_function)
    {
    case ActivationFunction::Softmax: softmax(activations); 
        return;

    default: 
        return;
    }
}


void ProbabilisticLayer3D::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                             unique_ptr<LayerForwardPropagation>& forward_propagation,
                                             const bool& is_training)
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    unique_ptr<ProbabilisticLayer3DForwardPropagation> probabilistic_layer_3d_forward_propagation
            (static_cast<ProbabilisticLayer3DForwardPropagation*>(forward_propagation.release()));
    
    Tensor<type, 3>& outputs = probabilistic_layer_3d_forward_propagation->outputs;
    
    calculate_combinations(inputs, outputs);

    if(is_training)
        calculate_activations(outputs);
    //else competitive(outputs, outputs);
}


void ProbabilisticLayer3D::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                          const vector<pair<type*, dimensions>>& delta_pairs,
                                          unique_ptr<LayerForwardPropagation>& forward_propagation,
                                          unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    // Forward propagation

    unique_ptr<ProbabilisticLayer3DForwardPropagation> probabilistic_layer_3d_forward_propagation
            (static_cast<ProbabilisticLayer3DForwardPropagation*>(forward_propagation.release()));

    const Tensor<type, 3>& outputs = probabilistic_layer_3d_forward_propagation->outputs;

    // Back propagation

    unique_ptr<ProbabilisticLayer3DBackPropagation> probabilistic_layer_3d_back_propagation 
            (static_cast<ProbabilisticLayer3DBackPropagation*>(back_propagation.release()));

    const Tensor<type, 2>& targets = probabilistic_layer_3d_back_propagation->targets;
    Tensor<type, 2>& mask = probabilistic_layer_3d_back_propagation->mask;
    bool& built_mask = probabilistic_layer_3d_back_propagation->built_mask;

    Tensor<type, 3>& combinations_derivatives = probabilistic_layer_3d_back_propagation->combinations_derivatives;

    Tensor<type, 1>& biases_derivatives = probabilistic_layer_3d_back_propagation->biases_derivatives;
    Tensor<type, 2>& synaptic_weights_derivatives = probabilistic_layer_3d_back_propagation->synaptic_weights_derivatives;

    Tensor<type, 3>& input_derivatives = probabilistic_layer_3d_back_propagation->input_derivatives;

    const Eigen::array<IndexPair<Index>, 2> double_contraction_indices = { IndexPair<Index>(0, 0), IndexPair<Index>(1, 1) };
    const Eigen::array<IndexPair<Index>, 1> single_contraction_indices = { IndexPair<Index>(2, 1) };

    if(!built_mask)
    {
        mask.device(*thread_pool_device) = (targets != targets.constant(0)).cast<type>();
 
        const Tensor<type, 0> mask_sum = mask.sum();
        
        mask.device(*thread_pool_device) = mask / mask_sum(0);
        
        built_mask = true;
    }

    calculate_combinations_derivatives(outputs, targets, mask, combinations_derivatives);

    biases_derivatives.device(*thread_pool_device) 
        = combinations_derivatives.sum(Eigen::array<Index, 2>({ 0, 1 }));

    synaptic_weights_derivatives.device(*thread_pool_device) 
        = inputs.contract(combinations_derivatives, double_contraction_indices);

    input_derivatives.device(*thread_pool_device) 
        = combinations_derivatives.contract(synaptic_weights, single_contraction_indices);
}


void ProbabilisticLayer3D::calculate_combinations_derivatives(const Tensor<type, 3>& outputs, 
                                                              const Tensor<type, 2>& targets,
                                                              const Tensor<type, 2>& mask,
                                                              Tensor<type, 3>& combinations_derivatives) const
{
    const Index batch_samples_number = outputs.dimension(0);
    const Index outputs_number = outputs.dimension(1);

    // @todo Can we simplify this? For instance put the division in the last line somewhere else. 

    combinations_derivatives.device(*thread_pool_device) = outputs;

    #pragma omp parallel for

    for(Index i = 0; i < batch_samples_number; i++)
        for(Index j = 0; j < outputs_number; j++)
            combinations_derivatives(i, j, Index(targets(i, j)))--;

    multiply_matrices(thread_pool_device, combinations_derivatives, mask);
}


void ProbabilisticLayer3D::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                           const Index& index,
                                           Tensor<type, 1>& gradient) const
{
    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    const unique_ptr<ProbabilisticLayer3DBackPropagation> probabilistic_layer_3d_back_propagation 
        (static_cast<ProbabilisticLayer3DBackPropagation*>(back_propagation.release()));

    const type* synaptic_weights_derivatives_data = probabilistic_layer_3d_back_propagation->synaptic_weights_derivatives.data();
    const type* biases_derivatives_data = probabilistic_layer_3d_back_propagation->biases_derivatives.data();

    type* gradient_data = gradient.data();

    #pragma omp parallel sections
    {
        #pragma omp section
        memcpy(gradient_data + index, synaptic_weights_derivatives_data, synaptic_weights_number * sizeof(type));

        #pragma omp section
        memcpy(gradient_data + index + synaptic_weights_number, biases_derivatives_data, biases_number * sizeof(type));
    }
}


void ProbabilisticLayer3D::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Probabilistic layer

    const tinyxml2::XMLElement* probabilistic_layer_element = document.FirstChildElement("ProbabilisticLayer3D");

    if(!probabilistic_layer_element)
        throw runtime_error("ProbabilisticLayer3D element is nullptr.\n");

    // Layer name

    const tinyxml2::XMLElement* layer_name_element = probabilistic_layer_element->FirstChildElement("Name");

    if(!layer_name_element)
        throw runtime_error("LayerName element is nullptr.\n");

    if(layer_name_element->GetText())
        set_name(layer_name_element->GetText());

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = probabilistic_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
        throw runtime_error("InputsNumber element is nullptr.\n");

    if(inputs_number_element->GetText())
        set_inputs_number(Index(stoi(inputs_number_element->GetText())));

    // Inputs depth

    const tinyxml2::XMLElement* inputs_depth_element = probabilistic_layer_element->FirstChildElement("InputsDepth");

    if(!inputs_depth_element)
        throw runtime_error("InputsDepth element is nullptr.\n");

    if(inputs_depth_element->GetText())
        set_inputs_depth(Index(stoi(inputs_depth_element->GetText())));

    // Neurons number

    const tinyxml2::XMLElement* neurons_number_element = probabilistic_layer_element->FirstChildElement("NeuronsNumber");

    if(!neurons_number_element)
        throw runtime_error("NeuronsNumber element is nullptr.\n");

    if(neurons_number_element->GetText())
        set_neurons_number(Index(stoi(neurons_number_element->GetText())));

    // Decision threshold

    const tinyxml2::XMLElement* decision_threshold_element = probabilistic_layer_element->FirstChildElement("DecisionThreshold");

    if(!decision_threshold_element)
        throw runtime_error("DecisionThreshold element is nullptr.\n");

    if(decision_threshold_element->GetText())
        set_decision_threshold(type(stod(decision_threshold_element->GetText())));

    // Activation function

    const tinyxml2::XMLElement* activation_function_element = probabilistic_layer_element->FirstChildElement("ActivationFunction");

    if(!activation_function_element)
        throw runtime_error("ActivationFunction element is nullptr.\n");

    if(activation_function_element->GetText())
        set_activation_function(activation_function_element->GetText());

    // Parameters

    const tinyxml2::XMLElement* parameters_element = probabilistic_layer_element->FirstChildElement("Parameters");

    if(!parameters_element)
        throw runtime_error("Parameters element is nullptr.\n");

    if(parameters_element->GetText())
        set_parameters(to_type_vector(parameters_element->GetText(), " "));
}


void ProbabilisticLayer3D::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Probabilistic layer

    file_stream.OpenElement("ProbabilisticLayer3D");

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

    // Neurons number

    file_stream.OpenElement("NeuronsNumber");
    file_stream.PushText(to_string(get_neurons_number()).c_str());
    file_stream.CloseElement();

    // Decision threshold

    file_stream.OpenElement("DecisionThreshold");
    file_stream.PushText(to_string(get_decision_threshold()).c_str());
    file_stream.CloseElement();

    // Activation function

    file_stream.OpenElement("ActivationFunction");
    file_stream.PushText(write_activation_function().c_str());
    file_stream.CloseElement();

    // Biases

    file_stream.OpenElement("Parameters");
    file_stream.PushText(tensor_to_string(get_parameters()).c_str());
    file_stream.CloseElement();

    // Probabilistic layer (end tag)

    file_stream.CloseElement();
}


pair<type*, dimensions> ProbabilisticLayer3DForwardPropagation::get_outputs_pair() const
{
    ProbabilisticLayer3D* probabilistic_layer_3d = static_cast<ProbabilisticLayer3D*>(layer);

    const Index neurons_number = probabilistic_layer_3d->get_neurons_number();
    const Index inputs_number = probabilistic_layer_3d->get_inputs_number();

    return {outputs_data, {batch_samples_number, inputs_number, neurons_number}};
}


void ProbabilisticLayer3DForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    ProbabilisticLayer3D* probabilistic_layer_3d = static_cast<ProbabilisticLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = probabilistic_layer_3d->get_neurons_number();
    const Index inputs_number = probabilistic_layer_3d->get_inputs_number();
    
    outputs.resize(batch_samples_number, inputs_number, neurons_number);
    
    outputs_data = outputs.data();
}


void ProbabilisticLayer3DBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    ProbabilisticLayer3D* probabilistic_layer_3d = static_cast<ProbabilisticLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index neurons_number = probabilistic_layer_3d->get_neurons_number();
    const Index inputs_number = probabilistic_layer_3d->get_inputs_number();
    const Index inputs_depth = probabilistic_layer_3d->get_inputs_depth();

    targets.resize(batch_samples_number, inputs_number);
    mask.resize(batch_samples_number, inputs_number);

    biases_derivatives.resize(neurons_number);

    synaptic_weights_derivatives.resize(inputs_depth, neurons_number);

    combinations_derivatives.resize(batch_samples_number, inputs_number, neurons_number);

    input_derivatives.resize(batch_samples_number, inputs_number, inputs_depth);
}


vector<pair<type*, dimensions>> ProbabilisticLayer3DBackPropagation::get_input_derivative_pairs() const
{
    ProbabilisticLayer3D* probabilistic_layer_3d = static_cast<ProbabilisticLayer3D*>(layer);

    const Index inputs_number = probabilistic_layer_3d->get_inputs_number();
    const Index inputs_depth = probabilistic_layer_3d->get_inputs_depth();

    return {{(type*)(input_derivatives.data()), {batch_samples_number, inputs_number, inputs_depth}} };
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
