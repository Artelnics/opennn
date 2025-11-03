//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "dense_layer_3d.h"

namespace opennn
{

Dense3d::Dense3d(const Index& new_sequence_length,
                 const Index& new_input_embedding,
                 const Index& new_output_embedding,
                 const string& new_activation_function,
                 const string& new_name) : Layer()
{
    set(new_sequence_length, new_input_embedding, new_output_embedding, new_activation_function, new_name);
}


Index Dense3d::get_sequence_length() const
{
    return sequence_length;
}


Index Dense3d::get_input_embedding() const
{
    return weights.dimension(0);
}


Index Dense3d::get_output_embedding() const
{
    return biases.size();
}


void Dense3d::set_dropout_rate(const type& new_dropout_rate)
{
    dropout_rate = new_dropout_rate;
}


dimensions Dense3d::get_input_dimensions() const
{
    return { sequence_length, get_input_embedding() };
}


dimensions Dense3d::get_output_dimensions() const
{
    return { sequence_length, get_output_embedding() };
}


type Dense3d::get_dropout_rate() const
{
    return dropout_rate;
}


string Dense3d::get_activation_function() const
{
    return activation_function;
}


void Dense3d::set(const Index& new_sequence_length,
                  const Index& new_input_dimension,
                  const Index& new_output_dimension,
                  const string& new_activation_function,
                  const string& new_label)
{
    sequence_length = new_sequence_length;

    biases.resize(new_output_dimension);

    weights.resize(new_input_dimension, new_output_dimension);

    set_parameters_glorot();

    set_activation_function(new_activation_function);

    label = new_label;

    name = "Dense3d";

    dropout_rate = 0;
}


void Dense3d::set_activation_function(const string& new_activation_function)
{
    if(new_activation_function == "HyperbolicTangent"
    || new_activation_function == "Linear"
    || new_activation_function == "RectifiedLinear"
    || new_activation_function == "Softmax")
        activation_function = new_activation_function;
    else
        throw runtime_error("Unknown activation function: " + new_activation_function);
}


void Dense3d::calculate_combinations(const Tensor<type, 3>& inputs,
                                     Tensor<type, 3>& combinations) const
{
    combinations.device(*thread_pool_device) = inputs.contract(weights, axes(2,0))
                                               + biases.reshape(array<Index, 3>{1, 1, combinations.dimension(2)})
                                                     .broadcast(array<Index, 3>{combinations.dimension(0), combinations.dimension(1), 1});
}


void Dense3d::forward_propagate(const vector<TensorView>& input_views,
                                unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                const bool& is_training)
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map<3>(input_views[0]);

    Dense3dForwardPropagation* this_forward_propagation =
        static_cast<Dense3dForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 3>& outputs = this_forward_propagation->outputs;

    calculate_combinations(inputs,
                           outputs);

    if(is_training && dropout_rate > type(0))
        dropout(outputs, dropout_rate);

    is_training
        ? calculate_activations(activation_function, outputs, this_forward_propagation->activation_derivatives)
        : calculate_activations(activation_function, outputs, empty_3);
}


void Dense3d::back_propagate(const vector<TensorView>& input_views,
                             const vector<TensorView>& delta_views,
                             unique_ptr<LayerForwardPropagation>& forward_propagation,
                             unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map<3>(input_views[0]);

    if(delta_views.size() > 1)
        add_deltas(delta_views);

    TensorMap<Tensor<type, 3>> deltas = tensor_map<3>(delta_views[0]);

    // Forward propagation

    const Dense3dForwardPropagation* dense3d_layer_forward_propagation =
        static_cast<Dense3dForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 3>& activation_derivatives = dense3d_layer_forward_propagation->activation_derivatives;

    // Back propagation

    Dense3dBackPropagation* dense3d_back_propagation =
        static_cast<Dense3dBackPropagation*>(back_propagation.get());

    Tensor<type, 1>& bias_deltas = dense3d_back_propagation->bias_deltas;
    Tensor<type, 2>& weight_deltas = dense3d_back_propagation->weight_deltas;

    Tensor<type, 3>& input_deltas = dense3d_back_propagation->input_deltas;

    deltas.device(*thread_pool_device) = deltas * activation_derivatives;

    bias_deltas.device(*thread_pool_device) = deltas.sum(array<Index, 2>({0,1}));

    weight_deltas.device(*thread_pool_device) = inputs.contract(deltas, axes(0,0,1,1));

    input_deltas.device(*thread_pool_device) = deltas.contract(weights, axes(2,1));
}


void Dense3d::from_XML(const XMLDocument& document)
{
    const XMLElement* dense2d_layer_element = document.FirstChildElement("Dense3d");

    if(!dense2d_layer_element)
        throw runtime_error("Dense3d element is nullptr.\n");

    const Index new_sequence_length = read_xml_index(dense2d_layer_element, "InputsNumber");
    const Index new_input_dimension = read_xml_index(dense2d_layer_element, "InputsDepth");
    const Index new_output_dimension = read_xml_index(dense2d_layer_element, "NeuronsNumber");

    set(new_sequence_length, new_input_dimension, new_output_dimension);

    set_label(read_xml_string(dense2d_layer_element, "Label"));
    set_activation_function(read_xml_string(dense2d_layer_element, "Activation"));

    string_to_tensor<type, 1>(read_xml_string(dense2d_layer_element, "Biases"), biases);
    string_to_tensor<type, 2>(read_xml_string(dense2d_layer_element, "Weights"), weights);
}


void Dense3d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Dense3d");

    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "InputsNumber", to_string(get_sequence_length()));
    add_xml_element(printer, "InputsDepth", to_string(get_input_embedding()));
    add_xml_element(printer, "NeuronsNumber", to_string(get_output_embedding()));
    add_xml_element(printer, "Activation", activation_function);
    add_xml_element(printer, "Biases", tensor_to_string<type, 1>(biases));
    add_xml_element(printer, "Weights", tensor_to_string<type, 2>(weights));

    printer.CloseElement();
}


void Dense3dForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl
         << "Activation derivatives:" << endl
         << activation_derivatives << endl;
}


Dense3dForwardPropagation::Dense3dForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


TensorView Dense3dForwardPropagation::get_output_pair() const
{
    Dense3d* dense_3d = static_cast<Dense3d*>(layer);

    const Index sequence_length = dense_3d->get_sequence_length();
    const Index output_embedding = dense_3d->get_output_embedding();

    return { (type*)outputs.data(), { batch_size, sequence_length, output_embedding } };
}


void Dense3dForwardPropagation::initialize()
{
    Dense3d* dense_3d = static_cast<Dense3d*>(layer);

    const Index output_embedding = dense_3d->get_output_embedding();

    const Index sequence_length = dense_3d->get_sequence_length();

    outputs.resize(batch_size, sequence_length, output_embedding);

    activation_derivatives.resize(batch_size, sequence_length, output_embedding);
}


void Dense3dBackPropagation::initialize()
{
    Dense3d* dense_3d = static_cast<Dense3d*>(layer);

    const Index output_embedding = dense_3d->get_output_embedding();
    const Index sequence_length = dense_3d->get_sequence_length();
    const Index input_embedding = dense_3d->get_input_embedding();

    bias_deltas.resize(output_embedding);
    weight_deltas.resize(input_embedding, output_embedding);
    input_deltas.resize(batch_size, sequence_length, input_embedding);
}


void Dense3dBackPropagation::print() const
{
    cout << "Biases derivatives:" << endl
         << bias_deltas << endl
         << "Synaptic weights derivatives:" << endl
         << weight_deltas << endl;
}


Dense3dBackPropagation::Dense3dBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<TensorView> Dense3dBackPropagation::get_input_derivative_views() const
{
    Dense3d* dense_3d = static_cast<Dense3d*>(layer);

    const Index sequence_length = dense_3d->get_sequence_length();
    const Index input_embedding = dense_3d->get_input_embedding();

    return {{(type*)(input_deltas.data()), {batch_size, sequence_length, input_embedding}}};
}

REGISTER(Layer, Dense3d, "Dense3d")
REGISTER(LayerForwardPropagation, Dense3dForwardPropagation, "Dense3d")
REGISTER(LayerBackPropagation, Dense3dBackPropagation, "Dense3d")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
