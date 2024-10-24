//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "strings_utilities.h"
#include "tensors.h"
#include "normalization_layer_3d.h"

namespace opennn
{

NormalizationLayer3D::NormalizationLayer3D() : Layer()
{
    set();

    layer_type = Type::Normalization3D;
}


NormalizationLayer3D::NormalizationLayer3D(const Index& new_inputs_number,
                                            const Index& new_inputs_depth) : Layer()
{
    set(new_inputs_number, new_inputs_depth);

    layer_type = Type::Normalization3D;

    name = "normalization_layer_3d";
}


Index NormalizationLayer3D::get_inputs_number() const
{
    return inputs_number;
}


Index NormalizationLayer3D::get_inputs_depth() const
{
    return inputs_depth;
}


dimensions NormalizationLayer3D::get_output_dimensions() const
{
    return { inputs_number, inputs_depth };
}


Index NormalizationLayer3D::get_gammas_number() const
{
    return gammas.size();
}


Index NormalizationLayer3D::get_betas_number() const
{
    return betas.size();
}


Index NormalizationLayer3D::get_parameters_number() const
{
    return gammas.size() + betas.size();
}


Tensor<type, 1> NormalizationLayer3D::get_parameters() const
{
    Tensor<type, 1> parameters(gammas.size() + betas.size());

    memcpy(parameters.data(), gammas.data(), gammas.size()*sizeof(type));

    memcpy(parameters.data() + gammas.size(), betas.data(), betas.size()*sizeof(type));

    return parameters;
}


const bool& NormalizationLayer3D::get_display() const
{
    return display;
}


void NormalizationLayer3D::set()
{
    set_default();
}


void NormalizationLayer3D::set(const Index& new_inputs_number, const Index& new_inputs_depth)
{
    inputs_number = new_inputs_number;

    inputs_depth = new_inputs_depth;

    gammas.resize(inputs_depth);
    betas.resize(inputs_depth);

    set_default();
}


void NormalizationLayer3D::set_default()
{
    name = "normalization_layer_3d";

    display = true;

    layer_type = Type::Normalization3D;

    set_parameters_default();
}


void NormalizationLayer3D::set_inputs_number(const Index& new_inputs_number)
{
    inputs_number = new_inputs_number;
}


void NormalizationLayer3D::set_inputs_depth(const Index& new_inputs_depth)
{
    inputs_depth = new_inputs_depth;

    gammas.resize(inputs_depth);
    betas.resize(inputs_depth);
}


void NormalizationLayer3D::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    memcpy(gammas.data(), new_parameters.data() + index, gammas.size()*sizeof(type));

    memcpy(betas.data(), new_parameters.data() + index + gammas.size(), betas.size()*sizeof(type));
}


void NormalizationLayer3D::set_display(const bool& new_display)
{
    display = new_display;
}


void NormalizationLayer3D::set_gammas_constant(const type& value)
{
    gammas.setConstant(value);
}


void NormalizationLayer3D::set_betas_constant(const type& value)
{
    betas.setConstant(value);
}


void NormalizationLayer3D::set_parameters_default()
{
    gammas.setConstant(1);
    betas.setZero();
}


void NormalizationLayer3D::set_parameters_constant(const type& value)
{
    gammas.setConstant(value);
    betas.setConstant(value);
}


void NormalizationLayer3D::set_parameters_random()
{
    set_random(gammas);
    set_random(betas);
}


void NormalizationLayer3D::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                             unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                             const bool& is_training)
{
    const Index samples_number = input_pairs[0].second[0];
    const Index inputs_number = input_pairs[0].second[1];
    const Index inputs_depth = input_pairs[0].second[2];

    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    NormalizationLayer3DForwardPropagation* normalization_layer_3d_forward_propagation =
        static_cast<NormalizationLayer3DForwardPropagation*>(layer_forward_propagation.get());

    // @todo Can we avoid normalized_inputs

    Tensor<type, 3>& normalized_inputs = normalization_layer_3d_forward_propagation->normalized_inputs;
    Tensor<type, 3>& outputs = normalization_layer_3d_forward_propagation->outputs;

    Tensor<type, 3>& means = normalization_layer_3d_forward_propagation->means;
    Tensor<type, 3>& standard_deviations = normalization_layer_3d_forward_propagation->standard_deviations;
    const type& epsilon = normalization_layer_3d_forward_propagation->epsilon;

    const Eigen::array<Index, 1> normalization_axis{ { 2 }};
    const Eigen::array<Index, 3> range_3{ { samples_number, inputs_number, 1 }};
    const Eigen::array<Index, 3> expand_normalization_axis{ { 1, 1, inputs_depth }};
    
    means.device(*thread_pool_device) = inputs.mean(normalization_axis)
                                       .reshape(range_3).broadcast(expand_normalization_axis);

    standard_deviations.device(*thread_pool_device) = (inputs - means).pow(2).mean(normalization_axis).sqrt()
                                           .reshape(range_3).broadcast(expand_normalization_axis);

    normalized_inputs.device(*thread_pool_device) = (inputs - means) / (standard_deviations + epsilon);

    // @todo outputs is assigned twice!!!

    outputs.device(*thread_pool_device) = normalized_inputs;

    outputs.device(*thread_pool_device) = (inputs - means) / (standard_deviations + epsilon);

    multiply_matrices(thread_pool_device, outputs, gammas);

    sum_matrices(thread_pool_device, betas, outputs);
}


void NormalizationLayer3D::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                          const vector<pair<type*, dimensions>>& delta_pairs,
                                          unique_ptr<LayerForwardPropagation>& forward_propagation,
                                          unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_samples_number = input_pairs[0].second[0];

    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    if(delta_pairs.size() > 1)     
        add_deltas(delta_pairs);

    const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    // Forward propagation

    const NormalizationLayer3DForwardPropagation* normalization_layer_3d_forward_propagation =
        static_cast<NormalizationLayer3DForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 3>& normalized_inputs = normalization_layer_3d_forward_propagation->normalized_inputs;

    const Tensor<type, 3>& standard_deviations = normalization_layer_3d_forward_propagation->standard_deviations;

    const TensorMap<Tensor<type, 2>> standard_deviations_matrix((type*)standard_deviations.data(), batch_samples_number, inputs_number);

    const type& epsilon = normalization_layer_3d_forward_propagation->epsilon;

    // Back propagation

    NormalizationLayer3DBackPropagation* normalization_layer_3d_back_propagation =
        static_cast<NormalizationLayer3DBackPropagation*>(back_propagation.get());
    
    Tensor<type, 1>& gammas_derivatives = normalization_layer_3d_back_propagation->gammas_derivatives;
    Tensor<type, 1>& betas_derivatives = normalization_layer_3d_back_propagation->betas_derivatives;

    Tensor<type, 3>& scaled_deltas = normalization_layer_3d_back_propagation->scaled_deltas;
    Tensor<type, 3>& standard_deviation_derivatives = normalization_layer_3d_back_propagation->standard_deviation_derivatives;
    Tensor<type, 2>& aux_2d = normalization_layer_3d_back_propagation->aux_2d;

    Tensor<type, 3>& input_derivatives = normalization_layer_3d_back_propagation->input_derivatives;
    
    // Parameters derivatives
    
    gammas_derivatives.device(*thread_pool_device) = (normalized_inputs * deltas).sum(Eigen::array<Index, 2>({ 0, 1 }));
    
    betas_derivatives.device(*thread_pool_device) = deltas.sum(Eigen::array<Index, 2>({ 0, 1 }));
    
    // Input derivatives

    standard_deviation_derivatives.device(*thread_pool_device) = normalized_inputs;

    scaled_deltas.device(*thread_pool_device) = deltas;

    multiply_matrices(thread_pool_device, scaled_deltas, gammas);

    aux_2d.device(*thread_pool_device) = 1 / type(inputs_depth) * (scaled_deltas * normalized_inputs).sum(Eigen::array<Index, 1>({ 2 })) / (standard_deviations_matrix + epsilon);

    multiply_matrices(thread_pool_device, standard_deviation_derivatives, aux_2d);

    scaled_deltas.device(*thread_pool_device) = scaled_deltas / (standard_deviations + epsilon);

    input_derivatives.device(*thread_pool_device) = scaled_deltas - standard_deviation_derivatives;

    aux_2d.device(*thread_pool_device) = 1 / type(inputs_depth) * scaled_deltas.sum(Eigen::array<Index, 1>({ 2 }));

    substract_matrices(thread_pool_device, aux_2d, input_derivatives);
}


void NormalizationLayer3D::add_deltas(const vector<pair<type*, dimensions>>& delta_pairs) const
{
    TensorMap<Tensor<type, 3>> deltas= tensor_map_3(delta_pairs[0]);

    for(Index i = 1; i < Index(delta_pairs.size()); i++)
        deltas.device(*thread_pool_device) += tensor_map_3(delta_pairs[i]);
}


void NormalizationLayer3D::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                           const Index& index,
                                           Tensor<type, 1>& gradient) const
{
    const Index gammas_number = get_gammas_number();
    const Index betas_number = get_betas_number();

    NormalizationLayer3DBackPropagation* normalization_layer_3d_back_propagation =
        static_cast<NormalizationLayer3DBackPropagation*>(back_propagation.get());

    const type* gammas_derivatives_data = normalization_layer_3d_back_propagation->gammas_derivatives.data();
    const type* betas_derivatives_data = normalization_layer_3d_back_propagation->betas_derivatives.data();
    type* gradient_data = gradient.data();

    #pragma omp parallel sections
    {
        #pragma omp section
        memcpy(gradient_data + index, gammas_derivatives_data, gammas_number * sizeof(type));

        #pragma omp section
        memcpy(gradient_data + index + gammas_number, betas_derivatives_data, betas_number * sizeof(type));
    }
}


void NormalizationLayer3D::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Normalization layer

    const tinyxml2::XMLElement* normalization_layer_element = document.FirstChildElement("NormalizationLayer3D");

    if(!normalization_layer_element)
        throw runtime_error("NormalizationLayer3D element is nullptr.\n");

    // Layer name

    const tinyxml2::XMLElement* layer_name_element = normalization_layer_element->FirstChildElement("Name");

    if(!layer_name_element)
        throw runtime_error("LayerName element is nullptr.\n");

    if(layer_name_element->GetText())
        set_name(layer_name_element->GetText());

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = normalization_layer_element->FirstChildElement("InputsNumber");

    if(!inputs_number_element)
        throw runtime_error("InputsNumber element is nullptr.\n");

    if(inputs_number_element->GetText())
        set_inputs_number(Index(stoi(inputs_number_element->GetText())));

    // Inputs depth

    const tinyxml2::XMLElement* inputs_depth_element = normalization_layer_element->FirstChildElement("InputsDepth");

    if(!inputs_depth_element)
        throw runtime_error("InputsDepth element is nullptr.\n");

    if(inputs_depth_element->GetText())
        set_inputs_depth(Index(stoi(inputs_depth_element->GetText())));

    // Gammas

    const tinyxml2::XMLElement* parameters_element = normalization_layer_element->FirstChildElement("Parameters");

    if(!parameters_element)
        throw runtime_error("Parameters element is nullptr.\n");

    if(parameters_element->GetText())
        set_parameters(to_type_vector(parameters_element->GetText(), " "));
}


void NormalizationLayer3D::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    // Normalization layer

    file_stream.OpenElement("NormalizationLayer3D");

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

    // Parameters

    file_stream.OpenElement("Parameters");
    file_stream.PushText(tensor_to_string(get_parameters()).c_str());
    file_stream.CloseElement();

    // Normalization layer (end tag)

    file_stream.CloseElement();
}


pair<type*, dimensions> NormalizationLayer3DForwardPropagation::get_outputs_pair() const
{
    NormalizationLayer3D* normalization_layer_3d = static_cast<NormalizationLayer3D*>(layer);

    const Index inputs_number = normalization_layer_3d->get_inputs_number();
    const Index inputs_depth = normalization_layer_3d->get_inputs_depth();

    return { (type*)outputs_data, { batch_samples_number, inputs_number, inputs_depth } };
}


void NormalizationLayer3DForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    NormalizationLayer3D* normalization_layer_3d = static_cast<NormalizationLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index inputs_number = normalization_layer_3d->get_inputs_number();
    const Index inputs_depth = normalization_layer_3d->get_inputs_depth();

    outputs.resize(batch_samples_number, inputs_number, inputs_depth);

    outputs_data = outputs.data();

    normalized_inputs.resize(batch_samples_number, inputs_number, inputs_depth);

    means.resize(batch_samples_number, inputs_number, inputs_depth);
    standard_deviations.resize(batch_samples_number, inputs_number, inputs_depth);
}


void NormalizationLayer3DBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    NormalizationLayer3D* normalization_layer_3d = static_cast<NormalizationLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index inputs_number = normalization_layer_3d->get_inputs_number();
    const Index inputs_depth = normalization_layer_3d->get_inputs_depth();

    gammas_derivatives.resize(inputs_depth);
    betas_derivatives.resize(inputs_depth);

    scaled_deltas.resize(batch_samples_number, inputs_number, inputs_depth);
    standard_deviation_derivatives.resize(batch_samples_number, inputs_number, inputs_depth);
    aux_2d.resize(batch_samples_number, inputs_number);

    input_derivatives.resize(batch_samples_number, inputs_number, inputs_depth);

}


vector<pair<type*, dimensions>> NormalizationLayer3DBackPropagation::get_input_derivative_pairs() const
{
    NormalizationLayer3D* normalization_layer_3d = static_cast<NormalizationLayer3D*>(layer);

    const Index inputs_number = normalization_layer_3d->get_inputs_number();
    const Index inputs_depth = normalization_layer_3d->get_inputs_depth();

    return { {(type*)(input_derivatives.data()), {batch_samples_number, inputs_number, inputs_depth}} };
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
