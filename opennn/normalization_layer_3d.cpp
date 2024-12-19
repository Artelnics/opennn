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

NormalizationLayer3D::NormalizationLayer3D(const Index& new_inputs_number,
                                           const Index& new_inputs_depth,
                                           const string& new_name) : Layer()
{
    set(new_inputs_number, new_inputs_depth, new_name);

    layer_type = Type::Normalization3D;

    name = new_name;
}


Index NormalizationLayer3D::get_sequence_length() const
{
    return sequence_length;
}


Index NormalizationLayer3D::get_embedding_dimension() const
{
    return gammas.size();
}


dimensions NormalizationLayer3D::get_input_dimensions() const
{
    const Index embedding_dimension = get_embedding_dimension();

    return { sequence_length, embedding_dimension };
}


dimensions NormalizationLayer3D::get_output_dimensions() const
{
    const Index embedding_dimension = get_embedding_dimension();

    return { sequence_length, embedding_dimension };
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


void NormalizationLayer3D::set(const Index& new_sequence_length, 
                               const Index& new_inputs_depth, 
                               const string& new_name)
{
    sequence_length = new_sequence_length;

    gammas.resize(new_inputs_depth);
    gammas.setConstant(1);

    betas.resize(new_inputs_depth);
    betas.setZero();

    name = new_name;

    layer_type = Type::Normalization3D;
}


void NormalizationLayer3D::set_sequence_length(const Index& new_sequence_length)
{
    sequence_length = new_sequence_length;
}


void NormalizationLayer3D::set_inputs_depth(const Index& new_inputs_depth)
{
    gammas.resize(new_inputs_depth);
    betas.resize(new_inputs_depth);
}


void NormalizationLayer3D::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
{
    memcpy(gammas.data(), new_parameters.data() + index, gammas.size()*sizeof(type));

    memcpy(betas.data(), new_parameters.data() + index + gammas.size(), betas.size()*sizeof(type));
}


void NormalizationLayer3D::set_gammas_constant(const type& value)
{
    gammas.setConstant(value);
}


void NormalizationLayer3D::set_betas_constant(const type& value)
{
    betas.setConstant(value);
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
                                             const bool&)
{
    const Index samples_number = input_pairs[0].second[0];
    const Index inputs_number = input_pairs[0].second[1];
    const Index inputs_depth = input_pairs[0].second[2];

    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    NormalizationLayer3DForwardPropagation* normalization_layer_3d_forward_propagation =
        static_cast<NormalizationLayer3DForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 3>& outputs = normalization_layer_3d_forward_propagation->outputs;

    Tensor<type, 3>& means = normalization_layer_3d_forward_propagation->means;
    Tensor<type, 3>& standard_deviations = normalization_layer_3d_forward_propagation->standard_deviations;
    const type& epsilon = normalization_layer_3d_forward_propagation->epsilon;

    const Eigen::array<Index, 3> range_3{ { samples_number, inputs_number, 1 }};
    const Eigen::array<Index, 3> expand_normalization_axis{ { 1, 1, inputs_depth }};
       
    means.device(*thread_pool_device) = inputs.mean(normalization_axis)
                                            .reshape(range_3).broadcast(expand_normalization_axis);

    standard_deviations.device(*thread_pool_device) = (inputs - means).square().mean(normalization_axis).sqrt()
                                           .reshape(range_3).broadcast(expand_normalization_axis);

    outputs.device(*thread_pool_device) = (inputs - means) / (standard_deviations + epsilon);

    multiply_matrices(thread_pool_device.get(), outputs, gammas);
    sum_matrices(thread_pool_device.get(), betas, outputs);
}


void NormalizationLayer3D::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                          const vector<pair<type*, dimensions>>& delta_pairs,
                                          unique_ptr<LayerForwardPropagation>& forward_propagation,
                                          unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_samples_number = input_pairs[0].second[0];

    const Index embedding_dimension = get_embedding_dimension();

    if(delta_pairs.size() > 1)     
        add_deltas(delta_pairs);

    const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    // Forward propagation

    const NormalizationLayer3DForwardPropagation* normalization_layer_3d_forward_propagation =
        static_cast<NormalizationLayer3DForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 3>& outputs = normalization_layer_3d_forward_propagation->outputs;

    const Tensor<type, 3>& standard_deviations = normalization_layer_3d_forward_propagation->standard_deviations;

    const TensorMap<Tensor<type, 2>> standard_deviations_matrix((type*)standard_deviations.data(), batch_samples_number, sequence_length);

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
    
    gammas_derivatives.device(*thread_pool_device) = (outputs * deltas).sum(sum_dimensions_2);
    
    betas_derivatives.device(*thread_pool_device) = deltas.sum(sum_dimensions_2);
    
    // Input derivatives

    standard_deviation_derivatives.device(*thread_pool_device) = outputs;

    scaled_deltas.device(*thread_pool_device) = deltas;

    multiply_matrices(thread_pool_device.get(), scaled_deltas, gammas);

    //aux_2d.device(*thread_pool_device) = 1 / type(inputs_depth) * (scaled_deltas * outputs).sum(sum_dimensions_1) / (standard_deviations_matrix + epsilon);

    aux_2d.device(*thread_pool_device) = (scaled_deltas * outputs).sum(sum_dimensions_1) / (type(embedding_dimension) * (standard_deviations_matrix + epsilon));

    multiply_matrices(thread_pool_device.get(), standard_deviation_derivatives, aux_2d);

    scaled_deltas.device(*thread_pool_device) = scaled_deltas / (standard_deviations + epsilon);

    input_derivatives.device(*thread_pool_device) = scaled_deltas - standard_deviation_derivatives;

    aux_2d.device(*thread_pool_device) = 1 / type(embedding_dimension) * scaled_deltas.sum(sum_dimensions_1);

    substract_matrices(thread_pool_device.get(), aux_2d, input_derivatives);
}


void NormalizationLayer3D::add_deltas(const vector<pair<type*, dimensions>>& delta_pairs) const
{
    TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    for(Index i = 1; i < Index(delta_pairs.size()); i++)
        deltas.device(*thread_pool_device) += tensor_map_3(delta_pairs[i]);
}


void NormalizationLayer3D::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                           const Index& index,
                                           Tensor<type, 1>& gradient) const
{
    const Index gammas_number = gammas.size();
    const Index betas_number = betas.size();

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


void NormalizationLayer3D::from_XML(const XMLDocument& document)
{
    const XMLElement* normalization_layer_element = document.FirstChildElement("Normalization3D");

    if(!normalization_layer_element)
        throw runtime_error("Normalization3D element is nullptr.\n");

    set_name(read_xml_string(normalization_layer_element, "Name"));
    set_sequence_length(read_xml_index(normalization_layer_element, "SequenceLength"));
    set_inputs_depth(read_xml_index(normalization_layer_element, "InputsDepth"));
    set_parameters(to_type_vector(read_xml_string(normalization_layer_element, "Parameters"), " "));
}


void NormalizationLayer3D::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Normalization3D");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "SequenceLength", to_string(get_sequence_length()));
    add_xml_element(printer, "EmbeddingDimension", to_string(get_embedding_dimension()));
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();
}


NormalizationLayer3DForwardPropagation::NormalizationLayer3DForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_samples_number, new_layer);
}


pair<type*, dimensions> NormalizationLayer3DForwardPropagation::get_outputs_pair() const
{
    NormalizationLayer3D* normalization_layer_3d = static_cast<NormalizationLayer3D*>(layer);

    const Index sequence_length = normalization_layer_3d->get_sequence_length();
    const Index embedding_dimension = normalization_layer_3d->get_embedding_dimension();

    return { (type*)outputs.data(), { batch_samples_number, sequence_length, embedding_dimension } };
}


void NormalizationLayer3DForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    NormalizationLayer3D* normalization_layer_3d = static_cast<NormalizationLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index sequence_length = normalization_layer_3d->get_sequence_length();
    const Index embedding_dimension = normalization_layer_3d->get_embedding_dimension();

    outputs.resize(batch_samples_number, sequence_length, embedding_dimension);

    means.resize(batch_samples_number, sequence_length, embedding_dimension);
    standard_deviations.resize(batch_samples_number, sequence_length, embedding_dimension);
}


void NormalizationLayer3DForwardPropagation::print() const
{
    cout << "Outputs:" << endl
        << outputs << endl;
}


void NormalizationLayer3DBackPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    NormalizationLayer3D* normalization_layer_3d = static_cast<NormalizationLayer3D*>(layer);

    batch_samples_number = new_batch_samples_number;

    const Index sequence_length = normalization_layer_3d->get_sequence_length();
    const Index embedding_dimension = normalization_layer_3d->get_embedding_dimension();

    gammas_derivatives.resize(embedding_dimension);
    betas_derivatives.resize(embedding_dimension);

    scaled_deltas.resize(batch_samples_number, sequence_length, embedding_dimension);
    standard_deviation_derivatives.resize(batch_samples_number, sequence_length, embedding_dimension);
    aux_2d.resize(batch_samples_number, sequence_length);

    input_derivatives.resize(batch_samples_number, sequence_length, embedding_dimension);

}


void NormalizationLayer3DBackPropagation::print() const
{
    cout << "Gammas derivatives:" << endl
        << gammas_derivatives << endl
        << "Betas derivatives:" << endl
        << betas_derivatives << endl;
}


NormalizationLayer3DBackPropagation::NormalizationLayer3DBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_samples_number, new_layer);
}


vector<pair<type*, dimensions>> NormalizationLayer3DBackPropagation::get_input_derivative_pairs() const
{
    NormalizationLayer3D* normalization_layer_3d = static_cast<NormalizationLayer3D*>(layer);

    const Index sequence_length = normalization_layer_3d->get_sequence_length();
    const Index embedding_dimension = normalization_layer_3d->get_embedding_dimension();

    return { {(type*)(input_derivatives.data()), {batch_samples_number, sequence_length, embedding_dimension}} };
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
