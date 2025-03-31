//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "tensors.h"
#include "normalization_layer_3d.h"
#include "strings_utilities.h"

namespace opennn
{

Normalization3d::Normalization3d(const Index& new_sequence_length,
                                 const Index& new_embedding_dimension,
                                 const string& new_name) : Layer()
{
    layer_type = Type::Normalization3D;

    set(new_sequence_length, new_embedding_dimension, new_name);
}


Index Normalization3d::get_sequence_length() const
{
    return sequence_length;
}


Index Normalization3d::get_embedding_dimension() const
{
    return gammas.size();
}


dimensions Normalization3d::get_input_dimensions() const
{
    const Index embedding_dimension = get_embedding_dimension();

    return { sequence_length, embedding_dimension };
}


dimensions Normalization3d::get_output_dimensions() const
{
    const Index embedding_dimension = get_embedding_dimension();

    return { sequence_length, embedding_dimension };
}


Index Normalization3d::get_parameters_number() const
{
    return gammas.size() + betas.size();
}


Tensor<type, 1> Normalization3d::get_parameters() const
{
    Tensor<type, 1> parameters(gammas.size() + betas.size());

    Index index = 0;

    copy_to_vector(parameters, gammas, index);
    copy_to_vector(parameters, betas, index);

    return parameters;
}


void Normalization3d::set(const Index& new_sequence_length, 
                          const Index& new_embedding_dimension, 
                          const string& new_name)
{
    sequence_length = new_sequence_length;

    gammas.resize(new_embedding_dimension);
    gammas.setConstant(1);

    betas.resize(new_embedding_dimension);
    betas.setZero();

    name = new_name;
}


void Normalization3d::set_parameters(const Tensor<type, 1>& new_parameters, Index& index)
{
    copy_from_vector(gammas, new_parameters, index);
    copy_from_vector(betas, new_parameters, index);
}


void Normalization3d::set_parameters_constant(const type& value)
{
    gammas.setConstant(value);
    betas.setConstant(value);
}


void Normalization3d::set_parameters_random()
{
    set_random(gammas);
    set_random(betas);
}


void Normalization3d::standarization(const Tensor<type, 3>& inputs,
                                     Tensor<type, 3>& outputs,
                                     Tensor<type, 2>& means,
                                     Tensor<type, 2>& standard_deviations) const
{

    const Index batch_size = inputs.dimension(0);
    const Index sequence_length = inputs.dimension(1);
    const Index embedding_dimension = inputs.dimension(2);

    const Eigen::array<Index, 3> range_3{ { batch_size, sequence_length, 1 } };
    const Eigen::array<Index, 3> expand_normalization_axis{ { 1, 1, embedding_dimension } };

    means.device(*thread_pool_device) = inputs.mean(normalization_axis);
       // .reshape(range_3).broadcast(expand_normalization_axis);

    standard_deviations.device(*thread_pool_device) = (inputs - means.reshape(range_3).broadcast(expand_normalization_axis)).square().mean(normalization_axis).sqrt();
       // .reshape(range_3).broadcast(expand_normalization_axis);

    outputs.device(*thread_pool_device) = (inputs - means.reshape(range_3).broadcast(expand_normalization_axis)) / (standard_deviations.reshape(range_3).broadcast(expand_normalization_axis) + epsilon);
}


void Normalization3d::affine_transformation(Tensor<type, 3>& outputs) const
{
    multiply_matrices(thread_pool_device.get(), outputs, gammas);
    sum_matrices(thread_pool_device.get(), betas, outputs);
}


void Normalization3d::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                        unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                        const bool&)
{
    const TensorMap<Tensor<type, 3>> inputs = tensor_map_3(input_pairs[0]);

    Normalization3dForwardPropagation* this_forward_propagation =
        static_cast<Normalization3dForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 3>& outputs = this_forward_propagation->outputs;

    Tensor<type, 2>& means = this_forward_propagation->means;
    Tensor<type, 2>& standard_deviations = this_forward_propagation->standard_deviations;
 
    standarization(inputs, outputs, means, standard_deviations);

    affine_transformation(outputs);
}


void Normalization3d::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                     const vector<pair<type*, dimensions>>& delta_pairs,
                                     unique_ptr<LayerForwardPropagation>& forward_propagation,
                                     unique_ptr<LayerBackPropagation>& back_propagation) const
{
    // @todo simplify                                                                                                  
    const Index batch_size = input_pairs[0].second[0];
    const Index embedding_dimension = get_embedding_dimension();

    const Eigen::array<Index, 3> range_3{ { batch_size, sequence_length, 1 } };
    const Eigen::array<Index, 3> expand_normalization_axis{ { 1, 1, embedding_dimension } };

    if(delta_pairs.size() > 1)     
        add_deltas(delta_pairs);

    const TensorMap<Tensor<type, 3>> deltas = tensor_map_3(delta_pairs[0]);

    // Forward propagation

    const Normalization3dForwardPropagation* this_forward_propagation 
        = static_cast<Normalization3dForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 3>& outputs = this_forward_propagation->outputs;
    const Tensor<type, 2>& standard_deviations = this_forward_propagation->standard_deviations;

    // Back propagation

    Normalization3dBackPropagation* this_back_propagation =
        static_cast<Normalization3dBackPropagation*>(back_propagation.get());
    
    Tensor<type, 1>& gamma_derivatives = this_back_propagation->gamma_derivatives;
    Tensor<type, 1>& beta_derivatives = this_back_propagation->beta_derivatives;

    Tensor<type, 3>& scaled_deltas = this_back_propagation->scaled_deltas;
    Tensor<type, 3>& standard_deviation_derivatives = this_back_propagation->standard_deviation_derivatives;

    Tensor<type, 3>& input_derivatives = this_back_propagation->input_derivatives;

    Tensor<type, 2>& aux_2d = this_back_propagation->aux_2d;

    // Parameters derivatives
    
    gamma_derivatives.device(*thread_pool_device) = (outputs * deltas).sum(sum_dimensions_2);    
    beta_derivatives.device(*thread_pool_device) = deltas.sum(sum_dimensions_2);
    
    // Input derivatives
    //@TODO REVISAR LA NUEVA FUNCIÓN DE SOFTMAX DERIVATIVES TIMES TENSOR PARA PODER QUITAR LOS ATTENTION SCORES

    scaled_deltas.device(*thread_pool_device) = deltas;
    multiply_matrices(thread_pool_device.get(), scaled_deltas, gammas);

    aux_2d.device(*thread_pool_device) = (scaled_deltas * outputs).sum(sum_dimensions_1) 
        / (embedding_dimension * (standard_deviations + epsilon));

    standard_deviation_derivatives.device(*thread_pool_device) = outputs;
    multiply_matrices(thread_pool_device.get(), standard_deviation_derivatives, aux_2d);

    scaled_deltas.device(*thread_pool_device) = scaled_deltas / (standard_deviations.reshape(range_3).broadcast(expand_normalization_axis) + epsilon);

    input_derivatives.device(*thread_pool_device) = scaled_deltas - standard_deviation_derivatives;

    aux_2d.device(*thread_pool_device) = 1 / type(embedding_dimension) * scaled_deltas.sum(sum_dimensions_1);

    substract_matrices(thread_pool_device.get(), aux_2d, input_derivatives);
}


void Normalization3d::insert_gradient(unique_ptr<LayerBackPropagation>& back_propagation,
                                      Index& index,
                                      Tensor<type, 1>& gradient) const
{
    Normalization3dBackPropagation* this_back_propagation =
        static_cast<Normalization3dBackPropagation*>(back_propagation.get());

    copy_to_vector(gradient, this_back_propagation->gamma_derivatives, index);
    copy_to_vector(gradient, this_back_propagation->beta_derivatives, index);
}


void Normalization3d::from_XML(const XMLDocument& document)
{
    const XMLElement* normalization_layer_element = document.FirstChildElement("Normalization3D");

    if(!normalization_layer_element)
        throw runtime_error("Normalization3D element is nullptr.\n");

    const string new_name = read_xml_string(normalization_layer_element, "Name");
    const Index new_sequence_length = read_xml_index(normalization_layer_element, "SequenceLength");
    const Index new_embedding_dimension = read_xml_index(normalization_layer_element, "EmbeddingDimension");

    set(new_sequence_length, new_embedding_dimension, new_name);

    Index index = 0;

    set_parameters(to_type_vector(read_xml_string(normalization_layer_element, "Parameters"), " "), index);
}


void Normalization3d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Normalization3D");

    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "SequenceLength", to_string(get_sequence_length()));
    add_xml_element(printer, "EmbeddingDimension", to_string(get_embedding_dimension()));
    add_xml_element(printer, "Parameters", tensor_to_string(get_parameters()));

    printer.CloseElement();
}


Normalization3dForwardPropagation::Normalization3dForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> Normalization3dForwardPropagation::get_outputs_pair() const
{
    Normalization3d* normalization_3d = static_cast<Normalization3d*>(layer);

    const Index sequence_length = normalization_3d->get_sequence_length();
    const Index embedding_dimension = normalization_3d->get_embedding_dimension();

    return { (type*)outputs.data(), { batch_size, sequence_length, embedding_dimension } };
}


void Normalization3dForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    Normalization3d* normalization_3d = static_cast<Normalization3d*>(layer);

    batch_size = new_batch_size;

    const Index sequence_length = normalization_3d->get_sequence_length();
    const Index embedding_dimension = normalization_3d->get_embedding_dimension();

    outputs.resize(batch_size, sequence_length, embedding_dimension);

    means.resize(batch_size, sequence_length);
    standard_deviations.resize(batch_size, sequence_length);
}


void Normalization3dForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs << endl;
}


void Normalization3dBackPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    layer = new_layer;

    Normalization3d* normalization_layer_3d = static_cast<Normalization3d*>(layer);

    batch_size = new_batch_size;

    const Index sequence_length = normalization_layer_3d->get_sequence_length();
    const Index embedding_dimension = normalization_layer_3d->get_embedding_dimension();

    gamma_derivatives.resize(embedding_dimension);
    beta_derivatives.resize(embedding_dimension);

    scaled_deltas.resize(batch_size, sequence_length, embedding_dimension);
    standard_deviation_derivatives.resize(batch_size, sequence_length, embedding_dimension);
    aux_2d.resize(batch_size, sequence_length);

    input_derivatives.resize(batch_size, sequence_length, embedding_dimension);
}


void Normalization3dBackPropagation::print() const
{
    cout << "Gammas derivatives:" << endl
        << gamma_derivatives << endl
        << "Betas derivatives:" << endl
        << beta_derivatives << endl;
}


Normalization3dBackPropagation::Normalization3dBackPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<pair<type*, dimensions>> Normalization3dBackPropagation::get_input_derivative_pairs() const
{
    Normalization3d* normalization_layer_3d = static_cast<Normalization3d*>(layer);

    const Index sequence_length = normalization_layer_3d->get_sequence_length();
    const Index embedding_dimension = normalization_layer_3d->get_embedding_dimension();

    return { {(type*)(input_derivatives.data()), {batch_size, sequence_length, embedding_dimension}} };
}

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
