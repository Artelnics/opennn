//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensors.h"
#include "normalization_layer_3d.h"

namespace opennn
{

Normalization3d::Normalization3d(const Shape& new_input_shape,
                                 const string& new_name) : Layer()
{
    set(new_input_shape[0], new_input_shape[1], new_name);
}


Index Normalization3d::get_sequence_length() const
{
    return sequence_length;
}


Index Normalization3d::get_embedding_dimension() const
{
    return gammas.size();
}


Shape Normalization3d::get_input_shape() const
{
    const Index embedding_dimension = get_embedding_dimension();

    return { sequence_length, embedding_dimension };
}


Shape Normalization3d::get_output_shape() const
{
    const Index embedding_dimension = get_embedding_dimension();

    return { sequence_length, embedding_dimension };
}


vector<TensorView*> Normalization3d::get_parameter_views()
{
    return {&gammas, &betas};
}


void Normalization3d::set(const Index new_sequence_length,
                          Index new_embedding_dimension,
                          const string& new_label)
{
    sequence_length = new_sequence_length;

    gammas.shape = {new_embedding_dimension};
/*
    gammas.setConstant(1);
*/
    betas.shape = {new_embedding_dimension};
/*
    betas.setZero();
*/
    label = new_label;

    name = "Normalization3d";
}


void Normalization3d::forward_propagate(const vector<TensorView>& input_views,
                                        unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                        bool)
{
    const Index batch_size = layer_forward_propagation->batch_size;
    //    const Index sequence_length = get_sequence_length();
    const Index embedding_dimension = get_embedding_dimension();

    const TensorMap3 inputs(input_views[0].data, batch_size, sequence_length, embedding_dimension);

    TensorMap3 outputs = tensor_map<3>(layer_forward_propagation->outputs);

    Normalization3dForwardPropagation* this_forward_propagation =
        static_cast<Normalization3dForwardPropagation*>(layer_forward_propagation.get());

    Tensor2& means = this_forward_propagation->means;
    Tensor2& standard_deviations = this_forward_propagation->standard_deviations;

    const array<Index, 3> reshape_dims({batch_size, sequence_length, 1});
    const array<Index, 3> broadcast_dims({1, 1, embedding_dimension});

    // Standarization

    means.device(get_device()) = inputs.mean(array<Index, 1>({2}));

    standard_deviations.device(get_device())
        = (inputs - means.reshape(reshape_dims).broadcast(broadcast_dims)).square().mean(array<Index, 1>({2})).sqrt();

    outputs.device(get_device())
        = (inputs - means.reshape(reshape_dims).broadcast(broadcast_dims))
          / (standard_deviations.reshape(reshape_dims).broadcast(broadcast_dims) + numeric_limits<type>::epsilon());

    // Affine transformation
/*
    multiply_matrices(get_device(), outputs, gammas);

    outputs.device(get_device()) = outputs + betas.reshape(array<Index, 3>{1, 1, betas.dimension(0)})
                                             .broadcast(array<Index, 3>{outputs.dimension(0), outputs.dimension(1), 1});
*/
}


void Normalization3d::back_propagate(const vector<TensorView>& input_views,
                                     const vector<TensorView>& output_gradient_views,
                                     unique_ptr<LayerForwardPropagation>& forward_propagation,
                                     unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_size = input_views[0].shape[0];
    const Index embedding_dimension = get_embedding_dimension();

    if(output_gradient_views.size() > 1)
        add_gradients(output_gradient_views);

    const TensorMap3 output_gradients = tensor_map<3>(output_gradient_views[0]);

    const TensorMap3 outputs = tensor_map<3>(forward_propagation->outputs);

    const Normalization3dForwardPropagation* this_forward_propagation
        = static_cast<Normalization3dForwardPropagation*>(forward_propagation.get());

    const Tensor2& standard_deviations = this_forward_propagation->standard_deviations;

    Normalization3dBackPropagation* this_back_propagation =
        static_cast<Normalization3dBackPropagation*>(back_propagation.get());

    VectorMap gamma_derivatives = vector_map(this_back_propagation->gamma_derivatives);
    VectorMap beta_derivatives = vector_map(this_back_propagation->beta_derivatives);

    Tensor3& scaled_gradients = this_back_propagation->scaled_gradients;
    Tensor3& standard_deviation_derivatives = this_back_propagation->standard_deviation_derivatives;

    TensorMap3 input_gradients = tensor_map<3>(this_back_propagation->input_gradients[0]);

    Tensor2& aux_2d = this_back_propagation->aux_2d;

    constexpr type epsilon = numeric_limits<type>::epsilon();

    // Parameters derivatives

    gamma_derivatives.device(get_device()) = (outputs * output_gradients).sum(array<Index, 2>({0, 1}));
    beta_derivatives.device(get_device()) = output_gradients.sum(array<Index, 2>({0, 1}));

    // Input derivatives
/*
    scaled_gradients.device(get_device()) = output_gradients;
    multiply_matrices(get_device(), scaled_gradients, gammas);

    aux_2d.device(get_device()) = (scaled_gradients * outputs).sum(array<Index, 1>({2}))
                                         / (embedding_dimension * (standard_deviations + epsilon));

    standard_deviation_derivatives.device(get_device()) = outputs;
    multiply_matrices(get_device(), standard_deviation_derivatives, aux_2d);

    scaled_gradients.device(get_device()) = scaled_gradients
                                                / (standard_deviations.reshape(array<Index, 3>({batch_size, sequence_length, 1}))
                                                       .broadcast(array<Index, 3>({1, 1, embedding_dimension})) + epsilon);

    input_gradients.device(get_device()) = scaled_gradients - standard_deviation_derivatives;

    aux_2d.device(get_device()) = 1 / type(embedding_dimension) * scaled_gradients.sum(array<Index, 1>({2}));
    /*
    input_derivatives.device(get_device()) = input_derivatives
        - aux_2d.reshape(array<Index, 3>{input_derivatives.dimension(0), input_derivatives.dimension(1), 1})
                .broadcast(array<Index, 3>{1, 1, input_derivatives.dimension(2)});
*/
    //substract_matrices(get_device(), aux_2d, input_derivatives);
}


void Normalization3d::from_XML(const XMLDocument& document)
{
    const XMLElement* normalization_layer_element = document.FirstChildElement("Normalization3d");

    if(!normalization_layer_element)
        throw runtime_error("Normalization3d element is nullptr.\n");

    const string new_name = read_xml_string(normalization_layer_element, "Name");
    const Index new_sequence_length = read_xml_index(normalization_layer_element, "SequenceLength");
    const Index new_embedding_dimension = read_xml_index(normalization_layer_element, "EmbeddingDimension");

    set(new_sequence_length, new_embedding_dimension, new_name);
/*
    string_to_tensor<type, 1>(read_xml_string(normalization_layer_element, "Betas"), betas);
    string_to_tensor<type, 1>(read_xml_string(normalization_layer_element, "Gammas"), gammas);
*/
}


void Normalization3d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Normalization3d");
    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "SequenceLength", to_string(get_sequence_length()));
    add_xml_element(printer, "EmbeddingDimension", to_string(get_embedding_dimension()));
/*
    add_xml_element(printer, "Betas", tensor_to_string<type, 1>(betas));
    add_xml_element(printer, "Gammas", tensor_to_string<type, 1>(gammas));
*/
    printer.CloseElement();
}


Normalization3dForwardPropagation::Normalization3dForwardPropagation(const Index new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


void Normalization3dForwardPropagation::initialize()
{
    const Normalization3d* normalization_3d = static_cast<Normalization3d*>(layer);

    const Index sequence_length = normalization_3d->get_sequence_length();
    const Index embedding_dimension = normalization_3d->get_embedding_dimension();

    outputs.shape = {batch_size, sequence_length, embedding_dimension};

    means.resize(batch_size, sequence_length);
    standard_deviations.resize(batch_size, sequence_length);
}


void Normalization3dForwardPropagation::print() const
{
    cout << "Outputs:" << endl
         << outputs.shape << endl;
}


void Normalization3dBackPropagation::initialize()
{
    const Normalization3d* normalization_layer_3d = static_cast<Normalization3d*>(layer);

    const Index sequence_length = normalization_layer_3d->get_sequence_length();
    const Index embedding_dimension = normalization_layer_3d->get_embedding_dimension();

    gamma_derivatives.shape = {embedding_dimension};
    beta_derivatives.shape = {embedding_dimension};

    scaled_gradients.resize(batch_size, sequence_length, embedding_dimension);
    standard_deviation_derivatives.resize(batch_size, sequence_length, embedding_dimension);
    aux_2d.resize(batch_size, sequence_length);

    input_gradients.resize(1);
    input_gradients[0].shape = {batch_size, sequence_length, embedding_dimension};
}


void Normalization3dBackPropagation::print() const
{
/*
    cout << "Gammas derivatives:" << endl
         << gamma_derivatives << endl
         << "Betas derivatives:" << endl
         << beta_derivatives << endl;
*/
}


Normalization3dBackPropagation::Normalization3dBackPropagation(const Index new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


vector<TensorView*> Normalization3dBackPropagation::get_gradient_views()
{
    return {&gamma_derivatives, &beta_derivatives};
}


REGISTER(Layer, Normalization3d, "Normalization3d")
REGISTER(LayerForwardPropagation, Normalization3dForwardPropagation, "Normalization3d")
REGISTER(LayerBackPropagation, Normalization3dBackPropagation, "Normalization3d")

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
