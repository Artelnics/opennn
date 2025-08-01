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

Normalization3d::Normalization3d(const dimensions& new_input_dimensions,
                                 const string& new_name) : Layer()
{
    set(new_input_dimensions[0], new_input_dimensions[1], new_name);
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


vector<pair<type *, Index> > Normalization3d::get_parameter_pairs() const
{
    return {
        {(type*)gammas.data(), gammas.size()},
        {(type*)betas.data(), betas.size()}
    };
}


void Normalization3d::set(const Index& new_sequence_length,
                          const Index& new_embedding_dimension,
                          const string& new_label)
{
    sequence_length = new_sequence_length;

    gammas.resize(new_embedding_dimension);
    gammas.setConstant(1);

    betas.resize(new_embedding_dimension);
    betas.setZero();

    label = new_label;

    name = "Normalization3d";
}


void Normalization3d::forward_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                        unique_ptr<LayerForwardPropagation>& layer_forward_propagation,
                                        const bool&)
{
    const Index batch_size = layer_forward_propagation->batch_size;
//    const Index sequence_length = get_sequence_length();
    const Index embedding_dimension = get_embedding_dimension();

    const TensorMap<Tensor<type, 3>> inputs(input_pairs[0].first, batch_size, sequence_length, embedding_dimension);

    Normalization3dForwardPropagation* this_forward_propagation =
        static_cast<Normalization3dForwardPropagation*>(layer_forward_propagation.get());

    Tensor<type, 3>& outputs = this_forward_propagation->outputs;
    Tensor<type, 2>& means = this_forward_propagation->means;
    Tensor<type, 2>& standard_deviations = this_forward_propagation->standard_deviations;

    // Standarization

    #pragma omp parallel for collapse(2)
    for (Index i = 0; i < batch_size; ++i)
    {
        for (Index j = 0; j < sequence_length; ++j)
        {
            type mean_sum = 0;

            for (Index k = 0; k < embedding_dimension; ++k)
                mean_sum += inputs(i, j, k);

            means(i, j) = mean_sum / type(embedding_dimension);

            type std_dev_sum = 0;
            const type current_mean = means(i, j);

            for (Index k = 0; k < embedding_dimension; ++k)
            {
                type diff = inputs(i, j, k) - current_mean;
                std_dev_sum += diff * diff;
            }

            standard_deviations(i, j) = sqrt(std_dev_sum / type(embedding_dimension));

            const type current_std_dev = standard_deviations(i, j);

            for (Index k = 0; k < embedding_dimension; ++k)
                outputs(i, j, k) = (inputs(i, j, k) - current_mean) / (current_std_dev + epsilon);
        }
    }

    // Affine transformation

    #pragma omp parallel for collapse(3)
    for (Index i = 0; i < batch_size; ++i)
        for (Index j = 0; j < sequence_length; ++j)
            for (Index k = 0; k < embedding_dimension; ++k)
                outputs(i, j, k) = outputs(i, j, k) * gammas(k) + betas(k);
}


void Normalization3d::back_propagate(const vector<pair<type*, dimensions>>& input_pairs,
                                     const vector<pair<type*, dimensions>>& delta_pairs,
                                     unique_ptr<LayerForwardPropagation>& forward_propagation,
                                     unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_size = input_pairs[0].second[0];
    const Index embedding_dimension = get_embedding_dimension();

    if(delta_pairs.size() > 1)     
        add_deltas(delta_pairs);

    const TensorMap<Tensor<type, 3>> deltas = tensor_map<3>(delta_pairs[0]);

    const Normalization3dForwardPropagation* this_forward_propagation 
        = static_cast<Normalization3dForwardPropagation*>(forward_propagation.get());

    const Tensor<type, 3>& outputs = this_forward_propagation->outputs;
    const Tensor<type, 2>& standard_deviations = this_forward_propagation->standard_deviations;

    Normalization3dBackPropagation* this_back_propagation =
        static_cast<Normalization3dBackPropagation*>(back_propagation.get());
    
    Tensor<type, 1>& gamma_derivatives = this_back_propagation->gamma_derivatives;
    Tensor<type, 1>& beta_derivatives = this_back_propagation->beta_derivatives;

    Tensor<type, 3>& scaled_deltas = this_back_propagation->scaled_deltas;
    Tensor<type, 3>& input_deltas = this_back_propagation->input_deltas;

    // Parameters derivatives

    gamma_derivatives.setZero();
    beta_derivatives.setZero();

    #pragma omp parallel
    {
        Tensor<type, 1> private_gamma_derivs(embedding_dimension); private_gamma_derivs.setZero();
        Tensor<type, 1> private_beta_derivs(embedding_dimension); private_beta_derivs.setZero();

        #pragma omp for
        for (Index i = 0; i < batch_size; ++i)
            for (Index j = 0; j < sequence_length; ++j)
                for (Index k = 0; k < embedding_dimension; ++k)
                {
                    private_gamma_derivs(k) += outputs(i, j, k) * deltas(i, j, k);
                    private_beta_derivs(k) += deltas(i, j, k);
                }

        #pragma omp critical
        {
            gamma_derivatives += private_gamma_derivs;
            beta_derivatives += private_beta_derivs;
        }
    }
    
    // Input derivatives

    #pragma omp parallel for collapse(3)
    for (Index i = 0; i < batch_size; ++i)
        for (Index j = 0; j < sequence_length; ++j)
            for (Index k = 0; k < embedding_dimension; ++k)
                scaled_deltas(i, j, k) = deltas(i, j, k) * gammas(k);

    #pragma omp parallel for collapse(2)
    for (Index i = 0; i < batch_size; ++i)
        for (Index j = 0; j < sequence_length; ++j)
        {
            type d_var_sum = 0;
            type d_mean_sum = 0;
            const type inv_std_dev = type(1) / (standard_deviations(i, j) + epsilon);

            for (Index k = 0; k < embedding_dimension; ++k)
                d_var_sum += scaled_deltas(i, j, k) * outputs(i, j, k);

            d_var_sum *= -type(0.5) * pow(inv_std_dev, 3);

            for (Index k = 0; k < embedding_dimension; ++k)
                d_mean_sum += scaled_deltas(i, j, k);

            d_mean_sum *= -inv_std_dev;

            const type inv_embed_dim = type(1) / type(embedding_dimension);

            for (Index k = 0; k < embedding_dimension; ++k)
                input_deltas(i, j, k) = scaled_deltas(i, j, k) * inv_std_dev
                                        + d_var_sum * type(2) * outputs(i, j, k) * inv_std_dev * inv_embed_dim
                                        + d_mean_sum * inv_embed_dim;
        }
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

    string_to_tensor<type, 1>(read_xml_string(normalization_layer_element, "Betas"), betas);
    string_to_tensor<type, 1>(read_xml_string(normalization_layer_element, "Gammas"), gammas);
}


void Normalization3d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Normalization3d");
    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "SequenceLength", to_string(get_sequence_length()));
    add_xml_element(printer, "EmbeddingDimension", to_string(get_embedding_dimension()));
    add_xml_element(printer, "Betas", tensor_to_string<type, 1>(betas));
    add_xml_element(printer, "Gammas", tensor_to_string<type, 1>(gammas));

    printer.CloseElement();
}


Normalization3dForwardPropagation::Normalization3dForwardPropagation(const Index& new_batch_size, Layer* new_layer)
    : LayerForwardPropagation()
{
    set(new_batch_size, new_layer);
}


pair<type*, dimensions> Normalization3dForwardPropagation::get_output_pair() const
{
    Normalization3d* normalization_3d = static_cast<Normalization3d*>(layer);

    const Index sequence_length = normalization_3d->get_sequence_length();
    const Index embedding_dimension = normalization_3d->get_embedding_dimension();

    return { (type*)outputs.data(), { batch_size, sequence_length, embedding_dimension } };
}


void Normalization3dForwardPropagation::set(const Index& new_batch_size, Layer* new_layer)
{
    if (!new_layer) return;

    layer = new_layer;

    batch_size = new_batch_size;

    Normalization3d* normalization_3d = static_cast<Normalization3d*>(layer);

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
    if (!new_layer) return;

    batch_size = new_batch_size;

    layer = new_layer;

    Normalization3d* normalization_layer_3d = static_cast<Normalization3d*>(layer);

    const Index sequence_length = normalization_layer_3d->get_sequence_length();
    const Index embedding_dimension = normalization_layer_3d->get_embedding_dimension();

    gamma_derivatives.resize(embedding_dimension);
    beta_derivatives.resize(embedding_dimension);

    scaled_deltas.resize(batch_size, sequence_length, embedding_dimension);
    standard_deviation_derivatives.resize(batch_size, sequence_length, embedding_dimension);
    aux_2d.resize(batch_size, sequence_length);

    input_deltas.resize(batch_size, sequence_length, embedding_dimension);
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

    return { {(type*)(input_deltas.data()), {batch_size, sequence_length, embedding_dimension}} };
}

vector<pair<type*, Index>> Normalization3dBackPropagation::get_parameter_delta_pairs() const
{
    return {
        {(type*)gamma_derivatives.data(), gamma_derivatives.size()},
        {(type*)beta_derivatives.data(), beta_derivatives.size()}
    };
}


REGISTER(Layer, Normalization3d, "Normalization3d")
REGISTER(LayerForwardPropagation, Normalization3dForwardPropagation, "Normalization3d")
REGISTER(LayerBackPropagation, Normalization3dBackPropagation, "Normalization3d")

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
