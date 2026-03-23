//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N O R M A L I Z A T I O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "tensor_utilities.h"
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
    return gammas.shape[0];
}


Shape Normalization3d::get_input_shape() const
{
    return { sequence_length, get_embedding_dimension() };
}


Shape Normalization3d::get_output_shape() const
{
    return { sequence_length, get_embedding_dimension() };
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
    betas.shape = {new_embedding_dimension};

    label = new_label;
    name = "Normalization3d";

#ifdef OPENNN_CUDA
    gammas_device.set_descriptor({1, 1, 1, new_embedding_dimension});
    betas_device.set_descriptor({1, 1, 1, new_embedding_dimension});
#endif
}


void Normalization3d::set_parameters_random()
{
    if(gammas.size() > 0)
        VectorMap(gammas.data, gammas.size()).setOnes();

    if(betas.size() > 0)
        VectorMap(betas.data, betas.size()).setZero();
}

void Normalization3d::set_parameters_glorot()
{
    set_parameters_random();
}


void Normalization3d::forward_propagate(unique_ptr<LayerForwardPropagation>& forward_propagation,
                                        bool)
{
    const Index batch_size = forward_propagation->batch_size;
    const Index embedding_dimension = get_embedding_dimension();

    const TensorMap3 inputs(forward_propagation->inputs[0].data, batch_size, sequence_length, embedding_dimension);

    TensorMap3 outputs = tensor_map<3>(forward_propagation->outputs);

    Normalization3dForwardPropagation* this_forward_propagation =
        static_cast<Normalization3dForwardPropagation*>(forward_propagation.get());

    Tensor2& means = this_forward_propagation->means;
    Tensor2& standard_deviations = this_forward_propagation->standard_deviations;
    Tensor3& normalized_inputs = this_forward_propagation->normalized_inputs;

    const array<Index, 3> reshape_dims({batch_size, sequence_length, 1});
    const array<Index, 3> broadcast_dims({1, 1, embedding_dimension});

    means.device(get_device()) = inputs.mean(array<Index, 1>({2}));

    auto centered_inputs = inputs - means.reshape(reshape_dims).broadcast(broadcast_dims);
    auto variance = centered_inputs.square().mean(array<Index, 1>({2}));
    standard_deviations.device(get_device()) = (variance + EPSILON).sqrt();

    normalized_inputs.device(get_device()) = centered_inputs / standard_deviations.reshape(reshape_dims).broadcast(broadcast_dims);

    TensorMap1 gamma_map(gammas.data, embedding_dimension);
    TensorMap1 beta_map(betas.data, embedding_dimension);

    auto gamma_bcast = gamma_map.reshape(array<Index, 3>({1, 1, embedding_dimension}))
                           .broadcast(array<Index, 3>({batch_size, sequence_length, 1}));

    auto beta_bcast = beta_map.reshape(array<Index, 3>({1, 1, embedding_dimension}))
                          .broadcast(array<Index, 3>({batch_size, sequence_length, 1}));

    outputs.device(get_device()) = normalized_inputs * gamma_bcast + beta_bcast;
}


void Normalization3d::back_propagate(unique_ptr<LayerForwardPropagation>& forward_propagation,
                                     unique_ptr<LayerBackPropagation>& back_propagation) const
{
    const Index batch_size = forward_propagation->inputs[0].shape[0];
    const Index embedding_dimension = get_embedding_dimension();

    if(back_propagation->output_gradients.size() > 1)
        add_gradients(back_propagation->output_gradients);

    const TensorMap3 output_gradients = tensor_map<3>(back_propagation->output_gradients[0]);

    const Normalization3dForwardPropagation* this_forward_propagation =
        static_cast<Normalization3dForwardPropagation*>(forward_propagation.get());

    const Tensor2& standard_deviations = this_forward_propagation->standard_deviations;
    const Tensor3& X_hat = this_forward_propagation->normalized_inputs;

    Normalization3dBackPropagation* this_back_propagation =
        static_cast<Normalization3dBackPropagation*>(back_propagation.get());

    VectorMap dGamma_map = vector_map(this_back_propagation->gamma_gradients);
    VectorMap dBeta_map = vector_map(this_back_propagation->beta_gradients);
    TensorMap3 dX = tensor_map<3>(this_back_propagation->input_gradients[0]);

    Tensor1 dGamma_tensor = (output_gradients * X_hat).sum(array<Index, 2>({0, 1}));
    Tensor1 dBeta_tensor = output_gradients.sum(array<Index, 2>({0, 1}));

    for(Index i = 0; i < embedding_dimension; ++i)
    {
        dGamma_map(i) = dGamma_tensor(i);
        dBeta_map(i) = dBeta_tensor(i);
    }

    TensorMap1 gamma_map(gammas.data, embedding_dimension);
    auto gamma_bcast = gamma_map.reshape(array<Index, 3>({1, 1, embedding_dimension}))
                           .broadcast(array<Index, 3>({batch_size, sequence_length, 1}));

    // D = dY * Gamma
    Tensor3 D = output_gradients * gamma_bcast;

    // sum_D = sum(D, axis=2)
    Tensor2 sum_D = D.sum(array<Index, 1>({2}));

    // sum_D_xhat = sum(D * X_hat, axis=2)
    Tensor2 sum_D_xhat = (D * X_hat).sum(array<Index, 1>({2}));

    // Broadcast components for dX calculation
    auto sum_D_bcast = sum_D.reshape(array<Index, 3>({batch_size, sequence_length, 1}))
                           .broadcast(array<Index, 3>({1, 1, embedding_dimension}));

    auto sum_D_xhat_bcast = sum_D_xhat.reshape(array<Index, 3>({batch_size, sequence_length, 1}))
                                .broadcast(array<Index, 3>({1, 1, embedding_dimension}));

    auto std_dev_bcast = standard_deviations.reshape(array<Index, 3>({batch_size, sequence_length, 1}))
                             .broadcast(array<Index, 3>({1, 1, embedding_dimension}));

    const type inv_E = type(1.0) / static_cast<type>(embedding_dimension);

    // dX = (1 / sigma) * (D - mean(D) - X_hat * mean(D * X_hat))
    dX.device(get_device()) = (D - sum_D_bcast * inv_E - X_hat * sum_D_xhat_bcast * inv_E) / std_dev_bcast;
}

#ifdef OPENNN_CUDA

vector<TensorViewCuda*> Normalization3d::get_parameter_views_device()
{
    return {&gammas_device, &betas_device};
}


void Normalization3d::forward_propagate(unique_ptr<LayerForwardPropagationCuda>& forward_propagation, bool)
{
    Normalization3dForwardPropagationCuda* fp_cuda = static_cast<Normalization3dForwardPropagationCuda*>(forward_propagation.get());

    const int N = static_cast<int>(fp_cuda->batch_size * sequence_length);
    const int D = static_cast<int>(get_embedding_dimension());

    layernorm_forward_cuda(
        N, D,
        forward_propagation->inputs[0].data,
        fp_cuda->outputs.data,
        fp_cuda->means_device.data,
        fp_cuda->inv_variances_device.data,
        gammas_device.data,
        betas_device.data,
        static_cast<float>(EPSILON)
        );
}


void Normalization3d::back_propagate(unique_ptr<LayerForwardPropagationCuda>& forward_propagation,
                                     unique_ptr<LayerBackPropagationCuda>& back_propagation) const
{
    Normalization3dForwardPropagationCuda* fp_cuda = static_cast<Normalization3dForwardPropagationCuda*>(forward_propagation.get());
    Normalization3dBackPropagationCuda* bp_cuda = static_cast<Normalization3dBackPropagationCuda*>(back_propagation.get());

    const int N = static_cast<int>(fp_cuda->batch_size * sequence_length);
    const int D = static_cast<int>(get_embedding_dimension());

    CHECK_CUDA(cudaMemset(bp_cuda->gamma_gradients.data, 0, D * sizeof(float)));
    CHECK_CUDA(cudaMemset(bp_cuda->beta_gradients.data, 0, D * sizeof(float)));

    layernorm_backward_cuda(
        N, D,
        back_propagation->output_gradients[0].data,
        forward_propagation->inputs[0].data,
        fp_cuda->means_device.data,
        fp_cuda->inv_variances_device.data,
        gammas_device.data,
        bp_cuda->input_gradients[0].data,
        bp_cuda->gamma_gradients.data,
        bp_cuda->beta_gradients.data
        );
}

#endif


void Normalization3d::from_XML(const XMLDocument& document)
{
    const XMLElement* element = document.FirstChildElement("Normalization3d");
    if(!element) throw runtime_error("Normalization3d element is nullptr.\n");

    const string new_name = read_xml_string(element, "Label");
    const Index new_sequence_length = read_xml_index(element, "SequenceLength");
    const Index new_embedding_dimension = read_xml_index(element, "EmbeddingDimension");

    set(new_sequence_length, new_embedding_dimension, new_name);
}


void Normalization3d::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Normalization3d");
    add_xml_element(printer, "Label", label);
    add_xml_element(printer, "SequenceLength", to_string(get_sequence_length()));
    add_xml_element(printer, "EmbeddingDimension", to_string(get_embedding_dimension()));
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
    normalized_inputs.resize(batch_size, sequence_length, embedding_dimension);
}


void Normalization3dForwardPropagation::print() const
{
    cout << "Normalization3d Outputs shape: " << outputs.shape << endl;
}


Normalization3dBackPropagation::Normalization3dBackPropagation(const Index new_batch_size, Layer* new_layer)
    : LayerBackPropagation()
{
    set(new_batch_size, new_layer);
}


void Normalization3dBackPropagation::initialize()
{
    const Normalization3d* normalization_layer_3d = static_cast<Normalization3d*>(layer);

    const Index sequence_length = normalization_layer_3d->get_sequence_length();
    const Index embedding_dimension = normalization_layer_3d->get_embedding_dimension();

    input_gradients = {{nullptr, {batch_size, sequence_length, embedding_dimension}}};

    gamma_gradients.shape = {embedding_dimension};
    beta_gradients.shape = {embedding_dimension};
}


vector<TensorView*> Normalization3dBackPropagation::get_gradient_views()
{
    return {&gamma_gradients, &beta_gradients};
}


void Normalization3dBackPropagation::print() const
{
    cout << "Normalization3d BackPropagation initialized." << endl;
}

#ifdef OPENNN_CUDA

Normalization3dForwardPropagationCuda::Normalization3dForwardPropagationCuda(const Index new_batch_size, Layer* new_layer) : LayerForwardPropagationCuda()
{
    set(new_batch_size, new_layer);
}

void Normalization3dForwardPropagationCuda::initialize()
{
    const Normalization3d* norm_layer = static_cast<Normalization3d*>(layer);
    const Index seq = norm_layer->get_sequence_length();
    const Index dim = norm_layer->get_embedding_dimension();

    outputs.set_descriptor({batch_size, seq, dim});

    means_device.resize({batch_size * seq});
    inv_variances_device.resize({batch_size * seq});
}


Normalization3dBackPropagationCuda::Normalization3dBackPropagationCuda(const Index new_batch_size, Layer* new_layer) : LayerBackPropagationCuda()
{
    set(new_batch_size, new_layer);
}

void Normalization3dBackPropagationCuda::initialize()
{
    const Normalization3d* norm_layer = static_cast<Normalization3d*>(layer);
    const Index seq = norm_layer->get_sequence_length();
    const Index dim = norm_layer->get_embedding_dimension();

    input_gradients = {TensorViewCuda({batch_size, seq, dim})};

    gamma_gradients.set_descriptor({dim});
    beta_gradients.set_descriptor({dim});
}

vector<TensorViewCuda*> Normalization3dBackPropagationCuda::get_gradient_views()
{
    return {&gamma_gradients, &beta_gradients};
}


REGISTER(LayerForwardPropagationCuda, Normalization3dForwardPropagationCuda, "Normalization3d")
REGISTER(LayerBackPropagationCuda, Normalization3dBackPropagationCuda, "Normalization3d")

#endif

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
